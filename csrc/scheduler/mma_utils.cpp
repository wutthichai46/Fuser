// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <id_model/id_model.h>
#include <ir/printer.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <val_graph.h>
#include <variant>
#include "mma_type.h"
namespace nvfuser {

namespace mma_utils {

//! A wrapper to get MMA Tensor data types
//!   The order of returned types: INPUT_A, INPUT_B, OUTPUT_D
inline mma_utils::MmaDataTypes getMmaDataTypes(
    const std::map<MatmulRole, std::vector<TensorView*>>& roles_map) {
  auto getMMADataType = [&](MatmulRole role) {
    auto entry = roles_map.find(role);
    if (entry != roles_map.end() && !entry->second.empty()) {
      return entry->second.front()->dtype();
    }
    NVF_ERROR(false, "Get MMA Tensor data type failed!");
  };
  const auto a_type = getMMADataType(MatmulRole::INPUT_A);
  const auto b_type = getMMADataType(MatmulRole::INPUT_B);
  const auto c_type = getMMADataType(MatmulRole::OUTPUT_D);
  return mma_utils::MmaDataTypes{a_type, b_type, c_type};
}

//! Return sizes of smem_a, smem_b, smem_c in bytes
std::tuple<int64_t, int64_t, int64_t> computeSharedMemorySizes(
    const MatMulTileOptions& gemm_tile,
    const MatmulParams::DoubleBufferOptions& double_buffer_options,
    const MmaDataTypes& data_types) {
  const auto properties = at::cuda::getCurrentDeviceProperties();

  auto warp_dims = gemm_tile.cta_tile / gemm_tile.warp_tile;

  int64_t ab_factor = double_buffer_options.double_buffer_smem_write
      ? double_buffer_options.smem_double_buffer_stage
      : 1;

  // see scheduleContiguousVectorLoad
  const int64_t vector_word = 8;
  const int64_t round_to_factor = warp_dims.m * warp_dims.n * warp_dims.k *
      properties->warpSize * vector_word;
  const int64_t mk = gemm_tile.cta_tile.m * gemm_tile.cta_tile.k;
  const int64_t nk = gemm_tile.cta_tile.n * gemm_tile.cta_tile.k;
  const int64_t smem_a = ceilDiv(mk, round_to_factor) * round_to_factor *
      ab_factor * dataTypeSize(data_types[0]);
  const int64_t smem_b = ceilDiv(nk, round_to_factor) * round_to_factor *
      ab_factor * dataTypeSize(data_types[1]);
  const int64_t smem_c =
      gemm_tile.cta_tile.m * gemm_tile.cta_tile.n * dataTypeSize(data_types[2]);

  return {smem_a, smem_b, smem_c};
}

int64_t computeExpectedSharedMemoryUsage(
    const MatmulParams& params,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed,
    bool smem_b_reuse_guaranteed) {
  const auto [smem_a, smem_b, smem_c] = computeSharedMemorySizes(
      params.tile_sizes, params.double_buffer_options, data_types);

  if (params.use_smem_epilogue) {
    if (params.promote_prologue_smem_reuse) {
      return (int64_t)std::max(
          smem_c + (smem_a_reuse_guaranteed ? 0 : smem_a) +
              (smem_b_reuse_guaranteed ? 0 : smem_b),
          smem_a + smem_b);
    } else {
      return (int64_t)(smem_a + smem_b + smem_c);
    }
  } else {
    return (int64_t)(smem_a + smem_b);
  }
}

std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    int smem_double_buffer_stage,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed,
    bool smem_b_reuse_guaranteed,
    bool ignore_occupancy_drop) {
  const size_t shared_memory_available = deviceAvailableSharedMemoryBytes();

  // We clip smem_double_buffer_stage to 1 since we will always load operands
  // to smem even if stages=0. That is, we interpret stages <= 1 as requesting
  // "no double-buffering", but we still stage incoming data to smem.
  if (smem_double_buffer_stage < 1) {
    smem_double_buffer_stage = 1;
  }

  // Create a temporary DoubleBufferOptions with full double buffering, for
  // estimating shared memory size.
  MatmulParams::DoubleBufferOptions double_buffer_options{
      true, true, smem_double_buffer_stage};

  const auto [smem_a, smem_b, smem_c] =
      computeSharedMemorySizes(gemm_tile, double_buffer_options, data_types);

  // NOTE: we can simply add these sizes since they should be integer multiples
  // of 16 bytes, so they will automatically be aligned. This may change with
  // FP8, in which case the expressions below should be updated to insert
  // alignment expressions, using the expected stack ordering in
  // StackBasedSharedMemAllocator.
  NVF_CHECK(smem_a % 16 == 0 && smem_b % 16 == 0 && smem_b % 16 == 0);

  const size_t total_without_smem_epilogue = smem_a + smem_b;
  const size_t total_with_noreuse_smem_epilogue = smem_a + smem_b + smem_c;
  // Even if we actually do wind up re-claiming smem_a and smem_b, if we
  // cannot prove it at this point then we have to assume it will not be
  // reclaimed.
  const size_t total_with_reused_smem_epilogue = std::max(
      smem_a + smem_b,
      (smem_a_reuse_guaranteed ? 0 : smem_a) +
          (smem_b_reuse_guaranteed ? 0 : smem_b) + smem_c);

  // Regardless of occupancy considerations, if we cannot fit an smem epilogue
  // without reuse then we must promote reuse
  bool must_reuse = shared_memory_available < total_with_noreuse_smem_epilogue;

  // shortcut where occupancy change is ignored.
  if (ignore_occupancy_drop) {
    if (must_reuse) {
      return {shared_memory_available >= total_with_reused_smem_epilogue, true};
    } else {
      return {true, false};
    }
  }

  // use additional shared memory for epilogue if occupancy is not changed.
  // occupancy is estimated using register and shared memory usage.
  auto warp_dims = gemm_tile.cta_tile / gemm_tile.warp_tile;
  const auto warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;
  const auto threads_per_block =
      warp_dims.m * warp_dims.n * warp_dims.k * warp_size;
  const auto threads_per_sm = getThreadsPerSMGivenRegPerThread(255);
  const auto blocks_per_sm_by_register = threads_per_sm / threads_per_block;
  const auto blocks_per_sm_without_smem_epilogue = std::min(
      shared_memory_available / total_without_smem_epilogue,
      (size_t)blocks_per_sm_by_register);
  const auto blocks_per_sm_with_reused_smem_epilogue = std::min(
      shared_memory_available / total_with_reused_smem_epilogue,
      (size_t)blocks_per_sm_by_register);
  const auto blocks_per_sm_with_noreuse_smem_epilogue = std::min(
      shared_memory_available / total_with_noreuse_smem_epilogue,
      (size_t)blocks_per_sm_by_register);

  // Return whether we should use smem for epilogue, and whether syncing for
  // re-use is desired. We avoid the sync if omitting it does not decrease
  // occupancy.
  bool promote_prologue_smem_reuse = must_reuse ||
      blocks_per_sm_with_reused_smem_epilogue !=
          blocks_per_sm_with_noreuse_smem_epilogue;

  return {
      blocks_per_sm_with_reused_smem_epilogue ==
          blocks_per_sm_without_smem_epilogue,
      promote_prologue_smem_reuse};
}

std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_double_buffer_stage,
    const RolesMap& roles_map,
    const bool ignore_occupancy_drop) {
  auto data_types = getMmaDataTypes(roles_map);
  // getMmaDataTypes provides the dtypes of INPUT_A, INPUT_B, and OUTPUT_D.
  // These are the problem types that indicate the gmem IO. We use smem to load
  // INPUT_A and INPUT_B, but instead of OUTPUT_D which is the result of the
  // epilogue, we store mma_result which is the _input_ to the epilogue. In
  // cases where the epilogue contains a cast back down to reduced precision, we
  // will still use Float for the epilogue smem. If we support Double or
  // Complex in the future then we might need a better way to determine this
  // data type.
  data_types[2] = DataType::Float;

  // smem_a and smem_b are guaranteed to be re-used for smem_c as long as:
  //   - they are marked for re-use using promoteReuse
  //   - they are not aliased by another tensor whose lifetime extends past the
  //   start of smem_epilogue's.
  //   - their lifetimes do not overlap smem_epilogue
  //
  // We can guarantee the first condition by calling tv->promoteReuse() in
  // scheduleProlog.
  //
  // The second condition would only be the case if another smem tensor had the
  // same indexing and its lifetime did not overlap. Matmul scheduler only uses
  // smem for these three arrays, so the only candidate for aliasing is C. If C
  // aliases either A or B, the following expression is still valid.
  //
  // The third condition is satisfied in the simple cases where the inputs to
  // the matmul have only this use. However, it could be violated if a or b has
  // other uses that get ordered after the matmul; for example when computing
  // matmul(A, B) + A for square matrices A and B. In that case, the smem tensor
  // resulting from A->cacheAfter() will be used in both the matmul as well as
  // the addition that occurs in the epilogue, extending the lifetime such that
  // it violates the third condition above. In order to avoid errors in these
  // cases, we check that there is no re-use when there is more than one use of
  // either a or b. If there are multiple uses we might wind up re-using memory,
  // but in that case the calculation below will be overly conservative.
  TensorView* a = roles_map.at(MatmulRole::INPUT_A).front();
  TensorView* b = roles_map.at(MatmulRole::INPUT_B).front();
  bool smem_a_reuse_guaranteed = a->uses().size() == 1;
  bool smem_b_reuse_guaranteed = b->uses().size() == 1;

  return generateSharedMemoryEpilogueHeuristics(
      gemm_tile,
      smem_double_buffer_stage,
      data_types,
      smem_a_reuse_guaranteed,
      smem_b_reuse_guaranteed,
      ignore_occupancy_drop);
}

void scheduleWarpTileWithReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  // Do not split K dimension of CTA tile into multiple warp tiles
  NVF_CHECK(
      cta_tile.k == warp_tile.k,
      "CTA tile and warp tile must have same K dimension");

  mma_utils::checkDimSize(
      tv, {-3, -2, -1}, {cta_tile.m, cta_tile.n, cta_tile.k});

  //       -3   -2  -1
  //[...    M,   N,  K]
  // Distribute warp tile:
  tv->split(-3, warp_tile.m);
  tv->split(-2, warp_tile.n);

  //  -5   -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw   K]
  tv->split(-4, instruction_tile.m);
  tv->split(-2, instruction_tile.n);
  tv->split(-1, instruction_tile.k);

  //   -8  -7 -6 -5 -4 -3  -2 -1
  // [Mwo Mw Mi Nwo Nw Ni Kwo Ki]

  tv->reorder({{-7, -5}, {-6, -3}, {-5, -6}, {-3, -2}, {-2, -8}, {-8, -7}});
  //   -8  -7 -6  -5 -4 -3 -2 -1
  // [Kwo Mwo Nwo Mw Nw Mi Ni Ki]
}

void scheduleWarpTileWithNoReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  mma_utils::checkDimSize(tv, {-2, -1}, {cta_tile.m, cta_tile.n});

  NVF_CHECK(
      cta_tile.k % warp_tile.k == 0,
      "Number of warp on k dimension need to be integer");

  int64_t num_warp_k = cta_tile.k / warp_tile.k;

  //        -2  -1
  //[...    M,   N]

  // Distribute warp tile:
  tv->split(-2, warp_tile.m);
  tv->split(-1, warp_tile.n);

  //  -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw ]
  tv->split(-3, instruction_tile.m);
  tv->split(-1, instruction_tile.n);

  //  -6 -5  -4 -3 -2 -1
  // [Mwo Mw Mi Nwo Nw Ni]

  tv->reorder({{-5, -4}, {-4, -2}, {-3, -5}, {-2, -3}});

  //  -6   -5  -4 -3 -2 -1
  // [Mwo  Nwo Mw Nw Mi Ni]

  if (num_warp_k != 1) {
    // The non reduction warps are merged together
    //  to save one thread dim for cross dim reduce.
    tv->merge(-6);
    //  -5  -4 -3 -2 -1
    // [MNo Mw Nw Mi Ni]
  }
}

//! Split the innermost dim to a vectorized load
void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int64_t vector_word,
    bool vectorize) {
  auto warp_dims = tile.cta_tile / tile.warp_tile;
  int64_t num_of_thread = warp_dims.m * warp_dims.n * warp_dims.k * 32;

  tv->split(-1, num_of_thread * vector_word);
  tv->split(-1, vector_word);
  // [..., thread, vec]
  // distribute to warp: for tidx
  tv->split(-2, 32);

  //      -3    -2    -1
  // [...warp, lane, vec]

  if (warp_dims.k == 1) {
    //      -4     -3    -2    -1
    // [...warpM, warpN, lane, vec]
    tv->split(-3, warp_dims.n);
  } else {
    //      -4     -3    -2    -1
    // [...warpMN, warpR, lane, vec]
    tv->split(-3, warp_dims.k);
  }

  if (vectorize) {
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  tv->axis(-2)->parallelize(ParallelType::TIDx);
  tv->axis(-3)->parallelize(ParallelType::TIDy);
  tv->axis(-4)->parallelize(ParallelType::TIDz);
}

void makeTile(TensorView* tv, std::vector<int64_t> tile_sizes) {
  NVF_CHECK(
      tv->getLeafDomain().size() >= tile_sizes.size(),
      "Tensor dimension less than tile dimension!");

  // Number of inner dimensions we are tiling.
  const int64_t tile_dimension_size = (int64_t)tile_sizes.size();

  // Split the inner dimensions:
  for (int64_t idx : c10::irange(tile_dimension_size)) {
    // Using negative indexing to accomodate potential batching
    //  dimensions on the further left. Eg.:
    //  0, 1, 2   ->         -3,-2,-1
    // [M, N, K]  -> [B0, B1, M, N, K]
    tv->split(idx - tile_dimension_size, tile_sizes.at(idx));
  }

  // The transformation happened should look like:
  //   Before               After
  // [..., M, N, K] -> [..., Mo, Mi, No, Ni, Ko, Ki]

  // Re-order the tiles so that all the outer tiles are
  //  on the left of all the inner tiles
  std::unordered_map<int64_t, int64_t> reorder_map_old_to_new;

  // Number of tiled inner dimensions after we split.
  const auto split_tile_dimension_size = 2 * tile_dimension_size;
  for (auto idx : c10::irange(split_tile_dimension_size)) {
    // We want to reorder as follows:
    //           Before
    //
    // [..., Mo, Mi, No, Ni, Ko, Ki] ->
    //                 After
    //      vvv group0 vvv  vvv group1 vvv
    // [..., Mo, No, Ko,     Mi, Ni, Ki]

    // The index offset within group of current
    //  iterdomain, with grouping specified above.
    auto index_within_group = idx / 2;

    // The index of the group the current id belongs
    //  to, as specified above.
    auto group_index = idx % 2;

    // Calculate the actual index after reordering
    auto index_after_reorder =
        group_index * tile_dimension_size + index_within_group;

    // Add pair {idx_before, idx_after} to re-order map.
    reorder_map_old_to_new.insert(std::make_pair(
        idx - split_tile_dimension_size,
        index_after_reorder - split_tile_dimension_size));
  }

  // Apply the re-order map to tensor
  tv->reorder(reorder_map_old_to_new);
}

namespace {

std::optional<IterDomain*> getMaybeRootIfInnermostTiled(
    IterDomain* id,
    const std::unordered_set<IterDomain*>& maybe_rfactor_id_set) {
  // Root id defaults to an "innermost id".
  while (id->definition() && !maybe_rfactor_id_set.count(id)) {
    if (auto split = dynamic_cast<Split*>(id->definition())) {
      if (id == split->inner()) {
        id = split->in();
        continue;
      }
    }
    // Didn't pass the inner most check, return empty.
    return std::nullopt;
  }

  return id;
}

} // namespace

void orderTiledConcreteIdAsRoot(TensorView* tv) {
  int64_t ndims = tv->nDims();

  // Keep track of the left most position where we will
  //  be reordering the axes.
  int64_t leftmost_pos = ndims;

  // Pull the root id's of the given tv.
  std::unordered_set<IterDomain*> maybe_rfactor_id_set{
      tv->getRFactorDomain().begin(), tv->getRFactorDomain().end()};

  // Keep track of leaf positions that is either a reduction
  //  or a broadcast.
  // Note: Currently don't really see a case where this function
  //  should be called on a reduction output tv, but adding them
  //  here for completeness.
  std::deque<int64_t> broadcast_or_reduction_pos;

  // Map the root id's to their innermost concrete id's
  //  on the leaf.
  std::unordered_map<IterDomain*, int64_t> root_id_to_inner_leaf_pos;

  // Try to re-order inner iterdomains from the innermost
  //  position backward. This utility only tries to re-order
  //  inner tiles on the innermost positions, like the resulting
  //  tensor from makeTile utility.
  // The re-ordering would first try to decide the inner iterdomains
  //  we want to re-order. For this we start from the innermost position
  //  and move back and collect all the iterdomains that we know
  //  are inner tiles of some root domain or broadcast/reduction domains
  //  that won't affect the concrete id layout.
  // The collection process would stop whenever a iterdomain that is
  //  neither an inner tile nor reduction/broadcast is found, and would
  //  not re-order any iterdomain beyond that point to keep the
  //  outer loop structure unchanged.
  for (int64_t i = ndims - 1; i >= 0; i--) {
    auto leaf_id = tv->axis(i);
    if (leaf_id->isBroadcast() || leaf_id->isReduction()) {
      // Register this reduction or broadcast axis
      //  to reorder.
      broadcast_or_reduction_pos.push_front(i);
      leftmost_pos = i;
      continue;
    }
    auto maybe_root =
        getMaybeRootIfInnermostTiled(leaf_id, maybe_rfactor_id_set);

    if (maybe_root.has_value()) {
      // Found an innermost id, add them to the
      //  axes to reorder.
      NVF_ERROR(
          root_id_to_inner_leaf_pos
              .insert(std::make_pair(maybe_root.value(), i))
              .second,
          "Multiple \"innermost\" id seen for root id :",
          maybe_root.value()->toString(),
          " on ",
          tv->toString(),
          " very likely an invariant is broken.");
      leftmost_pos = i;
    } else {
      break;
    }
  }

  // Calculate the ordering:

  // pointer to the current target postion after
  //  repordering
  int64_t current_pos = (int64_t)leftmost_pos;
  std::unordered_map<int64_t, int64_t> reorder_map_old_to_new;

  // first place all the broadcast and reduction on the left:
  for (auto original_broadcast_or_reduction_pos : broadcast_or_reduction_pos) {
    reorder_map_old_to_new[original_broadcast_or_reduction_pos] = current_pos++;
  }

  // Next put all the innermost leaf id's, we make sure that
  //  the inner tile ordering follows the corresponding root
  //  domain ordering by iterating on the root domain and
  //  find their corresponding inner tile iterdomains from
  //  the populated root_id_to_inner_leaf_pos.
  for (auto root_id : tv->getRFactorDomain()) {
    auto leaf_id_pos_it = root_id_to_inner_leaf_pos.find(root_id);
    if (leaf_id_pos_it != root_id_to_inner_leaf_pos.end()) {
      reorder_map_old_to_new[leaf_id_pos_it->second] = current_pos++;
    }
  }

  // Validate that we have processed all inner ids or broadcast/reduction
  //  ids we have registered.
  NVF_ERROR(current_pos == ndims, "Inconsistent ordering logic");

  // Apply the new order:
  tv->reorder(reorder_map_old_to_new);
}

namespace {

// Utility for mma dimension matching
enum class MmaDimension { M = 0, N, K };

// Preliminary checks to try to validate that leaf is
//  a innermost dim of root of exactly the given size.
bool canValidateIsInnerDim(
    IterDomain* root,
    IterDomain* leaf,
    int inner_dim_size) {
  auto expr = leaf->definition();
  if (!leaf->extent()->isConstInt()) {
    return false;
  }
  if (leaf->extent()->evaluate() != inner_dim_size) {
    return false;
  }

  while (expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      // Inner split only
      if (leaf != split->inner()) {
        return false;
      }
      // Const split only
      if (!split->factor()->isConstInt()) {
        return false;
      }
      leaf = split->in();
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // Might consider just rejecting merge.
      auto outer = merge->outer();
      if (outer->isBroadcast()) {
        return false;
      }

      // Only support merging with constant sized dims
      if (!leaf->extent()->isConstInt()) {
        return false;
      }
      leaf = merge->inner();
    } else {
      // No support for swizzled inner dim for now.
      //  Might need to add transpose swizzle here.
      return false;
    }
    expr = leaf->definition();
  }
  return leaf == root;
}

} // namespace

void checkDimSize(
    TensorView* tv,
    std::vector<int64_t> axis,
    std::vector<int64_t> expect) {
  NVF_ERROR(
      axis.size() == expect.size(),
      "CheckDimSize: Mismatched axis and expect size");
  for (auto axis_index : c10::irange(axis.size())) {
    NVF_ERROR(
        ((axis[axis_index] + tv->nDims()) >= 0) &&
            (axis[axis_index] < tv->nDims()),
        "CheckDimSize: axis position out of bound ",
        axis[axis_index],
        " ",
        tv->nDims());
    auto id = tv->axis(axis[axis_index]);
    NVF_CHECK(
        id->extent()->isConstInt(),
        "Mma warp mapping: instruction tile has to be constant");
    NVF_CHECK(
        id->extent()->evaluate() == expect[axis_index],
        "Mma warp mapping: unexpected tile size at",
        axis_index,
        ":",
        id->extent()->evaluate(),
        "vs",
        expect[axis_index],
        "\n for tv: ",
        tv->toString());
  }
}

static void setWarpMapped(TensorView* tv, int64_t number_of_dims) {
  for (int64_t id : c10::irange(number_of_dims)) {
    tv->axis(-id - 1)->toMmaSwizzled();
  }
}

namespace {

// Utility function for mma domain mapping:
//  returns the Iterdomain from the accumulator tv that corresponds
//  to the given mma dimension. See [MMA dimension matching].
std::vector<IterDomain*> getMmaDomains(MmaOp* mma, MmaDimension dimension) {
  // This utility is user facing so shouldn't ever see tensor index here.

  // Note: [Use Root Domain in Accumulator TV]
  //  Have to use root domain for accumulator tv since the operands do not have
  //  root/rfactor domains that map to the rfactor domain of output.
  //  For example:
  //   C[I,I,R,R] = mma (A[I,B,I,I], B[B,I,I,I]),
  //  if we do
  //    c->split(-1,4);
  //    c->rfactor(-1);
  //  on the mma stage we get:
  //   C[I,I,R,Io,R(4)] = mma (A[I,B,I,I], B[B,I,I,I]),
  //  and in this case Io and R(4) would not be able to find root mapping
  //  in A or B.
  //
  //  Essentially in the case of rfactor, this utility does producer side
  //   matching so looking at root domain would be required.
  //  This matching pattern should support most common matmul applications,
  //   but in follow ups we may need to extend RFactor matching if there
  //   are more complex scheduling patterns that we want to support.
  auto accumulator_domain = mma->out()->as<TensorView>()->getMaybeRootDomain();
  auto a_domain = TensorDomain::noReductions(
      mma->inA()->as<TensorView>()->getRFactorDomain());
  auto b_domain = TensorDomain::noReductions(
      mma->inB()->as<TensorView>()->getRFactorDomain());
  NVF_CHECK(
      a_domain.size() == b_domain.size() &&
          a_domain.size() == accumulator_domain.size(),
      "Inconsistent dimensions in mma op",
      a_domain.size(),
      " ",
      b_domain.size(),
      " ",
      accumulator_domain.size());

  std::vector<IterDomain*> result;

  for (auto id_idx : c10::irange(a_domain.size())) {
    // checks if this id should be included in the result
    bool include_this_id = false;
    bool is_broadcast_in_a = a_domain[id_idx]->isBroadcast();
    bool is_broadcast_in_b = b_domain[id_idx]->isBroadcast();
    bool is_reduction_id = accumulator_domain[id_idx]->isReduction();

    switch (dimension) {
      case MmaDimension::K:
        // K dimension is the dimension that is concrete in
        //  operands, and is reduced by mma. This complies with
        //  tensor contraction definition.
        include_this_id =
            !is_broadcast_in_a && !is_broadcast_in_b && is_reduction_id;
        break;
      // M and N dimension below are defined as the iterdomains
      //  that are not reduced by mma, and are concretized in this stage.
      case MmaDimension::M:
        include_this_id =
            !is_broadcast_in_a && is_broadcast_in_b && !is_reduction_id;
        break;
      case MmaDimension::N:
        include_this_id =
            is_broadcast_in_a && !is_broadcast_in_b && !is_reduction_id;
        break;

      default:
        NVF_ERROR(false, "unreachable");
    }

    if (include_this_id) {
      result.push_back(accumulator_domain.at(id_idx));
    }
  }

  return result;
}

//! Variant of getMmaDomains that returns a set
std::unordered_set<IterDomain*> getMmaDomainSet(
    MmaOp* mma,
    MmaDimension dimension) {
  auto mma_domains = getMmaDomains(mma, dimension);
  return {mma_domains.begin(), mma_domains.end()};
}

} // namespace

void WarpMmaSwizzler::scheduleLdMatrix(TensorView* tv, MmaOperand operand) {
  bool transpose = tv->definition()->as<LoadStoreOp>()->opType() ==
      LoadStoreOpType::LdMatrixTranspose;
  // For A, we have an extra outer dim (-6), which is the "warp group". For
  // Hopper, mma instructions executes on warp group level. For Turing/Ampere,
  // this dim will just have extent 1.

  //               A                                   B
  //  -6    -5  -4   -3   -2   -1     or     -5  -4   -3   -2   -1
  //[4moo, 8mi, 4k, 2ko, 2mo, 2ki]         [8ni, 4k, 2ko, 1no, 2ki]
  tv->reorder({{-2, -4}, {-3, -5}});
  //                A                                   B
  //  -6    -5   -4   -3  -2   -1     or     -5   -4   -3  -2   -1
  //[4moo, 2ko, 2mo, 8mi, 4k, 2ki]         [2ko, 1no, 8ni, 4k, 2ki]
  tv->merge(-2);
  //              A                                      B
  //  -5    -4   -3   -2  -1         or          -4   -3   -2   -1
  //[4moo, 2ko, 2mo, 8mi, 8k]                  [2ko, 1no, 8ni, 8k]
  if (transpose) {
    tv->reorder({{-2, -1}});
    //              A                                     B
    //  -5    -4   -3  -2   -1        or          -4   -3  -2   -1
    //[4moo, 2ko, 2mo, 8k, 8mi]                 [2ko, 1no, 8k, 8ni]
  }

  tv->merge(-4);
  tv->merge(-3);
  if (operand == MmaOperand::A) {
    // For A, we have an extra outer dim which is the warp group. Merge it back
    // here so that TIDx represent a warp group, instead of a single warp.
    tv->merge(-3);
  }
  //    A                         B
  // -2  -1         or          -2 -1
  //[128, 8]                   [16, 8]

  // The extent of axis(-2) is the number of threads that contains useful
  // addresses. We can not parallelize axis(-2) directly if the extent is less
  // than 32. Instead, we should split axis(-1) and merge it to axis(-2) to
  // get a complete warp of 32 threads. This makes sure that, during lowering,
  // our system can correctly compute the buffer size.
  int64_t num_tidx_with_addr = tv->axis(-2)->extent()->evaluate().as<int64_t>();
  if (num_tidx_with_addr < 32) {
    int64_t factor = 32 / num_tidx_with_addr;
    tv->split(-1, factor, false);
    tv->reorder({{-2, -3}, {-3, -2}});
    //    -3           -2              -1
    // [factor, num_tidx_with_addr, 8/factor]
    // For indexing, we only care about what we get when the index of axis(-3)
    // is 0. For higher values, they are garbage, and abandoned.
    tv->merge(-3);
  }

  //    A                      B
  // -2  -1        or        -2 -1
  //[128, 8]                [32, 4]

  tv->axis(-2)->parallelize(ParallelType::TIDx);
  // TODO: this is not really vectorization. Change its parallel type to Mma.
  tv->axis(-1)->parallelize(ParallelType::Vectorize);
  setWarpMapped(tv, 2);
}

void WarpMmaSwizzler::scheduleOperandRead(TensorView* tv, MmaOperand operand) {
  // This function works for all mma ops, regardless of the architecture.
  // Operand A and B are slightly different in the sense that operand A can be
  // (>=16)x16 matrix, but operand B can only be 8x16 or 16x16. For operand A,
  // the Hopper one is the most general one. For earlier architectures, we will
  // have some dimensions with size 1 after split, this is fine. Memory format
  // for hopper mma:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-a
  NVF_ERROR(tv->nDims() >= 2);

  //     A                            B
  //  -2   -1          or          -2   -1
  //[64m, 16k]                    [8n, 16k]
  tv->split(-2, 8);
  tv->split(-1, 2);
  tv->split(-2, 4);

  //          A                               B
  // -5  -4  -3  -2  -1      or      -5  -4  -3  -2  -1
  //[8m, 8m, 2k, 4k, 2k']           [1n, 8n, 2k, 4k, 2k']

  if (operand == MmaOperand::A) {
    // For A, we need to have an extra outer dim (-6) for warp group.
    tv->split(-5, 2);
    // On Ampere and Turing, the extent of dim -6 after the split below will be
    // just 1. On Hopper, the dim -6 will be 4 because Hopper warp group
    // instructions have 4x larger m extend than Ampere/Turing.
  }

  //            A                                 B
  // -6  -5  -4  -3  -2  -1      or      -5  -4  -3  -2  -1
  //[4m, 2m, 8m, 2k, 4k, 2k']           [1n, 8n, 2k, 4k, 2k']

  tv->reorder({{-4, -5}, {-5, -2}, {-2, -4}});

  //            A                                B
  // -6  -5  -4  -3  -2  -1     or      -5  -4  -3  -2  -1
  //[4m, 8m, 4k, 2k, 2m, 2k']          [8n, 4k, 2k, 1n, 2k']

  // ldmatrix loads multiple 8x8 matrices from shared memory to registers in a
  // swizzled memory format.
  //   +--------+--------+
  //   |        |        |
  //   |  8x8   |  8x8   |
  //   |        |        |
  //   +--------+--------+
  //   |        |        |
  //   |  8x8   |  8x8   |
  //   |        |        |
  //   +--------+--------+
  // If n_major is true, these 8x8 matrices are visited in the order of:
  // top left -> top right -> bottom left -> bottom right.
  // If n_major is false, these 8x8 matrices are visited in the order of:
  // top left -> bottom left -> top right -> bottom right.
  //
  // In principle, only `n_major = false` should be needed. But unfortunately,
  // we are taking advantage of the ldmatrix large load in a pretty hacky way.
  // For example, for Turing, only m16n8k8 is supported by hardware. But we are
  // also using a fake m16n8k16 and m16n16k16, which uses a single large
  // ldmatrix to load data to register, and run multiple mma instructions to
  // consume these data. In the future, we should only keep the m16n8k8 macro,
  // and schedule m16n8k16 and m16n16k16 more correctly than this current way.
  bool n_major =
      operand == MmaOperand::B && tv->axis(-2)->extent()->evaluate() > 1;
  if (n_major) {
    tv->reorder({{-2, -3}, {-3, -2}});
    // -5  -4  -2  -3  -1
    //[8n, 4k, 1n, 2k, 2k']
  }

  bool set_allocation = ir_utils::isLdMatrixOp(tv->definition());
  if (!set_allocation) {
    for (auto u : tv->uses()) {
      if (u->isA<MmaOp>()) {
        set_allocation = true;
        break;
      }
    }
  }
  if (set_allocation) {
    tv->setAllocationDomain(tv->getLeafDomain(), true);
  }
}

// Reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-swizzling-modes
void WarpMmaSwizzler::scheduleOperandRead(
    TensorView* tv,
    MmaInputSmemSwizzle swizzle) {
  if (swizzle == MmaInputSmemSwizzle::None) {
    // For no-swizzle case, the entire tile are divided into 8x8 core matrices,
    // and each core matrix resides in a contiguous 8*8*2 bytes region in shared
    // memory. [K, M]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, Mo, M8]
    tv->reorder({{-2, -3}});
    // [Ko, Mo, K8, M8]
  } else {
    auto swizzle_size = getBytesFromSwizzle(swizzle) / 16;
    // For example, [K, M]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // For example transpose2 == false
    // [Ko, K8, Mo, M8]
    // Note: the extent of Mo may not be a multiple of swizzle_size, but we
    // still split swizzle_size. If this is the case, effectively we are
    // padding it to a multiple of swizzle_size.
    tv->split(-2, swizzle_size);
    // For example, swizzle_size = 2
    // [Ko, K8, Moo, Mo2, M8]
    tv->split(-4, 8 / swizzle_size);
    // [Ko, K2, K4, Moo, Mo2, M8]
    tv->swizzle(SwizzleType::XOR, -5, -2);
    tv->reorder({{-3, -5}});
    // [Ko, Moo, K2, K4, Mo2, M8]
  }
  tv->setAllocationDomain(tv->getLeafDomain(), true);
}

void WarpMmaSwizzler::scheduleMmaWarpOutput(TensorView* tv) {
  // This function works for all mma ops, regardless of the architecture. The
  // Hopper one is the most general one. For earlier architectures, we will have
  // some dimensions with size 1 after split, this is fine.
  // Memory format for hopper mma:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d

  // Assume last 2 dims, for example [M64, N24] or [M64, N24, R]
  NVF_ERROR(tv->nDims() >= 2);
  bool is_mma_output = tv->definition()->isA<MmaOp>();

  int m_pos = is_mma_output ? -3 : -2;
  int n_pos = is_mma_output ? -2 : -1;

  //   m    n
  // [M64, N24  (,R)]
  tv->split(m_pos--, 8);
  tv->split(m_pos--, 2);
  //   m           n
  // [M4, M2, M8, N24  (,R)]
  tv->split(n_pos, 8);
  tv->split(n_pos, 2);

  n_pos -= 2;
  m_pos -= 2;
  //  m           n
  // [M4, M2, M8, N3, N4, N2  (,R)]

  tv->reorder({{m_pos + 1, n_pos + 1}, {n_pos + 1, m_pos + 2}});
  //  m           n
  // [M4, M8, N4, N3, M2, N2  (,R)]
  tv->merge(m_pos++);
  tv->merge(m_pos++);

  //       m
  // [WarpGroup128, N3, M2, N2  (,R)]

  if (is_mma_output) {
    tv->split(-1, 2);
    tv->split(-2, 4);
    m_pos -= 2;
    //       m
    // [WarpGroup128, N3, M2, N2, Ro, R4, R2]
  }

  NVF_CHECK(tv->definition() != nullptr);

  tv->axis(m_pos)->parallelize(ParallelType::TIDx);

  if (is_mma_output) {
    // Set instruction loops for mma reduce
    int pos = -1;
    while (pos > m_pos) {
      tv->axis(pos--)->parallelize(ParallelType::Mma);
    }
    setWarpMapped(tv, 7);
  }
}

void canonicalizeMmaTvOrdering(TensorView* tv) {
  std::unordered_set<IterDomain*> root_id_set{
      tv->getRFactorDomain().begin(), tv->getRFactorDomain().end()};

  auto mma = dynamic_cast<MmaOp*>(tv->definition());
  NVF_CHECK(
      mma != nullptr, "canonicalizeMmaTvOrdering : only support mma op output");

  auto m_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::M);
  auto n_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::N);
  auto k_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::K);

  std::vector<int64_t> device_pos, batch_pos, prev_reduction_pos, m_pos, n_pos,
      k_pos;

  int64_t ndims = tv->nDims();

  for (auto idx : c10::irange(ndims)) {
    auto id = tv->axis(idx);
    NVF_CHECK(root_id_set.count(id), id->toString(), " not a root id.");

    // Categorize each original iterdomain position
    if (m_id_set.count(id)) {
      m_pos.push_back(idx);
    } else if (n_id_set.count(id)) {
      n_pos.push_back(idx);
    } else if (k_id_set.count(id)) {
      k_pos.push_back(idx);
    } else if (id->isReduction()) {
      prev_reduction_pos.push_back(idx);
    } else if (id->isDeviceDim()) {
      device_pos.push_back(idx);
    } else {
      batch_pos.push_back(idx);
    }
  }

  // Collect all mma id's, other id's would be either
  //  batch or incoming reduction.

  // Ordering map from old position to new position
  //  that we wil build using the position vectors.
  std::unordered_map<int64_t, int64_t> order_map;

  // Running position counter keeping track of the
  //  current insert position in order_map.
  int64_t current_pos = 0;

  // Utility to insert the ordered pos sequences to
  //  the ordering map.
  auto insert_to_order_map =
      [&order_map, &current_pos](const std::vector<int64_t>& original_pos) {
        for (auto pos : original_pos) {
          order_map[pos] = current_pos++;
        }
      };

  // Order the categories, while keeping the original
  //  intra-category ordering.
  insert_to_order_map(device_pos);
  insert_to_order_map(batch_pos);
  insert_to_order_map(prev_reduction_pos);
  insert_to_order_map(m_pos);
  insert_to_order_map(n_pos);
  insert_to_order_map(k_pos);

  // Validate that all of the root ids are covered by
  //  the inserted categories.
  NVF_ERROR(current_pos == ndims, "Id not completely categorized");

  // Apply the new ordering
  tv->reorder(order_map);
}

namespace {

inline void resolveTvToMatmulDomainsMapping(
    DependenciesMap& deps_map,
    const std::vector<TensorView*>& tensors,
    IterDomain* m,
    IterDomain* n,
    IterDomain* k,
    const ComputeAtMap& ca_map) {
  for (const auto tv : tensors) {
    // This ensures all inputs are added to the deps_map.
    // There could be inputs such as a zero-dimensional bias which
    // would otherwise be skipped.
    deps_map[tv] = {};
    for (const auto domain : tv->getLeafDomain()) {
      if (ca_map.areMapped(m, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDomain::M);
        continue;
      }
      if (ca_map.areMapped(n, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDomain::N);
        continue;
      }
      if (ca_map.areMapped(k, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDomain::K);
        continue;
      }
    }
  }
}

} // anonymous namespace

MatmulProblemLayoutOpt getProblemLayout(Fusion* fusion) {
  const std::vector<MatmulPattern> patterns = findMatmulPatterns(fusion);
  if (patterns.size() != 1) {
    std::stringstream ss;
    ss << "Invalid number of MmaOp instances in fusion, expected 1, got "
       << patterns.size();
    return ss.str();
  }
  const MatmulPattern& pattern = patterns[0];
  IdModel id_model(fusion);
  const auto id_roles = pattern.getDimRoles(id_model);
  const auto tensor_roles_opt = getTensorRoles(fusion, id_model, id_roles);
  if (!tensor_roles_opt.isValid()) {
    return {tensor_roles_opt.getErrorMsg()};
  }
  return getProblemLayout(id_model, id_roles, tensor_roles_opt.getData());
}

MatmulProblemLayoutOpt getProblemLayout(
    const IdModel& id_model,
    const std::unordered_map<ValGroup, MatmulDomain>& dim_roles,
    const RolesMap& tensor_roles) {
  // Assumes the exact graph has already been built, since we've been provided
  // dim_roles
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  // Note: using DataWrapperOpt<MatmulDomain> would be preferable here. However,
  // using DataWrapperOpt<MatmulDomain>(std::move(dom)) leads to a clang-tidy
  // warning because MatmulDomain is trivially movable. There is only a move
  // constructor for DataWrapperOpt to prevent inadvertent copying. To avoid
  // this complication I'm using an unwrapped variant for the lambda's result
  // type.
  using UnitDimOpt = std::variant<std::string, UnitDim>;
  const auto findUnitDim =
      [&tensor_roles, &dim_roles, &exact_graph](MatmulRole role) -> UnitDimOpt {
    const auto role_it = tensor_roles.find(role);
    if (role_it == tensor_roles.end()) {
      return "Could not find role in tensor_roles";
    }
    std::optional<MatmulDomain> group_inner_dom = std::nullopt;
    for (TensorView* tv : role_it->second) {
      IterDomain* inner_id =
          TensorDomain::noReductions(tv->getMaybeAllocationDomain()).back();
      const ValGroup& g = exact_graph.toGroup(inner_id);
      auto g_it = dim_roles.find(g);
      if (g_it == dim_roles.end()) {
        return "Inner domain of tensor was not mapped to a MatmulDomain";
      }
      if (!group_inner_dom.has_value()) {
        group_inner_dom = g_it->second;
      } else if (group_inner_dom.value() != g_it->second) {
        return "Group contains multiple inner dimension domains";
      }
    }
    if (!group_inner_dom.has_value()) {
      return "No tensor found in role";
    }
    return group_inner_dom.value() == MatmulDomain::K ? UnitDim::K
                                                      : UnitDim::M_or_N;
  };

  const UnitDimOpt unitdim_a_opt = findUnitDim(MatmulRole::INPUT_A);
  if (std::holds_alternative<std::string>(unitdim_a_opt)) {
    std::string err = std::get<std::string>(unitdim_a_opt);
    return err;
  }
  const UnitDimOpt unitdim_b_opt = findUnitDim(MatmulRole::INPUT_B);
  if (std::holds_alternative<std::string>(unitdim_b_opt)) {
    std::string err = std::get<std::string>(unitdim_b_opt);
    return err;
  }
  const UnitDim unitdim_a = std::get<UnitDim>(unitdim_a_opt);
  const UnitDim unitdim_b = std::get<UnitDim>(unitdim_b_opt);

  if (unitdim_a == UnitDim::K && unitdim_b == UnitDim::K) {
    return MmaLayout::TN;
  } else if (unitdim_a == UnitDim::K && unitdim_b == UnitDim::M_or_N) {
    return MmaLayout::TT;
  } else if (unitdim_a == UnitDim::M_or_N && unitdim_b == UnitDim::M_or_N) {
    return MmaLayout::NT;
  } else if (unitdim_a == UnitDim::M_or_N && unitdim_b == UnitDim::K) {
    return MmaLayout::NN;
  }
  NVF_ERROR(false, "Reached unreachable section of getProblemLayout");
}

RolesMapOpt getTensorRoles(
    Fusion* fusion,
    const IdModel& id_model,
    const std::unordered_map<ValGroup, MatmulDomain>& dim_roles) {
  const auto mma_input_candidates =
      ir_utils::filterByType<TensorView>(fusion->inputs()).vector();
  if (mma_input_candidates.empty()) {
    return {"Failed to find any TV that is fusion input"};
  }
  const auto mma_output_candidates =
      ir_utils::filterByType<TensorView>(fusion->outputs()).vector();
  if (mma_output_candidates.empty()) {
    return {"Failed to find any TV that is fusion output"};
  }

  RolesMap tensor_roles;

  // Assumes the exact graph has already been built, since we've been provided
  // dim_roles
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  struct DimPresence {
    bool m = false;
    bool n = false;
    bool k = false;
    bool unmapped = false;
  };

  const auto findDims = [&dim_roles, &exact_graph](TensorView* tv) {
    DimPresence has;
    for (IterDomain* id : TensorDomain::noReductions(tv->getLeafDomain())) {
      if (id->isBroadcast() || id->isDeviceDim()) {
        // Broadcast and device domains won't exact map to concrete domains so
        // skip them
        continue;
      }
      const ValGroup& g = exact_graph.toGroup(id);
      auto it = dim_roles.find(g);
      if (it == dim_roles.end()) {
        // tv has an unmapped non-broadcast and non-reduction dimension
        has.unmapped = true;
        continue;
      }
      has.m = has.m || it->second == MatmulDomain::M;
      has.n = has.n || it->second == MatmulDomain::N;
      has.k = has.k || it->second == MatmulDomain::K;
    }
    return has;
  };

  for (TensorView* tv : mma_input_candidates) {
    DimPresence has = findDims(tv);
    if (has.unmapped) {
      // Don't map TVs to roles if they have unmapped dims
      continue;
    }
    if (has.m && has.k && !has.n) {
      tensor_roles[MatmulRole::INPUT_A].push_back(tv);
      continue;
    }
    if (has.n && has.k && !has.m) {
      tensor_roles[MatmulRole::INPUT_B].push_back(tv);
      continue;
    }
    // Bias vectors are assigned to INPUT_C role
    if (!has.k) {
      tensor_roles[MatmulRole::INPUT_C].push_back(tv);
      continue;
    }
  }

  std::vector<TensorView*> storage;
  for (TensorView* tv : mma_output_candidates) {
    DimPresence has = findDims(tv);
    // NOTE: depending on fusion definition k domain may appear in the output:
    //  - for mma_output == fusion output k domain is present
    //  - for mma_output != fusion output (fusion with epilogue) k domain
    //    is not present
    if (has.k || has.unmapped) {
      // Don't map TVs to output roles if they have unmapped dims, or if they
      // have K dimension
      continue;
    }

    // NOTE: the core fusion output tensors are the ones with m and n
    //  domains
    if (has.m && has.n) {
      storage.push_back(tv);
    }
  }

  if (!storage.empty()) {
    tensor_roles[MatmulRole::OUTPUT_D] = storage;
  }

  for (auto& [role, tvs] : tensor_roles) {
    // NOTE: sort role tvs in descending order by uses() size, and
    //  if equal then by name() to ensure the stable ordering of tensor
    //  views in collections assigned to the supported roles
    std::sort(tvs.begin(), tvs.end(), [](TensorView* a, TensorView* b) {
      return (a->uses().size() == b->uses().size())
          ? (a->name() < b->name())
          : (a->uses().size() > b->uses().size());
    });
  }

  return tensor_roles;
}

namespace {

// Check the val (in) is the output of broadcast.
// Then check the output of the broadcast is 3D (4D for bmm).
bool hasValidBroadcastOp(TensorView* bcast_out) {
  // First check the tensorsview is 3D (4D)
  // and has one broadcast dim.
  // Ignore device dimensions in this analysis.
  auto non_device_dims =
      TensorDomain::noDevices(bcast_out->getLeafDomain()).size();
  if (!((non_device_dims == 3 || non_device_dims == 4) &&
        TensorDomain::noDevices(bcast_out->domain()->noBroadcasts()).size() ==
            non_device_dims - 1)) {
    return false;
  }

  // Check if the definition is a broadcast op.
  if (dynamic_cast<BroadcastOp*>(bcast_out->definition())) {
    return true;
  }

  return false;
}

int64_t numBroadcastDeviceDims(TensorView* tv) {
  return std::count_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      [](IterDomain* id) { return id->isDeviceDim() && id->isBroadcast(); });
}

// This function checks if the mul-sum can be replace with a mma op. The checks
// are:
// 1. The inputs to the muls are broadcast ops.
// 2. The broadcasts have 2D or 3D(bmm) inputs.
// 3. The broadcasts only broadcast one dim and the dims are different for the 2
// muls.
// 4. There is a single reduction dim, and that dim that is not either of the
// broadcast dims.
bool broadcastsAreValid(
    TensorView* left,
    TensorView* right,
    unsigned int reduction_axis) {
  if (!(hasValidBroadcastOp(left) && hasValidBroadcastOp(right))) {
    return false;
  }

  auto bcast_l = dynamic_cast<BroadcastOp*>(left->definition());
  auto bcast_r = dynamic_cast<BroadcastOp*>(right->definition());

  // Ensure that only one non-device dim is getting broadcast.
  auto bcastFlags_l = bcast_l->getBroadcastDimFlags();
  auto bcastFlags_r = bcast_r->getBroadcastDimFlags();
  auto bcast_l_devices = numBroadcastDeviceDims(left);
  auto bcast_r_devices = numBroadcastDeviceDims(right);
  auto count_l = std::count(bcastFlags_l.begin(), bcastFlags_l.end(), true) -
      bcast_l_devices;
  auto count_r = std::count(bcastFlags_r.begin(), bcastFlags_r.end(), true) -
      bcast_r_devices;
  if ((count_l != 1) || (count_l != count_r)) {
    return false;
  }

  // Also ensure that it's not the same dim for the two muls. that's
  // getting broadcast.
  auto idx_l = std::find(bcastFlags_l.begin(), bcastFlags_l.end(), true) -
      bcastFlags_l.begin();
  auto idx_r = std::find(bcastFlags_r.begin(), bcastFlags_r.end(), true) -
      bcastFlags_r.begin();
  if (idx_l == idx_r) {
    return false;
  }

  // Also ensure that the reduction dim is not either of the broadcast dim.
  if (reduction_axis == idx_l || reduction_axis == idx_r) {
    return false;
  }

  // Check different dimensions are the broadcast dims.
  return true;
}

// If the tensorview is a output of a cast operation, then
// return the input to the cast operation, else return the tensorview.
TensorView* getTensorviewPriorToCast(TensorView* in) {
  if (auto uCastOp = dynamic_cast<UnaryOp*>(in->definition());
      uCastOp && uCastOp->getUnaryOpType() == UnaryOpType::Cast) {
    return static_cast<TensorView*>(uCastOp->in());
  }
  return in;
}

} // namespace

char dtypeToChar(const DataType& dtype) {
  if (dtype == DataType::Half) {
    return 'H';
  } else if (dtype == DataType::BFloat16) {
    return 'T';
  } else if (dtype == DataType::Float) {
    return 'S';
  } else if (dtype == DataType::Double) {
    return 'D';
  }
  NVF_ERROR(false, "Unsupported dtype for matmul: ", dtype);
  return 0;
}

namespace {

class MatmulPatternMatcher : IterVisitor {
 public:
  static std::vector<MatmulPattern> run(Fusion* fusion) {
    MatmulPatternMatcher matcher;
    matcher.traverse(fusion);
    return matcher.patterns_;
  }

 private:
  using IterVisitor::handle;

  // TODO: These methods currently assume the output will have allocation domain
  // equal to its rfactor. However, if the rfactor domain is specified, or if
  // there is a transpose operation in the epilogue, then this assumption will
  // be violated. In such cases we should actually swap and transpose A and B.

  // Handle the case when no translation is needed.
  void handle(MmaOp* mop) override {
    MatmulPattern& pattern = patterns_.emplace_back();
    pattern.A = mop->inA()->as<TensorView>();
    pattern.B = mop->inB()->as<TensorView>();
    pattern.output = mop->out()->as<TensorView>();
  }

  void handle(ReductionOp* rop) override {
    // Check if operation is a sum.
    if (rop->getReductionOpType() != BinaryOpType::Add) {
      return;
    }
    // Then check if the producer of the sum is a mul.
    if (auto bop = dynamic_cast<BinaryOp*>(rop->in()->definition())) {
      if (bop->getBinaryOpType() != BinaryOpType::Mul) {
        return;
      }
      // Remember that we are just gathering the immediate inputs to the
      // matmul, so there should be no prologue between a, b and the mul/sum.

      // Check that the inputs have broadcasts that are not all in common, i.e.
      // that there is at least one M and at least one N dimension.

      // Note that there might be a cast to Float just before the multiply. This
      // happens when using the `mul` op with reduced precision inputs. It can
      // also happen if the inputs to `mul` in the definition were Float, but
      // the Fusion was segmented and casts to half precision were inserted at
      // the segmentation edge (see castInputOutputToLowerPrecision in
      // fusion_segmenter.cpp).
      TensorView* ltv = dynamic_cast<TensorView*>(bop->lhs());
      TensorView* rtv = dynamic_cast<TensorView*>(bop->rhs());
      if (ltv == nullptr || rtv == nullptr) {
        // Found a scalar input
        return;
      }
      ltv = getTensorviewPriorToCast(ltv);
      rtv = getTensorviewPriorToCast(rtv);

      std::vector<IterDomain*> lrf = TensorDomain::noDevices(
          TensorDomain::noReductions(ltv->getLeafDomain()));
      std::vector<IterDomain*> rrf = TensorDomain::noDevices(
          TensorDomain::noReductions(rtv->getLeafDomain()));

      // These sizes should match since ops::maybeBroadcast places BroadcastOps
      // for implicit broadcasting.
      NVF_ERROR(lrf.size() == rrf.size());
      const std::vector<IterDomain*>& red_root = TensorDomain::noDevices(
          rop->out()->as<TensorView>()->getMaybeRootDomain());
      NVF_ERROR(red_root.size() == lrf.size());
      // Find innermost M or N dimension in output
      // We will assume for now that the output rfactor domain matches the
      // fusion output's allocation domain; in particular that the innermost
      // dimension is an N dimension. This allows us to determine which of lhs
      // and rhs is A and B.
      // TODO: analyze fusion outputs to determine N dimensions
      bool lhs_is_A = true;
      bool has_m = false, has_n = false;
      // Loop backwards to find inner-most Iteration domain in output
      for (int64_t i = (int64_t)red_root.size() - 1; i >= 0; --i) {
        IterDomain* lhs_id = lrf[(size_t)i];
        IterDomain* rhs_id = rrf[(size_t)i];
        IterDomain* out_id = red_root[(size_t)i];
        if (out_id->isIteration()) {
          if (lhs_id->isBroadcast() != rhs_id->isBroadcast()) {
            // This is either an M or N dimension

            // Operand domains must be Broadcast and Iteration
            NVF_ERROR(lhs_id->isIteration() || rhs_id->isIteration());

            if (!has_n) {
              // This is the inner-most output non-batch dim, so it is N
              has_n = true;
              // rhs is B if it has this dimension
              lhs_is_A = rhs_id->isIteration();
              continue;
            }
            // We have found the inner-most N dim, so we can now use lhs_is_A to
            // tell whether this is M or N
            has_m = has_m || (lhs_is_A && lhs_id->isIteration()) ||
                (!lhs_is_A && (rhs_id->isIteration()));
          }
          // out_id could also be a batch dim
        } else if (out_id->isReduction()) {
          // matmul must be contraction of non-broadcast dimensions
          if (!lhs_id->isIteration() || !rhs_id->isIteration()) {
            return;
          }
        } else if (!out_id->isBroadcast()) {
          // Reduction output ID should be iteration, reduction, or broadcast
          return;
        }
      }
      if (!has_m || !has_n) {
        // This is an ordinary reduction or mat-vec, not a matmul
        return;
      }

      MatmulPattern& pattern = patterns_.emplace_back();
      pattern.A = lhs_is_A ? ltv : rtv;
      pattern.B = lhs_is_A ? rtv : ltv;
      pattern.output = rop->out()->as<TensorView>();
    }
  }

 private:
  std::vector<MatmulPattern> patterns_;
};

} // namespace

std::vector<MatmulPattern> findMatmulPatterns(Fusion* fusion) {
  return MatmulPatternMatcher::run(fusion);
}

std::string MatmulPattern::toString() const {
  std::stringstream ss;
  ss << "MatmulPattern{";
  ss << "\n  A=" << A->toString();
  ss << "\n  B=" << B->toString();
  ss << "\n  output=" << output->toString() << "\n}";
  return ss.str();
}

MmaOp* MatmulPattern::translateToMmaOp() {
  if (auto mma_op = dynamic_cast<MmaOp*>(output->definition())) {
    // No translation needed
    return mma_op;
  } else if (output->definition()->isA<ReductionOp>()) {
    Val* init = IrBuilder::create<Val>(0.0, output->dtype());
    // This replaces the mul and sum by overwriting output->definition()
    return IrBuilder::create<MmaOp>(output, A, B, init);
  }
  NVF_ERROR(
      false,
      "Could not translate matmul pattern with output ",
      output->toString(),
      " to MmaOp");
}

std::unordered_map<ValGroup, MatmulDomain> MatmulPattern::getDimRoles(
    IdModel& id_model) const {
  id_model.maybeBuildGraph(IdMappingMode::EXACT);
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  // There are four types of ValGroup involved in a MatmulPattern: M, N, K, and
  // Batch. These are enumerated in the MatmulDomain enum class. They are
  // defined by their membership as follows:
  //   M: present in A and output, but not B
  //   N: present in B and output, but not A
  //   K: present in A and B, but not output
  //   Batch: present in all A, B, and output
  // If there are other patterns, for example a ValGroup present in only A, then
  // we should raise an exception here.

  // Indicates whether a ValGroup is present in A (bit 0), B (bit 1), or output
  // (bit 2)
  using DimPresence = std::bitset<3>;

  std::unordered_map<ValGroup, DimPresence> present_flags;
  const auto recordPresence = [&exact_graph, &present_flags](
                                  TensorView* tv, size_t tensor_num) {
    for (IterDomain* id : tv->getLeafDomain()) {
      if (id->isReduction() || id->isBroadcast() || id->isDeviceDim()) {
        // ignore device, reductions, and broadcasts since they don't exact map
        // to problem dims in the generated kernel
        continue;
      }
      const ValGroup& g = exact_graph.toGroup(id);
      present_flags[g].set(tensor_num);
    }
  };
  recordPresence(A, 0);
  recordPresence(B, 1);
  recordPresence(output, 2);

  std::unordered_map<ValGroup, MatmulDomain> dim_roles;
  for (const auto& [g, flags] : present_flags) {
    if (flags.all()) {
      dim_roles[g] = MatmulDomain::Batch;
    } else if (flags.test(0) && flags.test(1)) {
      dim_roles[g] = MatmulDomain::K;
    } else if (flags.test(0) && flags.test(2)) {
      dim_roles[g] = MatmulDomain::M;
    } else if (flags.test(1) && flags.test(2)) {
      dim_roles[g] = MatmulDomain::N;
    } else {
      NVF_ERROR(
          false,
          "IterDomain ValGroup should be present in at least two of A, B, and output. flags: ",
          flags);
    }
  }

  return dim_roles;
}

} // namespace mma_utils

} // namespace nvfuser
