// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <regex>

#include <debug.h>
#include <fusion.h>
#include <inlining.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <options.h>
#include <scheduler/cache_policy_refiner.h>
#include <test/utils.h>
#include <test/validator.h>
#include <type.h>

namespace nvfuser {

class MemoryTest
    : public NVFuserTest,
      public testing::WithParamInterface<std::tuple<CacheOp, std::string>> {
 protected:
  void expectMatchCount(
      const std::string& text,
      const std::string& pattern,
      const int num_matches) {
    std::regex regex(pattern);
    std::smatch match;
    std::regex_search(text, match, regex);
    EXPECT_EQ(match.size(), num_matches)
        << "Expect " << pattern << " to occur " << num_matches << " time(s).";
  }
};

TEST_P(MemoryTest, LoadCache) {
  CacheOp cache_op = std::get<0>(GetParam());
  std::string cache_op_str = std::get<1>(GetParam());

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 =
      ops::newValLike(tv0, tv0->getDataType().value())->as<TensorView>();
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, tv1, tv0, cache_op);
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  TensorView* tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->split(0, 4);
  tv1->split(0, 32);
  TransformPropagatorWithCheck propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  // Parallelize LoadStoreOps. Other TensorViews don't support vectorization.
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);
  scheduler_utils::parallelizeAllLike(tv1, {tv3});

  inlineMost();

  at::Tensor input = at::randn(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor expected_output = input + 1.0f;

  FusionExecutor fe;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    fe.compileFusion(&fusion, {input});
  }

  // Verify PTX.
  const executor_utils::CompiledKernel& compiled_kernel = fe.compiledKernel();
  std::string ptx(compiled_kernel.ptx.begin(), compiled_kernel.ptx.end());
  std::regex regex(R"(ld\.global\.)" + cache_op_str + R"(\.\S+)");
  std::smatch match;
  std::regex_search(ptx, match, regex);
  EXPECT_EQ(match.size(), 1);

  // Clean up the dumped PTX file.
  debug() << "Removing " << compiled_kernel.ptx_filename << std::endl;
  std::filesystem::remove(compiled_kernel.ptx_filename);

  // Verify output tensors.
  std::vector<at::Tensor> actual_ts = fe.runFusion({input});
  testValidate(
      &fusion, actual_ts, {input}, {expected_output}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    CacheGlobalLoads,
    MemoryTest,
    testing::Values(
        std::make_tuple(CacheOp::AllLevels, "ca"),
        std::make_tuple(CacheOp::Global, "cg"),
        std::make_tuple(CacheOp::Streaming, "cs")),
    [](const testing::TestParamInfo<std::tuple<CacheOp, std::string>>& info) {
      std::ostringstream os;
      os << std::get<0>(info.param);
      return os.str();
    });

// Use ld.cs when loading streaming data and ld.ca otherwise.
TEST_F(MemoryTest, RefineCachePolicy) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv_a = makeContigTensor(2);
  TensorView* tv_b = makeContigTensor(1);
  fusion.addInput(tv_a);
  fusion.addInput(tv_b);
  TensorView* tv_a2 = set(tv_a);
  TensorView* tv_b2 = set(tv_b);
  TensorView* tv_c = add(tv_a2, tv_b2);
  TensorView* tv_c2 = set(tv_c);
  fusion.addOutput(tv_c2);

  tv_a2->merge(0);
  tv_a2->split(0, 4);
  tv_a2->split(0, 32);
  TransformPropagatorWithCheck propagator(tv_a2);
  MaxRootDomainInfoSpanningTree(tv_a2).traverse(&propagator);

  tv_a2->axis(0)->parallelize(ParallelType::BIDx);
  tv_a2->axis(1)->parallelize(ParallelType::TIDx);
  tv_a2->axis(2)->parallelize(ParallelType::Vectorize);
  tv_b2->axis(2)->parallelize(ParallelType::Vectorize);
  tv_c2->axis(2)->parallelize(ParallelType::Vectorize);

  refineCachePolicy(&fusion);

  inlineMost();

  at::Tensor a = at::randn(
      {1024, 1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor b = at::randn(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor c = a + b;

  FusionExecutor fe;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    fe.compileFusion(&fusion, {a, b});
  }

  // Verify PTX.
  const executor_utils::CompiledKernel& compiled_kernel = fe.compiledKernel();
  std::string ptx(compiled_kernel.ptx.begin(), compiled_kernel.ptx.end());
  expectMatchCount(ptx, R"(ld\.global\.ca\.v4\.\S+)", 1);
  expectMatchCount(ptx, R"(ld\.global\.cs\.v4\.\S+)", 1);

  // Clean up the dumped PTX file.
  debug() << "Removing " << compiled_kernel.ptx_filename << std::endl;
  std::filesystem::remove(compiled_kernel.ptx_filename);

  std::vector<at::Tensor> actual_outputs = fe.runFusion({a, b});
  testValidate(&fusion, actual_outputs, {a, b}, {c}, __LINE__, __FILE__);
}

class TMATest : public NVFuserTest {
  void SetUp() override {
    // requires Hopper or newer
    if (!deviceMajorMinorCheck(9)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

TEST_F(TMATest, StoreCompleteTensor1D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor2D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor3D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor4D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor5D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(5);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);
  tv2->axis(4)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Basically just StoreCompleteTensor1D, but with index hoisting disabled.
// Because index hoisting is responsible making sure that tensor maps are
// created on the host and passed as kernel argument, we need to make sure
// that disabling index hoisting doesn't break this.
TEST_F(TMATest, DisableIndexHoisting) {
  DisableOptionsGuard opt_guard;
  opt_guard.getCurOptions().set(DisableOption::IndexHoist);

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

class ThreadBlockClusterTest : public NVFuserTest {
  void SetUp() override {
    // requires Hopper or newer
    if (!deviceMajorMinorCheck(9)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// Group all blocks into a cluster
TEST_F(ThreadBlockClusterTest, OneCluster) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::KIDx);
  tv1->axis(1)->parallelize(ParallelType::KIDy);
  tv1->axis(2)->parallelize(ParallelType::KIDz);
  scheduler_utils::parallelizeAllLike(tv1);
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto test = [&](int cidx, int cidy, int cidz) {
    constexpr int tidx = 32;
    at::Tensor t0 = at::randn({cidx, cidy, cidz, tidx}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    int64_t gdimx = cidx, gdimy = cidy, gdimz = cidz;
    int64_t cdimx = cidx, cdimy = cidy, cdimz = cidz;
    int64_t bdimx = tidx, bdimy = 1, bdimz = 1;
    LaunchParams lp(
        gdimx, gdimy, gdimz, bdimx, bdimy, bdimz, cdimx, cdimy, cdimz);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lp);

    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs, lp);
    testValidate(&fusion, outputs, aten_inputs, {t0}, __LINE__, __FILE__);
  };
  constexpr int max_cluster_size = 8;
  for (auto x = 1; x <= max_cluster_size; x++) {
    for (auto y = 1; y <= max_cluster_size / x; y++) {
      for (auto z = 1; z <= max_cluster_size / x / y; z++) {
        test(x, y, z);
      }
    }
  }
}

// Group all the blocks in the same rwo into a cluster
// GridDim.x  == clusterDim.x
//            +----+----+----+----+
// cluster-0  | 00 | 01 | 02 | 03 |
//            +----+----+----+----+
// cluster-1  | 10 | 11 | 12 | 13 |
//            +----+----+----+----+
// cluster-2  | 20 | 21 | 22 | 23 |
//            +----+----+----+----+
// This pattern can transform grid reduction using GridDim.x blocks into a
// thread block cluster reduction.
TEST_F(ThreadBlockClusterTest, OneClusterEachRow) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->axis(-1)->parallelize(ParallelType::KIDx);
  tv1->axis(-2)->parallelize(ParallelType::BIDy);
  scheduler_utils::parallelizeAllLike(tv1);
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto test = [&](int cidx, int bidy) {
    at::Tensor t0 = at::randn({bidy, cidx}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    int64_t gdimx = cidx, gdimy = bidy, gdimz = 1;
    int64_t cdimx = cidx, cdimy = 1, cdimz = 1;
    int64_t bdimx = 1, bdimy = 1, bdimz = 1;
    LaunchParams lp(
        gdimx, gdimy, gdimz, bdimx, bdimy, bdimz, cdimx, cdimy, cdimz);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lp);

    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs, lp);
    testValidate(&fusion, outputs, aten_inputs, {t0}, __LINE__, __FILE__);
  };
  constexpr int max_cluster_size = 8;
  for (auto x = 1; x <= max_cluster_size; x++) {
    test(x, 100);
  }
}

TEST_F(NVFuserTest, ClusterReduce) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {-1});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // vectorization
  const int vect = 4;
  tv2->split(-1, vect);

  // block reduction
  const int tidx = 512;
  tv2->split(-2, tidx);

  // cluster reduction
  const int kidx = 8;
  tv2->split(-3, kidx);

  // tv2: [i0, r0/tidx/kidx/vect, kidx, tidx, vect]
  auto ref = tv2->rFactor({1,4});

  TransformPropagator propagator(ref);
  MaxRootDomainInfoSpanningTree(ref).traverse(&propagator);

  // ref = [i0, r0/tidx/kidx, kidx, tidx]
  ref->axis(0)->parallelize(ParallelType::BIDy);
  ref->axis(1)->parallelize(ParallelType::Serial);
  if (std::getenv("USE_CLUSTER")) {
    ref->axis(2)->parallelize(ParallelType::KIDx);
  } else {
    ref->axis(2)->parallelize(ParallelType::BIDx);
  }
  ref->axis(3)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(ref);

  fusion.printMath();
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto test = [&](int num_elements) {
    at::Tensor t0 = at::ones({132 * kidx, num_elements}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs);
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(
        &fusion, outputs, aten_inputs, {t0.sum({-1})}, __LINE__, __FILE__);
  };
  // 0.055 ms for grid
  // 0.122 ms for cluster
  for (auto n : {vect * tidx * kidx}) {
    test(n);
  }
}

TEST_F(NVFuserTest, TMP1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {-1});
  auto tv3 = broadcast(tv2, {false,true});
  auto tv4 = add(tv1,tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  // vectorization
  const int vect = 4;
  tv2->split(-1, vect);

  // persistent batch
  const int persistent_batch = 5;
  tv2->split(-2, persistent_batch);

  // block reduction
  const int tidx = 512;
  tv2->split(-3, tidx);

  // // cluster reduction
  const int kidx = 8;
  // tv2->split(-3, kidx);

  // tv2: [i0, r0/tidx/batch/vect, tidx, batch, vect]
  auto ref = tv2->rFactor({-1,-2});

  TransformPropagator propagator(ref);
  MaxRootDomainInfoSpanningTree(ref).traverse(&propagator);

  // ref = [i0, r0/tidx/batch/vect, tidx, batch, vect]
  ref->axis(0)->parallelize(ParallelType::BIDy);
  ref->axis(1)->parallelize(ParallelType::KIDx);
  ref->axis(2)->parallelize(ParallelType::TIDx);
  ref->axis(3)->parallelize(ParallelType::Serial);
  ref->axis(4)->parallelize(ParallelType::Serial);
  scheduler_utils::parallelizeAllLike(ref);

  fusion.printMath();
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv5->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto test = [&](int num_elements) {
    at::Tensor t0 = at::ones({132 * kidx, num_elements}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs);
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    testValidate(
        &fusion, outputs, aten_inputs, {t0.sum({-1})}, __LINE__, __FILE__);
  };
  // 0.055 ms for grid
  // 0.122 ms for cluster
  for (auto n : {vect * persistent_batch * tidx * kidx}) {
    test(n);
  }
}

TEST_F(NVFuserTest, TMP2) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  auto dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  if (dtype == DataType::Half) {
    tv0 = castOp(DataType::Float, tv0);
  }
  auto tv1 = sum(tv0, {-1});
  if (dtype == DataType::Half) {
    tv1 = castOp(DataType::Float, tv1);
  }
  fusion.addOutput(tv1);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto test = [&](int num_elements) {
    at::Tensor t0 = at::ones({66, num_elements}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    FusionExecutorCache fec(std::move(fusion_ptr));
    auto outputs = fec.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, outputs, aten_inputs, {t0.sum({-1})}, __LINE__, __FILE__);
  };
  // grid reduction: 7 us
  // cluster reduction: 5.7 us
  for (auto n : {49152}) {
    test(n);
  }
}

TEST_F(NVFuserTest, TMP3) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(0.0));
  auto tv2 = add(tv0, IrBuilder::create<Val>(0.0));
  auto tv3 = add(tv0, IrBuilder::create<Val>(0.0));
  auto tv4 = add(tv0, IrBuilder::create<Val>(0.0));

  auto tv1sum = sum(tv1, {1});
  auto tv2sum = sum(tv2, {1});
  auto tv3sum = sum(tv3, {1});
  auto tv4sum = sum(tv4, {1});

  auto tvout12 = add(tv1sum, tv2sum);
  auto tvout34 = add(tv3sum, tv4sum);
  auto tvout = add(tvout12, tvout34);
  fusion.addOutput(tvout);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto test = [&](int num_elements) {
    at::Tensor t0 = at::ones({66, num_elements}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    FusionExecutorCache fec(std::move(fusion_ptr));
    auto outputs = fec.runFusionWithInputs(aten_inputs);
    auto t1 = t0 + 0.0;
    auto t2 = t0 + 0.0;
    auto t3 = t0 + 0.0;
    auto t4 = t0 + 0.0;
    auto t5 = t1 + t2 + t3 + t4;
    auto t6 = t5.sum({-1});
    testValidate(&fusion, outputs, aten_inputs, {t6}, __LINE__, __FILE__);
  };
  // grid reduction: 16 us
  // cluster reduction: 5.7 us
  for (auto n : {49152}) {
    test(n);
  }
}
} // namespace nvfuser
