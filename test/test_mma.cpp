// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <test/utils.h>
#include <test/validator.h>

#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>

namespace nvfuser {

using MmaTestParams = std::tuple<MmaMacro, DataType>;

class MmaTest : public NVFuserFixtureParamTest<MmaTestParams> {
  void SetUp() override {
    // requires Hopper or newer
    if (cudaArchGuardShouldSkip(7, 5)) {
      GTEST_SKIP() << "skipping tests on pre-Turing GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// MMA unit test on Turing
TEST_P(MmaTest, TN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto dtype = std::get<1>(GetParam());

  if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
  }

  // [M, K]
  auto tv0 = makeConcreteTensor({getM(macro), getK(macro)}, dtype);
  // [N, K]
  auto tv1 = makeConcreteTensor({getN(macro), getK(macro)}, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // [M, N, K] -> [N, M, K]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  tv0b->merge(1);
  tv0b->merge(1);
  tv0b->axis(1)->parallelize(ParallelType::TIDx);
  tv1b->merge(1);
  tv1b->axis(1)->parallelize(ParallelType::TIDx);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({getM(macro), getK(macro)}, options);
  auto t1 = at::randn({getN(macro), getK(macro)}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Turing
TEST_P(MmaTest, TT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto dtype = std::get<1>(GetParam());

  if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
  }

  // [M, K]
  auto tv0 = makeConcreteTensor({getM(macro), getK(macro)}, dtype);
  // [K, N]
  auto tv1 = makeConcreteTensor({getK(macro), getN(macro)}, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  // [M, K, N]
  auto tv1b = broadcast(tv1, {true, false, false});
  // [M, N, K]
  auto tv1t = transpose(tv1b, 1, 2);

  auto tv2 = fusedMultiplySum(tv0b, tv1t, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // [M, N, K] -> [N, M, K]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);
  tv1t->applyMmaSwizzle(MmaOperand::B);

  tv0b->merge(1);
  tv0b->merge(1);
  tv0b->axis(1)->parallelize(ParallelType::TIDx);
  tv1t->merge(1);
  tv1t->axis(1)->parallelize(ParallelType::TIDx);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({getM(macro), getK(macro)}, options);
  auto t1 = at::randn({getK(macro), getN(macro)}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Turing
TEST_P(MmaTest, NT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto dtype = std::get<1>(GetParam());

  if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
  }

  // [K, M]
  auto tv0 = makeConcreteTensor({getK(macro), getM(macro)}, dtype);
  // [K, N]
  auto tv1 = makeConcreteTensor({getK(macro), getN(macro)}, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv1t = permute(tv1b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1t, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // [K,M,N] -> [N,M,K]
  tv0t->reorder({{-2, -3}, {-3, -2}});
  tv0t->applyMmaSwizzle(MmaOperand::A);
  tv1t->applyMmaSwizzle(MmaOperand::B);

  tv0t->merge(1);
  tv0t->merge(1);
  tv0t->axis(1)->parallelize(ParallelType::TIDx);
  tv1t->merge(1);
  tv1t->axis(1)->parallelize(ParallelType::TIDx);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({getK(macro), getM(macro)}, options);
  auto t1 = at::randn({getK(macro), getN(macro)}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_P(MmaTest, NN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto dtype = std::get<1>(GetParam());

  if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
  }

  // [K, M]
  auto tv0 = makeConcreteTensor({getK(macro), getM(macro)}, dtype);
  // [N, K]
  auto tv1 = makeConcreteTensor({getN(macro), getK(macro)}, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  // [M, N, K]
  auto tv1b = broadcast(tv1, {true, false, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1b, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // [M, N, K] -> [N, M, K]
  tv0t->reorder({{-2, -3}, {-3, -2}});
  tv0t->applyMmaSwizzle(MmaOperand::A);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  tv0t->merge(1);
  tv0t->merge(1);
  tv0t->axis(1)->parallelize(ParallelType::TIDx);
  tv1b->merge(1);
  tv1b->axis(1)->parallelize(ParallelType::TIDx);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({getK(macro), getM(macro)}, options);
  auto t1 = at::randn({getN(macro), getK(macro)}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    SingleTile,
    MmaTest,
    testing::Values(
        std::make_tuple(MmaMacro::Turing_16_8_8, DataType::Half),
        std::make_tuple(MmaMacro::Turing_16_8_16, DataType::Half),
        std::make_tuple(MmaMacro::Turing_16_16_16, DataType::Half),
        std::make_tuple(MmaMacro::Ampere_16_8_16, DataType::Half),
        std::make_tuple(MmaMacro::Ampere_16_16_16, DataType::Half),
        std::make_tuple(MmaMacro::Ampere_16_8_16, DataType::BFloat16),
        std::make_tuple(MmaMacro::Ampere_16_16_16, DataType::BFloat16)),
    [](const testing::TestParamInfo<MmaTestParams>& info) {
      std::ostringstream os;
      auto macro = std::get<0>(info.param);
      auto dtype = std::get<1>(info.param);
      os << toString(macro) << "_" << dtype;
      return os.str();
    });

} // namespace nvfuser
