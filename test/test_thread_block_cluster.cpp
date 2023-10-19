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
  auto ref = tv2->rFactor({1, 4});

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
  // 0.108 ms for cluster
  for (auto n : {vect * tidx * kidx}) {
    test(n);
  }
}


TEST_F(NVFuserTest, GridNormalization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {-1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv1, tv3);
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
  auto ref = tv2->rFactor({-1, -2});

  TransformPropagator propagator(ref);
  MaxRootDomainInfoSpanningTree(ref).traverse(&propagator);

  // ref = [i0, r0/tidx/batch/vect, tidx, batch, vect]
  ref->axis(0)->parallelize(ParallelType::BIDy);
  ref->axis(1)->parallelize(ParallelType::BIDx);
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
    at::Tensor t0 = at::ones({132/kidx, num_elements}, options);
    std::vector<c10::IValue> aten_inputs = {t0};

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs);
    std::vector<at::Tensor> outputs = fe.runFusion(aten_inputs);
    auto t1 = t0.sum({-1});
    auto t2 = t1.unsqueeze(-1);
    auto t3 = t0 + t2;
    testValidate(
        &fusion, outputs, aten_inputs, {t3}, __LINE__, __FILE__);
  };
  // 0.055 ms for grid
  // 0.122 ms for cluster
  for (auto n : {vect * persistent_batch * tidx * kidx}) {
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
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv1, tv3);
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
  auto ref = tv2->rFactor({-1, -2});

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
    auto t1 = t0.sum({-1});
    auto t2 = t1.unsqueeze(-1);
    auto t3 = t0 + t2;
    testValidate(
        &fusion, outputs, aten_inputs, {t3}, __LINE__, __FILE__);
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

TEST_F(NVFuserTest, SIMPLE) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int x = 4096, y = 2048;
  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x, y}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({-1});
  auto t2 = t1.unsqueeze(-1);
  auto t3 = t0 + t2;
  testValidate(&fusion, cg_outputs, aten_inputs, {t3}, __LINE__, __FILE__);
}

} // namespace nvfuser
