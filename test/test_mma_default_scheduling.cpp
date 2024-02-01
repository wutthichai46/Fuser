// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>

#include <expr_evaluator.h>
#include <fusion.h>
#include <ops/all_ops.h>

namespace nvfuser {

class MmaDefaultSchedTest : public NVFuserTest {};

TEST (MmaDefaultSchedTest, MarkForExprEval){
    Fusion fusion;
    FusionGuard fg(&fusion);
    fusion.markMmaForExprEval();
    NVF_CHECK(fusion.isMmaExprEval(), "MmaOp was not marked to be expression evaluated.");
}

} // namespace nvfuser