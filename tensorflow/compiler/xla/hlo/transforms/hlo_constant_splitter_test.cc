/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/hlo/transforms/hlo_constant_splitter.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using HloConstantSplitterTest = HloTestBase;

TEST_F(HloConstantSplitterTest, SplitConstants) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      param = (f32[], f32[]) parameter(0),
        sharding={{maximal device=0}, {maximal device=0}}
      gte0 = f32[] get-tuple-element(param), index=0
      gte1 = f32[] get-tuple-element(param), index=1
      constant = f32[] constant(94.1934)
      add1 = f32[] add(constant, gte0)
      add2 = f32[] add(constant, gte1)
      ROOT root = (f32[], f32[], f32[]) tuple(constant, add1, add2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(HloConstantSplitter().Run(module.get()).status());

  // Check that every constant has at most one user.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        EXPECT_LE(instruction->user_count(), 1);
      }
    }
  }
}

TEST_F(HloConstantSplitterTest, PreservingConstantsWithZeroUsers) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      param = (f32[], f32[]) parameter(0),
        sharding={{maximal device=0}, {maximal device=0}}
      gte0 = f32[] get-tuple-element(param), index=0
      gte1 = f32[] get-tuple-element(param), index=1
      constant1 = f32[] constant(94.1934)
      constant2 = f32[] constant(9.1934)
      ROOT root = (f32[], f32[]) tuple(gte0, gte1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter();
  const auto status_or = HloTestBase::RunHloPass(&pass, module.get());
  TF_ASSERT_OK(status_or.status());
  // Verify that the changed flag returned is correct.
  EXPECT_FALSE(status_or.value());
}

TEST_F(HloConstantSplitterTest, SplittingExpressions) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      gte0 = f32[1024] parameter(0)
      gte1 = f32[1024] parameter(1)
      constant1 = f32[1024] iota(), iota_dimension=0
      constant2 = f32[] constant(9.1934)
      constant3 = f32[] constant(0.0)
      constant4 = f32[] constant(1.0)
      b = f32[1024] broadcast(constant2), dimensions={}
      b2 = f32[1024] broadcast(constant3), dimensions={}
      b3 = f32[1024] broadcast(constant4), dimensions={}
      cmp = pred[1024] compare(constant1, b), direction=LT
      %s = f32[1024] select(cmp, b2, b3)
      a1 = f32[1024] add(s, gte0)
      a2 = f32[1024] add(s, gte1)
      ROOT root = (f32[1024], f32[1024]) tuple(a1, a2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  HloConstantSplitter pass = HloConstantSplitter(/*split_expressions=*/true);
  const auto status_or = HloTestBase::RunHloPass(&pass, module.get());
  TF_ASSERT_OK(status_or.status());
  // Verify that the changed flag returned is correct.
  EXPECT_TRUE(status_or.value());
  XLA_VLOG_LINES(1, module->entry_computation()->ToString());
  EXPECT_EQ(module->entry_computation()->instruction_count(), 23);
}

}  // namespace
}  // namespace xla
