/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_analysis.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class WhileLoopAnalysisTest : public HloTestBase {};

TEST_F(WhileLoopAnalysisTest, SingleIterationUpperBound) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      const = s32[] constant(-1)
      ROOT root = (f32[2], s32[]) tuple(val, const)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  EXPECT_EQ(*ComputeWhileLoopTripCountUpperBound(while_op), 1);
}

TEST_F(WhileLoopAnalysisTest, NoUpperBound) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      const = s32[] constant(42)
      ROOT root = (f32[2], s32[]) tuple(val, const)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCountUpperBound(while_op), absl::nullopt);
}

TEST_F(WhileLoopAnalysisTest, ExactBound) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      index = s32[] get-tuple-element(p_body), index=1
      one = s32[] constant(1)
      inc = s32[] add(index, one)
      ROOT root = (f32[2], s32[]) tuple(val, inc)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=LT
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] constant(0)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  EXPECT_EQ(*ComputeWhileLoopTripCountUpperBound(while_op), 42);
}

}  // namespace
}  // namespace xla
