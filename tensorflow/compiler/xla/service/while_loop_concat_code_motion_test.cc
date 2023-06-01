/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_concat_code_motion.h"

#include <algorithm>
#include <iterator>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class WhileLoopConcatCodeMotionTest : public HloTestBase {};

TEST_F(WhileLoopConcatCodeMotionTest, SimpleMotion) {
  constexpr absl::string_view kHloModule = R"(
    HloModule test

    %cond {
      %param = (s32[], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %constant = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %constant), direction=LT
    }

    %body {
      %param = (s32[], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024,1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %concat = f32[2048,1024] concatenate(%gte.1, %gte.2), dimensions={0}
      %ccall = f32[2048,1024] custom-call(%concat), custom_call_target="test"
      %slice.0 = f32[1024,1024] slice(%ccall), slice={[0:1024], [0:1024]}
      %slice.1 = f32[1024,1024] slice(%ccall), slice={[1024:2048], [0:1024]}
      %ccall2 = f32[1024,1024] custom-call(), custom_call_target="test2"
      %add.0 = f32[1024,1024] add(%slice.0, %ccall2)
      %add.1 = f32[1024,1024] add(%slice.1, %ccall2)
      %t0 = token[] after-all()
      %outfeed = token[] outfeed(%slice.1, %t0)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024,1024], f32[1024,1024])
        tuple(%increment_iteration, %add.0, %add.1)
    }

    ENTRY test_main {
      %param.0 = f32[1024,1024] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %constant.0 = s32[] constant(0)
      %while_init = (s32[], f32[1024,1024], f32[1024,1024]) tuple(%constant.0, %param.0, %param.1)
      ROOT %while = (s32[], f32[1024,1024], f32[1024,1024]) while(%while_init), condition=%cond, body=%body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConcatCodeMotion(2).Run(module.get()));
  ASSERT_TRUE(changed);
  VLOG(1) << module->ToString();
  auto loop = op::While(
      op::Tuple(op::Constant(),
                AllOf(op::Shape("f32[2048,1024]"),
                      op::Concatenate(op::Parameter(0), op::Parameter(1)))));
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(loop), op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop))));
  auto while_op =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(while_op->while_body()->root_instruction(),
              op::Tuple(op::Add(),
                        op::Add(op::CustomCall(),
                                op::Reshape(op::Broadcast(op::CustomCall())))));
}

TEST_F(WhileLoopConcatCodeMotionTest, NoMotionWithChangedElementOrder) {
  constexpr absl::string_view kHloModule = R"(
    HloModule test

    %cond {
      %param = (s32[], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %constant = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %constant), direction=LT
    }

    %body {
      %param = (s32[], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024,1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %concat = f32[2048,1024] concatenate(%gte.1, %gte.2), dimensions={0}
      %ccall = f32[2048,1024] custom-call(%concat), custom_call_target="test"
      %slice.0 = f32[1024,1024] slice(%ccall), slice={[0:1024], [0:1024]}
      %slice.1 = f32[1024,1024] slice(%ccall), slice={[1024:2048], [0:1024]}
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024,1024], f32[1024,1024])
        tuple(%increment_iteration, %slice.1, %slice.0)
    }

    ENTRY test_main {
      %param.0 = f32[1024,1024] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %constant.0 = s32[] constant(0)
      %while_init = (s32[], f32[1024,1024], f32[1024,1024]) tuple(%constant.0, %param.0, %param.1)
      ROOT %while = (s32[], f32[1024,1024], f32[1024,1024]) while(%while_init), condition=%cond, body=%body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConcatCodeMotion(2).Run(module.get()));
  ASSERT_FALSE(changed);
}

TEST_F(WhileLoopConcatCodeMotionTest, CascadedConcats) {
  constexpr absl::string_view kHloModule = R"(
    HloModule test

    %cond {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %constant = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %constant), direction=LT
    }

    %body {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024,1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %concat = f32[2048,1024] concatenate(%gte.1, %gte.2), dimensions={0}
      %gte.3 = f32[1024,1024] get-tuple-element(%param), index=3
      %gte.4 = f32[1024,1024] get-tuple-element(%param), index=4
      %ccall = f32[2048,1024] custom-call(%concat), custom_call_target="test"
      %slice.0 = f32[1024,1024] slice(%ccall), slice={[0:1024], [0:1024]}
      %slice.1 = f32[1024,1024] slice(%ccall), slice={[1024:2048], [0:1024]}
      %add.0 = f32[1024,1024] add(%slice.0, %gte.3)
      %add.1 = f32[1024,1024] add(%slice.1, %gte.4)
      %add.2 = f32[1024,1024] add(%gte.3, %gte.3)
      %add.3 = f32[1024,1024] add(%gte.4, %gte.4)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        tuple(%increment_iteration, %add.0, %add.1, %add.2, %add.3)
    }

    ENTRY test_main {
      %param.0 = f32[1024,1024] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %param.2 = f32[1024,1024] parameter(2)
      %param.3 = f32[1024,1024] parameter(3)
      %constant.0 = s32[] constant(0)
      %while_init = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        tuple(%constant.0, %param.0, %param.1, %param.2, %param.3)
      ROOT %while = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        while(%while_init), condition=%cond, body=%body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConcatCodeMotion(2).Run(module.get()));
  ASSERT_TRUE(changed);
  VLOG(1) << module->ToString();
  auto loop = op::While(
      op::Tuple(op::Constant(),
                AllOf(op::Shape("f32[2048,1024]"),
                      op::Concatenate(op::Parameter(0), op::Parameter(1))),
                AllOf(op::Shape("f32[2048,1024]"),
                      op::Concatenate(op::Parameter(2), op::Parameter(3)))));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(loop), op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop))));
}

TEST_F(WhileLoopConcatCodeMotionTest, TwoConcatsSharedGroups) {
  constexpr absl::string_view kHloModule = R"(
    HloModule test

    %cond {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %constant = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %constant), direction=LT
    }

    %body {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024,1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %concat = f32[2048,1024] concatenate(%gte.1, %gte.2), dimensions={0}
      %ccall = f32[2048,1024] custom-call(%concat), custom_call_target="test"
      %slice.0 = f32[1024,1024] slice(%ccall), slice={[0:1024], [0:1024]}
      %slice.1 = f32[1024,1024] slice(%ccall), slice={[1024:2048], [0:1024]}
      %gte.3 = f32[1024,1024] get-tuple-element(%param), index=3
      %gte.4 = f32[1024,1024] get-tuple-element(%param), index=4
      %concat.1 = f32[2048,1024] concatenate(%gte.3, %gte.4), dimensions={0}
      %ccall.1 = f32[2048,1024] custom-call(%concat.1), custom_call_target="test"
      %slice.2 = f32[1024,1024] slice(%ccall.1), slice={[0:1024], [0:1024]}
      %slice.3 = f32[1024,1024] slice(%ccall.1), slice={[1024:2048], [0:1024]}
      %add.0 = f32[1024,1024] add(%slice.0, %slice.2)
      %add.1 = f32[1024,1024] add(%slice.1, %slice.3)
      %sub.0 = f32[1024,1024] subtract(%slice.0, %slice.2)
      %sub.1 = f32[1024,1024] subtract(%slice.1, %slice.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        tuple(%increment_iteration, %add.0, %add.1, %sub.0, %sub.1)
    }

    ENTRY test_main {
      %param.0 = f32[1024,1024] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %param.2 = f32[1024,1024] parameter(2)
      %param.3 = f32[1024,1024] parameter(3)
      %constant.0 = s32[] constant(0)
      %while_init = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        tuple(%constant.0, %param.0, %param.1, %param.2, %param.3)
      ROOT %while = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        while(%while_init), condition=%cond, body=%body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConcatCodeMotion(2).Run(module.get()));
  ASSERT_TRUE(changed);
  VLOG(1) << module->ToString();
  auto loop = op::While(
      op::Tuple(op::Constant(),
                AllOf(op::Shape("f32[2048,1024]"),
                      op::Concatenate(op::Parameter(0), op::Parameter(1))),
                AllOf(op::Shape("f32[2048,1024]"),
                      op::Concatenate(op::Parameter(2), op::Parameter(3)))));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(loop), op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop))));
}

// Two concats of the same shape and same element shapes. However, the updated
// value (at the end of the loop body) of one of them depends on elements
// concatenated in different orders. So we expect only the other concat to be
// optimized.
TEST_F(WhileLoopConcatCodeMotionTest, TwoConcatsDifferentOrders) {
  constexpr absl::string_view kHloModule = R"(
    HloModule test

    %cond {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %constant = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %constant), direction=LT
    }

    %body {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024,1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %concat = f32[2048,1024] concatenate(%gte.1, %gte.2), dimensions={0}
      %ccall = f32[2048,1024] custom-call(%concat), custom_call_target="test"
      %slice.0 = f32[1024,1024] slice(%ccall), slice={[0:1024], [0:1024]}
      %slice.1 = f32[1024,1024] slice(%ccall), slice={[1024:2048], [0:1024]}
      %gte.3 = f32[1024,1024] get-tuple-element(%param), index=3
      %gte.4 = f32[1024,1024] get-tuple-element(%param), index=4
      %concat.1 = f32[2048,1024] concatenate(%gte.3, %gte.4), dimensions={0}
      %ccall.1 = f32[2048,1024] custom-call(%concat.1), custom_call_target="test"
      %slice.2 = f32[1024,1024] slice(%ccall.1), slice={[0:1024], [0:1024]}
      %slice.3 = f32[1024,1024] slice(%ccall.1), slice={[1024:2048], [0:1024]}
      %add.0 = f32[1024,1024] add(%slice.0, %slice.3)
      %add.1 = f32[1024,1024] add(%slice.1, %slice.2)
      %sub.0 = f32[1024,1024] subtract(%slice.0, %slice.2)
      %sub.1 = f32[1024,1024] subtract(%slice.1, %slice.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        tuple(%increment_iteration, %add.0, %add.1, %sub.0, %sub.1)
    }

    ENTRY test_main {
      %param.0 = f32[1024,1024] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %param.2 = f32[1024,1024] parameter(2)
      %param.3 = f32[1024,1024] parameter(3)
      %constant.0 = s32[] constant(0)
      %while_init = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        tuple(%constant.0, %param.0, %param.1, %param.2, %param.3)
      ROOT %while = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024])
        while(%while_init), condition=%cond, body=%body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConcatCodeMotion(2).Run(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  auto loop = op::While(
      op::Tuple(op::Constant(), op::Parameter(0), op::Parameter(1),
                AllOf(op::Shape("f32[2048,1024]"),
                      op::Concatenate(op::Parameter(2), op::Parameter(3)))));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(loop), op::GetTupleElement(loop),
                op::GetTupleElement(loop), op::Slice(op::GetTupleElement(loop)),
                op::Slice(op::GetTupleElement(loop))));
}

TEST_F(WhileLoopConcatCodeMotionTest, NonElementwiseOps) {
  constexpr absl::string_view kHloModule = R"(
    HloModule test

    %cond {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024], f32[1024], f32[1], f32[1]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %constant = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %constant), direction=LT
    }

    %sum {
      %a = f32[] parameter(0)
      %b = f32[] parameter(1)
      ROOT %add = f32[] add(%a, %b)
    }

    %body {
      %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024], f32[1024], f32[1], f32[1]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024,1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024,1024] get-tuple-element(%param), index=2
      %reshape.0 = f32[1,1024,1024] reshape(%gte.1)
      %reshape.1 = f32[1,1024,1024] reshape(%gte.2)
      %concat = f32[2,1024,1024] concatenate(%reshape.0, %reshape.1), dimensions={0}
      %ccall = f32[2,1024,1024] custom-call(%concat), custom_call_target="test"
      %slice.0 = f32[1,1024,1024] slice(%ccall), slice={[0:1], [0:1024], [0:1024]}
      %slice.1 = f32[1,1024,1024] slice(%ccall), slice={[1:2], [0:1024], [0:1024]}
      %reshape.2 = f32[1024,1024] reshape(%slice.0 )
      %reshape.3 = f32[1024,1024] reshape(%slice.1)
      %gte.3 = f32[1024] get-tuple-element(%param), index=3
      %gte.4 = f32[1024] get-tuple-element(%param), index=4
      %constant.0 = f32[] constant(0)
      %reduce.0 = f32[1024] reduce(%reshape.0, %constant.0), to_apply=%sum, dimensions={0,1}
      %reduce.1 = f32[1024] reduce(%reshape.1, %constant.0), to_apply=%sum, dimensions={0,1}
      %add.0 = f32[1024] add(%reduce.0, %gte.3)
      %add.1 = f32[1024] add(%reduce.1, %gte.4)
      %br0 = f32[1024,1024] broadcast(%add.0), dimensions={1}
      %br1 = f32[1024,1024] broadcast(%add.1), dimensions={1}
      %sub.0 = f32[1024,1024] subtract(%reshape.2, %br0)
      %sub.1 = f32[1024,1024] subtract(%reshape.3, %br1)
      %gte.5 = f32[1] get-tuple-element(%param), index=5
      %gte.6 = f32[1] get-tuple-element(%param), index=6
      %reshape.4 = f32[] reshape(%gte.5)
      %reshape.5 = f32[] reshape(%gte.6)
      %br2 = f32[1024] broadcast(%reshape.4), dimensions={}
      %br3 = f32[1024] broadcast(%reshape.5), dimensions={}
      %add.2 = f32[1024] add(%add.0, %br2)
      %add.3 = f32[1024] add(%add.1, %br3)
      %inc0 = f32[] add(%constant.0, %reshape.4)
      %inc1 = f32[] add(%constant.0, %reshape.5)
      %reshape.6 = f32[1] reshape(%inc0)
      %reshape.7 = f32[1] reshape(%inc1)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024,1024], f32[1024,1024], f32[1024], f32[1024], f32[1], f32[1])
        tuple(%increment_iteration, %sub.0, %sub.1, %add.2, %add.3, %reshape.6, %reshape.7)
    }

    ENTRY test_main {
      %param.0 = f32[1024,1024] parameter(0)
      %param.1 = f32[1024,1024] parameter(1)
      %param.2 = f32[1024] parameter(2)
      %param.3 = f32[1024] parameter(3)
      %param.4 = f32[1] parameter(4)
      %param.5 = f32[1] parameter(5)
      %constant.0 = s32[] constant(0)
      %while_init = (s32[], f32[1024,1024], f32[1024,1024], f32[1024], f32[1024], f32[1], f32[1])
        tuple(%constant.0, %param.0, %param.1, %param.2, %param.3, %param.4, %param.5)
      ROOT %while = (s32[], f32[1024,1024], f32[1024,1024], f32[1024], f32[1024], f32[1], f32[1])
        while(%while_init), condition=%cond, body=%body
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConcatCodeMotion(2).Run(module.get()));
  ASSERT_TRUE(changed);
  VLOG(1) << module->ToString();
  auto loop = op::While(
      op::Tuple(op::Constant(),
                AllOf(op::Shape("f32[2,1024,1024]"),
                      op::Concatenate(op::Reshape(op::Parameter(0)),
                                      op::Reshape(op::Parameter(1)))),
                AllOf(op::Shape("f32[2,1024]"),
                      op::Concatenate(op::Reshape(op::Parameter(2)),
                                      op::Reshape(op::Parameter(3)))),
                AllOf(op::Shape("f32[2]"),
                      op::Concatenate(op::Parameter(4), op::Parameter(5)))));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(loop),
                        op::Reshape(op::Slice(op::GetTupleElement(loop))),
                        op::Reshape(op::Slice(op::GetTupleElement(loop))),
                        op::Reshape(op::Slice(op::GetTupleElement(loop))),
                        op::Reshape(op::Slice(op::GetTupleElement(loop))),
                        op::Slice(op::GetTupleElement(loop)),
                        op::Slice(op::GetTupleElement(loop))));
}

}  // namespace
}  // namespace xla
