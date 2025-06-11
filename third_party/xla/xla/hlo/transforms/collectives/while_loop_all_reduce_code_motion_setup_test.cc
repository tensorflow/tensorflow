/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/while_loop_all_reduce_code_motion_setup.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class ReorderReduceTransposeTest : public HloHardwareIndependentTestBase {
 protected:
  ReorderReduceTransposeTest() = default;
};

TEST_F(ReorderReduceTransposeTest, SimpleReduceScatterTransposeInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%while_cond {
  %param = (s32[4,4], s32[4,2], s32[4,2]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (s32[4,4], s32[4,2], s32[4,2]) parameter(0)
  %gte.0 = s32[4,4] get-tuple-element(%param), index=0
  %gte.1 = s32[4,2] get-tuple-element(%param), index=1
  %reduce_scatter.0 = s32[2,4] reduce-scatter(%gte.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %transpose.0 = s32[4,2] transpose(%reduce_scatter.0), dimensions={1,0}
  %add.0 = s32[4,2] add(%transpose.0, %gte.1)
  ROOT tuple = (s32[4,4], s32[4,2], s32[4,2]) tuple(%gte.0, %add.0, %gte.1)
}

ENTRY main {
  %init_param = (s32[4,4], s32[4,2], s32[4,2]) parameter(0)
  ROOT while = (s32[4,4], s32[4,2], s32[4,2]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderReduceTranspose rrt;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rrt.Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the transpose and reduce-scatter have been reordered inside the
  // while body.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  HloComputation* while_body = while_inst->while_body();
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(),
                        op::Add(op::ReduceScatter(op::Transpose()),
                                op::GetTupleElement()),
                        op::GetTupleElement()));
}

TEST_F(ReorderReduceTransposeTest,
       ReduceScatterConvertTransposeNotInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY main {
  arg.0 = f32[4,4] parameter(0)
  reduce_scatter.0 = f32[2,4] reduce-scatter(arg.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  convert.0 = s32[2,4] convert(reduce_scatter.0)
  ROOT transpose.0 = s32[4,2] transpose(convert.0), dimensions={1,0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderReduceTranspose rrt;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rrt.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReorderReduceTransposeTest, ReduceScatterConvertTransposeInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%while_cond {
  %param = (f32[4,4], s32[4,2], s32[4,2]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (f32[4,4], s32[4,2], s32[4,2]) parameter(0)
  %gte.0 = f32[4,4] get-tuple-element(%param), index=0
  %gte.1 = s32[4,2] get-tuple-element(%param), index=1
  %reduce_scatter.0 = f32[2,4] reduce-scatter(%gte.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %convert.0 = s32[2,4] convert(%reduce_scatter.0)
  %transpose.0 = s32[4,2] transpose(%convert.0), dimensions={1,0}
  %add.0 = s32[4,2] add(%transpose.0, %gte.1)
  ROOT tuple = (f32[4,4], s32[4,2], s32[4,2]) tuple(%gte.0, %add.0, %gte.1)
}

ENTRY main {
  %init_param = (f32[4,4], s32[4,2], s32[4,2]) parameter(0)
  ROOT while = (f32[4,4], s32[4,2], s32[4,2]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderReduceTranspose rrt;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rrt.Run(module.get()));
  EXPECT_TRUE(changed);
  // Check that the transpose, convert, and reduce-scatter have been reordered
  // inside the while body.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  HloComputation* while_body = while_inst->while_body();

  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(),
                        op::Add(op::ReduceScatter(op::Transpose(op::Convert())),
                                op::GetTupleElement()),
                        op::GetTupleElement()));
}

TEST_F(ReorderReduceTransposeTest,
       ReduceScatterTransposeReshapeDynamicUpdateSliceInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = s32[] parameter(0)
  %y = s32[] parameter(1)
  ROOT %add = s32[] add(s32[] %x, s32[] %y)
}

%while_cond {
  %param = (s32[4,4], s32[8], s32[]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (s32[4,4], s32[8], s32[]) parameter(0)
  %gte.0 = s32[4,4] get-tuple-element(%param), index=0
  %gte.1 = s32[8] get-tuple-element(%param), index=1
  %gte.2 = s32[] get-tuple-element(%param), index=2
  %reduce_scatter.0 = s32[2,4] reduce-scatter(%gte.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %transpose.0 = s32[4,2] transpose(%reduce_scatter.0), dimensions={1,0}
  %reshape.0 = s32[8] reshape(%transpose.0)
  %dynamic-update-slice.0 = s32[8] dynamic-update-slice(%gte.1, %reshape.0, %gte.2)
  ROOT tuple = (s32[4,4], s32[8], s32[]) tuple(%gte.0, %dynamic-update-slice.0, %gte.2)
}

ENTRY main {
  %init_param = (s32[4,4], s32[8], s32[]) parameter(0)
  ROOT while = (s32[4,4], s32[8], s32[]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderReduceTranspose rrt;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rrt.Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the transpose and reduce-scatter have been reordered inside the
  // while body.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  HloComputation* while_body = while_inst->while_body();
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(),
                        op::DynamicUpdateSlice(
                            op::GetTupleElement(),
                            op::Reshape(op::ReduceScatter(op::Transpose())),
                            op::GetTupleElement()),
                        op::GetTupleElement()));
}

class ReorderConvertReduceAddTest : public HloHardwareIndependentTestBase {
 protected:
  ReorderConvertReduceAddTest() = default;
};

TEST_F(ReorderConvertReduceAddTest, SimpleConvertReduceScatterAddInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%while_cond {
  %param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  %gte.0 = f32[4,4] get-tuple-element(%param), index=0
  %gte.1 = s32[2,4] get-tuple-element(%param), index=1
  %gte.2 = s32[] get-tuple-element(%param), index=2
  %reduce_scatter.0 = f32[2,4] reduce-scatter(%gte.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %convert.0 = s32[2,4] convert(%reduce_scatter.0)
  %add.0 = s32[2,4] add(%convert.0, %gte.1)
  ROOT tuple = (f32[4,4], s32[2,4], s32[]) tuple(%gte.0, %add.0, %gte.2)
}

ENTRY main {
  %init_param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  ROOT while = (f32[4,4], s32[2,4], s32[]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderConvertReduceAdd rcra(/*enable_reduce_scatter=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rcra.Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the convert, reduce-scatter, and add have been reordered inside
  // the while body.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  HloComputation* while_body = while_inst->while_body();
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(),
                        op::Add(op::ReduceScatter(op::Convert()),
                                op::GetTupleElement()),
                        op::GetTupleElement()));
  HloComputation* reduction =
      while_body->root_instruction()->operand(1)->operand(0)->to_apply();
  EXPECT_EQ(reduction->root_instruction()->shape().element_type(), S32);
}

TEST_F(ReorderConvertReduceAddTest, ConvertAllReduceAddNotInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY main {
  arg.0 = f32[4,4] parameter(0)
  all_reduce.0 = f32[4,4] all-reduce(arg.0), replica_groups={{0,1}}, to_apply=%reduction
  ROOT convert.0 = s32[4,4] convert(all_reduce.0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderConvertReduceAdd rcra;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rcra.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReorderConvertReduceAddTest, ConvertReduceScatterAddInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%while_cond {
  %param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  %gte.0 = f32[4,4] get-tuple-element(%param), index=0
  %gte.1 = s32[2,4] get-tuple-element(%param), index=1
  %gte.2 = s32[] get-tuple-element(%param), index=2
  %reduce_scatter.0 = f32[2,4] reduce-scatter(%gte.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %convert.0 = s32[2,4] convert(%reduce_scatter.0)
  %add.0 = s32[2,4] add(%convert.0, %gte.1)
  ROOT tuple = (f32[4,4], s32[2,4], s32[]) tuple(%gte.0, %add.0, %gte.2)
}

ENTRY main {
  %init_param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  ROOT while = (f32[4,4], s32[2,4], s32[]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderConvertReduceAdd rcra(/*enable_reduce_scatter=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rcra.Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the convert, reduce-scatter, and add have been reordered inside
  // the while body.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  HloComputation* while_body = while_inst->while_body();
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(),
                        op::Add(op::ReduceScatter(op::Convert()),
                                op::GetTupleElement()),
                        op::GetTupleElement()));
}

TEST_F(ReorderConvertReduceAddTest, DisableReduceScatter) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%while_cond {
  %param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  %gte.0 = f32[4,4] get-tuple-element(%param), index=0
  %gte.1 = s32[2,4] get-tuple-element(%param), index=1
  %gte.2 = s32[] get-tuple-element(%param), index=2
  %reduce_scatter.0 = f32[2,4] reduce-scatter(%gte.0), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %convert.0 = s32[2,4] convert(%reduce_scatter.0)
  %add.0 = s32[2,4] add(%convert.0, %gte.1)
  ROOT tuple = (f32[4,4], s32[2,4], s32[]) tuple(%gte.0, %add.0, %gte.2)
}

ENTRY main {
  %init_param = (f32[4,4], s32[2,4], s32[]) parameter(0)
  ROOT while = (f32[4,4], s32[2,4], s32[]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderConvertReduceAdd rcra(/*enable_reduce_scatter=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rcra.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReorderConvertReduceAddTest, ConvertAllReduceAddInWhileBody) {
  constexpr absl::string_view hlo = R"(
HloModule main

%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%while_cond {
  %param = (f32[2,4], s32[2,4], s32[]) parameter(0)
  ROOT cond = pred[] constant(true)
}

%while_body {
  %param = (f32[2,4], s32[2,4], s32[]) parameter(0)
  %gte.0 = f32[2,4] get-tuple-element(%param), index=0
  %gte.1 = s32[2,4] get-tuple-element(%param), index=1
  %gte.2 = s32[] get-tuple-element(%param), index=2
  %all_reduce.0 = f32[2,4] all-reduce(%gte.0), replica_groups={{0,1}}, to_apply=%reduction
  %convert.0 = s32[2,4] convert(%all_reduce.0)
  %add.0 = s32[2,4] add(%convert.0, %gte.1)
  ROOT tuple = (f32[2,4], s32[2,4], s32[]) tuple(%gte.0, %add.0, %gte.2)
}

ENTRY main {
  %init_param = (f32[2,4], s32[2,4], s32[]) parameter(0)
  ROOT while = (f32[2,4], s32[2,4], s32[]) while(%init_param), condition=%while_cond, body=%while_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReorderConvertReduceAdd rcra;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rcra.Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the convert, all-reduce, and add have been reordered inside the
  // while body.
  HloInstruction* while_inst = module->entry_computation()->root_instruction();
  HloComputation* while_body = while_inst->while_body();
  EXPECT_THAT(
      while_body->root_instruction(),
      op::Tuple(op::GetTupleElement(),
                op::Add(op::AllReduce(op::Convert()), op::GetTupleElement()),
                op::GetTupleElement()));
  HloComputation* reduction =
      while_body->root_instruction()->operand(1)->operand(0)->to_apply();
  EXPECT_EQ(reduction->root_instruction()->shape().element_type(), S32);
}

}  // namespace
}  // namespace xla
