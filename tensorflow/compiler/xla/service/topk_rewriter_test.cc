/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/topk_rewriter.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using TopkRewriterTest = HloTestBase;

std::string getComparator() {
  return R"(
%compare {
  %p.1.lhs.8 = s32[] parameter(2)
  %p.1.rhs.9 = s32[] parameter(3)
  %p.0.lhs.6 = f32[] parameter(0)
  %bitcast-convert.11 = s32[] bitcast-convert(%p.0.lhs.6)
  %constant.15 = s32[] constant(0)
  %compare.16 = pred[] compare(%bitcast-convert.11, %constant.15), direction=LT
  %constant.10 = u32[] constant(2147483647)
  %bitcast-convert.12 = u32[] bitcast-convert(%p.0.lhs.6)
  %subtract.13 = u32[] subtract(%constant.10, %bitcast-convert.12)
  %bitcast-convert.14 = s32[] bitcast-convert(%subtract.13)
  %select.17 = s32[] select(%compare.16, %bitcast-convert.14,
                            %bitcast-convert.11)
  %p.0.rhs.7 = f32[] parameter(1)
  %bitcast-convert.19 = s32[] bitcast-convert(%p.0.rhs.7)
  %constant.23 = s32[] constant(0)
  %compare.24 = pred[] compare(%bitcast-convert.19, %constant.23), direction=LT
  %constant.18 = u32[] constant(2147483647)
  %bitcast-convert.20 = u32[] bitcast-convert(%p.0.rhs.7)
  %subtract.21 = u32[] subtract(%constant.18, %bitcast-convert.20)
  %bitcast-convert.22 = s32[] bitcast-convert(%subtract.21)
  %select.25 = s32[] select(%compare.24, %bitcast-convert.22,
                            %bitcast-convert.19)
  ROOT %compare.26 = pred[] compare(%select.17, %select.25), direction=GT
})";
}

std::string getConvertMaxComparator() {
  return R"(
%compare {
  %p.1.lhs.6 = s32[] parameter(2)
  %p.1.rhs.7 = s32[] parameter(3)
  %p.0.lhs.4 = f32[] parameter(0)
  %bitcast-convert = s32[] bitcast-convert(f32[] %p.0.lhs.4)
  %constant = s32[] constant(0)
  %compare = pred[] compare(s32[] %bitcast-convert, s32[] %constant), direction=LT
  %constant.1 = s32[] constant(2147483647)
  %convert = u32[] convert(s32[] %constant.1)
  %bitcast-convert.1 = u32[] bitcast-convert(f32[] %p.0.lhs.4)
  %subtract = u32[] subtract(u32[] %convert, u32[] %bitcast-convert.1)
  %bitcast-convert.2 = s32[] bitcast-convert(u32[] %subtract)
  %select = s32[] select(pred[] %compare, s32[] %bitcast-convert.2, s32[] %bitcast-convert)
  %p.0.rhs.5 = f32[] parameter(1)
  %bitcast-convert.3 = s32[] bitcast-convert(f32[] %p.0.rhs.5)
  %compare.1 = pred[] compare(s32[] %bitcast-convert.3, s32[] %constant), direction=LT
  %bitcast-convert.4 = u32[] bitcast-convert(f32[] %p.0.rhs.5)
  %subtract.1 = u32[] subtract(u32[] %convert, u32[] %bitcast-convert.4)
  %bitcast-convert.5 = s32[] bitcast-convert(u32[] %subtract.1)
  %select.1 = s32[] select(pred[] %compare.1, s32[] %bitcast-convert.5, s32[] %bitcast-convert.3)
  ROOT %compare.2 = pred[] compare(s32[] %select, s32[] %select.1), direction=GT
})";
}

std::string getComparatorNoIota() {
  return R"(
%compare {
  %p.0.lhs.6 = f32[] parameter(0)
  %bitcast-convert.11 = s32[] bitcast-convert(%p.0.lhs.6)
  %constant.15 = s32[] constant(0)
  %compare.16 = pred[] compare(%bitcast-convert.11, %constant.15), direction=LT
  %constant.10 = u32[] constant(2147483647)
  %bitcast-convert.12 = u32[] bitcast-convert(%p.0.lhs.6)
  %subtract.13 = u32[] subtract(%constant.10, %bitcast-convert.12)
  %bitcast-convert.14 = s32[] bitcast-convert(%subtract.13)
  %select.17 = s32[] select(%compare.16, %bitcast-convert.14,
                            %bitcast-convert.11)
  %p.0.rhs.7 = f32[] parameter(1)
  %bitcast-convert.19 = s32[] bitcast-convert(%p.0.rhs.7)
  %constant.23 = s32[] constant(0)
  %compare.24 = pred[] compare(%bitcast-convert.19, %constant.23), direction=LT
  %constant.18 = u32[] constant(2147483647)
  %bitcast-convert.20 = u32[] bitcast-convert(%p.0.rhs.7)
  %subtract.21 = u32[] subtract(%constant.18, %bitcast-convert.20)
  %bitcast-convert.22 = s32[] bitcast-convert(%subtract.21)
  %select.25 = s32[] select(%compare.24, %bitcast-convert.22,
                            %bitcast-convert.19)
  ROOT %compare.26 = pred[] compare(%select.17, %select.25), direction=GT
})";
}

TEST_F(TopkRewriterTest, Rewrite) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %iota.4 = s32[8,1234567] iota(), iota_dimension=1
  %sort.27 = (f32[8,1234567], s32[8,1234567]) sort(%arg_tuple.1, %iota.4),
    dimensions={1}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[8,1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[8,5] slice(%get-tuple-element.28), slice={[0:8], [0:5]}
  %get-tuple-element.30 = s32[8,1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[8,5] slice(%get-tuple-element.30), slice={[0:8], [0:5]}
  ROOT %tuple.32 = (f32[8,5], s32[8,5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter([](const HloSortInstruction*, int64) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteWithConvertMaxComparator) {
  const std::string hlo_string = R"(
HloModule module
)" + getConvertMaxComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %iota.4 = s32[8,1234567] iota(), iota_dimension=1
  %sort.27 = (f32[8,1234567], s32[8,1234567]) sort(%arg_tuple.1, %iota.4),
    dimensions={1}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[8,1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[8,5] slice(%get-tuple-element.28), slice={[0:8], [0:5]}
  %get-tuple-element.30 = s32[8,1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[8,5] slice(%get-tuple-element.30), slice={[0:8], [0:5]}
  ROOT %tuple.32 = (f32[8,5], s32[8,5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter([](const HloSortInstruction*, int64) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteUnbatched) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[1234567] parameter(0)
  %iota.4 = s32[1234567] iota(), iota_dimension=0
  %sort.27 = (f32[1234567], s32[1234567]) sort(%arg_tuple.1, %iota.4),
    dimensions={0}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[5] slice(%get-tuple-element.28), slice={[0:5]}
  %get-tuple-element.30 = s32[1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[5] slice(%get-tuple-element.30), slice={[0:5]}
  ROOT %tuple.32 = (f32[5], s32[5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter([](const HloSortInstruction*, int64) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteTranspose) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[1234567,8] parameter(0)
  %iota.4 = s32[1234567,8] iota(), iota_dimension=0
  %sort.27 = (f32[1234567,8], s32[1234567,8]) sort(%arg_tuple.1, %iota.4),
    dimensions={0}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[1234567,8] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[5,8] slice(%get-tuple-element.28), slice={[0:5], [0:8]}
  %get-tuple-element.30 = s32[1234567,8] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[5,8] slice(%get-tuple-element.30), slice={[0:5], [0:8]}
  ROOT %tuple.32 = (f32[5,8], s32[5,8]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter([](const HloSortInstruction*, int64) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  LOG(INFO) << module->entry_computation()->ToString();
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Transpose(op::GetTupleElement(
                    op::CustomCall(op::Transpose(op::Parameter(0))), 0)),
                op::Transpose(op::GetTupleElement(
                    op::CustomCall(op::Transpose(op::Parameter(0))), 1))));
  const HloInstruction* cc = module->entry_computation()
                                 ->root_instruction()
                                 ->operand(0)
                                 ->operand(0)
                                 ->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RewriteNoIota) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparatorNoIota() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %sort.27 = f32[8,1234567] sort(%arg_tuple.1), dimensions={1}, is_stable=true, to_apply=%compare
  ROOT %slice.29 = f32[8,5] slice(%sort.27), slice={[0:8], [0:5]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TopkRewriter rewriter([](const HloSortInstruction*, int64) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

}  // namespace
}  // namespace xla
