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

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ::tsl::testing::IsOkAndHolds;
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

std::string getCompareComparator() {
  return R"(
  %compare {
  %Arg_0.100 = f32[] parameter(0)
  %Arg_1.101 = f32[] parameter(1)
  %Arg_2.102 = s32[] parameter(2)
  %Arg_3.103  = s32[] parameter(3)
  ROOT %compare.56364 = pred[] compare(f32[] %Arg_0.100, f32[] %Arg_1.101), direction=GT, type=TOTALORDER
})";
}

std::string getStableComparator() {
  return R"(
  %compare {
    %p.1.lhs.40628 = s32[] parameter(2)
    %p.1.rhs.40629 = s32[] parameter(3)
    %constant.40630 = pred[] constant(true)
    %broadcast.40631 = pred[] broadcast(pred[] %constant.40630), dimensions={}
    %p.0.lhs.40626 = f32[] parameter(0)
    %p.0.rhs.40627 = f32[] parameter(1)
    %compare.40632 = pred[] compare(f32[] %p.0.lhs.40626, f32[] %p.0.rhs.40627), direction=GT, type=TOTALORDER
    ROOT %select.40633 = pred[] select(pred[] %broadcast.40631, pred[] %compare.40632, pred[] %broadcast.40631)
  })";
}

TEST_F(TopkRewriterTest, Rewrite) {
  for (std::string comparator :
       {getComparator(), getCompareComparator(), getStableComparator()}) {
    const std::string hlo_string = R"(
HloModule module
)" + comparator + R"(
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
    TopkRewriter rewriter(
        [](const HloSortInstruction*, int64_t) { return true; });
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
}

TEST_F(TopkRewriterTest, RewriteWithBroadcast) {
  for (std::string comparator :
       {getComparator(), getCompareComparator(), getStableComparator()}) {
    const std::string hlo_string = R"(
HloModule module
)" + comparator + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %iota.4 = s32[1234567]{0} iota(), iota_dimension=0
  %broadcast.5 = s32[8,1234567]{1,0} broadcast(iota.4), dimensions={1}
  %sort.27 = (f32[8,1234567], s32[8,1234567]) sort(%arg_tuple.1, %broadcast.5),
    dimensions={1}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[8,1234567] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[8,5] slice(%get-tuple-element.28), slice={[0:8], [0:5]}
  %get-tuple-element.30 = s32[8,1234567] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[8,5] slice(%get-tuple-element.30), slice={[0:8], [0:5]}
  ROOT %tuple.32 = (f32[8,5], s32[8,5]) tuple(%slice.29, %slice.31)
})";
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    TopkRewriter rewriter(
        [](const HloSortInstruction*, int64_t) { return true; });
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
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
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
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
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
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
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
  TopkRewriter rewriter(
      [](const HloSortInstruction*, int64_t) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0));
  const HloInstruction* cc =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(cc->custom_call_target(), "TopK");
}

TEST_F(TopkRewriterTest, RoundTripNoIota) {
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
  auto run_topk_pass = [&] {
    TopkRewriter rewriter(
        [](const HloSortInstruction*, int64_t) { return true; });
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    TF_ASSERT_OK(HloDCE().Run(module.get()).status());
    ASSERT_TRUE(changed);
    ASSERT_THAT(module->entry_computation()->root_instruction(),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0));
    const HloInstruction* cc =
        module->entry_computation()->root_instruction()->operand(0);
    ASSERT_THAT(cc->custom_call_target(), "TopK");
  };
  // Start by producing a TopK...
  run_topk_pass();
  // ... ensuring it decomposes into sort+slice...
  TF_ASSERT_OK_AND_ASSIGN(bool decomposer_changed,
                          TopkDecomposer().Run(module.get()));
  EXPECT_TRUE(decomposer_changed);
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Slice(op::Sort(op::Parameter(0))));
  // ... and that it can become a topk again.
  run_topk_pass();
}

TEST_F(TopkRewriterTest, RoundTripOnlyIota) {
  const std::string hlo_string = R"(
HloModule module
)" + getComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[8,1234567] parameter(0)
  %iota.4 = s32[1234567]{0} iota(), iota_dimension=0
  %broadcast.5 = s32[8,1234567]{1,0} broadcast(iota.4), dimensions={1}
  %sort.27 = (f32[8,1234567], s32[8,1234567]) sort(%arg_tuple.1, %broadcast.5),
    dimensions={1}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = s32[8,1234567] get-tuple-element(%sort.27), index=1
  ROOT %slice.29 = s32[8,5] slice(%get-tuple-element.28), slice={[0:8], [0:5]}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto run_topk_pass = [&] {
    TopkRewriter rewriter(
        [](const HloSortInstruction*, int64_t) { return true; });
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    TF_ASSERT_OK(HloDCE().Run(module.get()).status());
    ASSERT_TRUE(changed);
    ASSERT_THAT(module->entry_computation()->root_instruction(),
                op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1));
    const HloInstruction* cc =
        module->entry_computation()->root_instruction()->operand(0);
    ASSERT_THAT(cc->custom_call_target(), "TopK");
  };
  // Start by producing a TopK...
  run_topk_pass();
  // ... ensuring it decomposes into sort+slice...
  TF_ASSERT_OK_AND_ASSIGN(bool decomposer_changed,
                          TopkDecomposer().Run(module.get()));
  EXPECT_TRUE(decomposer_changed);
  TF_ASSERT_OK(TupleSimplifier().Run(module.get()).status());
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Slice(op::GetTupleElement(
                  op::Sort(op::Parameter(0), op::Iota()), 1)));
  // ... and that it can become a topk again.
  run_topk_pass();
}

TEST_F(TopkRewriterTest, RoundTrip) {
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
  auto run_topk_pass = [&] {
    TopkRewriter rewriter(
        [](const HloSortInstruction*, int64_t) { return true; });
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    TF_ASSERT_OK(HloDCE().Run(module.get()).status());
    ASSERT_TRUE(changed);
    ASSERT_THAT(
        module->entry_computation()->root_instruction(),
        op::Tuple(op::GetTupleElement(op::CustomCall(op::Parameter(0)), 0),
                  op::GetTupleElement(op::CustomCall(op::Parameter(0)), 1)));
    const HloInstruction* cc =
        module->entry_computation()->root_instruction()->operand(0)->operand(0);
    ASSERT_THAT(cc->custom_call_target(), "TopK");
  };
  // Start by producing a TopK...
  run_topk_pass();
  // ... ensuring it decomposes into sort+slice...
  TF_ASSERT_OK_AND_ASSIGN(bool decomposer_changed,
                          TopkDecomposer().Run(module.get()));
  EXPECT_TRUE(decomposer_changed);
  TF_ASSERT_OK(HloDCE().Run(module.get()).status());
  TF_ASSERT_OK(TupleSimplifier().Run(module.get()).status());
  auto sort_matcher = op::Sort(op::Parameter(0), op::Iota());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Slice(op::GetTupleElement(sort_matcher, 0)),
                        op::Slice(op::GetTupleElement(sort_matcher, 1))));
  // ... and that it can become a topk again.
  run_topk_pass();
}

TEST_F(TopkRewriterTest, SanityCheckOutput) {
  const std::string hlo_string = R"(
HloModule module
)" + getCompareComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[1234] parameter(0)
  %iota.4 = s32[1234] iota(), iota_dimension=0
  %sort.27 = (f32[1234], s32[1234]) sort(%arg_tuple.1, %iota.4),
    dimensions={0}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[1234] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[5] slice(%get-tuple-element.28), slice={[0:5]}
  %get-tuple-element.30 = s32[1234] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[5] slice(%get-tuple-element.30), slice={[0:5]}
  ROOT %tuple.32 = (f32[5], s32[5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto source_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto topk_module = source_module->Clone();
  EXPECT_THAT(TopkRewriter([](const HloSortInstruction*, int64_t) {
                return true;
              }).Run(topk_module.get()),
              IsOkAndHolds(true));
  auto decomposed_module = topk_module->Clone();
  EXPECT_THAT(TopkDecomposer().Run(decomposed_module.get()),
              IsOkAndHolds(true));
  const size_t source_size = 1234;
  std::vector<float> source(source_size);
  std::iota(source.begin(), source.end(), 80000);
  auto input = LiteralUtil::CreateR1<float>(source);
  std::vector<float> top_k({81233, 81232, 81231, 81230, 81229});
  // Ensure all 3 modules produce the same output on the same input.
  auto check_result = [&](std::unique_ptr<HloModule> module) {
    TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&input}));
    LiteralTestUtil::ExpectR1Equal<float>(top_k, result.DecomposeTuple()[0]);
  };
  check_result(std::move(source_module));
  check_result(std::move(decomposed_module));
}

TEST_F(TopkRewriterTest, Equivalent) {
  const std::string hlo_string = R"(
HloModule module
)" + getCompareComparator() + R"(
ENTRY cluster {
  %arg_tuple.1 = f32[1234] parameter(0)
  %iota.4 = s32[1234] iota(), iota_dimension=0
  %sort.27 = (f32[1234], s32[1234]) sort(%arg_tuple.1, %iota.4),
    dimensions={0}, is_stable=true, to_apply=%compare
  %get-tuple-element.28 = f32[1234] get-tuple-element(%sort.27), index=0
  %slice.29 = f32[5] slice(%get-tuple-element.28), slice={[0:5]}
  %get-tuple-element.30 = s32[1234] get-tuple-element(%sort.27), index=1
  %slice.31 = s32[5] slice(%get-tuple-element.30), slice={[0:5]}
  ROOT %tuple.32 = (f32[5], s32[5]) tuple(%slice.29, %slice.31)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto source_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto round_trip = [](HloModule* module) {
    EXPECT_THAT(TopkRewriter([](const HloSortInstruction*, int64_t) {
                  return true;
                }).Run(module),
                IsOkAndHolds(true));
    EXPECT_THAT(TopkDecomposer().Run(module), IsOkAndHolds(true));
  };
  EXPECT_TRUE(
      RunAndCompare(std::move(source_module), std::nullopt, round_trip));
}

}  // namespace
}  // namespace xla
