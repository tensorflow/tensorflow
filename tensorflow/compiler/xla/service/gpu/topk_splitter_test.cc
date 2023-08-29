/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/topk_splitter.h"

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/topk_rewriter.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;
using TopkSplitterTest = HloTestBase;

constexpr absl::string_view kComparator = R"(
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

TEST_F(TopkSplitterTest, SplitsTopK) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %arg.1 = f32[1,1073741824] parameter(0)
  ROOT %cc.2 = (f32[1,5], s32[1,5]) custom-call(%arg.1), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(RunHloPass(TopKSplitter(), module.get()), IsOkAndHolds(true));
  auto first_topk = m::CustomCall(m::Reshape(m::Parameter(0)));
  auto slice_result = [&](auto input, size_t i) {
    return m::Reshape(m::Slice(m::GetTupleElement(input, i)));
  };
  auto index_correction =
      m::Broadcast(m::Multiply(m::Iota(), m::Broadcast(m::Constant())));
  auto sorted = m::Sort(
      m::Reshape(m::GetTupleElement(first_topk, 0)),
      m::Reshape(m::Add(m::GetTupleElement(first_topk, 1), index_correction)));
  EXPECT_TRUE(
      Match(module->entry_computation()->root_instruction(),
            m::Tuple(slice_result(sorted, 0), slice_result(sorted, 1))));
}

TEST_F(TopkSplitterTest, SplitsTopKNoBatchDimension) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %arg.1 = f32[1073741824] parameter(0)
  ROOT %cc.2 = (f32[5], s32[5]) custom-call(%arg.1), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(RunHloPass(TopKSplitter(), module.get()), IsOkAndHolds(true));
  auto first_topk = m::CustomCall(m::Reshape(m::Parameter(0)));
  auto slice_result = [&](auto input, size_t i) {
    return m::Reshape(m::Slice(m::GetTupleElement(input, i)));
  };
  auto index_correction =
      m::Broadcast(m::Multiply(m::Iota(), m::Broadcast(m::Constant())));
  auto sorted = m::Sort(
      m::Reshape(m::GetTupleElement(first_topk, 0)),
      m::Reshape(m::Add(m::GetTupleElement(first_topk, 1), index_correction)));
  EXPECT_TRUE(
      Match(module->entry_computation()->root_instruction(),
            m::Tuple(slice_result(sorted, 0), slice_result(sorted, 1))));
}

TEST_F(TopkSplitterTest, SplitFailsUnderThreshold) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %arg.1 = f32[1,524288] parameter(0)
  ROOT %cc.2 = (f32[1,5], s32[1,5]) custom-call(%arg.1), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(
      RunHloPass(TopKSplitter(/*split_threshold=*/1048576), module.get()),
      IsOkAndHolds(false));
}

TEST_F(TopkSplitterTest, SplitFailsUnaligned) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %arg.1 = f32[1,524289] parameter(0)
  ROOT %cc.2 = (f32[1,5], s32[1,5]) custom-call(%arg.1), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(RunHloPass(TopKSplitter(/*split_threshold=*/1024), module.get()),
              IsOkAndHolds(false));
}

TEST_F(TopkSplitterTest, SplitFailsLargeK) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %arg.1 = f32[1,524288] parameter(0)
  ROOT %cc.2 = (f32[1,1024], s32[1,1024]) custom-call(%arg.1), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(RunHloPass(TopKSplitter(/*split_threshold=*/1024), module.get()),
              IsOkAndHolds(false));
}

TEST_F(TopkSplitterTest, Equivalent) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %arg.1 = f32[1,16384] parameter(0)
  ROOT %cc.2 = (f32[1,5], s32[1,5]) custom-call(%arg.1), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(TopkDecomposer().Run(module.get()), IsOkAndHolds(true));
  auto round_trip = [](HloModule* module) {
    EXPECT_THAT(TopkRewriter([](const HloSortInstruction*, int64_t) {
                  return true;
                }).Run(module),
                IsOkAndHolds(true));
    EXPECT_THAT(TopKSplitter(1024).Run(module), IsOkAndHolds(true));
    EXPECT_THAT(TopkDecomposer().Run(module), IsOkAndHolds(true));
    EXPECT_TRUE(HloDCE().Run(module).status().ok());
  };
  EXPECT_TRUE(RunAndCompare(std::move(module), std::nullopt, round_trip));
}

TEST_F(TopkSplitterTest, StableSorts) {
  const std::string hlo_string = absl::Substitute(R"(
HloModule module
$0
ENTRY cluster {
  %constant.1 = f32[] constant(42)
  %broadcast.2= f32[1,16384] broadcast(f32[] %constant.1), dimensions={}
  ROOT %cc.3 = (f32[1,5], s32[1,5]) custom-call(%broadcast.2), custom_call_target= "TopK", to_apply=%compare
})",
                                                  kComparator);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(TopkDecomposer().Run(module.get()), IsOkAndHolds(true));
  auto round_trip = [](HloModule* module) {
    EXPECT_THAT(TopkRewriter([](const HloSortInstruction*, int64_t) {
                  return true;
                }).Run(module),
                IsOkAndHolds(true));
    EXPECT_THAT(TopKSplitter(1024).Run(module), IsOkAndHolds(true));
    EXPECT_THAT(TopkDecomposer().Run(module), IsOkAndHolds(true));
    EXPECT_TRUE(HloDCE().Run(module).status().ok());
  };
  EXPECT_TRUE(RunAndCompare(std::move(module), std::nullopt, round_trip));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
