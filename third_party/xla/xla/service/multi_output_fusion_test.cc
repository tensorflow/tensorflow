/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/multi_output_fusion.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// Fuses elementwise siblings and counts how often the sibling worklist build
// evaluates its predicates.
class CountingMultiOutputFusion : public MultiOutputFusion {
 public:
  explicit CountingMultiOutputFusion(const AliasInfo* alias_info)
      : MultiOutputFusion(alias_info) {}

  absl::string_view name() const override {
    return "counting_multi_output_fusion";
  }

  // Per instruction IsFusible calls, and per ordered pair LegalToFuse calls,
  // made while the worklist of the last computation was built. Keyed by name
  // because the fusions remove instructions afterwards.
  const absl::flat_hash_map<std::string, int>& is_fusible_calls() const {
    return is_fusible_calls_;
  }
  const absl::flat_hash_map<std::pair<std::string, std::string>, int>&
  legal_to_fuse_calls() const {
    return legal_to_fuse_calls_;
  }

 protected:
  bool ShapesCompatibleForFusion(HloInstruction* instr1,
                                 HloInstruction* instr2) override {
    return ShapeUtil::Equal(instr1->shape(), instr2->shape());
  }

  bool IsFusible(HloInstruction* instr) override {
    if (building_worklist_) {
      ++is_fusible_calls_[instr->name()];
    }
    return instr->IsElementwise();
  }

  int64_t GetProfit(HloInstruction* instr1, HloInstruction* instr2) override {
    return 1;
  }

  bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2) override {
    if (building_worklist_) {
      ++legal_to_fuse_calls_[{std::string(instr1->name()),
                              std::string(instr2->name())}];
    }
    return LegalToFuseMainConstraints(instr1, instr2);
  }

  void CreateFusionWorkListForCurrentComputation() override {
    is_fusible_calls_.clear();
    legal_to_fuse_calls_.clear();
    building_worklist_ = true;
    MultiOutputFusion::CreateFusionWorkListForCurrentComputation();
    building_worklist_ = false;
  }

 private:
  bool building_worklist_ = false;
  absl::flat_hash_map<std::string, int> is_fusible_calls_;
  absl::flat_hash_map<std::pair<std::string, std::string>, int>
      legal_to_fuse_calls_;
};

using MultiOutputFusionTest = HloHardwareIndependentTestBase;

TEST_F(MultiOutputFusionTest, WorklistBuildEvaluatesEachFactOnce) {
  // Every sibling pair shares both profitable operands, so every sibling is
  // visited twice per instruction.
  constexpr absl::string_view kHlo = R"(
HloModule m

ENTRY e {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  a = f32[8] add(p0, p1)
  b = f32[8] multiply(p0, p1)
  c = f32[8] subtract(p0, p1)
  ROOT t = (f32[8], f32[8], f32[8]) tuple(a, b, c)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  AliasInfo alias_info;
  CountingMultiOutputFusion fusion(&alias_info);
  ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(fusion.is_fusible_calls(),
              UnorderedElementsAre(Pair("p0", 1), Pair("p1", 1), Pair("a", 1),
                                   Pair("b", 1), Pair("c", 1), Pair("t", 1)));
  EXPECT_THAT(
      fusion.legal_to_fuse_calls(),
      UnorderedElementsAre(Pair(Pair("a", "b"), 1), Pair(Pair("a", "c"), 1),
                           Pair(Pair("b", "a"), 1), Pair(Pair("b", "c"), 1),
                           Pair(Pair("c", "a"), 1), Pair(Pair("c", "b"), 1)));
}

}  // namespace
}  // namespace xla
