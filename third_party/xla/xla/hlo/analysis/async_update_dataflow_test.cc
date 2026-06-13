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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class AsyncUpdateDataflowTest : public HloHardwareIndependentTestBase {
 protected:
  AsyncUpdateDataflowTest() : module_(CreateNewVerifiedModule()) {}

  const HloDataflowAnalysis& RunAnalysis(bool ssa_form = true) {
    auto status_or = HloDataflowAnalysis::Run(*module_, ssa_form);
    EXPECT_TRUE(status_or.ok()) << status_or.status();
    analysis_ = std::move(status_or).value();
    return *analysis_;
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloDataflowAnalysis> analysis_;
};

// Verifies that dataflow propagates correctly through an incremental binding
// chain where parameters are bound across multiple async-update instructions.
TEST_F(AsyncUpdateDataflowTest, LateBoundOperandDataflow) {
  // HLO program with two parameters bound incrementally using async-update.
  std::string hlo_text = R"hlo(
HloModule LateBoundOperandDataflow, is_scheduled=true

%Subcomputation (param0: f32[], param1: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  ROOT %add = f32[] add(%param0, %param1)
}

ENTRY %entry () -> f32[] {
  %constant.1 = f32[] constant(1)
  %constant.2 = f32[] constant(2)
  // Initially, no parameters are bound.
  %async-start = ((), f32[], s32[]{:S(2)}) async-start(),
      calls=%Subcomputation, async_execution_thread="sparsecore"
  // Bind the first parameter (param0).
  %async-update.0 = ((f32[]), f32[], s32[]{:S(2)}) async-update(%async-start, %constant.1)
  // Bind the second parameter (param1) incrementally.
  %async-update.1 = ((f32[], f32[]), f32[], s32[]{:S(2)}) async-update(%async-update.0, %constant.2)
  ROOT %async-done = f32[] async-done(%async-update.1)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  const HloDataflowAnalysis& analysis = RunAnalysis();

  const HloComputation* subcomputation =
      FindComputation(module_.get(), "Subcomputation");
  const HloInstruction* subparam0 = subcomputation->parameter_instruction(0);
  const HloInstruction* subparam1 = subcomputation->parameter_instruction(1);

  const HloInstruction* constant1 =
      FindInstruction(module_.get(), "constant.1");
  const HloInstruction* constant2 =
      FindInstruction(module_.get(), "constant.2");
  const HloInstruction* async_start =
      FindInstruction(module_.get(), "async-start");
  const HloInstruction* async_update_0 =
      FindInstruction(module_.get(), "async-update.0");
  const HloInstruction* async_update_1 =
      FindInstruction(module_.get(), "async-update.1");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  const HloInstruction* sub_add = FindInstruction(module_.get(), "add");

  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1, {}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2, {}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, {}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_update_0, {}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_update_1, {}));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_done, {}));

  EXPECT_TRUE(analysis.ValueIsDefinedAt(sub_add, {}));

  // Verify that constant1 correctly reaches param0 via the first async-update.
  EXPECT_EQ(analysis.GetUniqueValueAt(subparam0),
            analysis.GetValueDefinedAt(constant1));

  // Verify that constant2 correctly reaches param1 via the second async-update.
  EXPECT_EQ(analysis.GetUniqueValueAt(subparam1),
            analysis.GetValueDefinedAt(constant2));

  EXPECT_THAT(
      analysis.GetValueSet(async_done).values(),
      testing::UnorderedElementsAre(&analysis.GetValueDefinedAt(sub_add)));
}

TEST_F(AsyncUpdateDataflowTest, LateBoundOutputDataflow) {
  std::string hlo_text = R"hlo(
HloModule Module, is_scheduled=true

async_computation {
  p0 = f32[10] parameter(0)
  ROOT custom-call = f32[10] custom-call(p0), custom_call_target="foo"
}

ENTRY main {
  p = f32[10] parameter(0)
  async-start = ((f32[10]), (), s32[]) async-start(p), calls=async_computation
  async-update = ((f32[10]), f32[10], s32[]) async-update(async-start)
  ROOT async-done = f32[10] async-done(async-update)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  const HloDataflowAnalysis& analysis = RunAnalysis();

  const HloInstruction* custom_call =
      FindInstruction(module_.get(), "custom-call");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  const HloValueSet& async_done_values = analysis.GetValueSet(async_done);
  const HloValue& cc_value = analysis.GetValueDefinedAt(custom_call);

  EXPECT_TRUE(absl::c_linear_search(async_done_values.values(), &cc_value));
}

/*
TEST_F(AsyncUpdateDataflowTest, AliasInfoInPlacePairs) {
  std::string hlo_text = R"hlo(
HloModule Module, is_scheduled=true

Subcomputation (param0: f32[]) -> f32[] {
  ROOT p0 = f32[] parameter(0)
}

ENTRY main {
  constant = f32[] constant(1)
  async-start = ((), f32[], s32[]) async-start(), calls=Subcomputation
  async-update = ((f32[]), f32[], s32[]) async-update(async-start, constant)
  ROOT async-done = f32[] async-done(async-update)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  const HloInstruction* async_update =
      FindInstruction(module_.get(), "async-update");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  AliasInfo alias_info;

  auto update_pairs = alias_info.GetInPlaceInputOutputPairs(async_update);
  ASSERT_EQ(update_pairs.size(), 2);

  // Parameter binding pair
  EXPECT_EQ(update_pairs[0].first.operand_number, 1);
  EXPECT_EQ(update_pairs[0].first.operand_index, ShapeIndex({}));
  EXPECT_EQ(update_pairs[0].second, ShapeIndex({0, 0}));

  // Output aliasing pair
  EXPECT_EQ(update_pairs[1].first.operand_number, 0);
  EXPECT_EQ(update_pairs[1].first.operand_index, ShapeIndex({1}));
  EXPECT_EQ(update_pairs[1].second, ShapeIndex({1}));

  // Test AsyncDone in-place pairs
  auto done_pairs = alias_info.GetInPlaceInputOutputPairs(async_done);
  ASSERT_EQ(done_pairs.size(), 1);
  EXPECT_EQ(done_pairs[0].first.operand_number, 0);
  EXPECT_EQ(done_pairs[0].first.operand_index, ShapeIndex({1}));
  EXPECT_EQ(done_pairs[0].second, ShapeIndex({}));
}
*/

TEST_F(AsyncUpdateDataflowTest, AliasInfoPropagation) {
  std::string hlo_text = R"hlo(
HloModule Module, is_scheduled=true

Subcomputation (param0: f32[]) -> f32[] {
  ROOT p0 = f32[] parameter(0)
}

ENTRY main {
  constant = f32[] constant(1)
  async-start = ((f32[]), (), s32[]) async-start(constant), calls=Subcomputation
  async-update = ((f32[]), f32[], s32[]) async-update(async-start)
  ROOT async-done = f32[] async-done(async-update)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  HloInstruction* async_start = FindInstruction(module_.get(), "async-start");
  HloInstruction* constant = FindInstruction(module_.get(), "constant");

  // Programmatically set output-to-operand aliasing on AsyncStart.
  auto* async_start_concrete = Cast<HloAsyncStartInstruction>(async_start);

  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>> aliasing;
  aliasing.push_back({ShapeIndex({0, 0}), {0, ShapeIndex({})}});
  async_start_concrete->set_output_to_operand_aliasing(aliasing);

  // Run HloAliasAnalysis
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module_.get(), &alias_info));

  // Verify that constant and async-start at {0, 0} are in the same buffer!
  const HloBuffer& constant_buffer =
      alias_analysis->GetUniqueBufferAt(constant);
  const HloBuffer& async_start_buffer =
      alias_analysis->GetUniqueBufferAt(async_start, {0, 0});
  EXPECT_EQ(constant_buffer.id(), async_start_buffer.id());
}

TEST_F(AsyncUpdateDataflowTest, AsyncStartValueSet) {
  std::string hlo_text = R"hlo(
HloModule Module, is_scheduled=true

Subcomputation (param0: f32[]) -> f32[] {
  ROOT p0 = f32[] parameter(0)
}

ENTRY main {
  constant = f32[] constant(1)
  async-start = ((f32[]), (), s32[]) async-start(constant), calls=Subcomputation
  async-update = ((f32[]), f32[], s32[]) async-update(async-start)
  ROOT async-done = f32[] async-done(async-update)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  // Programmatically set output-to-operand aliasing on AsyncStart.
  HloInstruction* async_start_non_const =
      FindInstruction(module_.get(), "async-start");
  auto* async_start_concrete =
      Cast<HloAsyncStartInstruction>(async_start_non_const);

  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>> aliasing;
  aliasing.push_back({ShapeIndex({0, 0}), {0, ShapeIndex({})}});
  async_start_concrete->set_output_to_operand_aliasing(aliasing);

  const HloDataflowAnalysis& analysis = RunAnalysis();

  const HloInstruction* async_start =
      FindInstruction(module_.get(), "async-start");

  const HloValueSet& value_set = analysis.GetValueSet(async_start, {0, 0});

  EXPECT_EQ(value_set.values().size(), 1);
}

TEST_F(AsyncUpdateDataflowTest, LateBoundOutputBufferDataflow) {
  std::string hlo_text = R"hlo(
HloModule Module, is_scheduled=true

async_computation {
  p0 = f32[10] parameter(0)
  ROOT custom-call = f32[10] custom-call(p0), custom_call_target="foo"
}

ENTRY main {
  p = f32[10] parameter(0)
  async-start = ((f32[10]), (), s32[]) async-start(p), calls=async_computation
  async-update = ((f32[10]), f32[10], s32[]) async-update(async-start)
  ROOT async-done = f32[10] async-done(async-update)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  const HloDataflowAnalysis& analysis = RunAnalysis();

  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  const HloValueSet& async_done_values = analysis.GetValueSet(async_done);

  EXPECT_FALSE(async_done_values.values().empty());
}

TEST_F(AsyncUpdateDataflowTest, LateBoundNestedTupleOutputDataflow) {
  std::string hlo_text = R"hlo(
HloModule Module, is_scheduled=true

async_computation {
  p0 = f32[10] parameter(0)
  ROOT tuple = (f32[10], f32[10]) tuple(p0, p0)
}

ENTRY main {
  p = f32[10] parameter(0)
  async-start = ((f32[10]), (), s32[]) async-start(p), calls=async_computation
  // The output shape starts empty {}, and gets bound later.
  async-update = ((f32[10]), (f32[10], f32[10]), s32[]) async-update(async-start)
  ROOT async-done = (f32[10], f32[10]) async-done(async-update)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                       hlo_text, GetModuleConfigForTest()));

  const HloDataflowAnalysis& analysis = RunAnalysis();

  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  const HloValueSet& async_done_values_0 =
      analysis.GetValueSet(async_done, {0});

  EXPECT_FALSE(async_done_values_0.values().empty());
}

}  // namespace
}  // namespace xla
