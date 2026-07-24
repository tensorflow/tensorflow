/* Copyright 2019 The OpenXLA Authors.

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
#include "xla/hlo/utils/hlo_live_range.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using LiveRangeBounds = HloLiveRange::LiveRangeBounds;
class HloLiveRangeTest : public HloHardwareIndependentTestBase {
 protected:
  HloLiveRangeTest() : module_(CreateNewVerifiedModule()) {}
  ~HloLiveRangeTest() override {}

  void Analyze(const HloSchedule& schedule) {
    alias_analysis_ =
        HloAliasAnalysis::Run(module_.get(), &alias_info_).value();
    hlo_live_range_ = HloLiveRange::Run(schedule, *alias_analysis_,
                                        module_->entry_computation())
                          .value();
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloLiveRange> hlo_live_range_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  // Shapes for use in the examples.
  Shape f32scalar_ = ShapeUtil::MakeShape(xla::F32, {});
  Shape f32vec4_ = ShapeUtil::MakeShape(F32, {4});

  // Returns the buffer defined at the given instruction and index.
  const HloValue* BufferAt(const HloInstruction* instruction,
                           const ShapeIndex& index) const {
    return &alias_analysis_->dataflow_analysis().GetUniqueValueAt(instruction,
                                                                  index);
  }

  HloLiveRange::LiveRangeBounds LiveRangeAt(
      const HloInstruction* instruction, const ShapeIndex& index = {}) const {
    auto* value = BufferAt(instruction, index);
    return hlo_live_range_->buffer_live_ranges().at(value);
  }

  // Checks if the logical times reported by instruction_schedule() matches
  // flattened_instruction_sequence().
  void CheckSchedule() const {
    const auto& flattened_instructions =
        hlo_live_range_->flattened_instruction_sequence().instructions();
    EXPECT_EQ(flattened_instructions.size(),
              hlo_live_range_->instruction_schedule().size());
    for (const auto& inst_and_time : hlo_live_range_->instruction_schedule()) {
      EXPECT_EQ(flattened_instructions.at(inst_and_time.second),
                inst_and_time.first)
          << "(flattened_inst[" << inst_and_time.second
          << "] = " << flattened_instructions.at(inst_and_time.second)->name()
          << ") != (inst_schedule[" << inst_and_time.second
          << "] = " << inst_and_time.first->name() << ")";
    }
  }
  AliasInfo alias_info_;
};

TEST_F(HloLiveRangeTest, Multiply) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  module_->AddEntryComputation(builder.Build());

  HloSchedule schedule(module_.get());

  schedule.set_sequence(module_->entry_computation(), {paramA, paramX, mul});

  Analyze(schedule);

  CheckSchedule();

  // Parameters live from beginning to end.
  EXPECT_EQ(LiveRangeAt(paramA), LiveRangeBounds({0, 3}));
  EXPECT_EQ(LiveRangeAt(paramX), LiveRangeBounds({0, 3}));
  // Mul lives after parameters are defined to the end.
  EXPECT_EQ(LiveRangeAt(mul), LiveRangeBounds({2, 3}));
}

TEST_F(HloLiveRangeTest, MultiplyAdd) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec4_, "paramY"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, mul, paramY));
  module_->AddEntryComputation(builder.Build());

  HloSchedule schedule(module_.get());

  schedule.set_sequence(module_->entry_computation(),
                        {paramA, paramX, mul, paramY, add});

  Analyze(schedule);

  CheckSchedule();

  // Parameters live from beginning to end.
  EXPECT_EQ(LiveRangeAt(paramA), LiveRangeBounds({0, 5}));
  EXPECT_EQ(LiveRangeAt(paramX), LiveRangeBounds({0, 5}));
  EXPECT_EQ(LiveRangeAt(paramY), LiveRangeBounds({0, 5}));
  // Mul starts after parameter are defined (Note: all parameters are defined at
  // 0, mul starts at 2 which is an arbitrary number).
  EXPECT_EQ(LiveRangeAt(mul), LiveRangeBounds({2, 4}));
  // Add lives after mul is defined to the end of the program.
  EXPECT_EQ(LiveRangeAt(add), LiveRangeBounds({4, 5}));
}

TEST_F(HloLiveRangeTest, LiveOutBuffers) {
  // If a buffer is live out, its life range is extended to the end of
  // computation.
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec4_, "paramY"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, mul, paramY));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({mul, add}));
  module_->AddEntryComputation(builder.Build());

  HloSchedule schedule(module_.get());

  schedule.set_sequence(module_->entry_computation(),
                        {paramA, paramX, mul, paramY, add, tuple});

  Analyze(schedule);

  CheckSchedule();

  // Parameters live from beginning to end.
  EXPECT_EQ(LiveRangeAt(paramA), LiveRangeBounds({0, 6}));
  EXPECT_EQ(LiveRangeAt(paramX), LiveRangeBounds({0, 6}));
  EXPECT_EQ(LiveRangeAt(paramY), LiveRangeBounds({0, 6}));
  // Mul starts after parameter are defined (Note: all parameters are defined at
  // 0, mul starts at 2 which is an arbitrary number).
  EXPECT_EQ(LiveRangeAt(mul), LiveRangeBounds({2, 6}));
  // Add lives after mul is defined to the end of the program.
  EXPECT_EQ(LiveRangeAt(add), LiveRangeBounds({4, 6}));
}

TEST_F(HloLiveRangeTest, InstructionScheduledAfterRoot) {
  // If a buffer is live out, its life range is extended to the end of
  // computation.
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec4_, "paramY"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, mul, paramY));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, mul, paramY));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({mul, add}));
  module_->AddEntryComputation(builder.Build());

  HloSchedule schedule(module_.get());

  // Schedule another instruction after root.
  schedule.set_sequence(module_->entry_computation(),
                        {paramA, paramX, mul, paramY, add, tuple, add2});

  Analyze(schedule);

  CheckSchedule();

  // Parameters live from beginning to end.
  EXPECT_EQ(LiveRangeAt(paramA), LiveRangeBounds({0, 7}));
  EXPECT_EQ(LiveRangeAt(paramX), LiveRangeBounds({0, 7}));
  EXPECT_EQ(LiveRangeAt(paramY), LiveRangeBounds({0, 7}));
  // Live out buffers live through the computation.

  EXPECT_EQ(LiveRangeAt(mul), LiveRangeBounds({2, 7}));
  EXPECT_EQ(LiveRangeAt(add), LiveRangeBounds({4, 7}));
  EXPECT_EQ(LiveRangeAt(tuple), LiveRangeBounds({5, 7}));
  EXPECT_EQ(LiveRangeAt(add2), LiveRangeBounds({6, 6}));
}

TEST_F(HloLiveRangeTest, AliasedParameter) {
  // If a parameter is non-readonly(non-aliased), its live range can end in the
  // middle of the program.
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec4_, "paramY"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, mul, paramY));
  module_->AddEntryComputation(builder.Build());
  // Set up alias of the first parameter.
  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias({}, 0, {}));

  HloSchedule schedule(module_.get());

  schedule.set_sequence(module_->entry_computation(),
                        {paramA, paramX, mul, paramY, add});

  Analyze(schedule);

  CheckSchedule();

  // Non-readonly parameter live like other normal buffers.
  EXPECT_EQ(LiveRangeAt(paramA), LiveRangeBounds({0, 2}));

  // Readonly parameters live from beginning to end.
  EXPECT_EQ(LiveRangeAt(paramX), LiveRangeBounds({0, 5}));
  EXPECT_EQ(LiveRangeAt(paramY), LiveRangeBounds({0, 5}));
  // Mul starts after parameter are defined (Note: all parameters are defined at
  // 0, mul starts at 2 which is an arbitrary number).
  EXPECT_EQ(LiveRangeAt(mul), LiveRangeBounds({2, 4}));
  // Add lives after mul is defined to the end of the program.
  EXPECT_EQ(LiveRangeAt(add), LiveRangeBounds({4, 5}));
}

TEST_F(HloLiveRangeTest, While) {
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data_increment =
      body_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}})));
  HloInstruction* body_data_mul =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, body_data, body_data));
  HloInstruction* body_data_add =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data, body_data_increment));
  HloInstruction* body_data_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data_add, body_data_mul));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data_next, body_iter_next}));
  HloComputation* body_computation =
      module_->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_iter"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({data, iter}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloComputation* entry_computation =
      module_->AddEntryComputation(builder.Build());

  HloSchedule schedule(module_.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data_increment, body_data_mul,
                         body_data_add, body_data_next, body_out});
  schedule.set_sequence(entry_computation, {iter, data, tuple, while_op});

  Analyze(schedule);

  CheckSchedule();

  // Check that there are no gaps in the live-ranges of buffer-sharing values.
  EXPECT_EQ(LiveRangeAt(iter).end, LiveRangeAt(cond_iter).start);
  EXPECT_EQ(LiveRangeAt(cond_iter).end, LiveRangeAt(body_iter).start);
  EXPECT_EQ(LiveRangeAt(body_iter).end, LiveRangeAt(body_iter_next).start);
}

TEST_F(HloLiveRangeTest, Determinism) {
  std::string hlo_string = R"(
HloModule While, is_scheduled=true

%WhileBody {
  %body_param = (f32[2,3], f32[], f32[2,3]) parameter(0)
  %get-tuple-element.2 = f32[2,3] get-tuple-element(%body_param), index=0
  %constant.2 = f32[2,3] constant({ { 1, 2, 3 }, { 4, 5, 6 } })
  %add.1 = f32[2,3] add(%get-tuple-element.2, %constant.2)
  %multiply = f32[2,3] multiply(%get-tuple-element.2, %get-tuple-element.2)
  %add.2 = f32[2,3] add(%add.1, %multiply)
  %get-tuple-element.1 = f32[] get-tuple-element(%body_param), index=1
  %constant.1 = f32[] constant(1)
  %add = f32[] add(%get-tuple-element.1, %constant.1)
  %get-tuple-element.3 = f32[2,3] get-tuple-element(%body_param), index=2
  %add.3 = f32[2,3] add(%get-tuple-element.3, %constant.2)
  ROOT %tuple = (f32[2,3], f32[], f32[2,3]) tuple(%add.2, %add, %add.3)
}

%WhileCond {
  %cond_param = (f32[2,3], f32[], f32[2,3]) parameter(0)
  %get-tuple-element = f32[] get-tuple-element(%cond_param), index=1
  %constant = f32[] constant(50)
  ROOT %compare = pred[] compare(%get-tuple-element, %constant), direction=LT
}

ENTRY %While {
  %param_iter = f32[2,3] parameter(0)
  %param_data = f32[] parameter(1)
  %tuple.1 = (f32[2,3], f32[], f32[2,3]) tuple(%param_iter, %param_data,
                                               %param_iter)
  %while = (f32[2,3], f32[], f32[2,3]) while(%tuple.1), condition=%WhileCond,
    body=%WhileBody
  ROOT %get-tuple-element.4 = f32[2,3] get-tuple-element(%while), index=0
}

)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();

  const int32_t num_runs = 20;
  std::vector<std::unique_ptr<HloLiveRange>> hlo_live_ranges;
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module_.get(), &alias_info_).value();

  for (int i = 0; i < num_runs; ++i) {
    hlo_live_ranges.push_back(HloLiveRange::Run(schedule, *alias_analysis,
                                                module_->entry_computation())
                                  .value());
  }

  absl::flat_hash_map<const HloValue*, HloLiveRange::LiveRangeBounds>&
      buffer_live_ranges_0 = hlo_live_ranges[0]->buffer_live_ranges();
  for (const auto& iter : buffer_live_ranges_0) {
    for (size_t i = 1; i < num_runs; i++) {
      absl::flat_hash_map<const HloValue*, HloLiveRange::LiveRangeBounds>&
          buffer_live_ranges_i = hlo_live_ranges[i]->buffer_live_ranges();
      auto found_iter = buffer_live_ranges_i.find(iter.first);
      EXPECT_TRUE(found_iter != buffer_live_ranges_i.end())
          << "value does not exist: " << iter.first->ToString();
      EXPECT_EQ(found_iter->second.start, iter.second.start)
          << "value " << iter.first->ToString()
          << " has different start: " << found_iter->second.start << " vs "
          << iter.second.start;
      EXPECT_EQ(found_iter->second.end, iter.second.end)
          << "value " << iter.first->ToString()
          << " has different end: " << found_iter->second.end << " vs "
          << iter.second.end;
    }
  }
}

TEST_F(HloLiveRangeTest, AsyncCall) {
  std::string hlo_string = R"(
HloModule AsyncCall, is_scheduled=true,
  entry_computation_layout={(f32[4096],f32[4096])->f32[4096]}

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096] parameter(0)
  %param_1 = f32[4096] parameter(1)
  %negate_2 = f32[4096] negate(%param_0)
  %negate_3 = f32[4096] negate(%param_1)
  ROOT %result.1 = f32[4096] add(%negate_2, %negate_3)
}

%async_wrapped (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096] parameter(0)
  %async_param.1 = f32[4096] parameter(1)
  ROOT %call = f32[4096] call(%async_param, %async_param.1),
    to_apply=%called_computation
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096] parameter(0)
  %b = f32[4096] parameter(1)
  %negate_0 = f32[4096] negate(%a)
  %negate_1 = f32[4096] negate(%b)
  %async-start = ((f32[4096], f32[4096]), f32[4096], u32[]) async-start(
    %negate_0, %negate_1), calls=%async_wrapped
  %add_0 = f32[4096] add(%negate_0, %negate_1)
  %async-done = f32[4096] async-done(
    ((f32[4096], f32[4096]), f32[4096], u32[]) %async-start)
  ROOT %add_1 = f32[4096] add(%add_0, %async-done)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  Analyze(schedule);

  CheckSchedule();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> aa,
                          HloAliasAnalysis::Run(module_.get(), &alias_info_));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloLiveRange> hlo_live_range,
                          HloLiveRange::Run(module_->schedule(), *aa,
                                            module_->entry_computation()));

  absl::flat_hash_map<std::string, std::pair<int32_t, int32_t>> inst_ranges;
  for (auto& [value, time_bound] : hlo_live_range->buffer_live_ranges()) {
    inst_ranges[value->instruction()->name()] = {time_bound.start,
                                                 time_bound.end};
  }

  // Parameters live range spans whole computation.
  EXPECT_EQ(inst_ranges["a"], std::make_pair(0, 16));
  EXPECT_EQ(inst_ranges["b"], std::make_pair(0, 16));

  // Check `add` operations live range to make sure that `negate` values live
  // range spans past the last non-async use.
  EXPECT_EQ(inst_ranges["add_0"], std::make_pair(13, 15));
  EXPECT_EQ(inst_ranges["add_1"], std::make_pair(15, 16));

  // `negate_0` and `negate_1` live range ends after `async-done`.
  EXPECT_EQ(inst_ranges["negate_0"], std::make_pair(2, 14));
  EXPECT_EQ(inst_ranges["negate_1"], std::make_pair(3, 14));
}

TEST_F(HloLiveRangeTest, LateBoundAsyncCall) {
  std::string hlo_string = R"(
HloModule LateBoundAsyncCall, is_scheduled=true,
  entry_computation_layout={(f32[4096]{0})->f32[4096]{0}}

%called_computation (param_0: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  ROOT %negate_2 = f32[4096]{0} negate(%param_0)
}

%async_wrapped (async_param: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  ROOT %call = f32[4096]{0} call(%async_param), to_apply=%called_computation
}

ENTRY %main (a: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %negate_0 = f32[4096]{0} negate(%a)
  %async-start = ((), (), s32[]) async-start(), calls=%async_wrapped
  %negate_1 = f32[4096]{0} negate(%a)
  %async-update = ((f32[4096]{0}), f32[4096]{0}, s32[]) async-update(
    %async-start, %negate_0), calls=%async_wrapped
  %async-done = f32[4096]{0} async-done(%async-update), calls=%async_wrapped
  ROOT %add_1 = f32[4096]{0} add(%negate_1, %async-done)
}
)";

  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  Analyze(schedule);

  CheckSchedule();

  HloComputation* entry = module_->entry_computation();
  HloComputation* async_wrapped =
      module_->GetComputationWithName("async_wrapped");
  HloComputation* called_comp =
      module_->GetComputationWithName("called_computation");

  ASSERT_NE(async_wrapped, nullptr);
  ASSERT_NE(called_comp, nullptr);

  const HloInstruction* a = entry->GetInstructionWithName("a");
  const HloInstruction* negate_0 = entry->GetInstructionWithName("negate_0");
  const HloInstruction* async_start =
      entry->GetInstructionWithName("async-start");
  const HloInstruction* negate_1 = entry->GetInstructionWithName("negate_1");
  const HloInstruction* async_update =
      entry->GetInstructionWithName("async-update");
  const HloInstruction* async_done =
      entry->GetInstructionWithName("async-done");
  const HloInstruction* add_1 = entry->GetInstructionWithName("add_1");

  const HloInstruction* async_param =
      async_wrapped->GetInstructionWithName("async_param");
  const HloInstruction* call = async_wrapped->GetInstructionWithName("call");

  const HloInstruction* param_0 =
      called_comp->GetInstructionWithName("param_0");
  const HloInstruction* negate_2 =
      called_comp->GetInstructionWithName("negate_2");

  const auto& schedule_map = hlo_live_range_->instruction_schedule();

  EXPECT_EQ(schedule_map.at(a), 0);
  EXPECT_EQ(schedule_map.at(negate_0), 1);
  EXPECT_EQ(schedule_map.at(async_start), 2);
  EXPECT_EQ(schedule_map.at(negate_1), 3);
  EXPECT_EQ(schedule_map.at(async_param), 4);
  EXPECT_EQ(schedule_map.at(param_0), 5);
  EXPECT_EQ(schedule_map.at(negate_2), 6);
  EXPECT_EQ(schedule_map.at(call), 7);
  EXPECT_EQ(schedule_map.at(async_update), 8);
  EXPECT_EQ(schedule_map.at(async_done), 9);
  EXPECT_EQ(schedule_map.at(add_1), 10);
}

TEST_F(HloLiveRangeTest, ProgressiveLateBoundAsyncCall) {
  std::string hlo_string = R"(
HloModule ProgressiveLateBoundAsyncCall, is_scheduled=true,
  entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %add_0 = f32[4096]{0} add(%param_0, %param_1)
}

%async_wrapped (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(%async_param, %async_param.1), to_apply=%called_computation
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %negate_0 = f32[4096]{0} negate(%a)
  %negate_1 = f32[4096]{0} negate(%b)
  %async-start = ((), (), s32[]) async-start(), calls=%async_wrapped
  %async-update.1 = ((f32[4096]{0}), (), s32[]) async-update(
    %async-start, %negate_0), calls=%async_wrapped
  %async-update.2 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, s32[])
    async-update(%async-update.1, %negate_1), calls=%async_wrapped
  %async-done = f32[4096]{0} async-done(%async-update.2), calls=%async_wrapped
  ROOT %add_1 = f32[4096]{0} add(%async-done, %async-done)
}
)";

  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  Analyze(schedule);

  CheckSchedule();

  HloComputation* entry = module_->entry_computation();
  HloComputation* async_wrapped =
      module_->GetComputationWithName("async_wrapped");
  HloComputation* called_comp =
      module_->GetComputationWithName("called_computation");

  ASSERT_NE(async_wrapped, nullptr);
  ASSERT_NE(called_comp, nullptr);

  const HloInstruction* a = entry->GetInstructionWithName("a");
  const HloInstruction* b = entry->GetInstructionWithName("b");
  const HloInstruction* negate_0 = entry->GetInstructionWithName("negate_0");
  const HloInstruction* negate_1 = entry->GetInstructionWithName("negate_1");
  const HloInstruction* async_start =
      entry->GetInstructionWithName("async-start");
  const HloInstruction* async_update_1 =
      entry->GetInstructionWithName("async-update.1");
  const HloInstruction* async_update_2 =
      entry->GetInstructionWithName("async-update.2");
  const HloInstruction* async_done =
      entry->GetInstructionWithName("async-done");
  const HloInstruction* add_1 = entry->GetInstructionWithName("add_1");

  const HloInstruction* async_param =
      async_wrapped->GetInstructionWithName("async_param");
  const HloInstruction* async_param_1 =
      async_wrapped->GetInstructionWithName("async_param.1");
  const HloInstruction* call = async_wrapped->GetInstructionWithName("call");

  const HloInstruction* param_0 =
      called_comp->GetInstructionWithName("param_0");
  const HloInstruction* param_1 =
      called_comp->GetInstructionWithName("param_1");
  const HloInstruction* add_0 = called_comp->GetInstructionWithName("add_0");

  const auto& schedule_map = hlo_live_range_->instruction_schedule();

  EXPECT_EQ(schedule_map.at(a), 0);
  EXPECT_EQ(schedule_map.at(b), 1);
  EXPECT_EQ(schedule_map.at(negate_0), 2);
  EXPECT_EQ(schedule_map.at(negate_1), 3);
  EXPECT_EQ(schedule_map.at(async_start), 4);

  EXPECT_EQ(schedule_map.at(async_update_1), 5);

  // The called computation must be expanded right before async-update.2
  // (which fully binds them)
  EXPECT_EQ(schedule_map.at(async_param), 6);
  EXPECT_EQ(schedule_map.at(async_param_1), 7);
  EXPECT_EQ(schedule_map.at(param_0), 8);
  EXPECT_EQ(schedule_map.at(param_1), 9);
  EXPECT_EQ(schedule_map.at(add_0), 10);
  EXPECT_EQ(schedule_map.at(call), 11);

  EXPECT_EQ(schedule_map.at(async_update_2), 12);
  EXPECT_EQ(schedule_map.at(async_done), 13);
  EXPECT_EQ(schedule_map.at(add_1), 14);
}

TEST_F(HloLiveRangeTest, DoneBoundLateBoundAsyncCall) {
  std::string hlo_string = R"(
HloModule DoneBoundLateBoundAsyncCall, is_scheduled=true,
  entry_computation_layout={(f32[4096]{0})->f32[4096]{0}}

%called_computation (param_0: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  ROOT %negate_2 = f32[4096]{0} negate(%param_0)
}

%async_wrapped (async_param: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  ROOT %call = f32[4096]{0} call(%async_param), to_apply=%called_computation
}

ENTRY %main (a: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %negate_0 = f32[4096]{0} negate(%a)
  %async-start = ((), (), s32[]) async-start(), calls=%async_wrapped
  %async-update = ((f32[4096]{0}), (), s32[]) async-update(
    %async-start, %negate_0), calls=%async_wrapped
  %async-done = f32[4096]{0} async-done(%async-update), calls=%async_wrapped
  ROOT %add_1 = f32[4096]{0} add(%async-done, %async-done)
}
)";

  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  Analyze(schedule);

  CheckSchedule();

  HloComputation* entry = module_->entry_computation();
  HloComputation* async_wrapped =
      module_->GetComputationWithName("async_wrapped");
  HloComputation* called_comp =
      module_->GetComputationWithName("called_computation");

  ASSERT_NE(async_wrapped, nullptr);
  ASSERT_NE(called_comp, nullptr);

  const HloInstruction* a = entry->GetInstructionWithName("a");
  const HloInstruction* negate_0 = entry->GetInstructionWithName("negate_0");
  const HloInstruction* async_start =
      entry->GetInstructionWithName("async-start");
  const HloInstruction* async_update =
      entry->GetInstructionWithName("async-update");
  const HloInstruction* async_done =
      entry->GetInstructionWithName("async-done");
  const HloInstruction* add_1 = entry->GetInstructionWithName("add_1");

  const HloInstruction* async_param =
      async_wrapped->GetInstructionWithName("async_param");
  const HloInstruction* call = async_wrapped->GetInstructionWithName("call");

  const HloInstruction* param_0 =
      called_comp->GetInstructionWithName("param_0");
  const HloInstruction* negate_2 =
      called_comp->GetInstructionWithName("negate_2");

  const auto& schedule_map = hlo_live_range_->instruction_schedule();

  EXPECT_EQ(schedule_map.at(a), 0);
  EXPECT_EQ(schedule_map.at(negate_0), 1);
  EXPECT_EQ(schedule_map.at(async_start), 2);
  EXPECT_EQ(schedule_map.at(async_update), 3);

  // The called computation must be expanded right before async-done
  // (which fully binds the output result)
  EXPECT_EQ(schedule_map.at(async_param), 4);
  EXPECT_EQ(schedule_map.at(param_0), 5);
  EXPECT_EQ(schedule_map.at(negate_2), 6);
  EXPECT_EQ(schedule_map.at(call), 7);

  EXPECT_EQ(schedule_map.at(async_done), 8);
  EXPECT_EQ(schedule_map.at(add_1), 9);
}

TEST_F(HloLiveRangeTest, Call) {
  std::string hlo_string = R"(
    HloModule Call, is_scheduled=true

    %called_computation (param_0: f32[4096]) -> f32[4096] {
      %param_0 = f32[4096] parameter(0)
      ROOT %negate_0 = f32[4096] negate(%param_0)
    }

    ENTRY %main (a: f32[4096]) -> f32[4096] {
      %a = f32[4096] parameter(0)
      %b = f32[4096] negate(%a)
      %c = f32[4096] call(%b), to_apply=%called_computation
      %d = f32[4096] negate(%c)
      ROOT %e = f32[4096] add(%c, %d)
    })";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> aa,
                          HloAliasAnalysis::Run(module_.get(), &alias_info_));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloLiveRange> hlo_live_range,
                          HloLiveRange::Run(module_->schedule(), *aa,
                                            module_->entry_computation()));

  absl::flat_hash_map<std::string, std::pair<int32_t, int32_t>> inst_ranges;
  for (auto& [value, time_bound] : hlo_live_range->buffer_live_ranges()) {
    inst_ranges[value->instruction()->name()] = {time_bound.start,
                                                 time_bound.end};
  }

  EXPECT_EQ(inst_ranges["a"], std::make_pair(0, 7));
  EXPECT_EQ(inst_ranges["b"], std::make_pair(1, 3));
  EXPECT_EQ(inst_ranges["negate_0"], std::make_pair(3, 6));
  EXPECT_EQ(inst_ranges["d"], std::make_pair(5, 6));
  EXPECT_EQ(inst_ranges["e"], std::make_pair(6, 7));
}

TEST_F(HloLiveRangeTest, ToStringSimpleComputation) {
  std::string hlo_string = R"(
HloModule Module, is_scheduled=true

ENTRY %main {
  %paramA = f32[4] parameter(0)
  %paramX = f32[4] parameter(1)
  ROOT %multiply = f32[4] multiply(%paramA, %paramX)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  Analyze(schedule);

  // The peak is at LogicalTime=2, where all three buffers are live. Each array
  // of four F32 elements takes 16 bytes.
  std::string expected_string = R"(HloLiveRange (max 3):
  InstructionSequence:
    0:paramA
    1:paramX
    2:multiply
  BufferLiveRange:
    paramA{}:0-3
    paramX{}:0-3
    multiply{}:2-3
  Live ranges at 2 (peak):
    multiply{}: 16 bytes (cumulative: 16 bytes)
    paramA{}: 16 bytes (cumulative: 32 bytes)
    paramX{}: 16 bytes (cumulative: 48 bytes)
  Stack trace breakdown for peak usage: 48 bytes
    main (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
      ├── multiply (33.3%, total: 16 bytes, current: 16 bytes, remaining: 32 bytes)
      ├── paramA (33.3%, total: 16 bytes, current: 16 bytes, remaining: 16 bytes)
      └── paramX (33.3%, total: 16 bytes, current: 16 bytes, remaining: 0 bytes)
)";
  EXPECT_EQ(hlo_live_range_->ToString(), expected_string);
}

TEST_F(HloLiveRangeTest, ToStringComputationWithTuple) {
  std::string hlo_string = R"(
HloModule Module, is_scheduled=true

ENTRY %main {
  %paramA = f32[4] parameter(0)
  %constant = (f32[], f32[4]) constant((1.0, {2.0, 3.0, 4.0, 5.0}))
  %get-tuple-element = f32[4] get-tuple-element(%constant), index=1
  ROOT %add = f32[4] add(%paramA, %get-tuple-element)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  Analyze(schedule);

  // The peak time is at LogicalTime=1, when both constants in the tuple are
  // live. The tuple itself has two pointers of 8 bytes each, and the two
  // constants in the tuple have 4 and 16 bytes respectively.
  std::string expected_string = R"(HloLiveRange (max 4):
  InstructionSequence:
    0:paramA
    1:constant
    2:get-tuple-element
    3:add
  BufferLiveRange:
    paramA{}:0-4
    constant{}:1-2
    constant{0}:1-1
    constant{1}:1-3
    add{}:3-4
  Live ranges at 1 (peak):
    constant{}: 16 bytes (cumulative: 16 bytes)
    constant{1}: 16 bytes (cumulative: 32 bytes)
    paramA{}: 16 bytes (cumulative: 48 bytes)
    constant{0}: 4 bytes (cumulative: 52 bytes)
  Stack trace breakdown for peak usage: 52 bytes
    main (100.0%, total: 52 bytes, current: 0 bytes, remaining: 52 bytes)
      ├── constant (30.8%, total: 16 bytes, current: 16 bytes, remaining: 36 bytes)
      ├── constant{1} (30.8%, total: 16 bytes, current: 16 bytes, remaining: 20 bytes)
      ├── paramA (30.8%, total: 16 bytes, current: 16 bytes, remaining: 4 bytes)
      └── constant{0} (7.7%, total: 4 bytes, current: 4 bytes, remaining: 0 bytes)
)";
  EXPECT_EQ(hlo_live_range_->ToString(), expected_string);
}

TEST_F(HloLiveRangeTest, AsyncCallLateBinding) {
  std::string hlo_string = R"(
HloModule AsyncCall, is_scheduled=true,
  entry_computation_layout={(f32[4096]{0},f32[4096]{0})->f32[4096]{0}}

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  %negate_2 = f32[4096]{0} negate(%param_0)
  %negate_3 = f32[4096]{0} negate(%param_1)
  ROOT %result.1 = f32[4096]{0} add(%negate_2, %negate_3)
}

%async_wrapped (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(%async_param, %async_param.1),
    to_apply=%called_computation
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096] parameter(0)
  %b = f32[4096] parameter(1)
  %negate_0 = f32[4096] negate(%a)
  %negate_1 = f32[4096] negate(%b)
  %async-start = ((f32[4096]), (), u32[]) async-start(%negate_0),
    calls=%async_wrapped
  %async-update = ((f32[4096], f32[4096]), f32[4096], u32[]) async-update(
    %async-start, %negate_1)
  %add_0 = f32[4096] add(f32[4096]{0} %negate_0, f32[4096]{0} %negate_1)
  %async-done = f32[4096] async-done(%async-update)
  ROOT %add_1 = f32[4096] add(%add_0, %async-done)
}
)";

  /*
    0: a
    1: b
    2: negate_0
    3: negate_1
    4: async-param
    5: async-param.1
    6: param_0
    7: param_1
    8: negate_2
    9: negate_3
    10: result.1
    11: call
    12: async-start
    13: async-update
    14: add_0
    15: async-done
    16: add_1
  */

  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  const HloSchedule& schedule = module_->schedule();
  VLOG(0) << "schedule: " << schedule.ToString();
  Analyze(schedule);

  CheckSchedule();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> aa,
                       HloAliasAnalysis::Run(module_.get(), &alias_info_));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloLiveRange> hlo_live_range,
                       HloLiveRange::Run(module_->schedule(), *aa,
                                         module_->entry_computation()));

  absl::flat_hash_map<std::string, std::pair<int32_t, int32_t>> inst_ranges;
  for (auto& [value, time_bound] : hlo_live_range->buffer_live_ranges()) {
    inst_ranges[value->instruction()->name()] = {time_bound.start,
                                                 time_bound.end};
  }

  EXPECT_EQ(inst_ranges["a"], std::make_pair(0, 17));
  EXPECT_EQ(inst_ranges["b"], std::make_pair(0, 17));

  EXPECT_EQ(inst_ranges["add_0"], std::make_pair(14, 16));
  EXPECT_EQ(inst_ranges["add_1"], std::make_pair(16, 17));

  EXPECT_EQ(inst_ranges["negate_0"], std::make_pair(2, 15));
  EXPECT_EQ(inst_ranges["negate_1"], std::make_pair(3, 15));
}

}  // namespace
}  // namespace xla
