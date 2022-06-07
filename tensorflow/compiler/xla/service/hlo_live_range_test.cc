/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/hlo_live_range.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using TimeBound = HloLiveRange::TimeBound;
class HloLiveRangeTest : public HloTestBase {
 protected:
  HloLiveRangeTest() : module_(CreateNewVerifiedModule()) {}
  ~HloLiveRangeTest() override {}

  void Analyze(const HloSchedule& schedule) {
    alias_analysis_ = HloAliasAnalysis::Run(module_.get()).ValueOrDie();
    hlo_live_range_ = HloLiveRange::Run(schedule, *alias_analysis_,
                                        module_->entry_computation())
                          .ValueOrDie();
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

  HloLiveRange::TimeBound LiveRangeAt(const HloInstruction* instruction,
                                      const ShapeIndex& index = {}) const {
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
  EXPECT_EQ(LiveRangeAt(paramA), TimeBound({0, 3}));
  EXPECT_EQ(LiveRangeAt(paramX), TimeBound({0, 3}));
  // Mul lives after parameters are defined to the end.
  EXPECT_EQ(LiveRangeAt(mul), TimeBound({2, 3}));
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
  EXPECT_EQ(LiveRangeAt(paramA), TimeBound({0, 5}));
  EXPECT_EQ(LiveRangeAt(paramX), TimeBound({0, 5}));
  EXPECT_EQ(LiveRangeAt(paramY), TimeBound({0, 5}));
  // Mul starts after parameter are defined (Note: all parameters are defined at
  // 0, mul starts at 2 which is an arbitrary number).
  EXPECT_EQ(LiveRangeAt(mul), TimeBound({2, 4}));
  // Add lives after mul is defined to the end of the program.
  EXPECT_EQ(LiveRangeAt(add), TimeBound({4, 5}));
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
  EXPECT_EQ(LiveRangeAt(paramA), TimeBound({0, 6}));
  EXPECT_EQ(LiveRangeAt(paramX), TimeBound({0, 6}));
  EXPECT_EQ(LiveRangeAt(paramY), TimeBound({0, 6}));
  // Mul starts after parameter are defined (Note: all parameters are defined at
  // 0, mul starts at 2 which is an arbitrary number).
  EXPECT_EQ(LiveRangeAt(mul), TimeBound({2, 6}));
  // Add lives after mul is defined to the end of the program.
  EXPECT_EQ(LiveRangeAt(add), TimeBound({4, 6}));
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
  EXPECT_EQ(LiveRangeAt(paramA), TimeBound({0, 7}));
  EXPECT_EQ(LiveRangeAt(paramX), TimeBound({0, 7}));
  EXPECT_EQ(LiveRangeAt(paramY), TimeBound({0, 7}));
  // Live out buffers live through the computation.

  EXPECT_EQ(LiveRangeAt(mul), TimeBound({2, 7}));
  EXPECT_EQ(LiveRangeAt(add), TimeBound({4, 7}));
  EXPECT_EQ(LiveRangeAt(tuple), TimeBound({5, 7}));
  EXPECT_EQ(LiveRangeAt(add2), TimeBound({6, 6}));
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
  EXPECT_EQ(LiveRangeAt(paramA), TimeBound({0, 2}));

  // Readonly parameters live from beginning to end.
  EXPECT_EQ(LiveRangeAt(paramX), TimeBound({0, 5}));
  EXPECT_EQ(LiveRangeAt(paramY), TimeBound({0, 5}));
  // Mul starts after parameter are defined (Note: all parameters are defined at
  // 0, mul starts at 2 which is an arbitrary number).
  EXPECT_EQ(LiveRangeAt(mul), TimeBound({2, 4}));
  // Add lives after mul is defined to the end of the program.
  EXPECT_EQ(LiveRangeAt(add), TimeBound({4, 5}));
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
}  // namespace
}  // namespace xla
