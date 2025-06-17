/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/copy_insertion.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test_benchmark.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ::testing::NotNull;
using ::testing::UnorderedElementsAre;

int64_t CountCopies(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64_t CountCopies(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

int64_t CountControlEdges(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

int64_t CountControlEdges(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountControlEdges(*computation);
  }
  return count;
}

class CopyInsertionTest : public HloHardwareIndependentTestBase {
 protected:
  void InsertCopies(HloModule* module) {
    CopyInsertion copy_insertion;
    VLOG(3) << "Before copy inser: " << module->ToString();
    ASSERT_IS_OK(copy_insertion.Run(module).status());
    VLOG(2) << "After copy inser: " << module->ToString();
  }

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
  const AliasInfo alias_info_;
};

TEST_F(CopyInsertionTest, SingleParameter) {
  // Computation is a single parameter passed into a tuple. The parameter should
  // be copied before entering the tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "x"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({x}));

  EXPECT_THAT(x->users(), UnorderedElementsAre(tuple));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(x)));
}

TEST_F(CopyInsertionTest, SingleConstant) {
  // Computation is a single constant passed into a tuple. The parameter should
  // be copied before entering the tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant}));

  EXPECT_THAT(constant->users(), UnorderedElementsAre(tuple));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(constant)));
}

TEST_F(CopyInsertionTest, ExistingCopiesNotRemoved) {
  // Verify that kCopy instructions which change layout and exist before
  // copy-insertion remain in the graph after copy-insertion.
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}})));
  auto minor_to_major = LayoutUtil::MinorToMajor(constant->shape());
  Layout reversed_layout =
      LayoutUtil::MakeLayoutFromMajorToMinor(minor_to_major);
  Shape copy_shape = constant->shape();
  *copy_shape.mutable_layout() = reversed_layout;
  HloInstruction* copy_1 = builder.AddInstruction(
      HloInstruction::CreateUnary(copy_shape, HloOpcode::kCopy, constant));
  HloInstruction* copy_2 = builder.AddInstruction(
      HloInstruction::CreateUnary(copy_shape, HloOpcode::kCopy, constant));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kAdd, copy_1, copy_2));
  builder.AddInstruction(
      HloInstruction::CreateUnary(add->shape(), HloOpcode::kCopy, add));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(CountCopies(*module), 3);

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_EQ(module->entry_computation()->root_instruction(), add);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

TEST_F(CopyInsertionTest, MultipleConstantsAndParameters) {
  // Create a computation with more than one constant and parameter. Only one of
  // each constant/parameter is pointed to by the output tuple. Only these
  // instructions should be copied.
  auto builder = HloComputation::Builder(TestName());

  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));

  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "y"));

  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, constant1, y));

  builder.AddInstruction(HloInstruction::CreateTuple({constant2, x, add}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Copy(constant2), op::Copy(x), op::Add(constant1, y)));
}

TEST_F(CopyInsertionTest, BitcastParameter) {
  // The output of a bitcast is its operand (same buffer), so a bitcast
  // parameter feeding the result must have a copy added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "x"));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(ShapeUtil::MakeShape(F32, {2, 2}), x));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(x->users(), UnorderedElementsAre(bitcast));

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(old_root));
}

TEST_F(CopyInsertionTest, BitcastConstant) {
  // The output of a bitcast is its operand (same buffer), so a bitcast
  // constant feeding the result must have a copy added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1.0, 42.0})));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(ShapeUtil::MakeShape(F32, {2}), constant));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(constant->users(), UnorderedElementsAre(bitcast));

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(old_root));
}

TEST_F(CopyInsertionTest, BitcastTupleElementParameter) {
  // Same as BitcastParameter, but the bitcast is wrapped in a tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "x"));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(ShapeUtil::MakeShape(F32, {2, 2}), x));
  builder.AddInstruction(HloInstruction::CreateTuple({bitcast}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(x->users(), UnorderedElementsAre(bitcast));

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(bitcast)));
}

TEST_F(CopyInsertionTest, NestedTupleParameter) {
  // Construct a trivial computation where the root of the computation is a
  // nested tuple-shaped parameter. The parameter should be deep copied and the
  // copy should be the root of the computation.
  auto builder = HloComputation::Builder(TestName());

  // Param shape is: ((F32[], S32[1,2,3]), F32[42])
  builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {}),
                                      ShapeUtil::MakeShape(S32, {1, 2, 3})}),
           ShapeUtil::MakeShape(F32, {42})}),
      "param0"));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kParameter,
            module->entry_computation()->root_instruction()->opcode());

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 3);

  HloInstruction* new_root = module->entry_computation()->root_instruction();
  EXPECT_NE(old_root, new_root);

  EXPECT_THAT(
      new_root,
      op::Tuple(
          op::Tuple(
              op::Copy(op::GetTupleElement(op::GetTupleElement(old_root))),
              op::Copy(op::GetTupleElement(op::GetTupleElement(old_root)))),
          op::Copy(op::GetTupleElement(old_root))));
}

TEST_F(CopyInsertionTest, ElementOfNestedTupleParameter) {
  // Construct a computation where the root of the computation is a tuple
  // element of a nested tuple-shaped parameter.
  auto builder = HloComputation::Builder(TestName());

  // Param shape is: ((F32[], S32[1,2,3]), F32[42])
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {}),
                                      ShapeUtil::MakeShape(S32, {1, 2, 3})}),
           ShapeUtil::MakeShape(F32, {42})}),
      "param0"));

  // The return value of the computation is the zero-th element of the nested
  // tuple. This element is itself a tuple.
  auto gte = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(param->shape(), {0}), param, 0));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gte, module->entry_computation()->root_instruction());

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Copy(op::GetTupleElement(op::GetTupleElement(param))),
                op::Copy(op::GetTupleElement(op::GetTupleElement(param)))));
}

class WhileCopyInsertionTest : public CopyInsertionTest {
 protected:
  WhileCopyInsertionTest() : module_(CreateNewVerifiedModule()) {}

  // Builds a While condition computation which reads the induction variable
  // from the tuple parameter, and returns a predicate indicating whether this
  // value is less than the constant '10'.
  // The parameter 'nested' specifies the loop state shape from which to
  // read the induction variable.
  std::unique_ptr<HloComputation> BuildConditionComputation(
      const Shape& loop_state_shape) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(10)));
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            limit_const->shape(), loop_state, 0));
    builder.AddInstruction(HloInstruction::CreateCompare(
        condition_result_shape_, induction_variable, limit_const,
        ComparisonDirection::kLt));
    return builder.Build();
  }

  // Builds a While body computation with one output tuple element dependent on
  // both input tuple elements.
  // EX:
  // Body({in0, in1})
  //   out0 = Add(in0, 1)
  //   out1 = Add(BCast(in0), in1)
  //   Tuple(out0, out1)
  std::unique_ptr<HloComputation> BuildDependentBodyComputation() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(1).
    auto data = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));
    // Use 'induction_variable' in computation with no path to output tuple.
    Shape f32_scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto convert = builder.AddInstruction(
        HloInstruction::CreateConvert(f32_scalar_shape, induction_variable));
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, convert, {}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
    return builder.Build();
  }

  // Builds a While body computation with two output tuple elements dependent on
  // both input tuple elements.
  //
  // EX: Body({in0, in1, in2})
  //   out0 = Add(in0, 1)
  //   out1 = in1
  //   out2 = in2
  //   Tuple(out0, out1, out2)
  std::unique_ptr<HloComputation> BuildDependentBodyComputation2() {
    auto builder = HloComputation::Builder(TestName() + ".Body");

    const Shape& loop_state_shape = ShapeUtil::MakeTupleShape(
        {induction_variable_shape_, data_shape_, data_shape_});

    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));

    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));

    // add0 = Add(in0, 1)
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // data1 = GTE(1).
    HloInstruction* data1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));

    // data2 = GTE(2).
    HloInstruction* data2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 2));

    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({add0, data1, data2}));

    return builder.Build();
  }

  // Builds a While body computation with read-only tuple element 0.
  // EX:
  // Body({in0, in1})
  //   out0 = in0
  //   out1 = Add(BCast(in0), in1)
  //   Tuple(out0, out1)
  std::unique_ptr<HloComputation> BuildDependentBodyOneReadOnlyComputation() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    // Update data GTE(1).
    auto data = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));

    // Use 'induction_variable' in computation with no path to output tuple.
    Shape f32_scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto convert = builder.AddInstruction(
        HloInstruction::CreateConvert(f32_scalar_shape, induction_variable));
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, convert, {}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    builder.AddInstruction(
        HloInstruction::CreateTuple({induction_variable, add1}));
    return builder.Build();
  }

  // Builds a While body computation with independent outputs.
  // EX:
  // Body({in0, in1})
  //   out0 = Add(in0, 1)
  //   out1 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
  //   Tuple(out0, out1)
  std::unique_ptr<HloComputation> BuildIndependentBodyComputation(
      bool nested = false) {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    const Shape& loop_state_shape =
        nested ? nested_loop_state_shape_ : loop_state_shape_;

    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
    // add0 = Add(in0, 1)
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(1).
    HloInstruction* data = nullptr;
    if (nested) {
      data = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          nested_tuple_shape_, loop_state, 1));
      data = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(data_shape_, data, 0));
    } else {
      data = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));
    }
    auto update = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    // add1 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    if (nested) {
      auto nested_tuple =
          builder.AddInstruction(HloInstruction::CreateTuple({add1, add1}));
      builder.AddInstruction(HloInstruction::CreateTuple({add0, nested_tuple}));
    } else {
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
    }
    return builder.Build();
  }

  // Builds a While body computation with the following nested tuple
  // sub-computation:
  //                            |
  //                    GTE(loop_state, 1)
  //                       /           \
  // GTE(GTE(loop_state, 1), 0)     GTE(GTE(loop_state, 1), 1)
  //           |                              |
  //          Add                           Reverse
  //           |                              |
  std::unique_ptr<HloComputation> BuildNestedBodyComputation() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, nested_loop_state_shape_, "loop_state"));
    // Update GTE(0).
    auto gte0 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        gte0->shape(), HloOpcode::kAdd, gte0, inc));

    // GTE(loop_state, 1)
    auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        nested_tuple_shape_, loop_state, 1));
    // GTE(GTE(loop_state, 1), 0) -> Add
    auto gte10 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, gte1, 0));
    auto update10 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    auto add10 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, gte10, update10));

    // GTE(GTE(loop_state, 1), 1) -> Reverse
    auto gte11 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, gte1, 1));
    auto rev11 = builder.AddInstruction(
        HloInstruction::CreateReverse(data_shape_, gte11, {0}));

    // Create output Tuple.
    auto inner_tuple =
        builder.AddInstruction(HloInstruction::CreateTuple({add10, rev11}));
    builder.AddInstruction(HloInstruction::CreateTuple({add0, inner_tuple}));
    return builder.Build();
  }

  // Builds a While instruction using 'condition' and 'body' sub-computations.
  // Init operand is initialized to zeros of appropriate shape.
  HloInstruction* BuildWhileInstruction(HloComputation* condition,
                                        HloComputation* body,
                                        bool nested = false) {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto induction_var_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));

    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));

    if (nested) {
      auto inner_init = builder.AddInstruction(
          HloInstruction::CreateTuple({data_init, data_init}));
      auto loop_state_init = builder.AddInstruction(
          HloInstruction::CreateTuple({induction_var_init, inner_init}));
      auto while_hlo = builder.AddInstruction(HloInstruction::CreateWhile(
          loop_state_init->shape(), condition, body, loop_state_init));
      module_->AddEntryComputation(builder.Build());
      return while_hlo;
    }

    auto loop_state_init = builder.AddInstruction(
        HloInstruction::CreateTuple({induction_var_init, data_init}));
    auto while_hlo = builder.AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape_, condition, body, loop_state_init));
    module_->AddEntryComputation(builder.Build());
    return while_hlo;
  }

  HloInstruction* BuildWhileInstruction_InitPointsToConstant() {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));
    return BuildWhileInstructionWithCustomInit(loop_state_shape_, data_init,
                                               &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToParameter() {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape_, "data_init"));
    return BuildWhileInstructionWithCustomInit(loop_state_shape_, data_init,
                                               &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToNonDistinct() {
    auto builder = HloComputation::Builder(TestName() + ".While");

    auto one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto one_vec = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, one, {}));
    auto data_init =
        builder.AddInstruction(HloInstruction::CreateTuple({one_vec, one_vec}));

    return BuildWhileInstructionWithCustomInit(nested_loop_state_shape_,
                                               data_init, &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToInterfering() {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, one, {}));
    auto one_vec = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    // Take a reference to 'data_init' to make it interfere with while result.
    auto add = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data_init, one_vec));

    auto xla_while = BuildWhileInstructionWithCustomInit(loop_state_shape_,
                                                         data_init, &builder);

    // Add an additional binary operation operating on the while and the
    // interfering add so that neither operation is dead.
    auto gte = xla_while->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetSubshape(xla_while->shape(), {1}), xla_while, 1));
    auto sub = xla_while->parent()->AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kSubtract, add, gte));
    auto gte0 = xla_while->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetSubshape(xla_while->shape(), {0}), xla_while, 0));
    auto tuple = xla_while->parent()->AddInstruction(
        HloInstruction::CreateTuple({gte0, sub}));

    xla_while->parent()->set_root_instruction(tuple);

    return xla_while;
  }

  HloInstruction* BuildWhileInstructionWithCustomInit(
      const Shape& loop_state_shape, HloInstruction* data_init,
      HloComputation::Builder* builder) {
    const bool nested =
        ShapeUtil::Equal(loop_state_shape, nested_loop_state_shape_);
    auto induction_var_init = builder->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
    auto condition = module_->AddEmbeddedComputation(
        BuildConditionComputation(loop_state_shape));
    auto body = module_->AddEmbeddedComputation(
        BuildIndependentBodyComputation(nested));
    auto loop_state_init = builder->AddInstruction(
        HloInstruction::CreateTuple({induction_var_init, data_init}));
    auto while_hlo = builder->AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape, condition, body, loop_state_init));
    module_->AddEntryComputation(builder->Build());
    return while_hlo;
  }

  std::unique_ptr<HloModule> module_;
  Shape induction_variable_shape_ = ShapeUtil::MakeShape(S32, {});
  Shape data_shape_ = ShapeUtil::MakeShape(F32, {8});
  Shape loop_state_shape_ =
      ShapeUtil::MakeTupleShape({induction_variable_shape_, data_shape_});
  Shape nested_tuple_shape_ =
      ShapeUtil::MakeTupleShape({data_shape_, data_shape_});
  Shape nested_loop_state_shape_ = ShapeUtil::MakeTupleShape(
      {induction_variable_shape_, nested_tuple_shape_});
  Shape condition_result_shape_ = ShapeUtil::MakeShape(PRED, {});
};

// Tests while body computation with independent tuple elements:
//
//   While.Body({in0, in1})
//     out0 = Add(in0, 1)
//     out1 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
//     Tuple(out0, out1)
//
// CopyInsertion pass should not generate any copies.
//
TEST_F(WhileCopyInsertionTest, IndependentTupleElements) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body =
      module_->AddEmbeddedComputation(BuildIndependentBodyComputation());
  auto while_hlo = BuildWhileInstruction(condition, body);

  InsertCopies(module_.get());

  // Body should have no copies as the adds can be done inplace.
  EXPECT_EQ(CountCopies(*body), 0);
  EXPECT_EQ(CountControlEdges(*module_), 0);

  // Both init indices need copies as they are constants.
  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

// Tests Copy Insertion when a while feeds another while
//                         PARAMETER
//                        |        |
//                        GTE(0)   GTE(1)
//                        |        |
//                        X = CreateTuple(GTE(0), GTE(1))
//                                 |
//                        WHILE(X) (root)
TEST_F(WhileCopyInsertionTest, WhileFeedingWhileThruParameterWithCopies) {
  const std::string& hlo_string = R"(
HloModule DependentTupleElements

%DependentTupleElements.Body (loop_state.1: (s32[], f32[8])) -> (s32[], f32[8]) {
  %loop_state.1 = (s32[], f32[8]{0}) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], f32[8]{0}) %loop_state.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = f32[8]{0} get-tuple-element((s32[], f32[8]{0}) %loop_state.1), index=1
  %convert = f32[] convert(s32[] %get-tuple-element.1)
  %broadcast = f32[8]{0} broadcast(f32[] %convert), dimensions={}
  %add.1 = f32[8]{0} add(f32[8]{0} %get-tuple-element.2, f32[8]{0} %broadcast)
  ROOT %tuple = (s32[], f32[8]{0}) tuple(s32[] %add, f32[8]{0} %add.1)
}

%DependentTupleElements.Condition (loop_state: (s32[], f32[8])) -> pred[] {
  %loop_state = (s32[], f32[8]{0}) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], f32[8]{0}) %loop_state), index=0
  %constant = s32[] constant(10)
  ROOT %compare = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %DependentTupleElements.While () -> (s32[], f32[8]) {
  %constant.2 = s32[] constant(0)
  %constant.3 = f32[8]{0} constant({0, 0, 0, 0, 0, 0, 0, 0})
  %tuple.1 = (s32[], f32[8]{0}) tuple(s32[] %constant.2, f32[8]{0} %constant.3)
  ROOT %while.1 = (s32[], f32[8]{0}) while((s32[], f32[8]{0}) %tuple.1), condition=%DependentTupleElements.Condition, body=%DependentTupleElements.Body
}
)";
  auto module_ = ParseAndReturnVerifiedModule(hlo_string).value();
  auto while_hlo = module_->entry_computation()->root_instruction();
  // module_ and while_hlo are the pre-existing module and hlo, the below
  // code generates a clone of the existing while and replaces that while
  // with itself. The body of the new while calls the previous while
  HloComputation* outer_while_condition =
      module_->AddEmbeddedComputation(while_hlo->while_condition()->Clone());
  HloComputation* outer_while_body =
      module_->AddEmbeddedComputation(while_hlo->while_body()->Clone());
  HloInstruction* outer_while =
      while_hlo->parent()->AddInstruction(HloInstruction::CreateWhile(
          while_hlo->shape(), outer_while_condition, outer_while_body,
          while_hlo->mutable_operand(0)));
  HloInstruction* outer_param = outer_while_body->parameter_instruction(0);
  std::vector<HloInstruction*> materialized_gtes;
  for (int i = 0; i < outer_param->shape().tuple_shapes().size(); ++i) {
    materialized_gtes.push_back(
        outer_while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
            outer_param->shape().tuple_shapes(i), outer_param, i)));
  }
  HloInstruction* dual_init = outer_while_body->AddInstruction(
      HloInstruction::CreateTuple(materialized_gtes));
  HloInstruction* dual_while =
      outer_while_body->AddInstruction(HloInstruction::CreateWhile(
          while_hlo->shape(), while_hlo->while_condition(),
          while_hlo->while_body(), dual_init));
  TF_CHECK_OK(outer_while_body->ReplaceInstruction(
      outer_while_body->root_instruction(), dual_while));
  TF_CHECK_OK(while_hlo->parent()->ReplaceInstruction(while_hlo, outer_while));
  InsertCopies(module_.get());
}

// Tests Copy Insertion when a while feeds another while
//                         PARAMETER
//                        |        |
//                         \      /
//                           WHILE(PARAMETER) (root)
TEST_F(WhileCopyInsertionTest, WhileFeedingWhileThruParameterNoCopies) {
  const std::string& hlo_string = R"(
HloModule DependentTupleElements

%DependentTupleElements.Body (loop_state.1: (s32[], f32[8])) -> (s32[], f32[8]) {
  %loop_state.1 = (s32[], f32[8]{0}) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], f32[8]{0}) %loop_state.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = f32[8]{0} get-tuple-element((s32[], f32[8]{0}) %loop_state.1), index=1
  %convert = f32[] convert(s32[] %get-tuple-element.1)
  %broadcast = f32[8]{0} broadcast(f32[] %convert), dimensions={}
  %add.1 = f32[8]{0} add(f32[8]{0} %get-tuple-element.2, f32[8]{0} %broadcast)
  ROOT %tuple = (s32[], f32[8]{0}) tuple(s32[] %add, f32[8]{0} %add.1)
}

%DependentTupleElements.Condition (loop_state: (s32[], f32[8])) -> pred[] {
  %loop_state = (s32[], f32[8]{0}) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], f32[8]{0}) %loop_state), index=0
  %constant = s32[] constant(10)
  ROOT %compare = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %DependentTupleElements.While () -> (s32[], f32[8]) {
  %constant.2 = s32[] constant(0)
  %constant.3 = f32[8]{0} constant({0, 0, 0, 0, 0, 0, 0, 0})
  %tuple.1 = (s32[], f32[8]{0}) tuple(s32[] %constant.2, f32[8]{0} %constant.3)
  ROOT %while.1 = (s32[], f32[8]{0}) while((s32[], f32[8]{0}) %tuple.1), condition=%DependentTupleElements.Condition, body=%DependentTupleElements.Body
}
)";
  auto module_ = ParseAndReturnVerifiedModule(hlo_string).value();
  auto while_hlo = module_->entry_computation()->root_instruction();
  // module_ and while_hlo are the pre-existing module and hlo, the below
  // code generates a clone of the existing while and replaces that while
  // with itself. The body of the new while calls the previous while
  HloComputation* outer_while_condition =
      module_->AddEmbeddedComputation(while_hlo->while_condition()->Clone());
  HloComputation* outer_while_body =
      module_->AddEmbeddedComputation(while_hlo->while_body()->Clone());
  HloInstruction* outer_while =
      while_hlo->parent()->AddInstruction(HloInstruction::CreateWhile(
          while_hlo->shape(), outer_while_condition, outer_while_body,
          while_hlo->mutable_operand(0)));
  HloInstruction* outer_param = outer_while_body->parameter_instruction(0);
  HloInstruction* dual_while =
      outer_while_body->AddInstruction(HloInstruction::CreateWhile(
          while_hlo->shape(), while_hlo->while_condition(),
          while_hlo->while_body(), outer_param));
  TF_CHECK_OK(outer_while_body->ReplaceInstruction(
      outer_while_body->root_instruction(), dual_while));
  TF_CHECK_OK(while_hlo->parent()->ReplaceInstruction(while_hlo, outer_while));
  InsertCopies(module_.get());
}

// Tests Copy Insertion when a while feeds another while
//                         PARAMETER
//                        |        |
//                         \      /
//                           WHILE(PARAMETER) (root)
TEST_F(WhileCopyInsertionTest, WhileFeedingWhileThruParameterBig) {
  const std::string& hlo_string = R"(
HloModule DependentTupleElements

%DependentTupleElements.Body (loop_state.1: (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0})) -> (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) {
  %loop_state.1 = (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) %loop_state.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = f32[8]{0} get-tuple-element((s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) %loop_state.1), index=1
  %convert = f32[] convert(s32[] %get-tuple-element.1)
  %broadcast = f32[8]{0} broadcast(f32[] %convert), dimensions={}
  %add.1 = f32[8]{0} add(f32[8]{0} %get-tuple-element.2, f32[8]{0} %broadcast)
  ROOT %tuple = (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) tuple(s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1, s32[] %add, f32[8]{0} %add.1)
}

%DependentTupleElements.Condition (loop_state: (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0})) -> pred[] {
  %loop_state = (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) %loop_state), index=0
  %constant = s32[] constant(10)
  ROOT %compare = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %DependentTupleElements.While () -> (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) {
  %constant.2 = s32[] constant(0)
  %constant.3 = f32[8]{0} constant({0, 0, 0, 0, 0, 0, 0, 0})
  %tuple.1 = (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) tuple(s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3, s32[] %constant.2, f32[8]{0} %constant.3)
  ROOT %while.1 = (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) while( (s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}, s32[], f32[8]{0}) %tuple.1), condition=%DependentTupleElements.Condition, body=%DependentTupleElements.Body
}
)";
  auto module_ = ParseAndReturnVerifiedModule(hlo_string).value();
  auto while_hlo = module_->entry_computation()->root_instruction();
  // module_ and while_hlo are the pre-existing module and hlo, the below
  // code generates a clone of the existing while and replaces that while
  // with itself. The body of the new while calls the previous while
  HloComputation* outer_while_condition =
      module_->AddEmbeddedComputation(while_hlo->while_condition()->Clone());
  HloComputation* outer_while_body =
      module_->AddEmbeddedComputation(while_hlo->while_body()->Clone());
  HloInstruction* outer_while =
      while_hlo->parent()->AddInstruction(HloInstruction::CreateWhile(
          while_hlo->shape(), outer_while_condition, outer_while_body,
          while_hlo->mutable_operand(0)));
  HloInstruction* outer_param = outer_while_body->parameter_instruction(0);
  std::vector<HloInstruction*> materialized_gtes;
  for (int i = 0; i < outer_param->shape().tuple_shapes().size(); ++i) {
    materialized_gtes.push_back(
        outer_while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
            outer_param->shape().tuple_shapes(i), outer_param, i)));
  }
  HloInstruction* dual_init = outer_while_body->AddInstruction(
      HloInstruction::CreateTuple(materialized_gtes));
  HloInstruction* dual_while =
      outer_while_body->AddInstruction(HloInstruction::CreateWhile(
          while_hlo->shape(), while_hlo->while_condition(),
          while_hlo->while_body(), dual_init));
  TF_CHECK_OK(outer_while_body->ReplaceInstruction(
      outer_while_body->root_instruction(), dual_while));
  TF_CHECK_OK(while_hlo->parent()->ReplaceInstruction(while_hlo, outer_while));
  InsertCopies(module_.get());
}

// Tests while body computation with dependent tuple elements:
//
//   While.Body({in0, in1})
//     out0 = Add(in0, 1)
//     out1 = Add(BCast(in0), in1)
//     Tuple(out0, out1)
//
// CopyInsertion pass should convert the root instruction to:
//
//     Tuple(Copy(out0), out1)
//
TEST_F(WhileCopyInsertionTest, DependentTupleElements) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body = module_->AddEmbeddedComputation(BuildDependentBodyComputation());
  auto while_hlo = BuildWhileInstruction(condition, body);

  InsertCopies(module_.get());

  EXPECT_EQ(CountCopies(*body), 1);
  EXPECT_EQ(CountControlEdges(*body), 0);

  EXPECT_THAT(
      body->root_instruction(),
      op::Tuple(op::Add(), op::Add(op::GetTupleElement(), op::Broadcast())));

  auto add = body->root_instruction()->operand(0);
  auto bcast = body->root_instruction()->operand(1)->operand(1);
  ASSERT_EQ(add->opcode(), HloOpcode::kAdd);
  ASSERT_EQ(bcast->opcode(), HloOpcode::kBroadcast);

  EXPECT_THAT(while_hlo->while_body()->root_instruction(),
              op::Tuple(op::Add(op::Copy(), op::Constant()),
                        op::Add(op::GetTupleElement(),
                                op::Broadcast(op::Convert(op::Copy())))));

  // Both init indices need copies as they are constants.
  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

// Tests while body computation with read-only tuple element 0:
//
//                         PARAMETER
//                         /       \
//                      GTE(0)     GTE(1)
//                        |  \      |
//                        |   BCAST |
//                        |      \  |
//                        |       ADD
//                        |        |
//                         \      /
//                           TUPLE (root)
//
// CopyInsertion pass should not generate any copies for the while body.
TEST_F(WhileCopyInsertionTest, DependentTupleElements_OneReadOnly) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  BuildWhileInstruction(condition, body);

  InsertCopies(module_.get());

  // No copies or control edges should be inserted. The body is legal as is.
  EXPECT_EQ(CountCopies(*body), 0);
  EXPECT_EQ(CountControlEdges(*body), 0);
}

// Same as above, but with two while loops, sharing entry parameters.
TEST_F(WhileCopyInsertionTest,
       DependentTupleElements_OneReadOnly_TwoLoops_EntryParams) {
  auto condition1 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto condition2 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body1 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  auto body2 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());

  auto builder = HloComputation::Builder(TestName() + ".While");
  auto iter_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, induction_variable_shape_, "iter"));
  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "data"));
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({iter_param, data_param}));

  auto while_hlo1 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition1, body1, loop_init));
  auto while_hlo2 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition2, body2, loop_init));

  // Add a couple elements from each of the while so both whiles are live.
  auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo1, 0));
  auto gte2 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo2->shape(), {0}), while_hlo2, 0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(gte1->shape(), HloOpcode::kAdd, gte1, gte2));

  auto entry = module_->AddEntryComputation(builder.Build());

  InsertCopies(module_.get());

  // Neither body should have any copies or control edges in them.
  EXPECT_EQ(CountCopies(*body1), 0);
  EXPECT_EQ(CountCopies(*body2), 0);
  EXPECT_EQ(CountControlEdges(*body1), 0);
  EXPECT_EQ(CountControlEdges(*body2), 0);

  // Only two copies should be necessary. Each of the whiles should have
  // a copy of tuple element 1 (init value is a parameter, and the element is
  // not non-read-only) so each of the while bodies gets its own buffer to write
  // element 1 into.
  EXPECT_EQ(CountCopies(*entry), 2);

  EXPECT_EQ(while_hlo1->operand(0)->operand(1)->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(while_hlo2->operand(0)->operand(1)->opcode(), HloOpcode::kCopy);

  // The two copies of element 1 should be different.
  EXPECT_NE(while_hlo1->operand(0)->operand(1),
            while_hlo2->operand(0)->operand(1));
}

// Same as above, but with two while loops, sharing non-parameters.
TEST_F(WhileCopyInsertionTest,
       DependentTupleElements_OneReadOnly_TwoLoops_NonParams) {
  auto condition1 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto condition2 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body1 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  auto body2 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());

  auto builder = HloComputation::Builder(TestName() + ".While");
  auto iter_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, induction_variable_shape_, "iter"));
  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "data"));
  // Add dummy ops to ensure loop_init elements aren't entry parameters.
  Shape f32_scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto convert = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_scalar_shape, iter_param));
  auto iter_value = builder.AddInstruction(
      HloInstruction::CreateUnary(convert->shape(), HloOpcode::kExp, convert));
  auto convert2 = builder.AddInstruction(
      HloInstruction::CreateConvert(induction_variable_shape_, iter_value));
  auto data_value = builder.AddInstruction(HloInstruction::CreateUnary(
      data_param->shape(), HloOpcode::kExp, data_param));
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({convert2, data_value}));

  auto while_hlo1 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition1, body1, loop_init));
  auto while_hlo2 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition2, body2, loop_init));

  // Add a couple elements from each of the while so both whiles are not dead.
  auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo1, 0));
  auto gte2 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo2->shape(), {0}), while_hlo2, 0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(gte1->shape(), HloOpcode::kAdd, gte1, gte2));
  auto entry = module_->AddEntryComputation(builder.Build());

  InsertCopies(module_.get());

  // Ideally only one copy should be necessary. One of the whiles should
  // have a copy of tuple element 1 (the non-read-only element) so each of the
  // while bodies gets its own buffer to write element 1 into. However, the
  // analysis isn't perfect and adds an additional copy of element 0.
  EXPECT_EQ(CountCopies(*entry), 2);

  EXPECT_THAT(while_hlo1->operand(0),
              op::Tuple(op::Convert(op::Exp()), op::Copy(op::Exp())));
  EXPECT_THAT(while_hlo2->operand(0),
              op::Tuple(op::Convert(op::Exp()), op::Copy(op::Exp())));
}

// Tests while body computation with nested tuple elements:
//
//                            |
//                    GTE(loop_state, 1)
//                       /          \
// GTE(GTE(loop_state, 1), 0)     GTE(GTE(loop_state, 1), 1)
//           |                              |
//          Add                           Reverse
//           |                              |
//
// CopyInsertion pass will conceptually generate the following, but with the
// actual GTE and Tuple instructions optimized away:
//
//                    Tuple  // old root
//                   /     \
//                  /       \
//                GTE(0)   GTE(1)
//                  |       /  \
//                  |      /    \
//                  |    GTE(0) GTE(1)
//                  |       |    |
//                  |       |   Copy
//                  |       |    |
//                   \      |   /
//                    \    Tuple  // "inner" tuple.
//                     \    /
//                      \  /
//                     Tuple  // new root
//
TEST_F(WhileCopyInsertionTest, NestedTupleElements) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(nested_loop_state_shape_));
  auto body = module_->AddEmbeddedComputation(BuildNestedBodyComputation());
  BuildWhileInstruction(condition, body, true);

  //  HloInstruction* old_root = body->root_instruction();
  InsertCopies(module_.get());

  // The only copy necessary is for the kReverse as it cannot be done
  // in-place (instruction can share buffer with operand). The other elements of
  // the loop state are kAdd instructions which can be done in-place.
  EXPECT_EQ(CountCopies(*body), 1);

  // Each element of the init needs a copy as all are constants.
  EXPECT_EQ(CountCopies(*module_), 4);

  // Either the kReverse itself must be copied or the operand of the kReverse
  // must be copied.
  if (body->root_instruction()->operand(1)->operand(1)->opcode() ==
      HloOpcode::kCopy) {
    EXPECT_THAT(
        body->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Copy(op::Reverse()))));
  } else {
    EXPECT_THAT(
        body->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Reverse(op::Copy()))));
  }
}

// Tests while init instruction which points-to a constant.
//
//     init = Tuple(Constant(S32, {}), Constant(F32, {8}))
//
// CopyInsertion pass should add copies for both constants.
//
TEST_F(WhileCopyInsertionTest, InitPointsToConstant) {
  auto while_hlo = BuildWhileInstruction_InitPointsToConstant();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 0);
  EXPECT_EQ(CountCopies(*module_), 2);

  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

// Tests while init instruction which points-to a parameter.
//
//     init = Tuple(Constant(S32, {}), Parameter(F32, {8}))
//
// CopyInsertion pass should add copies for both the constant and parameter.
//
TEST_F(WhileCopyInsertionTest, InitPointsToParameter) {
  auto while_hlo = BuildWhileInstruction_InitPointsToParameter();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 0);
  EXPECT_EQ(CountCopies(*module_), 2);

  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Parameter())));
}

// Tests while init instruction which has a non-distinct points-to set.
//
//     init = Tuple(Constant(S32, {}), Tuple({vec_one, vec_one}))
//
// CopyInsertion pass will conceptually generate the following, but with some of
// the actual GTE and Tuple instructions optimized away:
//
//                    Tuple  // old init
//                   /     \
//                  /       \
//                GTE(0)   GTE(1)
//                  |       /  \
//                  |      /    \
//                  |    GTE(0) GTE(1)
//                  |       |    |
//                Copy   Copy   Copy
//                  |       |    |
//                   \      |   /
//                    \    Tuple
//                     \    /
//                      \  /
//                     Tuple  // new init
//
TEST_F(WhileCopyInsertionTest, InitPointsToNonDistinct) {
  auto while_hlo = BuildWhileInstruction_InitPointsToNonDistinct();

  InsertCopies(module_.get());

  // The entry computation requires two copies to resolve the non-distinctness
  // of two init elements and the constant passed in as one of the init
  // elements. Either element can be copied for the distinctness issue.
  EXPECT_EQ(CountCopies(*module_->entry_computation()), 2);
  if (while_hlo->operand(0)->operand(1)->operand(0)->opcode() ==
      HloOpcode::kCopy) {
    EXPECT_THAT(
        while_hlo->operand(0),
        op::Tuple(op::Copy(op::Constant()),
                  op::Tuple(op::Copy(op::Broadcast()), op::Broadcast())));
  } else {
    EXPECT_THAT(
        while_hlo->operand(0),
        op::Tuple(op::Copy(op::Constant()),
                  op::Tuple(op::Broadcast(), op::Copy(op::Broadcast()))));
  }

  // The body requires one copy because the buffer set is not distinct: the
  // result of one of the adds is written into two elements of the output of the
  // loop body. Either element might be copied.
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 1);
  if (while_hlo->while_body()
          ->root_instruction()
          ->operand(1)
          ->operand(0)
          ->opcode() == HloOpcode::kCopy) {
    EXPECT_THAT(
        while_hlo->while_body()->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Copy(op::Add()), op::Add())));
  } else {
    EXPECT_THAT(
        while_hlo->while_body()->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Copy(op::Add()))));
  }
}

// Tests while init instruction buffer which interferes with while result
// buffer.
//
//     init_data = Broadcast(...)
//     add_unrelated = Add(init_data) // takes a reference to cause interference
//     init = Tuple(Constant(S32, {}), init_data))
//
// CopyInsertion pass should copy both operands.
//
TEST_F(WhileCopyInsertionTest, InitPointsToInterfering) {
  auto while_hlo = BuildWhileInstruction_InitPointsToInterfering();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*module_), 2);
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 0);

  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Broadcast())));
}

// Tests while init instruction buffer which has a non-distinct points-to set:
//
//     init = Tuple(Parameter(S32, {}), Parameter(F32, {8},
//                  Parameter(F32, {8})))
//
// where the second and third parameters are identical *and* the tuple shared
// by another while instruction.
//
// Verifies that the resulting point-to set is distinct in the resulting Tuple
// (non-identical Copies). In other words, verifies that copy sharing does not
// insert identical copies to the resulting tuple.
TEST_F(WhileCopyInsertionTest, InitPointsToNonDistinctUsedByTwoWhileLoops) {
  // Loop body that outputs tuple comprises two elements dependent on the init
  // tuple.
  const Shape& loop_state_shape = ShapeUtil::MakeTupleShape(
      {induction_variable_shape_, data_shape_, data_shape_});

  auto condition1 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape));
  auto condition2 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape));
  auto body1 =
      module_->AddEmbeddedComputation(BuildDependentBodyComputation2());
  auto body2 =
      module_->AddEmbeddedComputation(BuildDependentBodyComputation2());

  auto builder = HloComputation::Builder(TestName() + ".While");

  auto iter_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, induction_variable_shape_, "iter"));
  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "data"));

  // Loop init tuple contains two identical parameter buffers.
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({iter_param, data_param, data_param}));

  // Two while loops share the same loop init tuple.
  auto while_hlo1 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape, condition1, body1, loop_init));
  auto while_hlo2 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape, condition2, body2, loop_init));

  // Add add instruction so neither while is dead.
  auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo1, 0));
  auto gte2 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo2, 0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(gte1->shape(), HloOpcode::kAdd, gte1, gte2));

  module_->AddEntryComputation(builder.Build());

  InsertCopies(module_.get());

  // None of the bodies should have copies or control flow edges.
  EXPECT_EQ(CountCopies(*body1), 0);
  EXPECT_EQ(CountCopies(*body2), 0);

  // The loop bodies pass through elements 1 and 2 in the init tuple, so ideally
  // these should not need to be copied before either while. However, copy
  // insertion is not able to reason about the transparency of elements through
  // while bodies in all circumstances so extra copies are added (b/xxx).
  EXPECT_EQ(CountCopies(*module_->entry_computation()), 2);

  EXPECT_THAT(while_hlo1->operand(0),
              op::Tuple(op::Copy(), op::Parameter(), op::Parameter()));
  EXPECT_THAT(while_hlo2->operand(0),
              op::Tuple(op::Copy(), op::Parameter(), op::Parameter()));
}

TEST_F(CopyInsertionTest, SwizzlingWhile) {
  // Test a while instruction with a body which permutes its tuple parameter
  // elements.
  auto module = CreateNewVerifiedModule();
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Body simply interchanges the two tuple elements in the loop state.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_1, body_element_0}));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  cond_builder.AddInstruction(HloInstruction::CreateUnary(
      cond_constant->shape(), HloOpcode::kNot, cond_constant));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition, body, tuple));
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 6);

  // The loop state elements should be copied at the parameter and at the root
  // with a control edge in between (see DeepCopyAndAddControlEdges). This is
  // technically one more copy than is strictly necessary, but in order to have
  // only three copies the copies of different loop state elements must be
  // ordered with a control edge.
  EXPECT_EQ(CountCopies(*body), 4);
  EXPECT_EQ(CountControlEdges(*body), 2);

  EXPECT_THAT(body->root_instruction(),
              op::Tuple(op::Copy(op::Copy()), op::Copy(op::Copy())));

  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  EXPECT_THAT(xla_while->operand(0), op::Tuple(op::Copy(), op::Copy()));
}

TEST_F(CopyInsertionTest, CrossingParameters) {
  // Test a case where two parameters' dataflow cross with each other while
  // input and output are aliased with same index:
  //
  //  (p0 ,  p1)
  //   | \   /|
  //   |  \ / |
  // alias X  alias
  //   |  / \ |
  //   | /   \|
  //  (p1  ,  p0)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte1, gte0}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1}));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 4);
}

TEST_F(CopyInsertionTest, ParametersAliasing) {
  // Test a case where two parameters' dataflow don't interfere with each other
  // while aliased.
  //
  //  (p0 ,  p1)
  //   |      |
  //   |      |
  // alias   alias
  //   |      |
  //   |      |
  //  (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1}));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, ParameterWithNoAliasing) {
  // Test a case where no parameter is aliased with result. In this case, copy
  // should be added
  //
  //  (p0 ,  p1)
  //   |      |
  //   |      |
  //   |      |
  //   |      |
  //   |      |
  //  (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  module->AddEntryComputation(builder.Build());
  InsertCopies(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(op::GetTupleElement(param, 0)),
                        op::Copy(op::GetTupleElement(param, 1))));

  EXPECT_EQ(CountCopies(*module), 2);
}

TEST_F(CopyInsertionTest, ParameterWithPartialAliasing) {
  // Test a case where one parameter is aliased with result while another one
  // isn't.
  //
  //  (p0 ,  p1)
  //   |      |
  //   |      |
  // alias    |
  //   |      |
  //   |      |
  //  (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  InsertCopies(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(param, 0),
                        op::Copy(op::GetTupleElement(param, 1))));

  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, ParameterAndParallelOpsWithPartialAliasing) {
  // Test a case where one parameter is aliased with result while another one
  // isn't.
  //
  //   +-- (p0 ,  p1)
  //   |    |      |
  //   |    |      |
  // alias Negate  Negate
  //   |    |      |
  //   |    |      |
  //   +-- (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));

  auto negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte0));

  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte1));
  builder.AddInstruction(HloInstruction::CreateTuple({negate0, negate1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, ParameterAndOpsWithPartialAliasing) {
  // Test a case where one parameter is aliased with result while another one
  // isn't.
  //
  //   +-- (p0 ,  p1)
  //   |    |      |
  //   |    |      |
  // alias Negate  Negate
  //   |    |      |
  //   |    Add----+
  //   |    |      |
  //   +-- (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));

  auto negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte0));

  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte1));

  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, negate0, negate1));
  builder.AddInstruction(HloInstruction::CreateTuple({add, negate1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, SwizzlingWhileWithOneOp) {
  // Test a while instruction with a body which permutes its tuple parameter
  // elements and applies one operation to one of the elements. The addition of
  // the operation (instruction) on the element makes the live range of the
  // respective input and output elements different than if the instruction were
  // not there (as in the SwizzlingWhile test above).
  auto module = CreateNewVerifiedModule();
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Body interchanges the two tuple elements in the loop state and negates one
  // of them.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, body_element_1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({negate, body_element_0}));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  cond_builder.AddInstruction(HloInstruction::CreateUnary(
      cond_constant->shape(), HloOpcode::kNot, cond_constant));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition, body, tuple));
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 6);

  // The loop state elements should be copied at the parameter and at the root
  // with a control edge in between (see DeepCopyAndAddControlEdges).
  EXPECT_EQ(CountCopies(*body), 4);
  EXPECT_EQ(CountControlEdges(*body), 2);

  EXPECT_THAT(
      body->root_instruction(),
      op::Tuple(op::Copy(op::Negate(op::Copy())), op::Copy(op::Copy())));

  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  EXPECT_THAT(xla_while->operand(0), op::Tuple(op::Copy(), op::Copy()));
}

TEST_F(CopyInsertionTest, SwizzlingWhileSharedInput) {
  // Test a while instruction with a body which permutes it's tuple parameter
  // elements similar to SwizzlinWhile above. However, in this test the input to
  // the while body is a single constant (both loop state elements are the same
  // constant). This means no copies are necessary because both loop state
  // elements are the same so interchanging them is a no-op.
  auto module = CreateNewVerifiedModule();
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Body simply interchanges the two tuple elements in the loop state.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_1, body_element_0}));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  cond_builder.AddInstruction(HloInstruction::CreateUnary(
      cond_constant->shape(), HloOpcode::kNot, cond_constant));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition, body, tuple));
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 2);
  EXPECT_EQ(CountCopies(*body), 0);

  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(), op::Copy()));
}

TEST_F(CopyInsertionTest, SequentialWhiles) {
  // Construct a computation with a series of sequential while instructions
  // containing four loop state elements:
  //
  //   element 0 is passed to each while directly from an entry parameter.
  //
  //   element 1 is passed transparently in series through all the while bodies.
  //
  //   element 2 is negated in each while body. (in-place possible)
  //
  //   element 3 is reversed in each while body. (in-place not possible)
  //
  const Shape element_shape = ShapeUtil::MakeShape(F32, {42});
  const Shape loop_state_shape = ShapeUtil::MakeTupleShape(
      {element_shape, element_shape, element_shape, element_shape});

  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, element_shape, "param_0"));
  auto param_1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, element_shape, "param_1"));
  auto param_2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, element_shape, "param_2"));
  auto param_3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, element_shape, "param_3"));

  // The number of sequential kWhile instructions.
  const int kNumWhiles = 3;

  HloInstruction* prev_element_1 = param_1;
  HloInstruction* prev_element_2 = param_2;
  HloInstruction* prev_element_3 = param_3;

  // Vector containing all of the while instructions.
  std::vector<const HloInstruction*> whiles;
  for (int i = 0; i < kNumWhiles; ++i) {
    auto body_builder = HloComputation::Builder("body");
    auto body_param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "param"));
    auto body_element_0 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 0));
    auto body_element_1 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 1));
    auto body_element_2 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 2));
    auto body_element_3 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 3));
    auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
        element_shape, HloOpcode::kNegate, body_element_2));
    auto reverse = body_builder.AddInstruction(
        HloInstruction::CreateReverse(element_shape, body_element_3, {0}));
    body_builder.AddInstruction(HloInstruction::CreateTuple(
        {body_element_0, body_element_1, negate, reverse}));
    HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

    auto cond_builder = HloComputation::Builder("condition");
    cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "param"));
    auto cond_constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    cond_builder.AddInstruction(HloInstruction::CreateUnary(
        cond_constant->shape(), HloOpcode::kNot, cond_constant));
    HloComputation* condition =
        module->AddEmbeddedComputation(cond_builder.Build());

    auto while_init = builder.AddInstruction(HloInstruction::CreateTuple(
        {param_0, prev_element_1, prev_element_2, prev_element_3}));

    auto xla_while = builder.AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape, condition, body, while_init));
    whiles.push_back(xla_while);
    if (i != kNumWhiles - 1) {
      prev_element_1 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, xla_while, 1));
      prev_element_2 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, xla_while, 2));
      prev_element_3 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, xla_while, 3));
    }
  }

  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  // Each while body has one copy. And each loop state element is copied once in
  // the entry computation.
  EXPECT_EQ(CountCopies(*module), 4 + kNumWhiles);

  // Each while body should have exactly one copy for element three which is an
  // op (kReverse) which cannot be done in place.
  for (const HloInstruction* xla_while : whiles) {
    EXPECT_EQ(CountCopies(*xla_while->while_body()), 1);
  }

  EXPECT_THAT(whiles[0]->operand(0), op::Tuple(op::Parameter(), op::Parameter(),
                                               op::Copy(), op::Copy()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(), op::Copy(), op::GetTupleElement(),
                        op::GetTupleElement()));
}

TEST_F(CopyInsertionTest, WhileBodyWithConstantRoot) {
  // Test a while body and condition which are each simply a constant (root of
  // computation is a constant). The body constant should be copied.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param_0"));

  auto body_builder = HloComputation::Builder("body");
  body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0)));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape_, condition, body, param_0));

  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(xla_while->operand(0), op::Copy(op::Parameter()));
  EXPECT_THAT(body->root_instruction(), op::Copy(op::Constant()));
  EXPECT_THAT(condition->root_instruction(), op::Constant());
}

TEST_F(CopyInsertionTest, TokensShouldNotBeCopied) {
  std::string module_string = R"(
HloModule TokensShouldNotBeCopied

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %TokensShouldNotBeCopied () -> s32[] {
  %one = s32[] constant(1)
  %negative_one = s32[] negate(%one)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %negative_one, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  InsertCopies(module.get());

  // There should be no copies added because tokens should not be copied.
  EXPECT_EQ(CountCopies(*module), 0);
}

std::unique_ptr<HloComputation> MakeTrivialCondition(const Shape& shape) {
  auto builder = HloComputation::Builder("trivial_condition");
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "loop_state"));
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNot, constant));
  return builder.Build();
}

std::unique_ptr<HloComputation> MakeBenchmarkWhileBody() {
  auto builder = HloComputation::Builder("benchmark_loop_body");
  const Shape element_shape = ShapeUtil::MakeShape(F32, {42});
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({element_shape, element_shape, element_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
  HloInstruction* element_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(element_shape, param, 0));
  HloInstruction* element_1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(element_shape, param, 1));
  HloInstruction* element_2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(element_shape, param, 2));

  HloInstruction* rev_1 = builder.AddInstruction(
      HloInstruction::CreateReverse(element_shape, element_1, {0}));
  HloInstruction* add_1_2 = builder.AddInstruction(HloInstruction::CreateBinary(
      element_shape, HloOpcode::kAdd, element_1, element_2));

  builder.AddInstruction(
      HloInstruction::CreateTuple({element_0, rev_1, add_1_2}));
  return builder.Build();
}

void BM_SequentialWhiles(::testing::benchmark::State& state) {
  const int num_whiles = state.range(0);

  // This benchmark constructs a chain of sequential while instructions.
  // Timer starts automatically at the first iteration of this loop
  // and ends after the last one.
  for (auto s : state) {
    state.PauseTiming();
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    HloModule module("BM_SequentialWhiles", config);

    auto builder = HloComputation::Builder("BM_SequentialWhiles");
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {42}), "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {42}), "y"));
    HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(F32, {42}), "z"));
    HloInstruction* init =
        builder.AddInstruction(HloInstruction::CreateTuple({x, y, z}));

    HloInstruction* prev_loop_state = init;
    for (int w = 0; w < num_whiles; ++w) {
      HloComputation* condition =
          module.AddEmbeddedComputation(MakeTrivialCondition(init->shape()));
      HloComputation* body =
          module.AddEmbeddedComputation(MakeBenchmarkWhileBody());
      prev_loop_state = builder.AddInstruction(HloInstruction::CreateWhile(
          init->shape(), condition, body, prev_loop_state));
    }
    module.AddEntryComputation(builder.Build());

    CopyInsertion copy_insertion;

    state.ResumeTiming();
    ASSERT_IS_OK(copy_insertion.Run(&module).status());
    state.PauseTiming();

    // The entry computation should have three copies, and each body has one.
    ASSERT_EQ(CountCopies(module), 3 + num_whiles);
    state.ResumeTiming();
  }
}

void BM_ParallelWhiles(::testing::benchmark::State& state) {
  const int num_whiles = state.range(0);

  // This benchmark constructs a fan-out of parallel while instructions.
  for (auto s : state) {
    state.PauseTiming();
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    HloModule module("BM_SequentialWhiles", config);

    auto builder = HloComputation::Builder("BM_ParallelWhiles");
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {42}), "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {42}), "y"));
    HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(F32, {42}), "z"));
    HloInstruction* init =
        builder.AddInstruction(HloInstruction::CreateTuple({x, y, z}));

    HloInstruction* sum = nullptr;
    for (int w = 0; w < num_whiles; ++w) {
      HloComputation* condition =
          module.AddEmbeddedComputation(MakeTrivialCondition(init->shape()));
      HloComputation* body =
          module.AddEmbeddedComputation(MakeBenchmarkWhileBody());

      HloInstruction* xla_while = builder.AddInstruction(
          HloInstruction::CreateWhile(init->shape(), condition, body, init));

      if (sum == nullptr) {
        sum = builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(x->shape(), xla_while, 0));
      } else {
        HloInstruction* element_0 = builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(x->shape(), xla_while, 0));
        sum = builder.AddInstruction(HloInstruction::CreateBinary(
            x->shape(), HloOpcode::kAdd, sum, element_0));
      }
    }
    module.AddEntryComputation(builder.Build());

    CopyInsertion copy_insertion;

    state.ResumeTiming();
    ASSERT_IS_OK(copy_insertion.Run(&module).status());
    state.PauseTiming();

    // Each body receives of copy of two of the parameters (the corresponding
    // elements in the body are modified), and there is one copy in each body.
    ASSERT_EQ(CountCopies(module), 3 * num_whiles);
  }
}

std::unique_ptr<HloComputation> MakeBenchmarkWhileBody(
    const int num_tuple_inputs) {
  auto builder = HloComputation::Builder("benchmark_loop_body");
  const Shape element_shape = ShapeUtil::MakeShape(F32, {});
  std::vector<Shape> input_shape(num_tuple_inputs, element_shape);
  const Shape loop_state_shape = ShapeUtil::MakeTupleShape(input_shape);
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
  std::vector<HloInstruction*> gte_nodes(num_tuple_inputs);
  for (int i = 0; i < num_tuple_inputs; ++i) {
    gte_nodes[i] = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, param, i));
  }
  builder.AddInstruction(HloInstruction::CreateTuple(gte_nodes));
  return builder.Build();
}

void BM_ManyElementTuple(::testing::benchmark::State& state) {
  const int num_tuple_inputs = state.range(0);
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  CopyInsertion copy_insertion;
  const Shape element_shape = ShapeUtil::MakeShape(F32, {});
  std::vector<HloInstruction*> tuple_params(num_tuple_inputs);
  for (auto s : state) {
    state.PauseTiming();
    auto builder = HloComputation::Builder("BM_ParallelWhiles");
    HloModule module("BM_ManyElementTuple", config);
    for (int j = 0; j < num_tuple_inputs; ++j) {
      tuple_params[j] = builder.AddInstruction(
          HloInstruction::CreateParameter(j, element_shape, ""));
    }
    HloInstruction* init =
        builder.AddInstruction(HloInstruction::CreateTuple(tuple_params));
    HloComputation* condition =
        module.AddEmbeddedComputation(MakeTrivialCondition(init->shape()));
    HloComputation* body =
        module.AddEmbeddedComputation(MakeBenchmarkWhileBody(num_tuple_inputs));
    HloInstruction* xla_while = builder.AddInstruction(
        HloInstruction::CreateWhile(init->shape(), condition, body, init));
    builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(F32, {}), xla_while, 0));
    module.AddEntryComputation(builder.Build());
    state.ResumeTiming();
    ASSERT_IS_OK(copy_insertion.Run(&module).status());
  }
}

BENCHMARK(BM_SequentialWhiles)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);
BENCHMARK(BM_ParallelWhiles)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);
BENCHMARK(BM_ManyElementTuple)->Arg(1024)->Arg(12288);

TEST_F(CopyInsertionTest, SimpleControlFlowTest) {
  const std::string& hlo_string = R"(
HloModule TestModule

if-body.v5 {
  constant.3 = s32[] constant(-1)
  p.1 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.18 = (s32[], s32[], s32[]) get-tuple-element(p.1), index=1
  get-tuple-element.65 = s32[] get-tuple-element(get-tuple-element.18), index=0
  get-tuple-element.66 = s32[] get-tuple-element(get-tuple-element.18), index=1
  add.3 = s32[] add(get-tuple-element.65, get-tuple-element.66)
  tuple.33 = (s32[]) tuple(add.3)
  ROOT tuple.34 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(constant.3, get-tuple-element.18, tuple.33)
}

if-condition.v4 {
  p.2 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.67 = s32[] get-tuple-element(p.2), index=0
  constant.4 = s32[] constant(0)
  ROOT equal-to = pred[] compare(get-tuple-element.67, constant.4), direction=EQ
}

_functionalize_body_1__.v28 {
  arg_tuple.4 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.68 = s32[] get-tuple-element(arg_tuple.4), index=0
  constant.7 = s32[] constant(1)
  add.4 = s32[] add(get-tuple-element.68, constant.7)
  get-tuple-element.69 = s32[] get-tuple-element(arg_tuple.4), index=1
  get-tuple-element.70 = s32[] get-tuple-element(arg_tuple.4), index=2
  less-than-or-equal-to = pred[] compare(get-tuple-element.69, get-tuple-element.70), direction=LE
  constant.8 = s32[] constant(0)
  select = s32[] select(less-than-or-equal-to, constant.8, constant.7)
  get-tuple-element.71 = s32[] get-tuple-element(arg_tuple.4), index=3
  tuple.35 = (s32[], s32[], s32[]) tuple(get-tuple-element.69, get-tuple-element.71, get-tuple-element.70)
  tuple.36 = (s32[]) tuple(constant.8)
  tuple.37 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(select, tuple.35, tuple.36)
  while = (s32[], (s32[], s32[], s32[]), (s32[])) while(tuple.37), condition=if-condition.v4, body=if-body.v5
  get-tuple-element.72 = (s32[]) get-tuple-element(while), index=2
  get-tuple-element.73 = s32[] get-tuple-element(get-tuple-element.72), index=0
  ROOT tuple.38 = (s32[], s32[], s32[], s32[]) tuple(add.4, get-tuple-element.69, get-tuple-element.70, get-tuple-element.73)
}

cond_wrapper.v3.1 {
  inputs.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.75 = s32[] get-tuple-element(inputs.1), index=0
  constant.11 = s32[] constant(7)
  ROOT less-than.2 = pred[] compare(get-tuple-element.75, constant.11), direction=LT
}

_functionalize_body_2__.v25 {
  arg_tuple.5 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.76 = s32[] get-tuple-element(arg_tuple.5), index=0
  get-tuple-element.77 = s32[] get-tuple-element(arg_tuple.5), index=2
  get-tuple-element.78 = s32[] get-tuple-element(arg_tuple.5), index=3
  get-tuple-element.79 = s32[] get-tuple-element(arg_tuple.5), index=4
  tuple.39 = (s32[], s32[], s32[], s32[]) tuple(get-tuple-element.76, get-tuple-element.77, get-tuple-element.78, get-tuple-element.79)
  while.2 = (s32[], s32[], s32[], s32[]) while(tuple.39), condition=cond_wrapper.v3.1, body=_functionalize_body_1__.v28
  get-tuple-element.80 = s32[] get-tuple-element(while.2), index=0
  get-tuple-element.81 = s32[] get-tuple-element(arg_tuple.5), index=1
  constant.12 = s32[] constant(1)
  add.5 = s32[] add(get-tuple-element.81, constant.12)
  get-tuple-element.82 = s32[] get-tuple-element(while.2), index=3
  ROOT tuple.40 = (s32[], s32[], s32[], s32[], s32[]) tuple(get-tuple-element.80, add.5, get-tuple-element.77, get-tuple-element.78, get-tuple-element.82)
}

cond_wrapper.v3.2 {
  inputs.2 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.83 = s32[] get-tuple-element(inputs.2), index=1
  constant.13 = s32[] constant(5)
  ROOT less-than.3 = pred[] compare(get-tuple-element.83, constant.13), direction=LT
}

ENTRY TestComputation {
  arg_tuple.6 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  ROOT while.3 = (s32[], s32[], s32[], s32[], s32[]) while(arg_tuple.6), condition=cond_wrapper.v3.2, body=_functionalize_body_2__.v25
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  InsertCopies(module.get());
}

TEST_F(CopyInsertionTest, ControlFlowTest) {
  const std::string& hlo_string = R"(
HloModule TestModule

if-body.v5 {
  constant.3 = s32[] constant(-1)
  p.1 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.18 = (s32[], s32[], s32[]) get-tuple-element(p.1), index=1
  get-tuple-element.65 = s32[] get-tuple-element(get-tuple-element.18), index=0
  get-tuple-element.66 = s32[] get-tuple-element(get-tuple-element.18), index=1
  add.3 = s32[] add(get-tuple-element.65, get-tuple-element.66)
  tuple.33 = (s32[]) tuple(add.3)
  ROOT tuple.34 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(constant.3, get-tuple-element.18, tuple.33)
}

if-condition.v4 {
  p.2 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.67 = s32[] get-tuple-element(p.2), index=0
  constant.4 = s32[] constant(0)
  ROOT equal-to = pred[] compare(get-tuple-element.67, constant.4), direction=EQ
}

if-body.v5.1 {
  constant.5 = s32[] constant(-1)
  p.3 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.68 = (s32[], s32[], s32[]) get-tuple-element(p.3), index=1
  get-tuple-element.70 = s32[] get-tuple-element(get-tuple-element.68), index=2
  multiply.1 = s32[] multiply(get-tuple-element.70, get-tuple-element.70)
  tuple.35 = (s32[]) tuple(multiply.1)
  ROOT tuple.36 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(constant.5, get-tuple-element.68, tuple.35)
}

if-condition.v4.1 {
  p.4 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.71 = s32[] get-tuple-element(p.4), index=0
  constant.6 = s32[] constant(1)
  ROOT equal-to.1 = pred[] compare(get-tuple-element.71, constant.6), direction=EQ
}

_functionalize_body_1__.v28 {
  arg_tuple.4 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.72 = s32[] get-tuple-element(arg_tuple.4), index=0
  constant.7 = s32[] constant(1)
  add.4 = s32[] add(get-tuple-element.72, constant.7)
  get-tuple-element.73 = s32[] get-tuple-element(arg_tuple.4), index=1
  get-tuple-element.74 = s32[] get-tuple-element(arg_tuple.4), index=2
  less-than-or-equal-to = pred[] compare(get-tuple-element.73, get-tuple-element.74), direction=LE
  constant.8 = s32[] constant(0)
  select = s32[] select(less-than-or-equal-to, constant.8, constant.7)
  get-tuple-element.75 = s32[] get-tuple-element(arg_tuple.4), index=3
  tuple.37 = (s32[], s32[], s32[]) tuple(get-tuple-element.73, get-tuple-element.75, get-tuple-element.74)
  tuple.38 = (s32[]) tuple(constant.8)
  tuple.39 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(select, tuple.37, tuple.38)
  while = (s32[], (s32[], s32[], s32[]), (s32[])) while(tuple.39), condition=if-condition.v4, body=if-body.v5
  while.1 = (s32[], (s32[], s32[], s32[]), (s32[])) while(while), condition=if-condition.v4.1, body=if-body.v5.1
  get-tuple-element.76 = (s32[]) get-tuple-element(while.1), index=2
  get-tuple-element.77 = s32[] get-tuple-element(get-tuple-element.76), index=0
  ROOT tuple.40 = (s32[], s32[], s32[], s32[]) tuple(add.4, get-tuple-element.73, get-tuple-element.74, get-tuple-element.77)
}

cond_wrapper.v3.1 {
  inputs.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.78 = s32[] get-tuple-element(inputs.1), index=0
  constant.11 = s32[] constant(7)
  ROOT less-than.2 = pred[] compare(get-tuple-element.78, constant.11), direction=LT
}

_functionalize_body_2__.v25 {
  arg_tuple.5 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.79 = s32[] get-tuple-element(arg_tuple.5), index=0
  get-tuple-element.80 = s32[] get-tuple-element(arg_tuple.5), index=2
  get-tuple-element.81 = s32[] get-tuple-element(arg_tuple.5), index=3
  get-tuple-element.82 = s32[] get-tuple-element(arg_tuple.5), index=4
  tuple.41 = (s32[], s32[], s32[], s32[]) tuple(get-tuple-element.79, get-tuple-element.80, get-tuple-element.81, get-tuple-element.82)
  while.2 = (s32[], s32[], s32[], s32[]) while(tuple.41), condition=cond_wrapper.v3.1, body=_functionalize_body_1__.v28
  get-tuple-element.83 = s32[] get-tuple-element(while.2), index=0
  get-tuple-element.84 = s32[] get-tuple-element(arg_tuple.5), index=1
  constant.12 = s32[] constant(1)
  add.5 = s32[] add(get-tuple-element.84, constant.12)
  get-tuple-element.85 = s32[] get-tuple-element(while.2), index=3
  ROOT tuple.42 = (s32[], s32[], s32[], s32[], s32[]) tuple(get-tuple-element.83, add.5, get-tuple-element.80, get-tuple-element.81, get-tuple-element.85)
}

cond_wrapper.v3.2 {
  inputs.2 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.86 = s32[] get-tuple-element(inputs.2), index=1
  constant.13 = s32[] constant(5)
  ROOT less-than.3 = pred[] compare(get-tuple-element.86, constant.13), direction=LT
}

ENTRY TestComputation {
  arg_tuple.6 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  ROOT while.3 = (s32[], s32[], s32[], s32[], s32[]) while(arg_tuple.6), condition=cond_wrapper.v3.2, body=_functionalize_body_2__.v25
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  InsertCopies(module.get());
}

TEST_F(CopyInsertionTest, NestedWhiles) {
  // Verify that only no unnecessary copies remain after copy insertion for
  // trivial nested whiles (b/112472605).
  const std::string& hlo_string = R"(
HloModule TestModule

cond.inner {
  ROOT param.cond.inner = pred[] parameter(0)
}

body.inner {
  param.body.inner = pred[] parameter(0)
  ROOT not = pred[] not(param.body.inner)
}

cond.outer {
  ROOT param.cond.outer = pred[] parameter(0)
}

body.outer {
  param.cond.outer = pred[] parameter(0)
  ROOT while = pred[] while(param.cond.outer), condition=cond.inner, body=body.inner
}

ENTRY TestComputation {
  entry_param = pred[] parameter(0)
  ROOT while = pred[] while(entry_param), condition=cond.outer, body=body.outer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());

  // There should only be a single copy inserted, and it's in the entry
  // computation.
  EXPECT_EQ(CountCopies(*module), 1);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::While(op::Copy(op::Parameter())));
}

TEST_F(CopyInsertionTest, NestedWhilesWithParamRoot) {
  // Test that when the root of a computation is before other side-effecting
  // operation (e.g. when the while body computation parameter is the root), we
  // introduce an interference edge and copy at the level of this outer loop
  // body and not one level out.
  const std::string& hlo_string = R"(
HloModule TestModule

cond.inner {
  ROOT param.cond.inner = pred[] parameter(0)
}

body.inner {
  param.body.inner = pred[] parameter(0)
  ROOT not = pred[] not(param.body.inner)
}

cond.outer {
  ROOT param.cond.outer = pred[] parameter(0)
}

body.outer {
  ROOT param.cond.outer = pred[] parameter(0)
  while = pred[] while(param.cond.outer), condition=cond.inner, body=body.inner
  after-all = token[] after-all()
  outfeed = token[] outfeed(while, after-all)
}

ENTRY TestComputation {
  entry_param = pred[] parameter(0)
  while = pred[] while(entry_param), condition=cond.outer, body=body.outer
  ROOT not = pred[] not(while)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());

  // There should only be a single copy inserted, and it's in the outer while
  // loop body.
  EXPECT_EQ(CountCopies(*module), 1);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Not(op::While(op::Parameter())));
  HloInstruction* outfeed = FindInstruction(module.get(), "outfeed");
  EXPECT_THAT(outfeed, op::Outfeed(op::While(op::Copy(op::Parameter(0))),
                                   op::AfterAll()));
}

TEST_F(CopyInsertionTest, NestedWhilesWithParamRoot2) {
  // Test that when the root of a computation is before other side-effecting
  // operation (e.g. when the while body computation parameter is the root), we
  // introduce an interference edge and copy at the level of this outer loop
  // body and not one level out.
  const std::string& hlo_string = R"(
HloModule TestModule

cond.inner {
  param.cond.inner = (pred[], pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param.cond.inner), index=0
}

body.inner {
  param.body.inner = (pred[], pred[]) parameter(0)
  gte.0 = pred[] get-tuple-element(param.body.inner), index=0
  gte.1 = pred[] get-tuple-element(param.body.inner), index=1
  and = pred[] and(gte.0, gte.1)
  not = pred[] not(gte.1)
  ROOT root = (pred[], pred[]) tuple(and, not)
}

cond.outer {
  param.cond.outer = (pred[], pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param.cond.outer), index=0
}

body.outer {
  param.body.outer = (pred[], pred[]) parameter(0)
  gte.0 = pred[] get-tuple-element(param.body.outer), index=0
  gte.1 = pred[] get-tuple-element(param.body.outer), index=1
  while.inner = (pred[], pred[]) while(param.body.outer), condition=cond.inner, body=body.inner
  gte.2 = pred[] get-tuple-element(while.inner), index=0
  after-all = token[] after-all()
  outfeed = token[] outfeed(gte.2, after-all)
  ROOT root = (pred[], pred[]) tuple(gte.0, gte.1)
}

ENTRY TestComputation {
  entry_param.1 = pred[] parameter(0)
  entry_param.2 = pred[] parameter(1)
  tuple = (pred[], pred[]) tuple(entry_param.1, entry_param.2)
  while.outer = (pred[], pred[]) while(tuple), condition=cond.outer, body=body.outer
  gte = pred[] get-tuple-element(while.outer), index=0
  ROOT not = pred[] not(gte)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());

  HloInstruction* while_inner = FindInstruction(module.get(), "while.inner");
  EXPECT_THAT(
      while_inner,
      op::While(op::Tuple(op::Copy(op::GetTupleElement(op::Parameter(0))),
                          op::Copy(op::GetTupleElement(op::Parameter(0))))));
}

TEST_F(CopyInsertionTest, NestedWhileAndConditional2) {
  const std::string& hlo_string = R"(
HloModule TestModule

on_true
 {
  v1 = f32[2] parameter(0)
  v2 = f32[2] add(v1,v1)
  ROOT t1 = (f32[2], f32[2]) tuple(v1,v2)
}

on_false
 {
  v1 = f32[2] parameter(0)
  v2 = f32[2] multiply(v1,v1)
  ROOT t2 = (f32[2], f32[2]) tuple(v1,v2)
}

cond.outer {
  param.1 = (pred[], f32[2], f32[2]) parameter(0)
  ROOT param.cond.outer = pred[] get-tuple-element(param.1), index=0
}

body.outer {
  param.1 = (pred[], f32[2], f32[2]) parameter(0)
  pred.1 = pred[] get-tuple-element(param.1), index=0
  arg_tuple.11 = f32[2] get-tuple-element(param.1), index=1
  if = (f32[2], f32[2]) conditional(pred.1, arg_tuple.11, arg_tuple.11), true_computation=on_true, false_computation=on_false
  e1 = f32[2] get-tuple-element(if), index=0
  e2 = f32[2] get-tuple-element(if), index=1
  ROOT res = (pred[], f32[2], f32[2]) tuple(pred.1,e1, e2)
}

ENTRY TestComputation {
  entry_param.1 = pred[] parameter(0)
  float_param = f32[2] parameter(1)
  entry_param = (pred[], f32[2], f32[2]) tuple(entry_param.1, float_param, float_param)
  ROOT while = (pred[], f32[2], f32[2]) while(entry_param), condition=cond.outer, body=body.outer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());

  // An extra copy must be kept inside the loop due to uses in the conditional.
  EXPECT_EQ(CountCopies(*module), 3);
}

TEST_F(CopyInsertionTest, NestedWhileAndConditional) {
  const std::string& hlo_string = R"(
HloModule TestModule

on_true
 {
  v1 = f32[2] parameter(0)
  ROOT v2 = f32[2] add(v1,v1)
}

on_false
 {
  v1 = f32[2] parameter(0)
  ROOT v2 = f32[2] multiply(v1,v1)
}

cond.outer {
  param.1 = (pred[], f32[2]) parameter(0)
  ROOT param.cond.outer = pred[] get-tuple-element(param.1), index=0
}

body.outer {
  param.1 = (pred[], f32[2]) parameter(0)
  pred.1 = pred[] get-tuple-element(param.1), index=0
  arg_tuple.11 = f32[2] get-tuple-element(param.1), index=1
  if = f32[2] conditional(pred.1, arg_tuple.11, arg_tuple.11), true_computation=on_true, false_computation=on_false
  ROOT res = (pred[], f32[2]) tuple(pred.1,if)
}

ENTRY TestComputation {
  entry_param.1 = pred[] parameter(0)
  float_param = f32[2] parameter(1)
  entry_param = (pred[], f32[2]) tuple(entry_param.1, float_param)
  ROOT while = (pred[], f32[2]) while(entry_param), condition=cond.outer, body=body.outer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  VLOG(2) << module->ToString() << "\n";

  // There should only be two copies inserted, and in the entry and exit of the
  // computation.
  EXPECT_EQ(CountCopies(*module), 2);
}

TEST_F(CopyInsertionTest, FixpointComputationRequired) {
  const std::string& hlo_string = R"(
HloModule Module

fused_computation {
  param0 = f32[3,3,96,1] parameter(0)
  param1 = f32[] parameter(1)
  broadcast = f32[3,3,96,1] broadcast(f32[] param1), dimensions={}
  ROOT %add.0 = f32[3,3,96,1] add(f32[3,3,96,1] param0, f32[3,3,96,1] broadcast)
}

ENTRY entry_computation {
  arg0 = f32[3,3,96,1] parameter(0)
  arg1 = f32[] parameter(1)
  fusion = f32[3,3,96,1] fusion(f32[3,3,96,1] arg0, f32[] arg1),
    kind=kLoop, calls=fused_computation
  negate = f32[] negate(f32[] arg1)
  ROOT tuple = (f32[3,3,96,1], f32[3,3,96,1], f32[], f32[]) tuple(
    f32[3,3,96,1] fusion,
    f32[3,3,96,1] arg0,
    f32[] negate,
    f32[] arg1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Set up the aliasing manually which normally would be set by
  // alias_passthrough_params pass.
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1},
      /*param_number=*/0,
      /*param_index=*/{}));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{3},
      /*param_number=*/1,
      /*param_index=*/{}));

  InsertCopies(module.get());

  // There should be no copies inserted.
  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, NoAliasCheckViolation) {
  const std::string& hlo_string = R"(
HloModule cluster

ENTRY Entry {
  %arg = f32[8,28,28,1] parameter(0)
  %bitcast.2 = f32[8,1,28,28] bitcast(f32[8,28,28,1] %arg)
  ROOT %tuple.1 = (f32[8,1,28,28], f32[8,28,28,1]) tuple(f32[8,1,28,28] %bitcast.2, f32[8,28,28,1] %arg)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1},
      /*param_number=*/0,
      /*param_index=*/{}));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, DynamicUpdateSliceNoCopy) {
  absl::string_view hlo_string = R"(
HloModule Module

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate = f32[1280,1,128] negate(param)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(negate, broadcast.6, constant.3, constant.3, constant.3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, FusedDynamicUpdateSliceNoCopy) {
  absl::string_view hlo_string = R"(
HloModule Module

fused_computation {
  param0 = f32[1280,1,128] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param0, broadcast.6, constant.3, constant.3, constant.3)
}

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate = f32[1280,1,128] negate(param)
  ROOT fusion = f32[1280,1,128] fusion(negate), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, DynamicUpdateSliceCopy) {
  absl::string_view hlo_string = R"(
HloModule Module

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate = f32[1280,1,128] negate(param)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  add = f32[1280,1,128] add(negate, negate)
  dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(negate, broadcast.6, constant.3, constant.3, constant.3)
  ROOT tuple = (f32[1280,1,128], f32[1280,1,128]) tuple(add, dynamic-update-slice.5)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, DynamicUpdateSliceParameterShareCopy) {
  absl::string_view hlo_string = R"(
HloModule Module

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param, broadcast.6, constant.3, constant.3, constant.3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, FusedDynamicUpdateSliceCopy) {
  absl::string_view hlo_string = R"(
HloModule Module

fused_computation {
  param0 = f32[1280,1,128] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param0, broadcast.6, constant.3, constant.3, constant.3)
}

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate = f32[1280,1,128] negate(param)
  add = f32[1280,1,128] add(negate, negate)
  fusion = f32[1280,1,128] fusion(negate), kind=kLoop, calls=fused_computation
  ROOT tuple = (f32[1280,1,128], f32[1280,1,128]) tuple(negate, fusion)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, ChainDynamicUpdateSliceCopy) {
  absl::string_view hlo_string = R"(
HloModule Module

ENTRY main {
  state = (s32[], f32[1280,1,128]{2,1,0}) parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128]{2,1,0} broadcast(constant.1), dimensions={}
  get-tuple-element.4 = f32[1280,1,128]{2,1,0} get-tuple-element(state), index=1
  get-tuple-element.3 = s32[] get-tuple-element(state), index=0
  constant.2 = s32[] constant(128)
  add.5 = s32[] add(get-tuple-element.3, constant.2)
  constant.3 = s32[] constant(0)
  dynamic-update-slice.5 = f32[1280,1,128]{2,1,0} dynamic-update-slice(get-tuple-element.4, broadcast.6, constant.3, constant.3, constant.3)
  dynamic-update-slice.9 = f32[1280,1,128]{2,1,0} dynamic-update-slice(dynamic-update-slice.5, broadcast.6, constant.3, constant.3, constant.3)
  ROOT tuple.85 = (s32[], f32[1280,1,128]{2,1,0}) tuple(add.5, dynamic-update-slice.9)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, FusedDynamicUpdateSliceCopy2) {
  absl::string_view hlo_string = R"(
HloModule Module

fused_computation.1 {
  param0 = f32[1280,1,128] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param0, broadcast.6, constant.3, constant.3, constant.3)
}

fused_computation.2 {
  param0 = f32[1280,1,128] parameter(0)
  param1 = f32[1280,1,128] parameter(1)
  slice = f32[128,1,128] slice(param1), slice={[0:128], [0:1], [0:128]}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param0, slice, constant.3, constant.3, constant.3)
}

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate = f32[1280,1,128] negate(param)
  add = f32[1280,1,128] add(negate, negate)
  fusion1 = f32[1280,1,128] fusion(negate), kind=kLoop, calls=fused_computation.1
  ROOT fusion2 = f32[1280,1,128] fusion(fusion1, negate), kind=kLoop, calls=fused_computation.2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, MultiOutputFusedDynamicUpdateSliceCopy) {
  // Tests multi-output fusion with two DUS outputs, requiring two copies.
  absl::string_view hlo_string = R"(
HloModule Module

fused_computation {
  param0 = f32[1280,1,128] parameter(0)
  param1 = f32[1280,1,128] parameter(1)
  param2 = f32[1280,1,128] parameter(2)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  add.1 = f32[1280,1,128] add(param0, param0)
  dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param1, broadcast.6, constant.3, constant.3, constant.3)
  dynamic-update-slice.6 = f32[1280,1,128] dynamic-update-slice(param2, broadcast.6, constant.3, constant.3, constant.3)
  ROOT tuple.1 = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) tuple(add.1, dynamic-update-slice.5, dynamic-update-slice.6)
}

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate0 = f32[1280,1,128] negate(param)
  negate1 = f32[1280,1,128] negate(param)
  negate2 = f32[1280,1,128] negate(param)
  fusion = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) fusion(negate0, negate1, negate2), kind=kLoop, calls=fused_computation
  gte0 = f32[1280,1,128] get-tuple-element(fusion), index=0
  gte1 = f32[1280,1,128] get-tuple-element(fusion), index=1
  gte2 = f32[1280,1,128] get-tuple-element(fusion), index=2
  add0 = f32[1280,1,128] add(negate0, gte0)
  add1 = f32[1280,1,128] add(negate1, gte1)
  add2 = f32[1280,1,128] add(negate2, gte2)
  ROOT tuple = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) tuple(add0, add1, add2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 2);
}

TEST_F(CopyInsertionTest, MultiOutputFusedDynamicUpdateSliceNoCopy) {
  // Same as above, but negate1 is not used beyond fusion, so it only needs one
  // copy for negate0.
  absl::string_view hlo_string = R"(
HloModule Module

fused_computation {
  param0 = f32[1280,1,128] parameter(0)
  param1 = f32[1280,1,128] parameter(1)
  param2 = f32[1280,1,128] parameter(2)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  add.1 = f32[1280,1,128] add(param0, param0)
  dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param1, broadcast.6, constant.3, constant.3, constant.3)
  dynamic-update-slice.6 = f32[1280,1,128] dynamic-update-slice(param2, broadcast.6, constant.3, constant.3, constant.3)
  ROOT tuple.1 = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) tuple(add.1, dynamic-update-slice.5, dynamic-update-slice.6)
}

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate0 = f32[1280,1,128] negate(param)
  negate1 = f32[1280,1,128] negate(param)
  negate2 = f32[1280,1,128] negate(param)
  fusion = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) fusion(negate0, negate1, negate2), kind=kLoop, calls=fused_computation
  gte0 = f32[1280,1,128] get-tuple-element(fusion), index=0
  gte1 = f32[1280,1,128] get-tuple-element(fusion), index=1
  gte2 = f32[1280,1,128] get-tuple-element(fusion), index=2
  add0 = f32[1280,1,128] add(negate0, gte0)
  add1 = f32[1280,1,128] add(gte1, gte1)
  add2 = f32[1280,1,128] add(negate2, gte2)
  ROOT tuple = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) tuple(add0, add1, add2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, ScatterSharedOperand) {
  // If an in-place op has an additional operand that has the same value as the
  // in-place buffer, a copy needs to be inserted on one of these values only.
  absl::string_view hlo_string = R"(
HloModule Module

update_s32 {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

fused_computation {
  iota.1 = s32[73729]{0} iota(), iota_dimension=0
  ROOT indices.1 = s32[73729]{0} reverse(iota.1), dimensions={0}
}

ENTRY main {
  iota.2 = s32[73729]{0} iota(), iota_dimension=0
  fusion = s32[73729]{0} fusion(), kind=kLoop, calls=fused_computation
  ROOT scatter = s32[73729]{0} scatter(iota.2, fusion, iota.2), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=update_s32
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Scatter(op::Copy(op::Iota()), op::Fusion(), op::Iota()));
}

TEST_F(CopyInsertionTest, HorizontalLoopFusionNoCopy) {
  const std::string& hlo_string = R"(
    HloModule test

    fused_computation {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      p2 = f32[10,10] parameter(2)
      p3 = f32[10,10] parameter(3)
      add0 = f32[10, 20] add(p0, p1)
      sub0 = f32[10, 10] subtract(p2, p3)
      reshape0 = f32[200] reshape(add0)
      reshape1 = f32[100] reshape(sub0)
      concat0 = f32[300] concatenate(reshape0, reshape1), dimensions={0}
      slice0 = f32[200] slice(concat0), slice={[0:200]}
      slice1 = f32[100] slice(concat0), slice={[200:300]}
      ROOT tuple = (f32[200], f32[100]) tuple(slice0, slice1)
    }

    ENTRY test {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      p2 = f32[10,10] parameter(2)
      p3 = f32[10,10] parameter(3)
      fusion = (f32[200], f32[100]) fusion(p0, p1, p2, p3), kind=kInput, calls=fused_computation
      gte0 = f32[200] get-tuple-element(fusion), index=0
      gte1 = f32[100] get-tuple-element(fusion), index=1
      bitcast0 = f32[10,20] bitcast(gte0)
      bitcast1 = f32[10,10] bitcast(gte1)
      ROOT tuple = (f32[10,20], f32[10,10]) tuple(bitcast0, bitcast1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0},
      /*param_number=*/0,
      /*param_index=*/{}));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1},
      /*param_number=*/3,
      /*param_index=*/{}));

  InsertCopies(module.get());

  // There should be no copies inserted.
  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, NestedWhileAndConditional3) {
  const std::string& hlo_string = R"(
HloModule TestModule

on_true.1
 {
  ROOT v1 = f32[2] parameter(0)
}

on_false.1
 {
  v1 = f32[2] parameter(0)
  ROOT v2 = f32[2] multiply(v1,v1)
}

on_true
 {
  v1 = f32[2] parameter(0)
  v2 = f32[2] add(v1,v1)
  v3 = (f32[2],f32[2]) tuple(v1,v2)
  v4 = f32[2] get-tuple-element(v3), index=1
  v5 = f32[2] multiply(v4,v2)
   ROOT t1 = (f32[2], f32[2]) tuple(v5,v2)
}

on_false
 {
  v1 = f32[2] parameter(0)
  v2 = f32[2] multiply(v1,v1)
  pred.1 = pred[] constant(true)
  v4 = f32[2] conditional(pred.1, v1, v2), true_computation=on_true.1, false_computation=on_false.1
  v5 = f32[2] multiply(v4,v2)
  ROOT t2 = (f32[2], f32[2]) tuple(v2,v5)

}

cond.outer {
  param.1 = (pred[], f32[2], f32[2]) parameter(0)
  ROOT param.cond.outer = pred[] get-tuple-element(param.1), index=0
}

body.outer {
  param.1 = (pred[], f32[2], f32[2]) parameter(0)
  pred.1 = pred[] get-tuple-element(param.1), index=0
  arg_tuple.11 = f32[2] get-tuple-element(param.1), index=1
  if = (f32[2], f32[2]) conditional(pred.1, arg_tuple.11, arg_tuple.11), true_computation=on_true, false_computation=on_false
  e1 = f32[2] get-tuple-element(if), index=0
  e2 = f32[2] get-tuple-element(if), index=1
  ROOT res = (pred[], f32[2], f32[2]) tuple(pred.1,e1, e2)
}

ENTRY TestComputation {
  entry_param.1 = pred[] parameter(0)
  float_param = f32[2] parameter(1)
  entry_param = (pred[], f32[2], f32[2]) tuple(entry_param.1, float_param, float_param)
  ROOT while = (pred[], f32[2], f32[2]) while(entry_param), condition=cond.outer, body=body.outer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  // An extra copy must be kept inside the loop due to uses in the conditional
  EXPECT_EQ(CountCopies(*module), 4);
}

TEST_F(CopyInsertionTest, ConditionalBranchMustCopy1) {
  const std::string& hlo_string = R"(
HloModule TestModule

 branch_0_comp.5.clone {
 %parameter.0 = (s32[2]{0:T(128)}) parameter(0)
 %get-tuple-element = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.0), index=0
 %negate = s32[2]{0:T(128)} negate(s32[2]{0:T(128)} %get-tuple-element)
 %copy = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %negate)
 ROOT tuple.5 = (s32[2]{0:T(128)}) tuple(%copy)
 }

 branch_1_comp.12.clone {
  %parameter.4 = (s32[2]{0:T(128)}) parameter(0)
  %get-tuple-element.5 = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.4), index=0
  %copy.1 = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %get-tuple-element.5)
  ROOT tuple.6 = (s32[2]{0:T(128)}) tuple(%copy.1)
 }

ENTRY TestComputation {
  %parameter.1 = s32[]{:T(128)} parameter(0), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.2 = s32[2]{0:T(128)} parameter(1), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.3 = s32[2]{0:T(128)} parameter(2), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %tuple.1 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.3)
  %tuple.3 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.2)
  %conditional.18 = (s32[2]{0:T(128)}) conditional(s32[]{:T(128)} %parameter.1, (s32[2]{0:T(128)}) %tuple.1, (s32[2]{0:T(128)}) %tuple.3), branch_computations={%branch_0_comp.5.clone, %branch_1_comp.12.clone}, metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %gte.1 = s32[2]{0:T(128)} get-tuple-element(conditional.18), index=0
  ROOT tuple.4 = (s32[2]{0:T(128)},s32[2]{0:T(128)}) tuple(parameter.2, gte.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();
  // The copy.1 must be kept due to modification in the other branch.
  auto conditional18 = FindInstruction(module.get(), "conditional.18");
  CHECK_NE(conditional18, nullptr);
  auto tuple6 = conditional18->branch_computation(1)->root_instruction();
  CHECK_EQ(tuple6->opcode(), HloOpcode::kTuple);
  auto copy1 = tuple6->operand(0);
  CHECK_EQ(copy1->opcode(), HloOpcode::kCopy);
}

TEST_F(CopyInsertionTest, ConditionalBranchMustCopy2) {
  const std::string& hlo_string = R"(
HloModule TestModule

 branch_0_comp.5.clone {
 %parameter.0 = (s32[2]{0:T(128)}) parameter(0)
 %get-tuple-element = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.0), index=0
 %negate = s32[2]{0:T(128)} negate(s32[2]{0:T(128)} %get-tuple-element)
 %copy = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %negate)
 ROOT tuple.5 = (s32[2]{0:T(128)}) tuple(%copy)
 }

 branch_1_comp.12.clone {
  %parameter.4 = (s32[2]{0:T(128)}) parameter(0)
  %get-tuple-element.5 = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.4), index=0
  %copy.1 = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %get-tuple-element.5)
  %constant.1 = s32[] constant(0)
  %broadcast.6 = s32[2] broadcast(constant.1), dimensions={}
  dynamic-update-slice.5 = s32[2]{0:T(128)} dynamic-update-slice(%copy.1, %broadcast.6, %constant.1)
  %add.1 = s32[2]{0:T(128)} add(dynamic-update-slice.5, %copy.1)
  ROOT tuple.6 = (s32[2]{0:T(128)}) tuple(%add.1)
 }

ENTRY TestComputation {
  %parameter.1 = s32[]{:T(128)} parameter(0), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.2 = s32[2]{0:T(128)} parameter(1), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.3 = s32[2]{0:T(128)} parameter(2), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %tuple.1 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.3)
  %tuple.3 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.2)
  %conditional.18 = (s32[2]{0:T(128)}) conditional(s32[]{:T(128)} %parameter.1, (s32[2]{0:T(128)}) %tuple.1, (s32[2]{0:T(128)}) %tuple.3), branch_computations={%branch_0_comp.5.clone, %branch_1_comp.12.clone}
  %gte.1 = s32[2]{0:T(128)} get-tuple-element(conditional.18), index=0
  ROOT tuple.4 = (s32[2]{0:T(128)},s32[2]{0:T(128)}) tuple(parameter.2, gte.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  // The copy.1 must be kept due to modification in the other branch.
  auto conditional18 = FindInstruction(module.get(), "conditional.18");
  CHECK_NE(conditional18, nullptr);
  auto tuple6 = conditional18->branch_computation(1)->root_instruction();
  CHECK_EQ(tuple6->opcode(), HloOpcode::kTuple);
  auto add1 = tuple6->operand(0);
  CHECK_EQ(add1->opcode(), HloOpcode::kAdd);
  auto dus = add1->operand(0);
  auto copy1 = dus->operand(0);
  CHECK_EQ(copy1->opcode(), HloOpcode::kCopy);
}

TEST_F(CopyInsertionTest, ConditionalBranchMustCopy3) {
  const std::string& hlo_string = R"(
HloModule primitive_computation_cond.19
%branch_0_comp.5.clone (parameter.0: (s32[2])) -> (s32[2]) {
  %parameter.0 = (s32[2]{0:T(128)}) parameter(0)
  %get-tuple-element = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.0), index=0
  %negate = s32[2]{0:T(128)} negate(s32[2]{0:T(128)} %get-tuple-element)
  %copy = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %negate)
  ROOT %tuple.5 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %copy)
}

%branch_1_comp.12.clone (parameter.4: (s32[2])) -> (s32[2]) {
  %parameter.4 = (s32[2]{0:T(128)}) parameter(0)
  %get-tuple-element.5 = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.4), index=0
  %copy.1 = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %get-tuple-element.5)
  ROOT %tuple.6 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %copy.1)
}

ENTRY %primitive_computation_cond.19 (parameter.1: s32[], parameter.2: s32[2], parameter.3: s32[2]) -> (s32[2]) {
  %parameter.1 = s32[]{:T(128)} parameter(0), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.3 = s32[2]{0:T(128)} parameter(2), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %tuple.1 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.3)
  %parameter.2 = s32[2]{0:T(128)} parameter(1), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %tuple.3 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.2)
  ROOT %conditional.18 = (s32[2]{0:T(128)}) conditional(s32[]{:T(128)} %parameter.1, (s32[2]{0:T(128)}) %tuple.1, (s32[2]{0:T(128)}) %tuple.3), branch_computations={%branch_0_comp.5.clone, %branch_1_comp.12.clone}, metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();
  // The copy.1 must be kept b/c aliasing of parameter and root is not allowed.
  auto conditional18 = FindInstruction(module.get(), "conditional.18");
  CHECK_NE(conditional18, nullptr);
  auto tuple6 = conditional18->branch_computation(1)->root_instruction();
  CHECK_EQ(tuple6->opcode(), HloOpcode::kTuple);
  auto copy1 = tuple6->operand(0);
  CHECK_EQ(copy1->opcode(), HloOpcode::kCopy);
}

TEST_F(CopyInsertionTest, ConditionalBranchDoNotCopy1) {
  const std::string& hlo_string = R"(
HloModule TestModule

 branch_0_comp.5.clone {
 %parameter.0 = (s32[2]{0:T(128)}) parameter(0)
 %get-tuple-element = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.0), index=0
 %negate = s32[2]{0:T(128)} negate(s32[2]{0:T(128)} %get-tuple-element)
 %copy = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %negate)
 ROOT tuple.5 = (s32[2]{0:T(128)}) tuple(%copy)
 }

 branch_1_comp.12.clone {
  %parameter.4 = (s32[2]{0:T(128)}) parameter(0)
  %get-tuple-element.5 = s32[2]{0:T(128)} get-tuple-element((s32[2]{0:T(128)}) %parameter.4), index=0
  %copy.1 = s32[2]{0:T(128)} copy(s32[2]{0:T(128)} %get-tuple-element.5)
  ROOT tuple.6 = (s32[2]{0:T(128)}) tuple(%copy.1)
 }

ENTRY TestComputation {
  %parameter.1 = s32[]{:T(128)} parameter(0), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.2 = s32[2]{0:T(128)} parameter(1), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %parameter.3 = s32[2]{0:T(128)} parameter(2), metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %tuple.1 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.3)
  %tuple.3 = (s32[2]{0:T(128)}) tuple(s32[2]{0:T(128)} %parameter.2)
  %conditional.18 = (s32[2]{0:T(128)}) conditional(s32[]{:T(128)} %parameter.1, (s32[2]{0:T(128)}) %tuple.1, (s32[2]{0:T(128)}) %tuple.3), branch_computations={%branch_0_comp.5.clone, %branch_1_comp.12.clone}, metadata={op_type="cond" op_name="cond[ linear=(False, False) ]"}
  %gte.1 = s32[2]{0:T(128)} get-tuple-element(conditional.18), index=0
  ROOT tuple.4 = (s32[2]{0:T(128)},s32[2]{0:T(128)}) tuple(gte.1, gte.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString() << "\n";

  // The copy.1 must be kept due to modification in the other branch.
  auto conditional18 = FindInstruction(module.get(), "conditional.18");
  CHECK_NE(conditional18, nullptr);
  auto tuple6 = conditional18->branch_computation(1)->root_instruction();
  CHECK_EQ(tuple6->opcode(), HloOpcode::kParameter);
}

TEST_F(CopyInsertionTest, ConditionalWithMultiOutputFusion) {
  const std::string& hlo_string = R"(
HloModule TestModule

branch_0 {
  param_0 = f64[] parameter(0)
  negate.2 = f64[] negate(f64[] param_0)
  ROOT tuple = (f64[], f64[]) tuple(f64[] negate.2, f64[] negate.2)
}

fused_computation {
  param_0.1 = f64[] parameter(0)
  abs.2 = f64[] abs(f64[] param_0.1)
  negate.1 = f64[] negate(f64[] param_0.1)
  ROOT %tuple.2 = (f64[], f64[]) tuple(f64[] negate.1, f64[] abs.2)
}

branch_1 {
  param_0.2 = f64[] parameter(0)
  ROOT fusion = (f64[], f64[]) fusion(f64[] param_0.2), kind=kLoop, calls=%fused_computation
}

ENTRY main {
  pred.0 = s32[] parameter(0)
  param_1 = f64[] parameter(1)
  param_2 = f64[] parameter(2)
  ROOT conditional.0 = (f64[], f64[]) conditional(s32[] pred.0, f64[] param_1, f64[] param_2), branch_computations={%branch_0, %branch_1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();

  // `branch_0` returns the same result of `negate.2` twice. Normally the result
  // would be put into one buffer and tuple would return two pointers to the
  // same buffer.
  // `branch_1` returns two results of multi-output fusion that should be put
  // into different buffers.
  // One copy is inserted in `branch_0` to ensure that result are put into two
  // different buffers.
  EXPECT_EQ(CountCopies(*module->GetComputationWithName("branch_0")), 1);

  EXPECT_EQ(CountCopies(*module->GetComputationWithName("branch_1")), 0);
  EXPECT_EQ(CountCopies(*module->GetComputationWithName("main")), 0);
}

TEST_F(CopyInsertionTest, ConditionalWithVariadicReduce) {
  const std::string& hlo_string = R"(
HloModule TestModule

branch_0 {
  empty_tuple.0 = () parameter(0)
  c_0 = f64[] constant(0)
  ROOT tuple.3 = (f64[], f64[]) tuple(c_0, c_0)
}

fused_computation {
  param_0.1 = f64[] parameter(0)
  abs.2 = f64[] abs(f64[] param_0.1)
  negate.1 = f64[] negate(f64[] param_0.1)
  ROOT %tuple.2 = (f64[], f64[]) tuple(f64[] negate.1, f64[] abs.2)
}

reduce_region {
  param_0.0 = f64[] parameter(0)
  param_2.0 = f64[] parameter(2)
  add.1.0 = f64[] add(param_0.0, param_2.0)
  param_1.0 = f64[] parameter(1)
  param_3.0 = f64[] parameter(3)
  multiply.1.0 = f64[] multiply(param_1.0, param_3.0)
  ROOT tuple.0.0 = (f64[], f64[]) tuple(add.1.0, multiply.1.0)
}

branch_1 {
  c_0 = f64[] constant(0)
  param_0.1 = f64[128]{0} parameter(0)
  ROOT reduce = (f64[], f64[]) reduce(param_0.1, param_0.1, c_0, c_0), dimensions={0}, to_apply=reduce_region
}

ENTRY main {
  pred.0 = s32[] parameter(0)
  empty_tuple = () tuple()
  param_2 = f64[128] parameter(1), sharding={replicated}
  ROOT conditional.0 = (f64[], f64[]) conditional(s32[] pred.0, () empty_tuple, f64[128] param_2), branch_computations={%branch_0, %branch_1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();

  // `branch_0` returns the same constant twice. Without copies it would return
  // pointers to read-only buffer for constant.
  // `branch_1` returns two results of a that should be put into different
  // buffers.
  // `conditional` needs to buffers for results, so the constant in `branch_0`
  // should be copied twice.
  EXPECT_EQ(CountCopies(*module->GetComputationWithName("branch_0")), 2);
  EXPECT_EQ(CountCopies(*module->GetComputationWithName("branch_1")), 0);
  EXPECT_EQ(CountCopies(*module->GetComputationWithName("main")), 0);
}

TEST_F(CopyInsertionTest, RootInstructionNotLast) {
  // This is a test for b/189219227. When the root instruction is scheduled not
  // as the last instruction, it still lives out. So, we make sure that the copy
  // after the root cannot be removed.
  const std::string& hlo_string = R"(
HloModule module, is_scheduled=true

body2 {
  p_body2 = (f32[2]{0}) parameter(0)
  p_body2.1 = f32[2]{0} get-tuple-element(p_body2), index=0
  add.3 = f32[2]{0} add(p_body2.1, p_body2.1)
  ROOT root2 = (f32[2]{0}) tuple(add.3)
}

condition2 {
  p_cond2 = (f32[2]{0}) parameter(0)
  ROOT result = pred[] constant(true)
}

body {
  p_body = (f32[2]{0}) parameter(0)
  p_body.1 = f32[2]{0} get-tuple-element(p_body), index=0
  ROOT root = (f32[2]{0}) tuple(p_body.1)
  copy = f32[2]{0} copy(p_body.1)
  tuple = (f32[2]{0}) tuple(copy)
  while.1 = (f32[2]{0}) while(tuple), condition=condition2, body=body2
}

condition {
  p_cond = (f32[2]{0}) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const0 = f32[2]{0} constant({1, 2})
  while_init = (f32[2]{0}) tuple(const0)
  ROOT while.0 = (f32[2]{0}) while(while_init), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.RemoveUnnecessaryCopies(module.get()));
  auto while_1 = FindInstruction(module.get(), "while.1");
  EXPECT_THAT(while_1, op::While(op::Tuple(op::Copy())));
}

TEST_F(CopyInsertionTest, InPlaceCollectivePermuteCopy) {
  absl::string_view hlo_string = R"(
HloModule hlo_runner_test_0.1
ENTRY hlo_runner_test_0.1 {
    replica_id = u32[] replica-id()
    broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] replica_id), dimensions={}
    constant.1 = u32[] constant(1000)
    broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    broadcast.2 = u32[4,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    constant.2 = s32[] constant(0)
    constant.3 = s32[] constant(1)
    tuple.input = (u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.0, u32[2,8,128]{2,1,0:T(2,128)} broadcast.0)
    tuple.output = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.1, u32[4,8,128]{2,1,0:T(2,128)} broadcast.2)
    tuple.2 = (s32[],s32[],s32[]) tuple(constant.2, constant.2, constant.2)
    tuple.3 = (s32[],s32[],s32[]) tuple(constant.3, constant.2, constant.2)
    tuple.4 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.3)
    constant.4 = s32[] constant(2)
    tuple.5 = (s32[],s32[],s32[]) tuple(constant.4, constant.2, constant.2)
    tuple.6 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.5)
    tuple.7 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.2)
    tuple.8 = (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple(((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.7)
    tuple.9 = (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple(((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.6)
    tuple.10 = (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple(((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.7)
    collective-permute.0 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute((u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple.input, (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple.output, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.8, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.9), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
    collective-permute.1 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute((u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple.input, (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple.output, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.8, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.10), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
    ROOT tuple = ((u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}), (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)})) tuple(collective-permute.0, collective-permute.1)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 4);
}

TEST_F(CopyInsertionTest, KeepCopyOfBroadcast) {
  absl::string_view hlo_string = R"(
HloModule Module

ENTRY main {
  param = f32[128,1,128] parameter(0)
  negate = f32[128,1,128] negate(param)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  broadcast.7 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  dynamic-update-slice.5 = f32[128,1,128] dynamic-update-slice(broadcast.6, broadcast.7, constant.3, constant.3, constant.3)
  add1 = f32[128,1,128] add(dynamic-update-slice.5, dynamic-update-slice.5)
  dynamic-update-slice.4 = f32[128,1,128] dynamic-update-slice(broadcast.6, broadcast.7, constant.3, constant.3, constant.3)
  add2 = f32[128,1,128] add(dynamic-update-slice.4, dynamic-update-slice.4)
  tuple = (f32[128,1,128], f32[128,1,128]) tuple(add1, add2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  EXPECT_EQ(CountCopies(*module), 2);
}

TEST_F(CopyInsertionTest, CustomCallAliasingCopyInsertedAliasedParam) {
  // The custom call specifies aliasing for an operand that is an input to the
  // computation, but it does not own that buffer so a precautionary copy
  // must be inserted.
  const char* const kModuleString = R"(
    HloModule xla_computation_f

    ENTRY xla_computation_f {
      parameter.1 = f32[2,3,4,5] parameter(0)
      parameter.2 = f32[2,3,4,5] parameter(1)
      ROOT custom-call = f32[2,3,4,5] custom-call(parameter.1, parameter.2), custom_call_target="dm_softmax", operand_layout_constraints={f32[2,3,4,5], f32[2,3,4,5]}, output_to_operand_aliasing={{}: (0, {})}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  InsertCopies(module.get());
  HloInstruction* custom_call = module->entry_computation()->root_instruction();
  EXPECT_THAT(custom_call->operand(0), op::Copy(op::Parameter(0)));
}

TEST_F(CopyInsertionTest, CustomCallAliasingCopyInsertedAliasedReuse) {
  // The custom call specifies aliasing for an operand that is later re-used
  // by a different instruction (add.2) A copy must be inserted so the correct
  // HloValue is passed to the add, and not the result of the aliased call.
  const char* const kModuleString = R"(
    HloModule xla_computation_f

    ENTRY xla_computation_f {
      parameter.1 = f32[2,3,4,5] parameter(0)
      parameter.2 = f32[2,3,4,5] parameter(1)
      add.1 = f32[2,3,4,5] add(parameter.1, parameter.2)
      custom-call = f32[2,3,4,5] custom-call(add.1, parameter.2), custom_call_target="dm_softmax", operand_layout_constraints={f32[2,3,4,5], f32[2,3,4,5]}, output_to_operand_aliasing={{}: (0, {})}
      ROOT add.2 = f32[2,3,4,5] add(custom-call, add.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  InsertCopies(module.get());
  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call");
  CHECK_NE(custom_call, nullptr);
  EXPECT_THAT(custom_call->operand(0), op::Copy(op::Add()));
}

TEST_F(CopyInsertionTest, CustomCallAliasingCopyRemoved) {
  // This custom call aliases an intermediate result, and the value is never
  // reused. There is no need for a copy.
  const char* const kModuleString = R"(
    HloModule xla_computation_f__1
    ENTRY xla_computation_f {
      parameter.1 = f32[2,3,4,5] parameter(0)
      parameter.2 = f32[2,3,4,5] parameter(1)
      add = f32[2,3,4,5] add(parameter.1, parameter.2)
      ROOT custom-call = f32[2,3,4,5] custom-call(add, parameter.2), custom_call_target="dm_softmax", operand_layout_constraints={f32[2,3,4,5], f32[2,3,4,5]}, output_to_operand_aliasing={{}: (0, {})}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  InsertCopies(module.get());
  HloInstruction* custom_call = module->entry_computation()->root_instruction();
  EXPECT_THAT(custom_call->operand(0), op::Add());
}

TEST_F(CopyInsertionTest, ReverseInConditional) {
  const char* const kModuleString = R"(
HloModule jit_f.0

%region_0.4 (Arg_.5: u8[300,451,3]) -> (u8[300,451,3]) {
  %Arg_.5 = u8[300,451,3]{1,0,2:T(8,128)(4,1)} parameter(0)
  ROOT %tuple = (u8[300,451,3]{1,0,2:T(8,128)(4,1)}) tuple(u8[300,451,3]{1,0,2:T(8,128)(4,1)} %Arg_.5)
}

%region_1.9 (Arg_.10: u8[300,451,3]) -> (u8[300,451,3]) {
  %Arg_.10 = u8[300,451,3]{1,0,2:T(8,128)(4,1)} parameter(0)
  %reverse = u8[300,451,3]{1,0,2:T(8,128)(4,1)} reverse(u8[300,451,3]{1,0,2:T(8,128)(4,1)} %Arg_.10), dimensions={0}
  ROOT %tuple.1 = (u8[300,451,3]{1,0,2:T(8,128)(4,1)}) tuple(u8[300,451,3]{1,0,2:T(8,128)(4,1)} %reverse)
}

ENTRY %main.13 (Arg_0.1: pred[], Arg_1.2: u8[300,451,3]) -> u8[300,451,3] {
  %Arg_0.1 = pred[]{:T(1024)} parameter(0)
  %convert.3 = s32[]{:T(256)} convert(pred[]{:T(1024)} %Arg_0.1)
  %Arg_1.2 = u8[300,451,3]{1,0,2:T(8,128)(4,1)} parameter(1)
  %conditional.12.clone = (u8[300,451,3]{1,0,2:T(8,128)(4,1)}) conditional(s32[]{:T(256)} %convert.3, u8[300,451,3]{1,0,2:T(8,128)(4,1)} %Arg_1.2, u8[300,451,3]{1,0,2:T(8,128)(4,1)} %Arg_1.2), branch_computations={%region_0.4, %region_1.9}
  ROOT %get-tuple-element = u8[300,451,3]{1,0,2:T(8,128)(4,1)} get-tuple-element((u8[300,451,3]{1,0,2:T(8,128)(4,1)}) %conditional.12.clone), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  HloInstruction* reverse = FindInstruction(module.get(), "reverse");
  EXPECT_THAT(reverse->operand(0), op::Copy());
}

TEST_F(CopyInsertionTest, InputOutputAliasCopy) {
  const char* const kModuleString = R"(
HloModule main_tf2xla.11, input_output_alias={ {0}: (0, {1}, may-alias) }

ENTRY %main_tf2xla.11 (arg_tuple.1: (f32[], f32[])) -> (f32[], f32[]) {
ROOT %arg_tuple.1 = (f32[]{:T(256)}, f32[]{:T(256)}) parameter(0), parameter_replication={false,false}, sharding={{replicated}, {replicated}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
}

TEST_F(CopyInsertionTest, AddControlDependencyForInputOutputAlias) {
  const char* const kModuleString = R"(
  HloModule test, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }

  ENTRY test {
    x = f32[3] parameter(0)
    y = f32[3] parameter(1)
    add = f32[3] add(x, y)
    mul = f32[3] multiply(x, y)
    ROOT result = (f32[3], f32[3]) tuple(add, mul)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  EXPECT_EQ(CountCopies(*module), 1);
  EXPECT_EQ(CountControlEdges(*module), 2);

  HloInstruction* add_instr = FindInstruction(module.get(), HloOpcode::kAdd);
  HloInstruction* mul_instr =
      FindInstruction(module.get(), HloOpcode::kMultiply);
  HloInstruction* copy_instr = FindInstruction(module.get(), HloOpcode::kCopy);
  EXPECT_TRUE(add_instr->control_predecessors()[0] == mul_instr);
  EXPECT_TRUE(copy_instr->control_predecessors()[0] == add_instr);
}

TEST_F(CopyInsertionTest, AsyncCallDUSNoCopy) {
  const char* const kModuleString = R"(
HloModule async_call

%called_computation {
  %out_param = s32[1024]{0} parameter(1)
  %input = s32[1024]{0} parameter(0)
  %size = s32[] constant(256)
  %index = s32[] custom-call(), custom_call_target="Baz"
  %start = s32[] multiply(s32[] %size, s32[] %index)
  %input2 = s32[256]{0} dynamic-slice(s32[1024]{0} %input, s32[] %start), dynamic_slice_sizes={256}
  %output = s32[256]{0} add(s32[256]{0} %input2, s32[256]{0} %input2)
  ROOT %output2 = s32[1024]{0} dynamic-update-slice(s32[1024]{0} %out_param, s32[256]{0} %output, s32[] %start)
}, execution_thread="foobar"

%async_wrapped {
  %async_param = s32[1024]{0} parameter(0)
  %async_param.1 = s32[1024]{0} parameter(1)
  ROOT %call = s32[1024]{0} call(s32[1024]{0} %async_param, s32[1024]{0} %async_param.1), to_apply=%called_computation
}, execution_thread="foobar"

ENTRY %main {
  %input.1 = s32[1024]{0} parameter(0)
  %buf = s32[1024]{0} custom-call(), custom_call_target="AllocateBuffer"
  %async-start = ((s32[1024]{0}, s32[1024]{0}), s32[1024]{0}, u32[]) async-start(s32[1024]{0} %input.1, s32[1024]{0} %buf), async_execution_thread="foobar", calls=%async_wrapped
  ROOT %async-done = s32[1024]{0} async-done(((s32[1024]{0}, s32[1024]{0}), s32[1024]{0}, u32[]) %async-start), async_execution_thread="foobar", calls=%async_wrapped
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get(), {"foobar"}).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, AsyncCallDUSCopy) {
  const char* const kModuleString = R"(
HloModule async_call

%called_computation {
  %out_param = s32[1024]{0} parameter(1)
  %input = s32[1024]{0} parameter(0)
  %size = s32[] constant(256)
  %index = s32[] custom-call(), custom_call_target="Baz"
  %start = s32[] multiply(s32[] %size, s32[] %index)
  %input2 = s32[256]{0} dynamic-slice(s32[1024]{0} %input, s32[] %start), dynamic_slice_sizes={256}
  %output = s32[256]{0} add(s32[256]{0} %input2, s32[256]{0} %input2)
  ROOT %output2 = s32[1024]{0} dynamic-update-slice(s32[1024]{0} %out_param, s32[256]{0} %output, s32[] %start)
}, execution_thread="foobar"

%async_wrapped {
  %async_param = s32[1024]{0} parameter(0)
  %async_param.1 = s32[1024]{0} parameter(1)
  ROOT %call = s32[1024]{0} call(s32[1024]{0} %async_param, s32[1024]{0} %async_param.1), to_apply=%called_computation
}, execution_thread="foobar"

ENTRY %main {
  %input.1 = s32[1024]{0} parameter(0)
  %input.2 = s32[1024]{0} parameter(1)
  %async-start = ((s32[1024]{0}, s32[1024]{0}), s32[1024]{0}, u32[]) async-start(s32[1024]{0} %input.1, s32[1024]{0} %input.2), async_execution_thread="foobar", calls=%async_wrapped
  ROOT %async-done = s32[1024]{0} async-done(((s32[1024]{0}, s32[1024]{0}), s32[1024]{0}, u32[]) %async-start), async_execution_thread="foobar", calls=%async_wrapped
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get(), {"foobar"}).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest,
       RegionAnalysisDoesNotAddUnnecessaryCopyOfInputTupleElements) {
  const char* const kModuleString = R"(
HloModule while_aliasing, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias) }

add {
  param_0 = f32[1,128] parameter(0)
  param_1 = f32[1,128] parameter(1)
  ROOT add = f32[1,128] add(param_0, param_1)
}

condition {
  input_tuple = (f32[1,128], f32[1,128], pred[]) parameter(0)
  ROOT cond = pred[] get-tuple-element(input_tuple), index=2
}

body {
  input_tuple = (f32[1,128], f32[1,128], pred[]) parameter(0)
  param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
  param_1 = f32[1,128] get-tuple-element(input_tuple), index=1
  cond = pred[] get-tuple-element(input_tuple), index=2
  add = f32[1,128] add(param_0, param_1)
  c0 = f32[] constant(0)
  splat_c0 = f32[1,128] broadcast(c0), dimensions={}
  ROOT output_tuple = (f32[1,128], f32[1,128], pred[]) tuple(add, splat_c0, cond)
}

ENTRY main {
  param_0 = f32[1,128] parameter(0)
  param_1 = f32[1,128] parameter(1)
  param_2 = pred[] parameter(2)
  tuple = (f32[1,128], f32[1,128], pred[]) tuple(param_0, param_1, param_2)
  ROOT while = (f32[1,128], f32[1,128], pred[]) while(tuple), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();

  // Both params of add should not be copy
  auto add = FindInstruction(module.get(), "add.1");
  EXPECT_NE(add, nullptr);
  EXPECT_EQ(add->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(add->operand(1)->opcode(), HloOpcode::kGetTupleElement);
}

TEST_F(CopyInsertionTest,
       RegionAnalysisDoesNotAddCopyForNonUpdateParameterOfDynamicSliceUpdate) {
  const char* const kModuleString = R"(
HloModule while_aliasing, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias) }

fused_computation {
  param_0 = f32[4,2,128,512]{3,2,1,0} parameter(0)
  param_1 = f32[2,128,512]{2,1,0} parameter(1)
  bitcast.1 = f32[1,2,128,512]{3,2,1,0} bitcast(param_1)
  param_2 = s32[] parameter(2)
  constant.1 = s32[] constant(0)
  compare.1 = pred[] compare(param_2, constant.1), direction=LT
  constant.2 = s32[] constant(4)
  add.1 = s32[] add(param_2, constant.2)
  select.1 = s32[] select(compare.1, add.1, param_2)
  ROOT dynamic-update-slice.73 = f32[4,2,128,512]{3,2,1,0} dynamic-update-slice(param_0, bitcast.1, select.1, constant.1, constant.1, constant.1)
}

condition {
  input_tuple = (s32[], f32[2,128,512], f32[4,2,128,512], pred[]) parameter(0)
  ROOT cond = pred[] get-tuple-element(input_tuple), index=3
}

body {
  input_tuple = (s32[], f32[2,128,512], f32[4,2,128,512], pred[]) parameter(0)
  get-tuple-element.0 = s32[] get-tuple-element(input_tuple), index=0
  get-tuple-element.1 = f32[4,2,128,512]{3,2,1,0} get-tuple-element(input_tuple), index=2
  get-tuple-element.2 = f32[2,128,512]{2,1,0} get-tuple-element(input_tuple), index=1
  fusion = f32[4,2,128,512]{3,2,1,0} fusion(get-tuple-element.1, get-tuple-element.2, get-tuple-element.0), kind=kLoop, calls=fused_computation
  cond = pred[] get-tuple-element(input_tuple), index=3
  c0 = f32[] constant(0)
  fusion.1 = f32[2,128,512]{2,1,0} broadcast(c0), dimensions={}
  ROOT output_tuple = (s32[], f32[2,128,512], f32[4,2,128,512], pred[]) tuple(get-tuple-element.0, fusion.1, fusion, cond)
}

ENTRY main {
  param_0 = f32[2,128,512] parameter(0)
  param_1 = f32[4,2,128,512] parameter(1)
  param_2 = pred[] parameter(2)
  param_3 = s32[] parameter(3)
  tuple = (s32[], f32[2,128,512], f32[4,2,128,512], pred[]) tuple(param_3, param_0, param_1, param_2)
  ROOT while = (s32[], f32[2,128,512], f32[4,2,128,512], pred[]) while(tuple), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();

  // The param 1 of fusion should not be a copy
  auto fusion = FindInstruction(module.get(), "fusion");
  EXPECT_NE(fusion, nullptr);
  EXPECT_EQ(fusion->operand(1)->opcode(), HloOpcode::kGetTupleElement);
}

TEST_F(CopyInsertionTest, RegionAnalysisNoCopyOfAddOutputInsideWhileBody) {
  const char* const kModuleString = R"(
HloModule while_aliasing
condition {
  input_tuple = (f32[1,128], f32[1,128], pred[]) parameter(0)
  ROOT cond = pred[] get-tuple-element(input_tuple), index=2
}

body {
  input_tuple = (f32[1,128], f32[1,128], pred[]) parameter(0)
  param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
  param_1 = f32[1,128] get-tuple-element(input_tuple), index=1
  cond = pred[] get-tuple-element(input_tuple), index=2
  c0 = f32[] constant(0)
  splat_c0 = f32[1,128] broadcast(c0), dimensions={}
  add = f32[1,128] add(splat_c0, param_1)
  add_1 = f32[1,128] add(splat_c0, splat_c0)
  ROOT output_tuple = (f32[1,128], f32[1,128], pred[]) tuple(add, add_1, cond)
}

ENTRY main {
  param_0 = f32[1,128] parameter(0)
  param_1 = f32[1,128] parameter(1)
  param_2 = pred[] parameter(2)
  tuple = (f32[1,128], f32[1,128], pred[]) tuple(param_0, param_1, param_2)
  ROOT while = (f32[1,128], f32[1,128], pred[]) while(tuple), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(3) << module->ToString();

  auto root = FindInstruction(module.get(), "tuple.3");
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kGetTupleElement);
}

TEST_F(CopyInsertionTest, DontInsertCopiesInAsyncComputation) {
  constexpr absl::string_view kModuleString = R"(
HloModule test

%async_computation {
  %param_0 = f32[10,32,512]{2,1,0:T(8,128)S(5)} parameter(0)
  %param_1 = f32[1,32,512]{2,1,0:T(8,128)} parameter(1)
  %param_2 = s32[]{:T(128)} parameter(2)
  %param_3 = s32[]{:T(128)} parameter(3)
  %param_4 = s32[]{:T(128)} parameter(4)
  ROOT %dynamic-update-slice.1 = f32[10,32,512]{2,1,0:T(8,128)S(5)}
    dynamic-update-slice(%param_0, %param_1, %param_2, %param_3, %param_4)
}

ENTRY %main {
  %param.1 = (s32[]{:T(128)}, f32[32,512]{1,0:T(8,128)},
              f32[10,32,512]{2,1,0:T(8,128)S(5)}) parameter(0)
  %get-tuple-element.132 = f32[10,32,512]{2,1,0:T(8,128)S(5)} get-tuple-element(
    %param.1), index=2
  %get-tuple-element.131 = f32[32,512]{1,0:T(8,128)} get-tuple-element(
    %param.1), index=1
  %cosine.0 = f32[32,512]{1,0:T(8,128)} cosine(%get-tuple-element.131)
  %reshape.6 = f32[1,32,512]{2,1,0:T(8,128)} reshape(%cosine.0)
  %get-tuple-element.130 = s32[]{:T(128)} get-tuple-element(%param.1), index=0
  %constant.49 = s32[]{:T(128)} constant(0)
  %compare.13 = pred[]{:T(512)} compare(
      %get-tuple-element.130, %constant.49), direction=LT
  %constant.50 = s32[]{:T(128)} constant(10)
  %add.22 = s32[]{:T(128)} add(%get-tuple-element.130, %constant.50)
  %select.6 = s32[]{:T(128)} select(
      %compare.13, %add.22, %get-tuple-element.130)
  %dynamic-update-slice-start = (
    (f32[10,32,512]{2,1,0:T(8,128)S(5)}, f32[1,32,512]{2,1,0:T(8,128)},
     s32[]{:T(128)}, s32[]{:T(128)}, s32[]{:T(128)}),
     f32[10,32,512]{2,1,0:T(8,128)S(5)}, u32[]) async-start(
      %get-tuple-element.132, %reshape.6, %select.6,
      %constant.49, %constant.49), calls=%async_computation
  ROOT %dynamic-update-slice-done = f32[10,32,512]{2,1,0:T(8,128)S(5)}
    async-done(%dynamic-update-slice-start), calls=%async_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion;
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  LOG(INFO) << module->ToString();

  auto* async_computation = module->GetComputationWithName("async_computation");
  ASSERT_THAT(async_computation, NotNull());
  EXPECT_EQ(CountCopies(*async_computation), 0);

  auto* main_computation = module->GetComputationWithName("main");
  ASSERT_THAT(main_computation, NotNull());
  EXPECT_EQ(CountCopies(*main_computation), 1);
}

TEST_F(CopyInsertionTest, AsyncDUSInLoop) {
  constexpr absl::string_view kModuleString = R"(
HloModule module

async_wrapped {
  async_param.1 = s32[1024]{0} parameter(0)
  async_param.2 = s32[256]{0} parameter(1)
  async_param.3 = s32[] parameter(2)
  ROOT dus = s32[1024]{0} dynamic-update-slice(async_param.1, async_param.2, async_param.3)
}

condition {
  input_tuple = (s32[1024]{0}, s32[256]{0}, s32[], pred[]) parameter(0)
  ROOT cond = pred[] get-tuple-element(input_tuple), index=3
}

body {
  input_tuple = (s32[1024]{0}, s32[256]{0}, s32[], pred[]) parameter(0)
  input.1 = s32[1024]{0} get-tuple-element(input_tuple), index=0
  input.2 = s32[256]{0} get-tuple-element(input_tuple), index=1
  input.3 = s32[] get-tuple-element(input_tuple), index=2
  input.4 = pred[] get-tuple-element(input_tuple), index=3
  async-start = ((s32[1024]{0}, s32[256]{0}, s32[]), s32[1024]{0}, u32[]) async-start(input.1, input.2, input.3), calls=%async_wrapped
  async-done = s32[1024]{0} async-done(async-start), calls=async_wrapped
  ROOT tuple = (s32[1024]{0}, s32[256]{0}, s32[], pred[]) tuple(async-done, input.2, input.3, input.4)
}

ENTRY main {
  input.1 = s32[256]{0} parameter(0)
  input.2 = s32[] parameter(1)
  input.3 = pred[] parameter(2)
  broadcast = s32[1024]{0} broadcast(input.2), dimensions={}
  while_tuple = (s32[1024]{0}, s32[256]{0}, s32[], pred[]) tuple(broadcast, input.1, input.2, input.3)
  while = (s32[1024]{0}, s32[256]{0}, s32[], pred[]) while(while_tuple), condition=condition, body=body
  ROOT gte = s32[1024]{0} get-tuple-element(while), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, PartiallyPipelinedAsyncRecv) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test, entry_computation_layout={()->f32[16]{0}}, num_partitions=4

    while_body {
      param = ((f32[16]{0}, u32[], token[])) parameter(0)
      prev_recv = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=0
      recv_done = (f32[16]{0}, token[]) recv-done(prev_recv), channel_id=1
      after_all = token[] after-all()
      recv = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=1,
          frontend_attributes={
            _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      ROOT tuple = ((f32[16]{0}, u32[], token[])) tuple(recv)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = ((f32[16]{0}, u32[], token[])) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main_spmd {
      after_all = token[] after-all()
      recv = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=1,
          frontend_attributes={
            _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      init = ((f32[16]{0}, u32[], token[])) tuple(recv)
      while = ((f32[16]{0}, u32[], token[])) while(init),
          condition=while_condition, body=while_body
      recv_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=0
      recv_done = (f32[16]{0}, token[]) recv-done(recv_ctx), channel_id=1
      ROOT result = f32[16]{0} get-tuple-element(recv_done), index=0
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();

  // All async start/end will be ordered so that all copies are removable.
  EXPECT_EQ(CountCopies(*module), 0);

  // Expect control dependency from recv-done to recv.
  HloComputation* while_body =
      hlo_query::FindComputation(module.get(), "while_body");
  HloInstruction* recv_done =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecvDone);
  HloInstruction* recv =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecv);
  EXPECT_THAT(recv->control_predecessors(), UnorderedElementsAre(recv_done));
}

TEST_F(CopyInsertionTest, PartiallyPipelinedAsyncRecvMultipleUses) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test, entry_computation_layout={(f32[16]{0})->f32[16]{0}},
        num_partitions=4

    while_body {
      param = ((f32[16]{0}, u32[], token[]), f32[16]{0}) parameter(0)
      prev_recv = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=0
      recv_done = (f32[16]{0}, token[]) recv-done(prev_recv), channel_id=1
      recv_data = f32[16]{0} get-tuple-element(recv_done), index=0
      after_all = token[] after-all()
      recv = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=1,
          frontend_attributes={
            _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}

      // `recv_data` is again here, which extends it's live range.
      ROOT tuple = ((f32[16]{0}, u32[], token[]), f32[16]{0}) tuple(recv,
          recv_data)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = ((f32[16]{0}, u32[], token[]), f32[16]{0}) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main_spmd {
      data = f32[16]{0} parameter(0)
      after_all = token[] after-all()
      recv = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=1,
          frontend_attributes={
            _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
      init = ((f32[16]{0}, u32[], token[]), f32[16]{0}) tuple(recv, data)
      while = ((f32[16]{0}, u32[], token[]), f32[16]{0}) while(init),
          condition=while_condition, body=while_body
      recv_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=0
      recv_done = (f32[16]{0}, token[]) recv-done(recv_ctx), channel_id=1
      ROOT result = f32[16]{0} get-tuple-element(recv_done), index=0
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();

  // All async start/end will be ordered so that all copies, except for an extra
  // use of the recv result, are removable. Additionally, there will be one copy
  // leading into the loop.
  HloComputation* while_body =
      hlo_query::FindComputation(module.get(), "while_body");
  EXPECT_EQ(CountCopies(*module), 2);
  EXPECT_EQ(CountCopies(*while_body), 1);

  // Expect control dependency from recv-done to recv.
  HloInstruction* recv_done =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecvDone);
  HloInstruction* recv =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecv);
  HloInstruction* recv_done_copy =
      hlo_query::FindInstruction(while_body, HloOpcode::kCopy);
  EXPECT_THAT(recv_done_copy, op::Copy(op::GetTupleElement(recv_done)));
  EXPECT_THAT(recv->control_predecessors(),
              UnorderedElementsAre(recv_done, recv_done_copy));
}

TEST_F(CopyInsertionTest, PartiallyPipelinedAsyncSendMultipleUses) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test, entry_computation_layout={(f32[16]{0})->f32[16]{0}},
        num_partitions=4

    while_body {
      param = ((f32[16]{0}, u32[], token[]), f32[16]{0}) parameter(0)
      prev_send = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=0
      data = f32[16]{0} get-tuple-element(param), index=1
      send_done = (f32[16]{0}, token[]) send-done(prev_send), channel_id=1
      after_all = token[] after-all()
      send = (f32[16]{0}, u32[], token[]) send(data, after_all), channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}

      // `data` is used again here, which extends it's live range beyond `send`.
      ROOT tuple = ((f32[16]{0}, u32[], token[]), f32[16]{0}) tuple(send, data)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = ((f32[16]{0}, u32[], token[]), f32[16]{0}) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main_spmd {
      data0 = f32[16]{0} parameter(0)
      data1 = f32[16] add(data0, data0)
      after_all = token[] after-all()
      send = (f32[16]{0}, u32[], token[]) send(data1, after_all), channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      init = ((f32[16]{0}, u32[], token[]), f32[16]{0}) tuple(send, data1)
      while = ((f32[16]{0}, u32[], token[]), f32[16]{0}) while(init),
          condition=while_condition, body=while_body
      send_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=0
      send_done = (f32[16]{0}, token[]) send-done(send_ctx), channel_id=1
      ROOT data2 = f32[16]{0} get-tuple-element(while), index=1
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();

  // All async start/end will be ordered so that all copies, except for an extra
  // use of the send operand, are removable. Additionally, there will be one
  // copy leading into the loop.
  EXPECT_EQ(CountCopies(*module), 2);

  // For While-body, check for a copy of the send operand and a control
  // dependency from send-done to the copy.
  HloComputation* while_body =
      hlo_query::FindComputation(module.get(), "while_body");
  HloInstruction* send_done =
      hlo_query::FindInstruction(while_body, HloOpcode::kSendDone);
  HloInstruction* send =
      hlo_query::FindInstruction(while_body, HloOpcode::kSend);
  HloInstruction* send_operand_copy =
      hlo_query::FindInstruction(while_body, HloOpcode::kCopy);
  EXPECT_THAT(send, op::Send(send_operand_copy, op::AfterAll()));
  EXPECT_THAT(send_operand_copy->control_predecessors(),
              UnorderedElementsAre(send_done));

  // For main, check for a copy of the send operand feeding into the loop and
  // no control dependency from send-done to the copy.
  HloComputation* main = hlo_query::FindComputation(module.get(), "main_spmd");
  HloInstruction* send_main =
      hlo_query::FindInstruction(main, HloOpcode::kSend);
  HloInstruction* send_main_operand_copy =
      hlo_query::FindInstruction(main, HloOpcode::kCopy);
  EXPECT_THAT(send_main, op::Send(send_main_operand_copy, op::AfterAll()));
  EXPECT_TRUE(send_main_operand_copy->control_predecessors().empty());
}

TEST_F(CopyInsertionTest, PartiallyPipelinedAsyncSendRecvPipelineParallelism) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test, entry_computation_layout={(f32[16]{0})->f32[16]{0}},
        num_partitions=4

    while_body {
      param = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]),
          f32[16]{0}, f32[16]{0}) parameter(0)

      prev_fwd = f32[16]{0} get-tuple-element(param), index=3

      prev_send = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=0
      send_done = (f32[16]{0}, token[]) send-done(prev_send), channel_id=1
      prev_recv = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=1
      recv_done = (f32[16]{0}, token[]) recv-done(prev_recv), channel_id=2

      fwd = f32[16]{0} get-tuple-element(recv_done), index=0

      after_all = token[] after-all()
      send = (f32[16]{0}, u32[], token[]) send(prev_fwd, after_all),
          channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=2,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}

      // Both, the data that was sent and the data that was received are live
      // until the end of the while loop.
      ROOT tuple = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]),
          f32[16]{0}, f32[16]{0}) tuple(send, recv, prev_fwd, fwd)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]),
          f32[16]{0}, f32[16]{0}) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main_spmd {
      data = f32[16]{0} parameter(0)
      after_all = token[] after-all()
      recv = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      send = (f32[16]{0}, u32[], token[]) send(data, after_all), channel_id=2,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      init = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]),
          f32[16]{0}, f32[16]{0}) tuple(send, recv, data, data)
      while = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]),
          f32[16]{0}, f32[16]{0}) while(init), condition=while_condition,
          body=while_body
      recv_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=0
      recv_done = (f32[16]{0}, token[]) recv-done(recv_ctx), channel_id=1
      send_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=0
      send_done = (f32[16]{0}, token[]) send-done(send_ctx), channel_id=2
      ROOT data_ = f32[16]{0} get-tuple-element(recv_done), index=0
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();

  // All async start/end will be ordered so that all copies but two are
  // removable:
  // - The copy for the extra use of the send operand.
  // - The copy for the extra use of the recv result.
  // The copy removal heuristic fails on removing one data copy, so the total
  // number of copies in the while loop body is 3.
  HloComputation* while_body =
      hlo_query::FindComputation(module.get(), "while_body");
  EXPECT_EQ(CountCopies(*module), 6);
  EXPECT_EQ(CountCopies(*while_body), 3);

  // Expect control dependency from send-done to send.
  HloInstruction* send_done =
      hlo_query::FindInstruction(while_body, HloOpcode::kSendDone);
  HloInstruction* send =
      hlo_query::FindInstruction(while_body, HloOpcode::kSend);
  HloInstruction* send_operand_copy = send->mutable_operand(0);
  EXPECT_THAT(send_operand_copy, op::Copy());
  EXPECT_THAT(send, op::Send(send_operand_copy, op::AfterAll()));
  EXPECT_THAT(send_operand_copy->control_predecessors(),
              UnorderedElementsAre(send_done));

  // Expect control dependency from recv-done to recv.
  HloInstruction* recv_done =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecvDone);
  HloInstruction* recv =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecv);
  HloInstruction* recv_done_copy = *absl::c_find_if(
      recv->control_predecessors(), HloPredicateIsOp<HloOpcode::kCopy>);
  EXPECT_THAT(recv_done_copy, op::Copy(op::GetTupleElement(recv_done)));
  EXPECT_THAT(recv->control_predecessors(),
              UnorderedElementsAre(recv_done, recv_done_copy));
}

TEST_F(CopyInsertionTest,
       PartiallyPipelinedAsyncSendRecvPipelineParallelismDirectFwd) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test, num_partitions=4

    while_body {
      param = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]))
          parameter(0)

      send_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=0
      recv_ctx = (f32[16]{0}, u32[], token[]) get-tuple-element(param), index=1
      send_done = (f32[16]{0}, token[]) send-done(send_ctx), channel_id=1
      recv_done = (f32[16]{0}, token[]) recv-done(recv_ctx), channel_id=2

      data = f32[16]{0} get-tuple-element(recv_done), index=0

      after_all = token[] after-all()
      send_ctx_ = (f32[16]{0}, u32[], token[]) send(data, after_all),
          channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv_ctx_ = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=2,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}

      ROOT tuple = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]))
          tuple(send_ctx_, recv_ctx_)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]))
          parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main_spmd {
      data = f32[16]{0} parameter(0)
      after_all = token[] after-all()
      send_ctx = (f32[16]{0}, u32[], token[]) send(data, after_all),
          channel_id=2,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      recv_ctx = (f32[16]{0}, u32[], token[]) recv(after_all), channel_id=1,
          frontend_attributes={
            _xla_send_send_source_target_pairs={{0,1},{1,2},{2,3}}}
      init = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]))
          tuple(send_ctx, recv_ctx)
      while = ((f32[16]{0}, u32[], token[]), (f32[16]{0}, u32[], token[]))
          while(init), condition=while_condition, body=while_body
      recv_ctx_ = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=0
      recv_done = (f32[16]{0}, token[]) recv-done(recv_ctx_), channel_id=1
      send_ctx_ = (f32[16]{0}, u32[], token[]) get-tuple-element(while), index=1
      send_done = (f32[16]{0}, token[]) send-done(send_ctx_), channel_id=2
      ROOT data_ = f32[16]{0} get-tuple-element(recv_done), index=0
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();

  // Expect that send and recv do not share the same buffer. Expect this to be
  // enforced by a copy before starting to send.
  HloComputation* while_body =
      hlo_query::FindComputation(module.get(), "while_body");
  EXPECT_EQ(CountCopies(*module), 2);
  EXPECT_EQ(CountCopies(*while_body), 1);
  HloInstruction* send =
      hlo_query::FindInstruction(while_body, HloOpcode::kSend);
  HloInstruction* recv_done =
      hlo_query::FindInstruction(while_body, HloOpcode::kRecvDone);
  EXPECT_THAT(
      send, op::Send(op::Copy(op::GetTupleElement(recv_done)), op::AfterAll()));
}

// Staightline non-copyable chain, with input to the start op. A copy is needed
// for transitioning from copyable to non-copyable.
TEST_F(CopyInsertionTest, NonCopyableOneChainOutsideWhileLoop) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test

    ENTRY main {
      p0 = f32[16] parameter(0)
      p1 = f32[16] parameter(1)
      // Avoid directly using parameters to distinguish special copies from
      // non-copyable transitioning copies.
      d0 = f32[16] add(p0, p1)
      d1 = f32[16] add(p1, p1)
      pin-d0 = b(f32[16]) custom-call(d0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call = (b(f32[16]), u32[], token[])
        custom-call(pin-d0, d1), custom_call_target="update",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable = b(f32[16]) get-tuple-element(call), index=0
      d2 = f32[16] custom-call(noncopyable), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      // The control-predecessor prevents copy insertion from making r0 the
      // control predecessor of pin_d0 and remove the added copy.
      d3 = f32[16] add(d0, d1), control-predecessors={pin-d0}
      ROOT d4 = f32[16] add(d3, d2)
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 1);
  HloInstruction* copy = FindInstruction(module.get(), HloOpcode::kCopy);
  HloInstruction* pin_d0 = FindInstruction(module.get(), "pin-d0");
  EXPECT_EQ(copy, pin_d0->operand(0));
}

// Similar to the previous test, but the non-copyable chain is completely
// inside a while-body.
TEST_F(CopyInsertionTest, NonCopyableOneChainInsideWhileLoop) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test

    while_body {
      param = (f32[16], f32[16]) parameter(0)
      pw0 = f32[16] get-tuple-element(param), index=0
      pin-pw0 = b(f32[16]) custom-call(pw0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call = b(f32[16]) custom-call(pin-pw0), custom_call_target="update",
        output_to_operand_aliasing={{}: (0, {})}
      unpin-call = f32[16] custom-call(call), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      pw1 = f32[16] get-tuple-element(param), index=1
      dw = f32[16] add(pw1, pw0), control-predecessors={pin-pw0}
      ROOT tuple = (f32[16], f32[16]) tuple(unpin-call, dw)
    }

    while_condition {
      pc = (f32[16], f32[16]) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main {
      p0 = f32[16] parameter(0)
      // Avoid directly using p0 in the while-body to avoid copies.
      d0 = f32[16] add(p0, p0)
      d1 = f32[16] add(p0, p0)
      init = (f32[16], f32[16]) tuple(d0, d1)
      while = (f32[16], f32[16]) while(init), condition=while_condition, body=while_body
      d2 = f32[16] get-tuple-element(while), index=0
      d3 = f32[16] get-tuple-element(while), index=1
      ROOT d4 = f32[16] add(d2, d3)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 2);
  // A copy of the pinned operand.
  HloInstruction* pin_pw0 = FindInstruction(module.get(), "pin-pw0");
  EXPECT_THAT(pin_pw0,
              op::CustomCall(op::Copy(op::GetTupleElement(op::Parameter(0)))));
  // A copy of the call1 result.
  HloInstruction* root = hlo_query::FindComputation(module.get(), "while_body")
                             ->root_instruction();
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCopy);
}

// Similar to the previous test, but there are two non-copyable chains.
TEST_F(CopyInsertionTest, NonCopyableTwoChainsInsideWhileLoop) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test

    while_body {
      param = (f32[16], f32[16]) parameter(0)
      pw0 = f32[16] get-tuple-element(param), index=0
      pin0-pw0 = b(f32[16]) custom-call(pw0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call0 = b(f32[16]) custom-call(pin0-pw0), custom_call_target="update0",
        output_to_operand_aliasing={{}: (0, {})}
      pin1-pw0 = b(f32[16]) custom-call(pw0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call1 = (b(f32[16]), b(f32[16])) custom-call(pin1-pw0, call0),
        custom_call_target="update1",
        output_to_operand_aliasing={{0}: (1, {}), {1}: (0, {})}
      call1-0 = b(f32[16]) get-tuple-element(call1), index=0
      unpin-call1-0 = f32[16] custom-call(call1-0), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      call1-1 = b(f32[16]) get-tuple-element(call1), index=1
      call2 = b(f32[16]) custom-call(call1-1), custom_call_target="update2",
        output_to_operand_aliasing={{}: (0, {})}
      unpin-call2 = f32[16] custom-call(call2), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      pw1 = f32[16] get-tuple-element(param), index=1
      dw0 = f32[16] add(pw1, pw0), control-predecessors={pin0-pw0, pin1-pw0}
      dw1 = f32[16] add(dw0, unpin-call1-0)
      ROOT tuple = (f32[16], f32[16]) tuple(unpin-call2, dw1)
    }

    while_condition {
      pc = (f32[16], f32[16]) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main {
      p0 = f32[16] parameter(0)
      // Avoid directly using p0 in the while-body to avoid copies.
      d0 = f32[16] add(p0, p0)
      d1 = f32[16] add(p0, p0)
      init = (f32[16], f32[16]) tuple(d0, d1)
      while = (f32[16], f32[16]) while(init), condition=while_condition, body=while_body
      d2 = f32[16] get-tuple-element(while), index=0
      d3 = f32[16] get-tuple-element(while), index=1
      ROOT d4 = f32[16] add(d2, d3)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 3);
  // A copy of the pin0-pw0 operand, for transitioning the non-copyable chain
  // containing pin0-pw0 -> call0(operand 0) -> call1(operand 1).
  HloInstruction* pin0_pw0 = FindInstruction(module.get(), "pin0-pw0");
  EXPECT_THAT(pin0_pw0,
              op::CustomCall(op::Copy(op::GetTupleElement(op::Parameter(0)))));
  // A copy of pin1-pw0 operand 0, for transitioning to the non-copyable chain
  // containing pin1-pw0 -> call1(operand 0) -> call2(operand 0).
  HloInstruction* pin1_pw0 = FindInstruction(module.get(), "pin1-pw0");
  EXPECT_EQ(pin1_pw0->operand(0)->opcode(), HloOpcode::kCopy);
  // A copy of the unpin-call2 result, for transitioning from non-copyable to
  // copyable at the end of the second chain. The copy for unpin-call1-1, for
  // transitioning from non-copyable to copyable at the end of the first chain
  // is removed.
  HloInstruction* root = hlo_query::FindComputation(module.get(), "while_body")
                             ->root_instruction();
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCopy);
}

// This test is very similar to NonCopyableOneChainInsideWhileLoop, but the
// while-init and while-root both contain non-copyable values. This is a case
// where the non-copyable chain is partially inside the while-loop and partially
// outside the while-loop, but the part that is inside the while-loop is fully
// connected (from loop parameter all the way to the while-root). As such,
// no copy is needed inside the while-loop.
TEST_F(CopyInsertionTest, NonCopyableOneChainPartiallyInsideWhileLoop) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test

    while_body {
      param = (b(f32[16]), f32[16]) parameter(0)
      pw0 = b(f32[16]) get-tuple-element(param), index=0
      call0 = b(f32[16]) custom-call(pw0), custom_call_target="update0",
        output_to_operand_aliasing={{}: (0, {})}
      call1 = b(f32[16]) custom-call(call0), custom_call_target="update1",
        output_to_operand_aliasing={{}: (0, {})}
      pw1 = f32[16] get-tuple-element(param), index=1
      dw = f32[16] add(pw1, pw1)
      ROOT tuple = (b(f32[16]), f32[16]) tuple(call1, dw)
    }

    while_condition {
      pc = (b(f32[16]), f32[16]) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main {
      p0 = f32[16] parameter(0)
      // Avoid directly using p0 in the while-body to avoid copies.
      d0 = f32[16] add(p0, p0)
      d1 = f32[16] add(p0, p0)
      pin-d0 = b(f32[16]) custom-call(d0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      init = (b(f32[16]), f32[16]) tuple(pin-d0, d1)
      while = (b(f32[16]), f32[16]) while(init), condition=while_condition, body=while_body
      noncopyable0 = b(f32[16]) get-tuple-element(while), index=0
      d2 = f32[16] custom-call(noncopyable0), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      d3 = f32[16] get-tuple-element(while), index=1
      d4 = f32[16] add(d2, d3)
      ROOT d5 = f32[16] add(d4, d0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 1);
  // A copy for the pin-d0 operand.
  HloInstruction* pin_d0 = FindInstruction(module.get(), "pin-d0");
  const HloInstruction* copy = FindInstruction(module.get(), "copy");
  EXPECT_EQ(copy, pin_d0->operand(0));
}

// The pipelined non-copyable chain inside the while-loop are separated
// into two parts. This is similar to the pipelined Send/Recv cases.
TEST_F(CopyInsertionTest, NonCopyableChainPipelinedSeparatedParts) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test

    while_body {
      param = (b(f32[16]), f32[16]) parameter(0)
      noncopyable0-w = b(f32[16]) get-tuple-element(param), index=0
      pw1 = f32[16] get-tuple-element(param), index=1
      dw0 = f32[16] add(pw1, pw1)
      call0-w = (b(f32[16]), f32[16], token[])
        custom-call(noncopyable0-w, dw0), custom_call_target="update0",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable1-w = b(f32[16]) get-tuple-element(call0-w), index=0
      dw1 = f32[16] get-tuple-element(call0-w), index=1
      dw2 = f32[16] add(pw1, dw1)
      dw3 = f32[16] add(dw2, dw2)
      pin-dw3 = b(f32[16]) custom-call(dw3), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call1-w = (b(f32[16]), f32[16], token[])
        custom-call(dw2, pin-dw3), custom_call_target="update1",
        output_to_operand_aliasing={{0}:(1, {})}
      noncopyable2-w = b(f32[16]) get-tuple-element(call1-w), index=0
      dw4 = f32[16] get-tuple-element(call1-w), index=1
      unpin-noncopyable1-w = f32[16] custom-call(noncopyable1-w),
        custom_call_target="Unpin", output_to_operand_aliasing={{}:(0, {})}
      add = f32[16] add(dw4, unpin-noncopyable1-w) // keep noncopyable1-w alive.
      dw5 = f32[16] add(add, dw3) // Keep dw3 alive.
      ROOT tuple = (b(f32[16]), f32[16]) tuple(noncopyable2-w, dw5)
    }

    // Infinite loop to keep IR small.
    while_condition {
      pc = (b(f32[16]), f32[16]) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main {
      p0 = f32[16] parameter(0)
      p1 = f32[16] parameter(1)
      d0 = f32[16] add(p0, p1)
      d1 = f32[16] add(p1, p1)
      pin-d0 = b(f32[16]) custom-call(d0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call0 = (b(f32[16]), f32[16], token[])
        custom-call(pin-d0, d1), custom_call_target="update0",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable0 = b(f32[16]) get-tuple-element(call0), index=0
      d2 = f32[16] get-tuple-element(call0), index=1
      init = (b(f32[16]), f32[16]) tuple(noncopyable0, d2)
      while = (b(f32[16]), f32[16]) while(init), condition=while_condition,
        body=while_body
      noncopyable1 = b(f32[16]) get-tuple-element(while), index=0
      d3 = f32[16] get-tuple-element(while), index=1
      call1 = (b(f32[16]), f32[16], token[])
        custom-call(noncopyable1), custom_call_target="update1",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable2 = b(f32[16]) get-tuple-element(call1), index=0
      unpin = f32[16] custom-call(noncopyable2), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      d4 = f32[16] add(d0, d3)
      d5 = f32[16] add(d1, unpin)
      ROOT result = (f32[16], f32[16]) tuple(d4, d5)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 3);
  // In while-body, a copy of call0-w result used as operand 1 in add
  // and a copy of the pin-dw3 operand.
  HloInstruction* add = FindInstruction(module.get(), "add");
  EXPECT_EQ(add->operand(1)->opcode(), HloOpcode::kCopy);
  HloInstruction* pin_dw3 = FindInstruction(module.get(), "pin-dw3");
  EXPECT_EQ(pin_dw3->operand(0)->opcode(), HloOpcode::kCopy);
  // A copy of the pin-d0 operand 0 in main.
  HloInstruction* pin_d0 = FindInstruction(module.get(), "pin-d0");
  EXPECT_EQ(pin_d0->operand(0)->opcode(), HloOpcode::kCopy);
}

// The pipelined non-copyable chain inside the while-loop is fully connected.
// There is no need to add copies for transitioning in/out of non-copyable
// inside the while-loop. There is one copy for transitioning in/out of
// non-copyable at the while-init.
TEST_F(CopyInsertionTest, NonCopyableChainPipelinedConnectedParts) {
  constexpr absl::string_view kModuleString = R"(
    HloModule test

    while_body {
      param = (b(f32[16]), f32[16]) parameter(0)
      noncopyable0-w = b(f32[16]) get-tuple-element(param), index=0
      pw1 = f32[16] get-tuple-element(param), index=1
      dw0 = f32[16] add(pw1, pw1)
      call0-w = (b(f32[16]), f32[16], token[])
        custom-call(noncopyable0-w, dw0), custom_call_target="update0",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable1-w = b(f32[16]) get-tuple-element(call0-w), index=0
      dw1 = f32[16] get-tuple-element(call0-w), index=1
      dw2 = f32[16] add(pw1, dw1)
      call1-w = (b(f32[16]), f32[16], token[])
        custom-call(noncopyable1-w, dw2), custom_call_target="update1",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable2-w = b(f32[16]) get-tuple-element(call1-w), index=0
      dw3 = f32[16] get-tuple-element(call1-w), index=1
      dw4 = f32[16] add(dw3, pw1)
      ROOT tuple = (b(f32[16]), f32[16]) tuple(noncopyable2-w, dw4)
    }

    // Infinite loop to keep IR small.
    while_condition {
      param = (b(f32[16]), f32[16]) parameter(0)
      ROOT infinite_loop = pred[] constant(true)
    }

    ENTRY main {
      p0 = f32[16] parameter(0)
      p1 = f32[16] parameter(1)
      d0 = f32[16] add(p0, p1)
      d1 = f32[16] add(p1, p1)
      pin-d0 = b(f32[16]) custom-call(d0), custom_call_target="Pin",
        output_to_operand_aliasing={{}:(0, {})}
      call0 = (b(f32[16]), f32[16], token[])
        custom-call(pin-d0, d1), custom_call_target="update0",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable0 = b(f32[16]) get-tuple-element(call0), index=0
      d2 = f32[16] get-tuple-element(call0), index=1
      init = (b(f32[16]), f32[16]) tuple(noncopyable0, d2)
      while = (b(f32[16]), f32[16]) while(init), condition=while_condition, body=while_body
      noncopyable1 = b(f32[16]) get-tuple-element(while), index=0
      d3 = f32[16] get-tuple-element(while), index=1
      call1 = (b(f32[16]), f32[16], token[])
        custom-call(noncopyable1), custom_call_target="update1",
        output_to_operand_aliasing={{0}:(0, {})}
      noncopyable2 = b(f32[16]) get-tuple-element(call1), index=0
      unpin = f32[16] custom-call(noncopyable2), custom_call_target="Unpin",
        output_to_operand_aliasing={{}:(0, {})}
      d4 = f32[16] add(d0, d3)
      d5 = f32[16] add(d1, unpin)
      ROOT result = (f32[16], f32[16]) tuple(d4, d5)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  CopyInsertion copy_insertion(&alias_info_,
                               /*use_region_based_live_range_analysis=*/-1);

  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 1);
  // A copy of the pin-d0 operand.
  HloInstruction* pin_d0 = FindInstruction(module.get(), "pin-d0");
  EXPECT_EQ(pin_d0->operand(0)->opcode(), HloOpcode::kCopy);
}

}  // namespace
}  // namespace xla
