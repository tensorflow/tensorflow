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

#include <cstdint>
#include <memory>

#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class WhileTransformerTest : public HloTestBase {
 protected:
  WhileTransformerTest()
      : module_(CreateNewVerifiedModule()),
        induction_variable_shape_(ShapeUtil::MakeShape(S32, {})),
        data_shape_(ShapeUtil::MakeShape(F32, {8})),
        condition_result_shape_(ShapeUtil::MakeShape(PRED, {})) {}

  std::unique_ptr<HloComputation> BuildConditionComputation(
      const int64_t tuple_index, const int64_t limit) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(limit)));
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, GetLoopStateShape(tuple_index), "loop_state"));
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            limit_const->shape(), loop_state, tuple_index));
    builder.AddInstruction(HloInstruction::CreateCompare(
        condition_result_shape_, induction_variable, limit_const,
        ComparisonDirection::kLt));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> BuildBodyComputation(
      const int64_t ind_var_tuple_index, const int64_t data_tuple_index,
      const int64_t increment) {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, GetLoopStateShape(ind_var_tuple_index), "loop_state"));
    // Update the induction variable GTE(ind_var_tuple_index).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, ind_var_tuple_index));
    auto inc = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(increment)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(data_tuple_index).
    auto data = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        data_shape_, loop_state, data_tuple_index));
    // Use 'induction_variable' in computation with no path to output tuple.
    auto cast = builder.AddInstruction(HloInstruction::CreateBitcastConvert(
        ShapeUtil::MakeShape(F32, {}), induction_variable));
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, cast, {}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    ind_var_tuple_index == 0
        ? builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}))
        : builder.AddInstruction(HloInstruction::CreateTuple({add1, add0}));
    return builder.Build();
  }

  HloInstruction* BuildWhileInstruction(HloComputation* condition,
                                        HloComputation* body,
                                        const int64_t ind_var_tuple_index,
                                        const int64_t ind_var_init) {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto induction_var_init =
        builder.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(ind_var_init)));
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));
    auto loop_state_init =
        ind_var_tuple_index == 0
            ? builder.AddInstruction(
                  HloInstruction::CreateTuple({induction_var_init, data_init}))
            : builder.AddInstruction(
                  HloInstruction::CreateTuple({data_init, induction_var_init}));
    auto while_hlo = builder.AddInstruction(
        HloInstruction::CreateWhile(GetLoopStateShape(ind_var_tuple_index),
                                    condition, body, loop_state_init));
    module_->AddEntryComputation(builder.Build());
    return while_hlo;
  }

  Shape GetLoopStateShape(const int64_t ind_var_tuple_index) {
    if (ind_var_tuple_index == 0) {
      return ShapeUtil::MakeTupleShape(
          {induction_variable_shape_, data_shape_});
    } else {
      return ShapeUtil::MakeTupleShape(
          {data_shape_, induction_variable_shape_});
    }
  }

  std::unique_ptr<HloModule> module_;
  Shape induction_variable_shape_;
  Shape data_shape_;
  Shape condition_result_shape_;
};

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement0) {
  // Build computation with induction variable at tuple element 0.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 0);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_TRUE(result);
  EXPECT_EQ(10, *result);
}

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement1) {
  // Build computation with induction variable at tuple element 1.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(1, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(1, 0, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 1, 0);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_TRUE(result);
  EXPECT_EQ(10, *result);
}

TEST_F(WhileTransformerTest, ImpossibleLoopLimit) {
  // Build computation with an impossible loop limit.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 5));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 10);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_TRUE(result);
  EXPECT_EQ(0, *result);
}

TEST_F(WhileTransformerTest, InvalidLoopIncrement) {
  // Build computation with invalid loop increment.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, -1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 0);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_FALSE(result);
}

}  // namespace
}  // namespace xla
