/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/while_transformer.h"

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class WhileTransformerTest : public HloTestBase {
 protected:
  WhileTransformerTest()
      : module_(TestName()),
        induction_variable_shape_(ShapeUtil::MakeShape(S32, {})),
        data_shape_(ShapeUtil::MakeShape(F32, {8})),
        loop_state_shape_(ShapeUtil::MakeTupleShape(
            {induction_variable_shape_, data_shape_})),
        condition_result_shape_(ShapeUtil::MakeShape(PRED, {})) {}

  std::unique_ptr<HloComputation> BuildConditionComputation(
      const int64 tuple_index, const int64 limit) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(limit)));
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            limit_const->shape(), loop_state, tuple_index));
    builder.AddInstruction(
        HloInstruction::CreateBinary(condition_result_shape_, HloOpcode::kLt,
                                     induction_variable, limit_const));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> BuildBodyComputation(
      const int64 ind_var_tuple_index, const int64 data_tuple_index,
      const int64 increment, bool insert_copies = false) {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    // Update the induction variable GTE(ind_var_tuple_index).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, ind_var_tuple_index));
    auto inc = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32>(increment)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(data_tuple_index).
    auto data = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        data_shape_, loop_state, data_tuple_index));
    // Use 'induction_variable' in computation with no path to output tuple.
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, induction_variable, {8}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    auto tuple0 =
        ind_var_tuple_index == 0
            ? builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}))
            : builder.AddInstruction(HloInstruction::CreateTuple({add1, add0}));
    if (insert_copies) {
      InsertTupleElementCopies(ind_var_tuple_index, tuple0, &builder);
    }
    return builder.Build();
  }

  HloInstruction* BuildWhileInstruction(HloComputation* condition,
                                        HloComputation* body,
                                        const int64 ind_var_tuple_index,
                                        const int64 ind_var_init,
                                        bool insert_copies = false) {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto induction_var_init =
        builder.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(ind_var_init)));
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));
    auto loop_state_init =
        ind_var_tuple_index == 0
            ? builder.AddInstruction(
                  HloInstruction::CreateTuple({induction_var_init, data_init}))
            : builder.AddInstruction(
                  HloInstruction::CreateTuple({data_init, induction_var_init}));
    if (insert_copies) {
      loop_state_init = InsertTupleElementCopies(ind_var_tuple_index,
                                                 loop_state_init, &builder);
    }
    auto while_hlo = builder.AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape_, condition, body, loop_state_init));
    module_.AddEntryComputation(builder.Build());
    return while_hlo;
  }

  HloInstruction* InsertTupleElementCopies(const int64 ind_var_tuple_index,
                                           HloInstruction* tuple0,
                                           HloComputation::Builder* builder) {
    auto gte0 = builder->AddInstruction(HloInstruction::CreateGetTupleElement(
        induction_variable_shape_, tuple0, ind_var_tuple_index));
    const int64 gte1_tuple_index = ind_var_tuple_index == 0 ? 1 : 0;
    auto gte1 = builder->AddInstruction(HloInstruction::CreateGetTupleElement(
        data_shape_, tuple0, gte1_tuple_index));
    // Insert copies.
    auto copy0 = builder->AddInstruction(
        HloInstruction::CreateUnary(gte0->shape(), HloOpcode::kCopy, gte0));
    auto copy1 = builder->AddInstruction(
        HloInstruction::CreateUnary(gte1->shape(), HloOpcode::kCopy, gte1));

    return ind_var_tuple_index == 0
               ? builder->AddInstruction(
                     HloInstruction::CreateTuple({copy0, copy1}))
               : builder->AddInstruction(
                     HloInstruction::CreateTuple({copy1, copy0}));
  }

  HloModule module_;
  Shape induction_variable_shape_;
  Shape data_shape_;
  Shape loop_state_shape_;
  Shape condition_result_shape_;
};

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement0) {
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_.AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto result =
      gpu::CanTransformWhileToFor(BuildWhileInstruction(condition, body, 0, 0));
  EXPECT_TRUE(result.ok());
  auto tuple = result.ConsumeValueOrDie();
  EXPECT_EQ(0, std::get<0>(tuple));
  EXPECT_EQ(10, std::get<1>(tuple));
  EXPECT_EQ(1, std::get<2>(tuple));
}

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement0_WithBodyCopies) {
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body =
      module_.AddEmbeddedComputation(BuildBodyComputation(0, 1, 1, true));
  auto result =
      gpu::CanTransformWhileToFor(BuildWhileInstruction(condition, body, 0, 0));
  EXPECT_TRUE(result.ok());
  auto tuple = result.ConsumeValueOrDie();
  EXPECT_EQ(0, std::get<0>(tuple));
  EXPECT_EQ(10, std::get<1>(tuple));
  EXPECT_EQ(1, std::get<2>(tuple));
}

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement0_WithInitCopies) {
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_.AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto result = gpu::CanTransformWhileToFor(
      BuildWhileInstruction(condition, body, 0, 0, true));
  EXPECT_TRUE(result.ok());
  auto tuple = result.ConsumeValueOrDie();
  EXPECT_EQ(0, std::get<0>(tuple));
  EXPECT_EQ(10, std::get<1>(tuple));
  EXPECT_EQ(1, std::get<2>(tuple));
}

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement1) {
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(1, 10));
  auto body = module_.AddEmbeddedComputation(BuildBodyComputation(1, 0, 1));
  auto result =
      gpu::CanTransformWhileToFor(BuildWhileInstruction(condition, body, 1, 0));
  EXPECT_TRUE(result.ok());
  auto tuple = result.ConsumeValueOrDie();
  EXPECT_EQ(0, std::get<0>(tuple));
  EXPECT_EQ(10, std::get<1>(tuple));
  EXPECT_EQ(1, std::get<2>(tuple));
}

TEST_F(WhileTransformerTest, InvalidLoopLimit) {
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(0, 5));
  auto body = module_.AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto result = gpu::CanTransformWhileToFor(
      BuildWhileInstruction(condition, body, 0, 10));
  EXPECT_FALSE(result.ok());
  EXPECT_MATCH(
      result.status().error_message(),
      testing::ContainsRegex("Loop start must be less than loop limit."));
}

TEST_F(WhileTransformerTest, InvalidLoopIncrement) {
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_.AddEmbeddedComputation(BuildBodyComputation(0, 1, -1));
  auto result =
      gpu::CanTransformWhileToFor(BuildWhileInstruction(condition, body, 0, 0));
  EXPECT_FALSE(result.ok());
  EXPECT_MATCH(
      result.status().error_message(),
      testing::ContainsRegex("Loop increment must greater than zero."));
}

}  // namespace
}  // namespace xla
