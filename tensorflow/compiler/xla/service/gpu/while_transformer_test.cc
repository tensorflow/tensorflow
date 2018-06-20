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

#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

class WhileTransformerTest : public HloTestBase {
 protected:
  WhileTransformerTest()
      : module_(CreateNewModule()),
        induction_variable_shape_(ShapeUtil::MakeShape(S32, {})),
        data_shape_(ShapeUtil::MakeShape(F32, {8})),
        condition_result_shape_(ShapeUtil::MakeShape(PRED, {})) {}

  std::unique_ptr<HloComputation> BuildConditionComputation(
      const int64 tuple_index, const int64 limit) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int32>(limit)));
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, GetLoopStateShape(tuple_index), "loop_state"));
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
      const int64 increment) {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, GetLoopStateShape(ind_var_tuple_index), "loop_state"));
    // Update the induction variable GTE(ind_var_tuple_index).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, ind_var_tuple_index));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int32>(increment)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(data_tuple_index).
    auto data = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        data_shape_, loop_state, data_tuple_index));
    // Use 'induction_variable' in computation with no path to output tuple.
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, induction_variable, {}));
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
                                        const int64 ind_var_tuple_index,
                                        const int64 ind_var_init) {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto induction_var_init = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int32>(ind_var_init)));
    auto data_init = builder.AddInstruction(HloInstruction::CreateConstant(
        Literal::CreateR1<float>({0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));
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

  void RunFusionPasses() {
    // Run standard fusion passes.
    EXPECT_TRUE(gpu::GpuInstructionFusion(/*may_duplicate=*/false)
                    .Run(module_.get())
                    .ValueOrDie());
    EXPECT_TRUE(gpu::GpuInstructionFusion(/*may_duplicate=*/true)
                    .Run(module_.get())
                    .ValueOrDie());
  }

  void RunCopyInsertionPass() {
    HloVerifier verifier;
    TF_ASSERT_OK(verifier.Run(module_.get()).status());
    CopyInsertion copy_insertion;
    TF_ASSERT_OK(copy_insertion.Run(module_.get()).status());
  }

  Shape GetLoopStateShape(const int64 ind_var_tuple_index) {
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

// TODO(b/68830972): The while transformer is far too fragile. It patterns
// matches the exact expressions of opcodes. Re-enable when transformation is
// more general
TEST_F(WhileTransformerTest, DISABLED_InductionVariableAtTupleElement0) {
  // Build computation with induction variable at tuple element 0.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 0);
  // Run HLO Optimization passes.
  RunFusionPasses();
  RunCopyInsertionPass();
  // Run WhileTransformer.
  auto result = gpu::CanTransformWhileToFor(while_hlo);
  TF_ASSERT_OK(result.status());
  // Check results.
  EXPECT_THAT(result.ConsumeValueOrDie(),
              Eq(std::tuple<int64, int64, int64>(0, 10, 1)));
}

// TODO(b/68830972): The while transformer is far too fragile. It patterns
// matches the exact expressions of opcodes. Re-enable when transformation is
// more general
TEST_F(WhileTransformerTest, DISABLED_InductionVariableAtTupleElement1) {
  // Build computation with induction variable at tuple element 1.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(1, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(1, 0, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 1, 0);
  // Run HLO Optimization passes.
  RunFusionPasses();
  RunCopyInsertionPass();
  // Run WhileTransformer.
  auto result = gpu::CanTransformWhileToFor(while_hlo);
  TF_ASSERT_OK(result.status());
  // Check results.
  EXPECT_THAT(result.ConsumeValueOrDie(),
              Eq(std::tuple<int64, int64, int64>(0, 10, 1)));
}

// TODO(b/68830972): The while transformer is far too fragile. It patterns
// matches the exact expressions of opcodes. Re-enable when transformation is
// more general
TEST_F(WhileTransformerTest, DISABLED_InvalidLoopLimit) {
  // Build computation with invalid loop limit.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 5));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 10);
  // Run HLO Optimization passes.
  RunFusionPasses();
  RunCopyInsertionPass();
  // Run WhileTransformer.
  auto result = gpu::CanTransformWhileToFor(while_hlo);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("Loop start must be less than loop limit."));
}

// TODO(b/68830972): The while transformer is far too fragile. It patterns
// matches the exact expressions of opcodes. Re-enable when transformation is
// more general
TEST_F(WhileTransformerTest, DISABLED_InvalidLoopIncrement) {
  // Build computation with invalid loop increment.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, -1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 0);
  // Run HLO Optimization passes.
  RunFusionPasses();
  RunCopyInsertionPass();
  // Run WhileTransformer.
  auto result = gpu::CanTransformWhileToFor(while_hlo);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("Loop increment must greater than zero."));
}

}  // namespace
}  // namespace xla
