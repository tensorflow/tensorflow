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

#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;

// Test is parameterized on a bool which is whether the dataflow analysis is
// performed with SSA form.
class HloDataflowAnalysisTest : public HloTestBase,
                                public ::testing::WithParamInterface<bool> {
 protected:
  HloDataflowAnalysisTest() : module_(CreateNewModule()) {}

  // Run dataflow analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  HloDataflowAnalysis& RunAnalysis(bool ssa_form,
                                   bool bitcast_defines_value = false) {
    analysis_ =
        HloDataflowAnalysis::Run(module_.get(), ssa_form, bitcast_defines_value)
            .ConsumeValueOrDie();
    return *analysis_;
  }

  // Return a vector of the HloValues at the given program position.
  std::vector<HloValue> HloValuesAt(const HloInstruction* instruction,
                                    const ShapeIndex& index = {}) {
    CHECK(analysis_ != nullptr);
    std::vector<HloValue> values;
    for (const HloValue* value :
         analysis_->GetValueSet(instruction, index).values()) {
      values.push_back(*value);
    }
    return values;
  }

  // Returns true if the top-level values for instructions 'a' and 'b' may
  // interfere. Precondition: 'a' and 'b' define array-shaped values.
  bool InstructionsMayInterfere(const HloOrdering& ordering,
                                const HloInstruction* a,
                                const HloInstruction* b) {
    EXPECT_FALSE(ShapeUtil::IsTuple(a->shape()));
    EXPECT_FALSE(ShapeUtil::IsTuple(b->shape()));
    return analysis_->MayInterfere(analysis_->GetValueDefinedAt(a),
                                   analysis_->GetValueDefinedAt(b), ordering);
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloDataflowAnalysis> analysis_;

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
  const Shape vector_shape_ = ShapeUtil::MakeShape(F32, {42});
};

TEST_P(HloDataflowAnalysisTest, BinaryOperation) {
  // Test the dataflow for a simple binary operation (Add).
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, constant1, constant2));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Each instruction should define a single value.
  EXPECT_EQ(analysis.values().size(), 3);
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(add));

  // Verify the positions of the values. These positions are all trivial because
  // there are no instructions which forward values.
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).positions(),
              UnorderedElementsAre(HloPosition{constant1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).positions(),
              UnorderedElementsAre(HloPosition{constant2, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(add).positions(),
              UnorderedElementsAre(HloPosition{add, {}}));

  // Verify the uses of the values.
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).uses().empty());

  // Verify liveout values from the module.
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, TupleAndGtes) {
  // Verify the dataflow through a Tuple and GetTupleElement instructions.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({param0, param1}));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple, 1));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape_, HloOpcode::kAdd, gte0, gte1));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // The two params, tuple, and add should each define one value.
  EXPECT_EQ(analysis.values().size(), 4);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(param0));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(param1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(tuple, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(tuple, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(tuple, /*index=*/{1}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(gte0));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(gte1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(add));

  // Verify the positions of the values.
  EXPECT_THAT(
      analysis.GetValueDefinedAt(param0).positions(),
      UnorderedElementsAre(HloPosition{param0, {}}, HloPosition{tuple, {0}},
                           HloPosition{gte0, {}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(param1).positions(),
      UnorderedElementsAre(HloPosition{param1, {}}, HloPosition{tuple, {1}},
                           HloPosition{gte1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple).positions(),
              UnorderedElementsAre(HloPosition{tuple, {}}));

  // Verify uses. Of interest is that a GetTupleElement instruction is only a
  // use of the top-level value in the tuple operand.
  EXPECT_THAT(analysis.GetValueDefinedAt(param0).uses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(param1).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple, /*index=*/{}).uses(),
              UnorderedElementsAre(HloUse{gte0, 0, {}}, HloUse{gte1, 0, {}}));
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, NestedTuple) {
  // Verify the dataflow through a nested tuple.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto nested_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({tuple, tuple, constant1}));
  auto gte_tuple = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple->shape(), nested_tuple, 1));
  auto gte_out = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, gte_tuple, 0));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 4);

  // Verify positions and uses.
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant1).positions(),
      UnorderedElementsAre(
          HloPosition{constant1, {}}, HloPosition{tuple, {0}},
          HloPosition{nested_tuple, {0, 0}}, HloPosition{nested_tuple, {1, 0}},
          HloPosition{nested_tuple, {2}}, HloPosition{gte_tuple, {0}},
          HloPosition{gte_out, {}}));
  // Constant values should have no uses though one is live out. The positions
  // where they appear as operands are on instructions which do not use the
  // values (eg, Tuple).
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).uses().empty());
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant2).uses().empty());

  // The top-level tuple values are used in GTE instructions.
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple, /*index=*/{}).uses(),
              UnorderedElementsAre(HloUse{gte_out, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(nested_tuple, /*index=*/{}).uses(),
              UnorderedElementsAre(HloUse{gte_tuple, 0, {}}));

  EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
  EXPECT_FALSE(
      analysis.GetValueDefinedAt(tuple, /*index=*/{}).live_out_of_module());
  EXPECT_FALSE(analysis.GetValueDefinedAt(nested_tuple, /*index=*/{})
                   .live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, SingleCall) {
  // Test a single call of a subcomputation. The subcomputation adds its two
  // array-shaped parameters.
  auto subbuilder = HloComputation::Builder("Subcomputation");
  auto subparam0 = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto subparam1 = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto add = subbuilder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, subparam0, subparam1));
  HloComputation* called_computation =
      module_->AddEmbeddedComputation(subbuilder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto call = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 3);

  // The parameters of the subcomputation and the call instruction itself should
  // not define values. Their values flow from elsewhere.
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(subparam0));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(subparam1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(add));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(call));

  EXPECT_EQ(analysis.GetUniqueValueAt(subparam0),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(subparam1),
            analysis.GetValueDefinedAt(constant2));
  EXPECT_EQ(analysis.GetUniqueValueAt(call), analysis.GetValueDefinedAt(add));

  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));

  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_computation());
}

TEST_P(HloDataflowAnalysisTest, ComputationCalledTwiceWithSameArguments) {
  // Test a subcomputation which is called twice with identical values.
  auto subbuilder = HloComputation::Builder("Subcomputation");
  auto subparam0 = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto subparam1 = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto add = subbuilder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, subparam0, subparam1));
  HloComputation* called_computation =
      module_->AddEmbeddedComputation(subbuilder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto call1 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  auto call2 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kSubtract, call1, call2));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 4);

  // Definitions should be identical to the single callsite case.
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(subparam0));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(subparam1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(add));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(call1));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(call2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(sub));

  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));
  // The Add from the subcomputation is used as both operands of the Subtract.
  EXPECT_THAT(analysis.GetValueDefinedAt(add).uses(),
              UnorderedElementsAre(HloUse{sub, 0, {}}, HloUse{sub, 1, {}}));

  EXPECT_FALSE(analysis.GetValueDefinedAt(add).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_computation());

  EXPECT_TRUE(analysis.GetValueDefinedAt(sub).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(sub).live_out_of_computation());
}

TEST_P(HloDataflowAnalysisTest, ComputationCalledTwiceWithDifferentArguments) {
  // Test a subcomputation which is called twice with different argument values.
  auto subbuilder = HloComputation::Builder("Subcomputation");
  auto subparam0 = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto subparam1 = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto add = subbuilder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, subparam0, subparam1));
  HloComputation* called_computation =
      module_->AddEmbeddedComputation(subbuilder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto call1 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  auto call2 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {call1, constant2}, called_computation));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_FALSE(analysis.ValueIsDefinedAt(call1));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(call2));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(subparam0));

  EXPECT_THAT(HloValuesAt(subparam0),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                   analysis.GetValueDefinedAt(add)));
  EXPECT_THAT(HloValuesAt(subparam1),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant2)));

  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, NestedCalls) {
  // Test a module with nested computations. HLO is:
  //
  // F32[] inner_computation(F32[] %param0, F32[] %param1):
  //   %add = Add(%param0, %param1)
  //
  // F32[] outer_computation((F32[] %param0, F32[] %param1):
  //  ;; Note that parameters are interchanged in the call.
  //   %nested_call = Call(inner_computation, {%param1, %param0})
  //
  // F32[] entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %call = Call(outer_computation, {%constant1, %constant2})
  //
  auto inner_builder = HloComputation::Builder("InnerComputation");
  auto inner_param0 = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto inner_param1 = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto add = inner_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, inner_param0, inner_param1));
  HloComputation* inner_computation =
      module_->AddEmbeddedComputation(inner_builder.Build());

  auto outer_builder = HloComputation::Builder("OuterComputation");
  auto outer_param0 = outer_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto outer_param1 = outer_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  // Swizzle parameters.
  outer_builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {outer_param1, outer_param0}, inner_computation));
  HloComputation* outer_computation =
      module_->AddEmbeddedComputation(outer_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, outer_computation));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Only three values should be defined. Most instructions just pass through
  // their operand values.
  EXPECT_EQ(analysis.values().size(), 3);

  // Verify that the uses of the constants are properly swizzled by parameter
  // permutation in nested_call.
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).uses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));

  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, SingleWhile) {
  // Test a simple single while instruction. The while body includes a
  // pass-through value. HLO:
  //
  // body((F32[], F32[]) %tuple_param):
  //   %add = Add(%tuple_param{0}, %tuple_param{1})
  //   return Tuple(%tuple_param{0}, %add)
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %tuple = Tuple(%constant1, %constant2)
  //   return While(%tuple, body, condition)
  //
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Element 0 passes transparently through the body.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  auto add = body_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, body_element_0, body_element_1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_0, add}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  // Condition computation trivially returns a constant "false".
  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_TRUE(
      analysis.GetValueDefinedAt(cond_constant).live_out_of_computation());
  EXPECT_FALSE(analysis.GetValueDefinedAt(cond_constant).live_out_of_module());

  if (ssa_form) {
    // While instruction should define phi values. The value at index {0} is a
    // degenerate phi with a single input 'constant1'.
    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{0}).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(xla_while, /*index=*/{0}),
              &analysis.GetValueDefinedAt(constant1));
    EXPECT_TRUE(analysis.ValueIsDefinedAt(body_param, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(body_param, /*index=*/{0}).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(body_param, /*index=*/{0}),
              &analysis.GetValueDefinedAt(constant1));
    EXPECT_TRUE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(cond_param, /*index=*/{0}).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(cond_param, /*index=*/{0}),
              &analysis.GetValueDefinedAt(constant1));

    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1}).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(xla_while, /*index=*/{1}), nullptr);
    EXPECT_TRUE(analysis.ValueIsDefinedAt(body_param, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(body_param, /*index=*/{1}).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(body_param, /*index=*/{1}), nullptr);
    EXPECT_TRUE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(cond_param, /*index=*/{1}).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(cond_param, /*index=*/{1}), nullptr);

    EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
                UnorderedElementsAre(HloUse{xla_while, 0, {0}}));

    EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{0})
                    .live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1})
                    .live_out_of_module());

    EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_computation());
    EXPECT_FALSE(analysis.GetValueDefinedAt(add).live_out_of_module());
  } else {
    // While instruction and subcomputation parameters should not define values
    // in non-ssa form.
    EXPECT_FALSE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{0}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{1}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(body_param, /*index=*/{0}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(body_param, /*index=*/{1}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{0}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{1}));

    EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_computation());
  }
}

TEST_P(HloDataflowAnalysisTest, SequentialWhiles) {
  // Test sequential while instructions. The while body includes a
  // pass-through value. HLO:
  //
  // body((F32[], F32[]) %tuple_param):
  //   %add = Add(%tuple_param{0}, %tuple_param{1})
  //   return Tuple(%tuple_param{0}, %add)
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %tuple = Tuple(%constant1, %constant2)
  //   %while0 = While(%tuple, body, condition)
  //   %while1 = While(%while0, body, condition)
  //   return While(%while1, body, condition)
  //
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Element 0 passes transparently through the body.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  auto add = body_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, body_element_0, body_element_1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_0, add}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  auto xla_while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, xla_while0));
  auto xla_while2 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, xla_while1));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  if (ssa_form) {
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while2).live_out_of_module());
    EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  } else {
    // Element 0 is passed through all the while instructions and out of the
    // module.
    EXPECT_EQ(analysis.GetUniqueValueAt(xla_while0, /*index=*/{0}),
              analysis.GetValueDefinedAt(constant1));
    EXPECT_EQ(analysis.GetUniqueValueAt(xla_while1, /*index=*/{0}),
              analysis.GetValueDefinedAt(constant1));
    EXPECT_EQ(analysis.GetUniqueValueAt(xla_while2, /*index=*/{0}),
              analysis.GetValueDefinedAt(constant1));
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  }
}

TEST_P(HloDataflowAnalysisTest, NestedWhiles) {
  // Test nested while instructions. The inner body passes through element 0 of
  // its parameter, and the outer body passes through element 1.  HLO:
  //
  // inner_body((F32[], F32[]) %tuple_param):
  //   %add = Add(%tuple_param{0}, %tuple_param{1})
  //   return Tuple(%tuple_param{0}, %add)
  //
  // outer_body((F32[], F32[]) %tuple_param):
  //   %negate = Negate(%tuple_param{0})
  //   %tuple = Tuple(%negate, %tuple_param{1})
  //   return While(%tuple, inner_body, condition)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %tuple = Tuple(%constant1, %constant2)
  //   return While(%tuple, outer_body, condition)
  //
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  // Element 0 passes transparently through the body.
  auto inner_builder = HloComputation::Builder("inner_body");
  auto inner_param = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto inner_element_0 = inner_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, inner_param, 0));
  auto inner_element_1 = inner_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, inner_param, 1));
  auto add = inner_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, inner_element_0, inner_element_1));
  inner_builder.AddInstruction(
      HloInstruction::CreateTuple({inner_element_0, add}));
  HloComputation* inner_body =
      module_->AddEmbeddedComputation(inner_builder.Build());

  // Element 1 passes transparently through the body.
  auto outer_builder = HloComputation::Builder("outer_body");
  auto outer_param = outer_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto outer_element_0 = outer_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, outer_param, 0));
  auto negate = outer_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, outer_element_0));
  auto outer_element_1 = outer_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, outer_param, 1));
  auto outer_tuple = outer_builder.AddInstruction(
      HloInstruction::CreateTuple({negate, outer_element_1}));
  auto nested_while = outer_builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, condition, inner_body, outer_tuple));
  HloComputation* outer_body =
      module_->AddEmbeddedComputation(outer_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto entry_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, outer_body, tuple));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  if (ssa_form) {
    EXPECT_TRUE(analysis.ValueIsDefinedAt(inner_param, /*index=*/{1}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(inner_param, /*index=*/{1}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(nested_while, /*index=*/{0}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(inner_param, /*index=*/{1}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(nested_while, /*index=*/{1}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(nested_while, /*index=*/{1}).is_phi());

    EXPECT_TRUE(analysis.ValueIsDefinedAt(entry_while, /*index=*/{0}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(entry_while, /*index=*/{0}).is_phi());

    EXPECT_TRUE(analysis.ValueIsDefinedAt(entry_while, /*index=*/{1}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(entry_while, /*index=*/{1}).is_phi());
  } else {
    EXPECT_THAT(HloValuesAt(inner_param, /*index=*/{0}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(inner_param, /*index=*/{1}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(add),
                                     analysis.GetValueDefinedAt(constant2)));

    EXPECT_THAT(HloValuesAt(nested_while, /*index=*/{0}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(nested_while, /*index=*/{1}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(add),
                                     analysis.GetValueDefinedAt(constant2)));

    EXPECT_THAT(HloValuesAt(entry_while, /*index=*/{0}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(negate),
                                     analysis.GetValueDefinedAt(constant1)));
    EXPECT_THAT(HloValuesAt(entry_while, /*index=*/{1}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(add),
                                     analysis.GetValueDefinedAt(constant2)));
  }
}

TEST_P(HloDataflowAnalysisTest, SwizzlingWhile) {
  // Test a while instruction with a body which permutes it's tuple parameter
  // elements. HLO:
  //
  // body((F32[], F32[]) %tuple_param):
  //   return Tuple(%tuple_param{1}, %tuple_param{0})
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %tuple = Tuple(%constant1, %constant2)
  //   return While(%tuple, body, condition)
  //
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_1, body_element_0}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  if (ssa_form) {
    // Element 0 and 1 in the while should both be phi values.
    EXPECT_TRUE(analysis.ValueIsDefinedAt(body_param, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(body_param, /*index=*/{0}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(body_param, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(body_param, /*index=*/{1}).is_phi());

    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{0}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1}).is_phi());

    EXPECT_TRUE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(cond_param, /*index=*/{0}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(cond_param, /*index=*/{1}).is_phi());

    EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_FALSE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{})
                    .live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{0})
                    .live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1})
                    .live_out_of_module());
  } else {
    // Elements 0 and 1 have both constants as reaching definitions.
    EXPECT_THAT(HloValuesAt(xla_while, /*index=*/{0}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                     analysis.GetValueDefinedAt(constant2)));
    EXPECT_THAT(HloValuesAt(xla_while, /*index=*/{1}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                     analysis.GetValueDefinedAt(constant2)));
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
  }
}

TEST_P(HloDataflowAnalysisTest, ArraySelect) {
  // Test a kSelect of an array value.
  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      scalar_shape_, HloOpcode::kSelect, pred, constant1, constant2));

  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(select));
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(select).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, TupleSelect) {
  // Test a kSelect of a tuple value. Non-top-level element flow through the
  // instruction.
  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(3.0)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(4.0)));
  auto tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant1}));
  auto tuple2 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant2}));
  auto tuple3 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant3}));
  auto tuple4 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant4}));
  const Shape tuple_shape = tuple1->shape();
  auto select11 = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple1, tuple1));
  auto select12 = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple1, tuple2));
  auto select34 = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple3, tuple4));
  auto select1234 = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, select12, select34));

  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Top-level value is always defined by a kSelect.
  EXPECT_TRUE(analysis.ValueIsDefinedAt(select11));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(select12));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(select34));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(select1234));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(select11, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(select12, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(select34, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(select1234, /*index=*/{0}));

  EXPECT_THAT(HloValuesAt(select11, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant1)));
  EXPECT_THAT(HloValuesAt(select12, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                   analysis.GetValueDefinedAt(constant2)));
  EXPECT_THAT(HloValuesAt(select34, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant3),
                                   analysis.GetValueDefinedAt(constant4)));
  EXPECT_THAT(HloValuesAt(select1234, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                   analysis.GetValueDefinedAt(constant2),
                                   analysis.GetValueDefinedAt(constant3),
                                   analysis.GetValueDefinedAt(constant4)));

  EXPECT_THAT(
      analysis.GetValueDefinedAt(tuple1, /*index=*/{}).uses(),
      UnorderedElementsAre(HloUse{select11, 1, {}}, HloUse{select11, 2, {}},
                           HloUse{select12, 1, {}}));

  // The two constant values just pass through the Selects and are not
  // used. They are live out however.
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).uses().empty());
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant2).uses().empty());
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, NestedTupleSelect) {
  // Test kSelect of a nested tuple.
  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(3.0)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(4.0)));
  auto constant5 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(5.0)));
  auto inner_tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant2, constant3}));
  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, inner_tuple1}));
  auto inner_tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant5, constant3}));
  auto tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant4, inner_tuple2}));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kSelect, pred, tuple1, tuple2));

  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(select));

  EXPECT_THAT(HloValuesAt(select, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                   analysis.GetValueDefinedAt(constant4)));
  EXPECT_THAT(HloValuesAt(select, /*index=*/{1}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(inner_tuple1),
                                   analysis.GetValueDefinedAt(inner_tuple2)));
  EXPECT_THAT(HloValuesAt(select, /*index=*/{1, 0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant2),
                                   analysis.GetValueDefinedAt(constant5)));
  EXPECT_THAT(HloValuesAt(select, /*index=*/{1, 1}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(constant3)));
}

TEST_P(HloDataflowAnalysisTest, TupleSelectToWhile) {
  // Test a tuple-shaped kSelect feeding a kWhile instruction. HLO:
  //
  // body((F32[], F32[]) %tuple_param):
  //   %add = Add(%tuple_param{0}, %tuple_param{1})
  //   return Tuple(%tuple_param{0}, %add)
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %constant3 = Constant(3.0)
  //   %tuple1 = Tuple(%constant1)
  //   %tuple2 = Tuple(%constant2)
  //   %select = Select(%tuple1, %tuple2)
  //   %gte = GetTupleElement(%select, 0)
  //   %tuple = Tuple(%gte, %constant3)
  //   return While(%tuple, body, condition)
  //
  auto builder = HloComputation::Builder(TestName());

  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Element 0 passes transparently through the body.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  auto add = body_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, body_element_0, body_element_1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_0, add}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(3.0)));
  auto tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant1}));
  auto tuple2 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant2}));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kSelect, pred, tuple1, tuple2));
  auto gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, select, 0));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({gte, constant3}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple->shape(), condition, body, tuple));

  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  if (ssa_form) {
    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{0}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{0}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1}).is_phi());

    EXPECT_FALSE(analysis.ValueIsDefinedAt(select, /*index=*/{0}));

    EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_FALSE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
    EXPECT_FALSE(analysis.GetValueDefinedAt(constant3).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1})
                    .live_out_of_module());
  } else {
    EXPECT_THAT(HloValuesAt(gte),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                     analysis.GetValueDefinedAt(constant2)));
    EXPECT_THAT(HloValuesAt(xla_while, /*index=*/{0}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant1),
                                     analysis.GetValueDefinedAt(constant2)));
    EXPECT_THAT(HloValuesAt(xla_while, /*index=*/{1}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(add),
                                     analysis.GetValueDefinedAt(constant3)));
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant3).live_out_of_module());
  }
}

TEST_P(HloDataflowAnalysisTest, BitcastDefinesValue) {
  // Test the bitcast_defines_value flag to the dataflow analysis.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kBitcast, constant));

  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  {
    const HloDataflowAnalysis& analysis =
        RunAnalysis(ssa_form, /*bitcast_defines_value=*/true);

    EXPECT_EQ(analysis.values().size(), 2);

    EXPECT_TRUE(analysis.ValueIsDefinedAt(constant));
    EXPECT_TRUE(analysis.ValueIsDefinedAt(bitcast));
    EXPECT_FALSE(analysis.GetValueDefinedAt(constant).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(bitcast).live_out_of_module());
  }
  {
    const HloDataflowAnalysis& analysis =
        RunAnalysis(ssa_form, /*bitcast_defines_value=*/false);
    EXPECT_EQ(analysis.values().size(), 1);

    EXPECT_TRUE(analysis.ValueIsDefinedAt(constant));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(bitcast));
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant).live_out_of_module());
  }
}

TEST_P(HloDataflowAnalysisTest, TupleCopy) {
  // Test that a tuple-shaped copy only copies (defines) the top-level value.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({param0, param1}));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(tuple->shape(), HloOpcode::kCopy, tuple));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 4);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(param0));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(param1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(tuple, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(tuple, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(tuple, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(copy, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(copy, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(copy, /*index=*/{1}));

  EXPECT_THAT(HloValuesAt(copy, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(param0)));
  EXPECT_THAT(HloValuesAt(copy, /*index=*/{1}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(param1)));
  EXPECT_TRUE(
      analysis.GetValueDefinedAt(copy, /*index=*/{}).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, ElementwiseChainInterference) {
  // A simple chain of elementwise operations. No values should interfere.
  //
  // param --> negate -> exp -> log
  //
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kExp, negate));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kLog, exp));

  module_->AddEntryComputation(builder.Build());
  RunAnalysis(GetParam());

  DependencyHloOrdering ordering(module_.get());

  // No values should interfere.
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, log));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, negate, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, negate, log));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, log));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, log, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, log, exp));

  // Values should interfere with itself.
  EXPECT_TRUE(InstructionsMayInterfere(ordering, exp, exp));
}

TEST_P(HloDataflowAnalysisTest, MultipleEntryParameters_Sequential) {
  // Two entry params, which interfere with each other.
  //
  // param0 --> negate ---------------\
  //                param1 --> exp --> add
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, vector_shape_, "param1"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kNegate, param0));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kExp, param1));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      vector_shape_, HloOpcode::kAdd, negate, exp));

  auto entry = module_->AddEntryComputation(builder.Build());
  RunAnalysis(GetParam());

  SequentialHloOrdering::HloModuleSequence sequence;
  sequence.insert({entry, {param0, negate, param1, exp, add}});
  SequentialHloOrdering ordering(module_.get(), sequence);

  // Entry parameters interfere as if they are defined simultaneously at
  // the very beginning.
  EXPECT_TRUE(InstructionsMayInterfere(ordering, param0, param1));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param0, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param0, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param0, add));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, param1, param0));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, param1, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param1, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param1, add));

  // Negate and exp still interfere.
  EXPECT_TRUE(InstructionsMayInterfere(ordering, negate, exp));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, exp, negate));

  // But {negate, add} and {exp, add} don't interfere.
  EXPECT_FALSE(InstructionsMayInterfere(ordering, negate, add));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, add, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, add));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, add, exp));
}

TEST_P(HloDataflowAnalysisTest, WhileParameters_Sequential) {
  // Similar to MultipleEntryParameters_Sequential, but the parameter is of
  // while body computation. Body computation in the sequential order:
  //
  //  %constant = Constant(...)
  //  %exp = Exp(%constant)
  //  %param = Param(0)
  //  %add = Add(%param, %exp)  ;; Root of body
  //  %dead_constant = Constant(...)
  //  %dead_negate = Negate(%dead_constant)
  //
  // %constant and its only use %exp are ordered before 'param'. However, the
  // %constant and %param values still interfere because the parameter is
  // considered live into the while body.
  //
  // Similarly, %dead_constant and %dead_negate are ordered after the root of
  // the body computation %add. However, %add is liveout of the computation so
  // %dead_constant and %add interfere.
  auto body_builder = HloComputation::Builder(TestName());
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "body_param"));
  auto constant = body_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto exp = body_builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kExp, constant));
  auto add = body_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, exp, body_param));
  auto dead_constant = body_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto dead_negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, dead_constant));
  HloComputation* body = module_->AddEmbeddedComputation(
      body_builder.Build(/*root_instruction=*/add));

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "cond_param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape_, condition, body, param));

  auto entry = module_->AddEntryComputation(builder.Build());
  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  SequentialHloOrdering::HloModuleSequence sequence;
  sequence.insert({entry, {param, xla_while}});
  sequence.insert({condition, {cond_param, cond_constant}});
  // Construct the order such that 'constant' and its use 'exp' are before
  // body_param.
  sequence.insert({body, {constant, exp, body_param, add}});

  SequentialHloOrdering ordering(module_.get(), sequence);

  // 'add' is the body root even though later instructions follow in the order
  // like 'dead_negate'. Only 'add' should be live out of the computation.
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_computation());
  EXPECT_FALSE(
      analysis.GetValueDefinedAt(dead_negate).live_out_of_computation());

  // 'add' is live out of the body and will interfere with an later instructions
  // such as 'dead_constant' and 'dead_negate'.
  EXPECT_TRUE(InstructionsMayInterfere(ordering, add, dead_constant));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, add, dead_negate));

  // The remaining checks test phi values defined by body and condition
  // parameters which only occur in the SSA form of the analysis.
  if (ssa_form) {
    // Though the ordering suggests 'constant' and 'param' should not interfere,
    // 'param' is live in and thus interferes with any earlier instruction of
    // the computation in the order (eg 'constant')'
    EXPECT_TRUE(InstructionsMayInterfere(ordering, body_param, constant));
    EXPECT_TRUE(InstructionsMayInterfere(ordering, body_param, exp));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, body_param, add));

    // The following values end up in the same buffer:
    //  (1) the init value: 'param'
    //  (2) the body parameter: 'body_param'
    //  (3) the condition parameter: 'cond_param'
    //  (4) the root value of the while body: 'add'
    //  (5) the while value: 'xla_while'
    // None should interfere.
    EXPECT_FALSE(InstructionsMayInterfere(ordering, param, body_param));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, param, cond_param));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, param, add));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, param, xla_while));

    EXPECT_FALSE(InstructionsMayInterfere(ordering, body_param, cond_param));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, body_param, add));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, body_param, xla_while));

    EXPECT_FALSE(InstructionsMayInterfere(ordering, cond_param, add));
    EXPECT_FALSE(InstructionsMayInterfere(ordering, cond_param, xla_while));

    EXPECT_FALSE(InstructionsMayInterfere(ordering, add, xla_while));
  }
}

TEST_P(HloDataflowAnalysisTest, NonElementwiseOperand) {
  // A chain of operations with two elementwise and one non-elementwise. The
  // elementwise op should not interfere with its operand, while the
  // non-elementwise op should interfere. Entry params always interfere.
  //
  // param --> exp -> negate -> reverse
  //
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kExp, param));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kNegate, exp));
  auto reverse = builder.AddInstruction(
      HloInstruction::CreateReverse(vector_shape_, negate, {0}));

  module_->AddEntryComputation(builder.Build());
  RunAnalysis(GetParam());

  DependencyHloOrdering ordering(module_.get());

  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, reverse));

  // Negate is elementwise, so doesn't interfere with its operand.
  // Reverse is non-elementwise, so does interfere with its operand.
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, negate));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, negate, reverse));
}

TEST_P(HloDataflowAnalysisTest, OverlappedValues) {
  // Verify simultaneously live values interfere (exp and negate).
  //
  // param --> negate -> add
  //     \---> exp -----/
  //
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kExp, param));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      vector_shape_, HloOpcode::kAdd, negate, exp));

  module_->AddEntryComputation(builder.Build());
  RunAnalysis(GetParam());

  DependencyHloOrdering ordering(module_.get());

  EXPECT_TRUE(InstructionsMayInterfere(ordering, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, param, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, add));

  // Negate and exp interfere with each other, but not with add.
  EXPECT_TRUE(InstructionsMayInterfere(ordering, negate, exp));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, negate, add));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, add, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, add));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, add, exp));
}

TEST_P(HloDataflowAnalysisTest, OverlappedValuesSequentialOrder) {
  // Identical to the test OverlappedValue but using a sequential ordering of
  // HLO instructions.
  //
  // param --> negate -> add
  //     \---> exp -----/
  //
  // Sequential order:
  //  param, negate, exp, add
  //
  // Liveness is identical to the DependencyHloOrdering.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kExp, param));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      vector_shape_, HloOpcode::kAdd, negate, exp));

  auto entry = module_->AddEntryComputation(builder.Build());
  RunAnalysis(GetParam());

  SequentialHloOrdering::HloModuleSequence sequence;
  std::vector<const HloInstruction*> order = {param, negate, exp, add};
  sequence.emplace(entry, order);

  SequentialHloOrdering ordering(module_.get(), sequence);

  EXPECT_TRUE(InstructionsMayInterfere(ordering, param, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, exp));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, param, add));

  // Negate and exp interfere with each other, but not with add.
  EXPECT_TRUE(InstructionsMayInterfere(ordering, negate, exp));
  EXPECT_TRUE(InstructionsMayInterfere(ordering, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, negate, add));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, add, negate));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, add));
  EXPECT_FALSE(InstructionsMayInterfere(ordering, add, exp));
}

TEST_P(HloDataflowAnalysisTest, EmbeddedComputationInterference) {
  // Test MayInterfere() for embedded computation, specifically the interference
  // of values in different computations.
  //
  // embedded_computation:
  //   %embedded_param = Param(0)
  //   %embedded_log = Log(%embedded_param)
  //
  // entry computation:
  //   %param = Param(0)
  //   %negate = Negate(%param)
  //   %exp = Negate(%exp)
  //   %call = Call(embedded_computation, {%exp})
  //   %add = Add(%negate, %call)
  //
  // Note %negate is live across the call and should interfere with all values
  // in the embedded computation.
  auto embedded_builder = HloComputation::Builder(TestName() + "_embedded");
  auto embedded_param = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "embedded_param"));
  auto embedded_log =
      embedded_builder.AddInstruction(HloInstruction::CreateUnary(
          vector_shape_, HloOpcode::kLog, embedded_param));
  auto embedded_computation =
      module_->AddEmbeddedComputation(embedded_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vector_shape_, HloOpcode::kExp, param));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(vector_shape_, {exp}, embedded_computation));
  builder.AddInstruction(HloInstruction::CreateBinary(
      vector_shape_, HloOpcode::kAdd, negate, call));
  module_->AddEntryComputation(builder.Build());
  RunAnalysis(GetParam());

  DependencyHloOrdering ordering(module_.get());

  // Exp only use is the call so it should not interfere with values inside the
  // embedded computation.
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, embedded_log));

  // Negate is live across the call and should interfere with values in the
  // embedded computation
  EXPECT_TRUE(InstructionsMayInterfere(ordering, negate, embedded_log));
}

TEST_P(HloDataflowAnalysisTest, UpdateAnalysisForWhile) {
  // Test updating dataflow after modifying a module with an array shaped while:
  //
  // body(F32[]  %param):
  //   %negate = Negate(%param)
  //
  // condition(F32[] %param):
  //   return Constant(false)
  //
  // entry:
  //   %constant = Constant(1.0)
  //   %exp = Exp(%constant)
  //   return While(%exp, body, condition)
  //
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, body_param));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  // Condition computation trivially returns a constant "false".
  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kExp, constant));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape_, condition, body, exp));
  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Sanity check the initial dataflow analysis before transforming the HLO
  // graph.
  if (ssa_form) {
    EXPECT_TRUE(analysis.ValueIsDefinedAt(body_param));
    EXPECT_TRUE(analysis.GetValueDefinedAt(body_param).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(body_param), nullptr);

    EXPECT_TRUE(analysis.ValueIsDefinedAt(cond_param));
    EXPECT_TRUE(analysis.GetValueDefinedAt(cond_param).is_phi());
    EXPECT_EQ(analysis.ResolvePhi(cond_param), nullptr);

    EXPECT_FALSE(analysis.GetValueDefinedAt(exp).live_out_of_module());
    EXPECT_FALSE(analysis.GetValueDefinedAt(negate).live_out_of_module());
  } else {
    EXPECT_THAT(HloValuesAt(body_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(exp),
                                     analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(cond_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(exp),
                                     analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(xla_while),
                UnorderedElementsAre(analysis.GetValueDefinedAt(exp),
                                     analysis.GetValueDefinedAt(negate)));

    EXPECT_TRUE(analysis.GetValueDefinedAt(negate).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(exp).live_out_of_module());
  }

  // Set the body root to the body_param. Previously it was Negate(body_param).
  body->set_root_instruction(body_param);

  // Prior to updating, verify that the dataflow analysis is no longer valid.
  Status verify_status = analysis.VerifyAgainstReference();
  EXPECT_FALSE(verify_status.ok());

  analysis.UpdateAfterChangingRoot(/*old_root=*/negate,
                                   /*new_root=*/body_param);

  // Analysis should be valid after the update.
  TF_EXPECT_OK(analysis.VerifyAgainstReference());

  if (ssa_form) {
    // The phis should now be resolvable as 'exp' is passed through the body
    // transparently.
    EXPECT_EQ(analysis.ResolvePhi(body_param),
              &analysis.GetValueDefinedAt(exp));
    EXPECT_EQ(analysis.ResolvePhi(cond_param),
              &analysis.GetValueDefinedAt(exp));
    EXPECT_EQ(analysis.ResolvePhi(xla_while), &analysis.GetValueDefinedAt(exp));
    EXPECT_FALSE(analysis.GetValueDefinedAt(exp).live_out_of_module());
  } else {
    EXPECT_THAT(HloValuesAt(body_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(exp)));
    EXPECT_THAT(HloValuesAt(cond_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(exp)));
    EXPECT_THAT(HloValuesAt(xla_while),
                UnorderedElementsAre(analysis.GetValueDefinedAt(exp)));
    EXPECT_TRUE(analysis.GetValueDefinedAt(exp).live_out_of_module());
  }
  EXPECT_FALSE(analysis.GetValueDefinedAt(negate).live_out_of_module());

  // Now replace the operand of the while with %constant (was %exp).
  TF_ASSERT_OK(exp->ReplaceUseWith(xla_while, constant));
  analysis.UpdateAfterChangingOperand(xla_while, /*old_operand=*/exp,
                                      /*new_operand=*/constant);

  // Verify that the dataflow is correct.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  if (ssa_form) {
    // The phis now resolve to 'constant'.
    EXPECT_EQ(analysis.ResolvePhi(body_param),
              &analysis.GetValueDefinedAt(constant));
    EXPECT_EQ(analysis.ResolvePhi(cond_param),
              &analysis.GetValueDefinedAt(constant));
    EXPECT_EQ(analysis.ResolvePhi(xla_while),
              &analysis.GetValueDefinedAt(constant));
  } else {
    EXPECT_THAT(HloValuesAt(body_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant)));
    EXPECT_THAT(HloValuesAt(cond_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant)));
    EXPECT_THAT(HloValuesAt(xla_while),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant)));
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant).live_out_of_module());
  }

  // And finally make the negate the root of the body again.
  body->set_root_instruction(negate);
  analysis.UpdateAfterChangingRoot(/*old_root=*/body_param,
                                   /*new_root=*/negate);

  // Verify that the dataflow is correct.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  if (ssa_form) {
    // Phis should no longer be resolvable.
    EXPECT_EQ(analysis.ResolvePhi(body_param), nullptr);
    EXPECT_EQ(analysis.ResolvePhi(cond_param), nullptr);
    EXPECT_EQ(analysis.ResolvePhi(xla_while), nullptr);
  } else {
    EXPECT_THAT(HloValuesAt(body_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant),
                                     analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(cond_param),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant),
                                     analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(xla_while),
                UnorderedElementsAre(analysis.GetValueDefinedAt(constant),
                                     analysis.GetValueDefinedAt(negate)));

    EXPECT_FALSE(analysis.GetValueDefinedAt(exp).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(negate).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant).live_out_of_module());
  }

  // After the updates, verify that the dataflow is correct.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());
}

TEST_P(HloDataflowAnalysisTest, UpdateOfATupleSelect) {
  // Test changing the operands of kSelects of a tuple value and updating the
  // dataflow.
  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(3.0)));
  auto d = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(4.0)));
  auto tuple_a = builder.AddInstruction(HloInstruction::CreateTuple({a}));
  auto tuple_b = builder.AddInstruction(HloInstruction::CreateTuple({b}));
  auto tuple_c = builder.AddInstruction(HloInstruction::CreateTuple({c}));
  auto tuple_d = builder.AddInstruction(HloInstruction::CreateTuple({d}));
  const Shape tuple_shape = tuple_a->shape();
  auto select_aa = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple_a, tuple_a));
  auto select_ab = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple_a, tuple_b));
  auto select_cd = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple_c, tuple_d));
  auto select_abcd = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, select_ab, select_cd));

  module_->AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Sanity check dataflow before changing the graph and updating.
  EXPECT_THAT(HloValuesAt(select_aa, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(a)));
  EXPECT_THAT(HloValuesAt(select_ab, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(a),
                                   analysis.GetValueDefinedAt(b)));
  EXPECT_THAT(HloValuesAt(select_cd, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(c),
                                   analysis.GetValueDefinedAt(d)));
  EXPECT_THAT(HloValuesAt(select_abcd, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(a),
                                   analysis.GetValueDefinedAt(b),
                                   analysis.GetValueDefinedAt(c),
                                   analysis.GetValueDefinedAt(d)));
  EXPECT_TRUE(analysis.GetValueDefinedAt(a).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(b).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(c).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(d).live_out_of_module());

  // Set the rhs of 'select_aa' to be 'd'.
  TF_ASSERT_OK(select_aa->ReplaceOperandWith(2, tuple_d));
  analysis.UpdateAfterChangingOperand(select_aa, /*old_operand=*/tuple_a,
                                      /*new_operand=*/tuple_d);

  // Verify that the dataflow is correct.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  EXPECT_THAT(HloValuesAt(select_aa, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(a),
                                   analysis.GetValueDefinedAt(d)));

  // Set the lhs of 'select_cd' to be 'a'.
  TF_ASSERT_OK(select_cd->ReplaceOperandWith(1, tuple_a));
  analysis.UpdateAfterChangingOperand(select_cd, /*old_operand=*/tuple_c,
                                      /*new_operand=*/tuple_a);

  // Verify that the dataflow is correct.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  EXPECT_THAT(HloValuesAt(select_cd, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(a),
                                   analysis.GetValueDefinedAt(d)));
  EXPECT_THAT(HloValuesAt(select_abcd, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(a),
                                   analysis.GetValueDefinedAt(b),
                                   analysis.GetValueDefinedAt(d)));
  EXPECT_TRUE(analysis.GetValueDefinedAt(a).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(b).live_out_of_module());
  EXPECT_FALSE(analysis.GetValueDefinedAt(c).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(d).live_out_of_module());

  // After the updates, verify that the dataflow is correct.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());
}

INSTANTIATE_TEST_CASE_P(HloDataflowAnalysisInstantiation,
                        HloDataflowAnalysisTest,
                        ::testing::Values(false, true));

}  // namespace
}  // namespace xla
