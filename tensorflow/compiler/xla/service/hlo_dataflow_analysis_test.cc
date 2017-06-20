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
  HloDataflowAnalysisTest() : module_(TestName()) {}

  // Run dataflow analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  const HloDataflowAnalysis& RunAnalysis(bool ssa_form,
                                         bool bitcast_defines_value = false) {
    analysis_ =
        HloDataflowAnalysis::Run(&module_, ssa_form, bitcast_defines_value)
            .ConsumeValueOrDie();
    return *analysis_;
  }

  // Return a vector of the HloValues at the given program location.
  std::vector<HloValue> HloValuesAt(const HloInstruction* instruction,
                                    const ShapeIndex& index = {}) {
    CHECK(analysis_ != nullptr);
    std::vector<HloValue> values;
    for (HloValue::Id value_id :
         analysis_->GetValueSet(instruction, index).value_ids()) {
      values.push_back(analysis_->GetValue(value_id));
    }
    return values;
  }

  HloModule module_;
  std::unique_ptr<HloDataflowAnalysis> analysis_;

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
};

TEST_P(HloDataflowAnalysisTest, BinaryOperation) {
  // Test the dataflow for a simple binary operation (Add).
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, constant1, constant2));
  module_.AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Each instruction should define a single value.
  EXPECT_EQ(analysis.values().size(), 3);
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(add));

  // Verify the locations of the values. These locations are all trivial because
  // there are no instructions which forward values.
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).locations(),
              UnorderedElementsAre(HloLocation{constant1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).locations(),
              UnorderedElementsAre(HloLocation{constant2, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(add).locations(),
              UnorderedElementsAre(HloLocation{add, {}}));

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
  module_.AddEntryComputation(builder.Build());

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

  // Verify the locations of the values.
  EXPECT_THAT(
      analysis.GetValueDefinedAt(param0).locations(),
      UnorderedElementsAre(HloLocation{param0, {}}, HloLocation{tuple, {0}},
                           HloLocation{gte0, {}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(param1).locations(),
      UnorderedElementsAre(HloLocation{param1, {}}, HloLocation{tuple, {1}},
                           HloLocation{gte1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple).locations(),
              UnorderedElementsAre(HloLocation{tuple, {}}));

  // Verify uses. Of interest is that a GetTupleElement instruction is only a
  // use of the top-level value in the tuple operand.
  EXPECT_THAT(analysis.GetValueDefinedAt(param0).uses(),
              UnorderedElementsAre(HloUse{tuple, 0, {}}, HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(param1).uses(),
              UnorderedElementsAre(HloUse{tuple, 1, {}}, HloUse{add, 1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple, /*index=*/{}).uses(),
              UnorderedElementsAre(HloUse{gte0, 0, {}}, HloUse{gte1, 0, {}}));
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, NestedTuple) {
  // Verify the dataflow through a nested tuple of the following form for two
  // constants %constant1 and %constant2:
  //
  // %nested_tuple = {{%constant1, %constant2},
  //                  {%constant1, %constant2},
  //                  %constant1}
  //
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto nested_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({tuple, tuple, constant1}));
  auto gte_tuple = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple->shape(), nested_tuple, 1));
  auto gte_out = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, gte_tuple, 0));
  module_.AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 4);

  // Verify locations and uses.
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant1).locations(),
      UnorderedElementsAre(
          HloLocation{constant1, {}}, HloLocation{tuple, {0}},
          HloLocation{nested_tuple, {0, 0}}, HloLocation{nested_tuple, {1, 0}},
          HloLocation{nested_tuple, {2}}, HloLocation{gte_tuple, {0}},
          HloLocation{gte_out, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
              UnorderedElementsAre(
                  HloUse{tuple, 0, {}}, HloUse{nested_tuple, 0, {0}},
                  HloUse{nested_tuple, 1, {0}}, HloUse{nested_tuple, 2, {}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant2).uses(),
      UnorderedElementsAre(HloUse{tuple, 1, {}}, HloUse{nested_tuple, 0, {1}},
                           HloUse{nested_tuple, 1, {1}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple, /*index=*/{}).uses(),
              UnorderedElementsAre(HloUse{nested_tuple, 0, {}},
                                   HloUse{nested_tuple, 1, {}},
                                   HloUse{gte_out, 0, {}}));
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
      module_.AddEmbeddedComputation(subbuilder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto call = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  module_.AddEntryComputation(builder.Build());

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
              UnorderedElementsAre(HloUse{add, 0, {}}, HloUse{call, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}, HloUse{call, 1, {}}));

  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
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
      module_.AddEmbeddedComputation(subbuilder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto call1 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  auto call2 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kSubtract, call1, call2));
  module_.AddEntryComputation(builder.Build());

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
              UnorderedElementsAre(HloUse{add, 0, {}}, HloUse{call1, 0, {}},
                                   HloUse{call2, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).uses(),
              UnorderedElementsAre(HloUse{add, 1, {}}, HloUse{call1, 1, {}},
                                   HloUse{call2, 1, {}}));
  // The Add from the subcomputation is used as both operands of the Subtract.
  EXPECT_THAT(analysis.GetValueDefinedAt(add).uses(),
              UnorderedElementsAre(HloUse{sub, 0, {}}, HloUse{sub, 1, {}}));

  EXPECT_FALSE(analysis.GetValueDefinedAt(add).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(sub).live_out_of_module());
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
      module_.AddEmbeddedComputation(subbuilder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto call1 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  auto call2 = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {call1, constant2}, called_computation));
  module_.AddEntryComputation(builder.Build());

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
      module_.AddEmbeddedComputation(inner_builder.Build());

  auto outer_builder = HloComputation::Builder("OuterComputation");
  auto outer_param0 = outer_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto outer_param1 = outer_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  // Swizzle parameters.
  auto nested_call = outer_builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {outer_param1, outer_param0}, inner_computation));
  HloComputation* outer_computation =
      module_.AddEmbeddedComputation(outer_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto call = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, outer_computation));
  module_.AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Only three values should be defined. Most instructions just pass through
  // their operand values.
  EXPECT_EQ(analysis.values().size(), 3);

  // Verify that the uses of the constants are properly swizzled by parameter
  // permutation in nested_call.
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant1).uses(),
      UnorderedElementsAre(HloUse{call, 0, {}}, HloUse{nested_call, 1, {}},
                           HloUse{add, 1, {}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant2).uses(),
      UnorderedElementsAre(HloUse{call, 1, {}}, HloUse{nested_call, 0, {}},
                           HloUse{add, 0, {}}));

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
  auto body_tuple = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_0, add}));
  HloComputation* body = module_.AddEmbeddedComputation(body_builder.Build());

  // Condition computation trivially returns a constant "false".
  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_.AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_.AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  if (ssa_form) {
    // Element 0 of the tuple passed through the body so no phi value is
    // defined.
    EXPECT_FALSE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{0}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(body_param, /*index=*/{0}));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{0}));

    // Element 1 of the tuple should be a phi value.
    EXPECT_TRUE(analysis.ValueIsDefinedAt(xla_while, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(body_param, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(body_param, /*index=*/{1}).is_phi());
    EXPECT_TRUE(analysis.ValueIsDefinedAt(cond_param, /*index=*/{1}));
    EXPECT_TRUE(analysis.GetValueDefinedAt(cond_param, /*index=*/{1}).is_phi());

    EXPECT_THAT(analysis.GetValueDefinedAt(constant1).uses(),
                UnorderedElementsAre(HloUse{add, 0, {}}, HloUse{tuple, 0, {}},
                                     HloUse{xla_while, 0, {0}},
                                     HloUse{body_tuple, 0, {}}));

    // Constant1 passes through the body and out of the module.
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1})
                    .live_out_of_module());
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
  HloComputation* body = module_.AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_.AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  auto xla_while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, xla_while0));
  auto xla_while2 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, xla_while1));
  module_.AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Element 0 is passed through all the while instructions and out of the
  // module..
  EXPECT_EQ(analysis.GetUniqueValueAt(xla_while0, /*index=*/{0}),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(xla_while1, /*index=*/{0}),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(xla_while2, /*index=*/{0}),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_.AddEmbeddedComputation(cond_builder.Build());

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
      module_.AddEmbeddedComputation(inner_builder.Build());

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
      module_.AddEmbeddedComputation(outer_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto entry_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, outer_body, tuple));
  module_.AddEntryComputation(builder.Build());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_THAT(HloValuesAt(inner_param, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetValueDefinedAt(negate)));
  if (ssa_form) {
    EXPECT_TRUE(analysis.ValueIsDefinedAt(inner_param, /*index=*/{1}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(inner_param, /*index=*/{1}).is_phi());

    // Element 0 of the nested while is %negate.
    EXPECT_FALSE(analysis.ValueIsDefinedAt(nested_while, /*index=*/{0}));
    EXPECT_THAT(HloValuesAt(inner_param, /*index=*/{0}),
                UnorderedElementsAre(analysis.GetValueDefinedAt(negate)));
    // Element 1 is a phi value (join of %add and %constant2).
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
  HloComputation* body = module_.AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_.AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_.AddEntryComputation(builder.Build());

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      scalar_shape_, HloOpcode::kSelect, pred, constant1, constant2));

  module_.AddEntryComputation(builder.Build());

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(4.0)));
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

  module_.AddEntryComputation(builder.Build());

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
      analysis.GetValueDefinedAt(constant1).uses(),
      UnorderedElementsAre(HloUse{tuple1, 0, {}}, HloUse{select11, 1, {0}},
                           HloUse{select11, 2, {0}}, HloUse{select12, 1, {0}},
                           HloUse{select1234, 1, {0}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant2).uses(),
      UnorderedElementsAre(HloUse{tuple2, 0, {}}, HloUse{select12, 2, {0}},
                           HloUse{select1234, 1, {0}}));
}

TEST_P(HloDataflowAnalysisTest, NestedTupleSelect) {
  // Test kSelect of a nested tuple.
  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(4.0)));
  auto constant5 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0)));
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

  module_.AddEntryComputation(builder.Build());

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
  HloComputation* body = module_.AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_.AddEmbeddedComputation(cond_builder.Build());

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
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

  module_.AddEntryComputation(builder.Build());

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kBitcast, constant));

  module_.AddEntryComputation(builder.Build());

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
  module_.AddEntryComputation(builder.Build());

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

INSTANTIATE_TEST_CASE_P(HloDataflowAnalysisInstantiation,
                        HloDataflowAnalysisTest,
                        ::testing::Values(false, true));

}  // namespace
}  // namespace xla
