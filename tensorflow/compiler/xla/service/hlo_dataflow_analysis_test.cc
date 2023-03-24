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

#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/async_op_canonicalizer.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// Test is parameterized on a bool which is whether the dataflow analysis is
// performed with SSA form.
class HloDataflowAnalysisTest : public HloTestBase,
                                public ::testing::WithParamInterface<bool> {
 protected:
  HloDataflowAnalysisTest() : module_(CreateNewVerifiedModule()) {}

  // Run dataflow analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  const HloDataflowAnalysis& RunAnalysis(bool ssa_form,
                                         bool bitcast_defines_value = false,
                                         bool run_dce = true) {
    AsyncOpCanonicalizer async_op_canonicalizer;
    EXPECT_TRUE(async_op_canonicalizer.Run(module_.get()).ok());
    if (run_dce) {
      HloDCE dce;
      EXPECT_TRUE(dce.Run(module_.get()).ok());
    }
    FlattenCallGraph flatten;
    EXPECT_TRUE(flatten.Run(module_.get()).ok());
    analysis_ =
        HloDataflowAnalysis::Run(*module_, ssa_form, bitcast_defines_value)
            .value();
    return *analysis_;
  }

  // Return a vector of the HloValues at the given program position.
  const std::vector<const HloValue*>& HloValuesAt(
      const HloInstruction* instruction, const ShapeIndex& index = {}) {
    CHECK(analysis_ != nullptr);
    return analysis_->GetValueSet(instruction, index).values();
  }

  // Returns true if the top-level values for instructions 'a' and 'b' may
  // interfere. Precondition: 'a' and 'b' define array-shaped values.
  bool InstructionsMayInterfere(const HloOrdering& ordering,
                                const HloInstruction* a,
                                const HloInstruction* b) {
    EXPECT_FALSE(a->shape().IsTuple());
    EXPECT_FALSE(b->shape().IsTuple());
    return ordering.MayInterfere(analysis_->GetValueDefinedAt(a),
                                 analysis_->GetValueDefinedAt(b), *analysis_);
  }

  std::unique_ptr<HloComputation> CreateR0F32UnaryOpComputation(
      HloOpcode opcode) {
    HloComputation::Builder builder(
        absl::StrCat(TestName(), ".", HloOpcodeString(opcode)));
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(scalar_shape_, opcode, param0));
    return builder.Build();
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloDataflowAnalysis> analysis_;

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
  const Shape vector_shape_ = ShapeUtil::MakeShape(F32, {42});
  const Shape tuple_shape_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});
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
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

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
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).GetUses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).GetUses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).GetUses().empty());

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
  SCOPED_TRACE(module_->ToString());

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
  EXPECT_THAT(analysis.GetValueDefinedAt(param0).GetUses(),
              UnorderedElementsAre(HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(param1).GetUses(),
              UnorderedElementsAre(HloUse{add, 1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple, /*index=*/{}).GetUses(),
              UnorderedElementsAre(HloUse{gte0, 0, {}}, HloUse{gte1, 0, {}}));
  EXPECT_TRUE(analysis.GetValueDefinedAt(add).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, NestedTuple) {
  // Verify the dataflow through a nested tuple.
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
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

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
  // Constant values should have only a single use, which is the root of the
  // computation.
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1, /*index=*/{}).GetUses(),
              UnorderedElementsAre(HloUse{gte_out, 0, {0}}));
  EXPECT_TRUE(analysis.GetValueDefinedAt(constant2).GetUses().empty());

  // The top-level tuple values are used in GTE instructions.
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple, /*index=*/{}).GetUses(),
              UnorderedElementsAre(HloUse{gte_out, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(nested_tuple, /*index=*/{}).GetUses(),
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto call = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, called_computation));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

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

  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).GetUses(),
              UnorderedElementsAre(HloUse{call, 0, {}}, HloUse{add, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).GetUses(),
              UnorderedElementsAre(HloUse{call, 1, {}}, HloUse{add, 1, {}}));

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
  auto nested_call = outer_builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {outer_param1, outer_param0}, inner_computation));
  HloComputation* outer_computation =
      module_->AddEmbeddedComputation(outer_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto call = builder.AddInstruction(HloInstruction::CreateCall(
      scalar_shape_, {constant1, constant2}, outer_computation));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Only three values should be defined. Most instructions just pass through
  // their operand values.
  EXPECT_EQ(analysis.values().size(), 3);

  // Verify that the uses of the constants are properly swizzled by parameter
  // permutation in nested_call.
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant1).GetUses(),
      UnorderedElementsAre(HloUse{call, 0, {}}, HloUse{nested_call, 1, {}},
                           HloUse{add, 1, {}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(constant2).GetUses(),
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
  auto body_root = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_0, add}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  // Condition computation trivially returns a constant "false".
  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_FALSE(analysis.GetValueDefinedAt(cond_constant).live_out_of_module());

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

    EXPECT_THAT(
        analysis.GetValueDefinedAt(constant1).GetUses(),
        UnorderedElementsAre(HloUse{add, 0, {}}, HloUse{body_root, 0, {}},
                             HloUse{xla_while, 0, {0}}));

    // Constant1 passes through the body and out of the module.
    EXPECT_TRUE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
    EXPECT_TRUE(analysis.GetValueDefinedAt(xla_while, /*index=*/{1})
                    .live_out_of_module());

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

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
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

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

TEST_P(HloDataflowAnalysisTest, MultiLevelNestedWhile) {
  // Test nested while instructions. The level0 body (most inner while) and
  // level1 body pass through the parameter, while level2 (most outer while)
  // modifies it.
  //
  // level0_body((F32[]) %tuple_param):
  //   return Tuple(%tuple_param{0})
  //
  // level1_body((F32[]) %tuple_param):
  //   return While(%tuple_param{0}), body=level0
  //
  // level2_body((F32[]) %tuple_param):
  //   while = While(%tuple_param{0}), body=level1
  //.  return negate(%while{0})
  //
  // entry:
  //   %constant = Constant(1.0)
  //   %tuple = Tuple(%constant)
  //   return While(%tuple), body=level2
  //
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape_});
  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  // level 0 passes transparently through the body.
  auto level0_builder = HloComputation::Builder("level0_body");
  auto level0_param = level0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto level0_element_0 = level0_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, level0_param, 0));
  auto level0_root = level0_builder.AddInstruction(
      HloInstruction::CreateTuple({level0_element_0}));
  HloComputation* level0_body =
      module_->AddEmbeddedComputation(level0_builder.Build());

  // Element 1 passes transparently through the body.
  auto level1_builder = HloComputation::Builder("level1_body");
  auto level1_param = level1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto level1_root = level1_builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, condition, level0_body, level1_param));
  HloComputation* level1_body =
      module_->AddEmbeddedComputation(level1_builder.Build());

  // Element 1 passes transparently through the body.
  auto level2_builder = HloComputation::Builder("level2_body");
  auto level2_param = level2_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto level2_while = level2_builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, condition, level1_body, level2_param));
  auto level2_element_0 = level2_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, level2_while, 0));
  auto negate = level2_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, level2_element_0));
  level2_builder.AddInstruction(HloInstruction::CreateTuple({negate}));
  HloComputation* level2_body =
      module_->AddEmbeddedComputation(level2_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({constant1}));
  builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, level2_body, tuple));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  if (!ssa_form) {
    return;
  }
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  // Phi node on inner parameters and roots should have been eliminated.
  EXPECT_FALSE(analysis.ValueIsDefinedAt(level1_param, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(level0_param, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(level1_root, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(level0_root, /*index=*/{0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(level2_param, /*index=*/{0}));
  EXPECT_EQ(HloValuesAt(level1_param, /*index=*/{0}),
            HloValuesAt(level2_param, /*index=*/{0}));
  EXPECT_EQ(HloValuesAt(level0_param, /*index=*/{0}),
            HloValuesAt(level2_param, /*index=*/{0}));
  EXPECT_EQ(HloValuesAt(level1_root, /*index=*/{0}),
            HloValuesAt(level2_param, /*index=*/{0}));
  EXPECT_EQ(HloValuesAt(level0_root, /*index=*/{0}),
            HloValuesAt(level2_param, /*index=*/{0}));
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto entry_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, outer_body, tuple));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_THAT(HloValuesAt(inner_param, /*index=*/{0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(negate)));
  if (ssa_form) {
    EXPECT_TRUE(analysis.ValueIsDefinedAt(inner_param, /*index=*/{1}));
    EXPECT_TRUE(
        analysis.GetValueDefinedAt(inner_param, /*index=*/{1}).is_phi());

    // Element 0 of the nested while is %negate.
    EXPECT_FALSE(analysis.ValueIsDefinedAt(nested_while, /*index=*/{0}));
    EXPECT_THAT(HloValuesAt(inner_param, /*index=*/{0}),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(negate)));
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
                UnorderedElementsAre(&analysis.GetValueDefinedAt(add),
                                     &analysis.GetValueDefinedAt(constant2)));

    EXPECT_THAT(HloValuesAt(nested_while, /*index=*/{0}),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(negate)));
    EXPECT_THAT(HloValuesAt(nested_while, /*index=*/{1}),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(add),
                                     &analysis.GetValueDefinedAt(constant2)));

    EXPECT_THAT(HloValuesAt(entry_while, /*index=*/{0}),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(negate),
                                     &analysis.GetValueDefinedAt(constant1)));
    EXPECT_THAT(HloValuesAt(entry_while, /*index=*/{1}),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(add),
                                     &analysis.GetValueDefinedAt(constant2)));
  }
}

TEST_P(HloDataflowAnalysisTest, SwizzlingWhileSharedInput) {
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
  //   %tuple = Tuple(%constant1, %constant1)
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
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant1}));
  builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);
  EXPECT_FALSE(analysis.ValueIsDefinedAt(body_param, /*index=*/{0}));
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

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
                UnorderedElementsAre(&analysis.GetValueDefinedAt(constant1),
                                     &analysis.GetValueDefinedAt(constant2)));
    EXPECT_THAT(HloValuesAt(xla_while, /*index=*/{1}),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(constant1),
                                     &analysis.GetValueDefinedAt(constant2)));
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

  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(select));
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant1).live_out_of_module());
  EXPECT_FALSE(analysis.GetValueDefinedAt(constant2).live_out_of_module());
  EXPECT_TRUE(analysis.GetValueDefinedAt(select).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, BitcastDefinesValue) {
  // Test the bitcast_defines_value flag to the dataflow analysis.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(scalar_shape_, constant));

  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

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
  SCOPED_TRACE(module_->ToString());

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
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param0)));
  EXPECT_THAT(HloValuesAt(copy, /*index=*/{1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param1)));
  EXPECT_TRUE(
      analysis.GetValueDefinedAt(copy, /*index=*/{}).live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, OptimizationBarrier) {
  // Test that an optimization barrier is a nop.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({param0, param1}));
  auto barrier = builder.AddInstruction(HloInstruction::CreateUnary(
      tuple->shape(), HloOpcode::kOptimizationBarrier, tuple));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 3);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(param0));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(param1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(tuple, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(tuple, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(tuple, /*index=*/{1}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(barrier, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(barrier, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(barrier, /*index=*/{1}));

  EXPECT_THAT(HloValuesAt(barrier, /*index=*/{0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param0)));
  EXPECT_THAT(HloValuesAt(barrier, /*index=*/{1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param1)));
}

TEST_P(HloDataflowAnalysisTest, CopyStartAndCopyDone) {
  // Test that a CopyDone forwards its operand tuple element at {0} to the
  // output.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto copy_start = builder.AddInstruction(HloInstruction::CreateCopyStart(
      ShapeUtil::MakeTupleShape({constant->shape(), constant->shape(),
                                 ShapeUtil::MakeShape(U32, {})}),
      constant));
  auto copy_done = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopyDone, copy_start));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 4);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(copy_start, /*index=*/{}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(copy_start, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(copy_start, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(copy_start, /*index=*/{2}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(copy_done, /*index=*/{}));
  EXPECT_THAT(
      HloValuesAt(copy_done, /*index=*/{}),
      UnorderedElementsAre(&analysis.GetValueDefinedAt(copy_start, {0})));
  EXPECT_TRUE(analysis.GetValueDefinedAt(copy_start, /*index=*/{0})
                  .live_out_of_module());
}

TEST_P(HloDataflowAnalysisTest, AsyncOps) {
  std::string hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), custom_call_target="foo"
    async-update = ((f32[2,3]), f32[2,3], u32[]) custom-call-update(async-start), custom_call_target="foo"
    ROOT async-done = f32[2,3] custom-call-done(async-update), custom_call_target="foo"
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  const HloInstruction* param =
      module_->entry_computation()->parameter_instruction(0);
  const HloInstruction* async_start =
      FindInstruction(module_.get(), "async-start");
  const HloInstruction* async_update =
      FindInstruction(module_.get(), "async-update");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");
  const HloInstruction* async_wrapped_instruction =
      async_start->async_wrapped_instruction();

  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_start, /*index=*/{0, 0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_start, /*index=*/{1}));
  EXPECT_THAT(HloValuesAt(async_start, {1}),
              UnorderedElementsAre(
                  &analysis.GetValueDefinedAt(async_wrapped_instruction, {})));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, /*index=*/{2}));
  EXPECT_THAT(HloValuesAt(async_start, /*index=*/{0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param, {})));
  EXPECT_TRUE(analysis.GetValueDefinedAt(async_wrapped_instruction, {})
                  .live_out_of_module());

  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_update, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_update, /*index=*/{0, 0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_update, /*index=*/{1}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_update, /*index=*/{2}));
  EXPECT_THAT(HloValuesAt(async_update, /*index=*/{0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param, {})));
  EXPECT_THAT(HloValuesAt(async_update, /*index=*/{1}),
              UnorderedElementsAre(
                  &analysis.GetValueDefinedAt(async_wrapped_instruction, {})));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_done, /*index=*/{}));
  EXPECT_THAT(HloValuesAt(async_done, /*index=*/{}),
              UnorderedElementsAre(
                  &analysis.GetValueDefinedAt(async_wrapped_instruction, {})));
}

TEST_P(HloDataflowAnalysisTest, AsyncCall) {
  std::string hlo_str = R"(
HloModule AsyncCall

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  %negate_0 = f32[4096]{0} negate(f32[4096]{0} %param_0)
  %negate_1 = f32[4096]{0} negate(f32[4096]{0} %param_1)
  ROOT %result.1 = f32[4096]{0} add(f32[4096]{0} %negate_0, f32[4096]{0} %negate_1)
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %async-start = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-start(f32[4096]{0} %a, f32[4096]{0} %b), to_apply=%called_computation
  %negate_2 = f32[4096]{0} negate(f32[4096]{0} %a)
  %async-update = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-update(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-start), to_apply=%called_computation
  %negate_3 = f32[4096]{0} negate(f32[4096]{0} %b)
  %add_0 = f32[4096]{0} add(f32[4096]{0} %negate_2, f32[4096]{0} %negate_3)
  %async-done = f32[4096]{0} call-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-update), to_apply=%called_computation
  ROOT %add_1 = f32[4096]{0} add(f32[4096]{0} %add_0, f32[4096]{0} %async-done)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  const HloInstruction* a = FindInstruction(module_.get(), "a");
  const HloInstruction* b = FindInstruction(module_.get(), "b");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  // For each of the async operations, ensure the called computation
  // parameter/root instructions have the same HloValues as the callees.
  for (std::string async_name : {"async-start", "async-update", "async-done"}) {
    const HloInstruction* async_op = FindInstruction(module_.get(), async_name);
    const HloComputation* called_computation =
        async_op->async_wrapped_instruction()->called_computations()[0];
    const HloInstruction* parameter0 =
        called_computation->parameter_instruction(0);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(parameter0));
    EXPECT_THAT(HloValuesAt(parameter0),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(a)));
    const HloInstruction* parameter1 =
        called_computation->parameter_instruction(1);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(parameter1));
    EXPECT_THAT(HloValuesAt(parameter1),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(b)));
    const HloInstruction* root = called_computation->root_instruction();
    EXPECT_TRUE(analysis.ValueIsDefinedAt(root));
    EXPECT_THAT(HloValuesAt(async_done),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(root)));
  }
}

TEST_P(HloDataflowAnalysisTest, SendAndSendDone) {
  // Test that a Send forwards its operand to the output tuple at {0}.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(
      HloInstruction::CreateSend(param, token, /*channel_id=*/0));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(send));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 6);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(param));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(send, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(send, /*index=*/{0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(send, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(send, /*index=*/{2}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(send_done));
  EXPECT_THAT(HloValuesAt(send, /*index=*/{0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param)));
}

TEST_P(HloDataflowAnalysisTest, SetDimensionSizeForwardsValue) {
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vector_shape_, "param"));
  auto size = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(3)));
  auto sds = builder.AddInstruction(
      HloInstruction::CreateSetDimensionSize(vector_shape_, param, size, 0));

  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  {
    const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);
    EXPECT_EQ(analysis.values().size(), 2);

    EXPECT_TRUE(analysis.ValueIsDefinedAt(param));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(sds));
    EXPECT_TRUE(analysis.GetValueDefinedAt(param).live_out_of_module());
  }
}

TEST_P(HloDataflowAnalysisTest, RecvAndRecvDone) {
  // Test that a RecvDone forwards its operand tuple element at {0} to element
  // {0} of the output.
  auto builder = HloComputation::Builder(TestName());
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto recv = builder.AddInstruction(
      HloInstruction::CreateRecv(scalar_shape_, token, /*channel_id=*/0));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(recv));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  EXPECT_EQ(analysis.values().size(), 7);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(recv, /*index=*/{}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(recv, /*index=*/{0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(recv, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(recv, /*index=*/{2}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(recv_done, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(recv_done, /*index=*/{0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(recv_done, /*index=*/{1}));
  EXPECT_THAT(HloValuesAt(recv_done, /*index=*/{0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(recv, {0})));
  EXPECT_TRUE(
      analysis.GetValueDefinedAt(recv, /*index=*/{0}).live_out_of_module());
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
  SCOPED_TRACE(module_->ToString());
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
  SCOPED_TRACE(module_->ToString());
  RunAnalysis(GetParam());

  HloSchedule schedule(module_.get());
  schedule.set_sequence(entry, {param0, negate, param1, exp, add});
  TF_ASSERT_OK(schedule.Verify());
  SequentialHloOrdering ordering(schedule);

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto exp = body_builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kExp, constant));
  auto add = body_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, exp, body_param));
  auto dead_constant = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto dead_negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, dead_constant));
  HloComputation* body = module_->AddEmbeddedComputation(
      body_builder.Build(/*root_instruction=*/add));

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "cond_param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module_->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape_, condition, body, param));

  auto entry = module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());
  bool ssa_form = GetParam();
  RunAnalysis(ssa_form, /*bitcast_defines_value=*/false,
              /*run_dce=*/false);

  HloSchedule schedule(module_.get());
  schedule.set_sequence(entry, {param, xla_while});
  schedule.set_sequence(condition, {cond_param, cond_constant});
  // Construct the order such that 'constant' and its use 'exp' are before
  // body_param.
  schedule.set_sequence(
      body, {constant, exp, body_param, add, dead_constant, dead_negate});
  TF_ASSERT_OK(schedule.Verify());

  SequentialHloOrdering ordering(schedule);

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
  SCOPED_TRACE(module_->ToString());
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
  SCOPED_TRACE(module_->ToString());
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
  SCOPED_TRACE(module_->ToString());
  RunAnalysis(GetParam());

  HloSchedule schedule(module_.get());
  schedule.set_sequence(entry, {param, negate, exp, add});
  TF_ASSERT_OK(schedule.Verify());
  SequentialHloOrdering ordering(schedule);

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
  SCOPED_TRACE(module_->ToString());
  RunAnalysis(GetParam());

  DependencyHloOrdering ordering(module_.get());

  // Exp only use is the call so it should not interfere with values inside
  // the embedded computation.
  EXPECT_FALSE(InstructionsMayInterfere(ordering, exp, embedded_log));

  // Negate is live across the call and should interfere with values in the
  // embedded computation
  EXPECT_TRUE(InstructionsMayInterfere(ordering, negate, embedded_log));
}

TEST_P(HloDataflowAnalysisTest, GetFlattenedValueSet) {
  const char* hlo_text = R"(
HloModule test_aliasing_module

ENTRY root {
  param = s32[1000] parameter(0)
  p0 = s32[1000] copy(param)
  p1 = s32[1000] copy(param)
  ROOT t = (s32[1000], s32[1000]) tuple(p0, p1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_text));
  auto entry = module_->entry_computation();
  entry->GetInstructionWithName("t");
  auto& dataflow_analysis = RunAnalysis(GetParam());
  auto set = dataflow_analysis.GetFlattenedValueSet(
      entry->GetInstructionWithName("t"));
  EXPECT_EQ(set.values().size(), 3);
}

TEST_P(HloDataflowAnalysisTest, ConditionalWithIdentity) {
  // Test conditional with identity computations in both true and false cases.
  //
  // true_computation(F32[] %true_param):
  //   return %true_param
  //
  // false_computation(F32[] %false_param):
  //   return %false_param
  //
  // entry:
  //   %pred = Constant(true)
  //   %constant1 = Constant(56.0)
  //   %constant2 = Constant(12.0)
  //   return Conditional(%pred, %constant1, true_computation,
  //                      %constant2, false_computation)

  auto true_builder = HloComputation::Builder(TestName() + "_true");
  auto true_param = true_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "true_param"));
  HloComputation* true_computation =
      module_->AddEmbeddedComputation(true_builder.Build());

  auto false_builder = HloComputation::Builder(TestName() + "_false");
  auto false_param = false_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "false_param"));
  HloComputation* false_computation =
      module_->AddEmbeddedComputation(false_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(56.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12.0f)));
  auto conditional = builder.AddInstruction(HloInstruction::CreateConditional(
      scalar_shape_, pred, constant1, true_computation, constant2,
      false_computation));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  const HloDataflowAnalysis& analysis = RunAnalysis(GetParam());

  EXPECT_TRUE(analysis.ValueIsDefinedAt(pred));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(true_param));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(false_param));

  EXPECT_EQ(analysis.GetUniqueValueAt(true_param),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(false_param),
            analysis.GetValueDefinedAt(constant2));

  EXPECT_THAT(analysis.GetValueDefinedAt(pred).GetUses(),
              ElementsAre(HloUse{conditional, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).GetUses(),
              ElementsAre(HloUse{conditional, 1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).GetUses(),
              ElementsAre(HloUse{conditional, 2, {}}));

  bool ssa_form = GetParam();
  if (ssa_form) {
    EXPECT_EQ(analysis.values().size(), 4);
    EXPECT_TRUE(analysis.ValueIsDefinedAt(conditional));
  } else {
    EXPECT_EQ(analysis.values().size(), 3);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(conditional));
    EXPECT_THAT(HloValuesAt(conditional),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(constant1),
                                     &analysis.GetValueDefinedAt(constant2)));
  }
}

TEST_P(HloDataflowAnalysisTest, ConditionalTakingTupleOperand) {
  // Test conditional with true and false computations taking a tuple operand.
  //
  // true_computation((F32[], F32[]) %true_param):
  //   %true_x = GetTupleElement(%true_param, 0)
  //   %true_y = GetTupleElement(%true_param, 1)
  //   return Add(%true_x, %true_y)
  //
  // false_computation((F32[], F32[]) %false_param):
  //   %false_x = GetTupleElement(%false_param, 0)
  //   %false_y = GetTupleElement(%false_param, 1)
  //   return Subtract(%false_x, %false_y)
  //
  // entry:
  //   %pred = Constant(true)
  //   %constant1 = Constant(56.0)
  //   %constant2 = Constant(12.0)
  //   %tuple_operand = Tuple(%constant1, %constant2)
  //   return Conditional(%pred, %tuple_operand, true_computation,
  //                      %tuple_operand, false_computation)

  auto true_builder = HloComputation::Builder(TestName() + "_true");
  auto true_param = true_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape_, "true_param"));
  auto true_x = true_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, true_param, 0));
  auto true_y = true_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, true_param, 1));
  auto add = true_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, true_x, true_y));
  HloComputation* true_computation =
      module_->AddEmbeddedComputation(true_builder.Build());

  auto false_builder = HloComputation::Builder(TestName() + "_false");
  auto false_param = false_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape_, "false_param"));
  auto false_x = false_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, false_param, 0));
  auto false_y = false_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, false_param, 1));
  auto sub = false_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kSubtract, false_x, false_y));
  HloComputation* false_computation =
      module_->AddEmbeddedComputation(false_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(56.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12.0f)));
  auto tuple_operand = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto conditional = builder.AddInstruction(HloInstruction::CreateConditional(
      scalar_shape_, pred, tuple_operand, true_computation, tuple_operand,
      false_computation));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  const HloDataflowAnalysis& analysis = RunAnalysis(GetParam());

  EXPECT_TRUE(analysis.ValueIsDefinedAt(pred));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(tuple_operand));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(add));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(sub));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(true_param));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(false_param));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(true_x));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(true_y));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(false_x));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(false_y));

  EXPECT_EQ(analysis.GetUniqueValueAt(true_param),
            analysis.GetValueDefinedAt(tuple_operand));
  EXPECT_EQ(analysis.GetUniqueValueAt(false_param),
            analysis.GetValueDefinedAt(tuple_operand));
  EXPECT_EQ(analysis.GetUniqueValueAt(true_x),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(true_y),
            analysis.GetValueDefinedAt(constant2));
  EXPECT_EQ(analysis.GetUniqueValueAt(false_x),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(false_y),
            analysis.GetValueDefinedAt(constant2));

  EXPECT_THAT(analysis.GetValueDefinedAt(pred).GetUses(),
              ElementsAre(HloUse{conditional, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant1).GetUses(),
              UnorderedElementsAre(HloUse{conditional, 1, {0}},
                                   HloUse{conditional, 2, {0}},
                                   HloUse{add, 0, {}}, HloUse{sub, 0, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(constant2).GetUses(),
              UnorderedElementsAre(HloUse{conditional, 1, {1}},
                                   HloUse{conditional, 2, {1}},
                                   HloUse{add, 1, {}}, HloUse{sub, 1, {}}));
  EXPECT_THAT(analysis.GetValueDefinedAt(tuple_operand).GetUses(),
              UnorderedElementsAre(
                  HloUse{conditional, 1, {}}, HloUse{conditional, 2, {}},
                  HloUse{true_x, 0, {}}, HloUse{true_y, 0, {}},
                  HloUse{false_x, 0, {}}, HloUse{false_y, 0, {}}));

  bool ssa_form = GetParam();
  if (ssa_form) {
    EXPECT_EQ(analysis.values().size(), 7);
    EXPECT_TRUE(analysis.ValueIsDefinedAt(conditional));
  } else {
    EXPECT_EQ(analysis.values().size(), 6);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(conditional));
    EXPECT_THAT(HloValuesAt(conditional),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(add),
                                     &analysis.GetValueDefinedAt(sub)));
  }
}

TEST_P(HloDataflowAnalysisTest, NestedConditionals) {
  // computation1(F32[] %param1):
  //   %ceil = Ceil(%param1)
  //   return %ceil
  //
  // computation2(F32[] %param2):
  //   %floor = Floor(%param2)
  //   return %floor
  //
  // computation3(F32[] %param3):
  //   %negate = Negate(%param3)
  //   return %negate
  //
  // inner_conditional((PRED, F32[], F32[]) %param_cond):
  //   %pred_cond = GetTupleElement(%param_cond, 0)
  //   %true_operand_cond = GetTupleElement(%param_cond, 1)
  //   %false_operand_cond = GetTupleElement(%param_cond, 2)
  //   return Conditional(%pred_cond, %true_operand_cond, computation1,
  //                      %false_operand_cond, computation2)
  //
  // entry:
  //   %pred1 = Constant(true)
  //   %pred2 = Constant(false)
  //   %constant1 = Constant(1.1);
  //   %constant2 = Constant(2.2);
  //   %constant3 = Constant(3.3);
  //   return Conditional(%pred1, (%pred2, %constant1, %constant2),
  //                      inner_conditional, %constant3, computation3)

  auto computation1 = module_->AddEmbeddedComputation(
      CreateR0F32UnaryOpComputation(HloOpcode::kCeil));
  auto computation2 = module_->AddEmbeddedComputation(
      CreateR0F32UnaryOpComputation(HloOpcode::kFloor));
  auto computation3 = module_->AddEmbeddedComputation(
      CreateR0F32UnaryOpComputation(HloOpcode::kNegate));

  // Build inner_conditional computation.
  const Shape scalar_bool_shape = ShapeUtil::MakeShape(PRED, {});
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {scalar_bool_shape, scalar_shape_, scalar_shape_});
  auto inner_builder =
      HloComputation::Builder(TestName() + "_inner_conditional");
  auto param_cond = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_param_shape, "param_cond"));
  auto pred_cond = inner_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_bool_shape, param_cond, 0));
  auto true_operand_cond = inner_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param_cond, 1));
  auto false_operand_cond = inner_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param_cond, 2));
  auto inner_conditional =
      inner_builder.AddInstruction(HloInstruction::CreateConditional(
          scalar_shape_, pred_cond, true_operand_cond, computation1,
          false_operand_cond, computation2));
  auto inner_conditional_computation =
      module_->AddEmbeddedComputation(inner_builder.Build());

  // Build entry computation.
  auto builder = HloComputation::Builder(TestName());
  auto pred1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto pred2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.2f)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.3f)));
  auto tuple_operand = builder.AddInstruction(
      HloInstruction::CreateTuple({pred2, constant1, constant2}));
  auto conditional = builder.AddInstruction(HloInstruction::CreateConditional(
      scalar_shape_, pred1, tuple_operand, inner_conditional_computation,
      constant3, computation3));
  module_->AddEntryComputation(builder.Build());
  SCOPED_TRACE(module_->ToString());

  const HloDataflowAnalysis& analysis = RunAnalysis(GetParam());

  EXPECT_TRUE(analysis.ValueIsDefinedAt(pred1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(pred2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant3));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(tuple_operand));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(computation1->root_instruction()));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(computation2->root_instruction()));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(computation3->root_instruction()));

  auto computation1_param = computation1->parameter_instruction(0);
  auto computation2_param = computation2->parameter_instruction(0);
  auto computation3_param = computation3->parameter_instruction(0);
  EXPECT_FALSE(analysis.ValueIsDefinedAt(computation1_param));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(computation2_param));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(computation3_param));
  EXPECT_EQ(analysis.GetUniqueValueAt(computation1_param),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(computation2_param),
            analysis.GetValueDefinedAt(constant2));
  EXPECT_EQ(analysis.GetUniqueValueAt(computation3_param),
            analysis.GetValueDefinedAt(constant3));

  EXPECT_FALSE(analysis.ValueIsDefinedAt(param_cond));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(pred_cond));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(true_operand_cond));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(false_operand_cond));
  EXPECT_EQ(analysis.GetUniqueValueAt(param_cond),
            analysis.GetValueDefinedAt(tuple_operand));
  EXPECT_EQ(analysis.GetUniqueValueAt(pred_cond),
            analysis.GetValueDefinedAt(pred2));
  EXPECT_EQ(analysis.GetUniqueValueAt(true_operand_cond),
            analysis.GetValueDefinedAt(constant1));
  EXPECT_EQ(analysis.GetUniqueValueAt(false_operand_cond),
            analysis.GetValueDefinedAt(constant2));

  bool ssa_form = GetParam();
  if (ssa_form) {
    EXPECT_EQ(analysis.values().size(), 11);
    EXPECT_TRUE(analysis.ValueIsDefinedAt(inner_conditional));
    EXPECT_TRUE(analysis.ValueIsDefinedAt(conditional));
  } else {
    EXPECT_EQ(analysis.values().size(), 9);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(inner_conditional));
    EXPECT_FALSE(analysis.ValueIsDefinedAt(conditional));
    EXPECT_THAT(
        HloValuesAt(inner_conditional),
        UnorderedElementsAre(
            &analysis.GetValueDefinedAt(computation1->root_instruction()),
            &analysis.GetValueDefinedAt(computation2->root_instruction())));
    EXPECT_THAT(
        HloValuesAt(conditional),
        UnorderedElementsAre(
            &analysis.GetValueDefinedAt(computation1->root_instruction()),
            &analysis.GetValueDefinedAt(computation2->root_instruction()),
            &analysis.GetValueDefinedAt(computation3->root_instruction())));
  }
}

TEST_P(HloDataflowAnalysisTest, AddDependency) {
  std::string module_string = R"(
HloModule AddDependency
ENTRY %AddDependency (p: f32[3]) -> f32[3] {
  %p = f32[3] parameter(0)
  %token0 = token[] after-all()
  ROOT %add_dep = f32[3] add-dependency(f32[3] %p, token[] %token0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(module_string, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloDataflowAnalysis> analysis,
                          HloDataflowAnalysis::Run(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAddDependency);

  // The after-all and parameter should define a value. Add-dependency should
  // not.
  EXPECT_EQ(analysis->values().size(), 2);
  EXPECT_FALSE(analysis->ValueIsDefinedAt(root));
}

TEST_F(HloDataflowAnalysisTest, AllReduceStartAndDone) {
  const char* hlo_text = R"(
    HloModule test
    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT add = f32[] add(x, y)
    }
    ENTRY entry {
      p0 = f32[2] parameter(0)
      start = f32[2] all-reduce-start(p0), to_apply=add
      ROOT done = f32[2] all-reduce-done(start)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloDataflowAnalysis> analysis,
                          HloDataflowAnalysis::Run(*module));

  HloInstruction* done = module->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  HloInstruction* param0 = start->mutable_operand(0);

  EXPECT_TRUE(analysis->ValueIsDefinedAt(start, /*index=*/{}));
  EXPECT_FALSE(analysis->ValueIsDefinedAt(done));

  EXPECT_THAT(analysis->GetValueDefinedAt(param0).GetUses(),
              UnorderedElementsAre(HloUse{start, 0, {}}));
  EXPECT_THAT(analysis->GetValueDefinedAt(start).GetUses(),
              UnorderedElementsAre(HloUse{done, 0, {}}));
}

TEST_F(HloDataflowAnalysisTest, AllReduceStartAndDoneTwoOperands) {
  const char* hlo_text = R"(
    HloModule test
    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT add = f32[] add(x, y)
    }
    ENTRY entry {
      p0 = f32[2] parameter(0)
      p1 = f32[2] parameter(1)
      start = (f32[2], f32[2]) all-reduce-start(p0, p1), to_apply=add
      ROOT done = (f32[2], f32[2]) all-reduce-done(start)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloDataflowAnalysis> analysis,
                          HloDataflowAnalysis::Run(*module));

  HloInstruction* done = module->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  HloInstruction* param0 = start->mutable_operand(0);
  HloInstruction* param1 = start->mutable_operand(1);

  EXPECT_TRUE(analysis->ValueIsDefinedAt(start, /*index=*/{}));
  EXPECT_TRUE(analysis->ValueIsDefinedAt(start, /*index=*/{0}));
  EXPECT_TRUE(analysis->ValueIsDefinedAt(start, /*index=*/{1}));
  EXPECT_FALSE(analysis->ValueIsDefinedAt(done));

  EXPECT_THAT(analysis->GetValueDefinedAt(param0).GetUses(),
              UnorderedElementsAre(HloUse{start, 0, {}}));
  EXPECT_THAT(analysis->GetValueDefinedAt(param1).GetUses(),
              UnorderedElementsAre(HloUse{start, 1, {}}));
  EXPECT_THAT(analysis->GetValueDefinedAt(start, {}).GetUses(),
              UnorderedElementsAre(HloUse{done, 0, {}}));
}

INSTANTIATE_TEST_SUITE_P(HloDataflowAnalysisInstantiation,
                         HloDataflowAnalysisTest,
                         ::testing::Values(false, true));

std::unique_ptr<HloDataflowAnalysis> RunAnalysis(
    const HloModule& module,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr) {
  return HloDataflowAnalysis::Run(module, /*ssa_form=*/false,
                                  /*bitcast_defines_value=*/false,
                                  can_share_buffer)
      .value();
}

using DoesNotUseOperandBufferTest = HloTestBase;

TEST_F(DoesNotUseOperandBufferTest, GetTupleElement) {
  auto builder = HloComputation::Builder(TestName());

  Shape elem_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({elem_shape, elem_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(elem_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(elem_shape, tuple, 1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(elem_shape, HloOpcode::kAdd, gte0, gte1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  // GetTupleElement instructions only access the top-level buffer of their
  // operand.
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {0}, gte0));
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {1}, gte1));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {}, gte0));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {}, gte1));
}

TEST_F(DoesNotUseOperandBufferTest, FusedDynamicUpdateSlice) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update,
          std::initializer_list<HloInstruction*>({starts})));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  // The fusion instruction never uses tuple element 0, but does use element 1.
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {0}, fusion));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {1}, fusion));
}

// Similar to FusedDynamicUpdateSlice above, but tests indirect uses of the
// parameter tuple.
TEST_F(DoesNotUseOperandBufferTest, IndirectUses) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto t0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple_param, 0));
  auto t1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple_param, 1));
  // Swap the tuple elements.
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({t1, t0}));

  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update,
          std::initializer_list<HloInstruction*>({starts})));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  // The fusion instruction never uses tuple element 0, but does use element 1.
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {0}, fusion));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {1}, fusion));
  // The same holds for the parameter tuple, except that the tuple elements
  // are swapped in 'tuple'.
  EXPECT_TRUE(
      dataflow_analysis->DoesNotUseOperandBuffer(tuple_param, {1}, fusion));
  EXPECT_FALSE(
      dataflow_analysis->DoesNotUseOperandBuffer(tuple_param, {0}, fusion));
}

using CanShareOperandBufferWithUserTest = HloTestBase;

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseSameShape) {
  auto builder = HloComputation::Builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kLog, exp));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(param, {}, exp, {}));
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(exp, {}, log, {}));
}

TEST_F(CanShareOperandBufferWithUserTest,
       NonElementwiseLoopFusionCantAliasOperandBuffer) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "param0"));

  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(data_shape, HloOpcode::kNegate, param0));

  auto reverse = builder.AddInstruction(
      HloInstruction::CreateReverse(data_shape, neg, {0, 1}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {reverse, neg}, HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(param0, {}, fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest,
       MultiOutputFusionCanAliasOperandBuffer) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  Shape in_shape = ShapeUtil::MakeShape(F32, {8});
  Shape out_shape = ShapeUtil::MakeShape(PRED, {8});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, in_shape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, in_shape, "param1"));

  auto copy0 = builder.AddInstruction(
      HloInstruction::CreateUnary(in_shape, HloOpcode::kCopy, param0));
  auto copy1 = builder.AddInstruction(
      HloInstruction::CreateUnary(in_shape, HloOpcode::kCopy, param1));

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({copy1, copy0}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {tuple, copy1, copy0}, HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                               fusion, {0}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                               fusion, {1}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                               fusion, {0}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                               fusion, {1}));
}

TEST_F(CanShareOperandBufferWithUserTest,
       ElementwiseLoopFusionCantAliasOperandBuffer) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {}));

  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(data_shape, HloOpcode::kNegate, operand));

  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(data_shape, HloOpcode::kExp, neg));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {exp, neg}, HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(operand, {},
                                                               fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest,
       CanShareOperandWhenDynamicUpdateSliceIsFedByDynamicSliceWithSameIndex) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {1, 2});

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "param0"));
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));
  auto ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      slice_shape, param, {zero, zero}, {1, 2}));

  auto dus = builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      data_shape, param, ds, {zero, zero}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {dus, ds, zero}, HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(param, {}, fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, DUSWithSliceWithSameIndices) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      p0 = f32[10,20,30] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      slice = f32[1,1,30] dynamic-slice(p0, p1, p2, p3), dynamic_slice_sizes={1,1,30}
      ROOT dus = f32[10,20,30] dynamic-update-slice(p0, slice, p1, p2, p3)
    }

    ENTRY test {
      p0 = f32[10,20,30] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT fusion = f32[10,20,30] fusion(p0, p1, p2, p3), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  auto* fusion = module->entry_computation()->root_instruction();
  auto* param = module->entry_computation()->parameter_instruction(0);

  auto dataflow_analysis = RunAnalysis(*module);
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(param, {}, fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseDifferentShape) {
  auto builder = HloComputation::Builder(TestName());

  Shape in_shape = ShapeUtil::MakeShape(F32, {8});
  Shape out_shape = ShapeUtil::MakeShape(PRED, {8});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, in_shape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, in_shape, "param1"));
  auto result = builder.AddInstruction(HloInstruction::CreateCompare(
      out_shape, param0, param1, ComparisonDirection::kEq));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(param0, {}, result, {}));
  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(param1, {}, result, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, CopyShares) {
  auto builder = HloComputation::Builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, exp));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(param, {}, exp, {}));
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(exp, {}, copy, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDynamicUpdateSlice) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update,
          std::initializer_list<HloInstruction*>({starts})));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  // The fusion instruction can share with tuple element 1.
  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(tuple, {0}, fusion, {}));
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(tuple, {1}, fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest,
       FusedDynamicUpdateSliceWithConvertCanShare) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  Shape data_shape_bf16 = ShapeUtil::MakeShape(BF16, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  auto convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(data_shape_bf16, gte1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape_bf16, convert1, update,
          std::initializer_list<HloInstruction*>({starts})));

  auto convert2 = builder.AddInstruction(
      HloInstruction::CreateConvert(data_shape, dynamic_update_slice));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, convert2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {convert2, dynamic_update_slice, starts, update, convert1},
      HloInstruction::FusionKind::kLoop);
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(gte1, {}, fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, DynamicUpdateSliceCanShare) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {1, 8});
  Shape update_shape = ShapeUtil::MakeShape(F32, {1, 4});
  Shape starts_shape = ShapeUtil::MakeShape(S32, {2});
  auto data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "data"));
  auto update = builder.AddInstruction(
      HloInstruction::CreateParameter(1, update_shape, "update"));
  auto start = builder.AddInstruction(
      HloInstruction::CreateParameter(2, starts_shape, "start"));

  auto dus = builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      data_shape, data, update, {start}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  // The DynamicUpdateSlice instruction can share with the data operand, but not
  // with update or start.
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(data, {}, dus, {}));
  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(update, {}, dus, {}));
  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(start, {}, dus, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, ScatterCanShare) {
  const char* hlo_text = R"(
    HloModule TensorFlowScatterV1

    update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1)
    }

    ENTRY main {
      operand = s32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = s32[2,3] parameter(2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=update_s32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto computation = module->entry_computation();
  auto dataflow_analysis = RunAnalysis(*module);

  HloInstruction* operand_param = computation->parameter_instruction(0);
  HloInstruction* indices_param = computation->parameter_instruction(1);
  HloInstruction* updates_param = computation->parameter_instruction(2);
  HloInstruction* scatter = computation->root_instruction();

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand_param, {}, scatter, {}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      indices_param, {}, scatter, {}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates_param, {}, scatter, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, MultioutputScatterCanShare) {
  const char* hlo_text = R"(
    HloModule MultioutputScatter

    update {
      lhs0 = s32[] parameter(0)
      lhs1 = f32[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = f32[] parameter(3)
      ROOT tuple = tuple(rhs0, rhs1)
    }

    ENTRY main {
      operand0 = s32[3,3] parameter(0)
      operand1 = f32[3,3] parameter(1)
      indices = s32[2] parameter(2)
      updates0 = s32[2,3] parameter(3)
      updates1 = f32[2,3] parameter(4)
      ROOT scatter = (s32[3,3], f32[3,3])
      scatter(operand0, operand1, indices, updates0, updates1),
          to_apply=update,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto computation = module->entry_computation();
  auto dataflow_analysis = RunAnalysis(*module);

  HloInstruction* operand0_param = computation->parameter_instruction(0);
  HloInstruction* operand1_param = computation->parameter_instruction(1);
  HloInstruction* indices_param = computation->parameter_instruction(2);
  HloInstruction* updates0_param = computation->parameter_instruction(3);
  HloInstruction* updates1_param = computation->parameter_instruction(4);
  HloInstruction* scatter = computation->root_instruction();

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand0_param, {}, scatter, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand0_param, {}, scatter, {1}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand1_param, {}, scatter, {0}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand1_param, {}, scatter, {1}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      indices_param, {}, scatter, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      indices_param, {}, scatter, {1}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates0_param, {}, scatter, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates0_param, {}, scatter, {1}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates1_param, {}, scatter, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates1_param, {}, scatter, {1}));
}

TEST_F(CanShareOperandBufferWithUserTest, TriangularSolveCanShare) {
  const char* hlo_text = R"(
    HloModule TensorFlowTriangularSolve

    ENTRY main {
      a = f32[4,4]{1,0} parameter(0)
      b = f32[3,4]{1,0} parameter(1)
      ROOT triangular-solve = f32[3,4]{1,0} triangular-solve(a, b), lower=true,
                                              transpose_a=NO_TRANSPOSE
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  auto computation = module->entry_computation();
  auto dataflow_analysis = RunAnalysis(*module);

  HloInstruction* lhs_param = computation->parameter_instruction(0);
  HloInstruction* rhs_param = computation->parameter_instruction(1);
  HloInstruction* triangular_solve = computation->root_instruction();

  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      lhs_param, {}, triangular_solve, {}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      rhs_param, {}, triangular_solve, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, SortCanShare) {
  auto builder = HloComputation::Builder(TestName());
  auto module = CreateNewVerifiedModule();

  Shape keys_shape = ShapeUtil::MakeShape(F32, {8});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  TF_ASSERT_OK_AND_ASSIGN(
      auto* sort, MakeSortHlo(keys_shape, {keys}, -1, /*is_stable=*/false,
                              &builder, module.get()));

  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(keys, {}, sort, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, SortCanShareWithTupleUser) {
  auto builder = HloComputation::Builder(TestName());
  auto module = CreateNewVerifiedModule();

  Shape keys_shape = ShapeUtil::MakeShape(F32, {8});
  Shape values_shape = ShapeUtil::MakeShape(F32, {8});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  auto values = builder.AddInstruction(
      HloInstruction::CreateParameter(1, values_shape, "values"));
  TF_ASSERT_OK_AND_ASSIGN(
      auto* sort,
      MakeSortHlo(ShapeUtil::MakeTupleShape({keys_shape, values_shape}),
                  {keys, values}, 0, /*is_stable=*/false, &builder,
                  module.get()));

  module->AddEntryComputation(builder.Build());
  auto dataflow_analysis = RunAnalysis(*module);

  // The buffer for the keys can be shared with the first tuple entry.
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(keys, {}, sort, {0}));
  // The buffer for the values can be shared with the second tuple entry.
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(values, {}, sort, {1}));
  // Verify that the buffers are not shared with the "wrong" tuple entry.
  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(keys, {}, sort, {1}));
  EXPECT_FALSE(
      dataflow_analysis->CanShareOperandBufferWithUser(values, {}, sort, {0}));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDotAdd) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto a = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 0.0}, {0.0, 1.0}})));
  auto b = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(data_shape, a, b, dot_dnums, precision_config));

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto add_operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {}));

  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      data_shape, HloOpcode::kAdd, dot, add_operand));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {add, dot}, HloInstruction::FusionKind::kOutput);
  auto dataflow_analysis = RunAnalysis(*module);

  // Output fused dot add should be able to share buffer with 'add_operand'.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(add_operand, {},
                                                               fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, OutputFusionCantAliasOperandBuffer) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {}));

  auto reverse = builder.AddInstruction(
      HloInstruction::CreateReverse(data_shape, operand, {0, 1}));

  auto two = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, reverse, two));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {add, two, reverse}, HloInstruction::FusionKind::kOutput);
  auto dataflow_analysis = RunAnalysis(*module);

  // Output fused operand->reverse->add cannot alias operand buffer 'operand'.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(operand, {},
                                                                fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, FusionCanShareBufferCustomized) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {}));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      data_shape, HloOpcode::kMultiply, operand, operand));
  auto two = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, mul, two));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  auto fusion = computation->CreateFusionInstruction(
      {add, two, mul}, HloInstruction::FusionKind::kInput);
  auto dataflow_analysis = RunAnalysis(
      *module,
      /*can_share_buffer=*/[](const HloInstruction* fusion,
                              const HloInstruction*, const ShapeIndex&) {
        return fusion->IsLoopFusion();
      });

  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(operand, {},
                                                                fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, WhileCanShare) {
  auto module = CreateNewVerifiedModule();
  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  Shape pred_scalar_shape = ShapeUtil::MakeShape(PRED, {});

  auto b = HloComputation::Builder(TestName() + ".And");
  auto p0 = b.AddInstruction(
      HloInstruction::CreateParameter(0, pred_scalar_shape, "p0"));
  auto p1 = b.AddInstruction(
      HloInstruction::CreateParameter(1, pred_scalar_shape, "p1"));
  b.AddInstruction(
      HloInstruction::CreateBinary(pred_scalar_shape, HloOpcode::kAnd, p0, p1));
  auto and_computation = module->AddEmbeddedComputation(b.Build());

  auto make_cond = [&data_shape, &and_computation]() {
    auto builder = HloComputation::Builder(TestName() + ".Cond");
    auto data = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape, "data"));
    auto compare = builder.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {8}), data, data, ComparisonDirection::kEq));
    auto true_value = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
    builder.AddInstruction(
        HloInstruction::CreateReduce(ShapeUtil::MakeShape(PRED, {}), compare,
                                     true_value, {0}, and_computation));
    return builder.Build();
  };

  auto make_body = [&data_shape]() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    auto data = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape, "data"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, data, data));
    return builder.Build();
  };

  HloComputation* cond_computation =
      module->AddEmbeddedComputation(make_cond());
  HloComputation* body_computation =
      module->AddEmbeddedComputation(make_body());

  auto builder = HloComputation::Builder(TestName());
  auto data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "data"));
  auto whil = builder.AddInstruction(HloInstruction::CreateWhile(
      data_shape, cond_computation, body_computation, data));
  module->AddEntryComputation(builder.Build());

  auto dataflow_analysis = RunAnalysis(*module);

  // The While instruction can share with the data operand.
  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(data, {}, whil, {}));
}

// Tests that Call can alias operand buffer if the only use of the operand
// in the called computation is an elementwise instruction.
TEST_F(CanShareOperandBufferWithUserTest, CallToComputationWithFusionRoot) {
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  // Build sub-computation with fusion root.
  auto sub_builder = HloComputation::Builder(TestName() + "_sub");
  auto sub_param = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "sub_param"));
  auto one = sub_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto ones = sub_builder.AddInstruction(
      HloInstruction::CreateBroadcast(shape, one, {}));
  auto add = sub_builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, sub_param, ones));

  auto module = CreateNewVerifiedModule();
  auto sub_computation = module->AddEmbeddedComputation(sub_builder.Build());
  sub_computation->CreateFusionInstruction({add, ones},
                                           HloInstruction::FusionKind::kLoop);

  // Build entry-computation with kCall which calls 'sub_computation'.
  auto builder = HloComputation::Builder(TestName());

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto reverse =
      builder.AddInstruction(HloInstruction::CreateReverse(shape, param, {0}));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {reverse}, sub_computation));
  module->AddEntryComputation(builder.Build());

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(
      dataflow_analysis->CanShareOperandBufferWithUser(reverse, {}, call, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, ConcatSliceWithElementwise) {
  const char* kModule = R"(
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
      ROOT fusion = (f32[200], f32[100]) fusion(p0, p1, p2, p3), kind=kInput, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  auto* fusion = module->entry_computation()->root_instruction();
  auto* param0 = module->entry_computation()->parameter_instruction(0);
  auto* param1 = module->entry_computation()->parameter_instruction(1);
  auto* param2 = module->entry_computation()->parameter_instruction(2);
  auto* param3 = module->entry_computation()->parameter_instruction(3);

  auto dataflow_analysis = RunAnalysis(*module);
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                               fusion, {0}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                               fusion, {0}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param2, {},
                                                               fusion, {1}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param3, {},
                                                               fusion, {1}));
  // Tensors of different sizes cannot share buffer.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {1}));
}

TEST_F(CanShareOperandBufferWithUserTest, ConcatSliceNegativeTest) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      // p0 has multiple transitive uses fed to concat. So, p0 cannot share
      // buffer with outputs because the aliased output could be written before
      // all the uses of p0 are finished.
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      add0 = f32[100] add(p0, p1)
      concat0 = f32[200] concatenate(p0, add0), dimensions={0}
      slice0 = f32[100] slice(concat0), slice={[0:100]}
      slice1 = f32[100] slice(concat0), slice={[100:200]}
      ROOT tuple = (f32[100], f32[100]) tuple(slice0, slice1)
    }

    ENTRY test {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT fusion = (f32[100], f32[100]) fusion(p0, p1),
                        kind=kInput, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  auto* fusion = module->entry_computation()->root_instruction();
  auto* param0 = module->entry_computation()->parameter_instruction(0);
  auto* param1 = module->entry_computation()->parameter_instruction(1);

  auto dataflow_analysis = RunAnalysis(*module);
  // p0 cannot share with either fusion{0} or fusion{1}.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {1}));
  // p1 cannot share with fusion{0} because we're not sure about their
  // relationship.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                                fusion, {0}));
  // p1 can share with fusion{1} because they will be executed in an
  // elementwise manner.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                               fusion, {1}));
}

TEST_F(CanShareOperandBufferWithUserTest, MultipleConcatenates) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      add0 = f32[100] add(p0, p1)
      sub0 = f32[100] subtract(p1, p1)
      concat0 = f32[200] concatenate(p0, add0), dimensions={0}
      slice0 = f32[100] slice(concat0), slice={[0:100]}
      slice1 = f32[100] slice(concat0), slice={[100:200]}
      concat1 = f32[200] concatenate(p0, sub0), dimensions={0}
      slice2 = f32[100] slice(concat1), slice={[0:100]}
      slice3 = f32[100] slice(concat1), slice={[100:200]}
      ROOT tuple = (f32[100], f32[100], f32[100], f32[100])
                       tuple(slice0, slice1, slice2, slice3)
    }

    ENTRY test {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT fusion = (f32[100], f32[100], f32[100], f32[100])
          fusion(p0, p1), kind=kInput, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  auto* fusion = module->entry_computation()->root_instruction();
  auto* param0 = module->entry_computation()->parameter_instruction(0);
  auto* param1 = module->entry_computation()->parameter_instruction(1);

  auto dataflow_analysis = RunAnalysis(*module);
  // p0 cannot share.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {1}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {2}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param0, {},
                                                                fusion, {3}));
  // p1 can share with either fusion{1} or fusion{3}.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                               fusion, {1}));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                               fusion, {3}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                                fusion, {0}));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(param1, {},
                                                                fusion, {2}));
}

using GetInPlaceInputOutputPairsTest = HloTestBase;

TEST_F(GetInPlaceInputOutputPairsTest, DUS) {
  const char* kModule = R"(
    HloModule test

    ENTRY test {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT dus = f32[10] dynamic-update-slice(p0, p1, p2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* dus = module->entry_computation()->root_instruction();

  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(dus);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, DUSFusion) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT dus = f32[10] dynamic-update-slice(p0, p1, p2)
    }

    ENTRY test {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT fusion = f32[10] fusion(p0, p1, p2), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, DUSFusionWithOutputOperandAliasing) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      dus = f32[10] dynamic-update-slice(p0, p1, p2)
      ROOT tuple = (f32[5], f32[10]) tuple(p1, dus)
    }

    ENTRY test {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT fusion = (f32[5], f32[10]) fusion(p0, p1, p2), kind=kLoop, output_to_operand_aliasing={{0}: (1, {})}, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {1}});  // discovered
  expected_pairs.push_back({HloOperandIndex{1, {}}, {0}});  // annotated
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, NonDUSFusion) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      p0 = f32[10] parameter(0)
      p1 = f32[10] parameter(1)
      ROOT add = f32[10] add(p0, p1)
    }

    ENTRY test {
      p0 = f32[10] parameter(0)
      p1 = f32[10] parameter(1)
      ROOT fusion = f32[10] fusion(p0, p1), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(fusion);
  EXPECT_THAT(in_place_pairs, IsEmpty());
}

TEST_F(GetInPlaceInputOutputPairsTest, NonDUSFusionWithOutputOperandAliasing) {
  const char* kModule = R"(
    HloModule test

    fused_computation {
      p0 = f32[10] parameter(0)
      p1 = f32[10] parameter(1)
      ROOT add = f32[10] add(p0, p1)
    }

    ENTRY test {
      p0 = f32[10] parameter(0)
      p1 = f32[10] parameter(1)
      ROOT fusion = f32[10] fusion(p0, p1), kind=kLoop, output_to_operand_aliasing={{}: (0, {})}, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(fusion);

  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, NestedDUSFusion) {
  const char* kModule = R"(
    HloModule test

    fused_computation1 {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT dus = f32[10] dynamic-update-slice(p0, p1, p2)
    }

    fused_computation2 {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT fusion = f32[10] fusion(p0, p1, p2), kind=kLoop, calls=fused_computation1
    }

    ENTRY test {
      p0 = f32[10] parameter(0)
      p1 = f32[5] parameter(1)
      p2 = s32[] parameter(2)
      ROOT fusion = f32[10] fusion(p0, p1, p2), kind=kLoop, calls=fused_computation2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, NestedMultiOutputDUSFusion) {
  const char* kModule = R"(
    HloModule test

    fused_computation1 {
      p0 = s32[] parameter(0)
      p1 = (f32[5],f32[10]) parameter(1)
      gte0 = f32[5] get-tuple-element(p1), index=0
      gte1 = f32[10] get-tuple-element(p1), index=1
      dus = f32[10] dynamic-update-slice(gte1, gte0, p0)
      negate = f32[5] negate(gte0)
      ROOT tuple = (f32[5],f32[10]) tuple(negate, dus)
    }

    fused_computation2 {
      p0 = f32[5] parameter(0)
      p1 = (f32[10],s32[]) parameter(1)
      gte0 = f32[10] get-tuple-element(p1), index=0
      gte1 = s32[] get-tuple-element(p1), index=1
      in_tuple = (f32[5],f32[10]) tuple(p0, gte0)
      inner_fusion = (f32[5],f32[10]) fusion(gte1, in_tuple), kind=kLoop, calls=fused_computation1
      fusion_gte0 = f32[5] get-tuple-element(inner_fusion), index=0
      fusion_gte1 = f32[10] get-tuple-element(inner_fusion), index=1
      negate = f32[5] negate(p0)
      ROOT tuple = (f32[5],f32[5],f32[10]) tuple(negate, fusion_gte0, fusion_gte1)
    }

    ENTRY test {
      p0 = f32[5] parameter(0)
      p1 = (f32[10],s32[]) parameter(1)
      ROOT fusion = (f32[5],f32[5],f32[10]) fusion(p0, p1), kind=kLoop, calls=fused_computation2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  HloInstruction* inner_fusion = FindInstruction(module.get(), "inner_fusion");

  auto inner_in_place_pairs =
      HloDataflowAnalysis::GetInPlaceInputOutputPairs(inner_fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> inner_expected_pairs;
  inner_expected_pairs.push_back({HloOperandIndex{1, {1}}, {1}});
  EXPECT_EQ(inner_in_place_pairs, inner_expected_pairs);

  auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{1, {0}}, {2}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

}  // namespace
}  // namespace xla
