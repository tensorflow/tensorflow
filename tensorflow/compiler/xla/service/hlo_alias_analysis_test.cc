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

#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"

#include <map>
#include <memory>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape_util.h"
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

class HloAliasAnalysisTest : public HloTestBase {
 protected:
  HloAliasAnalysisTest() : module_(CreateNewModule()) {}

  // Run alias analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  HloAliasAnalysis& RunAnalysis() {
    analysis_ = HloAliasAnalysis::Run(module_.get()).ConsumeValueOrDie();
    return *analysis_;
  }

  // Return a vector of the buffers in the buffer set at the current position
  // sorted by buffer id.
  std::vector<HloBuffer> GetBuffersAt(const HloInstruction* instruction,
                                      const ShapeIndex& index = {}) const {
    std::set<HloBuffer::Id> buffer_ids;
    for (const HloValue* value : analysis_->dataflow_analysis()
                                     .GetValueSet(instruction, index)
                                     .values()) {
      buffer_ids.insert(analysis_->GetBufferContainingValue(*value).id());
    }

    std::vector<HloBuffer> buffers;
    for (HloBuffer::Id id : buffer_ids) {
      buffers.push_back(analysis_->GetBuffer(id));
    }
    return buffers;
  }

  // Return a vector containing all of the HloValues in the given buffer.
  std::vector<HloValue> GetValuesInBuffer(const HloBuffer& buffer) {
    std::vector<HloValue> values;
    for (const HloValue* value : buffer.values()) {
      values.push_back(*value);
    }
    return values;
  }

  // Return the HloValue defined at the given position.
  const HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                                    const ShapeIndex& index = {}) const {
    return analysis_->dataflow_analysis().GetValueDefinedAt(instruction, index);
  }

  // Returns true if any values held in the same buffer interfere. Generally, in
  // the compiler pipeline copy-insertion will guarantee that this interference
  // never occurs, but HLO graphs with interference can be explicitly
  // constructed.
  bool AnyValuesInSameBufferInterfere() {
    DependencyHloOrdering ordering(module_.get());
    for (const HloBuffer* buffer : analysis_->buffers()) {
      for (const HloValue* value_a : buffer->values()) {
        for (const HloValue* value_b : buffer->values()) {
          if (*value_a != *value_b &&
              analysis_->dataflow_analysis().MayInterfere(*value_a, *value_b,
                                                          ordering)) {
            VLOG(1) << *value_a << " interferes with " << *value_b
                    << " in buffer: " << *buffer;
            return true;
          }
        }
      }
    }
    return false;
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloAliasAnalysis> analysis_;

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(HloAliasAnalysisTest, BinaryOperation) {
  // Test the analysis on a single binary operation (Add).
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, constant1, constant2));
  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.buffers().size(), 3);

  // All of the buffer sets should trivially contain a single buffer containing
  // a single value.
  for (const HloInstruction* instruction : {constant1, constant2, add}) {
    EXPECT_EQ(analysis.GetUniqueBufferAt(instruction).GetUniqueValue(),
              GetValueDefinedAt(instruction));
  }

  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(add));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(add));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, TupleAndGtes) {
  // Verify the analysis for a Tuple and GetTupleElement instructions.
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
  builder.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape_, HloOpcode::kAdd, gte0, gte1));
  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.buffers().size(), 4);

  // Verify the expected aliasing of the tuple elements.
  EXPECT_EQ(analysis.GetUniqueBufferAt(tuple, /*index=*/{}).GetUniqueValue(),
            GetValueDefinedAt(tuple, /*index=*/{}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(tuple, /*index=*/{0}).GetUniqueValue(),
            GetValueDefinedAt(param0));
  EXPECT_EQ(analysis.GetUniqueBufferAt(tuple, /*index=*/{1}).GetUniqueValue(),
            GetValueDefinedAt(param1));

  // The tuple operand, tuple element, and result of the GTE instruction should
  // all be the same buffer.
  EXPECT_EQ(analysis.GetUniqueBufferAt(param0),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{0}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(param0),
            analysis.GetUniqueBufferAt(gte0));

  // Verify the positions of an aliased buffer.
  EXPECT_THAT(
      analysis.GetUniqueBufferAt(param0).ComputePositions(),
      UnorderedElementsAre(HloPosition{param0, {}}, HloPosition{tuple, {0}},
                           HloPosition{gte0, {}}));

  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(tuple));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(tuple));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, NondistinctTuple) {
  // Test a expression with a non-distinct buffer set.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  // param0 is included twice in the tuple.
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({param0, param1, param0}));
  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_THAT(
      analysis.GetUniqueBufferAt(param0).ComputePositions(),
      UnorderedElementsAre(HloPosition{param0, {}}, HloPosition{tuple, {0}},
                           HloPosition{tuple, {2}}));

  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(tuple));
  EXPECT_FALSE(analysis.InstructionBuffersAreDistinct(tuple));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, SingleCall) {
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

  const HloAliasAnalysis& analysis = RunAnalysis();

  // Verify aliasing of the kCall operands and the subcomputation parameters.
  EXPECT_THAT(analysis.GetUniqueBufferAt(constant1).ComputePositions(),
              UnorderedElementsAre(HloPosition{constant1, {}},
                                   HloPosition{subparam0, {}}));
  EXPECT_THAT(analysis.GetUniqueBufferAt(constant2).ComputePositions(),
              UnorderedElementsAre(HloPosition{constant2, {}},
                                   HloPosition{subparam1, {}}));

  // The subcomputation root and the kCall itself should alias.
  EXPECT_THAT(
      analysis.GetUniqueBufferAt(add).ComputePositions(),
      UnorderedElementsAre(HloPosition{add, {}}, HloPosition{call, {}}));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, ComputationCalledTwice) {
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

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_THAT(analysis.GetUniqueBufferAt(constant1).ComputePositions(),
              UnorderedElementsAre(HloPosition{constant1, {}},
                                   HloPosition{subparam0, {}}));
  EXPECT_THAT(analysis.GetUniqueBufferAt(constant2).ComputePositions(),
              UnorderedElementsAre(HloPosition{constant2, {}},
                                   HloPosition{subparam1, {}}));

  // The 'add' (root of the subcomputation) aliases the two call instruction,
  // and the first parameter of the subcomputation because 'call1' it is passed
  // as an argument to the subcomputation in 'call2'.
  EXPECT_THAT(
      analysis.GetUniqueBufferAt(add).ComputePositions(),
      UnorderedElementsAre(HloPosition{add, {}}, HloPosition{call1, {}},
                           HloPosition{subparam0, {}}, HloPosition{call2, {}}));

  EXPECT_THAT(GetBuffersAt(subparam0),
              UnorderedElementsAre(analysis.GetUniqueBufferAt(constant1),
                                   analysis.GetUniqueBufferAt(add)));
  EXPECT_THAT(GetBuffersAt(subparam1),
              UnorderedElementsAre(analysis.GetUniqueBufferAt(constant2)));

  EXPECT_TRUE(analysis.InstructionBuffersAreAmbiguous(subparam0));
  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(subparam1));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(subparam0));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(subparam1));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, SingleWhile) {
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
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  // Condition computation trivially returns a constant "false".
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

  const HloAliasAnalysis& analysis = RunAnalysis();

  // Verify the positions of the aliased while buffers.
  EXPECT_THAT(
      analysis.GetUniqueBufferAt(xla_while, /*index=*/{}).ComputePositions(),
      UnorderedElementsAre(HloPosition{tuple, {}}, HloPosition{xla_while, {}},
                           HloPosition{body_param, {}},
                           HloPosition{body_tuple, {}},
                           HloPosition{cond_param, {}}));
  EXPECT_THAT(
      analysis.GetUniqueBufferAt(xla_while, /*index=*/{0}).ComputePositions(),
      UnorderedElementsAre(
          HloPosition{constant1, {}}, HloPosition{tuple, {0}},
          HloPosition{xla_while, {0}}, HloPosition{body_param, {0}},
          HloPosition{body_element_0, {}}, HloPosition{body_tuple, {0}},
          HloPosition{cond_param, {0}}));
  EXPECT_THAT(
      analysis.GetUniqueBufferAt(xla_while, /*index=*/{1}).ComputePositions(),
      UnorderedElementsAre(
          HloPosition{constant2, {}}, HloPosition{tuple, {1}},
          HloPosition{xla_while, {1}}, HloPosition{body_param, {1}},
          HloPosition{body_element_1, {}}, HloPosition{add, {}},
          HloPosition{body_tuple, {1}}, HloPosition{cond_param, {1}}));

  EXPECT_THAT(
      GetValuesInBuffer(analysis.GetUniqueBufferAt(xla_while, /*index=*/{0})),
      UnorderedElementsAre(GetValueDefinedAt(xla_while, /*index=*/{0}),
                           GetValueDefinedAt(body_param, /*index=*/{0}),
                           GetValueDefinedAt(cond_param, /*index=*/{0}),
                           GetValueDefinedAt(constant1)));
  EXPECT_THAT(
      GetValuesInBuffer(analysis.GetUniqueBufferAt(xla_while, /*index=*/{1})),
      UnorderedElementsAre(GetValueDefinedAt(constant2),
                           GetValueDefinedAt(xla_while, /*index=*/{1}),
                           GetValueDefinedAt(body_param, {1}),
                           GetValueDefinedAt(cond_param, {1}),
                           GetValueDefinedAt(add)));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, SequentialWhiles) {
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

  FlattenCallGraph flattener;
  TF_ASSERT_OK(flattener.Run(module_.get()).status());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.GetUniqueBufferAt(tuple, /*index=*/{}),
            analysis.GetUniqueBufferAt(xla_while2, /*index=*/{}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(xla_while2, /*index=*/{0}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant2),
            analysis.GetUniqueBufferAt(xla_while2, /*index=*/{1}));
}

TEST_F(HloAliasAnalysisTest, NestedWhiles) {
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

  auto build_cond_computation = [&tuple_shape]() {
    auto cond_builder = HloComputation::Builder("condition");
    cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "param"));
    cond_builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
    return cond_builder.Build();
  };
  // Build separate condition computations so the call graph is flat. The
  // callgraph is always flattened in the compiler pipeline, and the flattened
  // callgraph enables representative interference analysis.
  HloComputation* condition1 =
      module_->AddEmbeddedComputation(build_cond_computation());
  HloComputation* condition2 =
      module_->AddEmbeddedComputation(build_cond_computation());

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
      tuple_shape, condition1, inner_body, outer_tuple));
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
      HloInstruction::CreateWhile(tuple_shape, condition2, outer_body, tuple));
  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(entry_while, /*index=*/{0}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(nested_while, /*index=*/{0}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(inner_element_0));

  EXPECT_EQ(analysis.GetUniqueBufferAt(constant2),
            analysis.GetUniqueBufferAt(entry_while, /*index=*/{1}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant2),
            analysis.GetUniqueBufferAt(nested_while, /*index=*/{1}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant2),
            analysis.GetUniqueBufferAt(inner_element_1));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, SwizzlingWhile) {
  // Test a while instruction with a body which permutes it's tuple parameter
  // elements. HLO:
  //
  // body((F32[], F32[], F32[]) %tuple_param):
  //   return Tuple(%tuple_param{1}, %tuple_param{2}, %tuple_param{0})
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %constant3 = Constant(3.0)
  //   %tuple = Tuple(%constant1, %constant2, %constant3)
  //   return While(%tuple, body, condition)
  //
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_, scalar_shape_});

  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  auto body_element_2 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 2));
  body_builder.AddInstruction(HloInstruction::CreateTuple(
      {body_element_1, body_element_2, body_element_0}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
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
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(3.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2, constant3}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, tuple));
  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  // The swizzling while makes most positions in the module alias leaving only 3
  // HloBuffers.
  EXPECT_THAT(
      analysis.buffers(),
      UnorderedElementsAre(&analysis.GetUniqueBufferAt(constant1),
                           &analysis.GetUniqueBufferAt(tuple, /*index=*/{}),
                           &analysis.GetUniqueBufferAt(cond_constant)));

  // The tuple elements of the while and the three constant inputs should all be
  // smooshed into the same buffer.
  EXPECT_EQ(analysis.GetUniqueBufferAt(xla_while, /*index=*/{0}),
            analysis.GetUniqueBufferAt(xla_while, /*index=*/{1}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(xla_while, /*index=*/{0}),
            analysis.GetUniqueBufferAt(xla_while, /*index=*/{2}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(xla_while, /*index=*/{0}),
            analysis.GetUniqueBufferAt(constant1));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(constant2));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(constant3));

  // All elements in of the loop state tuple are forced into the same buffer
  // resulting liveness interference.
  EXPECT_TRUE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, TupleSelect) {
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

  const HloAliasAnalysis& analysis = RunAnalysis();

  // Verify the buffer sets of each select.
  EXPECT_THAT(GetBuffersAt(select11, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetUniqueBufferAt(constant1)));
  EXPECT_THAT(GetBuffersAt(select12, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetUniqueBufferAt(constant1),
                                   analysis.GetUniqueBufferAt(constant2)));
  EXPECT_THAT(GetBuffersAt(select34, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetUniqueBufferAt(constant3),
                                   analysis.GetUniqueBufferAt(constant4)));
  EXPECT_THAT(GetBuffersAt(select1234, /*index=*/{0}),
              UnorderedElementsAre(analysis.GetUniqueBufferAt(constant1),
                                   analysis.GetUniqueBufferAt(constant2),
                                   analysis.GetUniqueBufferAt(constant3),
                                   analysis.GetUniqueBufferAt(constant4)));

  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(select11));
  EXPECT_TRUE(analysis.InstructionBuffersAreAmbiguous(select12));
  EXPECT_TRUE(analysis.InstructionBuffersAreAmbiguous(select34));
  EXPECT_TRUE(analysis.InstructionBuffersAreAmbiguous(select1234));

  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(select11));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(select12));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(select34));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(select1234));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, TupleSelectToWhile) {
  // Test a tuple-shaped kSelect feeding a kWhile instruction. HLO:
  //
  // body((F32[], F32[]) %tuple_param):
  //   %negate = Negate(%tuple_param{0})
  //   return Tuple(%negate)
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %constant1 = Constant(1.0)
  //   %constant2 = Constant(2.0)
  //   %tuple1 = Tuple(%constant1)
  //   %tuple2 = Tuple(%constant2)
  //   %select = Select(%tuple1, %tuple2)
  //   return While(%select, body, condition)
  //
  auto builder = HloComputation::Builder(TestName());

  const Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape_});

  // Element 0 passes transparently through the body.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto body_element = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, body_element));
  body_builder.AddInstruction(HloInstruction::CreateTuple({negate}));
  HloComputation* body = module_->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
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
  auto tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant1}));
  auto tuple2 =
      builder.AddInstruction(HloInstruction::CreateTuple({constant2}));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred, tuple1, tuple2));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, condition, body, select));

  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  // The while should flatten the ambiguous select buffer set so that the buffer
  // set contents (constant1 and constant2) becomes a single buffer.
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(constant2));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant1),
            analysis.GetUniqueBufferAt(xla_while, /*index=*/{0}));

  EXPECT_THAT(GetValuesInBuffer(analysis.GetUniqueBufferAt(constant1)),
              UnorderedElementsAre(GetValueDefinedAt(constant1),
                                   GetValueDefinedAt(constant2),
                                   GetValueDefinedAt(xla_while, /*index=*/{0}),
                                   GetValueDefinedAt(body_param, /*index=*/{0}),
                                   GetValueDefinedAt(cond_param, /*index=*/{0}),
                                   GetValueDefinedAt(negate)));
  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(select));
  EXPECT_FALSE(analysis.InstructionBuffersAreAmbiguous(xla_while));

  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(select));
  EXPECT_TRUE(analysis.InstructionBuffersAreDistinct(xla_while));

  // The two operands of the select get flattened into the same buffer resulting
  // in liveness interference.
  EXPECT_TRUE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, Bitcast) {
  // Bitcasting a value should not produce a new buffer.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kBitcast, constant));

  module_->AddEntryComputation(builder.Build());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.buffers().size(), 1);

  EXPECT_EQ(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(bitcast));
}

TEST_F(HloAliasAnalysisTest, UpdateAnalysisForWhile) {
  // Test updating alias analysis after modifying a module with an array shaped
  // while:
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

  HloAliasAnalysis& analysis = RunAnalysis();

  // Sanity check some alias information.
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(body_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(cond_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(negate));
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(xla_while));

  // Set the body root to the body_param. Previously it was Negate(body_param).
  body->set_root_instruction(body_param);

  // Prior to updating, verify that the analysis is no longer valid.
  Status verify_status = analysis.VerifyAgainstReference();
  EXPECT_FALSE(verify_status.ok());

  analysis.UpdateAfterChangingRoot(/*old_root=*/negate,
                                   /*new_root*/ body_param);

  // Analysis should be valid after the update.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  // The exponential should now pass through the body transparently.
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(body_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(cond_param));
  EXPECT_NE(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(negate));
  EXPECT_EQ(analysis.GetUniqueBufferAt(exp),
            analysis.GetUniqueBufferAt(xla_while));

  // Now replace the operand of the while with %constant (was %exp).
  TF_ASSERT_OK(exp->ReplaceUseWith(xla_while, constant));
  analysis.UpdateAfterChangingOperand(xla_while, /*old_operand=*/exp,
                                      /*new_operand=*/constant);

  // Analysis should be valid after the update.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  EXPECT_EQ(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(body_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(cond_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(xla_while));
  EXPECT_NE(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(exp));
  EXPECT_NE(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(negate));

  // And finally make the negate the root of the body again.
  body->set_root_instruction(negate);
  analysis.UpdateAfterChangingRoot(/*old_root=*/body_param,
                                   /*new_root*/ negate);

  // Analysis should be valid after the update.
  TF_ASSERT_OK(analysis.VerifyAgainstReference());

  EXPECT_EQ(analysis.GetUniqueBufferAt(negate),
            analysis.GetUniqueBufferAt(body_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(negate),
            analysis.GetUniqueBufferAt(cond_param));
  EXPECT_EQ(analysis.GetUniqueBufferAt(negate),
            analysis.GetUniqueBufferAt(xla_while));
  EXPECT_EQ(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(negate));

  auto value_of = [&analysis](const HloInstruction* instruction) {
    return &analysis.dataflow_analysis().GetValueDefinedAt(instruction);
  };
  EXPECT_THAT(analysis.GetUniqueBufferAt(negate).values(),
              UnorderedElementsAre(value_of(body_param), value_of(cond_param),
                                   value_of(negate), value_of(constant),
                                   value_of(xla_while)));
}

// Test update tuple element.

}  // namespace
}  // namespace xla
