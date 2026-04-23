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

#include "xla/hlo/analysis/hlo_dataflow_analysis.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// Test is parameterized on a bool which is whether the dataflow analysis is
// performed with SSA form.
class HloDataflowAnalysisTest : public HloHardwareIndependentTestBase,
                                public ::testing::WithParamInterface<bool> {
 protected:
  HloDataflowAnalysisTest() : module_(CreateNewVerifiedModule()) {}

  // Run dataflow analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  const HloDataflowAnalysis& RunAnalysis(bool ssa_form,
                                         bool bitcast_defines_value = false,
                                         bool run_dce = true) {
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
                                 analysis_->GetValueDefinedAt(b), *analysis_,
                                 &alias_info_);
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
  AliasInfo alias_info_;

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
  const Shape vector_shape_ = ShapeUtil::MakeShape(F32, {42});
  const Shape tuple_shape_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});
};

TEST_P(HloDataflowAnalysisTest, BinaryOperation) {
  // Test the dataflow for a simple binary operation (Add).
  std::string hlo_str = R"(
HloModule BinaryOperation

ENTRY main {
  const1 = f32[] constant(1.0)
  const2 = f32[] constant(2.0)
  ROOT add = f32[] add(const1, const2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));
  HloInstruction* constant1 = FindInstruction(module_.get(), "const1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "const2");
  HloInstruction* add = FindInstruction(module_.get(), "add");
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

  // Check analysis ToString
  EXPECT_THAT(
      analysis.ToString(),
      testing::HasSubstr("HloDataflowAnalysis, module BinaryOperation"));
}

TEST_P(HloDataflowAnalysisTest, TupleAndGtes) {
  // Verify the dataflow through a Tuple and GetTupleElement instructions.
  std::string hlo_str = R"(
HloModule TupleAndGtes

ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(p0, p1)
  gte0 = f32[] get-tuple-element(tuple), index=0
  gte1 = f32[] get-tuple-element(tuple), index=1
  ROOT add = f32[] add(gte0, gte1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));
  HloInstruction* param0 = FindInstruction(module_.get(), "p0");
  HloInstruction* param1 = FindInstruction(module_.get(), "p1");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* gte0 = FindInstruction(module_.get(), "gte0");
  HloInstruction* gte1 = FindInstruction(module_.get(), "gte1");
  HloInstruction* add = FindInstruction(module_.get(), "add");
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
  std::string hlo_str = R"(
HloModule NestedTuple

ENTRY main {
  const1 = f32[] constant(1.0)
  const2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(const1, const2)
  nested_tuple = ((f32[], f32[]), (f32[], f32[]), f32[]) tuple(tuple, tuple, const1)
  gte_tuple = (f32[], f32[]) get-tuple-element(nested_tuple), index=1
  ROOT gte_out = f32[] get-tuple-element(gte_tuple), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));
  HloInstruction* constant1 = FindInstruction(module_.get(), "const1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "const2");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* nested_tuple = FindInstruction(module_.get(), "nested_tuple");
  HloInstruction* gte_tuple = FindInstruction(module_.get(), "gte_tuple");
  HloInstruction* gte_out = FindInstruction(module_.get(), "gte_out");
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
  std::string hlo_str = R"(
HloModule SingleCall

Subcomputation {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  ROOT add = f32[] add(param0, param1)
}

ENTRY main {
  const1 = f32[] constant(1.0)
  const2 = f32[] constant(2.0)
  ROOT call = f32[] call(const1, const2), to_apply=Subcomputation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));
  HloInstruction* subparam0 = FindInstruction(module_.get(), "param0");
  HloInstruction* subparam1 = FindInstruction(module_.get(), "param1");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* constant1 = FindInstruction(module_.get(), "const1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "const2");
  HloInstruction* call = FindInstruction(module_.get(), "call");
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
  std::string hlo_str = R"(
HloModule NestedCalls

InnerComputation {
  inner_p0 = f32[] parameter(0)
  inner_p1 = f32[] parameter(1)
  ROOT add = f32[] add(inner_p0, inner_p1)
}

OuterComputation {
  outer_p0 = f32[] parameter(0)
  outer_p1 = f32[] parameter(1)
  ROOT nested_call = f32[] call(outer_p1, outer_p0), to_apply=InnerComputation
}

ENTRY main {
  const1 = f32[] constant(1.0)
  const2 = f32[] constant(2.0)
  ROOT call = f32[] call(const1, const2), to_apply=OuterComputation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* nested_call = FindInstruction(module_.get(), "nested_call");
  HloInstruction* constant1 = FindInstruction(module_.get(), "const1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "const2");
  HloInstruction* call = FindInstruction(module_.get(), "call");
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
  // Element 0 passes transparently through the body.
  std::string hlo_str = R"(
HloModule SingleWhile

body (tuple_param: (f32[], f32[])) -> (f32[], f32[]) {
  body_param = (f32[], f32[]) parameter(0)
  body_gte0 = f32[] get-tuple-element(body_param), index=0
  body_gte1 = f32[] get-tuple-element(body_param), index=1
  add = f32[] add(body_gte0, body_gte1)
  ROOT body_root = (f32[], f32[]) tuple(body_gte0, add)
}

condition (tuple_param: (f32[], f32[])) -> pred[] {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

ENTRY main {
  const1 = f32[] constant(1.0)
  const2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(const1, const2)
  ROOT while_op = (f32[], f32[]) while(tuple), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));
  HloInstruction* body_param = FindInstruction(module_.get(), "body_param");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* body_root = FindInstruction(module_.get(), "body_root");
  HloInstruction* cond_param = FindInstruction(module_.get(), "cond_param");
  HloInstruction* cond_constant =
      FindInstruction(module_.get(), "cond_constant");
  HloInstruction* constant1 = FindInstruction(module_.get(), "const1");
  HloInstruction* xla_while = FindInstruction(module_.get(), "while_op");
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
  std::string hlo_str = R"(
HloModule SequentialWhiles

body {
  param = (f32[], f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(param), index=0
  gte1 = f32[] get-tuple-element(param), index=1
  add = f32[] add(gte0, gte1)
  ROOT tuple = (f32[], f32[]) tuple(gte0, add)
}

condition {
  param = (f32[], f32[]) parameter(0)
  ROOT const = pred[] constant(false)
}

ENTRY main {
  const1 = f32[] constant(1.0)
  const2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(const1, const2)
  while0 = (f32[], f32[]) while(tuple), condition=condition, body=body
  while1 = (f32[], f32[]) while(while0), condition=condition, body=body
  ROOT while2 = (f32[], f32[]) while(while1), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* xla_while0 = FindInstruction(module_.get(), "while0");
  HloInstruction* xla_while1 = FindInstruction(module_.get(), "while1");
  HloInstruction* xla_while2 = FindInstruction(module_.get(), "while2");
  HloInstruction* constant1 = FindInstruction(module_.get(), "const1");
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
  std::string hlo_str = R"(
HloModule MultiLevelNestedWhile

condition {
  cond_param = (f32[]) parameter(0)
  ROOT cond_const = pred[] constant(false)
}

level0_body {
  level0_param = (f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(level0_param), index=0
  ROOT level0_root = (f32[]) tuple(gte0)
}

level1_body {
  level1_param = (f32[]) parameter(0)
  ROOT level1_root = (f32[]) while(level1_param), condition=condition, body=level0_body
}

level2_body {
  level2_param = (f32[]) parameter(0)
  level2_while = (f32[]) while(level2_param), condition=condition, body=level1_body
  gte1 = f32[] get-tuple-element(level2_while), index=0
  negate = f32[] negate(gte1)
  ROOT level2_root = (f32[]) tuple(negate)
}

ENTRY main {
  const1 = f32[] constant(1.0)
  tuple = (f32[]) tuple(const1)
  ROOT main_while = (f32[]) while(tuple), condition=condition, body=level2_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* level0_param = FindInstruction(module_.get(), "level0_param");
  HloInstruction* level0_root = FindInstruction(module_.get(), "level0_root");
  HloInstruction* level1_param = FindInstruction(module_.get(), "level1_param");
  HloInstruction* level1_root = FindInstruction(module_.get(), "level1_root");
  HloInstruction* level2_param = FindInstruction(module_.get(), "level2_param");
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
  // its parameter, and the outer body negates element 0 and passes it through
  // element 0 of the output.
  std::string hlo_str = R"(
HloModule NestedWhiles

condition {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_const = pred[] constant(false)
}

inner_body {
  inner_param = (f32[], f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(inner_param), index=0
  gte1 = f32[] get-tuple-element(inner_param), index=1
  add = f32[] add(gte0, gte1)
  ROOT inner_root = (f32[], f32[]) tuple(gte0, add)
}

outer_body {
  outer_param = (f32[], f32[]) parameter(0)
  gte2 = f32[] get-tuple-element(outer_param), index=0
  negate = f32[] negate(gte2)
  gte3 = f32[] get-tuple-element(outer_param), index=1
  outer_tuple = (f32[], f32[]) tuple(negate, gte3)
  ROOT nested_while = (f32[], f32[]) while(outer_tuple), condition=condition, body=inner_body
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(constant1, constant2)
  ROOT entry_while = (f32[], f32[]) while(tuple), condition=condition, body=outer_body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* inner_param = FindInstruction(module_.get(), "inner_param");
  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* nested_while = FindInstruction(module_.get(), "nested_while");
  HloInstruction* entry_while = FindInstruction(module_.get(), "entry_while");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
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
  // elements.
  std::string hlo_str = R"(
HloModule SwizzlingWhileSharedInput

condition {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_const = pred[] constant(false)
}

body {
  body_param = (f32[], f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(body_param), index=0
  gte1 = f32[] get-tuple-element(body_param), index=1
  ROOT body_root = (f32[], f32[]) tuple(gte1, gte0)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  tuple = (f32[], f32[]) tuple(constant1, constant1)
  ROOT while = (f32[], f32[]) while(tuple), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* body_param = FindInstruction(module_.get(), "body_param");
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);
  EXPECT_FALSE(analysis.ValueIsDefinedAt(body_param, /*index=*/{0}));
}

TEST_P(HloDataflowAnalysisTest, SwizzlingWhile) {
  // Test a while instruction with a body which permutes it's tuple parameter
  // elements.
  //
  std::string hlo_str = R"(
HloModule SwizzlingWhile

condition {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_const = pred[] constant(false)
}

body {
  body_param = (f32[], f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(body_param), index=0
  gte1 = f32[] get-tuple-element(body_param), index=1
  ROOT body_root = (f32[], f32[]) tuple(gte1, gte0)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(constant1, constant2)
  ROOT while = (f32[], f32[]) while(tuple), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* body_param = FindInstruction(module_.get(), "body_param");
  HloInstruction* xla_while = FindInstruction(module_.get(), "while");
  HloInstruction* cond_param = FindInstruction(module_.get(), "cond_param");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
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
  std::string hlo_str = R"(
HloModule ArraySelect

ENTRY main {
  p0 = pred[] constant(false)
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  ROOT select = f32[] select(p0, constant1, constant2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* select = FindInstruction(module_.get(), "select");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
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
  std::string hlo_str = R"(
HloModule BitcastDefinesValue

ENTRY main {
  constant = f32[] constant(1.0)
  ROOT bitcast = f32[] bitcast(constant)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* constant = FindInstruction(module_.get(), "constant");
  HloInstruction* bitcast = FindInstruction(module_.get(), "bitcast");
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
  std::string hlo_str = R"(
HloModule TupleCopy

ENTRY main {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(param0, param1)
  ROOT copy = (f32[], f32[]) copy(tuple)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param0 = FindInstruction(module_.get(), "param0");
  HloInstruction* param1 = FindInstruction(module_.get(), "param1");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* copy = FindInstruction(module_.get(), "copy");
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
  std::string hlo_str = R"(
HloModule OptimizationBarrier

ENTRY main {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(param0, param1)
  ROOT barrier = (f32[], f32[]) opt-barrier(tuple)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param0 = FindInstruction(module_.get(), "param0");
  HloInstruction* param1 = FindInstruction(module_.get(), "param1");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* barrier = FindInstruction(module_.get(), "barrier");
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
  std::string hlo_str = R"(
HloModule CopyStartAndCopyDone

ENTRY main {
  constant = f32[] constant(1.0)
  copy-start = (f32[], f32[], u32[]) copy-start(constant)
  ROOT copy-done = f32[] copy-done(copy-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* copy_start = FindInstruction(module_.get(), "copy-start");
  HloInstruction* copy_done = FindInstruction(module_.get(), "copy-done");
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
    async-update = ((f32[2,3]), f32[2,3], u32[]) custom-call-update(async-start)
    ROOT async-done = f32[2,3] custom-call-done(async-update)
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
  EXPECT_THAT(
      HloValuesAt(async_update, /*index=*/{2}),
      UnorderedElementsAre(&analysis.GetValueDefinedAt(async_start, {2})));

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
  %async-update = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-update(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-start)
  %negate_3 = f32[4096]{0} negate(f32[4096]{0} %b)
  %add_0 = f32[4096]{0} add(f32[4096]{0} %negate_2, f32[4096]{0} %negate_3)
  %async-done = f32[4096]{0} call-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-update)
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
  const HloInstruction* async_start =
      FindInstruction(module_.get(), "async-start");
  const HloInstruction* async_update =
      FindInstruction(module_.get(), "async-update");

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

  // Track origin of all components for AsyncCall
  EXPECT_THAT(HloValuesAt(async_start, {0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(a)));
  EXPECT_THAT(HloValuesAt(async_start, {0, 1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(b)));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, {2}));

  EXPECT_THAT(HloValuesAt(async_update, {0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(a)));
  EXPECT_THAT(HloValuesAt(async_update, {0, 1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(b)));
  EXPECT_THAT(
      HloValuesAt(async_update, {2}),
      UnorderedElementsAre(&analysis.GetValueDefinedAt(async_start, {2})));
}

TEST_P(HloDataflowAnalysisTest, AsyncCallExcludedThread) {
  std::string hlo_str = R"(
HloModule AsyncCall

%called_computation {
  %param_0 = f32[4096] parameter(0)
  %param_1 = f32[4096] parameter(1)
  %negate_0 = f32[4096] negate(%param_0)
  %negate_1 = f32[4096] negate(%param_1)
  ROOT %result.1 = f32[4096] add(%negate_0, %negate_1)
}

ENTRY %main {
  %a = f32[4096] parameter(0)
  %b = f32[4096] parameter(1)
  %async-start = ((f32[4096], f32[4096]), f32[4096], u32[]) call-start(%a, %b),
    to_apply=%called_computation, async_execution_thread="excluded_thread"
  %negate_2 = f32[4096] negate(f32[4096] %a)
  %async-update = ((f32[4096], f32[4096]), f32[4096], u32[]) call-update(%async-start)
  %async-done = f32[4096] call-done(%async-update)
  ROOT %add_1 = f32[4096] add(%negate_2, %async-done)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  bool ssa_form = GetParam();

  // Run analysis with excluded thread.
  auto analysis_or = HloDataflowAnalysis::Run(
      *module_, ssa_form, /*bitcast_defines_value=*/false, {"main"});
  TF_ASSERT_OK(analysis_or.status());
  analysis_ = std::move(analysis_or).value();
  const HloDataflowAnalysis& analysis = *analysis_;

  const HloInstruction* async_start =
      FindInstruction(module_.get(), "async-start");
  const HloInstruction* async_update =
      FindInstruction(module_.get(), "async-update");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  // AsyncStart defines a new value at {1} because the thread is excluded.
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, {1}));

  // AsyncUpdate at {1} should contain that new value (it forwards it).
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_update, {1}));
  EXPECT_THAT(
      HloValuesAt(async_update, {1}),
      UnorderedElementsAre(&analysis.GetValueDefinedAt(async_start, {1})));

  // AsyncDone output should contain that new value (it forwards it from
  // AsyncUpdate at {1}).
  EXPECT_FALSE(analysis.ValueIsDefinedAt(async_done, {}));
  EXPECT_THAT(
      HloValuesAt(async_done, {}),
      UnorderedElementsAre(&analysis.GetValueDefinedAt(async_start, {1})));
}

TEST_P(HloDataflowAnalysisTest, AsyncCallWithConditional) {
  std::string hlo_str = R"(
HloModule AsyncCall

%cond_computation.1 (param_0: f32[4096]) -> f32[4096] {
  ROOT %param_0_t = f32[4096]{0} parameter(0)
}

%cond_computation.2 (param_1: f32[4096]) -> f32[4096] {
  %param_0_f = f32[4096]{0} parameter(0)
  ROOT %negate = f32[4096]{0} negate(f32[4096]{0} %param_0_f)
}

%called_computation (param_0: pred[], param_1: f32[4096]) -> f32[4096] {
  %param_0 = pred[] parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %conditional = f32[4096]{0} conditional(pred[] %param_0, f32[4096]{0} %param_1, f32[4096]{0} %param_1), true_computation=%cond_computation.1, false_computation=%cond_computation.2
}

ENTRY %main (a: f32[4096], pred: pred[]) -> f32[4096] {
  %a = f32[4096]{0} parameter(1)
  %p = pred[] parameter(0)
  %async-start = ((pred[], f32[4096]{0}), f32[4096]{0}, u32[]) call-start(pred[] %p, f32[4096]{0} %a), to_apply=%called_computation
  ROOT %async-done = f32[4096]{0} call-done(%async-start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  const HloInstruction* a = FindInstruction(module_.get(), "a");
  const HloInstruction* p = FindInstruction(module_.get(), "p");
  const HloInstruction* param_0_t = FindInstruction(module_.get(), "param_0_t");
  EXPECT_THAT(HloValuesAt(param_0_t),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(a)));
  const HloInstruction* param_0_f = FindInstruction(module_.get(), "param_0_f");
  EXPECT_THAT(HloValuesAt(param_0_f),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(a)));
  const HloInstruction* param_0 = FindInstruction(module_.get(), "param_0");
  EXPECT_THAT(HloValuesAt(param_0),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(p)));
  const HloInstruction* conditional =
      FindInstruction(module_.get(), "conditional");
  if (ssa_form) {
    EXPECT_EQ(HloValuesAt(conditional).size(), 1);
    EXPECT_TRUE(HloValuesAt(conditional)[0]->is_phi());
  } else {
    EXPECT_EQ(HloValuesAt(conditional).size(), 2);
  }

  for (std::string async_name : {"async-start", "async-done"}) {
    const HloInstruction* async_op = FindInstruction(module_.get(), async_name);
    const HloComputation* called_computation =
        async_op->async_wrapped_instruction()->called_computations()[0];
    const HloInstruction* parameter0 =
        called_computation->parameter_instruction(0);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(parameter0));
    EXPECT_THAT(HloValuesAt(parameter0),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(p)));
    const HloInstruction* parameter1 =
        called_computation->parameter_instruction(1);
    EXPECT_FALSE(analysis.ValueIsDefinedAt(parameter1));
    EXPECT_THAT(HloValuesAt(parameter1),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(a)));
  }
}

TEST_P(HloDataflowAnalysisTest, TupleShapedAsyncOp) {
  std::string hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), (f32[2,3], f32[2,3]), u32[]) custom-call-start(p0), custom_call_target="foo"
    async-update = ((f32[2,3]), (f32[2,3], f32[2,3]), u32[]) custom-call-update(async-start)
    ROOT async-done = (f32[2,3], f32[2,3]) custom-call-done(async-update)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  bool ssa_form = GetParam();
  const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);

  const HloInstruction* async_start =
      FindInstruction(module_.get(), "async-start");
  const HloInstruction* async_update =
      FindInstruction(module_.get(), "async-update");
  const HloInstruction* async_done =
      FindInstruction(module_.get(), "async-done");

  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_update, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_done));

  const HloInstruction* p0 = FindInstruction(module_.get(), "p0");
  EXPECT_THAT(HloValuesAt(async_start, {0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(p0)));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(async_start, {2}));

  EXPECT_THAT(HloValuesAt(async_update, {0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(p0)));
  EXPECT_THAT(
      HloValuesAt(async_update, {2}),
      UnorderedElementsAre(&analysis.GetValueDefinedAt(async_start, {2})));
}

TEST_P(HloDataflowAnalysisTest, SendAndSendDone) {
  // Test that a Send forwards its operand to the output tuple at {0}.
  std::string hlo_str = R"(
HloModule SendAndSendDone

ENTRY main {
  param0 = f32[] parameter(0)
  tok0 = token[] after-all()
  send = (f32[], u32[], token[]) send(param0, tok0), channel_id=0
  ROOT send-done = token[] send-done(send), channel_id=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module_.get(), "param0");
  HloInstruction* send = FindInstruction(module_.get(), "send");
  HloInstruction* send_done = FindInstruction(module_.get(), "send-done");
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

TEST_P(HloDataflowAnalysisTest, SetDimensionSizeCreatesValue) {
  std::string hlo_str = R"(
HloModule SetDimensionSizeCreatesValue

ENTRY main {
  param = f32[42] parameter(0)
  size = s32[] constant(3)
  ROOT sds = f32[42] set-dimension-size(param, size), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* sds = FindInstruction(module_.get(), "sds");
  SCOPED_TRACE(module_->ToString());

  bool ssa_form = GetParam();
  {
    const HloDataflowAnalysis& analysis = RunAnalysis(ssa_form);
    EXPECT_EQ(analysis.values().size(), 3);

    EXPECT_TRUE(analysis.ValueIsDefinedAt(param));
    EXPECT_TRUE(analysis.ValueIsDefinedAt(sds));
    EXPECT_TRUE(analysis.GetValueDefinedAt(sds).live_out_of_module());
  }
}

TEST_P(HloDataflowAnalysisTest, RecvAndRecvDone) {
  // Test that a RecvDone forwards its operand tuple element at {0} to element
  // {0} of the output.
  std::string hlo_str = R"(
HloModule RecvAndRecvDone

ENTRY main {
  tok0 = token[] after-all()
  recv = (f32[], u32[], token[]) recv(tok0), channel_id=0
  ROOT recv-done = (f32[], token[]) recv-done(recv), channel_id=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* recv = FindInstruction(module_.get(), "recv");
  HloInstruction* recv_done = FindInstruction(module_.get(), "recv-done");
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
  std::string hlo_str = R"(
HloModule ElementwiseChainInterference

ENTRY main {
  param = f32[42] parameter(0)
  negate = f32[42] negate(param)
  exp = f32[42] exponential(negate)
  ROOT log = f32[42] log(exp)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* log = FindInstruction(module_.get(), "log");
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
  std::string hlo_str = R"(
HloModule MultipleEntryParameters_Sequential

ENTRY main {
  param0 = f32[42] parameter(0)
  param1 = f32[42] parameter(1)
  negate = f32[42] negate(param0)
  exp = f32[42] exponential(param1)
  ROOT add = f32[42] add(negate, exp)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloComputation* entry = module_->entry_computation();
  HloInstruction* param0 = FindInstruction(module_.get(), "param0");
  HloInstruction* param1 = FindInstruction(module_.get(), "param1");
  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* add = FindInstruction(module_.get(), "add");
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
  std::string hlo_str = R"(
HloModule WhileParameters_Sequential

%condition (cond_param: f32[]) -> pred[] {
  cond_param = f32[] parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

%body (body_param: f32[]) -> f32[] {
  constant = f32[] constant(1.0)
  exp = f32[] exponential(constant)
  body_param = f32[] parameter(0)
  ROOT add = f32[] add(exp, body_param)
  dead_constant = f32[] constant(1.0)
  dead_negate = f32[] negate(dead_constant)
}

ENTRY main {
  param = f32[] parameter(0)
  ROOT while = f32[] while(param), condition=%condition, body=%body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloComputation* entry = module_->entry_computation();
  HloComputation* condition = FindComputation(module_.get(), "condition");
  HloComputation* body = FindComputation(module_.get(), "body");

  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* xla_while = FindInstruction(module_.get(), "while");
  HloInstruction* cond_param = FindInstruction(module_.get(), "cond_param");
  HloInstruction* cond_constant =
      FindInstruction(module_.get(), "cond_constant");
  HloInstruction* body_param = FindInstruction(module_.get(), "body_param");
  HloInstruction* constant = FindInstruction(module_.get(), "constant");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* dead_constant =
      FindInstruction(module_.get(), "dead_constant");
  HloInstruction* dead_negate = FindInstruction(module_.get(), "dead_negate");
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
  std::string hlo_str = R"(
HloModule NonElementwiseOperand

ENTRY main {
  param = f32[42] parameter(0)
  exp = f32[42] exponential(param)
  negate = f32[42] negate(exp)
  ROOT reverse = f32[42] reverse(negate), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* reverse = FindInstruction(module_.get(), "reverse");
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
  std::string hlo_str = R"(
HloModule OverlappedValues

ENTRY main {
  param = f32[42] parameter(0)
  negate = f32[42] negate(param)
  exp = f32[42] exponential(param)
  ROOT add = f32[42] add(negate, exp)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* add = FindInstruction(module_.get(), "add");
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
  std::string hlo_str = R"(
HloModule OverlappedValuesSequentialOrder

ENTRY main {
  param = f32[42] parameter(0)
  negate = f32[42] negate(param)
  exp = f32[42] exponential(param)
  ROOT add = f32[42] add(negate, exp)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloComputation* entry = module_->entry_computation();
  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* add = FindInstruction(module_.get(), "add");
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
  std::string hlo_str = R"(
HloModule EmbeddedComputationInterference

embedded_computation {
  embedded_param = f32[42] parameter(0)
  ROOT embedded_log = f32[42] log(embedded_param)
}

ENTRY main {
  param = f32[42] parameter(0)
  negate = f32[42] negate(param)
  exp = f32[42] exponential(param)
  call = f32[42] call(exp), to_apply=embedded_computation
  ROOT add = f32[42] add(negate, call)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* negate = FindInstruction(module_.get(), "negate");
  HloInstruction* exp = FindInstruction(module_.get(), "exp");
  HloInstruction* embedded_log = FindInstruction(module_.get(), "embedded_log");
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

  std::string hlo_str = R"(
HloModule ConditionalWithIdentity

true_computation {
  ROOT true_param = f32[] parameter(0)
}

false_computation {
  ROOT false_param = f32[] parameter(0)
}

ENTRY main {
  my_pred = pred[] constant(true)
  constant1 = f32[] constant(56)
  constant2 = f32[] constant(12)
  ROOT conditional = f32[] conditional(my_pred, constant1, constant2), true_computation=true_computation, false_computation=false_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* pred = FindInstruction(module_.get(), "my_pred");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* conditional = FindInstruction(module_.get(), "conditional");
  HloInstruction* true_param = FindInstruction(module_.get(), "true_param");
  HloInstruction* false_param = FindInstruction(module_.get(), "false_param");
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

  std::string hlo_str = R"(
HloModule ConditionalTakingTupleOperand

true_computation {
  true_param = (f32[], f32[]) parameter(0)
  true_x = f32[] get-tuple-element(true_param), index=0
  true_y = f32[] get-tuple-element(true_param), index=1
  ROOT add = f32[] add(true_x, true_y)
}

false_computation {
  false_param = (f32[], f32[]) parameter(0)
  false_x = f32[] get-tuple-element(false_param), index=0
  false_y = f32[] get-tuple-element(false_param), index=1
  ROOT sub = f32[] subtract(false_x, false_y)
}

ENTRY main {
  my_pred = pred[] constant(true)
  constant1 = f32[] constant(56)
  constant2 = f32[] constant(12)
  tuple_operand = (f32[], f32[]) tuple(constant1, constant2)
  ROOT conditional = f32[] conditional(my_pred, tuple_operand, tuple_operand), true_computation=true_computation, false_computation=false_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* pred = FindInstruction(module_.get(), "my_pred");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* tuple_operand =
      FindInstruction(module_.get(), "tuple_operand");
  HloInstruction* conditional = FindInstruction(module_.get(), "conditional");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* sub = FindInstruction(module_.get(), "sub");
  HloInstruction* true_param = FindInstruction(module_.get(), "true_param");
  HloInstruction* false_param = FindInstruction(module_.get(), "false_param");
  HloInstruction* true_x = FindInstruction(module_.get(), "true_x");
  HloInstruction* true_y = FindInstruction(module_.get(), "true_y");
  HloInstruction* false_x = FindInstruction(module_.get(), "false_x");
  HloInstruction* false_y = FindInstruction(module_.get(), "false_y");
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

  std::string hlo_str = R"(
HloModule NestedConditionals

computation1 {
  comp1_param = f32[] parameter(0)
  ROOT comp1_ceil = f32[] ceil(comp1_param)
}

computation2 {
  comp2_param = f32[] parameter(0)
  ROOT comp2_floor = f32[] floor(comp2_param)
}

computation3 {
  comp3_param = f32[] parameter(0)
  ROOT comp3_negate = f32[] negate(comp3_param)
}

inner_conditional {
  param_cond = (pred[], f32[], f32[]) parameter(0)
  pred_cond = pred[] get-tuple-element(param_cond), index=0
  true_operand_cond = f32[] get-tuple-element(param_cond), index=1
  false_operand_cond = f32[] get-tuple-element(param_cond), index=2
  ROOT inner_conditional = f32[] conditional(pred_cond, true_operand_cond, false_operand_cond), true_computation=computation1, false_computation=computation2
}

ENTRY main {
  pred1 = pred[] constant(true)
  pred2 = pred[] constant(false)
  constant1 = f32[] constant(1.1)
  constant2 = f32[] constant(2.2)
  constant3 = f32[] constant(3.3)
  tuple_operand = (pred[], f32[], f32[]) tuple(pred2, constant1, constant2)
  ROOT conditional = f32[] conditional(pred1, tuple_operand, constant3), true_computation=inner_conditional, false_computation=computation3
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* pred1 = FindInstruction(module_.get(), "pred1");
  HloInstruction* pred2 = FindInstruction(module_.get(), "pred2");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* constant3 = FindInstruction(module_.get(), "constant3");
  HloInstruction* tuple_operand =
      FindInstruction(module_.get(), "tuple_operand");
  HloInstruction* comp1_ceil = FindInstruction(module_.get(), "comp1_ceil");
  HloInstruction* comp2_floor = FindInstruction(module_.get(), "comp2_floor");
  HloInstruction* comp3_negate = FindInstruction(module_.get(), "comp3_negate");
  HloInstruction* comp1_param = FindInstruction(module_.get(), "comp1_param");
  HloInstruction* comp2_param = FindInstruction(module_.get(), "comp2_param");
  HloInstruction* comp3_param = FindInstruction(module_.get(), "comp3_param");
  HloInstruction* param_cond = FindInstruction(module_.get(), "param_cond");
  HloInstruction* pred_cond = FindInstruction(module_.get(), "pred_cond");
  HloInstruction* true_operand_cond =
      FindInstruction(module_.get(), "true_operand_cond");
  HloInstruction* false_operand_cond =
      FindInstruction(module_.get(), "false_operand_cond");
  HloInstruction* conditional = FindInstruction(module_.get(), "conditional");
  HloInstruction* inner_conditional =
      FindInstruction(module_.get(), "inner_conditional");

  SCOPED_TRACE(module_->ToString());

  const HloDataflowAnalysis& analysis = RunAnalysis(GetParam());

  EXPECT_TRUE(analysis.ValueIsDefinedAt(pred1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(pred2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant1));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant2));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(constant3));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(tuple_operand));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(comp1_ceil));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(comp2_floor));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(comp3_negate));

  HloInstruction* computation1_param = comp1_param;
  HloInstruction* computation2_param = comp2_param;
  HloInstruction* computation3_param = comp3_param;
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
    EXPECT_THAT(HloValuesAt(inner_conditional),
                UnorderedElementsAre(&analysis.GetValueDefinedAt(comp1_ceil),
                                     &analysis.GetValueDefinedAt(comp2_floor)));
    EXPECT_THAT(
        HloValuesAt(conditional),
        UnorderedElementsAre(&analysis.GetValueDefinedAt(comp1_ceil),
                             &analysis.GetValueDefinedAt(comp2_floor),
                             &analysis.GetValueDefinedAt(comp3_negate)));
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

TEST_F(HloDataflowAnalysisTest, CombinedCollectivePermuteStartAndDone) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[2] parameter(0)
      p1 = f32[2] parameter(1)
      start = ((f32[2], f32[2]), (f32[2], f32[2])) collective-permute-start(p0, p1), source_target_pairs={{0,1},{1,0}}
      ROOT done = (f32[2], f32[2]) collective-permute-done(start)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_text));
  const HloDataflowAnalysis& analysis = RunAnalysis(/*ssa_form=*/false);
  absl::Status status = analysis.Verify();
  EXPECT_TRUE(status.ok()) << status;

  HloInstruction* done = module_->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  HloInstruction* param0 = start->mutable_operand(0);
  HloInstruction* param1 = start->mutable_operand(1);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{1, 0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{1, 1}));

  EXPECT_TRUE(analysis.ValueIsDefinedAt(done, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(done, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(done, /*index=*/{1}));

  EXPECT_THAT(
      analysis.GetValueDefinedAt(param0).GetUses(),
      UnorderedElementsAre(HloUse{start, 0, {}}, HloUse{done, 0, {0, 0}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(param1).GetUses(),
      UnorderedElementsAre(HloUse{start, 1, {}}, HloUse{done, 0, {0, 1}}));

  EXPECT_THAT(HloValuesAt(start, /*index=*/{0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param0, {})));
  EXPECT_THAT(HloValuesAt(start, /*index=*/{0, 1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param1, {})));
  EXPECT_THAT(HloValuesAt(done, /*index=*/{0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(start, {1, 0})));
  EXPECT_THAT(HloValuesAt(done, /*index=*/{1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(start, {1, 1})));
}

TEST_F(HloDataflowAnalysisTest, AllGatherStartAndDoneWithTuple) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[2] parameter(0)
      p1 = bf16[2] parameter(1)
      start = ((f32[2], bf16[2]), (f32[4], bf16[4])) all-gather-start(p0, p1), dimensions={0}
      ROOT done = (f32[4], bf16[4]) all-gather-done(start)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_text));
  const HloDataflowAnalysis& analysis = RunAnalysis(/*ssa_form=*/false);
  absl::Status status = analysis.Verify();
  EXPECT_TRUE(status.ok()) << status;

  HloInstruction* done = module_->entry_computation()->root_instruction();
  HloInstruction* start = done->mutable_operand(0);
  HloInstruction* param0 = start->mutable_operand(0);
  HloInstruction* param1 = start->mutable_operand(1);

  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{1}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(start, /*index=*/{0, 0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(start, /*index=*/{0, 1}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{1, 0}));
  EXPECT_TRUE(analysis.ValueIsDefinedAt(start, /*index=*/{1, 1}));

  EXPECT_TRUE(analysis.ValueIsDefinedAt(done, /*index=*/{}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(done, /*index=*/{0}));
  EXPECT_FALSE(analysis.ValueIsDefinedAt(done, /*index=*/{1}));

  EXPECT_THAT(
      analysis.GetValueDefinedAt(param0).GetUses(),
      UnorderedElementsAre(HloUse{start, 0, {}}, HloUse{done, 0, {0, 0}}));
  EXPECT_THAT(
      analysis.GetValueDefinedAt(param1).GetUses(),
      UnorderedElementsAre(HloUse{start, 1, {}}, HloUse{done, 0, {0, 1}}));

  EXPECT_THAT(HloValuesAt(start, /*index=*/{0, 0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param0, {})));
  EXPECT_THAT(HloValuesAt(start, /*index=*/{0, 1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(param1, {})));
  EXPECT_THAT(HloValuesAt(done, /*index=*/{0}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(start, {1, 0})));
  EXPECT_THAT(HloValuesAt(done, /*index=*/{1}),
              UnorderedElementsAre(&analysis.GetValueDefinedAt(start, {1, 1})));
}

TEST_F(HloDataflowAnalysisTest, OptimizePhiInNonSingletonValueSets) {
  // We have an identity while loop (i.e. just passes its parameter through),
  // that creates a phi value.
  // We then call a subcomputation with the while value and a constant.
  // The subcomputation therefore has a ValueSet with {phi, constant}.
  // The phi value should be optimized to the while's input (a constant).
  // The parameter ValueSet should then contain {constant1, constant2}.
  // Note that if when optimizing phi values, we skip non-singleton ValueSets,
  // then the phi value won't be optimized, and we'll try to return a dangling
  // pointer to the phi value.
  const char* kModule = R"(
    HloModule OptimizePhiInNonSingletonValueSets

    subcomp {
      sub_param = f32[] parameter(0)
      ROOT negate = f32[] negate(sub_param)
    }

    body {
      ROOT body_param = f32[] parameter(0)
    }

    condition {
      cond_param = f32[] parameter(0)
      ROOT cond_root = pred[] constant(false)
    }

    ENTRY entry {
      constant1 = f32[] constant(1.0)
      constant2 = f32[] constant(2.0)
      while = f32[] while(constant1), condition=condition, body=body
      call1 = f32[] call(while), to_apply=subcomp
      call2 = f32[] call(constant2), to_apply=subcomp
      ROOT tuple = (f32[], f32[]) tuple(call1, call2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));

  // We must not run FlattenCallGraph, otherwise the calls are inlined/cloned
  // and the parameter won't have two values in its ValueSet. So we call
  // HloDataflowAnalysis::Run directly.
  // Note that we can have more complex cases where we still have this situation
  // after FlattenCallGraph, for example, when calling subcomp from ENTRY as
  // above, and also from another computation, like another while body.
  TF_ASSERT_OK_AND_ASSIGN(auto analysis_ptr,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  const HloDataflowAnalysis& analysis = *analysis_ptr;

  const HloInstruction* sub_param = FindInstruction(module.get(), "sub_param");
  const HloInstruction* constant1 = FindInstruction(module.get(), "constant1");
  const HloInstruction* constant2 = FindInstruction(module.get(), "constant2");

  const HloValueSet& param_set = analysis.GetValueSet(sub_param);
  const HloValue& val1 = analysis.GetValueDefinedAt(constant1);
  const HloValue& val2 = analysis.GetValueDefinedAt(constant2);

  EXPECT_THAT(param_set.values(), UnorderedElementsAre(&val1, &val2));
}

INSTANTIATE_TEST_SUITE_P(HloDataflowAnalysisInstantiation,
                         HloDataflowAnalysisTest,
                         ::testing::Values(false, true));

std::unique_ptr<HloDataflowAnalysis> RunAnalysis(const HloModule& module) {
  return HloDataflowAnalysis::Run(module, /*ssa_form=*/false,
                                  /*bitcast_defines_value=*/false)
      .value();
}

using DoesNotUseOperandBufferTest = HloHardwareIndependentTestBase;

TEST_F(DoesNotUseOperandBufferTest, GetTupleElement) {
  std::string hlo_str = R"(
HloModule GetTupleElement

ENTRY main {
  my_tuple = (f32[8], f32[8]) parameter(0)
  gte0 = f32[8] get-tuple-element(my_tuple), index=0
  gte1 = f32[8] get-tuple-element(my_tuple), index=1
  ROOT add = f32[8] add(gte0, gte1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* tuple = FindInstruction(module.get(), "my_tuple");
  HloInstruction* gte0 = FindInstruction(module.get(), "gte0");
  HloInstruction* gte1 = FindInstruction(module.get(), "gte1");

  auto dataflow_analysis = RunAnalysis(*module);

  // GetTupleElement instructions only access the top-level buffer of their
  // operand.
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {0}, gte0));
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {1}, gte1));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {}, gte0));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {}, gte1));
}

TEST_F(DoesNotUseOperandBufferTest, FusedDynamicUpdateSlice) {
  std::string hlo_str = R"(
HloModule FusedDynamicUpdateSlice

fused_computation {
  p0 = (f32[8], f32[8]) parameter(0)
  gte1 = f32[8] get-tuple-element(p0), index=1
  p1 = f32[3] parameter(1)
  p2 = s32[] parameter(2)
  ROOT dynamic_update_slice = f32[8] dynamic-update-slice(gte1, p1, p2)
}

ENTRY main {
  my_tuple = (f32[8], f32[8]) parameter(0)
  gte0 = f32[8] get-tuple-element(my_tuple), index=0
  starts = s32[] constant(2)
  update = f32[3] constant({2, 2, 2})
  fusion = f32[8] fusion(my_tuple, update, starts), kind=kLoop, calls=fused_computation
  ROOT result_tuple = (f32[8], f32[8]) tuple(gte0, fusion)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* tuple = FindInstruction(module.get(), "my_tuple");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  // The fusion instruction never uses tuple element 0, but does use element 1.
  EXPECT_TRUE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {0}, fusion));
  EXPECT_FALSE(dataflow_analysis->DoesNotUseOperandBuffer(tuple, {1}, fusion));
}

// Similar to FusedDynamicUpdateSlice above, but tests indirect uses of the
// parameter tuple.
TEST_F(DoesNotUseOperandBufferTest, IndirectUses) {
  std::string hlo_str = R"(
HloModule IndirectUses

fused_computation {
  p0 = f32[8] parameter(0)
  p1 = f32[3] parameter(1)
  p2 = s32[] parameter(2)
  ROOT dynamic_update_slice = f32[8] dynamic-update-slice(p0, p1, p2)
}

ENTRY main {
  tuple_param = (f32[8], f32[8]) parameter(0)
  t0 = f32[8] get-tuple-element(tuple_param), index=0
  t1 = f32[8] get-tuple-element(tuple_param), index=1
  my_tuple = (f32[8], f32[8]) tuple(t1, t0)
  gte0 = f32[8] get-tuple-element(my_tuple), index=0
  gte1 = f32[8] get-tuple-element(my_tuple), index=1
  starts = s32[] constant(2)
  update = f32[3] constant({2, 2, 2})
  fusion = f32[8] fusion(gte1, update, starts), kind=kLoop, calls=fused_computation
  ROOT result_tuple = (f32[8], f32[8]) tuple(gte0, fusion)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* tuple_param = FindInstruction(module.get(), "tuple_param");
  HloInstruction* tuple = FindInstruction(module.get(), "my_tuple");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

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

class CanShareOperandBufferWithUserTest
    : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseSameShape) {
  std::string hlo_str = R"(
HloModule ElementWiseSameShape

ENTRY main {
  param = f32[8] parameter(0)
  exp = f32[8] exponential(param)
  ROOT log = f32[8] log(exp)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module.get(), "param");
  HloInstruction* exp = FindInstruction(module.get(), "exp");
  HloInstruction* log = FindInstruction(module.get(), "log");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param, {}, exp, {}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(exp, {}, log, {},
                                                               &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest,
       NonElementwiseLoopFusionCantAliasOperandBuffer) {
  std::string hlo_str = R"(
HloModule NonElementwiseLoopFusionCantAliasOperandBuffer

fused_computation {
  p0 = f32[2,2] parameter(0)
  neg = f32[2,2] negate(p0)
  ROOT reverse = f32[2,2] reverse(neg), dimensions={0,1}
}

ENTRY main {
  param0 = f32[2,2] parameter(0)
  ROOT fusion = f32[2,2] fusion(param0), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* param0 = FindInstruction(module.get(), "param0");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      param0, {}, fusion, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest,
       MultiOutputFusionCanAliasOperandBuffer) {
  std::string hlo_str = R"(
HloModule MultiOutputFusionCanAliasOperandBuffer

fused_computation {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  copy0 = f32[8] copy(p0)
  copy1 = f32[8] copy(p1)
  ROOT result_tuple = (f32[8], f32[8]) tuple(copy1, copy0)
}

ENTRY main {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  ROOT fusion = (f32[8], f32[8]) fusion(param0, param1), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* param0 = FindInstruction(module.get(), "param0");
  HloInstruction* param1 = FindInstruction(module.get(), "param1");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param0, {}, fusion, {0}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param0, {}, fusion, {1}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param1, {}, fusion, {0}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param1, {}, fusion, {1}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest,
       ElementwiseLoopFusionCantAliasOperandBuffer) {
  std::string hlo_str = R"(
HloModule ElementwiseLoopFusionCantAliasOperandBuffer

fused_computation {
  p0 = f32[2,2] parameter(0)
  neg = f32[2,2] negate(p0)
  ROOT exp = f32[2,2] exponential(neg)
}

ENTRY main {
  one = f32[] constant(1)
  operand = f32[2,2] broadcast(one), dimensions={}
  ROOT fusion = f32[2,2] fusion(operand), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* operand = FindInstruction(module.get(), "operand");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand, {}, fusion, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest,
       CanShareOperandWhenDynamicUpdateSliceIsFedByDynamicSliceWithSameIndex) {
  std::string hlo_str = R"(
HloModule CanShareOperandWhenDynamicUpdateSliceIsFedByDynamicSliceWithSameIndex

fused_computation {
  p0 = f32[2,2] parameter(0)
  zero = s64[] constant(0)
  ds = f32[1,2] dynamic-slice(p0, zero, zero), dynamic_slice_sizes={1,2}
  ROOT dus = f32[2,2] dynamic-update-slice(p0, ds, zero, zero)
}

ENTRY main {
  param0 = f32[2,2] parameter(0)
  ROOT fusion = f32[2,2] fusion(param0), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module.get(), "param0");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param, {}, fusion, {}, &alias_info_));
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
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param, {}, fusion, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseDifferentShape) {
  std::string hlo_str = R"(
HloModule ElementWiseDifferentShape

ENTRY main {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  ROOT result = pred[8] compare(param0, param1), direction=EQ
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* param0 = FindInstruction(module.get(), "param0");
  HloInstruction* param1 = FindInstruction(module.get(), "param1");
  HloInstruction* result = FindInstruction(module.get(), "result");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      param0, {}, result, {}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      param1, {}, result, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, CopyShares) {
  std::string hlo_str = R"(
HloModule CopyShares

ENTRY main {
  param = f32[8] parameter(0)
  exp = f32[8] exponential(param)
  ROOT copy = f32[8] copy(exp)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* param = FindInstruction(module.get(), "param");
  HloInstruction* exp = FindInstruction(module.get(), "exp");
  HloInstruction* copy = FindInstruction(module.get(), "copy");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      param, {}, exp, {}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      exp, {}, copy, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDynamicUpdateSlice) {
  std::string hlo_str = R"(
HloModule FusedDynamicUpdateSlice

fused_computation {
  p0 = (f32[8], f32[8]) parameter(0)
  gte1 = f32[8] get-tuple-element(p0), index=1
  p1 = f32[3] parameter(1)
  p2 = s32[] parameter(2)
  ROOT dynamic_update_slice = f32[8] dynamic-update-slice(gte1, p1, p2)
}

ENTRY main {
  my_tuple = (f32[8], f32[8]) parameter(0)
  gte0 = f32[8] get-tuple-element(my_tuple), index=0
  starts = s32[] constant(2)
  update = f32[3] constant({2, 2, 2})
  fusion = f32[8] fusion(my_tuple, update, starts), kind=kLoop, calls=fused_computation
  ROOT result_tuple = (f32[8], f32[8]) tuple(gte0, fusion)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* tuple = FindInstruction(module.get(), "my_tuple");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  // The fusion instruction can share with tuple element 1.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      tuple, {0}, fusion, {}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      tuple, {1}, fusion, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest,
       FusedDynamicUpdateSliceWithConvertCanShare) {
  std::string hlo_str = R"(
HloModule FusedDynamicUpdateSliceWithConvertCanShare

fused_computation {
  p0 = f32[8] parameter(0)
  convert1 = bf16[8] convert(p0)
  starts = s32[] constant(2)
  update = f32[3] constant({2, 2, 2})
  dynamic_update_slice = bf16[8] dynamic-update-slice(convert1, update, starts)
  ROOT convert2 = f32[8] convert(dynamic_update_slice)
}

ENTRY main {
  my_tuple = (f32[8], f32[8]) parameter(0)
  gte0 = f32[8] get-tuple-element(my_tuple), index=0
  gte1 = f32[8] get-tuple-element(my_tuple), index=1
  fusion = f32[8] fusion(gte1), kind=kLoop, calls=fused_computation
  ROOT result_tuple = (f32[8], f32[8]) tuple(gte0, fusion)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      gte1, {}, fusion, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, DynamicUpdateSliceCanShare) {
  std::string hlo_str = R"(
HloModule DynamicUpdateSliceCanShare

ENTRY main {
  data = f32[1,8] parameter(0)
  update = f32[1,4] parameter(1)
  start = s32[2] parameter(2)
  ROOT dus = f32[1,8] dynamic-update-slice(data, update, start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* data = FindInstruction(module.get(), "data");
  HloInstruction* update = FindInstruction(module.get(), "update");
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* dus = FindInstruction(module.get(), "dus");

  auto dataflow_analysis = RunAnalysis(*module);

  // The DynamicUpdateSlice instruction can share with the data operand, but not
  // with update or start.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      data, {}, dus, {}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      update, {}, dus, {}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      start, {}, dus, {}, &alias_info_));
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
      operand_param, {}, scatter, {}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      indices_param, {}, scatter, {}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates_param, {}, scatter, {}, &alias_info_));
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
      operand0_param, {}, scatter, {0}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand0_param, {}, scatter, {1}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand1_param, {}, scatter, {0}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand1_param, {}, scatter, {1}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      indices_param, {}, scatter, {0}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      indices_param, {}, scatter, {1}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates0_param, {}, scatter, {0}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates0_param, {}, scatter, {1}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates1_param, {}, scatter, {0}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      updates1_param, {}, scatter, {1}, &alias_info_));
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
      lhs_param, {}, triangular_solve, {}, &alias_info_));
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      rhs_param, {}, triangular_solve, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, SortCanShare) {
  std::string hlo_str = R"(
HloModule SortCanShare

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT cmp = pred[] compare(p0, p1), direction=LT
}

ENTRY main {
  keys = f32[8] parameter(0)
  ROOT sort = f32[8] sort(keys), dimensions={0}, to_apply=compare
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* keys = FindInstruction(module.get(), "keys");
  HloInstruction* sort = FindInstruction(module.get(), "sort");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      keys, {}, sort, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, SortCanShareWithTupleUser) {
  std::string hlo_str = R"(
HloModule SortCanShareWithTupleUser

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  ROOT cmp = pred[] compare(p0, p2), direction=LT
}

ENTRY main {
  keys = f32[8] parameter(0)
  values = f32[8] parameter(1)
  ROOT sort = (f32[8], f32[8]) sort(keys, values), dimensions={0}, to_apply=compare
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* keys = FindInstruction(module.get(), "keys");
  HloInstruction* values = FindInstruction(module.get(), "values");
  HloInstruction* sort = FindInstruction(module.get(), "sort");

  auto dataflow_analysis = RunAnalysis(*module);

  // The buffer for the keys can be shared with the first tuple entry.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      keys, {}, sort, {0}, &alias_info_));
  // The buffer for the values can be shared with the second tuple entry.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      values, {}, sort, {1}, &alias_info_));
  // Verify that the buffers are not shared with the "wrong" tuple entry.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      keys, {}, sort, {1}, &alias_info_));
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      values, {}, sort, {0}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDotAdd) {
  std::string hlo_str = R"(
HloModule FusedDotAdd

fused_computation {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  p2 = f32[2,2] parameter(2)
  dot = f32[2,2] dot(p1, p2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT add = f32[2,2] add(dot, p0)
}

ENTRY main {
  a = f32[2,2] constant({{1.0, 0.0}, {0.0, 1.0}})
  b = f32[2,2] constant({{2.0, 2.0}, {2.0, 2.0}})
  one = f32[] constant(1)
  add_operand = f32[2,2] broadcast(one), dimensions={}
  ROOT fusion = f32[2,2] fusion(add_operand, a, b), kind=kOutput, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* add_operand = FindInstruction(module.get(), "add_operand");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  // Output fused dot add should be able to share buffer with 'add_operand'.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      add_operand, {}, fusion, {}, &alias_info_));
}

TEST_F(CanShareOperandBufferWithUserTest, OutputFusionCantAliasOperandBuffer) {
  std::string hlo_str = R"(
HloModule OutputFusionCantAliasOperandBuffer

fused_computation {
  p0 = f32[2,2] parameter(0)
  reverse = f32[2,2] reverse(p0), dimensions={0,1}
  two = f32[2,2] constant({{2.0, 2.0}, {2.0, 2.0}})
  ROOT add = f32[2,2] add(reverse, two)
}

ENTRY main {
  one = f32[] constant(1)
  operand = f32[2,2] broadcast(one), dimensions={}
  ROOT fusion = f32[2,2] fusion(operand), kind=kOutput, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* operand = FindInstruction(module.get(), "operand");
  HloInstruction* fusion = FindInstruction(module.get(), "fusion");

  auto dataflow_analysis = RunAnalysis(*module);

  // Output fused operand->reverse->add cannot alias operand buffer 'operand'.
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand, {}, fusion, {}, &alias_info_));
}

class CustomAliasInfo : public AliasInfo {
 public:
  std::optional<bool> MayAlias(const HloInstruction*, const ShapeIndex&,
                               const HloInstruction* fusion,
                               const ShapeIndex&) const override {
    return fusion->IsLoopFusion();
  }
};

TEST_F(CanShareOperandBufferWithUserTest,
       FusionCanShareBufferCustomizedAliasInfo) {
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
  auto dataflow_analysis = RunAnalysis(*module);

  CustomAliasInfo alias_info;
  EXPECT_FALSE(dataflow_analysis->CanShareOperandBufferWithUser(
      operand, {}, fusion, {}, &alias_info));
}

TEST_F(CanShareOperandBufferWithUserTest, WhileCanShare) {
  std::string hlo_str = R"(
HloModule WhileCanShare

and_computation {
  p0 = pred[] parameter(0)
  p1 = pred[] parameter(1)
  ROOT result = pred[] and(p0, p1)
}

cond_computation {
  cond_data = f32[8] parameter(0)
  compare = pred[8] compare(cond_data, cond_data), direction=EQ
  true_val = pred[] constant(true)
  ROOT reduce = pred[] reduce(compare, true_val), dimensions={0}, to_apply=and_computation
}

body_computation {
  body_data = f32[8] parameter(0)
  ROOT add = f32[8] add(body_data, body_data)
}

ENTRY main {
  main_data = f32[8] parameter(0)
  ROOT whil = f32[8] while(main_data), condition=cond_computation, body=body_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* data = FindInstruction(module.get(), "main_data");
  HloInstruction* whil = FindInstruction(module.get(), "whil");

  auto dataflow_analysis = RunAnalysis(*module);

  // The While instruction can share with the data operand.
  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      data, {}, whil, {}, &alias_info_));
}

// Tests that Call can alias operand buffer if the only use of the operand
// in the called computation is an elementwise instruction.
TEST_F(CanShareOperandBufferWithUserTest, CallToComputationWithFusionRoot) {
  std::string hlo_str = R"(
HloModule CallToComputationWithFusionRoot

fused_computation {
  p0 = f32[8] parameter(0)
  one = f32[] constant(1)
  ones = f32[8] broadcast(one), dimensions={}
  ROOT add = f32[8] add(p0, ones)
}

sub_computation {
  sub_param = f32[8] parameter(0)
  ROOT fusion = f32[8] fusion(sub_param), kind=kLoop, calls=fused_computation
}

ENTRY main {
  param = f32[8] parameter(0)
  reverse = f32[8] reverse(param), dimensions={0}
  ROOT call = f32[8] call(reverse), to_apply=sub_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_str, GetModuleConfigForTest()));

  HloInstruction* reverse = FindInstruction(module.get(), "reverse");
  HloInstruction* call = FindInstruction(module.get(), "call");

  auto dataflow_analysis = RunAnalysis(*module);

  EXPECT_TRUE(dataflow_analysis->CanShareOperandBufferWithUser(
      reverse, {}, call, {}, &alias_info_));
}

class GetInPlaceInputOutputPairsTest : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

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

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(dus);
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

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, AsyncStartWithOutputOperandAliasing) {
  const char* kModule = R"(
  HloModule module

  %async_computation {
    %param_0.2 = (f32[8,4,1], (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)})) parameter(0)
    %get-tuple-element = f32[8,4,1] get-tuple-element(%param_0.2), index=0
    ROOT %all-to-all0.0 = f32[8,4,1] all-to-all(%get-tuple-element), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}
  }

  ENTRY %Comp_spmd {
    %param = f32[8,4,1] parameter(0)
    %copy = f32[8,4,1] copy(%param)
    %custom-call = (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)}) custom-call(), custom_call_target="BarrierStart"
    %tuple = (f32[8,4,1], (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)})) tuple(%copy, %custom-call)
    %all-to-all-start.1 = (((f32[8,4,1], (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)}))), f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)}) async-start(%tuple), output_to_operand_aliasing={{0,0,1,0}: (0, {1,0}), {0,0,1,1}: (0, {1,1}), {0,0,1,2}: (0, {1,2})}, calls=%async_computation
    %all-to-all-done = f32[8,4,1] async-done(%all-to-all-start.1)
    ROOT %copy.1 = f32[8,4,1] copy(%all-to-all-done)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* async_start = module->entry_computation()
                                    ->root_instruction()
                                    ->mutable_operand(0)
                                    ->mutable_operand(0);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(async_start);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back(
      {HloOperandIndex{0, {1, 0}}, {0, 0, 1, 0}});  // annotated
  expected_pairs.push_back(
      {HloOperandIndex{0, {1, 1}}, {0, 0, 1, 1}});  // annotated
  expected_pairs.push_back(
      {HloOperandIndex{0, {1, 2}}, {0, 0, 1, 2}});  // annotated
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

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
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

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
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
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);

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

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
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
      alias_info_.GetInPlaceInputOutputPairs(inner_fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> inner_expected_pairs;
  inner_expected_pairs.push_back({HloOperandIndex{1, {1}}, {1}});
  EXPECT_EQ(inner_in_place_pairs, inner_expected_pairs);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{1, {0}}, {2}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, NestedLoopWithAliasingInDUSFusion) {
  const char* kModule = R"(
    HloModule test

    copy_fusion {
      input = s8[8,256,1,256] parameter(0)
      ROOT copy.3 = s8[8,256,1,256] copy(input)
    }

    fused_computation.0 {
      p0 = (s8[8,256,1,256],s8[1,256,1,256]) parameter(0)
      gte0 = s8[8,256,1,256] get-tuple-element(p0), index=0
      gte1 = s8[1,256,1,256] get-tuple-element(p0), index=1
      fusion = s8[8,256,1,256] fusion(gte0), kind=kLoop, output_to_operand_aliasing={{}: (0, {})}, calls=copy_fusion
      p1 = s8[1,256,1,256] parameter(1)
      added = s8[1,256,1,256] add(gte1, p1)
      p2 = s32[] parameter(2)
      c0 = s32[] constant(0)
      ROOT dynamic-update-slice.0 = s8[8,256,1,256] dynamic-update-slice(fusion, added, p2, c0, c0, c0)
    }

    ENTRY test {
      p0 = (s8[8,256,1,256],s8[1,256,1,256]) parameter(0)
      p1 = s8[1,256,1,256] parameter(1)
      p2 = s32[] parameter(2)
      ROOT fusion = s8[8,256,1,256] fusion(p0, p1, p2), kind=kLoop, calls=fused_computation.0
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {0}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, DUSLoopFusionWithCollective) {
  const char* kModule = R"(
    HloModule LoopFusionAllReduce

    fused_computation.1 {
      p0 = bf16[2,8192,6144]{2,1,0:T(8,128)(2,1)} parameter(0)
      ROOT slice = bf16[2,2048,6144]{2,1,0:T(8,128)(2,1)} slice(p0), slice={[0:2], [6144:8192], [0:6144]}
    }

    fused_computation.2 {
      p0 = bf16[2,8192]{1,0:T(2,128)(2,1)} parameter(0)
      ROOT slice = bf16[2,2048]{1,0:T(2,128)(2,1)} slice(p0), slice={[0:2], [6144:8192]}
    }

    sum {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT add = bf16[] add(lhs, rhs)
    }

    fused_computation {
      p0 = bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)} parameter(0)
      p1 = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} parameter(1)
      p2 = bf16[2,8192,6144]{2,1,0:T(8,128)(2,1)} parameter(2)
      p3 = bf16[2,8192]{1,0:T(2,128)(2,1)} parameter(3)
      fusion.1 = bf16[2,2048,6144]{2,1,0:T(8,128)(2,1)} fusion(p2), kind=kLoop, calls=fused_computation.1
      bitcast = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} bitcast(fusion.1)
      fusion.2 = bf16[2,2048]{1,0:T(2,128)(2,1)} fusion(p3), kind=kLoop, calls=fused_computation.2
      broadcast = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} broadcast(fusion.2), dimensions={1,2}
      multiply = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)S(1)} multiply(bitcast, broadcast)
      all-reduce = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} all-reduce(p1), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum
      c0 = u32[] constant(0)
      c1 = u32[] constant(4096)
      dynamic-update-slice = bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)} dynamic-update-slice(p0, all-reduce, c0, c0, c1, c0)
      ROOT tuple = (bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)S(1)}, bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)}) tuple(multiply, dynamic-update-slice)
    }

    ENTRY entry {
      p0 = bf16[2,8192,6144]{2,1,0:T(8,128)(2,1)} parameter(0)
      p1 = bf16[2,8192]{1,0:T(2,128)(2,1)} parameter(1)
      p2 = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} parameter(2)
      p3 = bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)} parameter(3)
      ROOT fusion = (bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)S(1)}, bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)}) fusion(p3, p2, p0, p1), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {1}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, DUSOutputFusionWithCollective) {
  const char* kModule = R"(
    HloModule OutputFusionAllReduce

    fused_computation.0 {
      p0 = bf16[4096,9216]{1,0:T(8,128)(2,1)} parameter(0)
      ROOT slice = bf16[1024,9216]{1,0:T(8,128)(2,1)} slice(p0), slice={[3072:4096], [0:9216]}
    }

    fused_computation.1 {
      p0 = s8[9216,6144]{1,0:T(8,128)(4,1)S(1)} parameter(0)
      ROOT bitcast = s8[9216,6144]{1,0:T(8,128)(4,1)} bitcast(p0)
    }

    add {
      x = bf16[] parameter(0)
      y = bf16[] parameter(1)
      ROOT add = bf16[] add(x, y)
    }

    fused_computation {
      p0 = bf16[4096,6144]{1,0:T(8,128)(2,1)} parameter(0)
      p1 = bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)} parameter(1)
      p2 = bf16[4096,9216]{1,0:T(8,128)(2,1)} parameter(2)
      p3 = s8[9216,6144]{1,0:T(8,128)(4,1)S(1)} parameter(3)
      fusion1 = bf16[1024,9216]{1,0:T(8,128)(2,1)} fusion(p2), kind=kLoop, calls=fused_computation.0
      fusion2 = s8[9216,6144]{1,0:T(8,128)(4,1)} fusion(p3), kind=kLoop, calls=fused_computation.1
      convolution = bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)} convolution(fusion1, fusion2), dim_labels=bf_io->bf
      all-reduce = bf16[1024,6144]{1,0:T(8,128)(2,1)} all-reduce(p1), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
      c1 = u32[] constant(2048)
      c0 = u32[] constant(0)
      dynamic-update-slice = bf16[4096,6144]{1,0:T(8,128)(2,1)} dynamic-update-slice(p0, all-reduce, c1, c0)
      ROOT tuple = (bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)}, bf16[4096,6144]{1,0:T(8,128)(2,1)}) tuple(convolution, dynamic-update-slice)
    }

    ENTRY entry {
      p0 = bf16[4096,9216]{1,0:T(8,128)(2,1)} parameter(0)
      p1 = s8[9216,6144]{1,0:T(8,128)(4,1)S(1)} parameter(1)
      p2 = bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)} parameter(2)
      p3 = bf16[4096,6144]{1,0:T(8,128)(2,1)} parameter(3)
      ROOT fusion = (bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)}, bf16[4096,6144]{1,0:T(8,128)(2,1)}) fusion(p3, p2, p0, p1), kind=kOutput, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {1}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, DUSLoopFusionWithBitcast) {
  const char* kModule = R"(
    HloModule DUSLoopFusionWithBitcast

    fused_dynamic_update_slice {
      param_1.133 = bf16[32,1,4096,18432]{2,3,1,0} parameter(1)
      bitcast.8539.1 = bf16[32,1,18432,4096]{3,2,1,0} bitcast(param_1.133)
      param_0.168 = bf16[1,4096,18432]{1,0,2} parameter(0)
      bitcast.8543.1 = bf16[1,1,18432,4096]{3,2,1,0} bitcast(param_0.168)
      param_2.98 = s32[] parameter(2)
      constant_2153_8 = s32[] constant(0)
      compare.753.6 = pred[] compare(param_2.98, constant_2153_8), direction=LT
      constant_2154_12 = s32[] constant(96)
      add.950.6 = s32[] add(param_2.98, constant_2154_12)
      select.883.5 = s32[] select(compare.753.6, add.950.6, param_2.98)
      ROOT dynamic-update-slice.178.1 = bf16[32,1,18432,4096]{3,2,1,0} dynamic-update-slice(bitcast.8539.1, bitcast.8543.1, select.883.5, constant_2153_8, constant_2153_8, /*index=5*/constant_2153_8)
    }

    ENTRY entry {
      p0 = bf16[1,4096,18432]{1,0,2} parameter(0)
      p1 = bf16[32,1,4096,18432]{2,3,1,0} parameter(1)
      p2 = s32[] parameter(2)
      ROOT fusion1 = bf16[32,1,18432,4096]{3,2,1,0} fusion(p0, p1, p2), kind=kLoop, calls=fused_dynamic_update_slice
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  // p1 should be aliased with fusion1
  expected_pairs.push_back({HloOperandIndex{1, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, RaggedAllToAll) {
  const char* kModule = R"(
HloModule RaggedAllToAll, is_scheduled=true

ENTRY AllToAll {
  input = f32[24,56,119] parameter(0)
  copy-start = (f32[24,56,119], f32[24,56,119], u32[]) copy-start(input)
  c0 = f32[] constant(0)
  output = f32[24,56,119] broadcast(c0), dimensions={}
  input_offsets = s32[8] parameter(1)
  send_sizes = s32[8] parameter(2)
  output_offsets = s32[8] parameter(3)
  recv_sizes = s32[8] parameter(4)
  copy-done = f32[24,56,119] copy-done(copy-start)
  ROOT ra2a = f32[24,56,119] ragged-all-to-all(copy-done, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3,4,5,6,7}}
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  HloInstruction* ragged_all_to_all =
      module->entry_computation()->root_instruction();

  auto in_place_pairs =
      alias_info_.GetInPlaceInputOutputPairs(ragged_all_to_all);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{1, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Test to check that the dataflow analysis works with a module that has scalar
// bitcast user.
TEST_P(HloDataflowAnalysisTest, b409416499) {
  const char* after_layout_bitcast = R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)})->(s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)})}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, allow_spmd_sharding_propagation_to_output={true,true,true,true}, num_partitions=4
  %region_0.13_spmd (param.1: s32[]) -> s32[] {
    %param.1 = s32[]{:T(128)} parameter(0), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/while"}
    %constant.1 = s32[]{:T(128)} constant(1)
    ROOT %add.0 = s32[]{:T(128)} add(%param.1, %constant.1), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/while/body/add" source_file="third_party/py/jax/tests/shard_map_test.py" source_line=1052}
  }

  %region_1.17_spmd (param: s32[]) -> pred[] {
    %param = s32[]{:T(128)} parameter(0), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/while"}
    %constant = s32[]{:T(128)} constant(1)
    ROOT %compare.0 = pred[]{:T(512)} compare(%param, %constant), direction=LT, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/while/cond/lt" source_file="third_party/py/jax/tests/shard_map_test.py" source_line=1049}
  }

  ENTRY %main.44_spmd (param.2: s32[1], param.3: s32[1], param.4: s32[1], param.5: s32[1]) -> (s32[1], s32[1], s32[1], s32[1]) {
    %param.2 = s32[1]{0:T(128)} parameter(0), sharding={devices=[4]<=[4]}, metadata={op_name="args[0]"}
    %bitcast.2 = s32[]{:T(128)} bitcast(%param.2), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/squeeze" source_file="third_party/py/jax/tests/shard_map_test.py" source_line=1053}
    %while.1 = s32[]{:T(128)} while(%bitcast.2), condition=%region_1.17_spmd, body=%region_0.13_spmd, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/while" source_file="third_party/py/jax/tests/shard_map_test.py" source_line=1053}
    %bitcast.3 = s32[1]{0:T(128)} bitcast(%while.1), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/jit(shmap_body)/broadcast_in_dim" source_file="third_party/py/jax/tests/shard_map_test.py" source_line=1053}
    %param.3 = s32[1]{0:T(128)} parameter(1), sharding={devices=[4]<=[4]}, metadata={op_name="args[1]"}
    %param.4 = s32[1]{0:T(128)} parameter(2), sharding={devices=[4]<=[4]}, metadata={op_name="args[2]"}
    %param.5 = s32[1]{0:T(128)} parameter(3), sharding={devices=[4]<=[4]}, metadata={op_name="args[3]"}
    ROOT %tuple.1 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(%bitcast.3, %param.3, %param.4, %param.5)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto after_layout_bitcast_module,
                          ParseAndReturnVerifiedModule(after_layout_bitcast));
  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloDataflowAnalysis::Run(*after_layout_bitcast_module,
                                                   /*ssa_form=*/false));
  HloInstruction* bitcast3 =
      FindInstruction(after_layout_bitcast_module.get(), "bitcast.3");
  HloInstruction* param2 =
      FindInstruction(after_layout_bitcast_module.get(), "param.2");
  HloComputation* while_body =
      FindComputation(after_layout_bitcast_module.get(), "region_0.13_spmd");
  HloInstruction* add0 = while_body->root_instruction();
  std::vector<HloInstruction*> defining_instructions;
  for (const HloValue* value : analysis->GetValueSet(bitcast3, {}).values()) {
    defining_instructions.push_back(value->defining_instruction());
  }
  EXPECT_THAT(defining_instructions, UnorderedElementsAre(param2, add0));
}

TEST_P(HloDataflowAnalysisTest, b409756077) {
  const char* after_layout_bitcast = R"(
  HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(f32[1,256,256]{2,1,0:T(8,128)})->f32[1,256,256]{2,1,0:T(8,128)}}
  add_f32 {
    %add_lhs = f32[] parameter(0)
    %add_rhs = f32[] parameter(1)
    ROOT %add = f32[] add(%add_lhs, %add_rhs)
  }

  %while_body (param.1: f32[256,256]) -> f32[256,256] {
    %param.1 = f32[256,256]{1,0:T(8,128)} parameter(0)
    %constant.0 = f32[]{:T(8,128)} constant(1)
    %constant.1 = f32[256,256]{1,0:T(8,128)} broadcast(%constant.0), dimensions={}
    ROOT %add.0 = f32[256,256]{1,0:T(8,128)} add(%param.1, %constant.1)
  }

  %while_condition (param: f32[256,256]) -> pred[] {
    %param.0 = f32[256,256]{1,0:T(8,128)} parameter(0)
    %zero = f32[]{:T(8,128)} constant(0)
    %sum_of_values_in_param = f32[]{:T(8,128)} reduce(%param.0, %zero), dimensions={0,1}, to_apply=%add_f32
    %constant = f32[]{:T(8,128)} constant(512)
    ROOT %compare.0 = pred[] compare(%sum_of_values_in_param, %constant), direction=LT
  }

  ENTRY %main (param.2: f32[1,256,256]) -> f32[1,256,256] {
    %param.2 = f32[1,256,256]{2,1,0:T(8,128)} parameter(0)
    %bitcast.2 = f32[256,256]{1,0:T(8,128)} bitcast(%param.2)
    %while.1 = f32[256,256]{1,0:T(8,128)} while(%bitcast.2), condition=%while_condition, body=%while_body
    ROOT %bitcast.3 = f32[1,256,256]{2,1,0:T(8,128)} bitcast(%while.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto after_layout_bitcast_module,
                          ParseAndReturnVerifiedModule(after_layout_bitcast));
  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloDataflowAnalysis::Run(*after_layout_bitcast_module,
                                                   /*ssa_form=*/false));
  HloInstruction* bitcast3 =
      FindInstruction(after_layout_bitcast_module.get(), "bitcast.3");
  HloInstruction* param2 =
      FindInstruction(after_layout_bitcast_module.get(), "param.2");
  HloComputation* while_body =
      FindComputation(after_layout_bitcast_module.get(), "while_body");
  HloInstruction* add0 = while_body->root_instruction();
  std::vector<HloInstruction*> defining_instructions;
  for (const HloValue* value : analysis->GetValueSet(bitcast3, {}).values()) {
    defining_instructions.push_back(value->defining_instruction());
  }
  EXPECT_THAT(defining_instructions, UnorderedElementsAre(param2, add0));
}

TEST_F(GetInPlaceInputOutputPairsTest, nvshmem_ar) {
  const char* kModule = R"(
    HloModule test_ar
    region_add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT ret = f32[] add(lhs, rhs)
    }

    ENTRY test {
      p0 = f32[10] parameter(0)
      ar = f32[10] all-reduce-start(p0), replica_groups={}, to_apply=region_add, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
      ROOT ar.done = f32[10] all-reduce-done(ar)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  const HloInstruction* ar_start =
      module->entry_computation()->root_instruction()->operand(0);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(ar_start);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  // For nvshmem allreduce, we expect no aliasing for input and output buffers
  // therefore empty inplace pairs.
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

TEST_F(GetInPlaceInputOutputPairsTest, CombinedCollectivePermute) {
  const char* kModule = R"(
    HloModule test_cp
    ENTRY test {
      p0 = f32[2,128]{1,0} parameter(0)
      p1 = f32[2,128]{1,0} parameter(1)
      p2 = f32[2,128]{1,0} parameter(2)
      p3 = f32[2,128]{1,0} parameter(3)
      collective-permute-start.0 = ((f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0}), (f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0})) collective-permute-start(p0, p1, p2, p3), channel_id=0, source_target_pairs={{0,2},{2,4},{1,3},{3,5}}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":false,"is_pipelined":false,"backend":"DEFAULT"},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
      ROOT collective-permute-done.0 = (f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0}) collective-permute-done(collective-permute-start.0)

    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  const HloInstruction* ar_start =
      module->entry_computation()->root_instruction()->operand(0);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(ar_start);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  // We expect no aliasing for input and output buffers
  // therefore empty inplace pairs.
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

}  // namespace
}  // namespace xla
