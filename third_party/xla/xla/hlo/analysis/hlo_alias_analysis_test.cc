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

#include "xla/hlo/analysis/hlo_alias_analysis.h"

#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;

class HloAliasAnalysisTest : public HloHardwareIndependentTestBase {
 protected:
  HloAliasAnalysisTest() : HloHardwareIndependentTestBase() {
    module_ = CreateNewVerifiedModule();
  }

  // Run alias analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  HloAliasAnalysis& RunAnalysis() {
    analysis_ = HloAliasAnalysis::Run(module_.get(), &alias_info_).value();
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
    buffers.reserve(buffer_ids.size());
    for (HloBuffer::Id id : buffer_ids) {
      buffers.push_back(analysis_->GetBuffer(id));
    }
    return buffers;
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
    for (const HloBuffer& buffer : analysis_->buffers()) {
      for (const HloValue* value_a : buffer.values()) {
        for (const HloValue* value_b : buffer.values()) {
          if (*value_a != *value_b &&
              ordering.MayInterfere(*value_a, *value_b,
                                    analysis_->dataflow_analysis(),
                                    &alias_info_)) {
            VLOG(1) << *value_a << " interferes with " << *value_b
                    << " in buffer: " << buffer;
            return true;
          }
        }
      }
    }
    return false;
  }

  // Returns true if any index in the output of the given instruction has more
  // than one buffer. That is, ComputeBuffersAt returns a vector with more than
  // one element.
  bool InstructionBuffersAreAmbiguous(const HloInstruction* instruction) const {
    for (const auto& pair :
         analysis_->dataflow_analysis().GetInstructionValueSet(instruction)) {
      const HloValueSet& value_set = pair.second;
      const HloBuffer* buffer = nullptr;
      for (const HloValue* value : value_set.values()) {
        if (buffer == nullptr) {
          buffer = &analysis_->GetBufferContainingValue(*value);
        } else if (buffer != &analysis_->GetBufferContainingValue(*value)) {
          return true;
        }
      }
    }
    return false;
  }

  // Returns true if no HloBuffer appears in more than one shape index in the
  // output of the given instruction.
  bool InstructionBuffersAreDistinct(const HloInstruction* instruction) const {
    absl::flat_hash_set<const HloBuffer*> buffers_seen;
    for (const auto& pair :
         analysis_->dataflow_analysis().GetInstructionValueSet(instruction)) {
      const HloValueSet& value_set = pair.second;
      // It's possible for multiple values at this index to have the same
      // HloBuffer. This does not result in non-distinctness.
      absl::flat_hash_set<const HloBuffer*> buffers_at_this_index;
      for (const HloValue* value : value_set.values()) {
        buffers_at_this_index.insert(
            &analysis_->GetBufferContainingValue(*value));
      }
      buffers_seen.merge(buffers_at_this_index);
      // If `buffer_at_this_index` is not empty after the merge, a buffer must
      // have already been present in `buffers_seen`.
      if (!buffers_at_this_index.empty()) return false;
    }
    return true;
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloAliasAnalysis> analysis_;
  AliasInfo alias_info_;

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(HloAliasAnalysisTest, BinaryOperation) {
  // Test the analysis on a single binary operation (Add).
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  ROOT add = f32[] add(constant1, constant2)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.buffers().size(), 3);

  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");

  // All of the buffer sets should trivially contain a single buffer containing
  // a single value.
  for (const HloInstruction* instruction : {constant1, constant2, add}) {
    EXPECT_EQ(analysis.GetUniqueBufferAt(instruction).GetUniqueValue(),
              GetValueDefinedAt(instruction));
  }

  EXPECT_FALSE(InstructionBuffersAreAmbiguous(add));
  EXPECT_TRUE(InstructionBuffersAreDistinct(add));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, TupleAndGtes) {
  // Verify the analysis for a Tuple and GetTupleElement instructions.
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  tuple = (f32[], f32[]) tuple(param0, param1)
  gte0 = f32[] get-tuple-element(tuple), index=0
  gte1 = f32[] get-tuple-element(tuple), index=1
  ROOT add = f32[] add(gte0, gte1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  EXPECT_EQ(analysis.buffers().size(), 4);

  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* param0 = FindInstruction(module_.get(), "param0");
  HloInstruction* param1 = FindInstruction(module_.get(), "param1");
  HloInstruction* gte0 = FindInstruction(module_.get(), "gte0");

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

  EXPECT_FALSE(InstructionBuffersAreAmbiguous(tuple));
  EXPECT_TRUE(InstructionBuffersAreDistinct(tuple));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, NondistinctTuple) {
  // Test a expression with a non-distinct buffer set.
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  ROOT tuple = (f32[], f32[], f32[]) tuple(param0, param1, param0)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* param0 = FindInstruction(module_.get(), "param0");

  EXPECT_THAT(
      analysis.GetUniqueBufferAt(param0).ComputePositions(),
      UnorderedElementsAre(HloPosition{param0, {}}, HloPosition{tuple, {0}},
                           HloPosition{tuple, {2}}));

  EXPECT_FALSE(InstructionBuffersAreAmbiguous(tuple));
  EXPECT_FALSE(InstructionBuffersAreDistinct(tuple));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, ParametersWithAliasing) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  param = (f32[], f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(param), index=0
  gte1 = f32[] get-tuple-element(param), index=1
  negate0 = f32[] negate(gte0)
  negate1 = f32[] negate(gte1)
  ROOT tuple = (f32[], f32[]) tuple(negate0, negate1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1}));

  // Cannot alias an output twice.
  ASSERT_IS_NOT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{0}));

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* gte0 = FindInstruction(module_.get(), "gte0");
  HloInstruction* gte1 = FindInstruction(module_.get(), "gte1");

  EXPECT_EQ(analysis.GetUniqueBufferAt(gte0),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{0}));

  EXPECT_EQ(analysis.GetUniqueBufferAt(gte1),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{1}));
}

TEST_F(HloAliasAnalysisTest, ParametersWithCrossAliasing) {
  // parameter 0 aliased with output 1 and parameter 1 aliased with output 0.
  //
  //  (p0 ,  p1)
  //     \   /
  //      \ /
  // alias X
  //      / \
  //     /   \
  //  (p0  ,  p1)
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  param = (f32[], f32[]) parameter(0)
  gte0 = f32[] get-tuple-element(param), index=0
  gte1 = f32[] get-tuple-element(param), index=1
  ROOT tuple = (f32[], f32[]) tuple(gte0, gte1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{1}));
  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{0}));

  // Cannot alias an output twice.
  ASSERT_IS_NOT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1}));

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* gte0 = FindInstruction(module_.get(), "gte0");
  HloInstruction* gte1 = FindInstruction(module_.get(), "gte1");

  // Every Ops in this graph are aliased with each other.
  EXPECT_EQ(analysis.GetUniqueBufferAt(gte0),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{0}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(gte0),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{1}));

  EXPECT_EQ(analysis.GetUniqueBufferAt(gte1),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{0}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(gte1),
            analysis.GetUniqueBufferAt(tuple, /*index=*/{1}));
}

TEST_F(HloAliasAnalysisTest, InputOutputAliasingWithWhile) {
  // Test a simple single while instruction can be aliased with input and output
  // of the computation.
  //
  // body((F32[], F32[]) %tuple_param):
  //   %add = Add(%tuple_param{0}, %tuple_param{1})
  //   return Tuple(%tuple_param{0}, %add)
  //
  // condition((F32[], F32[]) %tuple_param):
  //   return Constant(false)
  //
  // entry:
  //   %param1 = param1
  //   %while = While(%param1, body, condition)
  //   %while_1 = GTE(%while, 0)
  //   %while_2 = GTE(%while, 1)
  //   %negate_1 = Negate(%while_1)
  //   %negate_2 = Negate(%while_2)
  //   return Tuple(negate_1, negate_2)
  //
  const char* hlo_string = R"(
HloModule test

body {
  body_param = (f32[], f32[]) parameter(0)
  body_element_0 = f32[] get-tuple-element(body_param), index=0
  body_element_1 = f32[] get-tuple-element(body_param), index=1
  add = f32[] add(body_element_0, body_element_1)
  ROOT body_tuple = (f32[], f32[]) tuple(body_element_0, add)
}

condition {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

ENTRY main {
  param = (f32[], f32[]) parameter(0)
  xla_while = (f32[], f32[]) while(param), condition=condition, body=body
  while_element_1 = f32[] get-tuple-element(xla_while), index=0
  while_element_2 = f32[] get-tuple-element(xla_while), index=1
  negate_1 = f32[] negate(while_element_1)
  negate_2 = f32[] negate(while_element_2)
  ROOT tuple = (f32[], f32[]) tuple(negate_1, negate_2)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0}));
  TF_ASSERT_OK(module_->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1}));

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* param = FindInstruction(module_.get(), "param");
  HloInstruction* xla_while = FindInstruction(module_.get(), "xla_while");
  HloInstruction* body_param = FindInstruction(module_.get(), "body_param");
  HloInstruction* cond_param = FindInstruction(module_.get(), "cond_param");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* negate_2 = FindInstruction(module_.get(), "negate_2");
  HloInstruction* while_element_2 =
      FindInstruction(module_.get(), "while_element_2");
  HloInstruction* body_element_1 =
      FindInstruction(module_.get(), "body_element_1");
  HloInstruction* body_tuple = FindInstruction(module_.get(), "body_tuple");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");

  EXPECT_THAT(analysis.GetUniqueBufferAt(xla_while, /*index=*/{1}).values(),
              UnorderedElementsAre(&GetValueDefinedAt(param, {1}),
                                   &GetValueDefinedAt(xla_while, /*index=*/{1}),
                                   &GetValueDefinedAt(body_param, {1}),
                                   &GetValueDefinedAt(cond_param, {1}),
                                   &GetValueDefinedAt(add),
                                   &GetValueDefinedAt(negate_2)));

  EXPECT_THAT(
      analysis.GetUniqueBufferAt(xla_while, /*index=*/{1}).ComputePositions(),
      UnorderedElementsAre(
          HloPosition{param, {1}}, HloPosition{xla_while, {1}},
          HloPosition{while_element_2, {}}, HloPosition{body_param, {1}},
          HloPosition{body_element_1, {}}, HloPosition{add, {}},
          HloPosition{body_tuple, {1}}, HloPosition{tuple, {1}},
          HloPosition{cond_param, {1}}, HloPosition{negate_2, {}}));

  EXPECT_FALSE(AnyValuesInSameBufferInterfere());
}

TEST_F(HloAliasAnalysisTest, SingleCall) {
  // Test a single call of a subcomputation. The subcomputation adds its two
  // array-shaped parameters.
  const char* hlo_string = R"(
HloModule test

Subcomputation {
  subparam0 = f32[] parameter(0)
  subparam1 = f32[] parameter(1)
  ROOT add = f32[] add(subparam0, subparam1)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  ROOT call = f32[] call(constant1, constant2), to_apply=Subcomputation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* subparam0 = FindInstruction(module_.get(), "subparam0");
  HloInstruction* subparam1 = FindInstruction(module_.get(), "subparam1");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* call = FindInstruction(module_.get(), "call");

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
  const char* hlo_string = R"(
HloModule test

Subcomputation {
  subparam0 = f32[] parameter(0)
  subparam1 = f32[] parameter(1)
  ROOT add = f32[] add(subparam0, subparam1)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  call1 = f32[] call(constant1, constant2), to_apply=Subcomputation
  ROOT call2 = f32[] call(call1, constant2), to_apply=Subcomputation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* subparam0 = FindInstruction(module_.get(), "subparam0");
  HloInstruction* subparam1 = FindInstruction(module_.get(), "subparam1");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* call1 = FindInstruction(module_.get(), "call1");
  HloInstruction* call2 = FindInstruction(module_.get(), "call2");

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

  EXPECT_TRUE(InstructionBuffersAreAmbiguous(subparam0));
  EXPECT_FALSE(InstructionBuffersAreAmbiguous(subparam1));
  EXPECT_TRUE(InstructionBuffersAreDistinct(subparam0));
  EXPECT_TRUE(InstructionBuffersAreDistinct(subparam1));

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
  const char* hlo_string = R"(
HloModule test

body {
  body_param = (f32[], f32[]) parameter(0)
  body_element_0 = f32[] get-tuple-element(body_param), index=0
  body_element_1 = f32[] get-tuple-element(body_param), index=1
  add = f32[] add(body_element_0, body_element_1)
  ROOT body_tuple = (f32[], f32[]) tuple(body_element_0, add)
}

condition {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(constant1, constant2)
  ROOT xla_while = (f32[], f32[]) while(tuple), condition=condition, body=body
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* xla_while = FindInstruction(module_.get(), "xla_while");
  HloInstruction* body_param = FindInstruction(module_.get(), "body_param");
  HloInstruction* body_element_0 =
      FindInstruction(module_.get(), "body_element_0");
  HloInstruction* body_element_1 =
      FindInstruction(module_.get(), "body_element_1");
  HloInstruction* add = FindInstruction(module_.get(), "add");
  HloInstruction* body_tuple = FindInstruction(module_.get(), "body_tuple");
  HloInstruction* cond_param = FindInstruction(module_.get(), "cond_param");

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

  EXPECT_THAT(analysis.GetUniqueBufferAt(xla_while, /*index=*/{0}).values(),
              UnorderedElementsAre(&GetValueDefinedAt(constant1)));
  EXPECT_THAT(analysis.GetUniqueBufferAt(xla_while, /*index=*/{1}).values(),
              UnorderedElementsAre(&GetValueDefinedAt(constant2),
                                   &GetValueDefinedAt(xla_while, /*index=*/{1}),
                                   &GetValueDefinedAt(body_param, {1}),
                                   &GetValueDefinedAt(cond_param, {1}),
                                   &GetValueDefinedAt(add)));

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
  const char* hlo_string = R"(
HloModule test

body {
  body_param = (f32[], f32[]) parameter(0)
  body_element_0 = f32[] get-tuple-element(body_param), index=0
  body_element_1 = f32[] get-tuple-element(body_param), index=1
  add = f32[] add(body_element_0, body_element_1)
  ROOT body_tuple = (f32[], f32[]) tuple(body_element_0, add)
}

condition {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(constant1, constant2)
  xla_while0 = (f32[], f32[]) while(tuple), condition=condition, body=body
  xla_while1 = (f32[], f32[]) while(xla_while0), condition=condition, body=body
  ROOT xla_while2 = (f32[], f32[]) while(xla_while1), condition=condition, body=body
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));

  FlattenCallGraph flattener;
  TF_ASSERT_OK(flattener.Run(module_.get()).status());
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* xla_while2 = FindInstruction(module_.get(), "xla_while2");

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
  const char* hlo_string = R"(
HloModule test

inner_body {
  inner_param = (f32[], f32[]) parameter(0)
  inner_element_0 = f32[] get-tuple-element(inner_param), index=0
  inner_element_1 = f32[] get-tuple-element(inner_param), index=1
  add = f32[] add(inner_element_0, inner_element_1)
  ROOT tuple = (f32[], f32[]) tuple(inner_element_0, add)
}

condition1 {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

outer_body {
  outer_param = (f32[], f32[]) parameter(0)
  outer_element_0 = f32[] get-tuple-element(outer_param), index=0
  negate = f32[] negate(outer_element_0)
  outer_element_1 = f32[] get-tuple-element(outer_param), index=1
  outer_tuple = (f32[], f32[]) tuple(negate, outer_element_1)
  ROOT nested_while = (f32[], f32[]) while(outer_tuple), condition=condition1, body=inner_body
}

condition2 {
  cond_param = (f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  tuple = (f32[], f32[]) tuple(constant1, constant2)
  ROOT entry_while = (f32[], f32[]) while(tuple), condition=condition2, body=outer_body
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* entry_while = FindInstruction(module_.get(), "entry_while");
  HloInstruction* nested_while = FindInstruction(module_.get(), "nested_while");
  HloInstruction* inner_element_0 =
      FindInstruction(module_.get(), "inner_element_0");
  HloInstruction* inner_element_1 =
      FindInstruction(module_.get(), "inner_element_1");

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
  const char* hlo_string = R"(
HloModule test

body {
  body_param = (f32[], f32[], f32[]) parameter(0)
  body_element_0 = f32[] get-tuple-element(body_param), index=0
  body_element_1 = f32[] get-tuple-element(body_param), index=1
  body_element_2 = f32[] get-tuple-element(body_param), index=2
  ROOT body_tuple = (f32[], f32[], f32[]) tuple(body_element_1, body_element_2, body_element_0)
}

condition {
  cond_param = (f32[], f32[], f32[]) parameter(0)
  ROOT cond_constant = pred[] constant(false)
}

ENTRY main {
  constant1 = f32[] constant(1.0)
  constant2 = f32[] constant(2.0)
  constant3 = f32[] constant(3.0)
  tuple = (f32[], f32[], f32[]) tuple(constant1, constant2, constant3)
  ROOT xla_while = (f32[], f32[], f32[]) while(tuple), condition=condition, body=body
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant1 = FindInstruction(module_.get(), "constant1");
  HloInstruction* constant2 = FindInstruction(module_.get(), "constant2");
  HloInstruction* constant3 = FindInstruction(module_.get(), "constant3");
  HloInstruction* tuple = FindInstruction(module_.get(), "tuple");
  HloInstruction* xla_while = FindInstruction(module_.get(), "xla_while");
  HloInstruction* cond_constant =
      FindInstruction(module_.get(), "cond_constant");

  // The swizzling while makes most positions in the module alias leaving only 3
  // HloBuffers.
  EXPECT_THAT(
      analysis.buffers(),
      UnorderedElementsAre(analysis.GetUniqueBufferAt(constant1),
                           analysis.GetUniqueBufferAt(tuple, /*index=*/{}),
                           analysis.GetUniqueBufferAt(cond_constant)));

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



TEST_F(HloAliasAnalysisTest, Bitcast) {
  // Bitcasting a value should not produce a new buffer.
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  constant = f32[] constant(1.0)
  ROOT bitcast = f32[] bitcast(constant)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  const HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* constant = FindInstruction(module_.get(), "constant");
  HloInstruction* bitcast = FindInstruction(module_.get(), "bitcast");

  EXPECT_EQ(analysis.buffers().size(), 1);

  EXPECT_EQ(analysis.GetUniqueBufferAt(constant),
            analysis.GetUniqueBufferAt(bitcast));
}

TEST_F(HloAliasAnalysisTest, DynamicUpdateSlice) {
  const char* hlo_string = R"(
HloModule test

ENTRY main {
  param0 = f32[8] parameter(0)
  param1 = f32[4] parameter(1)
  param2 = s32[] parameter(2)
  copy0 = f32[8] copy(param0)
  ROOT dynamic_update_slice = f32[8] dynamic-update-slice(copy0, param1, param2)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));
  SCOPED_TRACE(module_->ToString());

  HloAliasAnalysis& analysis = RunAnalysis();

  HloInstruction* copy0 = FindInstruction(module_.get(), "copy0");
  HloInstruction* dynamic_update_slice =
      FindInstruction(module_.get(), "dynamic_update_slice");

  EXPECT_EQ(analysis.GetUniqueBufferAt(copy0),
            analysis.GetUniqueBufferAt(dynamic_update_slice));
}

TEST_F(HloAliasAnalysisTest, DynamicUpdateSliceMultiOutputFusion) {
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
  ROOT fusion = (f32[1280,1,128], f32[1280,1,128], f32[1280,1,128]) fusion(negate0, negate1, negate2), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));

  SCOPED_TRACE(module_->ToString());

  HloAliasAnalysis& analysis = RunAnalysis();
  LOG(INFO) << analysis.ToString();

  // Expect negate1 and negate2 to alias with fusion{1} and fusion{2}
  // respectively (due to DUS), but not negate0 and fusion{0}.
  const HloInstruction* fusion =
      module_->entry_computation()->GetInstructionWithName("fusion");
  const HloInstruction* negate0 =
      module_->entry_computation()->GetInstructionWithName("negate0");
  const HloInstruction* negate1 =
      module_->entry_computation()->GetInstructionWithName("negate1");
  const HloInstruction* negate2 =
      module_->entry_computation()->GetInstructionWithName("negate2");
  EXPECT_EQ(analysis.GetUniqueBufferAt(negate1),
            analysis.GetUniqueBufferAt(fusion, {1}));
  EXPECT_EQ(analysis.GetUniqueBufferAt(negate2),
            analysis.GetUniqueBufferAt(fusion, {2}));
  EXPECT_NE(analysis.GetUniqueBufferAt(negate0),
            analysis.GetUniqueBufferAt(fusion, {0}));
}

TEST_F(HloAliasAnalysisTest, ChainedDynamicUpdateSliceFusion) {
  // CPU and GPU backends may generate fusions with dynamic update slices
  // feeding each other. They expect the fusion to not be in-place if that is
  // the case.
  absl::string_view hlo_string = R"(
HloModule Module

fused_computation {
  param0 = f32[1280,1,128] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1,128] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  dynamic-update-slice.5 = f32[1280,1,128] dynamic-update-slice(param0, broadcast.6, constant.3, constant.3, constant.3)
  ROOT dynamic-update-slice.6 = f32[1280,1,128] dynamic-update-slice(dynamic-update-slice.5, broadcast.6, constant.3, constant.3, constant.3)
}

ENTRY main {
  param = f32[1280,1,128] parameter(0)
  negate0 = f32[1280,1,128] negate(param)
  ROOT fusion = f32[1280,1,128] fusion(negate0), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_string));

  SCOPED_TRACE(module_->ToString());

  HloAliasAnalysis& analysis = RunAnalysis();
  LOG(INFO) << analysis.ToString();

  const HloInstruction* fusion =
      module_->entry_computation()->GetInstructionWithName("fusion");
  const HloInstruction* negate0 =
      module_->entry_computation()->GetInstructionWithName("negate0");
  EXPECT_NE(analysis.GetUniqueBufferAt(negate0),
            analysis.GetUniqueBufferAt(fusion));
}

}  // namespace
}  // namespace xla
