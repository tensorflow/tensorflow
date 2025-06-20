// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

class HloGumgraphTest : public HloHardwareIndependentTestBase {};

const HloInstructionNode* SelectNodeByName(const HloGumgraph& graph,
                                           absl::string_view name) {
  const HloInstructionNode* result = nullptr;
  for (const auto* node : graph.AllNodes()) {
    if (!node->is_root && node->instruction->name() == name) {
      result = node;
      break;
    }
  }
  return result;
}

// Returns true if the subgraph fingerprint of the roots are the same.
bool FingerprintEqualTo(const HloGumgraph& first, const HloGumgraph& second) {
  return first.GetRoot().props.subgraph_fingerprint ==
         second.GetRoot().props.subgraph_fingerprint;
}

void AssertNode(const HloInstructionNode* actual_node,
                absl::string_view expected_node_name, int expected_num_children,
                int expected_num_parents) {
  EXPECT_EQ(actual_node->instruction->name(), expected_node_name);
  ASSERT_EQ(actual_node->children.size(), expected_num_children);
  ASSERT_EQ(actual_node->parents.size(), expected_num_parents);
}

TEST_F(HloGumgraphTest, CreateSimpleHloModuleWithoutFusionInstructionWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | Add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  const auto* entry = graph->GetRoot().children[0];
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry, "add_0", 2, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry->children[0], "add_1", 2, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry->children[1], "baz", 0, 1));
  ASSERT_NO_FATAL_FAILURE(
      AssertNode(entry->children[0]->children[0], "foo", 0, 1));
  ASSERT_NO_FATAL_FAILURE(
      AssertNode(entry->children[0]->children[1], "bar", 0, 1));

  EXPECT_THAT(
      graph->AllComputationProps(),
      UnorderedElementsAre(Pair(
          Pointee(Property(&HloComputation::name, "entry")),
          Field(&CallGraphNodeProps::fingerprint, 10150663182810228731U))));
}

TEST_F(HloGumgraphTest, CreateHloModuleWithFusionInstructionWorks) {
  // Create a module with entry computation containing the following structure:
  // [Param p0] ---> [Param p2] ---> ┌-------┐      ┌----------┐      ┌------┐
  //                                 | add.1 | ---> | fusion.1 | ---> | ROOT |
  // [Param p1] ---> [Param p3] ---> └-------┘      └----------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

fused_computation.1 {
  p2 = s32[32,16]{0,1:T(1,128)}  parameter(0)
  p3 = s32[32,16]{0,1:T(1,128)} parameter(1)
  add.1 = s32[32,16]{0,1:T(1,128)} add(p2, p3)
}

ENTRY entry {
  p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  ROOT fusion.1 = s32[32,16]{0,1:T(1,128)} fusion(p0,p1), kind=kLoop, calls=fused_computation.1
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  const auto* entry = graph->GetRoot().children[0];
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry, "fusion.1", 1, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry->children[0], "add.1", 2, 1));
  ASSERT_NO_FATAL_FAILURE(
      AssertNode(entry->children[0]->children[0], "p2", 1, 1));
  ASSERT_NO_FATAL_FAILURE(
      AssertNode(entry->children[0]->children[0]->children[0], "p0", 0, 1));

  EXPECT_THAT(
      graph->AllComputationProps(),
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloComputation::name, "entry")),
               Field(&CallGraphNodeProps::fingerprint, 17918193494741257405U)),
          Pair(
              Pointee(Property(&HloComputation::name, "fused_computation.1")),
              Field(&CallGraphNodeProps::fingerprint, 18256571801256786953U))));
}

TEST_F(HloGumgraphTest, CreateHloModuleWithConditionalInstructionWorks) {
  // Create a module with entry computation containing the following structure:
  // [constant.2] ---> [y] ---> [identity] ---> ┌-------------┐
  //                                            |             |      ┌------┐
  // [constant.1] ---> [x] ---> [negate] -----> | conditional | ---> | ROOT |
  // [constant] ------------------------------> └-------------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

Negate {
  x = f32[] parameter(0)
  ROOT negate = f32[] negate(x)
}

Identity {
  y = f32[] parameter(0)
  ROOT identity = f32[] copy(y)
}

ENTRY entry {
  constant = pred[] constant(true)
  constant.1 = f32[] constant(56)
  constant.2 = f32[] constant(12)
  ROOT conditional = f32[] conditional(constant, constant.1, constant.2), true_computation=Negate, false_computation=Identity
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  const auto* entry = graph->GetRoot().children[0];
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry, "conditional", 3, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry->children[2], "identity", 1, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry->children[1], "negate", 1, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(entry->children[0], "constant", 0, 1));
  ASSERT_NO_FATAL_FAILURE(
      AssertNode(entry->children[2]->children[0], "y", 1, 1));
  ASSERT_NO_FATAL_FAILURE(
      AssertNode(entry->children[1]->children[0], "x", 1, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(
      entry->children[2]->children[0]->children[0], "constant.2", 0, 1));
  ASSERT_NO_FATAL_FAILURE(AssertNode(
      entry->children[1]->children[0]->children[0], "constant.1", 0, 1));

  EXPECT_THAT(
      graph->AllComputationProps(),
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloComputation::name, "entry")),
               Field(&CallGraphNodeProps::fingerprint, 9646443073508437215U)),
          Pair(Pointee(Property(&HloComputation::name, "Identity")),
               Field(&CallGraphNodeProps::fingerprint, 7593821242743477274U)),
          Pair(
              Pointee(Property(&HloComputation::name, "Negate")),
              Field(&CallGraphNodeProps::fingerprint, 11882609566947793238U))));
}

TEST_F(HloGumgraphTest, PreComputationsWorksWithoutShapeInFingerprint) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | Add_1 |
  // ┌------------┐ ---> └-------┘ ---> ┌-------┐      ┌------┐
  // |Constant bar|                     | add_0 | ---> | ROOT |
  // └------------┘ ------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, bar)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const HloGumgraph> graph,
      HloGumgraph::Create(module.get(), {.ignore_shape = true}));

  const auto* entry = graph->GetRoot().children[0];
  EXPECT_THAT(entry->props,
              FieldsAre(
                  /*generation=*/1,
                  /*height=*/3, /*subgraph_fingerprint=*/8543065396480500811U,
                  /*fingerprint=*/7968662072287666665U,
                  /*canonical_fingerprint=*/962574172336760684U));
  EXPECT_THAT(entry->children[0]->props,
              FieldsAre(
                  /*generation=*/2,
                  /*height=*/2, /*subgraph_fingerprint=*/12467718903949982030U,
                  /*fingerprint=*/7968662072287666665U,
                  /*canonical_fingerprint=*/962574172336760684U));
  EXPECT_THAT(entry->children[1]->props,
              FieldsAre(
                  /*generation=*/3,
                  /*height=*/1, /*subgraph_fingerprint=*/3183718271480206887U,
                  /*fingerprint=*/3183718271480206887U,
                  /*canonical_fingerprint=*/1545292564424961499U));
  EXPECT_THAT(entry->children[0]->children[0]->props,
              FieldsAre(
                  /*generation=*/3,
                  /*height=*/1, /*subgraph_fingerprint=*/856105463456541506U,
                  /*fingerprint=*/856105463456541506U,
                  /*canonical_fingerprint=*/2283891754502192697U));

  EXPECT_THAT(
      graph->AllComputationProps(),
      UnorderedElementsAre(
          Pair(Pointee(Property(&HloComputation::name, "entry")),
               Field(&CallGraphNodeProps::fingerprint, 8543065396480500811U))));
}

TEST_F(HloGumgraphTest, PreComputationsWorksWithShapeInFingerprint) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | Add_1 |
  // ┌------------┐ ---> └-------┘ ---> ┌-------┐      ┌------┐
  // |Constant bar|                     | add_0 | ---> | ROOT |
  // └------------┘ ------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, bar)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const HloGumgraph> graph,
      HloGumgraph::Create(module.get(), {.ignore_shape = false}));

  const auto* entry = graph->GetRoot().children[0];
  EXPECT_THAT(entry->props,
              FieldsAre(
                  /*generation=*/1,
                  /*height=*/3, /*subgraph_fingerprint=*/11491866794545709423U,
                  /*fingerprint=*/13023796333337170182U,
                  /*canonical_fingerprint=*/962574172336760684U));

  EXPECT_THAT(entry->children[0]->props,
              FieldsAre(
                  /*generation=*/2,
                  /*height=*/2, /*subgraph_fingerprint=*/11413025457497517292U,
                  /*fingerprint=*/13023796333337170182U,
                  /*canonical_fingerprint=*/962574172336760684U));
  EXPECT_THAT(entry->children[1]->props,
              FieldsAre(
                  /*generation=*/3,
                  /*height=*/1, /*subgraph_fingerprint=*/18045659843081992748U,
                  /*fingerprint=*/18045659843081992748U,
                  /*canonical_fingerprint=*/1545292564424961499U));
  EXPECT_THAT(entry->children[0]->children[0]->props,
              FieldsAre(
                  /*generation=*/3,
                  /*height=*/1, /*subgraph_fingerprint=*/7851455295828926644U,
                  /*fingerprint=*/7851455295828926644U,
                  /*canonical_fingerprint=*/2283891754502192697U));

  EXPECT_THAT(
      graph->AllComputationProps(),
      UnorderedElementsAre(Pair(
          Pointee(Property(&HloComputation::name, "entry")),
          Field(&CallGraphNodeProps::fingerprint, 11491866794545709423U))));
}

TEST_F(HloGumgraphTest, PreComputationsWorksMultiRoot) {
  // Create a module with entry computation containing the following structure:
  //                      ┌--------┐           ┌-----------┐
  // ┌-----------┐ -----> |  recv  | --------> | recv-done | ---> ┌------┐
  // | after-all |        └--------┘           └-----------┘      | ROOT |
  // └-----------┘ -----> ┌--------┐           ┌-----------┐ ---> └------┘
  // ┌----------┐         |  send  | --------> | send-done |
  // | constant | ------> └--------┘           └-----------┘
  // └----------┘
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          R"(HloModule TwoSendRecvBothWayRecvFist_module, entry_computation_layout={()->(f32[], token[])}

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  %constant = f32[] constant(2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, control-predecessors={%recv}
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
}

)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));

  EXPECT_EQ(SelectNodeByName(*graph, "recv")->props.generation, 2);
  EXPECT_EQ(SelectNodeByName(*graph, "recv-done")->props.generation, 1);
  EXPECT_EQ(SelectNodeByName(*graph, "send")->props.generation, 2);
  EXPECT_EQ(SelectNodeByName(*graph, "send-done")->props.generation, 1);
  EXPECT_EQ(SelectNodeByName(*graph, "token0")->props.generation, 3);
  EXPECT_EQ(SelectNodeByName(*graph, "constant")->props.generation, 3);
}

TEST_F(HloGumgraphTest, PreComputationsWorksSubgraphFingerprint) {
  // Create left module with entry computation containing the following
  // structure:
  // [Const 0] ---> ┌-------┐
  //                | add_0 |
  // [Const 1] ---> └-------┘ ---> ┌-------┐      ┌------┐
  //                               | add_3 | ---> | ROOT |
  // [Const 2] ---> ┌-------┐ ---> └-------┘      └------┘
  //                | add_1 |
  // [Const 3] ---> └-------┘
  //
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  constant.0 = f32[] constant(0)
  constant.1 = f32[] constant(0)
  constant.2 = f32[] constant(0)
  constant.3 = f32[] constant(0)
  add.0 = f32[] add(constant.0, constant.1)
  add.1 = f32[] add(constant.2, constant.3)
  add.3 = f32[] add(add.0, add.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_l,
                          HloGumgraph::Create(module_l.get()));

  // Create right module with entry computation containing the following
  // structure:
  // [Const 0] ---> ┌-------┐
  //                | add_0 |
  // ┌-------┐ ---> └-------┘ ---> ┌-------┐      ┌------┐
  // |Const 1|                     | add_3 | ---> | ROOT |
  // └-------┘ ---> ┌-------┐ ---> └-------┘      └------┘
  //                | add_1 |
  // [Const 3] ---> └-------┘

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  constant.0 = f32[] constant(0)
  constant.1 = f32[] constant(0)
  constant.3 = f32[] constant(0)
  add.0 = f32[] add(constant.0, constant.1)
  add.1 = f32[] add(constant.1, constant.3)
  add.3 = f32[] add(add.0, add.1)
}
)"));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph_r,
                          HloGumgraph::Create(module_r.get()));

  // TODO(b/365855856): The subgraph fingerprint should not be the same.
  // EXPECT_NE(graph_l->GetRoot().props.subgraph_fingerprint,
  //           graph_r->GetRoot().props.subgraph_fingerprint);
  EXPECT_EQ(graph_l->GetRoot().props.subgraph_fingerprint,
            graph_r->GetRoot().props.subgraph_fingerprint);
}

TEST_F(HloGumgraphTest, CalledComputationWithMultipleCallsitesAreNotInlined) {
  const absl::string_view hlo_string = R"(
    HloModule MultipleCallerComputationChainedExecution
  
    _where_26.3690 (Arg_0.3686: pred[], Arg_1.3687: s32[], Arg_2.3688: s32[]) -> s32[] {
      Arg_0.3686 = pred[] parameter(0)
      Arg_1.3687 = s32[] parameter(1)
      Arg_2.3688 = s32[] parameter(2)
      ROOT select.3689 = s32[] select(Arg_0.3686, Arg_1.3687, Arg_2.3688)
    }
  
    ENTRY main {
      parameter.1 = pred[] parameter(0)
      parameter.2 = s32[] parameter(1)
      parameter.3 = s32[] parameter(2)
      parameter.4 = pred[] parameter(3)
      parameter.5 = s32[] parameter(4)
      call.1 = s32[] call(parameter.1, parameter.2, parameter.3), to_apply=_where_26.3690
      ROOT call.2 = s32[] call(parameter.4, parameter.5, call.1), to_apply=_where_26.3690
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> graph,
                          HloGumgraph::Create(module.get()));
  EXPECT_EQ(SelectNodeByName(*graph, "parameter.1")->parents.size(), 1);
  EXPECT_EQ(SelectNodeByName(*graph, "parameter.4")->parents.size(), 1);
  EXPECT_EQ(
      SelectNodeByName(*graph, "parameter.1")->parents[0]->instruction->name(),
      "call.1");
  EXPECT_EQ(
      SelectNodeByName(*graph, "parameter.4")->parents[0]->instruction->name(),
      "call.2");
}

using HloGumgraphDeathTest = HloGumgraphTest;

TEST_F(HloGumgraphDeathTest, CreateWithNullHloModuleFails) {
  // The `hlo_module` parameter is annotated nonnull, but we want to test the
  // defensive null check. Use a variable instead of passing nullptr directly
  // to avoid a `-Wnonnull` warning.
  HloModule* null_hlo_module = nullptr;
  ASSERT_DEATH(auto unused = HloGumgraph::Create(null_hlo_module), "");
}

TEST_F(HloGumgraphDeathTest, CreateWithNullEntryComputationFails) {
  HloModule hlo_module("module", HloModuleConfig());

  ASSERT_DEATH(auto unused = HloGumgraph::Create(&hlo_module), "");
}

TEST_F(HloGumgraphTest, CheckEqualityForIdenticalGraphs) {
  // Create two identical modules with entry computation containing the
  // following structure:
  // [Param foo] ------> ┌-------┐
  //                     | Add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  const auto* hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto first_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto second_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> first_graph,
                          HloGumgraph::Create(first_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> second_graph,
                          HloGumgraph::Create(second_module.get()));

  EXPECT_TRUE(FingerprintEqualTo(*first_graph, *second_graph));
}

TEST_F(HloGumgraphTest, CheckEqualityForDifferentGraphs) {
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | Add_1 | ---> ┌-------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | add_0 | ---> | ROOT |
  // [Param baz] ---------------------> └-------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(auto first_module, ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  add_0 = f32[8,2048]{1,0:T(8,128)} add(add_1, baz)
}
)"));
  // Create a module with entry computation containing the following structure:
  // [Param foo] ------> ┌-------┐
  //                     | Add_1 | ---> ┌------------┐      ┌------┐
  // [Constant bar] ---> └-------┘      | subtract_0 | ---> | ROOT |
  // [Param baz] ---------------------> └------------┘      └------┘
  TF_ASSERT_OK_AND_ASSIGN(auto second_module, ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true

ENTRY entry {
  foo = f32[8,2048]{1,0:T(8,128)} parameter(0)
  bar = f32[8,2048]{1,0:T(8,128)} constant(0)
  baz = f32[8,2048]{1,0:T(8,128)} parameter(1)
  add_1 = f32[8,2048]{1,0:T(8,128)} add(foo, bar)
  subtract_0 = f32[8,2048]{1,0:T(8,128)} subtract(add_1, baz)
}
)"));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> first_graph,
                          HloGumgraph::Create(first_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const HloGumgraph> second_graph,
                          HloGumgraph::Create(second_module.get()));

  EXPECT_FALSE(FingerprintEqualTo(*first_graph, *second_graph));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
