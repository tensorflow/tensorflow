/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/comparison_hlo_dumper.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(CombineSummariesTest, CombinesMultipleSummaries) {
  std::vector<TensorSummaryProto> summaries;
  summaries.push_back(ParseTextProtoOrDie<TensorSummaryProto>(R"pb(
    block_summaries { min: 1 max: 3 mean: 2 stddev: 1 count: 2 }
  )pb"));
  summaries.push_back(ParseTextProtoOrDie<TensorSummaryProto>(R"pb(
    block_summaries { min: 4 max: 6 mean: 5 stddev: 1 count: 2 }
  )pb"));

  FloatBlockSummary combined = CombineSummaries(summaries);

  EXPECT_THAT(combined.min, Eq(1.0));
  EXPECT_THAT(combined.max, Eq(6.0));
  EXPECT_THAT(combined.count, Eq(4.0));
  EXPECT_THAT(combined.mean, Eq(3.5));
}

TEST(CombineSummariesTest, EmptyInput) {
  std::vector<TensorSummaryProto> summaries;
  FloatBlockSummary combined = CombineSummaries(summaries);
  EXPECT_THAT(combined.count, Eq(0.0));
}

TEST(GetTooltipDataTest, Basic) {
  HloNodeComparisonStats comp;
  comp.score_count = 10;
  comp.score_min = 0.1;
  comp.score_max = 0.9;
  comp.score_mean = 0.5;

  FloatBlockSummary baseline{};
  baseline.mean = 1.0;
  baseline.count = 10;

  FloatBlockSummary target{};
  target.mean = 1.1;
  target.count = 10;

  std::string json_str = GetTooltipData(&comp, &baseline, &target);

  EXPECT_THAT(json_str, HasSubstr("\"diffScore\""));
  EXPECT_THAT(json_str, HasSubstr("\"baseline\""));
  EXPECT_THAT(json_str, HasSubstr("\"target\""));
  EXPECT_THAT(json_str, HasSubstr("1"));
  EXPECT_THAT(json_str, HasSubstr("1.1"));
}

TEST(GetTooltipDataTest, NotComparable) {
  HloNodeComparisonStats comp;
  comp.not_comparable = true;

  FloatBlockSummary baseline{};
  baseline.mean = 1.0;

  std::string json_str = GetTooltipData(&comp, &baseline, nullptr);

  EXPECT_THAT(json_str, HasSubstr("\"notComparable\":true"));
  EXPECT_THAT(json_str, Not(HasSubstr("\"baseline\"")));
  EXPECT_THAT(json_str, Not(HasSubstr("\"target\"")));
}

TEST(GetTooltipDataTest, NoComparisonStats) {
  FloatBlockSummary run{};
  run.mean = 1.0;

  std::string json_str = GetTooltipData(nullptr, &run, nullptr);

  EXPECT_THAT(json_str, Not(HasSubstr("\"diffScore\"")));
  EXPECT_THAT(json_str, HasSubstr("1"));
}

TEST(HloHtmlTensorKeyTest, EqualityAndHashing) {
  HloHtmlTensorKey key1{"abc", {1, 2}};
  HloHtmlTensorKey key2{"abc", {1, 2}};
  HloHtmlTensorKey key3{"def", {1, 2}};
  HloHtmlTensorKey key4{"abc", {1}};

  EXPECT_THAT(key1, Eq(key2));
  EXPECT_THAT(key1, Not(Eq(key3)));
  EXPECT_THAT(key1, Not(Eq(key4)));
}

class ComparisonHloDumperDagTest
    : public ::xla::HloHardwareIndependentTestBase {};

TEST_F(ComparisonHloDumperDagTest, GenerateComputationDagCollectionLocal) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule local_test
ENTRY %main (p0: f32[], p1: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %add = f32[] add(%p0, %p1)
  %exp = f32[] exponential(%add)
  ROOT %sine = f32[] sine(%exp)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;
  AbsoluteScopedTensorKey rep_add =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add"));
  reported_keys.push_back(rep_add);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  // In entry computation, key nodes are:
  // p0, p1 (parameters)
  // sine (root)
  // add (reported)
  // exp is NOT a key node, so it should be forwarded through.

  auto it = dag_collection.graphs.find("main");
  ASSERT_NE(it, dag_collection.graphs.end());
  const auto& graph = it->second;

  LocalGraphNode p0_node = {"p0", -1};
  LocalGraphNode p1_node = {"p1", -1};
  LocalGraphNode add_node = {"add", -1};
  LocalGraphNode sine_node = {"sine", -1};

  // Check consumers
  // p0 -> add
  // p1 -> add
  // add -> sine (via exp, which is forwarded)

  EXPECT_THAT(graph.consumers.at(p0_node), ::testing::Contains(add_node));
  EXPECT_THAT(graph.consumers.at(p1_node), ::testing::Contains(add_node));
  EXPECT_THAT(graph.consumers.at(add_node), ::testing::Contains(sine_node));

  // Check suppliers
  EXPECT_THAT(graph.suppliers.at(add_node), ::testing::Contains(p0_node));
  EXPECT_THAT(graph.suppliers.at(add_node), ::testing::Contains(p1_node));
  EXPECT_THAT(graph.suppliers.at(sine_node), ::testing::Contains(add_node));
}

TEST_F(ComparisonHloDumperDagTest, FindConsumersAndSuppliersCall) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule call_test

%called (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %inner_add = f32[] add(%p, %p)
}

ENTRY %main (p0: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %call = f32[] call(%p0), to_apply=%called
  ROOT %sine = f32[] sine(%call)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_p0 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p0"));
  reported_keys.push_back(rep_p0);

  AbsoluteScopedTensorKey rep_inner_add = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inner_add"), {ScopeInstruction::Create("call", 0)});
  reported_keys.push_back(rep_inner_add);

  AbsoluteScopedTensorKey rep_sine =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("sine"));
  reported_keys.push_back(rep_sine);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  // Test FindConsumers from p0
  auto consumers_p0 = FindConsumers(rep_p0, dag_collection);
  // Should find inner_add
  ASSERT_EQ(consumers_p0.size(), 1);
  EXPECT_EQ(consumers_p0[0].tensor_key.instruction_name, "inner_add");

  // Test FindSuppliers from inner_add
  auto suppliers_inner_add = FindSuppliers(rep_inner_add, dag_collection);
  // Should find p0
  ASSERT_EQ(suppliers_inner_add.size(), 1);
  EXPECT_EQ(suppliers_inner_add[0].tensor_key.instruction_name, "p0");

  // Test FindConsumers from inner_add
  auto consumers_inner_add = FindConsumers(rep_inner_add, dag_collection);
  // Should find sine
  ASSERT_EQ(consumers_inner_add.size(), 1);
  EXPECT_EQ(consumers_inner_add[0].tensor_key.instruction_name, "sine");

  // Test FindSuppliers from sine
  auto suppliers_sine = FindSuppliers(rep_sine, dag_collection);
  // Should find inner_add
  ASSERT_EQ(suppliers_sine.size(), 1);
  EXPECT_EQ(suppliers_sine[0].tensor_key.instruction_name, "inner_add");
}

TEST_F(ComparisonHloDumperDagTest, FindConsumersWildcardAndTuple) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule while_tuple_test

%cond (p: (f32[])) -> pred[] {
  %p = (f32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%body (p: (f32[])) -> (f32[]) {
  %p = (f32[]) parameter(0)
  %gte = f32[] get-tuple-element(%p), index=0
  %inner_add = f32[] add(%gte, %gte)
  ROOT %tuple = (f32[]) tuple(%inner_add)
}

ENTRY %main (p0: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %tuple0 = (f32[]) tuple(%p0)
  %while = (f32[]) while(%tuple0), condition=%cond, body=%body
  %gte_out = f32[] get-tuple-element(%while), index=0
  ROOT %sine = f32[] sine(%gte_out)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_p0 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p0"));
  reported_keys.push_back(rep_p0);

  // Reported inside loop with concrete iteration 0
  AbsoluteScopedTensorKey rep_inner_add_concrete =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inner_add"),
                                      {ScopeInstruction::Create("while", 0)});
  reported_keys.push_back(rep_inner_add_concrete);

  // Reported inside loop with wildcard
  AbsoluteScopedTensorKey rep_inner_add_wildcard =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inner_add"),
                                      {ScopeInstruction::Create("while", -1)});
  reported_keys.push_back(rep_inner_add_wildcard);

  AbsoluteScopedTensorKey rep_sine =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("sine"));
  reported_keys.push_back(rep_sine);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  // 1. FindConsumers from p0 (Outside) -> Should enter While as Wildcard
  // Should match both Concrete and Wildcard reported inner_add
  auto consumers_p0 = FindConsumers(rep_p0, dag_collection);
  EXPECT_THAT(consumers_p0,
              ::testing::UnorderedElementsAre(rep_inner_add_concrete,
                                              rep_inner_add_wildcard));

  // 2. FindConsumers from rep_inner_add_concrete (Inside) -> Should exit While
  // Should match sine
  auto consumers_inner_concrete =
      FindConsumers(rep_inner_add_concrete, dag_collection);
  ASSERT_EQ(consumers_inner_concrete.size(), 1);
  EXPECT_EQ(consumers_inner_concrete[0].tensor_key.instruction_name, "sine");

  // 3. FindConsumers from rep_inner_add_wildcard (Inside) -> Should exit While
  // Should match sine
  auto consumers_inner_wildcard =
      FindConsumers(rep_inner_add_wildcard, dag_collection);
  ASSERT_EQ(consumers_inner_wildcard.size(), 1);
  EXPECT_EQ(consumers_inner_wildcard[0].tensor_key.instruction_name, "sine");

  // 4. FindSuppliers from rep_inner_add_concrete (Inside) -> Should exit While
  // Should match p0
  auto suppliers_inner_concrete =
      FindSuppliers(rep_inner_add_concrete, dag_collection);
  ASSERT_EQ(suppliers_inner_concrete.size(), 1);
  EXPECT_EQ(suppliers_inner_concrete[0].tensor_key.instruction_name, "p0");

  // 5. FindSuppliers from rep_sine (Outside) -> Should enter While
  // Should match both Wildcard and Concrete reported inner_add because
  // we enter with concrete max_iter, which can match wildcard.
  auto suppliers_sine = FindSuppliers(rep_sine, dag_collection);
  EXPECT_THAT(suppliers_sine,
              ::testing::UnorderedElementsAre(rep_inner_add_wildcard,
                                              rep_inner_add_concrete));
}

TEST_F(ComparisonHloDumperDagTest,
       FindConsumersAndSuppliersLoopCarriedComplex) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule while_complex_test

%cond (p: (f32[], f32[], f32[])) -> pred[] {
  %p = (f32[], f32[], f32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%body (p: (f32[], f32[], f32[])) -> (f32[], f32[], f32[]) {
  %p = (f32[], f32[], f32[]) parameter(0)
  %a = f32[] get-tuple-element(%p), index=0
  %b = f32[] get-tuple-element(%p), index=1
  %c = f32[] get-tuple-element(%p), index=2
  
  %a_next = f32[] add(%a, %b)
  %b_next = f32[] multiply(%b, %c)
  %c_next = f32[] sine(%a)
  
  ROOT %tuple = (f32[], f32[], f32[]) tuple(%a_next, %b_next, %c_next)
}

ENTRY %main (p0: f32[], p1: f32[], p2: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %p2 = f32[] parameter(2)
  %tuple0 = (f32[], f32[], f32[]) tuple(%p0, %p1, %p2)
  %while = (f32[], f32[], f32[]) while(%tuple0), condition=%cond, body=%body
  %gte_out = f32[] get-tuple-element(%while), index=0
  ROOT %sine = f32[] sine(%gte_out)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_a_next_0 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("a_next"), {ScopeInstruction::Create("while", 0)});
  reported_keys.push_back(rep_a_next_0);

  AbsoluteScopedTensorKey rep_a_next_1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("a_next"), {ScopeInstruction::Create("while", 1)});
  reported_keys.push_back(rep_a_next_1);

  AbsoluteScopedTensorKey rep_b_next_0 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("b_next"), {ScopeInstruction::Create("while", 0)});
  reported_keys.push_back(rep_b_next_0);

  AbsoluteScopedTensorKey rep_b_next_1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("b_next"), {ScopeInstruction::Create("while", 1)});
  reported_keys.push_back(rep_b_next_1);

  AbsoluteScopedTensorKey rep_c_next_0 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("c_next"), {ScopeInstruction::Create("while", 0)});
  reported_keys.push_back(rep_c_next_0);

  AbsoluteScopedTensorKey rep_c_next_1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("c_next"), {ScopeInstruction::Create("while", 1)});
  reported_keys.push_back(rep_c_next_1);

  AbsoluteScopedTensorKey rep_sine =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("sine"));
  reported_keys.push_back(rep_sine);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  // 1. Consumers of a_next#0
  // - c_next#1 (via 'a' loop carried to next iteration)
  // - a_next#1 (via 'a' loop carried to next iteration)
  // - NOT sine (since iteration 0 is not the final iteration)
  auto consumers_a_0 = FindConsumers(rep_a_next_0, dag_collection);
  EXPECT_THAT(consumers_a_0,
              ::testing::UnorderedElementsAre(rep_c_next_1, rep_a_next_1));

  // 1b. Consumers of a_next#1 (Final iteration)
  // - sine (via exit)
  auto consumers_a_1 = FindConsumers(rep_a_next_1, dag_collection);
  EXPECT_THAT(consumers_a_1, ::testing::UnorderedElementsAre(rep_sine));

  // 2. Consumers of b_next#0
  // - a_next#1 (via 'b' loop carried to next iteration)
  // - b_next#1 (via 'b' loop carried to next iteration)
  auto consumers_b_0 = FindConsumers(rep_b_next_0, dag_collection);
  EXPECT_THAT(consumers_b_0,
              ::testing::UnorderedElementsAre(rep_a_next_1, rep_b_next_1));

  // 3. Consumers of c_next#0
  // - b_next#1 (via 'c' loop carried to next iteration)
  auto consumers_c_0 = FindConsumers(rep_c_next_0, dag_collection);
  EXPECT_THAT(consumers_c_0, ::testing::UnorderedElementsAre(rep_b_next_1));

  // 4. Suppliers of a_next#1
  // - a_next#0 (via 'a' loop carried)
  // - b_next#0 (via 'b' loop carried)
  auto suppliers_a_1 = FindSuppliers(rep_a_next_1, dag_collection);
  EXPECT_THAT(suppliers_a_1,
              ::testing::UnorderedElementsAre(rep_a_next_0, rep_b_next_0));
  // It also depends on p0, p1 from entry!
  // Wait, does it?
  // a_next depends on 'a' and 'b'.
  // 'a' comes from param[0]. 'b' comes from param[1].
  // param comes from while loop.
  // So it goes to while loop operands in parent!
  // Operands are tuple0 -> p0, p1, p2.
  // So it should depend on p0, p1!
  // Let's add them to reported keys to verify.
  AbsoluteScopedTensorKey rep_p0 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p0"));
  AbsoluteScopedTensorKey rep_p1 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p1"));
  // Re-generate dag collection with all keys.
  std::vector<AbsoluteScopedTensorKey> all_keys = reported_keys;
  all_keys.push_back(rep_p0);
  all_keys.push_back(rep_p1);
  ComputationDagCollection dag_collection_expanded =
      GenerateComputationDagCollection(*module, all_keys);

  auto suppliers_a_0_expanded =
      FindSuppliers(rep_a_next_0, dag_collection_expanded);
  // Should depend on rep_p0, rep_p1 (via left loop).
  EXPECT_THAT(suppliers_a_0_expanded,
              ::testing::UnorderedElementsAre(rep_p0, rep_p1));
}

TEST_F(ComparisonHloDumperDagTest, FindConsumersAndSuppliersNestedLoops) {
  // Let's fix the HLO string to be valid.
  constexpr absl::string_view hlo_string_valid = R"hlo(
HloModule nested_while_test

%cond_inner (p: (f32[])) -> pred[] {
  %p = (f32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%body_inner (p: (f32[])) -> (f32[]) {
  %p = (f32[]) parameter(0)
  %gte = f32[] get-tuple-element(%p), index=0
  %inner_add = f32[] add(%gte, %gte)
  ROOT %tuple = (f32[]) tuple(%inner_add)
}

%cond_outer (p: (f32[])) -> pred[] {
  %p = (f32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%body_outer (p: (f32[])) -> (f32[]) {
  %p = (f32[]) parameter(0)
  ROOT %while_inner = (f32[]) while(%p), condition=%cond_inner, body=%body_inner
}

ENTRY %main (p0: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %tuple0 = (f32[]) tuple(%p0)
  %while_outer = (f32[]) while(%tuple0), condition=%cond_outer, body=%body_outer
  %gte_out = f32[] get-tuple-element(%while_outer), index=0
  ROOT %sine = f32[] sine(%gte_out)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_string_valid));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_inner_add = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inner_add"),
      {ScopeInstruction::Create("while_outer", -1),
       ScopeInstruction::Create("while_inner", -1)});
  reported_keys.push_back(rep_inner_add);

  AbsoluteScopedTensorKey rep_sine =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("sine"));
  reported_keys.push_back(rep_sine);

  AbsoluteScopedTensorKey rep_p0 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p0"));
  reported_keys.push_back(rep_p0);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  // 1. Consumers of inner_add#-1#-1
  // Should exit inner loop, then exit outer loop, reach sine.
  auto consumers_inner = FindConsumers(rep_inner_add, dag_collection);
  EXPECT_THAT(consumers_inner, ::testing::UnorderedElementsAre(rep_sine));

  // 2. Suppliers of sine
  // Should enter outer loop, enter inner loop, match inner_add.
  auto suppliers_sine = FindSuppliers(rep_sine, dag_collection);
  EXPECT_THAT(suppliers_sine, ::testing::UnorderedElementsAre(rep_inner_add));

  // 3. Consumers of p0
  // Should enter outer, enter inner, match inner_add.
  auto consumers_p0 = FindConsumers(rep_p0, dag_collection);
  EXPECT_THAT(consumers_p0, ::testing::UnorderedElementsAre(rep_inner_add));

  // 4. Suppliers of inner_add#-1#-1
  // Should match p0.
  auto suppliers_inner = FindSuppliers(rep_inner_add, dag_collection);
  EXPECT_THAT(suppliers_inner, ::testing::UnorderedElementsAre(rep_p0));
}
TEST_F(ComparisonHloDumperDagTest, PreventOverlapSiblingCalls) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule sibling_calls_test

%called1 (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %add1 = f32[] add(%p, %p)
}

%called2 (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %add2 = f32[] add(%p, %p)
}

ENTRY %main (p0: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %call1 = f32[] call(%p0), to_apply=%called1
  %call2 = f32[] call(%p0), to_apply=%called2
  ROOT %add = f32[] add(%call1, %call2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_add1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("add1"), {ScopeInstruction::Create("call1", 0)});
  reported_keys.push_back(rep_add1);

  AbsoluteScopedTensorKey rep_add2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("add2"), {ScopeInstruction::Create("call2", 0)});
  reported_keys.push_back(rep_add2);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      node_consumers;
  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      node_suppliers;
  for (const auto& key : reported_keys) {
    auto consumers = FindConsumers(key, dag_collection);
    node_consumers[key] = consumers;
    for (const auto& c : consumers) {
      node_suppliers[c].push_back(key);
    }
  }

  auto layout = ComputeDagLayoutForTesting(dag_collection, reported_keys,
                                           node_consumers, node_suppliers);

  ASSERT_TRUE(layout.contains(rep_add1));
  ASSERT_TRUE(layout.contains(rep_add2));
  EXPECT_LT(layout[rep_add1].x, layout[rep_add2].x);
}

TEST_F(ComparisonHloDumperDagTest,
       FindConsumersEntersLoopAtSequentialIterationZero) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule loop_entry_sequential

%cond (p: (f32[])) -> pred[] {
  %p = (f32[]) parameter(0)
  ROOT %c = pred[] constant(true)
}

%body (p: (f32[])) -> (f32[]) {
  %p = (f32[]) parameter(0)
  %gte = f32[] get-tuple-element(%p), index=0
  %inner_add = f32[] add(%gte, %gte)
  ROOT %tuple = (f32[]) tuple(%inner_add)
}

ENTRY %main (p0: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %tuple0 = (f32[]) tuple(%p0)
  %while = (f32[]) while(%tuple0), condition=%cond, body=%body
  %gte_out = f32[] get-tuple-element(%while), index=0
  ROOT %sine = f32[] sine(%gte_out)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_p0 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p0"));
  reported_keys.push_back(rep_p0);

  // Logged node at initial iteration 0
  AbsoluteScopedTensorKey rep_add_0 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inner_add"), {ScopeInstruction::Create("while", 0)});
  reported_keys.push_back(rep_add_0);

  // Logged node at future iteration 1
  AbsoluteScopedTensorKey rep_add_1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inner_add"), {ScopeInstruction::Create("while", 1)});
  reported_keys.push_back(rep_add_1);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  // Since p0 is outside the loop, it enters at iteration 0. Because inner_add
  // is logged at iteration 0, it intercepts the sequential dependency,
  // preventing transit to iteration 1.
  auto consumers = FindConsumers(rep_p0, dag_collection);
  EXPECT_THAT(consumers, ::testing::UnorderedElementsAre(rep_add_0));
}

TEST_F(ComparisonHloDumperDagTest, SupplierLessInScopePushedForward) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule test_module
%called (p0: f32[], p1: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}
ENTRY %main (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  %early = f32[] exponential(%p)
  %v1 = f32[] exponential(%early)
  %v2 = f32[] exponential(%v1)
  %v3 = f32[] exponential(%v2)
  %call = f32[] call(%early, %v3), to_apply=%called
  ROOT %sine = f32[] sine(%call)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::vector<AbsoluteScopedTensorKey> reported_keys;

  AbsoluteScopedTensorKey rep_v1 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("v1"));
  reported_keys.push_back(rep_v1);

  AbsoluteScopedTensorKey rep_v2 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("v2"));
  reported_keys.push_back(rep_v2);

  AbsoluteScopedTensorKey rep_v3 =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("v3"));
  reported_keys.push_back(rep_v3);

  AbsoluteScopedTensorKey rep_p0 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("p0"), {ScopeInstruction::Create("call", 0)});
  reported_keys.push_back(rep_p0);

  AbsoluteScopedTensorKey rep_p1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("p1"), {ScopeInstruction::Create("call", 0)});
  reported_keys.push_back(rep_p1);

  ComputationDagCollection dag_collection =
      GenerateComputationDagCollection(*module, reported_keys);

  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      node_consumers;
  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      node_suppliers;
  for (const auto& key : reported_keys) {
    auto consumers = FindConsumers(key, dag_collection);
    node_consumers[key] = consumers;
    for (const auto& c : consumers) {
      node_suppliers[c].push_back(key);
    }
  }

  auto layout = ComputeDagLayoutForTesting(dag_collection, reported_keys,
                                           node_consumers, node_suppliers);

  ASSERT_TRUE(layout.contains(rep_p0));
  ASSERT_TRUE(layout.contains(rep_p1));

  // 'rep_p0' corresponds to an operand ('early') computed early. However,
  // because it has no suppliers *within the called scope*, it is considered
  // supplier-less in that scope. This ensures it receives virtual dependencies
  // from nodes entering the scope (like 'v3' entering via 'p1'), pushing it
  // forward to align with the rest of the call's execution rather than being
  // dragged back to early X coordinates.
  EXPECT_GE(layout[rep_p0].x, 4.0);
  EXPECT_GE(layout[rep_p1].x, 4.0);
}

}  // namespace
}  // namespace xla::numerics::comparison
