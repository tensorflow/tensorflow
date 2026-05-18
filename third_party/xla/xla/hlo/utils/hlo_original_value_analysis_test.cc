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

#include "xla/hlo/utils/hlo_original_value_analysis.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class HloOriginalValueAnalysisTest : public HloHardwareIndependentTestBase {};

TEST_F(HloOriginalValueAnalysisTest, SimpleDirectMapping) {
  std::string hlo_text = R"hlo(
HloModule module
ENTRY main {
  %p0 = f32[2,3] parameter(0), origin={{"p0_orig" {1,0}}}, sharding={maximal device=0}
  %p1 = f32[2,3] parameter(1), origin={{"p1_orig"}}, sharding={maximal device=0}
  ROOT %tuple = (f32[2,3], f32[2,3]) tuple(%p0, %p1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  EXPECT_THAT(
      analysis->optimized_tensor_dimensions(),
      UnorderedElementsAre(Pair(TensorKey{"p0", {}}, ElementsAre(2, 3)),
                           Pair(TensorKey{"p1", {}}, ElementsAre(2, 3))));

  EXPECT_THAT(analysis->optimized_tensor_sharding(), IsEmpty());

  auto p0_orig = RelativeScopedTensorKey::FromString("p0_orig", {1, 0});
  auto p1_orig = RelativeScopedTensorKey::FromString("p1_orig", {});

  EXPECT_THAT(
      analysis->original_tensor_by_optimized_tensor_key(),
      UnorderedElementsAre(
          Pair(TensorKey{"p0", {}},
               ElementsAre(Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 p0_orig))),
          Pair(TensorKey{"p1", {}},
               ElementsAre(Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 p1_orig)))));
}

TEST_F(HloOriginalValueAnalysisTest, RecoveryMapping) {
  std::string hlo_text = R"hlo(
HloModule module, origin_recovery_table={
  {"goal_orig"}: {"__ovp1"}
  {"goal_unshard"}: {"__ovp2"}, "HloModule rec1 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %g = f32[2,3] all-gather(%p), dimensions={0}, replica_groups={} }"
  {"goal_complex"}: {"__ovp3"}, "HloModule rec2 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %a = f32[2,3] add(%p, %p) }"
}
ENTRY main {
  %p1 = f32[2,3] parameter(0), origin={{"__ovp1"}}
  %p2 = f32[2,3] parameter(1), origin={{"__ovp2"}}
  %p3 = f32[2,3] parameter(2), origin={{"__ovp3"}}
  ROOT %tuple = (f32[2,3], f32[2,3], f32[2,3]) tuple(%p1, %p2, %p3)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto goal_orig = RelativeScopedTensorKey::FromString("goal_orig", {});
  auto goal_unshard = RelativeScopedTensorKey::FromString("goal_unshard", {});
  auto goal_complex = RelativeScopedTensorKey::FromString("goal_complex", {});

  EXPECT_THAT(analysis->original_tensor_by_optimized_tensor_key(),
              UnorderedElementsAre(
                  Pair(TensorKey{"p1", {}},
                       ElementsAre(AllOf(
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_orig),
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     recovery_modules,
                                 IsEmpty())))),
                  Pair(TensorKey{"p2", {}},
                       ElementsAre(AllOf(
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_unshard),
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     recovery_modules,
                                 SizeIs(1))))),
                  Pair(TensorKey{"p3", {}},
                       ElementsAre(AllOf(
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_complex),
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     recovery_modules,
                                 SizeIs(1)))))));
}

TEST_F(HloOriginalValueAnalysisTest, ChainedRecoveryModules) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule chained_module, entry_computation_layout={(f32[4,8]{1,0})->f32[4,8]{1,0}}, origin_recovery_table={
  {"original_add"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: f32[4,16]) -> f32[8,8] {
      %p = f32[4,16]{1,0} parameter(0)
      ROOT %reshape = f32[8,8]{1,0} reshape(%p)
    }
  "
  {"__ovp_1"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: f32[2,16]) -> f32[4,16] {
      %p = f32[2,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
      ROOT %ag = f32[4,16]{1,0} all-gather(%p), dimensions={0}
    }
  "
  {"__ovp_2"} : {"__ovp_3"},
  "
    ENTRY %recovery_3 (p: f32[4,8]) -> f32[2,16] {
      %p = f32[4,8]{1,0} parameter(0)
      ROOT %reshape = f32[2,16]{1,0} reshape(%p)
    }
  "
}

ENTRY %main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
  ROOT %opt = f32[4,8]{1,0} add(%p, %p), origin={{"__ovp_3"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  TensorKey opt_key = TensorKey::Create("opt");
  auto it = analysis->original_tensor_by_optimized_tensor_key().find(opt_key);
  ASSERT_NE(it, analysis->original_tensor_by_optimized_tensor_key().end());
  ASSERT_THAT(it->second, SizeIs(1));

  const auto& info = it->second[0];
  EXPECT_THAT(info.original_scoped_tensor_key.tensor_key.instruction_name,
              Eq("original_add"));
  ASSERT_THAT(info.recovery_modules, SizeIs(3));

  EXPECT_THAT(info.recovery_modules[0]->entry_computation()->name(),
              Eq("recovery_3"));
  EXPECT_THAT(info.recovery_modules[1]->entry_computation()->name(),
              Eq("recovery_2"));
  EXPECT_THAT(info.recovery_modules[2]->entry_computation()->name(),
              Eq("recovery_1"));

  // Check sharding extraction
  auto sharding_it = analysis->optimized_tensor_sharding().find(opt_key);
  ASSERT_NE(sharding_it, analysis->optimized_tensor_sharding().end());
  EXPECT_TRUE(sharding_it->second.IsTiled());
}

TEST_F(HloOriginalValueAnalysisTest, PruningWorks) {
  std::string hlo_text = R"hlo(
HloModule module, origin_recovery_table={
  {"goal_orig"}: {"__ovp1"}
  {"goal_unshard"}: {"__ovp2"}, "HloModule rec1 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %g = f32[2,3] all-gather(%p), dimensions={0}, replica_groups={} }"
  {"goal_complex"}: {"__ovp3"}, "HloModule rec2 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %a = f32[2,3] add(%p, %p) }"
}
ENTRY main {
  %p1 = f32[2,3] parameter(0), origin={{"__ovp1"}}
  %p2 = f32[2,3] parameter(1), origin={{"__ovp2"}}
  %p3 = f32[2,3] parameter(2), origin={{"__ovp3"}}
  ROOT %tuple = (f32[2,3], f32[2,3], f32[2,3]) tuple(%p1, %p2, %p3)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  absl::flat_hash_set<TensorKey> logged_keys = {TensorKey{"p1", {}},
                                                TensorKey{"p2", {}}};

  ASSERT_OK_AND_ASSIGN(auto analysis, HloOriginalValueAnalysis::Create(
                                          module.get(), logged_keys));

  auto goal_orig = RelativeScopedTensorKey::FromString("goal_orig", {});
  auto goal_unshard = RelativeScopedTensorKey::FromString("goal_unshard", {});

  EXPECT_THAT(
      analysis->original_tensor_by_optimized_tensor_key(),
      UnorderedElementsAre(
          Pair(TensorKey{"p1", {}},
               ElementsAre(Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_orig))),
          Pair(TensorKey{"p2", {}},
               ElementsAre(Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_unshard)))));
}

TEST_F(HloOriginalValueAnalysisTest, PruningWithEmptyLoggedKeys) {
  std::string hlo_text = R"hlo(
HloModule module, origin_recovery_table={
  {"goal_orig"}: {"__ovp1"}
}
ENTRY main {
  %p1 = f32[2,3] parameter(0), origin={{"__ovp1"}}
  ROOT %tuple = (f32[2,3]) tuple(%p1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  absl::flat_hash_set<TensorKey> logged_keys = {};

  ASSERT_OK_AND_ASSIGN(auto analysis, HloOriginalValueAnalysis::Create(
                                          module.get(), logged_keys));

  EXPECT_THAT(analysis->original_tensor_by_optimized_tensor_key(), IsEmpty());
}

TEST_F(HloOriginalValueAnalysisTest, PruningWithMultiplePaths) {
  std::string hlo_text = R"hlo(
HloModule module
ENTRY main {
  %p1 = f32[2,3] parameter(0), origin={{"goal_orig"}}
  %p2 = f32[2,3] parameter(1), origin={{"goal_orig"}}
  ROOT %tuple = (f32[2,3], f32[2,3]) tuple(%p1, %p2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  absl::flat_hash_set<TensorKey> logged_keys = {TensorKey{"p1", {}}};

  ASSERT_OK_AND_ASSIGN(auto analysis, HloOriginalValueAnalysis::Create(
                                          module.get(), logged_keys));

  auto goal_orig_scoped = RelativeScopedTensorKey::FromString("goal_orig", {});
  auto goal_orig_key = goal_orig_scoped.tensor_key;

  EXPECT_THAT(
      analysis->original_tensor_by_optimized_tensor_key(),
      UnorderedElementsAre(
          Pair(TensorKey{"p1", {}},
               ElementsAre(Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_orig_scoped)))));

  auto it = analysis->original_to_optimized_tensor_map().find(goal_orig_key);
  ASSERT_NE(it, analysis->original_to_optimized_tensor_map().end());
  EXPECT_THAT(it->second, SizeIs(1));
  EXPECT_THAT(it->second[0].first, Eq(TensorKey{"p1", {}}));
}

TEST_F(HloOriginalValueAnalysisTest, PruningWithComplexPaths) {
  std::string hlo_text = R"hlo(
HloModule module, origin_recovery_table={
  {"goal_orig"}: {"__ovp_p1_1"}, "HloModule rec_p1_1 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %a = f32[2,3] add(%p, %p) }"
  {"__ovp_p1_1"}: {"__ovp_p1_2"}, "HloModule rec_p1_2 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %s = f32[2,3] add(%p, %p) }"
  {"goal_orig"}: {"__ovp_p2_1"}, "HloModule rec_p2_1 ENTRY %e { %p = f32[2,3] parameter(0) ROOT %m = f32[2,3] add(%p, %p) }"
}
ENTRY main {
  %p1 = f32[2,3] parameter(0), origin={{"__ovp_p1_2"}}
  %p2 = f32[2,3] parameter(1), origin={{"__ovp_p2_1"}}
  ROOT %tuple = (f32[2,3], f32[2,3]) tuple(%p1, %p2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  absl::flat_hash_set<TensorKey> logged_keys = {TensorKey{"p1", {}}};

  ASSERT_OK_AND_ASSIGN(auto analysis, HloOriginalValueAnalysis::Create(
                                          module.get(), logged_keys));

  auto goal_orig_scoped = RelativeScopedTensorKey::FromString("goal_orig", {});
  auto goal_orig_key = goal_orig_scoped.tensor_key;

  EXPECT_THAT(analysis->original_tensor_by_optimized_tensor_key(),
              UnorderedElementsAre(
                  Pair(TensorKey{"p1", {}},
                       ElementsAre(AllOf(
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     original_scoped_tensor_key,
                                 goal_orig_scoped),
                           Field(&HloOriginalValueAnalysis::OriginalTensorInfo::
                                     recovery_modules,
                                 SizeIs(2)))))));

  auto it = analysis->original_to_optimized_tensor_map().find(goal_orig_key);
  ASSERT_NE(it, analysis->original_to_optimized_tensor_map().end());
  EXPECT_THAT(it->second, SizeIs(1));
  EXPECT_THAT(it->second[0].first, Eq(TensorKey{"p1", {}}));
}
TEST_F(HloOriginalValueAnalysisTest, RecoverableTracking) {
  const char* module_str = R"(
HloModule TestModule

ENTRY main {
  src0 = f32[2] parameter(0)
  ROOT res0 = f32[2] copy(src0), origin={{"orig0res0"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  // The original tensor key should be recoverable.
  auto rel_key = RelativeScopedTensorKey::FromString("orig0res0");
  auto abs_key = AbsoluteScopedTensorKey::Create({}, rel_key, {});

  EXPECT_TRUE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key));

  // An unknown key should not be recoverable.
  auto unknown_rel_key = RelativeScopedTensorKey::FromString("unknown");
  auto unknown_abs_key =
      AbsoluteScopedTensorKey::Create({}, unknown_rel_key, {});
  EXPECT_FALSE(
      analysis->IsOriginalAbsoluteTensorKeyRecoverable(unknown_abs_key));
}

TEST_F(HloOriginalValueAnalysisTest, WildcardMatching) {
  const char* module_str = R"(
HloModule TestModule
ENTRY main {
  src = f32[2] parameter(0)
  ROOT res = f32[2] copy(src), origin={{"loop#*/res"}}
}
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto abs_key = AbsoluteScopedTensorKey::FromString("loop#5/res");

  EXPECT_TRUE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key));
}

TEST_F(HloOriginalValueAnalysisTest, PlaceholderMatching) {
  const char* module_str = R"(
HloModule TestModule

called_comp {
  param = f32[2] parameter(0)
  ROOT res_inner = f32[2] copy(param), origin={{"inner_scope/res_inner"}}
}

ENTRY main {
  src = f32[2] parameter(0)
  my_call = f32[2] call(src), to_apply=called_comp, origin={{"outer_scope#$/my_call"}}
  ROOT root = f32[2] copy(my_call)
}
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto abs_key = AbsoluteScopedTensorKey::FromString(
      "outer_scope#3/my_call/inner_scope/res_inner");

  EXPECT_TRUE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key));
}

TEST_F(HloOriginalValueAnalysisTest, MismatchIterationIndex) {
  const char* module_str = R"(
HloModule TestModule
ENTRY main {
  src = f32[2] parameter(0)
  ROOT res = f32[2] copy(src), origin={{"loop#1/res"}}
}
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto abs_key = AbsoluteScopedTensorKey::FromString("loop#2/res");

  EXPECT_FALSE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key));
}

TEST_F(HloOriginalValueAnalysisTest, ComplexRecoveryPath) {
  const char* module_str = R"(
HloModule TestModule

leaf_comp {
  p = f32[2] parameter(0)
  ROOT leaf_res = f32[2] copy(p), origin={{"leaf_scope/leaf_res"}}
}

mid_comp_A {
  p = f32[2] parameter(0)
  ROOT call_leaf_A = f32[2] call(p), to_apply=leaf_comp, origin={{"A_scope/call_leaf_A"}}
}

mid_comp_B {
  p = f32[2] parameter(0)
  ROOT call_leaf_B = f32[2] call(p), to_apply=leaf_comp, origin={{"B_scope/call_leaf_B"}}
}

ENTRY main {
  src0 = f32[2] parameter(0)
  src1 = f32[2] parameter(1)
  call_A = f32[2] call(src0), to_apply=mid_comp_A, origin={{"main_scope/call_A"}}
  call_B = f32[2] call(src1), to_apply=mid_comp_B, origin={{"main_scope/call_B"}}
  ROOT root = f32[2] add(call_A, call_B)
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  // Query 1: Path via A is valid.
  auto abs_key_A = AbsoluteScopedTensorKey::FromString(
      "main_scope/call_A/A_scope/call_leaf_A/leaf_scope/leaf_res");

  EXPECT_TRUE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key_A));

  // Query 2: Path via B is valid.
  auto abs_key_B = AbsoluteScopedTensorKey::FromString(
      "main_scope/call_B/B_scope/call_leaf_B/leaf_scope/leaf_res");

  EXPECT_TRUE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key_B));

  // Query 3: Broken path (mixes A and B).
  // E.g., main_scope -> call_B -> A_scope -> ...
  auto abs_key_broken = AbsoluteScopedTensorKey::FromString(
      "main_scope/call_B/A_scope/call_leaf_A/leaf_scope/leaf_res");

  EXPECT_FALSE(
      analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key_broken));
}

TEST_F(HloOriginalValueAnalysisTest, BrokenPathMissingOriginOnCall) {
  const char* module_str = R"(
HloModule TestModule

leaf_comp {
  p = f32[2] parameter(0)
  ROOT leaf_res = f32[2] copy(p), origin={{"leaf_scope/leaf_res"}}
}

mid_comp {
  p = f32[2] parameter(0)
  ROOT leaf_call = f32[2] call(p), to_apply=leaf_comp
}

ENTRY main {
  src = f32[2] parameter(0)
  ROOT mid_call = f32[2] call(src), to_apply=mid_comp, origin={{"main_scope/mid_call"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto abs_key = AbsoluteScopedTensorKey::FromString(
      "main_scope/mid_call/leaf_call/leaf_scope/leaf_res");

  EXPECT_FALSE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key));
}

TEST_F(HloOriginalValueAnalysisTest, FusionTracking) {
  const char* module_str = R"(
HloModule TestModule

fused_comp {
  p = f32[2] parameter(0)
  ROOT fused_res = f32[2] copy(p), origin={{"fused_scope/fused_res"}}
}

ENTRY main {
  src = f32[2] parameter(0)
  ROOT fusion = f32[2] fusion(src), kind=kLoop, calls=fused_comp, origin={{"main_scope/fusion"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto abs_key = AbsoluteScopedTensorKey::FromString(
      "main_scope/fusion/fused_scope/fused_res");

  EXPECT_TRUE(analysis->IsOriginalAbsoluteTensorKeyRecoverable(abs_key));
}

TEST_F(HloOriginalValueAnalysisTest, RequestedOriginalArraysFiltering) {
  const char* module_str = R"(
HloModule TestModule
ENTRY main {
  src = f32[2] parameter(0)
  ROOT res = f32[2] copy(src), origin={{"orig_array"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));

  auto make_debug_attr = [](int64_t callback_id) {
    HloModule::DebugAttributes attr;
    attr.callback_id = callback_id;
    return attr;
  };

  OriginalArray oa = {"orig_array", {}};
  module->AddDebugAttributes(oa, make_debug_attr(0));
  module->AddDebugAttributes(oa, make_debug_attr(42));
  module->AddDebugAttributes(oa, make_debug_attr(100));

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto it = analysis->requested_original_arrays().find(oa);
  ASSERT_NE(it, analysis->requested_original_arrays().end());
  EXPECT_THAT(it->second, SizeIs(2));
  EXPECT_THAT(it->second[0].callback_id, Eq(42));
  EXPECT_THAT(it->second[1].callback_id, Eq(100));
}

TEST_F(HloOriginalValueAnalysisTest, WhileInstructionPlaceholderHeuristic) {
  const char* module_str = R"(
HloModule TestModule

cond_comp {
  p = f32[2] parameter(0)
  ROOT res = pred[] constant(true)
}

body_comp {
  p = f32[2] parameter(0)
  ROOT res = f32[2] copy(p)
}

ENTRY main {
  src = f32[2] parameter(0)
  ROOT loop = f32[2] while(src), condition=cond_comp, body=body_comp, origin={{"outer/while_loop/inner"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  auto it = analysis->call_map().find("loop");
  ASSERT_NE(it, analysis->call_map().end());
  ASSERT_THAT(it->second, SizeIs(3));
  EXPECT_THAT(it->second[0].instruction_name, Eq("outer"));
  EXPECT_THAT(it->second[1].instruction_name, Eq("while_loop"));
  EXPECT_THAT(it->second[1].iteration_index, Eq(-2));
  EXPECT_THAT(it->second[2].instruction_name, Eq("inner"));
}

TEST_F(HloOriginalValueAnalysisTest, CallMapAndComputationMapping) {
  const char* module_str = R"(
HloModule TestModule

called_comp {
  p = f32[2] parameter(0)
  ROOT res = f32[2] copy(p)
}

ENTRY main {
  src = f32[2] parameter(0)
  ROOT caller = f32[2] call(src), to_apply=called_comp, origin={{"scope1/scope2"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  EXPECT_THAT(analysis->instruction_to_computation().at("caller"), Eq("main"));
  EXPECT_THAT(analysis->instruction_to_computation().at("res"),
              Eq("called_comp"));

  auto rev_it = analysis->reverse_call_map().find("called_comp");
  ASSERT_NE(rev_it, analysis->reverse_call_map().end());
  EXPECT_THAT(rev_it->second, ElementsAre("caller"));

  auto call_it = analysis->call_map().find("caller");
  ASSERT_NE(call_it, analysis->call_map().end());
  ASSERT_THAT(call_it->second, SizeIs(2));
  EXPECT_THAT(call_it->second[0].instruction_name, Eq("scope1"));
  EXPECT_THAT(call_it->second[1].instruction_name, Eq("scope2"));
}

}  // namespace
}  // namespace xla
