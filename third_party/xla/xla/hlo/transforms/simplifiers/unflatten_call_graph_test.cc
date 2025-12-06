/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/unflatten_call_graph.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class UnflattenCallGraphTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunUnflattenCallGraph(HloModule* module) {
    UnflattenCallGraph unflatten;
    return unflatten.Run(module);
  }
};

// Tests that pass makes no changes when there are no duplicate computations.
// The graph is:
// main -> called_computation
TEST_F(UnflattenCallGraphTest, NoChange) {
  std::string hlo_string = R"(
HloModule NoChange

  %called_computation (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  ENTRY %main (a: f32[]) -> f32[] {
    %a = f32[] parameter(0)
    ROOT %call = f32[] call(%a), to_apply=%called_computation
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  EXPECT_FALSE(result);
  ASSERT_EQ(module->computation_count(), 2);
}

// Tests that the pass merges simple duplicate computations.
// The initial graph is:
// main -> called_computation_1
//      -> called_computation_2
// where called_computation_1 and called_computation_2 are identical.
// The expected graph is:
// main -> called_computation_1
//      -> called_computation_1
TEST_F(UnflattenCallGraphTest, SimpleDuplicates) {
  std::string hlo_string =
      R"(HloModule SimpleDuplicates, entry_computation_layout={(f32[])->(f32[], f32[])}

  %called_computation_1 (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  %called_computation_2 (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  ENTRY %main (a: f32[], b: f32[]) -> (f32[], f32[]) {
    %a = f32[] parameter(0)
    %call1 = f32[] call(%a), to_apply=%called_computation_1
    %call2 = f32[] call(%a), to_apply=%called_computation_2
    ROOT %tuple = (f32[], f32[]) tuple(%call1, %call2)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  ASSERT_EQ(module->computation_count(), 2);

  // Check that call1 and call2 now point to the same computation.
  auto call1 = FindInstruction(module.get(), "call1");
  auto call2 = FindInstruction(module.get(), "call2");
  EXPECT_EQ(call1->to_apply(), call2->to_apply());

  // Check that one of the computations was removed.
  auto called_computation_1 =
      FindComputation(module.get(), "called_computation_1");
  auto called_computation_2 =
      FindComputation(module.get(), "called_computation_2");
  // Check that the only one computation is used and other is deleted
  ASSERT_NE(called_computation_1 == nullptr, called_computation_2 == nullptr);
}

// Tests that the pass merges duplicate while loops that are nested in a call.
// including the body and condition computations.
// The initial graph is:
// main -> called_computation -> while_cond, while_body
//      -> called_computation.clone -> while_cond.clone, while_body.clone
// where called_computation is identical to called_computation.clone,
// while_cond to while_cond.clone, and while_body to while_body.clone.
// The expected graph is:
// main -> called_computation -> while_cond, while_body
//      -> called_computation -> while_cond, while_body
TEST_F(UnflattenCallGraphTest, DuplicatesInWhile) {
  std::string hlo_string = R"(
HloModule WhileInCall, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%while_body (p: (f32[4096])) -> (f32[4096]) {
  ROOT %p = (f32[4096]{0}) parameter(0)
}

%while_cond (p.cond: (f32[4096])) -> pred[] {
  %p.cond = (f32[4096]{0}) parameter(0)
  ROOT %eq = pred[] constant(false)
}

%called_computation (arg: f32[4096]) -> f32[4096] {
  %arg = f32[4096]{0} parameter(0)
  %while_init = (f32[4096]{0}) tuple(%arg)
  %while = (f32[4096]{0}) while(%while_init), condition=%while_cond, body=%while_body
  ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
}

%while_body.clone (p.1: (f32[4096])) -> (f32[4096]) {
  ROOT %p.1 = (f32[4096]{0}) parameter(0)
}

%while_cond.clone (p.cond.1: (f32[4096])) -> pred[] {
  %p.cond.1 = (f32[4096]{0}) parameter(0)
  ROOT %eq.1 = pred[] constant(false)
}

%called_computation.clone (arg.1: f32[4096]) -> f32[4096] {
  %arg.1 = f32[4096]{0} parameter(0)
  %while_init.1 = (f32[4096]{0}) tuple(%arg.1)
  %while.1 = (f32[4096]{0}) while(%while_init.1), condition=%while_cond.clone, body=%while_body.clone
  ROOT %get-tuple-element.1 = f32[4096]{0} get-tuple-element(%while.1), index=0
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
  %b = f32[4096]{0} parameter(1)
  %call1 = f32[4096]{0} call(%b), to_apply=%called_computation.clone
  ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 7);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  ASSERT_EQ(module->computation_count(), 4);

  // check that call0 and call1 now point to the same computation
  auto call0 = FindInstruction(module.get(), "call0");
  auto call1 = FindInstruction(module.get(), "call1");
  EXPECT_EQ(call0->to_apply(), call1->to_apply());
}

// Tests that the pass handles multi-level nested calls with duplicates
// at different levels.
// Initial graph:
// Entry -> A.1 -> Scalar.1
//       |      -> Scalar.2
//       -> A.2 -> Scalar.3
//              -> Scalar.4
// All Scalar.X are identical to each other and A.1 is identical to A.2.
//
// Example of one of the expected graph:
// Entry -> A.1 -> Scalar.1
//       |      -> Scalar.1
//       -> A.1 -> Scalar.1
//              -> Scalar.1
TEST_F(UnflattenCallGraphTest, DuplicatedMultilevelNestedCalls) {
  std::string hlo_string = R"(
HloModule NestedCalls, entry_computation_layout={(f32[])->f32[]}

%Scalar.1 (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0)
}

%Scalar.2 (param0.3: f32[]) -> f32[] {
  %param0.3 = f32[] parameter(0)
  ROOT %negate.1 = f32[] negate(%param0.3)
}

%Scalar.3 (param0.5: f32[]) -> f32[] {
  %param0.5 = f32[] parameter(0)
  ROOT %negate.2 = f32[] negate(%param0.5)
}

%Scalar.4 (param0.6: f32[]) -> f32[] {
  %param0.6 = f32[] parameter(0)
  ROOT %negate.3 = f32[] negate(%param0.6)
}

%A.2 (param0.1: f32[]) -> f32[] {
  %param0.1 = f32[] parameter(0)
  %call = f32[] call(%param0.1), to_apply=%Scalar.3
  ROOT %call.1 = f32[] call(%call), to_apply=%Scalar.4
}

%A.1 (param0.4: f32[]) -> f32[] {
  %param0.4 = f32[] parameter(0)
  %call.4 = f32[] call(%param0.4), to_apply=%Scalar.1
  ROOT %call.5 = f32[] call(%call.4), to_apply=%Scalar.2
}

ENTRY %FlattenCalls.Entry (param0.2: f32[]) -> f32[] {
  %param0.2 = f32[] parameter(0)
  %call.2 = f32[] call(%param0.2), to_apply=%A.1
  ROOT %call.3 = f32[] call(%call.2), to_apply=%A.2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 7);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  ASSERT_EQ(module->computation_count(), 3);

  // Check that A.1 and A.2 are deduplicated.
  auto A1 = FindComputation(module.get(), "A.1");
  auto A2 = FindComputation(module.get(), "A.2");
  EXPECT_NE(A1 == nullptr, A2 == nullptr);

  // Check that Scalar.X are deduplicated.
  std::vector<HloComputation*> deduped_computations = {
      FindComputation(module.get(), "Scalar.1"),
      FindComputation(module.get(), "Scalar.2"),
      FindComputation(module.get(), "Scalar.3"),
      FindComputation(module.get(), "Scalar.4")};
  EXPECT_EQ(absl::c_count(deduped_computations, nullptr), 3);
}

// Tests that the pass handles multi-level nested calls with duplicates
// at different levels, where merging at a lower level enables further
// merging at higher levels.
//
// Initial graph structure:
// main -> A -> C -> E -> terminal
//      -> B -> D -> F -> terminal
//
// Duplicates:
// - E and F are identical (call terminal).
// - C and D become identical after E and F are merged
//       (add, then call the same computation).
// - A and B become identical after C and D are merged
//       (multiply, then call the same computation).
//
// The pass should first merge E and F. This makes C and D equivalent,
// so they are merged. Finally, this makes A and B equivalent, leading
// to their merge.
//
// Expected computation count reduction from 8 to 5.
TEST_F(UnflattenCallGraphTest, DuplicatedMultilevelNestedCalls2) {
  std::string hlo_string = R"(
HloModule LinearChain, entry_computation_layout={(f32[])->f32[]}

%terminal (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0)
}

%F (param0: f32[]) -> f32[] {
  %Fparam0 = f32[] parameter(0)
  ROOT %Fterminal = f32[] call(%Fparam0), to_apply=%terminal
}

%E (param0: f32[]) -> f32[] {
  %Eparam0 = f32[] parameter(0)
  ROOT %Eterminal = f32[] call(%Eparam0), to_apply=%terminal
}

%D (param0: f32[]) -> f32[] {
  %Dparam0 = f32[] parameter(0)
  %Dadd = f32[] add(%Dparam0, %Dparam0)
  ROOT %call = f32[] call(%Dadd), to_apply=%F
}

%C (param0: f32[]) -> f32[] {
  %Cparam = f32[] parameter(0)
  %Cadd = f32[] add(%Cparam, %Cparam)
  ROOT %call = f32[] call(%Cadd), to_apply=%E
}

%B (param0: f32[]) -> f32[] {
  %notparam = f32[] parameter(0)
  %notmult = f32[] multiply(%notparam, %notparam)
  ROOT %call = f32[] call(%notmult), to_apply=%D
}

%A (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  %mult = f32[] multiply(%param0, %param0)
  ROOT %call = f32[] call(%mult), to_apply=%C
}

ENTRY %main (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  %call1 = f32[] call(%param0), to_apply=%A
  ROOT %call2 = f32[] call(%param0), to_apply=%B
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 8);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));

  EXPECT_TRUE(result);
  ASSERT_EQ(module->computation_count(), 5);

  auto F = FindComputation(module.get(), "F");
  auto E = FindComputation(module.get(), "E");
  auto D = FindComputation(module.get(), "D");
  auto C = FindComputation(module.get(), "C");
  auto B = FindComputation(module.get(), "B");
  auto A = FindComputation(module.get(), "A");

  // Check that following pairs are deduplicated (A and B, C and D, E and F)
  EXPECT_NE(A == nullptr, B == nullptr);
  EXPECT_NE(C == nullptr, D == nullptr);
  EXPECT_NE(E == nullptr, F == nullptr);
}

// Tests that the pass merges duplicate computations even if the argument names
// in the computations are different.
// The initial graph is:
// main -> called_computation_1 (param_A)
//      -> called_computation_2 (param_B)
// where called_computation_1 and called_computation_2 are structurally
// identical but use different parameter names.
// The expected graph is:
// main -> called_computation_1 (param_A)
//      -> called_computation_1 (param_A)
TEST_F(UnflattenCallGraphTest, DifferentArgumentNames) {
  std::string hlo_string =
      R"(HloModule DifferentArgumentNames, entry_computation_layout={(f32[])->(f32[], f32[])}

  %called_computation_1 (param_A: f32[]) -> f32[] {
    ROOT %param_A = f32[] parameter(0)
  }

  %called_computation_2 (param_B: f32[]) -> f32[] {
    ROOT %param_B = f32[] parameter(0)
  }

  ENTRY %main (a: f32[], b: f32[]) -> (f32[], f32[]) {
    %a = f32[] parameter(0)
    %call1 = f32[] call(%a), to_apply=%called_computation_1
    %call2 = f32[] call(%a), to_apply=%called_computation_2
    ROOT %tuple = (f32[], f32[]) tuple(%call1, %call2)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  ASSERT_EQ(module->computation_count(), 2);

  // Check that call1 and call2 now point to the same computation.
  auto call1 = FindInstruction(module.get(), "call1");
  auto call2 = FindInstruction(module.get(), "call2");
  EXPECT_EQ(call1->to_apply(), call2->to_apply());

  // Check that one of the computations was removed.
  auto called_computation_1 =
      FindComputation(module.get(), "called_computation_1");
  auto called_computation_2 =
      FindComputation(module.get(), "called_computation_2");
  ASSERT_NE(called_computation_1 == nullptr, called_computation_2 == nullptr);
}

// Tests that pass does not deduplicate conditional computations.
// The initial graph should not be changed.
TEST_F(UnflattenCallGraphTest, DontDeDuplicateInConditional) {
  std::string hlo_string = R"(
HloModule DuplicatesInConditional

  %branch_1 (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  %branch_2 (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  ENTRY %main (pred: pred[], a: f32[]) -> f32[] {
    %pred_0 = pred[] parameter(0)
    %a = f32[] parameter(1)
    ROOT %conditional = f32[] conditional(%pred_0, %a, %a), true_computation=%branch_1, false_computation=%branch_2
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  // No change expected.
  EXPECT_FALSE(result);
}

// Test that the pass only deduplicates computations that are only kCalled.
// The initial graph is:
// main -> called_computation_1 (kCalled)
//      -> if pred { called_computation_1 } (kControlFlow)
//      -> else { called_computation_2 } (kControlFlow)
//      -> called_computation_2 (kCalled)
//      -> called_computation_1 (kMap)
//      -> called_computation_2 (kMap)
// The expected graph should change kCalled called_computation_1 and
// called_computation_2 to the same computation, but not kControlFlow or kMap.
// main -> called_computation_1 (kCalled)
//      -> if pred { called_computation_1 } (kControlFlow)
//      -> else { called_computation_2 }  (kControlFlow)
//      -> called_computation_1 (kCalled)
//      -> called_computation_1 (kMap)
//      -> called_computation_2 (kMap)
TEST_F(UnflattenCallGraphTest, OnlyDeduplicateCalledComputations) {
  std::string hlo_string = R"(
HloModule OnlyDeduplicateCalledComputations

  %called_computation_1 (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  %called_computation_2 (param_0: f32[]) -> f32[] {
    ROOT %param_0 = f32[] parameter(0)
  }

  ENTRY %main (pred: pred[], a: f32[]) -> (f32[], f32[]) {
    %pred_0 = pred[] parameter(0)
    %a = f32[] parameter(1)
    %call1 = f32[] call(%a), to_apply=%called_computation_1
    %conditional = f32[] conditional(%pred_0, %a, %call1), true_computation=%called_computation_1, false_computation=%called_computation_2
    %call2 = f32[] call(%a), to_apply=%called_computation_2
    %map1 = f32[] map(%a), to_apply=%called_computation_1
    %map2 = f32[] map(%a), to_apply=%called_computation_2
    ROOT %tuple = (f32[], f32[]) tuple(%conditional, %map1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunUnflattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  // Check that computations did not get removed.
  auto called_computation_1 =
      FindComputation(module.get(), "called_computation_1");
  auto called_computation_2 =
      FindComputation(module.get(), "called_computation_2");

  EXPECT_TRUE(called_computation_1 != nullptr);
  EXPECT_TRUE(called_computation_2 != nullptr);

  // %call1 and %call2's called_computation be deduplicated.
  // They should be calling the  same computation.
  auto call1 = FindInstruction(module.get(), "call1");
  auto call2 = FindInstruction(module.get(), "call2");
  EXPECT_EQ(call2->to_apply(), call1->to_apply());

  // Conditional should be unchanged and calling the same computations.
  auto conditional = FindInstruction(module.get(), "conditional");
  auto true_computation = conditional->called_computations()[0];
  auto false_computation = conditional->called_computations()[1];
  EXPECT_EQ(true_computation, called_computation_1);
  EXPECT_EQ(false_computation, called_computation_2);

  // %map1 and %map2 should be calling the same computations.
  auto map1 = FindInstruction(module.get(), "map1");
  EXPECT_EQ(map1->to_apply(), called_computation_1);
  auto map2 = FindInstruction(module.get(), "map2");
  EXPECT_EQ(map2->to_apply(), called_computation_2);
}

}  // namespace
}  // namespace xla
