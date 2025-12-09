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

#include "xla/service/call_outliner.h"

#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/call_inliner.h"
#include "xla/service/call_marker.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class CallOutlinerTest : public HloHardwareIndependentTestBase {
 protected:
  // Helper to parse, optionally run marker/inliner, and run outliner.
  absl::StatusOr<std::unique_ptr<HloModule>> OutlineAndReturnModule(
      absl::string_view hlo_string, bool run_marker_and_inliner = true,
      std::optional<bool> single_call_inlining = std::nullopt) {
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_string));

    if (run_marker_and_inliner) {
      CallInliner call_inliner(single_call_inlining.value_or(false));
      CallMarker call_marker(call_inliner);
      ASSIGN_OR_RETURN(bool marked, call_marker.Run(module.get()));
      EXPECT_TRUE(marked);

      ASSIGN_OR_RETURN(bool inlined, call_inliner.Run(module.get()));
      EXPECT_TRUE(inlined);
      LOG(INFO) << "Inlined module:\n" << module->ToString();
      LOG(INFO) << "--------------------------------------------------";
      LOG(INFO) << module->computation_count() << " computations";
      LOG(INFO) << "--------------------------------------------------";
    }

    CallOutliner call_outliner;
    ASSIGN_OR_RETURN(bool outlined, call_outliner.Run(module.get()));
    EXPECT_TRUE(outlined);
    return module;
  }

  // Helper to find a call instruction by the prefix name of its target
  // computation.
  HloInstruction* FindCallByName(HloComputation* computation,
                                 absl::string_view name) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall &&
          absl::StartsWith(instruction->to_apply()->name(), name)) {
        return instruction;
      }
    }
    return nullptr;
  }

  // Helper to count the number of call instructions in a computation.
  int CountCalls(HloComputation* computation) {
    int count = 0;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall) {
        count++;
      }
    }
    return count;
  }
};

TEST_F(CallOutlinerTest, OutlineSingleCall) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    a = f32[] call(c), to_apply=a
    ROOT tuple = (f32[], f32[]) tuple(a, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineAndReturnModule(hlo_string));

  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);
}

TEST_F(CallOutlinerTest, OutlineRootCall) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    ROOT a = f32[] call(c), to_apply=a
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineAndReturnModule(hlo_string));

  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction(), call_a);
}

TEST_F(CallOutlinerTest, OutlineNestedCalls) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  c {
    p = f32[] parameter(0)
    ROOT result = f32[] add(p, p)
  }

  b {
    p = f32[] parameter(0)
    ROOT result = f32[] call(p), to_apply=c
  }

  a {
    p = f32[] parameter(0)
    ROOT result = f32[] call(p), to_apply=b
  }

  ENTRY inline {
    c = f32[] constant(1)
    ROOT result = f32[] call(c), to_apply=a
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, true, /*single_call_inlining=*/false));

  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction(), call_a);

  HloInstruction* call_b = FindCallByName(call_a->to_apply(), "b");
  ASSERT_NE(call_b, nullptr);
  EXPECT_EQ(call_a->to_apply()->root_instruction(), call_b);

  HloInstruction* call_c = FindCallByName(call_b->to_apply(), "c");
  ASSERT_NE(call_c, nullptr);
  EXPECT_EQ(call_b->to_apply()->root_instruction(), call_c);
}

TEST_F(CallOutlinerTest, OutlineCallWithCapturedValue) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    ext = f32[] parameter(1)
    before = (f32[]) custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    gte = f32[] get-tuple-element(before), index=0
    add = f32[] add(gte, ext)
    ROOT after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));

  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);
  EXPECT_EQ(call_a->operand_count(),
            2);  // Should have captured `ext` as second operand.
}

TEST_F(CallOutlinerTest, MultipleSequentialCalls) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }
  b {
    p = f32[] parameter(0)
    ROOT mul = f32[] multiply(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    call_a = f32[] call(c), to_apply=a
    ROOT call_b = f32[] call(call_a), to_apply=b
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineAndReturnModule(hlo_string));

  EXPECT_EQ(CountCalls(module->entry_computation()), 2);
}

TEST_F(CallOutlinerTest, CallWithMultipleArguments) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY inline {
    c1 = f32[] constant(1)
    c2 = f32[] constant(2)
    ROOT result = f32[] call(c1, c2), to_apply=a
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineAndReturnModule(hlo_string));

  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);
  EXPECT_EQ(call_a->operand_count(), 2);
}

TEST_F(CallOutlinerTest, OutlineCallWithBitcastBody) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT result = f32[] bitcast(p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    ROOT result = f32[] call(c), to_apply=a
  })";

  // Use bitcast as a no-op that might be simplified away, or just return
  // parameter directly.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineAndReturnModule(hlo_string));

  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);
  EXPECT_EQ(call_a->to_apply()->root_instruction()->opcode(),
            HloOpcode::kBitcast);
}

TEST_F(CallOutlinerTest, UnbalancedDeeplyNestedCalls) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    a_before = (f32[]) custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    a_gte = f32[] get-tuple-element(a_before), index=0

    b_before = (f32[]) custom-call(a_gte), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="b"}
    b_gte = f32[] get-tuple-element(b_before), index=0

    c_before = (f32[]) custom-call(b_gte), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="c"}
    c_gte = f32[] get-tuple-element(c_before), index=0

    add = f32[] add(c_gte, c_gte)

    b_after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="b"}

    ROOT a_after = f32[] custom-call(b_after), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));

  // We expect a and b to be outlined. c_before will be stuck inside b's
  // computation because it has no c_after.
  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);

  HloInstruction* call_b = FindCallByName(call_a->to_apply(), "b");
  ASSERT_NE(call_b, nullptr);

  bool c_before_found = false;
  for (HloInstruction* instruction : call_b->to_apply()->instructions()) {
    if (instruction->opcode() == HloOpcode::kCustomCall &&
        instruction->custom_call_target() == kCallMarkerBeforeTarget &&
        instruction->has_frontend_attributes() &&
        instruction->frontend_attributes().map().at(
            kCallMarkedComputationAttribute) == "c") {
      c_before_found = true;
      break;
    }
  }
  EXPECT_TRUE(c_before_found);
}

TEST_F(CallOutlinerTest, MalformedModule_OverlappingMarkers) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    a_before = (f32[]) custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    a_gte = f32[] get-tuple-element(a_before), index=0

    b_before = (f32[]) custom-call(a_gte), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="b"}
    b_gte = f32[] get-tuple-element(b_before), index=0

    add = f32[] add(b_gte, b_gte)

    a_after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}

    ROOT b_after = f32[] custom-call(a_after), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="b"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));
  EXPECT_EQ(CountCalls(module->entry_computation()), 1);
}

TEST_F(CallOutlinerTest, CleanupAbandonedBeforeMarker) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    before = (f32[]) custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    gte = f32[] get-tuple-element(before), index=0
    ROOT add = f32[] add(gte, gte)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallOutliner call_outliner;
  TF_ASSERT_OK_AND_ASSIGN(bool outlined, call_outliner.Run(module.get()));
  EXPECT_TRUE(outlined);

  // Verify that no markers are left in the module.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerBeforeTarget);
    }
  }
}

TEST_F(CallOutlinerTest, CleanupMultipleAbandonedBeforeMarkers) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    before1 = (f32[]) custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    gte1 = f32[] get-tuple-element(before1), index=0

    before2 = (f32[]) custom-call(gte1), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="b"}
    gte2 = f32[] get-tuple-element(before2), index=0

    ROOT add = f32[] add(gte2, gte2)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallOutliner call_outliner;
  TF_ASSERT_OK_AND_ASSIGN(bool outlined, call_outliner.Run(module.get()));
  EXPECT_TRUE(outlined);

  // Verify that no before markers are left in the module.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerBeforeTarget);
    }
  }
}

TEST_F(CallOutlinerTest, CleanupAbandonedAfterMarkers) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    add1 = f32[] add(p0, p0)
    after1 = f32[] custom-call(add1), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
    add2 = f32[] add(after1, after1)
    ROOT after2 = f32[] custom-call(add2), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="b"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallOutliner call_outliner;
  TF_ASSERT_OK_AND_ASSIGN(bool outlined, call_outliner.Run(module.get()));
  EXPECT_TRUE(outlined);

  // Verify that no markers are left in the module.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerAfterTarget);
    }
  }
}

// Visualization of the call graph:
//
//       ENTRY
//       /   \
//    node3 --> node2
//     |         |  \
//     |         |   leaf
//     |         |
//     +------> node1
//               |
//              leaf
TEST_F(CallOutlinerTest, OutlineNestedAndSharedCalls) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  leaf {
    p = f32[] parameter(0)
    ROOT result = f32[] add(p, p)
  }

  node1 {
    p = f32[] parameter(0)
    ROOT result = f32[] call(p), to_apply=leaf
  }

  node2 {
    p = f32[] parameter(0)
    c1 = f32[] call(p), to_apply=leaf
    c2 = f32[] call(c1), to_apply=node1
    ROOT result = f32[] add(c1, c2)
  }

  node3 {
    p = f32[] parameter(0)
    c1 = f32[] call(p), to_apply=node2
    c2 = f32[] call(p), to_apply=node1
    ROOT result = f32[] add(c1, c2)
  }

  ENTRY inline {
    c = f32[] constant(1)
    c1 = f32[] call(c), to_apply=node3
    c2 = f32[] call(c), to_apply=node2
    ROOT result = f32[] add(c1, c2)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  std::string original_str = module->ToString();

  CallInliner call_inliner(false);
  CallMarker call_marker(call_inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool marked, call_marker.Run(module.get()));
  EXPECT_TRUE(marked);

  TF_ASSERT_OK_AND_ASSIGN(bool inlined, call_inliner.Run(module.get()));
  EXPECT_TRUE(inlined);

  CallOutliner call_outliner;
  TF_ASSERT_OK_AND_ASSIGN(bool outlined, call_outliner.Run(module.get()));
  EXPECT_TRUE(outlined);

  // Verify structure.
  HloComputation* entry = module->entry_computation();

  // Entry should have calls to node3 and node2.
  HloInstruction* call_node3 = FindCallByName(entry, "node3");
  ASSERT_NE(call_node3, nullptr);
  HloInstruction* call_node2 = FindCallByName(entry, "node2");
  ASSERT_NE(call_node2, nullptr);

  // node3 should have calls to node2 and node1.
  HloInstruction* call_node3_node2 =
      FindCallByName(call_node3->to_apply(), "node2");
  ASSERT_NE(call_node3_node2, nullptr);
  HloInstruction* call_node3_node1 =
      FindCallByName(call_node3->to_apply(), "node1");
  ASSERT_NE(call_node3_node1, nullptr);

  // node2 should have calls to leaf and node1.
  HloInstruction* call_node2_leaf =
      FindCallByName(call_node2->to_apply(), "leaf");
  ASSERT_NE(call_node2_leaf, nullptr);
  HloInstruction* call_node2_node1 =
      FindCallByName(call_node2->to_apply(), "node1");
  ASSERT_NE(call_node2_node1, nullptr);

  // call_node2_node1 (node1) should have calls to leaf.
  HloInstruction* call_node1_leaf =
      FindCallByName(call_node2_node1->to_apply(), "leaf");
  ASSERT_NE(call_node1_leaf, nullptr);

  // call_node3_node1 (node1) should also have calls to leaf.
  HloInstruction* call_node3_node1_leaf =
      FindCallByName(call_node3_node1->to_apply(), "leaf");
  ASSERT_NE(call_node3_node1_leaf, nullptr);

  EXPECT_NE(module->ToString(), original_str);
}
TEST_F(CallOutlinerTest, OutlineCallWithTupleUserOfBeforeMarker) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    before = (f32[], f32[]) custom-call(p0, p1), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    gte0 = f32[] get-tuple-element(before), index=0
    gte1 = f32[] get-tuple-element(before), index=1
    add = f32[] add(gte0, gte1)
    nested_tuple = ((f32[], f32[])) tuple(before)
    after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
    ROOT root = (f32[], ((f32[], f32[]))) tuple(after, nested_tuple)
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));

  // Verify that 'before' is gone.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerBeforeTarget);
    }
  }

  // Verify that nested_tuple now uses a reconstructed tuple.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* nested_tuple = root->operand(1);
  ASSERT_EQ(nested_tuple->opcode(), HloOpcode::kTuple);
  const HloInstruction* reconstructed_tuple = nested_tuple->operand(0);
  ASSERT_EQ(reconstructed_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(reconstructed_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(reconstructed_tuple->operand(1)->opcode(), HloOpcode::kParameter);
}

TEST_F(CallOutlinerTest, OutlineCallWithNonTupleBeforeMarker) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    before = f32[] custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    add = f32[] add(before, before)
    ROOT after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));

  // Verify that 'before' is gone.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerBeforeTarget);
    }
  }

  // Verify that 'add' now uses 'p0' directly.
  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);

  // The outlined computation root should be an add of parameter 0 and parameter
  // 0.
  HloComputation* outlined_comp = call_a->to_apply();
  HloInstruction* root = outlined_comp->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);

  // The call instruction itself should have p0 as operand.
  EXPECT_EQ(call_a->operand(0)->opcode(), HloOpcode::kParameter);
}

TEST_F(CallOutlinerTest, OutlineCallWithTupleUserOfBeforeMarker) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    before = (f32[], f32[]) custom-call(p0, p1), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    gte0 = f32[] get-tuple-element(before), index=0
    gte1 = f32[] get-tuple-element(before), index=1
    add = f32[] add(gte0, gte1)
    nested_tuple = ((f32[], f32[])) tuple(before)
    after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
    ROOT root = (f32[], ((f32[], f32[]))) tuple(after, nested_tuple)
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));

  // Verify that 'before' is gone.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerBeforeTarget);
    }
  }

  // Verify that nested_tuple now uses a reconstructed tuple.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* nested_tuple = root->operand(1);
  ASSERT_EQ(nested_tuple->opcode(), HloOpcode::kTuple);
  const HloInstruction* reconstructed_tuple = nested_tuple->operand(0);
  ASSERT_EQ(reconstructed_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(reconstructed_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(reconstructed_tuple->operand(1)->opcode(), HloOpcode::kParameter);
}

TEST_F(CallOutlinerTest, OutlineCallWithNonTupleBeforeMarker) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    before = f32[] custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    add = f32[] add(before, before)
    ROOT after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      OutlineAndReturnModule(hlo_string, /*run_marker_and_inliner=*/false));

  // Verify that 'before' is gone.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCustomCall) {
      EXPECT_NE(inst->custom_call_target(), kCallMarkerBeforeTarget);
    }
  }

  // Verify that 'add' now uses 'p0' directly.
  HloInstruction* call_a = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call_a, nullptr);

  // The outlined computation root should be an add of parameter 0 and parameter
  // 0.
  HloComputation* outlined_comp = call_a->to_apply();
  HloInstruction* root = outlined_comp->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);

  // The call instruction itself should have p0 as operand.
  EXPECT_EQ(call_a->operand(0)->opcode(), HloOpcode::kParameter);
}

}  // namespace
}  // namespace xla
