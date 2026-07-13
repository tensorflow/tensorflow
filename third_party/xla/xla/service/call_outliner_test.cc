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

#include "xla/service/call_outliner.h"

#include <algorithm>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/call_inliner.h"
#include "xla/service/call_marker.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class CallOutlinerTest : public HloHardwareIndependentTestBase {
 protected:
  // Helper to parse, run marker, inliner, and outliner.
  absl::StatusOr<std::unique_ptr<HloModule>> ParseInlineAndOutline(
      absl::string_view hlo_string) {
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_string));
    CallInliner call_inliner;
    CallMarker call_marker(call_inliner);
    ASSIGN_OR_RETURN(bool marked, call_marker.Run(module.get()));
    EXPECT_TRUE(marked);

    ASSIGN_OR_RETURN(bool inlined, call_inliner.Run(module.get()));
    EXPECT_TRUE(inlined);

    CallOutliner call_outliner;
    ASSIGN_OR_RETURN(bool outlined, call_outliner.Run(module.get()));
    EXPECT_TRUE(outlined);
    return module;
  }

  // Helper to parse and run outliner only (for pre-marked modules).
  absl::StatusOr<std::unique_ptr<HloModule>> OutlineModule(
      absl::string_view hlo_string) {
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_string));
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
  HloModule outline

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY entry {
    c = f32[] constant(1)
    call = f32[] call(c), to_apply=a
    ROOT tuple = (f32[], f32[]) tuple(call, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseInlineAndOutline(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule outline, entry_computation_layout={()->(f32[], f32[])}

a.1 {
  p0 = f32[] parameter(0)
  ROOT add.2 = f32[] add(p0, p0)
}

ENTRY entry {
  c = f32[] constant(1)
  call.1 = f32[] call(c), to_apply=a.1
  ROOT tuple = (f32[], f32[]) tuple(call.1, c)
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
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
                          ParseInlineAndOutline(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->f32[]}

a.1 {
  p0 = f32[] parameter(0)
  ROOT add.2 = f32[] add(p0, p0)
}

ENTRY inline {
  c = f32[] constant(1)
  ROOT call = f32[] call(c), to_apply=a.1
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseInlineAndOutline(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->f32[]}

c.1 {
  p0 = f32[] parameter(0)
  ROOT result.7 = f32[] add(p0, p0)
}

b.1 {
  p0.1 = f32[] parameter(0)
  ROOT call.1 = f32[] call(p0.1), to_apply=c.1
}

a.1 {
  p0.2 = f32[] parameter(0)
  ROOT call.3 = f32[] call(p0.2), to_apply=b.1
}

ENTRY inline {
  c = f32[] constant(1)
  ROOT call.4 = f32[] call(c), to_apply=a.1
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineModule(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={(f32[], f32[])->f32[]}

a {
  p0.1 = f32[] parameter(0)
  extra_1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(p0.1, extra_1)
}

ENTRY inline {
  p0 = f32[] parameter(0)
  ext = f32[] parameter(1)
  ROOT call = f32[] call(p0, ext), to_apply=a
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
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
                          ParseInlineAndOutline(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->f32[]}

a.1 {
  p0 = f32[] parameter(0)
  ROOT add.2 = f32[] add(p0, p0)
}

b.1 {
  p0.1 = f32[] parameter(0)
  ROOT mul.2 = f32[] multiply(p0.1, p0.1)
}

ENTRY inline {
  c = f32[] constant(1)
  call = f32[] call(c), to_apply=a.1
  ROOT call.1 = f32[] call(call), to_apply=b.1
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
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
                          ParseInlineAndOutline(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->f32[]}

a.1 {
  p0.1 = f32[] parameter(0)
  p1.1 = f32[] parameter(1)
  ROOT add.2 = f32[] add(p0.1, p1.1)
}

ENTRY inline {
  c1 = f32[] constant(1)
  c2 = f32[] constant(2)
  ROOT call = f32[] call(c1, c2), to_apply=a.1
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
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
                          ParseInlineAndOutline(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->f32[]}

a.1 {
  p0 = f32[] parameter(0)
  ROOT result.3 = f32[] bitcast(p0)
}

ENTRY inline {
  c = f32[] constant(1)
  ROOT call = f32[] call(c), to_apply=a.1
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
}

TEST_F(CallOutlinerTest, ErrorOnUnbalancedDeeplyNestedCalls) {
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

  auto module_or = OutlineModule(hlo_string);
  EXPECT_FALSE(module_or.status().ok());
  EXPECT_EQ(module_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(module_or.status().message(),
                                "Found _after marker for b but nested _before "
                                "marker for c was not closed."));
}

TEST_F(CallOutlinerTest, ErrorOnOverlappingMarkers) {
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

  auto module_or = OutlineModule(hlo_string);
  EXPECT_FALSE(module_or.status().ok());
  EXPECT_EQ(module_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(module_or.status().message(),
                                "Found _after marker for a but nested _before "
                                "marker for b was not closed."));
}

TEST_F(CallOutlinerTest, ErrorOnAbandonedBeforeMarker) {
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
  auto status = call_outliner.Run(module.get()).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Found _before marker without matching _after marker for a"));
}

TEST_F(CallOutlinerTest, ErrorOnMultipleAbandonedBeforeMarkers) {
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
  auto status = call_outliner.Run(module.get()).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Found _before marker without matching _after marker for a"));
}

TEST_F(CallOutlinerTest, ErrorOnAbandonedAfterMarkers) {
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
  auto status = call_outliner.Run(module.get()).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Found _after marker without matching _before marker for a"));
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

  CallInliner call_inliner;
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineModule(hlo_string));

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineModule(hlo_string));

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

TEST_F(CallOutlinerTest, OutlineCallWithMultipleOutputs) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    before = (f32[], f32[]) custom-call(p0, p1), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    gte0 = f32[] get-tuple-element(before), index=0
    gte1 = f32[] get-tuple-element(before), index=1
    add = f32[] add(gte0, gte1)
    sub = f32[] subtract(gte0, gte1)
    ROOT after = (f32[], f32[]) custom-call(add, sub), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineModule(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

a {
  p0.1 = f32[] parameter(0)
  p1.1 = f32[] parameter(1)
  add.1 = f32[] add(p0.1, p1.1)
  sub.1 = f32[] subtract(p0.1, p1.1)
  ROOT tuple = (f32[], f32[]) tuple(add.1, sub.1)
}

ENTRY inline {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT call = (f32[], f32[]) call(p0, p1), to_apply=a
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
}

TEST_F(CallOutlinerTest, OutlineConsecutiveCallsWithSharedConstant) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    before_a = f32[] custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    c1 = f32[] constant(1.0)
    add_a = f32[] add(before_a, c1)
    after_a = f32[] custom-call(add_a), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
    before_b = f32[] custom-call(after_a), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="b"}
    add_b = f32[] add(before_b, c1)
    ROOT after_b = f32[] custom-call(add_b), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="b"}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineModule(hlo_string));

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={(f32[])->f32[]}

a {
  p0.1 = f32[] parameter(0)
  c1.1 = f32[] constant(1)
  ROOT add_a.1 = f32[] add(p0.1, c1.1)
}

b {
  p0.2 = f32[] parameter(0)
  c1.clone = f32[] constant(1)
  ROOT add_b.1 = f32[] add(p0.2, c1.clone)
}

ENTRY inline {
  p0 = f32[] parameter(0)
  call = f32[] call(p0), to_apply=a
  ROOT call.1 = f32[] call(call), to_apply=b
}

)";
  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
}

TEST_F(CallOutlinerTest, RetainOpMetadata) {
  const absl::string_view hlo_string = R"(
  HloModule outline

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY entry {
    c = f32[] constant(1)
    call = f32[] call(c), to_apply=a, metadata={op_type="my_op" op_name="my_name" source_file="my_file.py" source_line=123}
    ROOT tuple = (f32[], f32[]) tuple(call, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseInlineAndOutline(hlo_string));

  HloInstruction* call = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->metadata().op_type(), "my_op");
  EXPECT_EQ(call->metadata().op_name(), "my_name");
  EXPECT_EQ(call->metadata().source_file(), "my_file.py");
  EXPECT_EQ(call->metadata().source_line(), 123);
}

TEST_F(CallOutlinerTest, RetainFrontendAttributes) {
  const absl::string_view hlo_string = R"(
  HloModule outline

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY entry {
    c = f32[] constant(1)
    call = f32[] call(c), to_apply=a, frontend_attributes={key1="val1", key2="val2"}
    ROOT tuple = (f32[], f32[]) tuple(call, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseInlineAndOutline(hlo_string));

  HloInstruction* call = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call, nullptr);
  ASSERT_TRUE(call->has_frontend_attributes());
  EXPECT_EQ(call->frontend_attributes().map().at("key1"), "val1");
  EXPECT_EQ(call->frontend_attributes().map().at("key2"), "val2");
}

TEST_F(CallOutlinerTest, RetainBackendConfig) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  ENTRY inline {
    p0 = f32[] parameter(0)
    before = f32[] custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    add = f32[] add(before, before)
    ROOT after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}, backend_config="my_backend_config"
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          OutlineModule(hlo_string));

  HloInstruction* call = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->raw_backend_config_string(), "my_backend_config");
}

TEST_F(CallOutlinerTest, RetainSharding) {
  const absl::string_view hlo_string = R"(
  HloModule outline

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY entry {
    c = f32[] constant(1)
    call = f32[] call(c), to_apply=a, sharding={replicated}
    ROOT tuple = (f32[], f32[]) tuple(call, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseInlineAndOutline(hlo_string));

  HloInstruction* call = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call, nullptr);
  ASSERT_TRUE(call->has_sharding());
  EXPECT_TRUE(call->sharding().IsReplicated());
}

TEST_F(CallOutlinerTest, RetainControlDependencies) {
  const absl::string_view hlo_string = R"(
  HloModule outline

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY entry {
    c = f32[] constant(1)
    other1 = f32[] constant(2)
    call = f32[] call(c), to_apply=a, control-predecessors={other1}
    other2 = f32[] constant(3), control-predecessors={call}
    ROOT tuple = (f32[], f32[], f32[], f32[]) tuple(call, c, other1, other2)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseInlineAndOutline(hlo_string));

  HloInstruction* call = FindCallByName(module->entry_computation(), "a");
  ASSERT_NE(call, nullptr);

  HloInstruction* other1 = FindInstruction(module.get(), "other1");
  ASSERT_NE(other1, nullptr);
  HloInstruction* other2 = FindInstruction(module.get(), "other2");
  ASSERT_NE(other2, nullptr);

  EXPECT_NE(std::find(call->control_predecessors().begin(),
                      call->control_predecessors().end(), other1),
            call->control_predecessors().end());
  EXPECT_NE(std::find(other2->control_predecessors().begin(),
                      other2->control_predecessors().end(), call),
            other2->control_predecessors().end());
}

TEST_F(CallOutlinerTest, OutlineWithExecutionThread) {
  const absl::string_view hlo_string = R"(
  HloModule outline

  ENTRY entry {
    p0 = f32[] parameter(0)
    before = f32[] custom-call(p0), custom_call_target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="a"}
    add = f32[] add(before, before)
    ROOT after = f32[] custom-call(add), custom_call_target="__xla_internal_call_marker_after", frontend_attributes={xla_call_marked_computation="a"}
  }, execution_thread="foo_thread")";

  // If we outline with execution_threads={"bar_thread"} (which does not
  // match the entry computation's execution thread "foo_thread"), it should not
  // outline.
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_string));
    CallOutliner call_outliner;
    TF_ASSERT_OK_AND_ASSIGN(bool outlined,
                            call_outliner.Run(module.get(), {"bar_thread"}));
    EXPECT_FALSE(outlined);
  }

  // If we outline with execution_threads={"foo_thread"}, it should outline.
  // The newly created outlined computation should inherit "foo_thread"
  // as its execution thread.
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_string));
    CallOutliner call_outliner;
    TF_ASSERT_OK_AND_ASSIGN(bool outlined,
                            call_outliner.Run(module.get(), {"foo_thread"}));
    EXPECT_TRUE(outlined);

    HloInstruction* call = FindCallByName(module->entry_computation(), "a");
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->to_apply()->execution_thread(), "foo_thread");
  }
}

}  // namespace
}  // namespace xla
