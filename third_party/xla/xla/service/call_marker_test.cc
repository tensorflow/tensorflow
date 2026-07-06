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

#include "xla/service/call_marker.h"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/call_inliner.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using CallMarkerTest = HloHardwareIndependentTestBase;

TEST_F(CallMarkerTest, MarkSingleCall) {
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
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->(f32[], f32[])}

a {
  p = f32[] parameter(0)
  ROOT add = f32[] add(p, p)
}

ENTRY inline {
  c = f32[] constant(1)
  custom-call = (f32[]) custom-call(c), custom_call_target="__xla_internal_call_marker_before", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a"}
  get-tuple-element = f32[] get-tuple-element(custom-call), index=0
  a = f32[] call(get-tuple-element), to_apply=a
  custom-call.1 = f32[] custom-call(a), custom_call_target="__xla_internal_call_marker_after", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a",xla_call_marked_instruction_name="a"}
  ROOT tuple = (f32[], f32[]) tuple(custom-call.1, c)
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
}

TEST_F(CallMarkerTest, MarkRootCall) {
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
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  // This test checks that if the call instruction is the root of the entry
  // computation, it is correctly wrapped and the "after" marker custom call
  // becomes the new root of the entry computation.
  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->f32[]}

a {
  p = f32[] parameter(0)
  ROOT add = f32[] add(p, p)
}

ENTRY inline {
  c = f32[] constant(1)
  custom-call = (f32[]) custom-call(c), custom_call_target="__xla_internal_call_marker_before", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a"}
  get-tuple-element = f32[] get-tuple-element(custom-call), index=0
  a = f32[] call(get-tuple-element), to_apply=a
  ROOT custom-call.1 = f32[] custom-call(a), custom_call_target="__xla_internal_call_marker_after", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a",xla_call_marked_instruction_name="a"}
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
}

TEST_F(CallMarkerTest, MarkRootCallPreservesEntryResultLayout) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module, entry_computation_layout={()->f32[4,4]{0,1}}

  a {
    p = f32[4,4]{1,0} parameter(0)
    ROOT add = f32[4,4]{1,0} add(p, p)
  }

  ENTRY inline {
    c = f32[4,4]{1,0} constant(1)
    ROOT a = f32[4,4]{1,0} call(c), to_apply=a
  })";

  HloParserOptions parser_options;
  parser_options.set_keep_module_auto_layouts(true);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest(),
                                   parser_options));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  EXPECT_EQ(
      module->entry_computation()->root_instruction()->shape().layout(),
      module->entry_computation_layout().result_layout().shape().layout());
}

TEST_F(CallMarkerTest, MarkNestedCalls) {
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
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);
  HloInstruction* call_before_c = FindInstruction(module.get(), "custom-call");
  EXPECT_TRUE(call_before_c != nullptr);
  EXPECT_EQ(call_before_c->custom_call_target(), kCallMarkerBeforeTarget);
  EXPECT_TRUE(call_before_c->has_frontend_attributes());
  EXPECT_EQ(call_before_c->frontend_attributes().map().at(
                kCallMarkedComputationAttribute),
            "c");

  HloInstruction* call_after_a = FindInstruction(module.get(), "custom-call.5");
  EXPECT_TRUE(call_after_a != nullptr);
  EXPECT_EQ(call_after_a->custom_call_target(), kCallMarkerAfterTarget);
  EXPECT_TRUE(call_after_a->has_frontend_attributes());
  EXPECT_EQ(call_after_a->frontend_attributes().map().at(
                kCallMarkedComputationAttribute),
            "a");
}

// This test verifies call marking when calls are nested inside a while loop.
//
// Call graph of computations:
//
//          [ body ]
//          /   | \
//         /    |  \
//        v     |   v
//  [callee1]   |   [callee2]
//    /   \     |
//   /     \    v
//  v       \  [callee4]
// [callee3] \  |
//            v v
//          [callee5]
//
TEST_F(CallMarkerTest, MarkCallInsideWhileLoop) {
  const absl::string_view hlo_string = R"(
  HloModule module

  callee5 {
    p = f32[] parameter(0)
    ROOT mul = f32[] multiply(p, p)
  }

  callee3 {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  callee4 {
    p = f32[] parameter(0)
    call_c5 = f32[] call(p), to_apply=callee5
    ROOT sub = f32[] subtract(call_c5, p)
  }

  callee1 {
    p0 = f32[] parameter(0)
    p1 = ((f32[], f32[]), f32[]) parameter(1)
    p1_0 = (f32[], f32[]) get-tuple-element(p1), index=0
    p1_1 = f32[] get-tuple-element(p1), index=1
    p1_0_0 = f32[] get-tuple-element(p1_0), index=0
    p1_0_1 = f32[] get-tuple-element(p1_0), index=1

    call_c3 = f32[] call(p0), to_apply=callee3
    call_c5 = f32[] call(p1_1), to_apply=callee5

    add0 = f32[] add(call_c3, p1_0_0)
    sub0 = f32[] subtract(p1_0_1, call_c5)

    tup_inner = (f32[], f32[]) tuple(add0, sub0)
    ROOT result = ((f32[], f32[]), f32[]) tuple(tup_inner, call_c3)
  }

  callee2 {
    p = ((f32[], f32[]), f32[]) parameter(0)
    p_0 = (f32[], f32[]) get-tuple-element(p), index=0
    p_1 = f32[] get-tuple-element(p), index=1
    p_0_0 = f32[] get-tuple-element(p_0), index=0
    p_0_1 = f32[] get-tuple-element(p_0), index=1

    mul = f32[] multiply(p_0_0, p_0_1)
    add = f32[] add(mul, p_1)
    ROOT result = (f32[], f32[]) tuple(add, mul)
  }

  cond {
    state = (s32[], f32[], ((f32[], f32[]), f32[])) parameter(0)
    iter = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(10)
    ROOT cmp = pred[] compare(iter, limit), direction=LT
  }

  body {
    state = (s32[], f32[], ((f32[], f32[]), f32[])) parameter(0)
    iter = s32[] get-tuple-element(state), index=0
    val = f32[] get-tuple-element(state), index=1
    nested_tup = ((f32[], f32[]), f32[]) get-tuple-element(state), index=2

    one = s32[] constant(1)
    next_iter = s32[] add(iter, one)

    call1 = ((f32[], f32[]), f32[]) call(val, nested_tup), to_apply=callee1
    call2 = (f32[], f32[]) call(call1), to_apply=callee2
    call4 = f32[] call(val), to_apply=callee4

    next_nested_tup = ((f32[], f32[]), f32[]) tuple(call2, call4)
    next_val = f32[] get-tuple-element(call2), index=0

    ROOT next_state = (s32[], f32[], ((f32[], f32[]), f32[])) tuple(next_iter, next_val, next_nested_tup)
  }

  ENTRY entry {
    iter0 = s32[] constant(0)
    val0 = f32[] constant(1.0)
    tup_inner0 = (f32[], f32[]) tuple(val0, val0)
    nested_tup0 = ((f32[], f32[]), f32[]) tuple(tup_inner0, val0)
    init_state = (s32[], f32[], ((f32[], f32[]), f32[])) tuple(iter0, val0, nested_tup0)
    ROOT loop = (s32[], f32[], ((f32[], f32[]), f32[])) while(init_state), condition=cond, body=body
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  auto status = verifier().Run(module.get()).status();
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(CallMarkerTest, MetadataAndAttributesCopiedToMarkers) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    my_call = f32[] call(c), to_apply=a, metadata={op_type="my_op" op_name="my_name" source_file="my_file.py" source_line=123}, frontend_attributes={key1="val1"}, sharding={replicated}
    ROOT tuple = (f32[], f32[]) tuple(my_call, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  HloInstruction* before = FindInstruction(module.get(), "custom-call");
  ASSERT_NE(before, nullptr);
  EXPECT_EQ(before->custom_call_target(), kCallMarkerBeforeTarget);

  // Check 'before' marker has ONLY xla_call_marked_computation attribute.
  ASSERT_TRUE(before->has_frontend_attributes());
  EXPECT_EQ(before->frontend_attributes().map().size(), 1);
  EXPECT_EQ(before->frontend_attributes().map().at(
                kCallMarkedComputationAttribute.data()),
            "a");

  // Check 'before' has no metadata, sharding or backend config
  EXPECT_TRUE(before->metadata().op_name().empty());
  EXPECT_FALSE(before->has_sharding());
  EXPECT_FALSE(before->has_backend_config());

  HloInstruction* after = FindInstruction(module.get(), "custom-call.1");
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(after->custom_call_target(), kCallMarkerAfterTarget);

  // Check 'after' marker has all frontend attributes.
  ASSERT_TRUE(after->has_frontend_attributes());
  EXPECT_EQ(after->frontend_attributes().map().at(
                kCallMarkedComputationAttribute.data()),
            "a");
  EXPECT_EQ(after->frontend_attributes().map().at("key1"), "val1");

  // Check 'after' has metadata.
  EXPECT_EQ(after->metadata().op_type(), "my_op");
  EXPECT_EQ(after->metadata().op_name(), "my_name");
  EXPECT_EQ(after->metadata().source_file(), "my_file.py");
  EXPECT_EQ(after->metadata().source_line(), 123);

  // Check 'after' has sharding.
  ASSERT_TRUE(after->has_sharding());
  EXPECT_TRUE(after->sharding().IsReplicated());
}

TEST_F(CallMarkerTest, ControlDependenciesSplitBetweenMarkers) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    other1 = f32[] constant(2)
    my_call = f32[] call(c), to_apply=a, control-predecessors={other1}
    other2 = f32[] constant(3), control-predecessors={my_call}
    ROOT tuple = (f32[], f32[], f32[], f32[]) tuple(my_call, c, other1, other2)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  HloInstruction* before = FindInstruction(module.get(), "custom-call");
  ASSERT_NE(before, nullptr);
  EXPECT_EQ(before->custom_call_target(), kCallMarkerBeforeTarget);

  HloInstruction* after = FindInstruction(module.get(), "custom-call.1");
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(after->custom_call_target(), kCallMarkerAfterTarget);

  HloInstruction* other1 = FindInstruction(module.get(), "other1");
  ASSERT_NE(other1, nullptr);
  HloInstruction* other2 = FindInstruction(module.get(), "other2");
  ASSERT_NE(other2, nullptr);

  HloInstruction* call = FindInstruction(module.get(), "my_call");
  ASSERT_NE(call, nullptr);

  // Control predecessors: other1 should point to before, not call.
  EXPECT_NE(std::find(before->control_predecessors().begin(),
                      before->control_predecessors().end(), other1),
            before->control_predecessors().end());
  EXPECT_EQ(std::find(call->control_predecessors().begin(),
                      call->control_predecessors().end(), other1),
            call->control_predecessors().end());

  // Control successors: after should point to other2, not call.
  EXPECT_NE(std::find(other2->control_predecessors().begin(),
                      other2->control_predecessors().end(), after),
            other2->control_predecessors().end());
  EXPECT_EQ(std::find(other2->control_predecessors().begin(),
                      other2->control_predecessors().end(), call),
            other2->control_predecessors().end());
}

TEST_F(CallMarkerTest, ReusedComputationNameAfterInliningAndCleanup) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    p = f32[] parameter(0)
    ROOT add = f32[] add(p, p)
  }

  ENTRY inline {
    c = f32[] constant(1)
    b = f32[] call(c), to_apply=a
    ROOT tuple = (f32[], f32[]) tuple(b, c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);

  // Run CallMarker and CallInliner.
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);
  TF_ASSERT_OK_AND_ASSIGN(bool inlined, inliner.Run(module.get()));
  EXPECT_TRUE(inlined);

  // Call Cleanup().
  module->Cleanup();

  // Create call computation with the same name as the initial one had ("a").
  HloComputation::Builder sub_builder("a");
  auto p = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p"));
  sub_builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kNegate, p));
  HloComputation* new_sub = module->AddEmbeddedComputation(sub_builder.Build());

  HloComputation* entry = module->entry_computation();
  // Find the constant instruction 'c' in the entry computation.
  HloInstruction* c = nullptr;
  for (HloInstruction* inst : entry->instructions()) {
    if (inst->name() == "c") {
      c = inst;
      break;
    }
  }
  ASSERT_NE(c, nullptr);

  HloInstruction* new_call = entry->AddInstruction(
      HloInstruction::CreateCall(ShapeUtil::MakeShape(F32, {}), {c}, new_sub));

  HloInstruction* root = entry->root_instruction();
  EXPECT_TRUE(root->ReplaceOperandWith(1, new_call).ok());

  // Run CallMarker and CallInliner again.
  TF_ASSERT_OK_AND_ASSIGN(mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  TF_ASSERT_OK_AND_ASSIGN(inlined, inliner.Run(module.get()));
  EXPECT_TRUE(inlined);

  module->Cleanup();

  const absl::string_view expected_hlo =
      R"(HloModule inline_module, entry_computation_layout={()->(f32[], f32[])}

ENTRY inline {
  c = f32[] constant(1)
  custom-call = (f32[]) custom-call(c), custom_call_target="__xla_internal_call_marker_before", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a"}
  get-tuple-element = f32[] get-tuple-element(custom-call), index=0
  add.1 = f32[] add(get-tuple-element, get-tuple-element)
  custom-call.1 = f32[] custom-call(add.1), custom_call_target="__xla_internal_call_marker_after", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a",xla_call_marked_instruction_name="b"}
  custom-call.2 = (f32[]) custom-call(c), custom_call_target="__xla_internal_call_marker_before", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a.1"}
  get-tuple-element.1 = f32[] get-tuple-element(custom-call.2), index=0
  negate.1 = f32[] negate(get-tuple-element.1)
  custom-call.3 = f32[] custom-call(negate.1), custom_call_target="__xla_internal_call_marker_after", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a.1",xla_call_marked_instruction_name="call"}
  ROOT tuple = (f32[], f32[]) tuple(custom-call.1, custom-call.3)
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), expected_hlo);
}

TEST_F(CallMarkerTest, SkipCallWithNoParameters) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    ROOT c = f32[] constant(1.0)
  }

  ENTRY inline {
    a = f32[] call(), to_apply=a
    ROOT tuple = (f32[]) tuple(a)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_FALSE(mutated);
}

TEST_F(CallMarkerTest, DontLoseLayoutOnParameters) {
  // Test check that we don't lose layout on parameters when we mark and inline
  // a call. Here in initial `hlo_string`, arugment (constant `c`) has layout
  // {1,0}, but in the callee `a` it has layout {0,1}. Which is layout mismatch.
  // Call marker should respect this pre-set layouts and not change them.
  // so after marking and inlining the call, the argument should be still the
  // same, with the same layout (in this case in `inlined` string it is
  // `get-tuple-element = f32[4,4]{0,1} get-tuple-element(custom-call),index=0`)
  const absl::string_view hlo_string =
      R"(HloModule inline_module, entry_computation_layout={()->f32[4,4]{1,0}}

  a {
    p = f32[4,4]{0,1} parameter(0)
    ROOT add = f32[4,4] add(p, p)
  }

  ENTRY inline {
    c = f32[4,4]{1,0} constant(1)
    ROOT a = f32[4,4]{1,0} call(c), to_apply=a
  })";

  HloParserOptions parser_options;
  parser_options.set_keep_module_auto_layouts(true);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest(),
                                   parser_options));
  CallInliner inliner;
  CallMarker call_marker(inliner);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_marker.Run(module.get()));
  EXPECT_TRUE(mutated);

  const absl::string_view just_call_marked =
      R"(HloModule inline_module, entry_computation_layout={()->f32[4,4]{1,0}}

a {
  p = f32[4,4]{0,1} parameter(0)
  ROOT add = f32[4,4]{1,0} add(p, p)
}

ENTRY inline {
  c = f32[4,4]{1,0} constant({ { 1, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } })
  custom-call = (f32[4,4]{0,1}) custom-call(c), custom_call_target="__xla_internal_call_marker_before", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a"}
  get-tuple-element = f32[4,4]{0,1} get-tuple-element(custom-call), index=0
  a = f32[4,4]{1,0} call(get-tuple-element), to_apply=a
  ROOT custom-call.1 = f32[4,4]{1,0} custom-call(a), custom_call_target="__xla_internal_call_marker_after", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a",xla_call_marked_instruction_name="a"}
}

)";

  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()),
            just_call_marked);

  TF_ASSERT_OK_AND_ASSIGN(mutated, inliner.Run(module.get()));
  EXPECT_TRUE(mutated);

  const absl::string_view inlined =
      R"(HloModule inline_module, entry_computation_layout={()->f32[4,4]{1,0}}

ENTRY inline {
  c = f32[4,4]{1,0} constant({ { 1, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } })
  custom-call = (f32[4,4]{0,1}) custom-call(c), custom_call_target="__xla_internal_call_marker_before", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a"}
  get-tuple-element = f32[4,4]{0,1} get-tuple-element(custom-call), index=0
  add.1 = f32[4,4]{1,0} add(get-tuple-element, get-tuple-element)
  ROOT custom-call.1 = f32[4,4]{1,0} custom-call(add.1), custom_call_target="__xla_internal_call_marker_after", custom_call_has_side_effect=true, frontend_attributes={xla_call_marked_computation="a",xla_call_marked_instruction_name="a"}
}

)";
  EXPECT_EQ(module->ToString(HloPrintOptions::ShortParsable()), inlined);
}
}  // namespace
}  // namespace xla
