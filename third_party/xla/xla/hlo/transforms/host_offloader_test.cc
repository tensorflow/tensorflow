/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/host_offloader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/host_offload_legalize.h"
#include "xla/layout.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/service/host_offload_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace m = ::xla::match;

namespace xla {
namespace {

class HostOffloaderTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunHostOffloader(HloModule* module,
                                        bool after_layout = false) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (module->has_schedule()) {
      return absl::InternalError("Expected a non-scheduled module");
    }
    bool changed = false;
    HostOffloadLegalize host_offload_legalize(Layout::kHostMemorySpace,
                                              after_layout);
    TF_ASSIGN_OR_RETURN(bool legal_changed, host_offload_legalize.Run(module));
    changed |= legal_changed;
    HostOffloader host_offloader(Layout::kHostMemorySpace);
    TF_ASSIGN_OR_RETURN(bool offload_changed, host_offloader.Run(module));
    changed |= offload_changed;
    return changed;
  }

  void TestShapeHasMemorySpace(const Shape& shape, int64_t memory_space) {
    ASSERT_TRUE(shape.has_layout());
    EXPECT_EQ(shape.layout().memory_space(), memory_space);
  }

  bool HaveRemainingOffloadAnnotations(const HloModule* module) {
    for (const HloComputation* computation : module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->IsCustomCall(
                {host_memory_offload_annotations::kMoveToHostCustomCallTarget,
                 host_memory_offload_annotations::
                     kMoveToDeviceCustomCallTarget})) {
          return true;
        }
      }
    }
    return false;
  }
};

absl::flat_hash_set<const HloInstruction*>
getInstructionsWithOpcodeFromComputation(const HloComputation* computation,
                                         HloOpcode target_opcode) {
  absl::flat_hash_set<const HloInstruction*> instructions;
  for (const HloInstruction* instruction : computation->instructions()) {
    if (instruction->opcode() == target_opcode) {
      instructions.emplace(instruction);
    }
  }
  return instructions;
}

TEST_F(HostOffloaderTest, BasicDusDs) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // "AllocateBuffer"  param_0  _...
  //               |  /        /
  //           dynamic-update-slice  _...
  //                          |     /
  //                       dynamic-slice
  HloInstruction* param;
  HloInstruction* allocate_buffer;
  HloInstruction* dynamic_update_slice;
  HloInstruction* dynamic_slice;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice(
                  &dynamic_slice,
                  m::DynamicUpdateSlice(
                      &dynamic_update_slice,
                      m::CustomCall(&allocate_buffer, {"AllocateBuffer"}),
                      m::Parameter(&param, 0), m::Op(), m::Op(), m::Op()),
                  m::Op(), m::Op(), m::Op())));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_slice->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, DusFirstOperandIsNotFromABroadcast) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  param_2 = f32[2,2048,2048] parameter(2)
  constant_s32_0 = s32[] constant(0)
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(param_2, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const absl::StatusOr<bool> result = RunHostOffloader(module.get());
  EXPECT_FALSE(result.ok());
}

TEST_F(HostOffloaderTest, DusDsWithTupleAfterBroadcast) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  tuple = (f32[2,2048,2048]) tuple(broadcast)
  gte = f32[2,2048,2048] get-tuple-element(tuple), index=0
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(gte, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // "AllocateBuffer"
  //               |
  //             tuple
  //               |
  //              gte  param_0  _...
  //               |  /        /
  //           dynamic-update-slice  _...
  //                          |     /
  //                       dynamic-slice
  HloInstruction* param;
  HloInstruction* allocate_buffer;
  HloInstruction* tuple;
  HloInstruction* gte;
  HloInstruction* dynamic_update_slice;
  HloInstruction* dynamic_slice;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice(
                  &dynamic_slice,
                  m::DynamicUpdateSlice(
                      &dynamic_update_slice,
                      m::GetTupleElement(
                          &gte,
                          m::Tuple(&tuple, m::CustomCall(&allocate_buffer,
                                                         {"AllocateBuffer"})),
                          0),
                      m::Parameter(&param, 0), m::Op(), m::Op(), m::Op()),
                  m::Op(), m::Op(), m::Op())));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_slice->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, DusWithoutDs) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  ROOT load_custom_call = f32[2,2048,2048] custom-call(dynamic_update_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // "AllocateBuffer"  param_0  _...
  //               |  /        /
  //           dynamic-update-slice
  //                          |
  //                         copy
  HloInstruction* param;
  HloInstruction* allocate_buffer;
  HloInstruction* dynamic_update_slice;
  HloInstruction* copy;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(
          &copy, m::DynamicUpdateSlice(
                     &dynamic_update_slice,
                     m::CustomCall(&allocate_buffer, {"AllocateBuffer"}),
                     m::Parameter(&param, 0), m::Op(), m::Op(), m::Op()))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, DusAndNoCopyFromSameCustomCall) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  tuple = (f32[1,2048,2048]) tuple(offload_custom_call)
  gte = f32[1,2048,2048] get-tuple-element(tuple), index=0
  load_custom_call_0 = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
  load_custom_call_1 = f32[1,2048,2048] custom-call(gte), custom_call_target="MoveToDevice"
  ROOT tuple_1 = (f32[1,2048,2048], f32[1,2048,2048]) tuple(load_custom_call_0, load_custom_call_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // "AllocateBuffer"      param_0
  //               |          /   \_________
  //               |         /              \
  //               |        /              copy
  //               |       /                 |
  //               |      /     _...       tuple
  //               |     /     /             |
  //           dynamic-update-slice  _...   gte
  //                          |     /        |
  //                       dynamic-slice   copy
  //                                   \    /
  //                                    tuple
  HloInstruction* param_match_1;
  HloInstruction* param_match_2;
  HloInstruction* allocate_buffer;
  HloInstruction* dynamic_update_slice;
  HloInstruction* dynamic_slice;
  HloInstruction* copy_to_host;
  HloInstruction* tuple_0;
  HloInstruction* gte;
  HloInstruction* copy_to_device;
  HloInstruction* tuple_1;

  const auto dynamic_slice_pattern = m::DynamicSlice(
      &dynamic_slice,
      m::DynamicUpdateSlice(&dynamic_update_slice,
                            m::CustomCall(&allocate_buffer, {"AllocateBuffer"}),
                            m::Parameter(&param_match_1, 0), m::Op(), m::Op(),
                            m::Op()),
      m::Op(), m::Op(), m::Op());
  const auto copy_pattern = m::Copy(
      &copy_to_device,
      m::GetTupleElement(
          &gte,
          m::Tuple(&tuple_0,
                   m::Copy(&copy_to_host, m::Parameter(&param_match_2, 0))),
          0));
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(&tuple_1, dynamic_slice_pattern, copy_pattern)));
  EXPECT_EQ(param_match_1, param_match_2);
  TestShapeHasMemorySpace(param_match_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_slice->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_0->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {0}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {1}),
                          Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, BasicAsyncCustomCallWithAliasing) {
  const std::string& hlo_string = R"(
HloModule m, input_output_alias={{}: (0, {}, must-alias)},
             entry_computation_layout={(f32[4096]{0:T(128)S(5)})->f32[4096]{0:T(128)S(5)}}

ENTRY %main (a: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %async-start = ((f32[4096]{0}), f32[4096]{0}, u32[]) custom-call-start(%a),
                 custom_call_target="Foo",
                 output_to_operand_aliasing={{}: (0, {})}
  ROOT %async-done = f32[4096]{0} custom-call-done(%async-start)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  TestShapeHasMemorySpace(async_done->shape(), Layout::kHostMemorySpace);
}

TEST_F(HostOffloaderTest, ParameterStreamingWithXposeCopyFeedingIntoWhile) {
  const std::string& hlo_string = R"(
HloModule jit__prefill_impl, entry_computation_layout={(bf16[2,16,16]{2,1,0:T(8,128)(2,1)S(5)})->bf16[2,16,16]{1,2,0:T(8,128)(2,1)}}

while_condition {
  condition_param = (s32[], bf16[2,16,16]{1,2,0:T(8,128)(2,1)}, bf16[2,16,16]{1,2,0:T(8,128)(2,1)}) parameter(0)
  condition_current_iteration_index = s32[] get-tuple-element(condition_param), index=0
  condition_iteration_count = s32[] constant(16)
  ROOT condition_result = pred[] compare(condition_current_iteration_index, condition_iteration_count), direction=LT
}

while_body {
  input_tuple.0 = (s32[], bf16[2,16,16]{1,2,0:T(8,128)(2,1)}, bf16[2,16,16]{1,2,0:T(8,128)(2,1)}) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  orig_data = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} get-tuple-element(input_tuple.0), index=1
  custom-call.0 = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} custom-call(orig_data), custom_call_target="MoveToDevice"
  sum = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} get-tuple-element(input_tuple.0), index=2
  sum.1 = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} add(custom-call.0, sum)

  constant_1 = s32[] constant(1)
  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1)
  ROOT tuple_result.0 = (s32[], bf16[2,16,16]{1,2,0:T(8,128)(2,1)}, bf16[2,16,16]{1,2,0:T(8,128)(2,1)}) tuple(incremented_index.0, custom-call.0, sum.1)
}

ENTRY main {
  param.0 = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} parameter(0)
  copy = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} copy(param.0)
  constant_0 = s32[] constant(0)
  constant_0.0 = bf16[] constant(0.0)
  broadcast = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} broadcast(constant_0.0), dimensions={}
  tuple_for_while = (s32[], bf16[2,16,16]{1,2,0:T(8,128)(2,1)}, bf16[2,16,16]{1,2,0:T(8,128)(2,1)}) tuple(constant_0, copy, broadcast)
  while = (s32[], bf16[2,16,16]{1,2,0:T(8,128)(2,1)}, bf16[2,16,16]{1,2,0:T(8,128)(2,1)}) while(tuple_for_while), condition=while_condition, body=while_body
  ROOT gte = bf16[2,16,16]{1,2,0:T(8,128)(2,1)} get-tuple-element(while), index=2
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunHostOffloader(module.get(), /*after_layout=*/true));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
  HloVerifier verifier(/*layout_sensitive=*/true,
                       /*allow_mixed_precision=*/true);
  TF_EXPECT_OK(verifier.Run(module.get()).status());
  VLOG(1) << "module after: " << module->ToString();
}

TEST_F(HostOffloaderTest, ParameterStreamingFeedingIntoWhile) {
  const std::string& hlo_string = R"(
HloModule jit__prefill_impl, entry_computation_layout={(bf16[2,16,16]{2,1,0:T(8,128)(2,1)S(5)})->bf16[2,16,16]{2,1,0:T(8,128)(2,1)}}

while_condition {
  condition_param = (s32[], bf16[2,16,16]{2,1,0:T(8,128)(2,1)}, bf16[2,16,16]{2,1,0:T(8,128)(2,1)}) parameter(0)
  condition_current_iteration_index = s32[] get-tuple-element(condition_param), index=0
  condition_iteration_count = s32[] constant(16)
  ROOT condition_result = pred[] compare(condition_current_iteration_index, condition_iteration_count), direction=LT
}

while_body {
  input_tuple.0 = (s32[], bf16[2,16,16]{2,1,0:T(8,128)(2,1)}, bf16[2,16,16]{2,1,0:T(8,128)(2,1)}) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  orig_data = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} get-tuple-element(input_tuple.0), index=1
  custom-call.0 = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} custom-call(orig_data), custom_call_target="MoveToDevice"
  sum = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} get-tuple-element(input_tuple.0), index=2
  sum.1 = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} add(custom-call.0, sum)

  constant_1 = s32[] constant(1)
  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1)
  ROOT tuple_result.0 = (s32[], bf16[2,16,16]{2,1,0:T(8,128)(2,1)}, bf16[2,16,16]{2,1,0:T(8,128)(2,1)}) tuple(incremented_index.0, custom-call.0, sum.1)
}

ENTRY main {
  param.0 = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} parameter(0)
  constant_0 = s32[] constant(0)
  constant_0.0 = bf16[] constant(0.0)
  broadcast = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} broadcast(constant_0.0), dimensions={}
  tuple_for_while = (s32[], bf16[2,16,16]{2,1,0:T(8,128)(2,1)}, bf16[2,16,16]{2,1,0:T(8,128)(2,1)}) tuple(constant_0, param.0, broadcast)
  while = (s32[], bf16[2,16,16]{2,1,0:T(8,128)(2,1)}, bf16[2,16,16]{2,1,0:T(8,128)(2,1)}) while(tuple_for_while), condition=while_condition, body=while_body
  ROOT gte = bf16[2,16,16]{2,1,0:T(8,128)(2,1)} get-tuple-element(while), index=2
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunHostOffloader(module.get(), /*after_layout=*/true));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
  HloVerifier verifier(/*layout_sensitive=*/true,
                       /*allow_mixed_precision=*/true);
  TF_EXPECT_OK(verifier.Run(module.get()).status());
  VLOG(1) << "module after: " << module->ToString();
}

TEST_F(HostOffloaderTest, ParameterStreamingInScanLoop) {
  const std::string& hlo_string = R"(
HloModule m,
  entry_computation_layout={(f32[8,2]{0,1:T(2,128)S(5)})->(f32[]{:T(256)}, f32[8,2]{0,1:T(2,128)})},
  allow_spmd_sharding_propagation_to_output={true,true}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

while_body {
  arg_tuple.8 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[8,2]{0,1:T(2,128)}) parameter(0)
  get-tuple-element.9 = s32[]{:T(256)} get-tuple-element(arg_tuple.8), index=0
  constant.12 = s32[]{:T(256)} constant(1)
  add.29 = s32[]{:T(256)} add(get-tuple-element.9, constant.12)
  get-tuple-element.10 = f32[8,2]{0,1:T(2,128)} get-tuple-element(arg_tuple.8), index=1
  get-tuple-element.11 = f32[8,2]{0,1:T(2,128)} get-tuple-element(arg_tuple.8), index=2
  constant.16 = s32[]{:T(256)} constant(0)
  dynamic-slice.20 = f32[1,2]{0,1:T(2,128)} dynamic-slice(get-tuple-element.11, get-tuple-element.9,  constant.16), dynamic_slice_sizes={1,2}
  constant.1 = f32[] constant(-0)
  reduce = f32[2]{0:T(256)} reduce(dynamic-slice.20, constant.1), dimensions={0}, to_apply=add
  custom-call = f32[2]{0:T(256)} custom-call(reduce), custom_call_target="MoveToDevice"
  constant.13 = f32[]{:T(256)} constant(1)
  broadcast.14 = f32[2]{0:T(256)} broadcast(constant.13), dimensions={}
  add.23 = f32[2]{0:T(256)} add(custom-call, broadcast.14)
  reshape.24 = f32[1,2]{0,1:T(2,128)} reshape(add.23)
  dynamic-update-slice.28 = f32[8,2]{0,1:T(2,128)} dynamic-update-slice(get-tuple-element.10, reshape.24, get-tuple-element.9, constant.16)
  ROOT tuple.30 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[8,2]{0,1:T(2,128)}) tuple(add.29, dynamic-update-slice.28,  get-tuple-element.11)
}

condition {
  arg_tuple.32 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[8,2]{0,1:T(2,128)}) parameter(0)
  get-tuple-element.33 = s32[]{:T(256)} get-tuple-element(arg_tuple.32), index=0
  constant.36 = s32[]{:T(256)} constant(8)
  ROOT compare.37 = pred[]{:T(1024)} compare(get-tuple-element.33, constant.36), direction=LT
}

ENTRY e {
  constant.3 = f32[]{:T(256)} constant(1)
  constant.2 = s32[]{:T(256)} constant(0)
  constant.4 = f32[]{:T(256)} constant(0)
  broadcast.5 = f32[8,2]{0,1:T(2,128)} broadcast(constant.4), dimensions={}
  Arg_0.1 = f32[8,2]{0,1:T(2,128)} parameter(0), sharding={replicated}
  tuple.6 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[8,2]{0,1:T(2,128)}) tuple(constant.2, broadcast.5, Arg_0.1)
  while.38 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[8,2]{0,1:T(2,128)}) while(tuple.6), condition=condition, body=while_body
  get-tuple-element.40 = f32[8,2]{0,1:T(2,128)} get-tuple-element(while.38), index=1
  ROOT tuple.42 = (f32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}) tuple(constant.3, get-tuple-element.40)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunHostOffloader(module.get(), /*after_layout=*/true));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
  HloVerifier verifier(/*layout_sensitive=*/true,
                       /*allow_mixed_precision=*/true);
  TF_EXPECT_OK(verifier.Run(module.get()).status());
}

TEST_F(HostOffloaderTest, OutputStreamingInScanLoop) {
  const std::string& hlo_string = R"(
HloModule m,
  entry_computation_layout={(f32[4,1]{0,1:T(2,128)})->(f32[]{:T(256)}, f32[8,2]{0,1:T(2,128)S(5)})},
  allow_spmd_sharding_propagation_to_output={true,true}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

while_body {
  param.1 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[4,1]{0,1:T(2,128)}) parameter(0)
  get-tuple-element.1 = s32[]{:T(256)} get-tuple-element(param.1), index=0
  constant.9 = s32[]{:T(256)} constant(1)
  add.1 = s32[]{:T(256)} add(get-tuple-element.1, constant.9)
  get-tuple-element.2 = f32[8,2]{0,1:T(2,128)} get-tuple-element(param.1), index=1
  get-tuple-element.3 = f32[4,1]{0,1:T(2,128)} get-tuple-element(param.1), index=2
  bitcast = f32[1,4,1]{1,2,0:T(2,128)} bitcast(get-tuple-element.3)
  all-gather.2 = f32[4,4,1]{1,2,0:T(2,128)} all-gather(bitcast), channel_id=2, replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
  constant.20 = f32[] constant(-0)
  reduce = f32[4,4]{1,0:T(4,128)} reduce(all-gather.2, constant.20), dimensions={2}, to_apply=add
  bitcast.1 = f32[2,4,2,1]{1,2,0,3:T(2,128)} bitcast(reduce)
  copy.1 = f32[2,4,2,1]{1,0,2,3:T(2,128)} copy(bitcast.1)
  reshape.6 = f32[8,2]{0,1:T(2,128)} reshape(copy.1)
  constant.10 = s32[]{:T(256)} constant(0)
  dynamic-slice.0 = f32[1,2]{0,1:T(2,128)} dynamic-slice(reshape.6, get-tuple-element.1, constant.10), dynamic_slice_sizes={1,2}
  constant.11 = f32[]{:T(256)} constant(1)
  broadcast.4 = f32[1,2]{0,1:T(2,128)} broadcast(constant.11), dimensions={}
  add.2 = f32[1,2]{0,1:T(2,128)} add(dynamic-slice.0, broadcast.4)
  reduce.1 = f32[2]{0:T(256)} reduce(add.2, constant.20), dimensions={0}, to_apply=add
  custom-call.1 = f32[2]{0:T(256)} custom-call(reduce.1), custom_call_target="MoveToHost"
  reshape.8 = f32[1,2]{0,1:T(2,128)} reshape(custom-call.1)
  dynamic-update-slice.0 = f32[8,2]{0,1:T(2,128)} dynamic-update-slice(get-tuple-element.2, reshape.8, get-tuple-element.1, constant.10)
  ROOT tuple = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[4,1]{0,1:T(2,128)}) tuple(add.1, dynamic-update-slice.0, get-tuple-element.3)
}

condition {
  param = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[4,1]{0,1:T(2,128)}) parameter(0)
  get-tuple-element = s32[]{:T(256)} get-tuple-element(param), index=0
  constant.8 = s32[]{:T(256)} constant(8)
  ROOT compare.0 = pred[]{:T(1024)} compare(get-tuple-element, constant.8), direction=LT
}

ENTRY e {
  constant.17 = f32[]{:T(256)} constant(1)
  constant.18 = s32[]{:T(256)} constant(0)
  constant.19 = f32[]{:T(256)} constant(0)
  broadcast.6 = f32[8,2]{0,1:T(2,128)} broadcast(constant.19), dimensions={}
  param.2 = f32[4,1]{0,1:T(2,128)} parameter(0), sharding={devices=[2,2]<=[4]}
  tuple.1 = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[4,1]{0,1:T(2,128)}) tuple(constant.18, broadcast.6, param.2)
  while = (s32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}, f32[4,1]{0,1:T(2,128)}) while(tuple.1), condition=condition, body=while_body
  get-tuple-element.4 = f32[8,2]{0,1:T(2,128)} get-tuple-element(while), index=1
  ROOT tuple.2 = (f32[]{:T(256)}, f32[8,2]{0,1:T(2,128)}) tuple(constant.17, get-tuple-element.4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunHostOffloader(module.get(), /*after_layout=*/true));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
  HloVerifier verifier(/*layout_sensitive=*/true,
                       /*allow_mixed_precision=*/true);
  TF_EXPECT_OK(verifier.Run(module.get()).status());
}

TEST_F(HostOffloaderTest, BasicNoCopy) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[2048] parameter(0)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  ROOT load_custom_call = f32[2048] custom-call(offload_custom_call), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(&copy_to_device,
                         m::Copy(&copy_to_host, m::Parameter(&param, 0)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyThroughTuple) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[2048] parameter(0)
  other_param = f32[2048] parameter(1)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  tuple = (f32[2048], f32[2048]) tuple(offload_custom_call, other_param)
  gte_0 = f32[2048] get-tuple-element(tuple), index=0
  gte_1 = f32[2048] get-tuple-element(tuple), index=1
  ROOT load_custom_call = f32[2048] custom-call(gte_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |   _
  //   |  /
  // tuple
  //   |
  //  gte
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* tuple;
  HloInstruction* gte;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(
          &copy_to_device,
          m::GetTupleElement(
              &gte,
              m::Tuple(&tuple, m::Copy(&copy_to_host, m::Parameter(&param, 0)),
                       m::Op()),
              0))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyThroughNestedTuple) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[2048] parameter(0)
  other_param_0 = f32[2048] parameter(1)
  other_param_1 = f32[2048] parameter(2)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  tuple_0 = (f32[2048], f32[2048]) tuple(offload_custom_call, other_param_0)
  tuple_1 = ((f32[2048], f32[2048]), f32[2048]) tuple(tuple_0, other_param_1)
  gte_0 = (f32[2048], f32[2048]) get-tuple-element(tuple_1), index=0
  gte_1 = f32[2048] get-tuple-element(tuple_1), index=1
  gte_2 = f32[2048] get-tuple-element(gte_0), index=0
  gte_3 = f32[2048] get-tuple-element(gte_0), index=1
  ROOT load_custom_call = f32[2048] custom-call(gte_2), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |   _
  //   |  /
  // tuple_0
  //   |   _
  //   |  /
  // tuple_1
  //   |
  //  gte_0
  //   |
  //  gte_1
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* tuple_0;
  HloInstruction* gte_0;
  HloInstruction* tuple_1;
  HloInstruction* gte_1;
  HloInstruction* copy_to_device;
  const auto copy_param_pattern =
      m::Copy(&copy_to_host, m::Parameter(&param, 0));
  const auto tuple_of_tuple_pattern = m::Tuple(
      &tuple_1, m::Tuple(&tuple_0, copy_param_pattern, m::Op()), m::Op());
  const auto gte_of_gte_pattern = m::GetTupleElement(
      &gte_1, m::GetTupleElement(&gte_0, tuple_of_tuple_pattern, 0), 0);
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(&copy_to_device, gte_of_gte_pattern)));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_0->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_0->shape(), {1}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(gte_0->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(gte_0->shape(), {1}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {0, 0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {0, 1}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {1}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyThroughComputation) {
  const std::string& hlo_string = R"(
HloModule my_module
other_computation {
  ROOT param = f32[2048] parameter(0)
}
ENTRY main {
  data_param = f32[2048] parameter(0)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  call = f32[2048] call(offload_custom_call), to_apply=other_computation
  ROOT load_custom_call = f32[2048] custom-call(call), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern in the entry computation:
  // param
  //   |
  // copy (to host)
  //   |             ___
  //   |            /   \
  // call===========   param
  //   |            \___/
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* call;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(
          &copy_to_device,
          m::Call(&call, m::Copy(&copy_to_host, m::Parameter(&param, 0))))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(call->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  ASSERT_THAT(call->called_computations(), ::testing::SizeIs(1));
  HloComputation* called_computation = call->called_computations()[0];
  HloInstruction* called_computation_param;
  ASSERT_THAT(called_computation->root_instruction(),
              GmockMatch(m::Parameter(&called_computation_param, 0)));
  TestShapeHasMemorySpace(called_computation_param->shape(),
                          Layout::kHostMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyThroughComputationAndTuple) {
  const std::string& hlo_string = R"(
HloModule my_module
other_computation {
  param_0 = f32[2048] parameter(0)
  param_1 = f32[2048] parameter(1)
  ROOT tuple = (f32[2048], f32[2048]) tuple(param_0, param_1)
}
ENTRY main {
  data_param = f32[2048] parameter(0)
  other_param = f32[2048] parameter(1)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  call = (f32[2048], f32[2048]) call(offload_custom_call, other_param), to_apply=other_computation
  gte_0 = f32[2048] get-tuple-element(call), index=0
  gte_1 = f32[2048] get-tuple-element(call), index=1
  ROOT load_custom_call = f32[2048] custom-call(gte_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param0
  //   |
  // copy (to host)
  //   |              __________
  //   |  _          /      /   \
  //   | /          /   param0 param1
  //  call==========         \ /
  //   |            \       tuple
  //  gte            \_______/
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* call;
  HloInstruction* gte;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(
          &copy_to_device,
          m::GetTupleElement(
              &gte,
              m::Call(&call, m::Copy(&copy_to_host, m::Parameter(&param, 0)),
                      m::Op())))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(call->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(call->shape(), {1}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_THAT(call->called_computations(), ::testing::SizeIs(1));
  HloComputation* called_computation = call->called_computations()[0];
  HloInstruction* called_computation_param_0;
  HloInstruction* called_computation_param_1;
  HloInstruction* tuple;
  ASSERT_THAT(
      called_computation->root_instruction(),
      GmockMatch(m::Tuple(&tuple, m::Parameter(&called_computation_param_0, 0),
                          m::Parameter(&called_computation_param_1, 1))));
  TestShapeHasMemorySpace(called_computation_param_0->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(called_computation_param_1->shape(),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                          Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyThroughWhile) {
  const std::string& hlo_string = R"(
HloModule my_module
while_body {
  ROOT param = f32[2048] parameter(0)
}
while_condition {
  param = f32[2048] parameter(0)
  constant_0 = s32[] constant(0)
  constant_1 = s32[] constant(1)
  ROOT pred_result = pred[] compare(constant_1, constant_0), direction=LT
}
ENTRY main {
  data_param = f32[2048] parameter(0)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  while = f32[2048] while(offload_custom_call), condition=while_condition, body=while_body
  ROOT load_custom_call = f32[2048] custom-call(while), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //                                     __
  //                                    /  \
  // param                             /  param
  //   |                 <CONDITION>===
  // copy (to host)     /
  //   |               /
  // while=============
  //   |               \           __
  // copy (to device)   \         /  \
  //                     <BODY>===  param
  //                              \__/

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* while_instr;
  HloInstruction* copy_to_device;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(
                  &copy_to_device,
                  m::While(&while_instr,
                           m::Copy(&copy_to_host, m::Parameter(&param, 0))))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(while_instr->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  HloComputation* while_condition = while_instr->while_condition();
  ASSERT_THAT(while_condition->parameter_instructions(), ::testing::SizeIs(1));
  TestShapeHasMemorySpace(while_condition->parameter_instruction(0)->shape(),
                          Layout::kHostMemorySpace);

  HloInstruction* while_body_param;
  HloComputation* while_body = while_instr->while_body();
  ASSERT_THAT(while_body->root_instruction(),
              GmockMatch(m::Parameter(&while_body_param, 0)));
  TestShapeHasMemorySpace(while_body_param->shape(), Layout::kHostMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyWithOptBarrier) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[2048] parameter(0)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  tuple = (f32[2048]) tuple(offload_custom_call)
  opt_barrier = (f32[2048]) opt-barrier(tuple)
  get_tuple_element = f32[2048] get-tuple-element(opt_barrier), index=0
  ROOT load_custom_call = f32[2048] custom-call(get_tuple_element), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |
  // tuple
  //   |
  // opt-barrier
  //   |
  // get-tuple-element
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* tuple;
  HloInstruction* opt_barrier;
  HloInstruction* gte;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(
          &copy_to_device,
          m::GetTupleElement(
              &gte, m::OptimizationBarrier(
                        &opt_barrier,
                        m::Tuple(&tuple, m::Copy(&copy_to_host,
                                                 m::Parameter(&param, 0))))))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyMultipleToDevice) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  constant = f32[] constant(0)
  custom_call_0 = f32[] custom-call(constant), custom_call_target="MoveToHost"
  tuple_0 = (f32[], f32[]) tuple(custom_call_0, custom_call_0)
  opt_barrier = (f32[], f32[]) opt-barrier(tuple_0)
  gte_0 = f32[] get-tuple-element(opt_barrier), index=0
  custom_call_1 = f32[] custom-call(gte_0), custom_call_target="MoveToDevice"
  gte_1 = f32[] get-tuple-element(opt_barrier), index=1
  custom_call_2 = f32[] custom-call(gte_1), custom_call_target="MoveToDevice"
  ROOT tuple_1 = (f32[], f32[]) tuple(custom_call_1, custom_call_2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //  constant
  //      |
  //    copy
  //    |  |
  //    tuple
  //      |
  // opt-barrier
  //    /  \
  //  gte  gte
  //   |    |
  //  copy copy
  //    \  /
  //   tuple
  HloInstruction* constant;
  HloInstruction* copy_to_host_1;
  HloInstruction* copy_to_host_2;
  HloInstruction* tuple_1;
  HloInstruction* opt_barrier;
  HloInstruction* gte_1;
  HloInstruction* copy_to_device_1;
  HloInstruction* gte_2;
  HloInstruction* copy_to_device_2;
  HloInstruction* tuple_2;
  const auto constant_pattern = m::ConstantScalar(&constant, 0);
  const auto opt_barrier_pattern = m::OptimizationBarrier(
      &opt_barrier,
      m::Tuple(&tuple_1, m::Copy(&copy_to_host_1, constant_pattern),
               m::Copy(&copy_to_host_2, constant_pattern)));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  &tuple_2,
                  m::Copy(&copy_to_device_1,
                          m::GetTupleElement(&gte_1, opt_barrier_pattern)),
                  m::Copy(&copy_to_device_2,
                          m::GetTupleElement(&gte_2, opt_barrier_pattern)))));
  TestShapeHasMemorySpace(constant->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host_1->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_host_2->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_1->shape(), {1}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {1}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device_1->shape(),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte_2->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device_2->shape(),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_2->shape(), {0}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple_2->shape(), {1}),
                          Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyWithOptBarrierMoreElaborate) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

ENTRY main.24 {
  Arg_0.1 = f32[16]{0} parameter(0), sharding={devices=[2]<=[2]}
  cosine.4 = f32[16]{0} cosine(Arg_0.1)
  custom-call.5 = f32[16]{0} custom-call(cosine.4), custom_call_target="MoveToHost"
  sine.3 = f32[16]{0} sine(Arg_0.1)
  cosine.7 = f32[16]{0} cosine(sine.3)
  custom-call.8 = f32[16]{0} custom-call(cosine.7), custom_call_target="MoveToHost"
  sine.6 = f32[16]{0} sine(sine.3)
  cosine.9 = f32[16]{0} cosine(sine.6)
  custom-call.10 = f32[16]{0} custom-call(cosine.9), custom_call_target="MoveToHost"
  constant.2 = f32[] constant(1)
  tuple.11 = (f32[16]{0}, f32[16]{0}, f32[16]{0}, f32[]) tuple(custom-call.5, custom-call.8, custom-call.10, constant.2)
  opt-barrier.12 = (f32[16]{0}, f32[16]{0}, f32[16]{0}, f32[]) opt-barrier(tuple.11)
  get-tuple-element.16 = f32[] get-tuple-element(opt-barrier.12), index=3
  broadcast.20 = f32[16]{0} broadcast(get-tuple-element.16), dimensions={}
  get-tuple-element.15 = f32[16]{0} get-tuple-element(opt-barrier.12), index=2
  custom-call.19 = f32[16]{0} custom-call(get-tuple-element.15), custom_call_target="MoveToDevice"
  multiply.21 = f32[16]{0} multiply(broadcast.20, custom-call.19)
  get-tuple-element.14 = f32[16]{0} get-tuple-element(opt-barrier.12), index=1
  custom-call.18 = f32[16]{0} custom-call(get-tuple-element.14), custom_call_target="MoveToDevice"
  multiply.22 = f32[16]{0} multiply(multiply.21, custom-call.18)
  get-tuple-element.13 = f32[16]{0} get-tuple-element(opt-barrier.12), index=0
  custom-call.17 = f32[16]{0} custom-call(get-tuple-element.13), custom_call_target="MoveToDevice"
  ROOT multiply.23 = f32[16]{0} multiply(multiply.22, custom-call.17)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //                          param                         constant
  //                __________/ |                             |
  //               /            |                             |
  //          cosine           sine                           |
  //            |               |  \____________              |
  //            |               |               \             |
  //            |               |              sine           |
  //            |               |                |            |
  //            |             cosine          cosine          |
  //            |               |               |             |
  //       copy(to host)   copy(to host)   copy(to host)      |
  //                  \                \   /                  |
  //                   \______________  | |  _________________/
  //                                  \ | | /
  //                                   tuple
  //                                     |
  //                                 opt-barrier
  //                   _____________/   /  \   \_____________
  //                  /                /    \                \
  // get-tuple-element  get-tuple-element  get-tuple-element  get-tuple-element
  //        |                   |                  |                  |
  //   copy(to device)     copy(to device)    copy(to device)     broadcast
  //                  \                   \                 \    /
  //                   \                   \__________     multiply
  //                    \                             \       /
  //                     \                             multiply
  //                      \_________________________        /
  //                                                \      /
  //                                                multiply

  HloInstruction* param;
  HloInstruction* constant;
  HloInstruction* sine_0;
  HloInstruction* sine_1;
  HloInstruction* cosine_0;
  HloInstruction* cosine_1;
  HloInstruction* cosine_2;
  HloInstruction* copy_to_host_0;
  HloInstruction* copy_to_host_1;
  HloInstruction* copy_to_host_2;
  HloInstruction* tuple;
  HloInstruction* opt_barrier;
  HloInstruction* gte_0;
  HloInstruction* gte_1;
  HloInstruction* gte_2;
  HloInstruction* gte_3;
  HloInstruction* broadcast;
  HloInstruction* copy_to_device_0;
  HloInstruction* copy_to_device_1;
  HloInstruction* copy_to_device_2;
  HloInstruction* multiply_0;
  HloInstruction* multiply_1;
  HloInstruction* multiply_2;

  auto parameter_matcher = m::Parameter(&param, 0);
  auto first_sine_matcher = m::Op(&sine_0)
                                .WithOpcode(xla::HloOpcode::kSin)
                                .WithOperand(0, parameter_matcher);
  auto opt_barrier_matcher = m::OptimizationBarrier(
      &opt_barrier,
      m::Tuple(
          &tuple,
          m::Copy(&copy_to_host_0, m::Op(&cosine_0)
                                       .WithOpcode(xla::HloOpcode::kCos)
                                       .WithOperand(0, parameter_matcher)),
          m::Copy(&copy_to_host_1, m::Op(&cosine_1)
                                       .WithOpcode(xla::HloOpcode::kCos)
                                       .WithOperand(0, first_sine_matcher)),
          m::Copy(&copy_to_host_2,
                  m::Op(&cosine_2)
                      .WithOpcode(xla::HloOpcode::kCos)
                      .WithOperand(0, m::Op(&sine_1)
                                          .WithOpcode(xla::HloOpcode::kSin)
                                          .WithOperand(0, first_sine_matcher))),
          m::Constant(&constant)));
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          &multiply_0,
          m::Multiply(
              &multiply_1,
              m::Multiply(
                  &multiply_2,
                  m::Broadcast(&broadcast, m::GetTupleElement(
                                               &gte_3, opt_barrier_matcher, 3)),
                  m::Copy(&copy_to_device_2,
                          m::GetTupleElement(&gte_2, opt_barrier_matcher, 2))),
              m::Copy(&copy_to_device_1,
                      m::GetTupleElement(&gte_1, opt_barrier_matcher, 1))),
          m::Copy(&copy_to_device_0,
                  m::GetTupleElement(&gte_0, opt_barrier_matcher, 0)))));

  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(constant->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(sine_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(sine_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(cosine_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(cosine_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(cosine_2->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host_0->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_host_1->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_host_2->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {3}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {1}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {2}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(opt_barrier->shape(), {3}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte_2->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte_3->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_device_0->shape(),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_device_1->shape(),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_device_2->shape(),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_2->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, NoCopyMultipleUsers) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[2048] parameter(0)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  sine = f32[2048] sine(data_param)
  load_custom_call = f32[2048] custom-call(offload_custom_call), custom_call_target="MoveToDevice"
  ROOT add = f32[2048] add(sine, load_custom_call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //   parameter
  //     /  \
  //  sine  copy
  //     |   |
  //     |  copy
  //     |  /
  //     add
  HloInstruction* param;
  HloInstruction* sine;
  HloInstruction* copy_to_host;
  HloInstruction* copy_to_device;
  HloInstruction* add;
  const auto param_pattern = m::Parameter(&param, 0);
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Add(
          &add, m::Sin(&sine, param_pattern),
          m::Copy(&copy_to_device, m::Copy(&copy_to_host, param_pattern)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(sine->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(add->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, BasicDusDsWithMultipleBroadcastUsers) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  tanh = f32[2,2048,2048] tanh(broadcast)
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // "AllocateBuffer"  param_0  _...
  //               |  /        /
  //           dynamic-update-slice  _...
  //                          |     /
  //                       dynamic-slice
  HloInstruction* param;
  HloInstruction* allocate_buffer;
  HloInstruction* dynamic_update_slice;
  HloInstruction* dynamic_slice;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice(
                  &dynamic_slice,
                  m::DynamicUpdateSlice(
                      &dynamic_update_slice,
                      m::CustomCall(&allocate_buffer, {"AllocateBuffer"}),
                      m::Parameter(&param, 0), m::Op(), m::Op(), m::Op()),
                  m::Op(), m::Op(), m::Op())));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_slice->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));

  // Look for the tanh and make sure that it still uses the original broadcast.
  HloInstruction* tanh = nullptr;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kTanh) {
      tanh = instruction;
      break;
    }
  }
  ASSERT_NE(tanh, nullptr);
  HloInstruction* broadcast;
  EXPECT_THAT(tanh, GmockMatch(m::Tanh(m::Broadcast(&broadcast))));
  TestShapeHasMemorySpace(broadcast->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(tanh->shape(), Layout::kDefaultMemorySpace);
}

TEST_F(HostOffloaderTest, BasicDusDsBitcastBeforeDus) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  offload_custom_call = f32[2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  bitcast = f32[1,2048,2048] bitcast(offload_custom_call)
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, bitcast, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //                   param_0
  //                     |
  // "AllocateBuffer"  bitcast  _...
  //               |  /        /
  //           dynamic-update-slice  _...
  //                          |     /
  //                       dynamic-slice
  HloInstruction* param;
  HloInstruction* bitcast;
  HloInstruction* allocate_buffer;
  HloInstruction* dynamic_update_slice;
  HloInstruction* dynamic_slice;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice(
                  &dynamic_slice,
                  m::DynamicUpdateSlice(
                      &dynamic_update_slice,
                      m::CustomCall(&allocate_buffer, {"AllocateBuffer"}),
                      m::Bitcast(&bitcast, m::Parameter(&param, 0)), m::Op(),
                      m::Op(), m::Op()),
                  m::Op(), m::Op(), m::Op())));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(bitcast->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_slice->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

// The annotation is mistakenly after the dynamic-update-slice; it should be
// before.
TEST_F(HostOffloaderTest, BasicDusDsDusAnnotationOnWrongSide) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, data_param, index_param, constant_s32_0, constant_s32_0)
  offload_custom_call = f32[1,2048,2048] custom-call(dynamic_update_slice), custom_call_target="MoveToHost"
  dynamic_slice = f32[1,2048,2048] dynamic-slice(offload_custom_call, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  absl::StatusOr<bool> statusOrChanged = RunHostOffloader(module.get());
  // The pass should return an error.
  ASSERT_FALSE(statusOrChanged.ok());
}

// The annotation is mistakenly before the dynamic-slice; it should be after.
TEST_F(HostOffloaderTest, BasicDusDsDsAnnotationOnWrongSide) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  load_custom_call = f32[2,2048,2048] custom-call(dynamic_update_slice), custom_call_target="MoveToDevice"
  ROOT dynamic_slice = f32[1,2048,2048] dynamic-slice(load_custom_call, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // "AllocateBuffer"  param_0  _...
  //               |  /        /
  //           dynamic-update-slice
  //                          |
  //                        copy     _...
  //                          |     /
  //                       dynamic-slice
  HloInstruction* param;
  HloInstruction* allocate_buffer;
  HloInstruction* dynamic_update_slice;
  HloInstruction* copy;
  HloInstruction* dynamic_slice;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::DynamicSlice(
          &dynamic_slice,
          m::Copy(&copy,
                  m::DynamicUpdateSlice(
                      &dynamic_update_slice,
                      m::CustomCall(&allocate_buffer, {"AllocateBuffer"}),
                      m::Parameter(&param, 0), m::Op(), m::Op(), m::Op())),
          m::Op(), m::Op(), m::Op())));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(allocate_buffer->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(dynamic_slice->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, LlmActivation) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.0), index=1
  data_1.0 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.0), index=2
  constant_0.0 = s32[] constant(0)
  constant_1.0 = s32[] constant(1)
  constant_96 = s32[] constant(96)

  /* Create dummy data used in DUS */
  slice_data_0 = f32[1,8,6,2048,2048]  constant({...})
  slice_data_1 = f32[1,8,6,2048,1]  constant({...})

  /* Build DUS index */
  compare_result.0 = pred[] compare(current_iteration_index.0, constant_0.0), direction=LT
  add_result = s32[] add(current_iteration_index.0, constant_96)
  select_result.0 = s32[] select(compare_result.0, add_result, current_iteration_index.0)

  /* Annotate DUS for offload */
  custom_call_0.0 = f32[1,8,6,2048,2048] custom-call(slice_data_0), custom_call_target="MoveToHost"
  custom_call_1.0 = f32[1,8,6,2048,1] custom-call(slice_data_1), custom_call_target="MoveToHost"

  dynamic_update_slice_0 = f32[96,8,6,2048,2048] dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)
  dynamic_update_slice_1 = f32[96,8,6,2048,1] dynamic-update-slice(data_1.0, custom_call_1.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.0, dynamic_update_slice_0, dynamic_update_slice_1)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.1), index=1
  data_1.1 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.1), index=2
  constant_0.1 = s32[] constant(0)
  constant_1.1 = s32[] constant(1)
  constant_95 = s32[] constant(95)
  constant_191 = s32[] constant(191)

  /* Build DS index */
  subtract_0 = s32[] subtract(constant_95, current_iteration_index.1)
  compare_result.1 = pred[] compare(subtract_0, constant_0.1), direction=LT
  subtract_1 = s32[] subtract(constant_191, current_iteration_index.1)
  select_result.1 = s32[] select(compare_result.1, subtract_1, subtract_0)

  dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(data_0.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,2048}
  dynamic_slice_1 = f32[1,8,6,2048,1] dynamic-slice(data_1.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,1}

  /* Annotate DS for offload */
  custom_call_0.1 = f32[1,8,6,2048,2048] custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
  custom_call_1.1 = f32[1,8,6,2048,1] custom-call(dynamic_slice_1), custom_call_target="MoveToDevice"

  /* Do some work with the dynamic slice outputs. */
  tanh_0 = f32[1,8,6,2048,2048] tanh(custom_call_0.1)
  tanh_1 = f32[1,8,6,2048,1] tanh(custom_call_1.1)

  /* Increment iteration index */
  incremented_index.1 = s32[] add(current_iteration_index.1, constant_1.1)
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.1, data_0.1, data_1.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  broadcast_0 = f32[96,8,6,2048,2048] broadcast(entry_param_0), dimensions={}
  broadcast_1 = f32[96,8,6,2048,1] broadcast(entry_param_0), dimensions={}
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, broadcast_0, broadcast_1)
  producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048] get-tuple-element(producing_while), index=1
  while_output_2 = f32[96,8,6,2048,1] get-tuple-element(producing_while), index=2
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, while_output_1, while_output_2)
  consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  ROOT result = s32[] get-tuple-element(consuming_while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // First, look for the pattern:
  //  producing_while
  //       /  \
  //     gte  gte  constant
  //       \  /   /
  //        \/   /
  //        tuple
  //         |
  //  consuming_while
  //         |
  //        gte
  HloInstruction* consuming_while;
  HloInstruction* producing_while_0;
  HloInstruction* producing_while_1;
  {
    HloInstruction* tuple;
    HloInstruction* gte_0;
    HloInstruction* gte_1;
    HloInstruction* gte_2;
    ASSERT_THAT(
        module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            &gte_2,
            m::While(
                &consuming_while,
                m::Tuple(
                    &tuple, m::Constant(),
                    m::GetTupleElement(&gte_0, m::While(&producing_while_0)),
                    m::GetTupleElement(&gte_1, m::While(&producing_while_1)))),
            0)));
    ASSERT_EQ(producing_while_0, producing_while_1);

    // Check that the memory spaces were properly set.
    TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {2}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while_0->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while_0->shape(), {2}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                            Layout::kHostMemorySpace);
  }

  // Now, look for the AllocateBuffers leading into the producing while.
  {
    HloInstruction* allocate_buffer_0;
    HloInstruction* allocate_buffer_1;
    ASSERT_THAT(producing_while_0,
                GmockMatch(m::While(m::Tuple(
                    m::Constant(),
                    m::CustomCall(&allocate_buffer_0, {"AllocateBuffer"}),
                    m::CustomCall(&allocate_buffer_1, {"AllocateBuffer"})))));
    // Check that the memory spaces were properly set.
    ASSERT_TRUE(allocate_buffer_0->shape().has_layout());
    EXPECT_EQ(allocate_buffer_0->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
    ASSERT_TRUE(allocate_buffer_1->shape().has_layout());
    EXPECT_EQ(allocate_buffer_1->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
  }

  // There are 4 computations to look at:
  //  - Consuming while's body
  //  - Consuming while's condition
  //  - Producing while's body
  //  - Producing while's condition

  // For the condition computations, just check that the parameters have the
  // right memory space.
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {1}),
      Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {2}),
      Layout::kHostMemorySpace);

  // Now, check the producing while body for the following pattern:
  //    param      param
  //      |          |
  //     gte  _...  gte  _...
  //     |   /      |   /
  //     |  /       |  /
  //     | /        | /
  //     dus       dus
  //      |       /
  //      |      /
  //  _   |     /
  //   \  |    /
  //    \ |   /
  //     \|  /
  //    tuple
  {
    HloInstruction* tuple;
    HloInstruction* dynamic_update_slice_0;
    HloInstruction* dynamic_update_slice_1;
    HloInstruction* dynamic_update_slice_second_param_0;
    HloInstruction* dynamic_update_slice_second_param_1;
    HloInstruction* gte_0;
    HloInstruction* gte_1;
    HloInstruction* param_0;
    HloInstruction* param_1;
    ASSERT_THAT(producing_while_0->while_body()->root_instruction(),
                GmockMatch(m::Tuple(
                    &tuple, m::Op(),
                    m::DynamicUpdateSlice(
                        &dynamic_update_slice_0,
                        m::GetTupleElement(&gte_0, m::Parameter(&param_0)),
                        m::Op(&dynamic_update_slice_second_param_0), m::Op(),
                        m::Op(), m::Op(), m::Op(), m::Op()),
                    m::DynamicUpdateSlice(
                        &dynamic_update_slice_1,
                        m::GetTupleElement(&gte_1, m::Parameter(&param_1)),
                        m::Op(&dynamic_update_slice_second_param_1), m::Op(),
                        m::Op(), m::Op(), m::Op(), m::Op()))));
    EXPECT_EQ(param_0, param_1);

    // Check that the memory spaces were properly set.
    // HOST:
    //  tuple subshape 1
    //  tuple subshape 2
    //  dynamic_update_slice_0 shape
    //  dynamic_update_slice_1 shape
    //  gte_0 shape
    //  gte_1 shape
    //  param_0 subshape 1
    //  param_0 subshape 2
    // DEVICE:
    //  dynamic_update_slice_second_param_0
    //  dynamic_update_slice_second_param_1

    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_0->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_1->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param_0->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param_0->shape(), {2}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param_0->shape(),
                            Layout::kDefaultMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param_1->shape(),
                            Layout::kDefaultMemorySpace);
  }

  // Now, check the consuming while body for the following pattern:
  //  param
  //  |   |
  // gte gte
  //  |   |
  //  ds  ds
  {
    const absl::flat_hash_set<const HloInstruction*> dynamic_slices =
        getInstructionsWithOpcodeFromComputation(consuming_while->while_body(),
                                                 HloOpcode::kDynamicSlice);
    // There should only be two dynamic-slices.
    ASSERT_EQ(dynamic_slices.size(), 2);
    for (const HloInstruction* dynamic_slice : dynamic_slices) {
      const HloInstruction* get_tuple_element;
      const HloInstruction* parameter;
      ASSERT_THAT(
          dynamic_slice,
          GmockMatch(m::DynamicSlice(
              m::GetTupleElement(&get_tuple_element, m::Parameter(&parameter)),
              m::Op(), m::Op(), m::Op(), m::Op(), m::Op())));

      // Check that the memory spaces were properly set.
      // HOST:
      //  parameter subshape 1
      //  parameter subshape 2
      //  get_tuple_element
      // DEVICE:
      //  dynamic_slice
      TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {1}),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {2}),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(get_tuple_element->shape(),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(dynamic_slice->shape(),
                              Layout::kDefaultMemorySpace);
    }
  }

  // Finally, ensure that all annotations have been removed.
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, LlmActivationSourceIsAllocateBuffer) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.0), index=1
  data_1.0 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.0), index=2
  constant_0.0 = s32[] constant(0)
  constant_1.0 = s32[] constant(1)
  constant_96 = s32[] constant(96)

  /* Create dummy data used in DUS */
  slice_data_0 = f32[1,8,6,2048,2048]  constant({...})
  slice_data_1 = f32[1,8,6,2048,1]  constant({...})

  /* Build DUS index */
  compare_result.0 = pred[] compare(current_iteration_index.0, constant_0.0), direction=LT
  add_result = s32[] add(current_iteration_index.0, constant_96)
  select_result.0 = s32[] select(compare_result.0, add_result, current_iteration_index.0)

  /* Annotate DUS for offload */
  custom_call_0.0 = f32[1,8,6,2048,2048] custom-call(slice_data_0), custom_call_target="MoveToHost"
  custom_call_1.0 = f32[1,8,6,2048,1] custom-call(slice_data_1), custom_call_target="MoveToHost"

  dynamic_update_slice_0 = f32[96,8,6,2048,2048] dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)
  dynamic_update_slice_1 = f32[96,8,6,2048,1] dynamic-update-slice(data_1.0, custom_call_1.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.0, dynamic_update_slice_0, dynamic_update_slice_1)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.1), index=1
  data_1.1 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.1), index=2
  constant_0.1 = s32[] constant(0)
  constant_1.1 = s32[] constant(1)
  constant_95 = s32[] constant(95)
  constant_191 = s32[] constant(191)

  /* Build DS index */
  subtract_0 = s32[] subtract(constant_95, current_iteration_index.1)
  compare_result.1 = pred[] compare(subtract_0, constant_0.1), direction=LT
  subtract_1 = s32[] subtract(constant_191, current_iteration_index.1)
  select_result.1 = s32[] select(compare_result.1, subtract_1, subtract_0)

  dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(data_0.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,2048}
  dynamic_slice_1 = f32[1,8,6,2048,1] dynamic-slice(data_1.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,1}

  /* Annotate DS for offload */
  custom_call_0.1 = f32[1,8,6,2048,2048] custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
  custom_call_1.1 = f32[1,8,6,2048,1] custom-call(dynamic_slice_1), custom_call_target="MoveToDevice"

  /* Do some work with the dynamic slice outputs. */
  tanh_0 = f32[1,8,6,2048,2048] tanh(custom_call_0.1)
  tanh_1 = f32[1,8,6,2048,1] tanh(custom_call_1.1)

  /* Increment iteration index */
  incremented_index.1 = s32[] add(current_iteration_index.1, constant_1.1)
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.1, data_0.1, data_1.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  allocate_buffer_0 = f32[96,8,6,2048,2048] custom-call(), custom_call_target="AllocateBuffer"
  allocate_buffer_1 = f32[96,8,6,2048,1] custom-call(), custom_call_target="AllocateBuffer"
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, allocate_buffer_0, allocate_buffer_1)
  producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048] get-tuple-element(producing_while), index=1
  while_output_2 = f32[96,8,6,2048,1] get-tuple-element(producing_while), index=2
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, while_output_1, while_output_2)
  consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  ROOT result = s32[] get-tuple-element(consuming_while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // First, look for the pattern:
  //  producing_while
  //       /  \
  //     gte  gte  constant
  //       \  /   /
  //        \/   /
  //        tuple
  //         |
  //  consuming_while
  //         |
  //        gte
  HloInstruction* consuming_while;
  HloInstruction* producing_while_0;
  HloInstruction* producing_while_1;
  {
    HloInstruction* tuple;
    HloInstruction* gte_0;
    HloInstruction* gte_1;
    HloInstruction* gte_2;
    ASSERT_THAT(
        module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            &gte_2,
            m::While(
                &consuming_while,
                m::Tuple(
                    &tuple, m::Constant(),
                    m::GetTupleElement(&gte_0, m::While(&producing_while_0)),
                    m::GetTupleElement(&gte_1, m::While(&producing_while_1)))),
            0)));
    ASSERT_EQ(producing_while_0, producing_while_1);

    // Check that the memory spaces were properly set.
    TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {2}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while_0->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while_0->shape(), {2}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                            Layout::kHostMemorySpace);
  }

  // Now, look for the AllocateBuffers leading into the producing while.
  {
    HloInstruction* allocate_buffer_0;
    HloInstruction* allocate_buffer_1;
    ASSERT_THAT(producing_while_0,
                GmockMatch(m::While(m::Tuple(
                    m::Constant(),
                    m::CustomCall(&allocate_buffer_0, {"AllocateBuffer"}),
                    m::CustomCall(&allocate_buffer_1, {"AllocateBuffer"})))));
    // Check that the memory spaces were properly set.
    ASSERT_TRUE(allocate_buffer_0->shape().has_layout());
    EXPECT_EQ(allocate_buffer_0->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
    ASSERT_TRUE(allocate_buffer_1->shape().has_layout());
    EXPECT_EQ(allocate_buffer_1->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
  }

  // There are 4 computations to look at:
  //  - Consuming while's body
  //  - Consuming while's condition
  //  - Producing while's body
  //  - Producing while's condition

  // For the condition computations, just check that the parameters have the
  // right memory space.
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {1}),
      Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {2}),
      Layout::kHostMemorySpace);

  // Now, check the producing while body for the following pattern:
  //    param      param
  //      |          |
  //     gte  _...  gte  _...
  //     |   /      |   /
  //     |  /       |  /
  //     | /        | /
  //     dus       dus
  //      |       /
  //      |      /
  //  _   |     /
  //   \  |    /
  //    \ |   /
  //     \|  /
  //    tuple
  {
    HloInstruction* tuple;
    HloInstruction* dynamic_update_slice_0;
    HloInstruction* dynamic_update_slice_1;
    HloInstruction* dynamic_update_slice_second_param_0;
    HloInstruction* dynamic_update_slice_second_param_1;
    HloInstruction* gte_0;
    HloInstruction* gte_1;
    HloInstruction* param_0;
    HloInstruction* param_1;
    ASSERT_THAT(producing_while_0->while_body()->root_instruction(),
                GmockMatch(m::Tuple(
                    &tuple, m::Op(),
                    m::DynamicUpdateSlice(
                        &dynamic_update_slice_0,
                        m::GetTupleElement(&gte_0, m::Parameter(&param_0)),
                        m::Op(&dynamic_update_slice_second_param_0), m::Op(),
                        m::Op(), m::Op(), m::Op(), m::Op()),
                    m::DynamicUpdateSlice(
                        &dynamic_update_slice_1,
                        m::GetTupleElement(&gte_1, m::Parameter(&param_1)),
                        m::Op(&dynamic_update_slice_second_param_1), m::Op(),
                        m::Op(), m::Op(), m::Op(), m::Op()))));
    EXPECT_EQ(param_0, param_1);

    // Check that the memory spaces were properly set.
    // HOST:
    //  tuple subshape 1
    //  tuple subshape 2
    //  dynamic_update_slice_0 shape
    //  dynamic_update_slice_1 shape
    //  gte_0 shape
    //  gte_1 shape
    //  param_0 subshape 1
    //  param_0 subshape 2
    // DEVICE:
    //  dynamic_update_slice_second_param_0
    //  dynamic_update_slice_second_param_1

    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_0->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_1->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param_0->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param_0->shape(), {2}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param_0->shape(),
                            Layout::kDefaultMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param_1->shape(),
                            Layout::kDefaultMemorySpace);
  }

  // Now, check the consuming while body for the following pattern:
  //  param
  //  |   |
  // gte gte
  //  |   |
  //  ds  ds
  {
    const absl::flat_hash_set<const HloInstruction*> dynamic_slices =
        getInstructionsWithOpcodeFromComputation(consuming_while->while_body(),
                                                 HloOpcode::kDynamicSlice);
    // There should only be two dynamic-slices.
    ASSERT_EQ(dynamic_slices.size(), 2);
    for (const HloInstruction* dynamic_slice : dynamic_slices) {
      const HloInstruction* get_tuple_element;
      const HloInstruction* parameter;
      ASSERT_THAT(
          dynamic_slice,
          GmockMatch(m::DynamicSlice(
              m::GetTupleElement(&get_tuple_element, m::Parameter(&parameter)),
              m::Op(), m::Op(), m::Op(), m::Op(), m::Op())));

      // Check that the memory spaces were properly set.
      // HOST:
      //  parameter subshape 1
      //  parameter subshape 2
      //  get_tuple_element
      // DEVICE:
      //  dynamic_slice
      TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {1}),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {2}),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(get_tuple_element->shape(),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(dynamic_slice->shape(),
                              Layout::kDefaultMemorySpace);
    }
  }

  // Finally, ensure that all annotations have been removed.
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, LlmActivationDsWithReshape) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.0), index=1
  data_1.0 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.0), index=2
  constant_0.0 = s32[] constant(0)
  constant_1.0 = s32[] constant(1)
  constant_96 = s32[] constant(96)

  /* Create dummy data used in DUS */
  slice_data_0 = f32[1,8,6,2048,2048]  constant({...})
  slice_data_1 = f32[1,8,6,2048,1]  constant({...})

  /* Build DUS index */
  compare_result.0 = pred[] compare(current_iteration_index.0, constant_0.0), direction=LT
  add_result = s32[] add(current_iteration_index.0, constant_96)
  select_result.0 = s32[] select(compare_result.0, add_result, current_iteration_index.0)

  /* Annotate DUS for offload */
  custom_call_0.0 = f32[1,8,6,2048,2048] custom-call(slice_data_0), custom_call_target="MoveToHost"
  custom_call_1.0 = f32[1,8,6,2048,1] custom-call(slice_data_1), custom_call_target="MoveToHost"

  dynamic_update_slice_0 = f32[96,8,6,2048,2048] dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)
  dynamic_update_slice_1 = f32[96,8,6,2048,1] dynamic-update-slice(data_1.0, custom_call_1.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.0, dynamic_update_slice_0, dynamic_update_slice_1)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.1), index=1
  data_1.1 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.1), index=2
  constant_0.1 = s32[] constant(0)
  constant_1.1 = s32[] constant(1)
  constant_95 = s32[] constant(95)
  constant_191 = s32[] constant(191)

  /* Build DS index */
  subtract_0 = s32[] subtract(constant_95, current_iteration_index.1)
  compare_result.1 = pred[] compare(subtract_0, constant_0.1), direction=LT
  subtract_1 = s32[] subtract(constant_191, current_iteration_index.1)
  select_result.1 = s32[] select(compare_result.1, subtract_1, subtract_0)

  dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(data_0.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,2048}
  dynamic_slice_1 = f32[1,8,6,2048,1] dynamic-slice(data_1.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,1}
  rs = f32[1,8,6,2048,2048] reshape(dynamic_slice_0)
  rs2 = f32[1,8,6,2048,1] reshape(dynamic_slice_1)
  /* Annotate DS for offload */
  custom_call_0.1 = f32[1,8,6,2048,2048] custom-call(rs), custom_call_target="MoveToDevice"
  custom_call_1.1 = f32[1,8,6,2048,1] custom-call(rs2), custom_call_target="MoveToDevice"

  /* Do some work with the dynamic slice outputs. */
  tanh_0 = f32[1,8,6,2048,2048] tanh(custom_call_0.1)
  tanh_1 = f32[1,8,6,2048,1] tanh(custom_call_1.1)

  /* Increment iteration index */
  incremented_index.1 = s32[] add(current_iteration_index.1, constant_1.1)
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.1, data_0.1, data_1.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  broadcast_0 = f32[96,8,6,2048,2048] broadcast(entry_param_0), dimensions={}
  broadcast_1 = f32[96,8,6,2048,1] broadcast(entry_param_0), dimensions={}
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, broadcast_0, broadcast_1)
  producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048] get-tuple-element(producing_while), index=1
  while_output_2 = f32[96,8,6,2048,1] get-tuple-element(producing_while), index=2
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, while_output_1, while_output_2)
  consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  ROOT result = s32[] get-tuple-element(consuming_while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // First, look for the pattern:
  //  producing_while
  //       /  \
  //     gte  gte  constant
  //       \  /   /
  //        \/   /
  //        tuple
  //         |
  //  consuming_while
  //         |
  //        gte
  HloInstruction* consuming_while;
  HloInstruction* producing_while_0;
  HloInstruction* producing_while_1;
  {
    HloInstruction* tuple;
    HloInstruction* gte_0;
    HloInstruction* gte_1;
    HloInstruction* gte_2;
    ASSERT_THAT(
        module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            &gte_2,
            m::While(
                &consuming_while,
                m::Tuple(
                    &tuple, m::Constant(),
                    m::GetTupleElement(&gte_0, m::While(&producing_while_0)),
                    m::GetTupleElement(&gte_1, m::While(&producing_while_1)))),
            0)));
    ASSERT_EQ(producing_while_0, producing_while_1);

    // Check that the memory spaces were properly set.
    TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {2}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while_0->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while_0->shape(), {2}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                            Layout::kHostMemorySpace);
  }

  // Now, look for the AllocateBuffers leading into the producing while.
  {
    HloInstruction* allocate_buffer_0;
    HloInstruction* allocate_buffer_1;
    ASSERT_THAT(producing_while_0,
                GmockMatch(m::While(m::Tuple(
                    m::Constant(),
                    m::CustomCall(&allocate_buffer_0, {"AllocateBuffer"}),
                    m::CustomCall(&allocate_buffer_1, {"AllocateBuffer"})))));
    // Check that the memory spaces were properly set.
    ASSERT_TRUE(allocate_buffer_0->shape().has_layout());
    EXPECT_EQ(allocate_buffer_0->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
    ASSERT_TRUE(allocate_buffer_1->shape().has_layout());
    EXPECT_EQ(allocate_buffer_1->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
  }

  // There are 4 computations to look at:
  //  - Consuming while's body
  //  - Consuming while's condition
  //  - Producing while's body
  //  - Producing while's condition

  // For the condition computations, just check that the parameters have the
  // right memory space.
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {1}),
      Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {2}),
      Layout::kHostMemorySpace);

  // Now, check the producing while body for the following pattern:
  //    param      param
  //      |          |
  //     gte  _...  gte  _...
  //     |   /      |   /
  //     |  /       |  /
  //     | /        | /
  //     dus       dus
  //      |       /
  //      |      /
  //  _   |     /
  //   \  |    /
  //    \ |   /
  //     \|  /
  //    tuple
  {
    HloInstruction* tuple;
    HloInstruction* dynamic_update_slice_0;
    HloInstruction* dynamic_update_slice_1;
    HloInstruction* dynamic_update_slice_second_param_0;
    HloInstruction* dynamic_update_slice_second_param_1;
    HloInstruction* gte_0;
    HloInstruction* gte_1;
    HloInstruction* param_0;
    HloInstruction* param_1;
    ASSERT_THAT(producing_while_0->while_body()->root_instruction(),
                GmockMatch(m::Tuple(
                    &tuple, m::Op(),
                    m::DynamicUpdateSlice(
                        &dynamic_update_slice_0,
                        m::GetTupleElement(&gte_0, m::Parameter(&param_0)),
                        m::Op(&dynamic_update_slice_second_param_0), m::Op(),
                        m::Op(), m::Op(), m::Op(), m::Op()),
                    m::DynamicUpdateSlice(
                        &dynamic_update_slice_1,
                        m::GetTupleElement(&gte_1, m::Parameter(&param_1)),
                        m::Op(&dynamic_update_slice_second_param_1), m::Op(),
                        m::Op(), m::Op(), m::Op(), m::Op()))));
    EXPECT_EQ(param_0, param_1);

    // Check that the memory spaces were properly set.
    // HOST:
    //  tuple subshape 1
    //  tuple subshape 2
    //  dynamic_update_slice_0 shape
    //  dynamic_update_slice_1 shape
    //  gte_0 shape
    //  gte_1 shape
    //  param_0 subshape 1
    //  param_0 subshape 2
    // DEVICE:
    //  dynamic_update_slice_second_param_0
    //  dynamic_update_slice_second_param_1

    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {2}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_0->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_1->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param_0->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param_0->shape(), {2}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param_0->shape(),
                            Layout::kDefaultMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param_1->shape(),
                            Layout::kDefaultMemorySpace);
  }

  // Now, check the consuming while body for the following pattern:
  //  param
  //  |   |
  // gte gte
  //  |   |
  //  ds  ds
  {
    const absl::flat_hash_set<const HloInstruction*> dynamic_slices =
        getInstructionsWithOpcodeFromComputation(consuming_while->while_body(),
                                                 HloOpcode::kDynamicSlice);
    // There should only be two dynamic-slices.
    ASSERT_EQ(dynamic_slices.size(), 2);
    for (const HloInstruction* dynamic_slice : dynamic_slices) {
      const HloInstruction* get_tuple_element;
      const HloInstruction* parameter;
      ASSERT_THAT(
          dynamic_slice,
          GmockMatch(m::DynamicSlice(
              m::GetTupleElement(&get_tuple_element, m::Parameter(&parameter)),
              m::Op(), m::Op(), m::Op(), m::Op(), m::Op())));

      // Check that the memory spaces were properly set.
      // HOST:
      //  parameter subshape 1
      //  parameter subshape 2
      //  get_tuple_element
      // DEVICE:
      //  dynamic_slice
      TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {1}),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {2}),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(get_tuple_element->shape(),
                              Layout::kHostMemorySpace);
      TestShapeHasMemorySpace(dynamic_slice->shape(),
                              Layout::kDefaultMemorySpace);
    }
  }

  // Finally, ensure that all annotations have been removed.
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, LlmActivationHostMemoryMultipleConsumers) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048]) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048]) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048]) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.0), index=1
  constant_0.0 = s32[] constant(0)
  constant_1.0 = s32[] constant(1)
  constant_96 = s32[] constant(96)

  /* Create dummy data used in DUS */
  slice_data_0 = f32[1,8,6,2048,2048]  constant({...})

  /* Build DUS index */
  compare_result.0 = pred[] compare(current_iteration_index.0, constant_0.0), direction=LT
  add_result = s32[] add(current_iteration_index.0, constant_96)
  select_result.0 = s32[] select(compare_result.0, add_result, current_iteration_index.0)

  /* Annotate DUS for offload */
  custom_call_0.0 = f32[1,8,6,2048,2048] custom-call(slice_data_0), custom_call_target="MoveToHost"

  dynamic_update_slice_0 = f32[96,8,6,2048,2048] dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048]) tuple(incremented_index.0, dynamic_update_slice_0)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048]) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.1), index=1
  constant_0.1 = s32[] constant(0)
  constant_1.1 = s32[] constant(1)
  constant_95 = s32[] constant(95)
  constant_191 = s32[] constant(191)

  /* Build DS index */
  subtract_0 = s32[] subtract(constant_95, current_iteration_index.1)
  compare_result.1 = pred[] compare(subtract_0, constant_0.1), direction=LT
  subtract_1 = s32[] subtract(constant_191, current_iteration_index.1)
  select_result.1 = s32[] select(compare_result.1, subtract_1, subtract_0)

  dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(data_0.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,2048}

  /* Annotate DS for offload */
  custom_call_0.1 = f32[1,8,6,2048,2048] custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"

  /* Do some work with the dynamic slice outputs. */
  tanh_0 = f32[1,8,6,2048,2048] tanh(custom_call_0.1)

  /* Increment iteration index */
  incremented_index.1 = s32[] add(current_iteration_index.1, constant_1.1)
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048]) tuple(incremented_index.1, data_0.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  entry_param_1 = s32[] parameter(1)
  entry_param_2 = s32[] parameter(2)
  broadcast_0 = f32[96,8,6,2048,2048] broadcast(entry_param_0), dimensions={}
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048]) tuple(constant_s32_0, broadcast_0)
  producing_while = (s32[], f32[96,8,6,2048,2048]) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048] get-tuple-element(producing_while), index=1
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048]) tuple(constant_s32_0, while_output_1)
  consuming_while = (s32[], f32[96,8,6,2048,2048]) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  second_while_output = f32[96,8,6,2048,2048] get-tuple-element(consuming_while), index=1
  final_dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(second_while_output, entry_param_1, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,8,6,2048,2048}
  final_host_to_device_custom_call_0 = f32[1,8,6,2048,2048] custom-call(final_dynamic_slice_0), custom_call_target="MoveToDevice"
  final_slice_0 = f32[1,8,6,2048,2048] slice(second_while_output), slice={[41:42], [0:8], [0:6], [0:2048], [0:2048]}
  final_host_to_device_custom_call_1 = f32[1,8,6,2048,2048] custom-call(final_slice_0), custom_call_target="MoveToDevice"
  ROOT add = f32[1,8,6,2048,2048] add(final_host_to_device_custom_call_0, final_host_to_device_custom_call_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // First, look for the pattern:
  //           producing_while
  //                 |
  //      constant  gte
  //           \     |
  //            \    |
  //             tuple
  //               |
  //       consuming_while
  //               |
  //              gte
  //             /   \
  // dynamic-slice   dynamic-slice
  //             \   /
  //              add
  // Note: The second dynamic-slice was originally a slice.
  HloInstruction* consuming_while;
  HloInstruction* producing_while;
  {
    HloInstruction* tuple;
    HloInstruction* gte_between_whiles;
    HloInstruction* final_gte;
    HloInstruction* dynamic_slice_0;
    HloInstruction* dynalic_slice_1;
    HloInstruction* add;
    auto pattern_ending_in_gte = m::GetTupleElement(
        &final_gte,
        m::While(&consuming_while,
                 m::Tuple(&tuple, m::Constant(),
                          m::GetTupleElement(&gte_between_whiles,
                                             m::While(&producing_while)))));
    ASSERT_THAT(
        module->entry_computation()->root_instruction(),
        GmockMatch(
            m::Add(&add,
                   m::DynamicSlice(&dynamic_slice_0, pattern_ending_in_gte,
                                   m::Op(), m::Op(), m::Op(), m::Op(), m::Op()),
                   m::DynamicSlice(&dynalic_slice_1, pattern_ending_in_gte,
                                   m::ConstantScalar(41), m::Op(), m::Op(),
                                   m::Op(), m::Op()))));

    // Check that the memory spaces were properly set.
    TestShapeHasMemorySpace(gte_between_whiles->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(consuming_while->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(
        ShapeUtil::GetSubshape(producing_while->shape(), {1}),
        Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(final_gte->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_slice_0->shape(),
                            Layout::kDefaultMemorySpace);
    TestShapeHasMemorySpace(dynalic_slice_1->shape(),
                            Layout::kDefaultMemorySpace);
    TestShapeHasMemorySpace(add->shape(), Layout::kDefaultMemorySpace);
  }

  // Now, look for the AllocateBuffers leading into the producing while.
  {
    HloInstruction* allocate_buffer;
    ASSERT_THAT(producing_while,
                GmockMatch(m::While(m::Tuple(
                    m::Constant(),
                    m::CustomCall(&allocate_buffer, {"AllocateBuffer"})))));
    // Check that the memory spaces were properly set.
    ASSERT_TRUE(allocate_buffer->shape().has_layout());
    EXPECT_EQ(allocate_buffer->shape().layout().memory_space(),
              Layout::kHostMemorySpace);
  }

  // There are 4 computations to look at:
  //  - Consuming while's body
  //  - Consuming while's condition
  //  - Producing while's body
  //  - Producing while's condition

  // For the condition computations, just check that the parameters have the
  // right memory space.
  TestShapeHasMemorySpace(
      ShapeUtil::GetSubshape(
          consuming_while->while_condition()->parameter_instruction(0)->shape(),
          {1}),
      Layout::kHostMemorySpace);

  // Now, check the producing while body for the following pattern:
  //    param
  //      |
  //     gte  _
  //      |  /
  //      | /
  //  _  dus
  //   \  |
  //   tuple
  {
    HloInstruction* tuple;
    HloInstruction* dynamic_update_slice;
    HloInstruction* dynamic_update_slice_second_param;
    HloInstruction* gte;
    HloInstruction* param;
    ASSERT_THAT(
        producing_while->while_body()->root_instruction(),
        GmockMatch(m::Tuple(&tuple, m::Op(),
                            m::DynamicUpdateSlice(
                                &dynamic_update_slice,
                                m::GetTupleElement(&gte, m::Parameter(&param)),
                                m::Op(&dynamic_update_slice_second_param),
                                m::Op(), m::Op(), m::Op(), m::Op(), m::Op()))));

    // Check that the memory spaces were properly set.
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(param->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_update_slice_second_param->shape(),
                            Layout::kDefaultMemorySpace);
  }

  // Now, check the consuming while body for the following pattern:
  //  param
  //    |
  //   gte
  //    |
  //    ds
  {
    const absl::flat_hash_set<const HloInstruction*> dynamic_slices =
        getInstructionsWithOpcodeFromComputation(consuming_while->while_body(),
                                                 HloOpcode::kDynamicSlice);
    // There should only be one dynamic-slice.
    ASSERT_EQ(dynamic_slices.size(), 1);
    const HloInstruction* dynamic_slice = *dynamic_slices.begin();
    const HloInstruction* get_tuple_element;
    const HloInstruction* parameter;
    ASSERT_THAT(
        dynamic_slice,
        GmockMatch(m::DynamicSlice(
            m::GetTupleElement(&get_tuple_element, m::Parameter(&parameter)),
            m::Op(), m::Op(), m::Op(), m::Op(), m::Op())));

    // Check that the memory spaces were properly set.
    TestShapeHasMemorySpace(ShapeUtil::GetSubshape(parameter->shape(), {1}),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(get_tuple_element->shape(),
                            Layout::kHostMemorySpace);
    TestShapeHasMemorySpace(dynamic_slice->shape(),
                            Layout::kDefaultMemorySpace);
  }

  // Finally, ensure that all annotations have been removed.
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, InsertExtraCopyForScheduling) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1], f32[1,8,6,2048,1]) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1], f32[1,8,6,2048,1]) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.0), index=1
  data_1.0 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.0), index=2
  data_2.1 = f32[1,8,6,2048,1] get-tuple-element(input_tuple.0), index=3
  constant_0.0 = s32[] constant(0)
  constant_1.0 = s32[] constant(1)
  constant_96 = s32[] constant(96)

  /* Create dummy data used in DUS */
  slice_data_0 = f32[1,8,6,2048,2048]  constant({...})
  slice_data_1 = f32[1,8,6,2048,1]  constant({...})

  /* Build DUS index */
  compare_result.0 = pred[] compare(current_iteration_index.0, constant_0.0), direction=LT
  add_result = s32[] add(current_iteration_index.0, constant_96)
  select_result.0 = s32[] select(compare_result.0, add_result, current_iteration_index.0)

  /* Annotate DUS for offload */
  custom_call_0.0 = f32[1,8,6,2048,2048] custom-call(slice_data_0), custom_call_target="MoveToHost"
  custom_call_1.0 = f32[1,8,6,2048,1] custom-call(data_2.1), custom_call_target="MoveToHost"

  dynamic_update_slice_0 = f32[96,8,6,2048,2048] dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)
  dynamic_update_slice_1 = f32[96,8,6,2048,1] dynamic-update-slice(data_1.0, custom_call_1.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1], f32[1,8,6,2048,1]) tuple(incremented_index.0, dynamic_update_slice_0, dynamic_update_slice_1, data_2.1)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048] get-tuple-element(input_tuple.1), index=1
  data_1.1 = f32[96,8,6,2048,1] get-tuple-element(input_tuple.1), index=2
  constant_0.1 = s32[] constant(0)
  constant_1.1 = s32[] constant(1)
  constant_95 = s32[] constant(95)
  constant_191 = s32[] constant(191)

  /* Build DS index */
  subtract_0 = s32[] subtract(constant_95, current_iteration_index.1)
  compare_result.1 = pred[] compare(subtract_0, constant_0.1), direction=LT
  subtract_1 = s32[] subtract(constant_191, current_iteration_index.1)
  select_result.1 = s32[] select(compare_result.1, subtract_1, subtract_0)

  dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(data_0.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,2048}
  dynamic_slice_1 = f32[1,8,6,2048,1] dynamic-slice(data_1.1, select_result.1, constant_0.1, constant_0.1, constant_0.1, constant_0.1), dynamic_slice_sizes={1,8,6,2048,1}

  /* Annotate DS for offload */
  custom_call_0.1 = f32[1,8,6,2048,2048] custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
  custom_call_1.1 = f32[1,8,6,2048,1] custom-call(dynamic_slice_1), custom_call_target="MoveToDevice"

  /* Do some work with the dynamic slice outputs. */
  tanh_0 = f32[1,8,6,2048,2048] tanh(custom_call_0.1)
  tanh_1 = f32[1,8,6,2048,1] tanh(custom_call_1.1)

  /* Increment iteration index */
  incremented_index.1 = s32[] add(current_iteration_index.1, constant_1.1)
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(incremented_index.1, data_0.1, data_1.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  broadcast_0 = f32[96,8,6,2048,2048] broadcast(entry_param_0), dimensions={}
  broadcast_1 = f32[96,8,6,2048,1] broadcast(entry_param_0), dimensions={}
  broadcast_2 = f32[1,8,6,2048,1] broadcast(entry_param_0), dimensions={}
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1], f32[1,8,6,2048,1]) tuple(constant_s32_0, broadcast_0, broadcast_1, broadcast_2)
  producing_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1], f32[1,8,6,2048,1]) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048] get-tuple-element(producing_while), index=1
  while_output_2 = f32[96,8,6,2048,1] get-tuple-element(producing_while), index=2
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) tuple(constant_s32_0, while_output_1, while_output_2)
  consuming_while = (s32[], f32[96,8,6,2048,2048], f32[96,8,6,2048,1]) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  ROOT result = s32[] get-tuple-element(consuming_while), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  // Finally, ensure that all annotations have been removed.
  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
  const HloInstruction* dus0 =
      FindInstruction(module.get(), "dynamic_update_slice_0");
  const HloInstruction* dus1 =
      FindInstruction(module.get(), "dynamic_update_slice_1");
  EXPECT_THAT(dus0, GmockMatch(m::DynamicUpdateSlice(m::Op(), m::Constant(),
                                                     m::Op(), m::Op(), m::Op(),
                                                     m::Op(), m::Op())));
  EXPECT_THAT(dus1, GmockMatch(m::DynamicUpdateSlice(m::Op(), m::Copy(),
                                                     m::Op(), m::Op(), m::Op(),
                                                     m::Op(), m::Op())));
}

TEST_F(HostOffloaderTest, ParameterStreaming) {
  const std::string& hlo_string = R"(
HloModule ParameterStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)S(5)}, s32[2,1]{1,0:T(2,128)})->(s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)})}

ENTRY main {
  param_0 = s32[2,1]{1,0} parameter(0)
  param_1 = s32[2,1]{1,0} parameter(1)
  constant_2 = s32[] constant(2)
  constant_4 = s32[] constant(4)
  broadcast_0 = s32[2,1]{1,0} broadcast(constant_2), dimensions={}
  multiply_0 = s32[2,1]{1,0} multiply(param_1, broadcast_0)
  custom_call = s32[2,1]{1,0} custom-call(param_0), custom_call_target="MoveToDevice"
  multiply_1 = s32[2,1]{1,0} multiply(multiply_0, custom_call)
  broadcast_1 = s32[2,1]{1,0} broadcast(constant_4), dimensions={}
  multiply_2 = s32[2,1]{1,0} multiply(multiply_1, broadcast_1)
  ROOT tuple = (s32[2,1]{1,0}, s32[2,1]{1,0}) tuple(multiply_2, multiply_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //         constant
  //            |
  // param1 broadcast  param0
  //     \  /           /
  //   multiply      copy
  //       \         /
  //        \       /
  //         multiply   constant
  //         |     |       |
  //         |  ---+---broadcast
  //         | /   |
  //      multiply |
  //            \  |
  //            tuple
  HloInstruction* param_1;
  HloInstruction* broadcast_0;
  HloInstruction* multiply_0;
  HloInstruction* param_0;
  HloInstruction* copy;
  HloInstruction* multiply_1;
  HloInstruction* broadcast_1;
  HloInstruction* multiply_2;
  HloInstruction* tuple;
  auto multiplyPattern =
      m::Multiply(&multiply_1,
                  m::Multiply(&multiply_0, m::Parameter(&param_1),
                              m::Broadcast(&broadcast_0, m::ConstantScalar(2))),
                  m::Copy(&copy, m::Parameter(&param_0)));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  &tuple,
                  m::Multiply(&multiply_2, multiplyPattern,
                              m::Broadcast(&broadcast_1, m::ConstantScalar(4))),
                  multiplyPattern)));
  TestShapeHasMemorySpace(param_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(param_0->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_2->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                          Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, TupleParameterStreaming) {
  const std::string& hlo_string = R"(
HloModule ParameterStreaming, entry_computation_layout={((s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)S(5)}))->s32[2,1]{1,0:T(2,128)}}

ENTRY main {
  param_tuple = (s32[2,1], s32[2,1]) parameter(0)
  x = get-tuple-element(param_tuple), index=0
  y_host = get-tuple-element(param_tuple), index=1
  y = s32[2,1] custom-call(y_host), custom_call_target="MoveToDevice"
  ROOT crs = s32[2,1] add(x, y)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param0: tuple(x   ,   y)
  //              /         \
  // get-tuple-element  get-tuple-element
  //             \      |
  //              \    copy
  //               \   /
  //                add
  HloInstruction* param;
  HloInstruction* gte_x;
  HloInstruction* gte_y;
  HloInstruction* copy;
  HloInstruction* add;
  auto parameter_pattern = m::Parameter(&param, 0);
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Add(
          &add, m::GetTupleElement(&gte_x, parameter_pattern),
          m::Copy(&copy, m::GetTupleElement(&gte_y, parameter_pattern)))));
  TestShapeHasMemorySpace(param->shape().tuple_shapes(0),
                          Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte_x->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(add->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(param->shape().tuple_shapes(1),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte_y->shape(), Layout::kHostMemorySpace);
}

TEST_F(HostOffloaderTest, ParameterStreamingNoOpToHost) {
  const std::string& hlo_string = R"(
HloModule ParameterStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)S(5)})->s32[2,1]{1,0:T(2,128)}}

ENTRY main {
  param = s32[2,1]{1,0} parameter(0)
  to_host = s32[2,1]{1,0} custom-call(param), custom_call_target="MoveToHost"
  ROOT to_device = s32[2,1]{1,0} custom-call(to_host), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);
  LOG(INFO) << module->ToString();

  // Look for the following pattern:
  //    param
  //      |
  // copy(to device)
  HloInstruction* param;
  HloInstruction* copy;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(&copy, m::Parameter(&param, 0))));
  TestShapeHasMemorySpace(param->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, OutputStreaming) {
  const std::string& hlo_string = R"(
    HloModule ParameterStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)})->(s32[2,1]{1,0:T(2,128)S(5)}, s32[2,1]{1,0:T(2,128)})}

    ENTRY main {
      param_0 = s32[2,1]{1,0} parameter(0)
      param_1 = s32[2,1]{1,0} parameter(1)
      constant_2 = s32[] constant(2)
      constant_4 = s32[] constant(4)
      broadcast_0 = s32[2,1]{1,0} broadcast(constant_2), dimensions={}
      multiply_0 = s32[2,1]{1,0} multiply(param_1, broadcast_0)
      multiply_1 = s32[2,1]{1,0} multiply(multiply_0, param_0)
      broadcast_1 = s32[2,1]{1,0} broadcast(constant_4), dimensions={}
      multiply_2 = s32[2,1]{1,0} multiply(multiply_1, broadcast_1)
      custom_call = s32[2,1]{1,0} custom-call(multiply_2), custom_call_target="MoveToHost"
      ROOT tuple = (s32[2,1]{1,0}, s32[2,1]{1,0}) tuple(custom_call, multiply_1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //         constant
  //            |
  // param1 broadcast  param0
  //     \  /          /
  //   multiply       /
  //       \         /
  //        \       /
  //         multiply   constant
  //         |     |       |
  //         |  ---+---broadcast
  //         | /   |
  //      multiply |
  //          |    |
  //         copy  |
  //           \   |
  //           tuple
  HloInstruction* param_1;
  HloInstruction* broadcast_0;
  HloInstruction* multiply_0;
  HloInstruction* param_0;
  HloInstruction* multiply_1;
  HloInstruction* broadcast_1;
  HloInstruction* multiply_2;
  HloInstruction* copy;
  HloInstruction* tuple;
  auto multiplyPattern =
      m::Multiply(&multiply_1,
                  m::Multiply(&multiply_0, m::Parameter(&param_1),
                              m::Broadcast(&broadcast_0, m::ConstantScalar(2))),
                  m::Parameter(&param_0));
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          &tuple,
          m::Copy(&copy, m::Multiply(
                             &multiply_2, multiplyPattern,
                             m::Broadcast(&broadcast_1, m::ConstantScalar(4)))),
          multiplyPattern)));
  TestShapeHasMemorySpace(param_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(param_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_2->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {1}),
                          Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, InvalidOutputStreaming) {
  const std::string& hlo_string = R"(
    HloModule ParameterStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)})->(s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)})}

    ENTRY main {
      param_0 = s32[2,1]{1,0} parameter(0)
      param_1 = s32[2,1]{1,0} parameter(1)
      constant_2 = s32[] constant(2)
      constant_4 = s32[] constant(4)
      broadcast_0 = s32[2,1]{1,0} broadcast(constant_2), dimensions={}
      multiply_0 = s32[2,1]{1,0} multiply(param_1, broadcast_0)
      multiply_1 = s32[2,1]{1,0} multiply(multiply_0, param_0)
      broadcast_1 = s32[2,1]{1,0} broadcast(constant_4), dimensions={}
      multiply_2 = s32[2,1]{1,0} multiply(multiply_1, broadcast_1)
      custom_call = s32[2,1]{1,0} custom-call(multiply_2), custom_call_target="MoveToHost"
      ROOT tuple = (s32[2,1]{1,0}, s32[2,1]{1,0}) tuple(custom_call, multiply_1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  absl::StatusOr<bool> result = RunHostOffloader(module.get());
  EXPECT_FALSE(result.ok());
}

TEST_F(HostOffloaderTest, OutputStreamingWithoutTuple) {
  const std::string& hlo_string = R"(
    HloModule ParameterStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)})->s32[2,1]{1,0:T(2,128)S(5)}}

    ENTRY main {
      param_0 = s32[2,1]{1,0} parameter(0)
      param_1 = s32[2,1]{1,0} parameter(1)
      constant_2 = s32[] constant(2)
      constant_4 = s32[] constant(4)
      broadcast_0 = s32[2,1]{1,0} broadcast(constant_2), dimensions={}
      multiply_0 = s32[2,1]{1,0} multiply(param_1, broadcast_0)
      multiply_1 = s32[2,1]{1,0} multiply(multiply_0, param_0)
      broadcast_1 = s32[2,1]{1,0} broadcast(constant_4), dimensions={}
      multiply_2 = s32[2,1]{1,0} multiply(multiply_1, broadcast_1)
      ROOT custom_call = s32[2,1]{1,0} custom-call(multiply_2), custom_call_target="MoveToHost"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //         constant
  //            |
  // param1 broadcast  param0
  //     \  /          /
  //   multiply       /
  //       \         /
  //        \       /
  //         multiply   constant
  //         |     |       |
  //         |  ---+---broadcast
  //         | /
  //      multiply
  //          |
  //         copy

  HloInstruction* param_1;
  HloInstruction* broadcast_0;
  HloInstruction* multiply_0;
  HloInstruction* param_0;
  HloInstruction* multiply_1;
  HloInstruction* broadcast_1;
  HloInstruction* multiply_2;
  HloInstruction* copy;
  auto multiplyPattern =
      m::Multiply(&multiply_1,
                  m::Multiply(&multiply_0, m::Parameter(&param_1),
                              m::Broadcast(&broadcast_0, m::ConstantScalar(2))),
                  m::Parameter(&param_0));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(
                  &copy, m::Multiply(&multiply_2, multiplyPattern,
                                     m::Broadcast(&broadcast_1,
                                                  m::ConstantScalar(4))))));
  TestShapeHasMemorySpace(param_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(param_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_2->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kHostMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, OutputStreamingCustomCallRoot) {
  const std::string& hlo_string = R"(
    HloModule ParameterStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)}, s32[2,1]{1,0:T(2,128)})->s32[2,1]{1,0:T(2,128)S(5)}}

    ENTRY main {
      param_0 = s32[2,1]{1,0} parameter(0)
      param_1 = s32[2,1]{1,0} parameter(1)
      constant_2 = s32[] constant(2)
      constant_4 = s32[] constant(4)
      broadcast_0 = s32[2,1]{1,0} broadcast(constant_2), dimensions={}
      multiply_0 = s32[2,1]{1,0} multiply(param_1, broadcast_0)
      multiply_1 = s32[2,1]{1,0} multiply(multiply_0, param_0)
      broadcast_1 = s32[2,1]{1,0} broadcast(constant_4), dimensions={}
      multiply_2 = s32[2,1]{1,0} multiply(multiply_1, broadcast_1)
      ROOT custom_call = s32[2,1]{1,0} custom-call(multiply_2), custom_call_target="MoveToHost"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //         constant
  //            |
  // param1 broadcast  param0
  //     \  /          /
  //   multiply       /
  //       \         /
  //        \       /
  //         multiply   constant
  //         |             |
  //         |  ---+---broadcast
  //         | /
  //      multiply
  //          |
  //         copy
  HloInstruction* param_1;
  HloInstruction* broadcast_0;
  HloInstruction* multiply_0;
  HloInstruction* param_0;
  HloInstruction* multiply_1;
  HloInstruction* broadcast_1;
  HloInstruction* multiply_2;
  HloInstruction* copy;
  auto multiplyPattern =
      m::Multiply(&multiply_1,
                  m::Multiply(&multiply_0, m::Parameter(&param_1),
                              m::Broadcast(&broadcast_0, m::ConstantScalar(2))),
                  m::Parameter(&param_0));
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(
                  &copy, m::Multiply(&multiply_2, multiplyPattern,
                                     m::Broadcast(&broadcast_1,
                                                  m::ConstantScalar(4))))));
  TestShapeHasMemorySpace(param_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(param_0->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(broadcast_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(multiply_2->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kHostMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, OutputStreamingInUnrolledScanLoop) {
  const std::string& hlo_string = R"(
HloModule m,
entry_computation_layout={(s32[16,16,8]{1,2,0:T(8,128)})->s32[16,16,8]{1,2,0:T(8,128)S(5)}},
allow_spmd_sharding_propagation_to_output={true}, num_partitions=2

body {
  loop_peel_param = (s32[]{:T(256)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[]{:T(256)}, s32[16,8]{0,1:T(8,128)}) parameter(0)
  get-tuple-element.12 = s32[]{:T(256)} get-tuple-element(loop_peel_param), index=0
  constant.29 = s32[]{:T(256)} constant(1)
  add.5 = s32[]{:T(256)} add(get-tuple-element.12, constant.29)
  get-tuple-element.13 = s32[16,16,8]{1,2,0:T(8,128)} get-tuple-element(loop_peel_param), index=1
  get-tuple-element.18 = s32[16,8]{0,1:T(8,128)} get-tuple-element(loop_peel_param), index=4
  custom-call.3 = s32[16,8]{0,1:T(8,128)} custom-call(get-tuple-element.18), custom_call_target="MoveToHost"
  bitcast = s32[1,16,8]{1,2,0:T(8,128)} bitcast(custom-call.3)
  get-tuple-element.15 = s32[]{:T(256)} get-tuple-element(loop_peel_param), index=3
  constant.30 = s32[]{:T(256)} constant(0)
  dynamic-update-slice.2 = s32[16,16,8]{1,2,0:T(8,128)} dynamic-update-slice(get-tuple-element.13, bitcast, get-tuple-element.15, constant.30, constant.30), backend_config={"flag_configs":[],"scoped_memory_configs":[],"indices_config":{"index_known_bits":[{"zeroes":"0","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"}]},"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_INVALID","used_scoped_memory_configs":[]}
  get-tuple-element.14 = s32[16,16,8]{1,2,0:T(8,128)} get-tuple-element(loop_peel_param), index=2
  dynamic-slice.2 = s32[1,16,8]{1,2,0:T(8,128)} dynamic-slice(get-tuple-element.14, get-tuple-element.12, constant.30, constant.30), dynamic_slice_sizes={1,16,8}
  broadcast.8 = s32[1,16,8]{1,2,0:T(8,128)} broadcast(constant.29), dimensions={}
  add.6 = s32[1,16,8]{1,2,0:T(8,128)} add(dynamic-slice.2, broadcast.8)
  bitcast.1 = s32[16,8]{0,1:T(8,128)} bitcast(add.6)
  ROOT tuple.3 = (s32[]{:T(256)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[]{:T(256)}, s32[16,8]{0,1:T(8,128)}) tuple(add.5, dynamic-update-slice.2, get-tuple-element.14, get-tuple-element.12, bitcast.1)
} // body

condition {
  loop_peel_cond_param = (s32[]{:T(256)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[]{:T(256)}, s32[16,8]{0,1:T(8,128)}) parameter(0)
  get-tuple-element.11 = s32[]{:T(256)} get-tuple-element(loop_peel_cond_param), index=0
  constant.28 = s32[]{:T(256)} constant(16)
  ROOT compare.1 = pred[]{:T(1024)} compare(get-tuple-element.11, constant.28), direction=LT
}

ENTRY entry {
  constant.26 = s32[]{:T(256)} constant(1)
  constant.24 = s32[]{:T(256)} constant(0)
  broadcast.6 = s32[16,16,8]{1,2,0:T(8,128)} broadcast(constant.24), dimensions={}
  param.2 = s32[16,16,8]{1,2,0:T(8,128)} parameter(0), sharding={devices=[1,1,2]<=[2]}
  slice = s32[1,16,8]{1,2,0:T(8,128)} slice(param.2), slice={[0:1], [0:16], [0:8]}
  broadcast.7 = s32[1,16,8]{1,2,0:T(8,128)} broadcast(constant.26), dimensions={}
  add.4 = s32[1,16,8]{1,2,0:T(8,128)} add(slice, broadcast.7)
  bitcast.2 = s32[16,8]{0,1:T(8,128)} bitcast(add.4)
  tuple.4 = (s32[]{:T(256)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[]{:T(256)}, s32[16,8]{0,1:T(8,128)}) tuple(constant.26, broadcast.6, param.2, constant.24, bitcast.2)
  while.1 = (s32[]{:T(256)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[16,16,8]{1,2,0:T(8,128)}, s32[]{:T(256)}, s32[16,8]{0,1:T(8,128)}) while(tuple.4), condition=condition, body=body
  get-tuple-element.17 = s32[16,16,8]{1,2,0:T(8,128)} get-tuple-element(while.1), index=1
  get-tuple-element.19 = s32[16,8]{0,1:T(8,128)} get-tuple-element(while.1), index=4
  custom-call.4 = s32[16,8]{0,1:T(8,128)} custom-call(get-tuple-element.19), custom_call_target="MoveToHost"
  bitcast.3 = s32[1,16,8]{1,2,0:T(8,128)} bitcast(custom-call.4)
  get-tuple-element.16 = s32[]{:T(256)} get-tuple-element(while.1), index=3
  ROOT dynamic-update-slice.3 = s32[16,16,8]{1,2,0:T(8,128)} dynamic-update-slice(get-tuple-element.17, bitcast.3, get-tuple-element.16, constant.24, constant.24), backend_config={"flag_configs":[],"scoped_memory_configs":[],"indices_config":{"index_known_bits":[{"zeroes":"0","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"}]},"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_INVALID","used_scoped_memory_configs":[]}
} // entry
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* bitcast;
  HloInstruction* gte_0;
  HloInstruction* gte_1;
  HloInstruction* dus;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice(
                  &dus, m::GetTupleElement(&gte_0), m::Bitcast(&bitcast),
                  m::GetTupleElement(&gte_1), m::ConstantScalar(0),
                  m::ConstantScalar(0))));

  TestShapeHasMemorySpace(bitcast->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(dus->shape(), Layout::kHostMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, OutputStreamingNoOpToDevice) {
  const std::string& hlo_string = R"(
HloModule OutputStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)})->s32[2,1]{1,0:T(2,128)S(5)}}

ENTRY main {
  param = s32[2,1]{1,0} parameter(0)
  to_device = s32[2,1]{1,0} custom-call(param), custom_call_target="MoveToDevice"
  ROOT to_host = s32[2,1]{1,0} custom-call(to_device), custom_call_target="MoveToHost"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);
  LOG(INFO) << module->ToString();

  // Look for the following pattern:
  //    param
  //      |
  // copy(to host)
  HloInstruction* param;
  HloInstruction* copy;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(&copy, m::Parameter(&param, 0))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy->shape(), Layout::kHostMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, ParameterAndOutputStreamingPassThrough) {
  const std::string& hlo_string = R"(
HloModule OutputStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)S(5)})->s32[2,1]{1,0:T(2,128)S(5)}}

ENTRY main {
  ROOT param = s32[2,1]{1,0} parameter(0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);
  HloInstruction* param;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(&param, 0)));
  TestShapeHasMemorySpace(param->shape(), Layout::kHostMemorySpace);
}

TEST_F(HostOffloaderTest, ParameterAndOutputStreamingPassThroughTuple) {
  const std::string& hlo_string = R"(
HloModule OutputStreaming, entry_computation_layout={(s32[2,1]{1,0:T(2,128)S(5)})->s32[2,1]{1,0:T(2,128)S(5)}}

ENTRY main {
  param = s32[2,1]{1,0} parameter(0)
  tuple = (s32[2,1]{1,0}) tuple(param)
  ROOT gte = s32[2,1]{1,0} get-tuple-element(tuple), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  //  param
  //    |
  //  tuple
  //    |
  //   gte
  HloInstruction* param;
  HloInstruction* tuple;
  HloInstruction* gte;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  &gte, m::Tuple(&tuple, m::Parameter(&param, 0)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(tuple->shape(), {0}),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(gte->shape(), Layout::kHostMemorySpace);
}

TEST_F(HostOffloaderTest, LoneMoveToDevice) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1})->f32[16,256]{1,0}}

ENTRY main {
  param_0 = f32[16,256]{0,1} parameter(0)
  ROOT custom_call_2 = f32[16,256]{0,1} custom-call(param_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* param;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(&param, 0)));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, RepeatedMoveToHost) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1})->f32[16,256]{1,0}}

ENTRY main {
  param_0 = f32[16,256]{0,1} parameter(0)
  custom_call_0 = f32[16,256]{0,1} custom-call(param_0), custom_call_target="MoveToHost"
  custom_call_1 = f32[16,256]{0,1} custom-call(custom_call_0), custom_call_target="MoveToHost"
  ROOT custom_call_2 = f32[16,256]{0,1} custom-call(custom_call_1), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(&copy_to_device,
                         m::Copy(&copy_to_host, m::Parameter(&param, 0)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, RepeatedMoveToDevice) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1})->f32[16,256]{1,0}}

ENTRY main {
  param_0 = f32[16,256]{0,1} parameter(0)
  custom_call_0 = f32[16,256]{0,1} custom-call(param_0), custom_call_target="MoveToHost"
  custom_call_1 = f32[16,256]{0,1} custom-call(custom_call_0), custom_call_target="MoveToDevice"
  ROOT custom_call_2 = f32[16,256]{0,1} custom-call(custom_call_1), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(&copy_to_device,
                         m::Copy(&copy_to_host, m::Parameter(&param, 0)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, RepeatedMoveToHostNonSequential) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1})->f32[16,256]{1,0}}

ENTRY main {
  param_0 = f32[16,256]{0,1} parameter(0)
  custom_call_0 = f32[16,256]{0,1} custom-call(param_0), custom_call_target="MoveToHost"
  custom_call_1 = f32[16,256]{0,1} custom-call(param_0), custom_call_target="MoveToHost"
  ROOT custom_call_2 = f32[16,256]{0,1} custom-call(custom_call_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |
  // copy (to device)

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(&copy_to_device,
                         m::Copy(&copy_to_host, m::Parameter(&param, 0)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, RepeatedMoveToDeviceNonSequential) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1})->f32[16,256]{1,0}}

ENTRY main {
  param_0 = f32[16,256]{0,1} parameter(0)
  custom_call_0 = f32[16,256]{0,1} custom-call(param_0), custom_call_target="MoveToHost"
  custom_call_1 = f32[16,256]{0,1} custom-call(custom_call_0), custom_call_target="MoveToDevice"
  ROOT custom_call_2 = f32[16,256]{0,1} custom-call(custom_call_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));

  EXPECT_TRUE(changed);

  // Look for the following pattern:
  // param
  //   |
  // copy (to host)
  //   |
  // copy (to device)
  // Note: There is another copy with another user, but that's not our problem.

  HloInstruction* param;
  HloInstruction* copy_to_host;
  HloInstruction* copy_to_device;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Copy(&copy_to_device,
                         m::Copy(&copy_to_host, m::Parameter(&param, 0)))));
  TestShapeHasMemorySpace(param->shape(), Layout::kDefaultMemorySpace);
  TestShapeHasMemorySpace(copy_to_host->shape(), Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(copy_to_device->shape(), Layout::kDefaultMemorySpace);

  EXPECT_FALSE(HaveRemainingOffloadAnnotations(module.get()));
}

TEST_F(HostOffloaderTest, BasicAsyncHostOffloadedCall_RemoveRedundantCopies) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={(f32[4096]{0:S(5)})->(f32[4096]{0:S(5)}, f32[4096]{0:S(5)})}

%async_computation {
  %param_0 = f32[4096] parameter(0)
  ROOT %offloaded-custom-call = (f32[4096], f32[4096]) custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

ENTRY %main {
  %a = f32[4096] parameter(0)
  %async-start = ((f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%a), async_execution_thread="host", calls=%async_computation
  %async-done = (f32[4096], f32[4096]) custom-call-done(%async-start)
  %gte_0 = f32[4096] get-tuple-element(%async-done), index=0
  %gte_1 = f32[4096] get-tuple-element(%async-done), index=1
  %gte_0_host = f32[4096] custom-call(%gte_0), custom_call_target="MoveToHost"
  %gte_1_host = f32[4096] custom-call(%gte_1), custom_call_target="MoveToHost"
  ROOT %tuple = (f32[4096], f32[4096]) tuple(%gte_0_host, %gte_1_host)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  ASSERT_NE(async_start, nullptr);
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  ASSERT_NE(async_done, nullptr);

  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte_1");
  ASSERT_NE(gte_1, nullptr);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);

  HloInstruction* gte_0_host = FindInstruction(module.get(), "gte_0_host");
  ASSERT_EQ(gte_0_host, nullptr);
  HloInstruction* gte_1_host = FindInstruction(module.get(), "gte_1_host");
  ASSERT_EQ(gte_1_host, nullptr);

  // Check all set of successors.
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  std::vector<HloInstruction*> expected = {gte_0, gte_1};
  EXPECT_THAT(tuple->operands(),
              ::testing::UnorderedElementsAreArray(expected));
}

TEST_F(HostOffloaderTest,
       BasicAsyncHostOffloadedCall_NoChangesWhenEntryLayoutExpectsHBM) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={(f32[4096]{0:S(5)})->(f32[4096]{0:S(0)}, f32[4096]{0:S(0)})}

%async_computation {
  %param_0 = f32[4096] parameter(0)
  ROOT %offloaded-custom-call = (f32[4096], f32[4096]) custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

ENTRY %main {
  %a = f32[4096] parameter(0)
  %async-start = ((f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%a), async_execution_thread="host", calls=%async_computation
  %async-done = (f32[4096], f32[4096]) custom-call-done(%async-start)
  %gte_0 = f32[4096] get-tuple-element(%async-done), index=0
  %gte_1 = f32[4096] get-tuple-element(%async-done), index=1
  ROOT %tuple = (f32[4096], f32[4096]) tuple(%gte_0, %gte_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK(RunHostOffloader(module.get()));

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  ASSERT_NE(async_start, nullptr);
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  ASSERT_NE(async_done, nullptr);

  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kDefaultMemorySpace);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte_1");
  ASSERT_NE(gte_1, nullptr);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kDefaultMemorySpace);
}

TEST_F(HostOffloaderTest,
       BasicAsyncHostOffloadedCall_RemoveOnlyRedundantCopies) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={(f32[4096]{0:S(5)})->(f32[4096]{0:S(5)}, f32[4096]{0:S(5)})}

%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add_res = f32[] add(%lhs, %rhs)
}

%async_computation {
  %param_0 = f32[4096] parameter(0)
  ROOT %offloaded-custom-call = (f32[4096], f32[4096]) custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

ENTRY %main {
  %a = f32[4096] parameter(0)
  %async-start = ((f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%a), async_execution_thread="host", calls=%async_computation
  %async-done = (f32[4096], f32[4096]) custom-call-done(%async-start)
  %gte_0 = f32[4096] get-tuple-element(%async-done), index=0
  %gte_1 = f32[4096] get-tuple-element(%async-done), index=1
  %sum = f32[4096] add(%gte_0, %gte_0)
  %gte_0_host = f32[4096] custom-call(%gte_0), custom_call_target="MoveToHost"
  %gte_1_host = f32[4096] custom-call(%gte_1), custom_call_target="MoveToHost"
  ROOT %tuple = (f32[4096]{0:S(5)}, f32[4096]) tuple(%gte_0_host, %gte_1_host)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  ASSERT_NE(async_start, nullptr);
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  ASSERT_NE(async_done, nullptr);

  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kDefaultMemorySpace);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte_1");
  ASSERT_NE(gte_1, nullptr);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);

  // Since gte_0 is used on device (we do not take dead code into account here
  // ..) gte_0 will be copied to device and be moved to host.
  HloInstruction* gte_0_host = FindInstruction(module.get(), "gte_0_host");
  ASSERT_EQ(gte_0_host, nullptr);  // replaced with copy
  HloInstruction* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->operands()[0], gte_0);

  HloInstruction* gte_1_host = FindInstruction(module.get(), "gte_1_host");
  ASSERT_EQ(gte_1_host, nullptr);
}

TEST_F(HostOffloaderTest,
       AsyncHostOffloadedCall_nonEntryPoint_RemoveRedundantCopies) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={(f32[4096]{0:S(5)})->(f32[4096]{0:S(5)}, f32[4096]{0:S(5)})}

%async_computation {
  %param_0 = f32[4096] parameter(0)
  ROOT %offloaded-custom-call = (f32[4096], f32[4096]) custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

%non_async_computation {
  %param_0 = f32[4096] parameter(0)
  %async-start = ((f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%param_0), async_execution_thread="host", calls=%async_computation
  %async-done = (f32[4096], f32[4096]) custom-call-done(%async-start)
  %gte_0 = f32[4096] get-tuple-element(%async-done), index=0
  %gte_1 = f32[4096] get-tuple-element(%async-done), index=1
  %gte_0_host = f32[4096] custom-call(%gte_0), custom_call_target="MoveToHost"
  %gte_1_host = f32[4096] custom-call(%gte_1), custom_call_target="MoveToHost"
  ROOT %tuple_non_async = (f32[4096]{0:S(5)}, f32[4096]) tuple(%gte_0_host, %gte_1_host)
}

ENTRY %main {
  %a = f32[4096] parameter(0)
  %call = (f32[4096], f32[4096]) call(%a), to_apply=%non_async_computation
  %call_0 = f32[4096] get-tuple-element(%call), index=0
  %call_1 = f32[4096] get-tuple-element(%call), index=1
  ROOT %tuple = (f32[4096], f32[4096]) tuple(%call_0, %call_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  ASSERT_NE(async_start, nullptr);
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  ASSERT_NE(async_done, nullptr);

  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte_1");
  ASSERT_NE(gte_1, nullptr);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);

  HloInstruction* gte_0_host = FindInstruction(module.get(), "gte_0_host");
  ASSERT_EQ(gte_0_host, nullptr);
  HloInstruction* gte_1_host = FindInstruction(module.get(), "gte_1_host");
  ASSERT_EQ(gte_1_host, nullptr);

  HloInstruction* tuple_non_async =
      FindInstruction(module.get(), "tuple_non_async");
  ASSERT_NE(tuple_non_async, nullptr);
  std::vector<HloInstruction*> expected = {gte_0, gte_1};
  EXPECT_THAT(tuple_non_async->operands(),
              ::testing::UnorderedElementsAreArray(expected));

  // Check the main output is on host.
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  TestShapeHasMemorySpace(tuple->shape().tuple_shapes(0),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(tuple->shape().tuple_shapes(1),
                          Layout::kHostMemorySpace);
}

TEST_F(HostOffloaderTest,
       AsyncHostOffloadedCall_passedToCall_RemoveRedundantCopies) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={(f32[4096]{0:S(5)})->(f32[4096]{0:S(5)}, f32[4096]{0:S(5)})}

%async_computation {
  %param_0 = f32[4096] parameter(0)
  ROOT %offloaded-custom-call = (f32[4096], f32[4096]) custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

%non_async_computation {
  %param_0_non_async = f32[4096] parameter(0)
  %param_1_non_async = f32[4096] parameter(1)
  ROOT %tuple_non_async = (f32[4096], f32[4096]) tuple(%param_0_non_async, %param_1_non_async)
}

ENTRY %main {
  %a = f32[4096] parameter(0)
  %async-start = ((f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%a), async_execution_thread="host", calls=%async_computation
  %async-done = (f32[4096], f32[4096]) custom-call-done(%async-start)
  %gte_0 = f32[4096] get-tuple-element(%async-done), index=0
  %gte_1 = f32[4096] get-tuple-element(%async-done), index=1
  %call = (f32[4096], f32[4096]) call(%gte_0, %gte_1), to_apply=%non_async_computation
  %call_0 = f32[4096] get-tuple-element(%call), index=0
  %call_1 = f32[4096] get-tuple-element(%call), index=1
  %call_0_host = f32[4096] custom-call(%call_0), custom_call_target="MoveToHost"
  %call_1_host = f32[4096] custom-call(%call_1), custom_call_target="MoveToHost"
  ROOT %tuple = (f32[4096], f32[4096]) tuple(%call_0_host, %call_1_host)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  ASSERT_NE(async_start, nullptr);
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  ASSERT_NE(async_done, nullptr);

  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kHostMemorySpace);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte_1");
  ASSERT_NE(gte_1, nullptr);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kHostMemorySpace);

  HloInstruction* call_0 = FindInstruction(module.get(), "call_0");
  ASSERT_NE(call_0, nullptr);
  HloInstruction* call_1 = FindInstruction(module.get(), "call_1");
  ASSERT_NE(call_1, nullptr);

  HloInstruction* call_0_host = FindInstruction(module.get(), "call_0_host");
  ASSERT_EQ(call_0_host, nullptr);
  HloInstruction* call_1_host = FindInstruction(module.get(), "call_1_host");
  ASSERT_EQ(call_1_host, nullptr);

  HloInstruction* param_0_non_async =
      FindInstruction(module.get(), "param_0_non_async");
  ASSERT_NE(param_0_non_async, nullptr);
  TestShapeHasMemorySpace(param_0_non_async->shape(), Layout::kHostMemorySpace);
  HloInstruction* param_1_non_async =
      FindInstruction(module.get(), "param_1_non_async");
  ASSERT_NE(param_1_non_async, nullptr);
  TestShapeHasMemorySpace(param_1_non_async->shape(), Layout::kHostMemorySpace);

  HloInstruction* tuple_non_async =
      FindInstruction(module.get(), "tuple_non_async");
  ASSERT_NE(tuple_non_async, nullptr);
  std::vector<HloInstruction*> expected_operands = {param_0_non_async,
                                                    param_1_non_async};
  EXPECT_THAT(tuple_non_async->operands(),
              ::testing::UnorderedElementsAreArray(expected_operands));

  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);
  TestShapeHasMemorySpace(tuple->shape().tuple_shapes(0),
                          Layout::kHostMemorySpace);
  TestShapeHasMemorySpace(tuple->shape().tuple_shapes(1),
                          Layout::kHostMemorySpace);

  std::vector<HloInstruction*> expected = {call_0, call_1};
  EXPECT_THAT(tuple->operands(),
              ::testing::UnorderedElementsAreArray(expected));
}

TEST_F(HostOffloaderTest,
       AsyncHostOffloadedCall_passedToAsyncHostOffloadedCall_NoCopiesRemoved) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={(f32[4096]{0:S(5)})->(f32[4096]{0:S(5)}, f32[4096]{0:S(5)}, f32[4096]{0:S(0)}, f32[4096]{0:S(0)})}

%async_computation {
  %param_0 = f32[4096] parameter(0)
  ROOT %offloaded-custom-call = (f32[4096], f32[4096]) custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

%extra_async_computation {
  %param_0_extra_async = f32[4096] parameter(0)
  %param_1_extra_async = f32[4096] parameter(1)
  ROOT %offloaded-extra-custom-call = (f32[4096], f32[4096]) custom-call(%param_0_extra_async, %param_1_extra_async), custom_call_target="HostExecute"
}, execution_thread="host"

ENTRY %main {
  %a = f32[4096] parameter(0)
  %async-start = ((f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%a), async_execution_thread="host", calls=%async_computation
  %async-done = (f32[4096], f32[4096]) custom-call-done(%async-start)
  %gte_0 = f32[4096] get-tuple-element(%async-done), index=0
  %gte_1 = f32[4096] get-tuple-element(%async-done), index=1
  %extra-async-start = ((f32[4096], f32[4096]), (f32[4096], f32[4096]), u32[]) async-start(%gte_0, %gte_1), async_execution_thread="host", calls=%extra_async_computation
  %extra-async-done = (f32[4096], f32[4096]) custom-call-done(%extra-async-start)
  %call_0 = f32[4096] get-tuple-element(%extra-async-done), index=0
  %call_1 = f32[4096] get-tuple-element(%extra-async-done), index=1
  %gte_0_host = f32[4096] custom-call(%gte_0), custom_call_target="MoveToHost"
  %gte_1_host = f32[4096] custom-call(%gte_1), custom_call_target="MoveToHost"
  ROOT %tuple = (f32[4096], f32[4096], f32[4096], f32[4096]) tuple(%gte_0_host, %gte_1_host, %call_0, %call_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  ASSERT_NE(async_start, nullptr);
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");
  ASSERT_NE(async_done, nullptr);

  // No changes are made since both outputs flow into an async call...
  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  TestShapeHasMemorySpace(gte_0->shape(), Layout::kDefaultMemorySpace);
  HloInstruction* gte_1 = FindInstruction(module.get(), "gte_1");
  ASSERT_NE(gte_1, nullptr);
  TestShapeHasMemorySpace(gte_1->shape(), Layout::kDefaultMemorySpace);
}

TEST_F(HostOffloaderTest, OffloadPassedToEntryComputationRoot) {
  const std::string& hlo_string = R"(
HloModule m, entry_computation_layout={()->(s32[]{:T(128)})}

ENTRY %main {
  c = s32[] constant(1)
  custom-call.331 = s32[]{:T(128)} custom-call(c), custom_call_target="MoveToHost"
  custom-call.332 = s32[]{:T(128)} custom-call(custom-call.331), custom_call_target="MoveToDevice"
  ROOT tuple = (s32[]{:T(128)}) tuple(custom-call.332)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << "module after: " << module->ToString();
}

// Test to ensure that HostOffloader can handle the case in which a
// MoveToHost(broadcast(...)) is shared between two
// DynamicUpdateSlice(MoveToHost(...)) in a while loop body.
TEST_F(HostOffloaderTest, MoveToHostInsideWhileLoopBodyShareSameBroadcast) {
  const absl::string_view hlo_string = R"(
    HloModule MoveToHostFoundOutsideAndInsideOfWhileLoop, entry_computation_layout={(s32[],f32[1,1,128,128],f32[1,1,128,128])->(f32[8,1,128,128]{3,2,1,0:T(8,128)S(5)}, f32[8,1,128,128]{3,2,1,0:T(8,128)S(5)}, f32[1,1,128,128], f32[1,1,128,128], s32[], s32[])} 
    
    while_condition {
      condition_param = (f32[8,1,128,128], f32[8,1,128,128], f32[1,1,128,128], f32[1,1,128,128], s32[], s32[]) parameter(0)
      condition_current_iteration_index = s32[] get-tuple-element(condition_param), index=5
      condition_iteration_count = s32[] constant(16)
      ROOT condition_result = pred[] compare(condition_current_iteration_index, condition_iteration_count), direction=LT
    }

    while_body {
      while_body_input_tuple = (f32[8,1,128,128], f32[8,1,128,128], f32[1,1,128,128], f32[1,1,128,128], s32[], s32[]) parameter(0)
      host_tensor_1 = f32[8,1,128,128] get-tuple-element(while_body_input_tuple), index=0
      host_tensor_2 = f32[8,1,128,128] get-tuple-element(while_body_input_tuple), index=1
      update_1 = f32[1,1,128,128] get-tuple-element(while_body_input_tuple), index=2
      update_2 = f32[1,1,128,128] get-tuple-element(while_body_input_tuple), index=3
      offset_dus = s32[] get-tuple-element(while_body_input_tuple), index=4
      while_body_num_iter = s32[] get-tuple-element(while_body_input_tuple), index=5
      mth_tensor_1 = f32[8,1,128,128] custom-call(host_tensor_1), custom_call_target="MoveToHost"
      mth_tensor_2 = f32[8,1,128,128] custom-call(host_tensor_2), custom_call_target="MoveToHost"
      constant_zero = s32[] constant(0)
      host_dus_1 = f32[8,1,128,128]{3,2,1,0:T(8,128)} dynamic-update-slice(mth_tensor_1, update_1, offset_dus, constant_zero, constant_zero, constant_zero)
      host_dus_2 = f32[8,1,128,128]{3,2,1,0:T(8,128)} dynamic-update-slice(mth_tensor_2, update_2, offset_dus, constant_zero, constant_zero, constant_zero)
      ROOT while_output_tuple = tuple(host_dus_1,host_dus_2, update_1, update_2, offset_dus, while_body_num_iter)
    }

    ENTRY main {
      offset = s32[] parameter(0)
      update = f32[1,1,128,128] parameter(1)
      update2 = f32[1,1,128,128] parameter(2)
      constant = f32[] constant(1.0)
      /*Shared broadcast between two MoveToHost inside while body.*/
      broadcast = f32[8,1,128,128] broadcast(constant)
      shared_host_memory = f32[8,1,128,128] custom-call(broadcast), custom_call_target="MoveToHost"
      tuple_for_while = (f32[8,1,128,128], f32[8,1,128,128], f32[1,1,128,128], f32[1,1,128,128], s32[], s32[]) tuple(shared_host_memory, shared_host_memory, update, update2, offset, offset)
      ROOT while = (f32[8,1,128,128], f32[8,1,128,128], f32[1,1,128,128], f32[1,1,128,128], s32[], s32[]) while(tuple_for_while), condition=while_condition, body=while_body
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
}

// Test to ensure HostOffloader removes redundant copies back to host when
// the output is a non-tuple.
TEST_F(HostOffloaderTest, RemoveRedundantCopiesBackToHostOutputIsNonTuple) {
  const absl::string_view hlo_string = R"(
    HloModule jit_main, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }, entry_computation_layout={(f32[1048576]{0:T(1024)}, f32[25769803776]{0:T(1024)S(5)})->(f32[1048576]{0:T(1024)}, f32[25769803776]{0:T(1024)S(5)})}, allow_spmd_sharding_propagation_to_parameters={false,false}, allow_spmd_sharding_propagation_to_output={false,false}

    %host_fn.6 (Arg_0.7: f32[25769803776]) -> f32[25769803776] {
      %Arg_0.7 = f32[25769803776]{0} parameter(0), metadata={op_name="jit(main)/jit(main)/pjit"}
      %constant.8 = f32[] constant(1)
      %broadcast.9 = f32[25769803776]{0} broadcast(f32[] %constant.8), dimensions={}, metadata={op_name="jit(main)/jit(main)/jit(host_fn)/add" source_file="third_party/py/jax/tests/memories_test.py" source_line=1448}
      ROOT %add.10 = f32[25769803776]{0} add(f32[25769803776]{0} %Arg_0.7, f32[25769803776]{0} %broadcast.9), frontend_attributes={_xla_compute_type="host"}, metadata={op_name="jit(main)/jit(main)/jit(host_fn)/add" source_file="third_party/py/jax/tests/memories_test.py" source_line=1448}
    }, execution_thread="host"

    ENTRY %main.17 (Arg_0.1: f32[1048576], Arg_1.2: f32[25769803776]) -> (f32[1048576], f32[25769803776]) {
      %Arg_0.1 = f32[1048576]{0:T(1024)} parameter(0), sharding={replicated}, metadata={op_name="a"}
      %constant.3 = f32[]{:T(128)} constant(1)
      %broadcast.4 = f32[1048576]{0:T(1024)} broadcast(f32[]{:T(128)} %constant.3), dimensions={}, metadata={op_name="jit(main)/jit(main)/add" source_file="third_party/py/jax/tests/memories_test.py" source_line=1454}
      %add.5 = f32[1048576]{0:T(1024)} add(f32[1048576]{0:T(1024)} %Arg_0.1, f32[1048576]{0:T(1024)} %broadcast.4), metadata={op_name="jit(main)/jit(main)/add" source_file="third_party/py/jax/tests/memories_test.py" source_line=1454}
      %custom-call = f32[1048576]{0:T(1024)} custom-call(f32[1048576]{0:T(1024)} %add.5), custom_call_target="MoveToDevice"
      %Arg_1.2 = f32[25769803776]{0:T(1024)} parameter(1), sharding={replicated}, metadata={op_name="b"}
      %host-async-start = ((f32[25769803776]{0:T(1024)}), f32[25769803776]{0:T(1024)}, u32[]{:T(128)}) custom-call-start(f32[25769803776]{0:T(1024)} %Arg_1.2), async_execution_thread="host", custom_call_target="HostExecute", called_computations={%host_fn.6}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
      %host-async-done = f32[25769803776]{0:T(1024)} custom-call-done(((f32[25769803776]{0:T(1024)}), f32[25769803776]{0:T(1024)}, u32[]{:T(128)}) %host-async-start), backend_config={"flag_configs":[],"scoped_memory_configs":[],"device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
      %redundant-move-to-host = f32[25769803776]{0:T(1024)} custom-call(f32[25769803776]{0:T(1024)} %host-async-done), custom_call_target="MoveToHost"
      ROOT %output_tuple = (f32[1048576]{0:T(1024)}, f32[25769803776]{0:T(1024)}) tuple(f32[1048576]{0:T(1024)} %custom-call, f32[25769803776]{0:T(1024)} %redundant-move-to-host), sharding={{replicated}, {replicated}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();

  HloInstruction* async_start =
      FindInstruction(module.get(), "host-async-start");
  ASSERT_NE(async_start, nullptr);
  // Input is on host.
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(async_start->shape(), {0, 0}),
                          Layout::kHostMemorySpace);
  // Output is on host.
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(async_start->shape(), {1}),
                          Layout::kHostMemorySpace);
  HloInstruction* async_done = FindInstruction(module.get(), "host-async-done");
  ASSERT_NE(async_done, nullptr);
  TestShapeHasMemorySpace(async_done->shape(), Layout::kHostMemorySpace);

  HloInstruction* output_tuple = FindInstruction(module.get(), "output_tuple");
  ASSERT_NE(output_tuple, nullptr);
  TestShapeHasMemorySpace(ShapeUtil::GetSubshape(output_tuple->shape(), {1}),
                          Layout::kHostMemorySpace);
}

// Test to ensure that redundant "MoveToHost" instructions do not produce
// redundant copy to host instructions after running the host offloader pass.
TEST_F(HostOffloaderTest, AvoidRedundantCopiesToHost) {
  const absl::string_view hlo_string = R"(
    HloModule AvoidRedundantCopiesToHost, entry_computation_layout={(bf16[65536,1024]{1,0:T(8,128)(2,1)})->bf16[65536,1024]{1,0:T(8,128)(2,1)S(5)}}, num_partitions=8

    body {
      param.1 = (s32[]{:T(128)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}) parameter(0)
      get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(param.1), index=0
      constant.22 = s32[]{:T(128)} constant(1)
      add.1 = s32[]{:T(128)} add(get-tuple-element.3, constant.22)
      get-tuple-element.4 = bf16[65536,1024]{1,0:T(8,128)(2,1)} get-tuple-element(param.1), index=1
      get-tuple-element.5 = bf16[65536,1024]{1,0:T(8,128)(2,1)} get-tuple-element(param.1), index=2
      constant.23 = s32[]{:T(128)} constant(8)
      multiply.1 = s32[]{:T(128)} multiply(get-tuple-element.3, constant.23)
      constant.24 = s32[]{:T(128)} constant(0)
      compare.3 = pred[]{:T(512)} compare(multiply.1, constant.24), direction=LT
      constant.25 = s32[]{:T(128)} constant(65536)
      add.2 = s32[]{:T(128)} add(multiply.1, constant.25)
      select.1 = s32[]{:T(128)} select(compare.3, add.2, multiply.1)
      dynamic-slice.1 = bf16[8,1024]{1,0:T(8,128)(2,1)} dynamic-slice(get-tuple-element.5, select.1, constant.24), dynamic_slice_sizes={8,1024}
      custom-call.4 = bf16[8,1024]{1,0:T(8,128)(2,1)} custom-call(dynamic-slice.1), custom_call_target="MoveToHost"
      dynamic-update-slice.0 = bf16[65536,1024]{1,0:T(8,128)(2,1)} dynamic-update-slice(get-tuple-element.4, custom-call.4, select.1, constant.24)
      ROOT tuple.1 = (s32[]{:T(128)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}) tuple(add.1, dynamic-update-slice.0, get-tuple-element.5)
    }

    or_comp {
      Arg_0.27 = pred[]{:T(512)} parameter(0)
      Arg_1.28 = pred[]{:T(512)} parameter(1)
      ROOT or.29 = pred[]{:T(512)} or(Arg_0.27, Arg_1.28)
    }

    condition {
      param = (s32[]{:T(128)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}) parameter(0)
      get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(param), index=0
      constant.15 = s32[]{:T(128)} constant(8)
      multiply.0 = s32[]{:T(128)} multiply(get-tuple-element.1, constant.15)
      constant.16 = s32[]{:T(128)} constant(65536)
      compare.0 = pred[]{:T(512)} compare(multiply.0, constant.16), direction=LT
      get-tuple-element.2 = bf16[65536,1024]{1,0:T(8,128)(2,1)} get-tuple-element(param), index=2
      constant.17 = s32[]{:T(128)} constant(0)
      compare.1 = pred[]{:T(512)} compare(multiply.0, constant.17), direction=LT
      add.0 = s32[]{:T(128)} add(multiply.0, constant.16)
      select.0 = s32[]{:T(128)} select(compare.1, add.0, multiply.0)
      dynamic-slice.0 = bf16[8,1024]{1,0:T(8,128)(2,1)} dynamic-slice(get-tuple-element.2, select.0, constant.17), dynamic_slice_sizes={8,1024}
      constant.20 = bf16[]{:T(256)} constant(0)
      broadcast.3 = bf16[8,1024]{1,0:T(8,128)(2,1)} broadcast(constant.20), dimensions={}
      compare.2 = pred[8,1024]{1,0:T(8,128)(4,1)} compare(dynamic-slice.0, broadcast.3), direction=GT
      constant.21 = pred[]{:T(512)} constant(false)
      reduce.0 = pred[]{:T(512)} reduce(compare.2, constant.21), dimensions={0,1}, to_apply=or_comp
      ROOT and.0 = pred[]{:T(512)} and(compare.0, reduce.0)
    }

    ENTRY main {
      constant.28 = s32[]{:T(128)} constant(0)
      constant.29 = bf16[]{:T(256)} constant(0)
      broadcast.4 = bf16[65536,1024]{1,0:T(8,128)(2,1)} broadcast(constant.29), dimensions={}
      param.2 = bf16[65536,1024]{1,0:T(8,128)(2,1)} parameter(0), sharding={devices=[8,1]<=[4,2]T(1,0)}
      tuple.2 = (s32[]{:T(128)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}) tuple(constant.28, broadcast.4, param.2)
      while.1 = (s32[]{:T(128)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}, bf16[65536,1024]{1,0:T(8,128)(2,1)}) while(tuple.2), condition=condition, body=body
      get-tuple-element.9 = bf16[65536,1024]{1,0:T(8,128)(2,1)} get-tuple-element(while.1), index=1
      ROOT custom-call.5 = bf16[65536,1024]{1,0:T(8,128)(2,1)} custom-call(get-tuple-element.9), custom_call_target="MoveToHost"
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();

  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    ASSERT_NE(instr->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(HostOffloaderTest, TanhOnHostMemory) {
  const absl::string_view hlo_string = R"(
    HloModule module, entry_computation_layout={(f32[1024]{0})->f32[1024]{0}}

    ENTRY main {
      param = f32[1024]{0} parameter(0)
      to_host = f32[1024]{0} custom-call(param), custom_call_target="MoveToHost"
      tanh = f32[1024]{0} tanh(to_host)
      ROOT to_device = f32[1024]{0} custom-call(tanh), custom_call_target="MoveToDevice"
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  HloInstruction* tanh = FindInstruction(module.get(), "tanh");
  EXPECT_TRUE(host_offload_utils::ComputeTypeIsHost(tanh));
}

TEST_F(HostOffloaderTest, DynamicSliceOnHostMemoryParamCopied) {
  const absl::string_view hlo_string = R"(
    HloModule module, entry_computation_layout={(f32[1024]{0}, s32[]{:T(128)})->f32[256]{0}}

    ENTRY main {
      param = f32[1024]{0} parameter(0)
      index = s32[]{:T(128)} parameter(1)
      to_host = f32[1024]{0} custom-call(param), custom_call_target="MoveToHost"
      dynamic_slice = f32[256]{0} dynamic-slice(to_host, index), dynamic_slice_sizes={256}
      tanh = f32[256]{0} tanh(dynamic_slice)
      ROOT to_device = f32[256]{0} custom-call(tanh), custom_call_target="MoveToDevice"
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  HloInstruction* tanh = FindInstruction(module.get(), "tanh");
  EXPECT_TRUE(host_offload_utils::ComputeTypeIsHost(tanh));
  HloInstruction* dynamic_slice =
      FindInstruction(module.get(), "dynamic_slice");
  EXPECT_TRUE(host_offload_utils::ComputeTypeIsHost(dynamic_slice));
  // Check memory spaces
  ASSERT_EQ(dynamic_slice->operand_count(), 2);
  HloInstruction* copy_of_param = dynamic_slice->mutable_operand(0);
  EXPECT_EQ(copy_of_param->opcode(), HloOpcode::kCopy);
  TestShapeHasMemorySpace(copy_of_param->shape(), Layout::kHostMemorySpace);
  // The below tests something which needn't always be true.
  // The current expected behavior of HostOffloader for this test is to detect
  // compute happening on data in host memory space, which is the ops
  // dynamic_slice and tanh. HostOffloader will mark these two as host compute.
  // The interesting thing here is that the index to the dynamic_slice has not
  // explicitly been moved to host memory space. The below check expects that
  // HostOffloader does not explicitly move the index to host memory space. If
  // HostOffloader changes to enable this, that is fine, I just wanted to make
  // sure that it doesn't happen by accident.
  HloInstruction* index = dynamic_slice->mutable_operand(1);
  EXPECT_EQ(index->opcode(), HloOpcode::kParameter);
  TestShapeHasMemorySpace(index->shape(), Layout::kDefaultMemorySpace);
}

TEST_F(HostOffloaderTest, DynamicSliceOnHostMemoryIndexCopied) {
  const absl::string_view hlo_string = R"(
    HloModule module, entry_computation_layout={(f32[1024]{0}, s32[]{:T(128)})->f32[256]{0}}

    ENTRY main {
      param = f32[1024]{0} parameter(0)
      index = s32[]{:T(128)} parameter(1)
      index_to_host = s32[]{:T(128)} custom-call(index), custom_call_target="MoveToHost"
      dynamic_slice = f32[256]{0} dynamic-slice(param, index_to_host), dynamic_slice_sizes={256}
      tanh = f32[256]{0} tanh(dynamic_slice)
      ROOT to_device = f32[256]{0} custom-call(tanh), custom_call_target="MoveToDevice"
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloader(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  HloInstruction* dynamic_slice =
      FindInstruction(module.get(), "dynamic_slice");
  EXPECT_TRUE(host_offload_utils::ComputeTypeIsHost(dynamic_slice));
  HloInstruction* tanh = FindInstruction(module.get(), "tanh");
  EXPECT_TRUE(host_offload_utils::ComputeTypeIsHost(tanh));
}

}  // namespace

}  // namespace xla
