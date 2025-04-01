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

#include "xla/hlo/transforms/host_offload_legalize.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class HostOffloadLegalizeTest : public HloHardwareIndependentTestBase {
 protected:
  static constexpr int64_t kHostMemorySpaceColor{5};

  absl::StatusOr<bool> RunHostOffloadLegalize(HloModule* module) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (module->has_schedule()) {
      return absl::InternalError("Expected a non-scheduled module");
    }
    HostOffloadLegalize host_offload_legalize(kHostMemorySpaceColor,
                                              /*after_layout=*/true);
    return host_offload_legalize.Run(module);
  }

  void TestShapeHasMemorySpace(const Shape& shape, int64_t memory_space) {
    ASSERT_TRUE(shape.has_layout());
    EXPECT_EQ(shape.layout().memory_space(), memory_space);
  }

  bool HaveRemainingOffloadAnnotations(const HloModule* module) {
    for (const HloComputation* computation : module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->IsCustomCall(
                {memory_annotations::kMoveToHostCustomCallTarget,
                 memory_annotations::kMoveToDeviceCustomCallTarget})) {
          return true;
        }
      }
    }
    return false;
  }
};

TEST_F(HostOffloadLegalizeTest, TestWithAsyncCall) {
  const std::string& hlo_string = R"(
HloModule jit_update, entry_computation_layout={(f32[20,3,256,133]{2,3,1,0:T(8,128)S(5)})->(f32[20,3,256,133]{2,1,0,3:T(4,128)}, f32[4096]{0:T(1024)})}

%async_computation {
  %param_0 = f32[20,3,256,133] parameter(0)
  ROOT %offloaded-custom-call = f32[4096] custom-call(%param_0), custom_call_target="HostExecute"
}, execution_thread="host"

ENTRY main {
  %param.246 = f32[20,3,256,133] parameter(0)
  %async-start = ((f32[20,3,256,133]), f32[4096], u32[]) async-start(%param.246), async_execution_thread="host", calls=%async_computation
  %async-done = f32[4096] custom-call-done(%async-start)
  copy.16744 = f32[20,3,256,133]{2,1,0,3:T(4,128)} copy(param.246)
  custom-call.7832 = f32[20,3,256,133]{2,1,0,3:T(4,128)} custom-call(copy.16744), custom_call_target="MoveToDevice"
  ROOT tuple.16745 = (f32[20,3,256,133]{2,1,0,3:T(4,128)}, f32[4096]{0:T(1024)}) tuple(custom-call.7832, %async-done)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* custom_call =
      FindInstruction(module.get(), "custom-call.7832");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(HostOffloadLegalizeTest, NoCopyWithOptBarrierMoreElaborate) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1})->f32[16,256]{1,0}}

ENTRY main.24 {
  Arg_0.1 = f32[16,256]{0,1} parameter(0)
  cosine.4 = f32[16,256]{0,1} cosine(Arg_0.1)
  custom-call.5 = f32[16,256]{0,1} custom-call(cosine.4), custom_call_target="MoveToHost"
  sine.3 = f32[16,256]{0,1} sine(Arg_0.1)
  cosine.7 = f32[16,256]{0,1} cosine(sine.3)
  custom-call.8 = f32[16,256]{0,1} custom-call(cosine.7), custom_call_target="MoveToHost"
  sine.6 = f32[16,256]{0,1} sine(sine.3)
  cosine.9 = f32[16,256]{0,1} cosine(sine.6)
  custom-call.10 = f32[16,256]{0,1} custom-call(cosine.9), custom_call_target="MoveToHost"
  constant.2 = f32[] constant(1)
  cp = f32[16,256]{1,0} copy(custom-call.8)
  tuple.11 = (f32[16,256]{0,1}, f32[16,256]{1,0}, f32[16,256]{0,1}, f32[]) tuple(custom-call.5, cp, custom-call.10, constant.2)
  opt-barrier.12 = (f32[16,256]{0,1}, f32[16,256]{1,0}, f32[16,256]{0,1}, f32[]) opt-barrier(tuple.11)
  get-tuple-element.16 = f32[] get-tuple-element(opt-barrier.12), index=3
  broadcast.20 = f32[16,256]{0,1} broadcast(get-tuple-element.16), dimensions={}
  get-tuple-element.15 = f32[16,256]{0,1} get-tuple-element(opt-barrier.12), index=2
  custom-call.19 = f32[16,256]{0,1} custom-call(get-tuple-element.15), custom_call_target="MoveToDevice"
  multiply.21 = f32[16,256]{0,1} multiply(broadcast.20, custom-call.19)
  cp2 = f32[16,256]{1,0} copy(multiply.21)
  get-tuple-element.14 = f32[16,256]{1,0} get-tuple-element(opt-barrier.12), index=1
  custom-call.18 = f32[16,256]{1,0} custom-call(get-tuple-element.14), custom_call_target="MoveToDevice"
  multiply.22 = f32[16,256]{1,0} multiply(cp2, custom-call.18)
  get-tuple-element.13 = f32[16,256]{0,1} get-tuple-element(opt-barrier.12), index=0
  custom-call.17 = f32[16,256]{0,1} custom-call(get-tuple-element.13), custom_call_target="MoveToDevice"
  cp3 = f32[16,256]{1,0} copy(custom-call.17)
  ROOT multiply.23 = f32[16,256]{1,0} multiply(multiply.22, cp3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());

  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call.18");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(), LayoutUtil::MakeLayout({0, 1}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));
}

TEST_F(HostOffloadLegalizeTest, XposeCopyOnParameterStreaming) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16,256]{0,1},f32[16,256]{0,1:T(8,128)S(5)})->f32[16,256]{1,0}}

ENTRY main.24 {
  Arg_0.1 = f32[16,256]{0,1} parameter(0)
  Arg_0.2 = f32[16,256]{0,1:T(8,128)} parameter(1)
  cp0 = f32[16,256]{1,0} copy(Arg_0.2)
  cosine.4 = f32[16,256]{0,1} cosine(Arg_0.1)
  custom-call.5 = f32[16,256]{0,1} custom-call(cosine.4), custom_call_target="MoveToHost"
  sine.3 = f32[16,256]{0,1} sine(Arg_0.1)
  cosine.7 = f32[16,256]{0,1} cosine(sine.3)
  custom-call.8 = f32[16,256]{0,1} custom-call(cosine.7), custom_call_target="MoveToHost"
  constant.2 = f32[] constant(1)
  cp1 = f32[16,256]{1,0} copy(custom-call.8)
  tuple.11 = (f32[16,256]{0,1}, f32[16,256]{1,0}, f32[16,256]{1,0}, f32[]) tuple(custom-call.5, cp1, cp0, constant.2)
  opt-barrier.12 = (f32[16,256]{0,1}, f32[16,256]{1,0}, f32[16,256]{1,0}, f32[]) opt-barrier(tuple.11)
  get-tuple-element.16 = f32[] get-tuple-element(opt-barrier.12), index=3
  broadcast.20 = f32[16,256]{0,1} broadcast(get-tuple-element.16), dimensions={}
  get-tuple-element.15 = f32[16,256]{1,0} get-tuple-element(opt-barrier.12), index=2
  custom-call.19 = f32[16,256]{1,0} custom-call(get-tuple-element.15), custom_call_target="MoveToDevice"
  multiply.21 = f32[16,256]{0,1} multiply(broadcast.20, custom-call.19)
  cp2 = f32[16,256]{1,0} copy(multiply.21)
  get-tuple-element.14 = f32[16,256]{1,0} get-tuple-element(opt-barrier.12), index=1
  custom-call.18 = f32[16,256]{1,0} custom-call(get-tuple-element.14), custom_call_target="MoveToDevice"
  multiply.22 = f32[16,256]{1,0} multiply(cp2, custom-call.18)
  get-tuple-element.13 = f32[16,256]{0,1} get-tuple-element(opt-barrier.12), index=0
  custom-call.17 = f32[16,256]{0,1} custom-call(get-tuple-element.13), custom_call_target="MoveToDevice"
  cp3 = f32[16,256]{1,0} copy(custom-call.17)
  ROOT multiply.23 = f32[16,256]{1,0} multiply(multiply.22, cp3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());

  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call.18");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(), LayoutUtil::MakeLayout({0, 1}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));

  custom_call = FindInstruction(module.get(), "custom-call.19");
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({0, 1}, {}, {}, {}, {Tile{{8, 128}}}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));
}

TEST_F(HostOffloadLegalizeTest, DUSSameLayoutForOperandAndUpdate_1) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(bf16[16,512,532]{1,2,0})->bf16[1,16,512,532]{2,3,1,0}}

ENTRY main.24 {
  constant_0 = s32[] constant(0)
  cs0 = bf16[] constant(0)
  broadcast = bf16[20,16,512,532]{3,2,1,0}  broadcast(cs0), dimensions={}
  cp = bf16[20,16,512,532]{3,2,1,0} copy(broadcast)
  custom-call.8 = bf16[20,16,512,532]{3,2,1,0} custom-call(cp), custom_call_target="MoveToHost"
  copy = bf16[20,16,512,532]{2,3,1,0} copy(custom-call.8)
  arg1 = bf16[16,512,532]{1,2,0} parameter(0)
  copy.17302 = bf16[16,512,532]{2,1,0} copy(arg1)
  bitcast.6100 = bf16[1,16,512,532]{3,2,1,0} bitcast(copy.17302)
  copy.20241 = bf16[1,16,512,532]{2,3,1,0} copy(bitcast.6100)
  custom-call.6720 = bf16[1,16,512,532]{2,3,1,0} custom-call(copy.20241), custom_call_target="MoveToHost"
  dynamic-update-slice.6830 = bf16[20,16,512,532]{2,3,1,0} dynamic-update-slice(copy, custom-call.6720, constant_0, constant_0, constant_0, constant_0)
  dynamic_slice_0 = bf16[1,16,512,532]{2,3,1,0} dynamic-slice(dynamic-update-slice.6830, constant_0, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,16,512,532}
  ROOT custom_call_0.1 = bf16[1,16,512,532]{2,3,1,0} custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());

  HloInstruction* dus =
      FindInstruction(module.get(), "dynamic-update-slice.6830");
  ASSERT_NE(dus, nullptr);
  EXPECT_EQ(dus->operand(0)->shape().layout(),
            dus->operand(1)->shape().layout());
  EXPECT_EQ(dus->shape().layout(), dus->operand(1)->shape().layout());

  const HloInstruction* custom_call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_TRUE(custom_call->IsCustomCall(
      memory_annotations::kMoveToDeviceCustomCallTarget));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({2, 3, 1, 0}));
}

TEST_F(HostOffloadLegalizeTest, DUSSameLayoutForOperandAndUpdate_2) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(bf16[16,512,532]{1,2,0})->bf16[1,16,512,532]{2,3,1,0}}

ENTRY main.24 {
  constant_0 = s32[] constant(0)
  cs0 = bf16[] constant(0)
  broadcast = bf16[20,16,512,532]{3,2,1,0}  broadcast(cs0), dimensions={}
  cp = bf16[20,16,512,532]{3,2,1,0} copy(broadcast)
  custom-call.8 = bf16[20,16,512,532]{3,2,1,0} custom-call(cp), custom_call_target="MoveToHost"
  copy = bf16[20,16,512,532]{2,3,1,0} copy(custom-call.8)
  arg1 = bf16[16,512,532]{1,2,0} parameter(0)
  copy.17302 = bf16[16,512,532]{2,1,0} copy(arg1)
  custom-call.6720 = bf16[16,512,532]{2,1,0} custom-call(copy.17302), custom_call_target="MoveToHost"
  bitcast.6100 = bf16[1,16,512,532]{3,2,1,0} bitcast(custom-call.6720)
  copy.20241 = bf16[1,16,512,532]{2,3,1,0} copy(bitcast.6100)
  dynamic-update-slice.6830 = bf16[20,16,512,532]{2,3,1,0} dynamic-update-slice(copy, copy.20241, constant_0, constant_0, constant_0, constant_0)
  dynamic_slice_0 = bf16[1,16,512,532]{2,3,1,0} dynamic-slice(dynamic-update-slice.6830, constant_0, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,16,512,532}
  ROOT custom_call_0.1 = bf16[1,16,512,532]{2,3,1,0} custom-call(dynamic_slice_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());

  HloInstruction* dus =
      FindInstruction(module.get(), "dynamic-update-slice.6830");
  ASSERT_NE(dus, nullptr);
  EXPECT_EQ(dus->operand(0)->shape().layout(),
            dus->operand(1)->shape().layout());
  EXPECT_EQ(dus->shape().layout(), dus->operand(1)->shape().layout());

  const HloInstruction* custom_call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_TRUE(custom_call->IsCustomCall(
      memory_annotations::kMoveToDeviceCustomCallTarget));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}));
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({2, 3, 1, 0}));
}

TEST_F(HostOffloadLegalizeTest, LlmActivationHostMemoryMultipleConsumers) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048]{0,1,2,3,4} get-tuple-element(input_tuple.0), index=1
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

  dynamic_update_slice_0 = f32[96,8,6,2048,2048]{0,1,2,3,4} dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) tuple(incremented_index.0, dynamic_update_slice_0)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048]{0,1,3,2,4} get-tuple-element(input_tuple.1), index=1
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
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) tuple(incremented_index.1, data_0.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  entry_param_1 = s32[] parameter(1)
  entry_param_2 = s32[] parameter(2)
  cs0 = f32[] constant(0)
  broadcast_0 = f32[96,8,6,2048,2048]{0,1,2,3,4} broadcast(cs0), dimensions={}
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) tuple(constant_s32_0, broadcast_0)
  producing_while = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048]{0,1,2,3,4} get-tuple-element(producing_while), index=1
  cp = f32[96,8,6,2048,2048]{0,1,3,2,4} copy(while_output_1)
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) tuple(constant_s32_0, cp)
  consuming_while = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  second_while_output = f32[96,8,6,2048,2048]{0,1,3,2,4} get-tuple-element(consuming_while), index=1
  final_dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(second_while_output, entry_param_1, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,8,6,2048,2048}
  final_host_to_device_custom_call_0 = f32[1,8,6,2048,2048] custom-call(final_dynamic_slice_0), custom_call_target="MoveToDevice"
  final_slice_0 = f32[1,8,6,2048,2048] slice(second_while_output), slice={[41:42], [0:8], [0:6], [0:2048], [0:2048]}
  final_host_to_device_custom_call_1 = f32[1,8,6,2048,2048] custom-call(final_slice_0), custom_call_target="MoveToDevice"
  ROOT add = f32[1,8,6,2048,2048] add(final_host_to_device_custom_call_0, final_host_to_device_custom_call_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  EXPECT_TRUE(changed);
  HloInstruction* copy = FindInstruction(module.get(), HloOpcode::kCopy);
  HloInstruction* consuming_while =
      FindInstruction(module.get(), "consuming_while");
  ASSERT_NE(copy, nullptr);
  ASSERT_NE(consuming_while, nullptr);
  EXPECT_NE(copy, nullptr);
  EXPECT_NE(consuming_while, nullptr);
  EXPECT_EQ(copy->parent(), consuming_while->while_body());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(HostOffloadLegalizeTest, LlmActivationHostMemoryMultipleCopies) {
  const std::string& hlo_string = R"(
HloModule llm_while

producing_while_condition {
  producing_condition_param = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) parameter(0)
  producing_condition_current_iteration_index = s32[] get-tuple-element(producing_condition_param), index=0
  producing_condition_iteration_count = s32[] constant(96)
  ROOT producing_condition_result = pred[] compare(producing_condition_current_iteration_index, producing_condition_iteration_count), direction=LT
}

consuming_while_condition {
  consuming_condition_param = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) parameter(0)
  consuming_condition_current_iteration_index = s32[] get-tuple-element(consuming_condition_param), index=0
  consuming_condition_iteration_count = s32[] constant(96)
  ROOT consuming_condition_result = pred[] compare(consuming_condition_current_iteration_index, consuming_condition_iteration_count), direction=LT
}

producing_while_body {
  input_tuple.0 = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) parameter(0)
  current_iteration_index.0 = s32[] get-tuple-element(input_tuple.0), index=0
  data_0.0 = f32[96,8,6,2048,2048]{0,1,2,3,4} get-tuple-element(input_tuple.0), index=1
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

  dynamic_update_slice_0 = f32[96,8,6,2048,2048]{0,1,2,3,4} dynamic-update-slice(data_0.0, custom_call_0.0, select_result.0, constant_0.0, constant_0.0, constant_0.0, constant_0.0)

  /* Increment iteration index */
  incremented_index.0 = s32[] add(current_iteration_index.0, constant_1.0)
  ROOT tuple_result.0 = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) tuple(incremented_index.0, dynamic_update_slice_0)
}

consuming_while_body {
  input_tuple.1 = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) parameter(0)
  current_iteration_index.1 = s32[] get-tuple-element(input_tuple.1), index=0
  data_0.1 = f32[96,8,6,2048,2048]{0,1,3,2,4} get-tuple-element(input_tuple.1), index=1
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
  ROOT tuple_result.1 = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) tuple(incremented_index.1, data_0.1)
}

ENTRY main {
  entry_param_0 = f32[] parameter(0)
  entry_param_1 = s32[] parameter(1)
  entry_param_2 = s32[] parameter(2)
  cs0 = f32[] constant(0)
  broadcast_0 = f32[96,8,6,2048,2048]{0,1,2,3,4} broadcast(cs0), dimensions={}
  constant_s32_0 = s32[] constant(0)
  tuple_for_producing_while = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) tuple(constant_s32_0, broadcast_0)
  producing_while = (s32[], f32[96,8,6,2048,2048]{0,1,2,3,4}) while(tuple_for_producing_while), condition=producing_while_condition, body=producing_while_body
  while_output_1 = f32[96,8,6,2048,2048]{0,1,2,3,4} get-tuple-element(producing_while), index=1
  cp = f32[96,8,6,2048,2048]{0,1,3,2,4} copy(while_output_1)
  cp1 = f32[96,8,6,2048,2048]{0,1,3,2,4} copy(cp)
  tuple_for_consuming_while = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) tuple(constant_s32_0, cp1)
  consuming_while = (s32[], f32[96,8,6,2048,2048]{0,1,3,2,4}) while(tuple_for_consuming_while), condition=consuming_while_condition, body=consuming_while_body
  second_while_output = f32[96,8,6,2048,2048]{0,1,3,2,4} get-tuple-element(consuming_while), index=1
  final_dynamic_slice_0 = f32[1,8,6,2048,2048] dynamic-slice(second_while_output, entry_param_1, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,8,6,2048,2048}
  final_host_to_device_custom_call_0 = f32[1,8,6,2048,2048] custom-call(final_dynamic_slice_0), custom_call_target="MoveToDevice"
  final_slice_0 = f32[1,8,6,2048,2048] slice(second_while_output), slice={[41:42], [0:8], [0:6], [0:2048], [0:2048]}
  final_host_to_device_custom_call_1 = f32[1,8,6,2048,2048] custom-call(final_slice_0), custom_call_target="MoveToDevice"
  ROOT add = f32[1,8,6,2048,2048] add(final_host_to_device_custom_call_0, final_host_to_device_custom_call_1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  EXPECT_TRUE(changed);
  HloInstruction* copy_0 = FindInstruction(module.get(), "cp.2");
  HloInstruction* copy_1 = FindInstruction(module.get(), "cp1.2");
  HloInstruction* consuming_while =
      FindInstruction(module.get(), "consuming_while");
  ASSERT_NE(copy_0, nullptr);
  ASSERT_NE(copy_1, nullptr);
  ASSERT_NE(consuming_while, nullptr);
  EXPECT_NE(copy_0, nullptr);
  EXPECT_NE(copy_1, nullptr);
  EXPECT_NE(consuming_while, nullptr);
  EXPECT_EQ(copy_0->parent(), module->entry_computation());
  EXPECT_EQ(copy_1->operand(0), copy_0);
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(HostOffloadLegalizeTest, MoveCopyOverBitcast) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(bf16[1,1,16384,4,256]{4,3,2,1,0:T(4,128)(2,1)S(5)})->bf16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)}}

ENTRY main {
  param = bf16[1,1,16384,4,256]{4,3,2,1,0:T(4,128)(2,1)} parameter(0)
  copy = bf16[1,1,16384,4,256]{4,2,3,1,0:T(8,128)(2,1)} copy(param)
  bitcast = bf16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)} bitcast(copy)
  custom-call = bf16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)} custom-call(bitcast), custom_call_target="MoveToDevice"
  ROOT add = bf16[1,16384,4,256]{3,1,2,0:T(8,128)(2,1)} add(custom-call, custom-call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call");
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}, {}, {}, {},
                                   {Tile{{4, 128}}, Tile{{2, 1}}}));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({3, 1, 2, 0}, {}, {}, {},
                                   {Tile{{8, 128}}, Tile{{2, 1}}}));
}

TEST_F(HostOffloadLegalizeTest, MoveCopyOverBitcast_2) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(bf16[1,16384,4,256]{3,2,1,0:T(4,128)(2,1)S(5)})->bf16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)}}

ENTRY main {
  param = bf16[1,16384,4,256]{3,2,1,0:T(4,128)(2,1)} parameter(0)
  copy = bf16[1,16384,4,256]{2,3,1,0:T(8,128)(2,1)} copy(param)
  bitcast = bf16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)} bitcast(copy)
  custom-call = bf16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)} custom-call(bitcast), custom_call_target="MoveToDevice"
  ROOT add = bf16[1,1,16384,4,256]{4,3,1,2,0:T(8,128)(2,1)} add(custom-call, custom-call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  HloInstruction* custom_call = FindInstruction(module.get(), "custom-call");
  EXPECT_EQ(custom_call->shape().layout(),
            LayoutUtil::MakeLayout({4, 3, 2, 1, 0}, {}, {}, {},
                                   {Tile{{4, 128}}, Tile{{2, 1}}}));
  EXPECT_EQ(custom_call->users()[0]->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(custom_call->users()[0]->shape().layout(),
            LayoutUtil::MakeLayout({3, 4, 2, 1, 0}, {}, {}, {},
                                   {Tile{{8, 128}}, Tile{{2, 1}}}));
}

TEST_F(HostOffloadLegalizeTest, MoveCopyUp) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(bf16[4,4,128,8]{2,0,1,3:T(8,128)(2,1)})->bf16[4,4,128,8]{2,3,1,0:T(8,128)(2,1)S(5)}}

ENTRY main {
  param = bf16[4,4,128,8]{2,0,1,3:T(8,128)(2,1)} parameter(0)
  custom_call = bf16[4,4,128,8]{2,0,1,3:T(8,128)(2,1)} custom-call(param), custom_call_target="MoveToHost"
  ROOT copy = bf16[4,4,128,8]{2,3,1,0:T(8,128)(2,1)S(5)} copy(custom_call)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  ASSERT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  // Check that the custom call and copy have swapped places.
  HloInstruction* custom_call = FindInstruction(module.get(), "custom_call");
  EXPECT_TRUE(custom_call->IsRoot());
  EXPECT_TRUE(custom_call->IsCustomCall(
      memory_annotations::kMoveToHostCustomCallTarget));
  const HloInstruction* copy = custom_call->operand(0);
  ASSERT_EQ(copy->opcode(), HloOpcode::kCopy);
  const HloInstruction* param = copy->operand(0);
  EXPECT_EQ(copy->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(param->opcode(), HloOpcode::kParameter);
}

// Check that HostOffloadLegalize doesn't crash when the base operand of the
// dynamic-update-slice is a parameter.
TEST_F(HostOffloadLegalizeTest, NoCrashBaseIsStreamed) {
  const std::string& hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[8,512,64]{1,2,0:T(8,128)S(5)}, f32[1,512,64]{1,2,0:T(8,128)}, s32[]{:T(128)})->f32[8,512,64]{1,2,0:T(8,128)S(5)}}

ENTRY main {
  param1 = f32[8,512,64]{1,2,0:T(8,128)S(5)} parameter(0)
  param2 = f32[1,512,64]{1,2,0:T(8,128)} parameter(1)
  custom_call = f32[1,512,64]{1,2,0:T(8,128)S(5)} custom-call(param2), custom_call_target="MoveToHost"
  param3 = s32[]{:T(128)} parameter(2)
  constant = s32[]{:T(128)} constant(0)
  ROOT dynamic-update-slice = f32[8,512,64]{1,2,0:T(8,128)} dynamic-update-slice(param1, param2, param3, constant, constant)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHostOffloadLegalize(module.get()));

  ASSERT_FALSE(changed);
}

}  // namespace

}  // namespace xla
