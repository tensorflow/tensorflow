/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/stream_attribute_annotator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

auto MakeDeviceDescription() {
  stream_executor::DeviceDescription device_description{
      stream_executor::GpuDeviceInfoProto{}};
  device_description.set_threads_per_warp(32);
  return device_description;
}

class StreamAttributeAnnotatorTest : public HloHardwareIndependentTestBase {
 public:
  const se::DeviceDescription& device_description() const {
    return device_description_;
  }

 private:
  const se::DeviceDescription device_description_{MakeDeviceDescription()};
};

TEST_F(StreamAttributeAnnotatorTest, AllUsersAreAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[1] parameter(0)
    p2_32 = f32[1] parameter(1)
    add_32 = f32[1] add(p1_32, p2_32), backend_config={"operation_queue_id":"1", "wait_on_operation_queues":[]}
    exp_32 = f32[1] exponential(add_32)

    neg32 = f32[1] negate(add_32)
    ROOT add_out_32 = f32[1] add(neg32, exp_32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator{device_description()};
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* add = FindInstruction(module.get(), "add_32");
  for (auto user : add->users()) {
    // Every user should have an annotation.
    EXPECT_TRUE(user->has_backend_config());
    TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                            user->backend_config<GpuBackendConfig>());
    EXPECT_EQ(gpu_config.wait_on_operation_queues()[0], 1);
  }
}

TEST_F(StreamAttributeAnnotatorTest, MultipleStreamsAreCombined) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[1] parameter(0)
    p2_32 = f32[1] parameter(1)
    add_32 = f32[1] add(p1_32, p2_32), backend_config={"operation_queue_id":"1", "wait_on_operation_queues":[]}
    exp_32 = f32[1] exponential(p2_32), backend_config={"operation_queue_id":"2", "wait_on_operation_queues":[]}

    ROOT add_out_32 = f32[1] add(add_32, exp_32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator{device_description()};
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  // Root should wait on 2 streams.
  EXPECT_TRUE(root->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          root->backend_config<GpuBackendConfig>());
  std::vector<int64_t> expected_stream_ids = {1, 2};
  for (auto id : expected_stream_ids) {
    auto it = absl::c_find(gpu_config.wait_on_operation_queues(), id);
    EXPECT_NE(it, gpu_config.wait_on_operation_queues().end());
  }
}

TEST_F(StreamAttributeAnnotatorTest, GTEUserIsAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[16,32] parameter(0)
    p2_32 = f32[32,16] parameter(1)

    custom-call.3 = (f32[16,16], s8[1028]{0}) custom-call(p1_32, p2_32), custom_call_target="__cublas$gemm", backend_config={"operation_queue_id":"1","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT","grad_x":false,"grad_y":false}}
    get-tuple-element.24 = f32[16,16] get-tuple-element(custom-call.3), index=0

    exp_32 = f32[16,16] exponential(get-tuple-element.24)

    ROOT neg32 = f32[16,16] negate(exp_32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator{device_description()};
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* exp = FindInstruction(module.get(), "exp_32");
  EXPECT_TRUE(exp->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          exp->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.wait_on_operation_queues()[0], 1);
}

TEST_F(StreamAttributeAnnotatorTest, FusionIsAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithFusion

  fused_computation.1 {
    fusion_p0_32 = f32[16,16] parameter(0)
    fusion_p2_32 = f32[16,16] parameter(1)
    ROOT add = f32[16,16] add(fusion_p0_32, fusion_p2_32), backend_config={"operation_queue_id":"1","wait_on_operation_queues":[]}
  }

  ENTRY entry {
    p1_32 = f32[16,16] parameter(0)
    p2_32 = f32[16,16] parameter(1)
    ROOT fusion.1 = f32[16,16] fusion(p1_32, p2_32), kind=kLoop, calls=fused_computation.1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator{device_description()};
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* fusion = FindInstruction(module.get(), "fusion.1");
  EXPECT_TRUE(fusion->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.operation_queue_id(), 1);
}

TEST_F(StreamAttributeAnnotatorTest, CopyStartIsAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule offloading, is_scheduled=true
    ENTRY %main (param_0: f32[1024], param_1: f32[1024]) -> f32[1024] {
    %param_1 = f32[1024]{0} parameter(1)
    %param_0 = f32[1024]{0} parameter(0)
    %res_3 = f32[1024]{0} add(f32[1024]{0} %param_0, f32[1024]{0} %param_1)
    %copy-start = (f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) copy-start(f32[1024]{0} %res_3)
    %res_4 = f32[1024]{0} tanh(f32[1024]{0} %res_3)
    %copy-start.2 = (f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) copy-start(f32[1024]{0} %res_4)
    %res_5 = f32[1024]{0} tanh(f32[1024]{0} %res_4)
    %copy-done = f32[1024]{0:S(5)} copy-done((f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) %copy-start)
    %res_6 = f32[1024]{0} tanh(f32[1024]{0} %res_5)
    %copy-done.2 = f32[1024]{0:S(5)} copy-done((f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) %copy-start.2)
    %copy-start.3 = (f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) copy-start(f32[1024]{0:S(5)} %copy-done.2)
    %res_7 = f32[1024]{0} add(f32[1024]{0} %res_6, f32[1024]{0} %res_6)
    %copy-start.1 = (f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) copy-start(f32[1024]{0:S(5)} %copy-done)
    %res_8 = f32[1024]{0} add(f32[1024]{0} %res_7, f32[1024]{0} %res_5)
    %copy-done.3 = f32[1024]{0} copy-done((f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) %copy-start.3)
    %res_9 = f32[1024]{0} add(f32[1024]{0} %res_8, f32[1024]{0} %copy-done.3)
    %copy-done.1 = f32[1024]{0} copy-done((f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) %copy-start.1)
    %res_10 = f32[1024]{0} add(f32[1024]{0} %res_9, f32[1024]{0} %copy-done.1)
    ROOT %res_11 = f32[1024]{0} tanh(f32[1024]{0} %res_10)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator{device_description()};
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  for (std::string i : {"", ".1", ".2", ".3"}) {
    const HloInstruction* cp_start =
        FindInstruction(module.get(), "copy-start" + i);
    EXPECT_TRUE(cp_start->has_backend_config());
    TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                            cp_start->backend_config<GpuBackendConfig>());
    EXPECT_EQ(gpu_config.operation_queue_id(), 1);
  }
}

TEST_F(StreamAttributeAnnotatorTest, DynamicUpdateSliceWrappedAndAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsyncDynamicUpdateSlice, is_scheduled=true

  ENTRY entry (param_0: f32[256,128,128], param_1: f32[1,128,128]) -> f32[256,128,128] {
    param_0 = f32[256,128,128]{2,1,0:S(5)} parameter(0), metadata={scheduling_name="param_0"}
    param_1 = f32[1,128,128]{2,1,0} parameter(1), metadata={scheduling_name="param_1"}
    izero = s32[] constant(0), metadata={scheduling_name="izero"}
    dynamic-update-slice-start.2 = ((f32[256,128,128]{2,1,0:S(5)}, f32[1,128,128]{2,1,0}, s32[], s32[], s32[]), f32[256,128,128]{2,1,0:S(5)}, u32[])
        dynamic-update-slice-start(param_0, param_1, izero, izero, izero),
        metadata={scheduling_name="dynamic-update-slice-start.2"}
    ROOT dynamic-update-slice-done.2 = f32[256,128,128]{2,1,0:S(5)}
        dynamic-update-slice-done(dynamic-update-slice-start.2),
        metadata={scheduling_name="dynamic-update-slice-done.2"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  EXPECT_TRUE(module->has_schedule());

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      StreamAttributeAnnotator(device_description()).Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the dynamic-update-slice instruction is wrapped in a fusion
  // and the fusion is annotated with the correct operation_queue_id.
  const HloInstruction* dus =
      FindInstruction(module.get(), HloOpcode::kDynamicUpdateSlice);
  const HloComputation* computation = dus->parent();
  EXPECT_TRUE(computation->IsFusionComputation());
  const HloInstruction* fusion = computation->FusionInstruction();
  EXPECT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(fusion->parent()->IsAsyncComputation());

  EXPECT_TRUE(fusion->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.operation_queue_id(), 1);
  // Check if the schedule name the same as the instruction name
  for (const auto* comp : module->computations()) {
    for (const auto* instruction : comp->instructions()) {
      if (!instruction->metadata().scheduling_name().empty()) {
        EXPECT_EQ(instruction->name(),
                  instruction->metadata().scheduling_name());
      }
    }
  }
  constexpr absl::string_view kExpectedSchedulingName = R"(
// CHECK: %wrapped_dynamic-update-slice_computation
// CHECK: ROOT %[[DYNAMIC_UPDATE_SLICE:.+]] = f32[256,128,128]{2,1,0:S(5)} dynamic-update-slice(
// CHECK-SAME: metadata={scheduling_name="[[DYNAMIC_UPDATE_SLICE]]"}
// CHECK: %[[DYNAMIC_UPDATE_SLICE_START:.+]] = {{.*}} fusion-start(
// CHECK-SAME: calls=%wrapped_dynamic-update-slice_computation
// CHECK-SAME: metadata={scheduling_name="[[DYNAMIC_UPDATE_SLICE_START]]"}
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(false)),
          kExpectedSchedulingName));
  EXPECT_TRUE(filecheck_matches);
}

TEST_F(StreamAttributeAnnotatorTest, DynamicSliceWrappedAndAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsyncDynamicSlice, is_scheduled=true

  ENTRY entry (param_0: f32[256,128,128]) -> f32[1,128,128] {
    param_0 = f32[256,128,128]{2,1,0:S(5)} parameter(0), metadata={scheduling_name="param_0"}
    izero = s32[] constant(0), metadata={scheduling_name="izero"}
    dynamic-slice-start.2 = ((f32[256,128,128]{2,1,0:S(5)}, s32[], s32[], s32[]), f32[1,128,128]{2,1,0}, u32[])
        dynamic-slice-start(param_0, izero, izero, izero), dynamic_slice_sizes={1,128,128},
        metadata={scheduling_name="dynamic-slice-start.2"}
    ROOT dynamic-slice-done.2 = f32[1,128,128]{2,1,0}
        dynamic-slice-done(dynamic-slice-start.2),
        metadata={scheduling_name="dynamic-slice-done.2"}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      StreamAttributeAnnotator(device_description()).Run(module.get()));
  EXPECT_TRUE(changed);

  // Check that the dynamic-slice instruction is wrapped in a fusion
  // and the fusion is annotated with the correct operation_queue_id.
  const HloInstruction* ds =
      FindInstruction(module.get(), HloOpcode::kDynamicSlice);
  const HloComputation* computation = ds->parent();
  EXPECT_TRUE(computation->IsFusionComputation());
  const HloInstruction* fusion = computation->FusionInstruction();
  EXPECT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(fusion->parent()->IsAsyncComputation());

  EXPECT_TRUE(fusion->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.operation_queue_id(), 1);
  // Check if the schedule name the same as the instruction name
  for (const auto* comp : module->computations()) {
    for (const auto* instruction : comp->instructions()) {
      if (!instruction->metadata().scheduling_name().empty()) {
        EXPECT_EQ(instruction->name(),
                  instruction->metadata().scheduling_name());
      }
    }
  }
  constexpr absl::string_view kExpectedSchedulingName = R"(
// CHECK: %wrapped_dynamic-slice_computation
// CHECK: ROOT %[[DYNAMIC_SLICE:.+]] = f32[1,128,128]{2,1,0} dynamic-slice(
// CHECK-SAME: metadata={scheduling_name="[[DYNAMIC_SLICE]]"}
// CHECK: %[[DYNAMIC_SLICE_START:.+]] = {{.*}} fusion-start(
// CHECK-SAME: calls=%wrapped_dynamic-slice_computation
// CHECK-SAME: metadata={scheduling_name="[[DYNAMIC_SLICE_START]]"}
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(false)),
          kExpectedSchedulingName));
  EXPECT_TRUE(filecheck_matches);
}
}  // namespace
}  // namespace xla::gpu
