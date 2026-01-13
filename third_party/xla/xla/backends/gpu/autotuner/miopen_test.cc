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

#include "xla/backends/gpu/autotuner/miopen.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/amdgpu_compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using MIOpenBackendConfig = stream_executor::dnn::AlgorithmProto;

using absl_testing::IsOkAndHolds;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

const char kMIOpenCustomCallHlo[] = R"(
  HloModule module

  ENTRY %main {
    %arg0 = f32[3,56,56,16]{2,1,0,3} parameter(0)
    %arg1 = f32[3,3,3,64]{2,1,0,3} parameter(1)
    %cudnn-conv = (f32[54,54,16,64]{1,0,3,2}, u8[0]{0})
      custom-call(%arg0, %arg1), custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=f01b_i01o->01bf,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        },
      }
    ROOT %get-tuple-element = f32[54,54,16,64]{1,0,3,2} get-tuple-element(%cudnn-conv), index=0
  })";

class MIOpenBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  AMDGPUCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  MIOpenBackend backend_;

  MIOpenBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(stream_executor_, &debug_options_, &compiler_,
                 &target_config_) {}

  bool IsRocm() {
    return stream_executor_->GetPlatform()->id() == se::rocm::kROCmPlatformId;
  }
};

TEST_F(MIOpenBackendTest, CanCreateMIOpenBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(MIOpenBackendTest, GetSupportedConfigsFromMIOpenCustomCall) {
  if (!IsRocm()) {
    GTEST_SKIP() << "Skipping test on non-ROCm platform";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kMIOpenCustomCallHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  ASSERT_THAT(configs, IsOkAndHolds(SizeIs(1)));
  MIOpenBackendConfig algorithm_config;
  ASSERT_TRUE((*configs)[0]->UnpackTo(&algorithm_config));
  EXPECT_NE(algorithm_config.algo_id(), 0);
}

TEST_F(MIOpenBackendTest, GetDefaultConfigFromMIOpenCustomCall) {
  if (!IsRocm()) {
    GTEST_SKIP() << "Skipping test on non-ROCm platform";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kMIOpenCustomCallHlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  TF_ASSERT_OK(config);
  MIOpenBackendConfig algorithm_config;
  ASSERT_TRUE(config->get()->UnpackTo(&algorithm_config));
  EXPECT_EQ(algorithm_config.algo_id(), 0);
}

TEST_F(MIOpenBackendTest, ApplyConfigToMIOpenCustomCall) {
  if (!IsRocm()) {
    GTEST_SKIP() << "Skipping test on non-ROCm platform";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kMIOpenCustomCallHlo));
  MIOpenBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_.ApplyConfig(*instr, any));
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
}

TEST_F(MIOpenBackendTest, ApplyConfigToMIOpenCustomCallWithWorkspace) {
  if (!IsRocm()) {
    GTEST_SKIP() << "Skipping test on non-ROCm platform";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kMIOpenCustomCallHlo));
  MIOpenBackendConfig config;
  config.set_algo_id(1);
  config.mutable_workspace_size()->set_value(1024);
  HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_ASSERT_OK(backend_.ApplyConfig(*instr, any));

  auto* replaced_instr =
      hlo_module->entry_computation()->GetInstructionWithName("cudnn-conv");

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          replaced_instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
  EXPECT_EQ(replaced_instr->shape().tuple_shapes(1).dimensions(0), 1024);
}

}  // namespace gpu
}  // namespace xla
