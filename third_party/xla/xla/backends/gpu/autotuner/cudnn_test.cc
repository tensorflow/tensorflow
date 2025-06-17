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

#include "xla/backends/gpu/autotuner/cudnn.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

using ::testing::Gt;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

const char kCudnnFusionHlo[] = R"(
  fusion1 {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    ROOT d = f32[3,32,32] dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  ENTRY e {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    ROOT _ = f32[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
  })";

const char kCudnnCustomCallHlo[] = R"(
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

const char kUnsupportedHlo[] = R"(
  HloModule module

  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

class CudnnBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  CudnnBackend backend_;

  CudnnBackendTest()
      : debug_options_([]() {
          DebugOptions debug_options;
          debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
          return debug_options;
        }()),
        backend_(PlatformUtil::GetDefaultPlatform()
                     .value()
                     ->ExecutorForDevice(0)
                     .value(),
                 &debug_options_, &compiler_) {}
};

TEST_F(CudnnBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest,
       GetSupportedConfigsFromNonCudnnFusionReturnsEmptyVector) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kUnsupportedHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(0)));
}

TEST_F(CudnnBackendTest, GetDefaultConfigFromCudnnFusionFails) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(config, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  TF_ASSERT_OK(backend_.ApplyConfig(*fusion_instr, config));
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  TF_ASSERT_OK(backend_.ApplyConfig(*fusion_instr, config));
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
}

}  // namespace gpu
}  // namespace xla
