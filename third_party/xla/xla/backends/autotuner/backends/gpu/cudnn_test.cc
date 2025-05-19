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

#include "xla/backends/autotuner/backends/gpu/cudnn.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

using ::testing::SizeIs;
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

class CudnnBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  Compiler::TargetConfig target_config_;
  CudnnBackend backend_;

  CudnnBackendTest()
      : debug_options_([]() {
          DebugOptions debug_options;
          debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
          return debug_options;
        }()),
        target_config_([]() {
          se::GpuTargetConfigProto target_config_proto;
          *target_config_proto.mutable_gpu_device_info() =
              TestGpuDeviceInfo().CudaOrRocmDeviceInfo().ToGpuProto();
          return Compiler::TargetConfig(target_config_proto);
        }()),
        backend_(&target_config_, &debug_options_, &compiler_) {}
};

TEST_F(CudnnBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  se::StreamExecutor* stream_executor =
      PlatformUtil::GetDefaultPlatform().value()->ExecutorForDevice(0).value();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()),
          stream_executor);
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(5)));
}

TEST_F(CudnnBackendTest, GetDefaultConfigFromCudnnFusionFails) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(config, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace gpu
}  // namespace xla
