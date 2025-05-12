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

#include "xla/backends/autotuner/backends/gpu/custom_kernel.h"

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

namespace xla {
namespace gpu {

using CublasBackendConfig = AutotuneResult::GemmKey;

using tsl::testing::IsOk;
using tsl::testing::StatusIs;

const char kCustomKernelFusionHlo[] = R"(
HloModule extracted

cutlass_gemm {
  p0 = f32[15,19]{1,0} parameter(0)
  p1 = f32[19,17]{1,0} parameter(1)
  ROOT r = f32[15, 17]{1,0} dot(p0, p1), lhs_contracting_dims={1},
  rhs_contracting_dims={0}
}

ENTRY region_198.14436 {
  p.0 = f32[15,19]{1,0} parameter(0)
  p.1 = f32[19,17]{1,0} parameter(1)
  ROOT cutlass_gemm = f32[15,17]{1,0} fusion(p.0, p.1), kind=kCustom,
  calls=cutlass_gemm,
  backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"cutlass_gemm","kernel_index":-1}},"force_earliest_schedule":false}
})";

class CustomKernelBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  Compiler::TargetConfig target_config_;
  CustomKernelBackend backend_;

  CustomKernelBackendTest()
      : target_config_([]() {
          se::GpuTargetConfigProto target_config_proto;
          *target_config_proto.mutable_gpu_device_info() =
              TestGpuDeviceInfo().CudaOrRocmDeviceInfo().ToGpuProto();
          return Compiler::TargetConfig(target_config_proto);
        }()),
        backend_(&target_config_, &debug_options_, &compiler_) {}
};

TEST_F(CustomKernelBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(CustomKernelBackendTest, GetSupportedConfigsFromCustomKernelFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCustomKernelFusionHlo));
  se::StreamExecutor* stream_executor =
      PlatformUtil::GetDefaultPlatform().value()->ExecutorForDevice(0).value();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()), stream_executor);
  EXPECT_THAT(configs, IsOk());
  EXPECT_FALSE(configs.value().empty());
}

TEST_F(CustomKernelBackendTest, GetDefaultConfigFails) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCustomKernelFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(config, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace gpu
}  // namespace xla
