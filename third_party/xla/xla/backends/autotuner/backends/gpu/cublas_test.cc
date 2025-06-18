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

#include "xla/backends/autotuner/backends/gpu/cublas.h"

#include <memory>
#include <string>
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
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CublasBackendConfig = AutotuneResult::GemmKey;

using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::IsOk;

const char kCublasCustomCallHlo[] = R"(
    HloModule module, entry_computation_layout={(f32[100,100]{1,0}, f32[100,100]{1,0})->f32[100,100]{1,0}}

    ENTRY %main (arg0: f32[100,100], arg1: f32[100,100]) -> f32[100,100] {
      %arg0 = f32[100,100]{1,0} parameter(0)
      %arg1 = f32[100,100]{1,0} parameter(1)
      %custom-call.1 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%arg0, %arg1), 
      custom_call_target="__cublas$gemm", 
      backend_config={
        "gemm_backend_config":{
          "dot_dimension_numbers":
            {
              "lhs_contracting_dimensions":["1"],
              "rhs_contracting_dimensions":["0"],
              "lhs_batch_dimensions":[],
              "rhs_batch_dimensions":[]
          }
        }
      }
      ROOT %get-tuple-element = f32[100,100]{1,0} get-tuple-element(%custom-call.1), index=0
    })";

class CublasBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  Compiler::TargetConfig target_config_;
  CublasBackend backend_;

  CublasBackendTest()
      : target_config_([]() {
          se::GpuTargetConfigProto target_config_proto;
          *target_config_proto.mutable_gpu_device_info() =
              TestGpuDeviceInfo().CudaOrRocmDeviceInfo().ToGpuProto();
          return Compiler::TargetConfig(target_config_proto);
        }()),
        backend_(&target_config_, &debug_options_, &compiler_) {}

  CublasBackendConfig ExpectedDefaultAlgorithm() {
    auto config = AutotuneResult::GemmKey();
    config.set_algorithm(se::blas::kDefaultAlgorithm);
    return config;
  }
};

TEST_F(CublasBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(CublasBackendTest, GetSupportedConfigsFromCublasCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));
  se::StreamExecutor* stream_executor =
      PlatformUtil::GetDefaultPlatform().value()->ExecutorForDevice(0).value();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()->operand(0)),
          stream_executor);
  EXPECT_THAT(configs, IsOk());
  EXPECT_GT(configs.value().size(), 0);
}

TEST_F(CublasBackendTest, GetDefaultConfigFromCublasCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(static_cast<const CublasBackendConfig&>(*config.value()),
              EqualsProto(ExpectedDefaultAlgorithm()));
}

}  // namespace gpu
}  // namespace xla
