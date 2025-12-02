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

#include "xla/backends/gpu/autotuner/cublas.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CublasBackendConfig = AutotuneResult::GemmKey;

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using ::testing::IsEmpty;
using ::testing::Not;
using ::tsl::proto_testing::EqualsProto;

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

const char kCublasLtCustomCallHlo[] = R"(
  HloModule test, entry_computation_layout={(f8e4m3fn[16,32]{1,0}, f8e5m2[32,16]{1,0}, f32[], f32[])->f32[16,16]{1,0}}

  ENTRY %test (x: f8e4m3fn[16,32], y: f8e5m2[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
    %x = f8e4m3fn[16,32]{1,0} parameter(0)
    %y = f8e5m2[32,16]{1,0} parameter(1)
    %transpose = f8e5m2[16,32]{1,0} transpose(%y), dimensions={1,0}
    %x_scale = f32[] parameter(2)
    %y_scale = f32[] parameter(3)
    %cublas-gemm.1 = (f32[16,16]{1,0}, s8[33554432]{0}) custom-call(%x, %transpose, %x_scale, %y_scale),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={
      "operation_queue_id":"0",
      "wait_on_operation_queues":[],
      "gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["1"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{
          "operand_precision":["DEFAULT","DEFAULT"],
          "algorithm":"ALG_UNSET"
        },
        "epilogue":"DEFAULT",
        "lhs_stride":"512",
        "rhs_stride":"512",
        "grad_x":false,
        "grad_y":false,
        "damax_output":false
      },
      "force_earliest_schedule":false,
      "reification_cost":[],
      "device_type":"DEVICE_TYPE_INVALID"
    }
    ROOT %get-tuple-element = f32[16,16]{1,0} get-tuple-element(%cublas-gemm.1), index=0
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

class CublasBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  CublasBackend backend_;

  CublasBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(stream_executor_, &debug_options_, &compiler_,
                 &target_config_) {}

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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(configs, IsOkAndHolds(Not(IsEmpty())));
}

TEST_F(CublasBackendTest, CublasLtCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCublasLtCustomCallHlo));
  const HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->operand(0);
  CublasBackend backend(stream_executor_, &debug_options_, &compiler_,
                        &target_config_, /*fp8_lt_fallback=*/true);
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend.GetSupportedConfigs(*instr);
  EXPECT_THAT(configs, IsOkAndHolds(Not(IsEmpty())));

  EXPECT_THAT(backend.GetDefaultConfig(*instr), IsOk());
  EXPECT_THAT(backend.Compile(*instr, *configs.value()[0]), IsOk());
}

TEST_F(CublasBackendTest,
       GetSupportedConfigsReturnsEmptyVectorNonCublasCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kUnsupportedHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(0)));
}

TEST_F(CublasBackendTest, GetDefaultConfigFromCublasCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  CublasBackendConfig config_proto;
  ASSERT_TRUE(config.value()->UnpackTo(&config_proto));
  EXPECT_THAT(config_proto, EqualsProto(ExpectedDefaultAlgorithm()));
}

TEST_F(CublasBackendTest, ApplyConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));
  CublasBackendConfig config;
  config.set_algorithm(2);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_EXPECT_OK(backend_.ApplyConfig(*hlo_module->entry_computation()
                                         ->root_instruction()
                                         ->mutable_operands()
                                         .at(0),
                                    any));
  EXPECT_THAT(RunFileCheck(hlo_module->ToString(),
                           "CHECK: \"selected_algorithm\":\"2\""),
              IsOkAndHolds(true));
}

TEST_F(CublasBackendTest, Compile) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()->operand(0))));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()), *config);
  EXPECT_THAT(executable, IsOk());
}

}  // namespace gpu
}  // namespace xla
