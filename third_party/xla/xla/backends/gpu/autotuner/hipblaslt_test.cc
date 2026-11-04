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

#include "xla/backends/gpu/autotuner/hipblaslt.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/amdgpu_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

// Returns the first device executor of the default platform.
se::StreamExecutor* GetDefaultStreamExecutor() {
  absl::StatusOr<se::Platform*> platform = PlatformUtil::GetDefaultPlatform();
  CHECK_OK(platform.status());
  absl::StatusOr<se::StreamExecutor*> stream_executor =
      (*platform)->ExecutorForDevice(0);
  CHECK_OK(stream_executor.status());
  return *stream_executor;
}

}  // namespace

using HipblasLtBackendConfig = AutotuneResult::GemmKey;

const char kHipblasLtCustomCallHlo[] = R"(
HloModule module

ENTRY main {
  p0 = f32[100,100] parameter(0)
  p1 = f32[100,100] parameter(1)
  %custom-call.1 = (f32[100,100]{1,0}, s8[4194304]{0}) custom-call(p0, p1),
    custom_call_target="__cublas$lt$matmul",
    backend_config={
      "gemm_backend_config":{
        "selected_algorithm":"0",
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{
          "operand_precision":["DEFAULT","DEFAULT"],
          "algorithm":"ALG_UNSET"
        },
        "epilogue":"DEFAULT",
        "lhs_stride":"10000",
        "rhs_stride":"10000",
        "grad_x":false,
        "grad_y":false,
        "damax_output":false
      }
    }
  ROOT %get-tuple-element = f32[100,100]{1,0} get-tuple-element(%custom-call.1), index=0
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

class HipblasLtBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  AMDGPUCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  HipblasLtBackend backend_;

  HipblasLtBackendTest()
      : debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()),
        stream_executor_(GetDefaultStreamExecutor()),
        target_config_(stream_executor_),
        backend_(stream_executor_, &debug_options_, &compiler_,
                 &target_config_) {}

  HipblasLtBackendConfig ExpectedDefaultAlgorithm() {
    auto config = AutotuneResult::GemmKey();
    config.set_algorithm(se::blas::kDefaultAlgorithm);
    return config;
  }
};

TEST_F(HipblasLtBackendTest, CanCreateHipblasLtBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(HipblasLtBackendTest, GetSupportedConfigs) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction()->operand(0));
  EXPECT_THAT(configs,
              absl_testing::IsOkAndHolds(testing::SizeIs(testing::Gt(0))));
}

TEST_F(HipblasLtBackendTest, GetSupportedConfigsReturnsErrorForDeviceless) {
  HipblasLtBackend backend_without_stream_executor(
      /*stream_executor=*/nullptr, &debug_options_, &compiler_,
      &target_config_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_without_stream_executor.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction()->operand(0));
  EXPECT_THAT(configs,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HipblasLtBackendTest,
       GetSupportedConfigsReturnsEmptyVectorForNonHipblasLtCustomCall) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kUnsupportedHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction());
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(testing::SizeIs(0)));
}

TEST_F(HipblasLtBackendTest, GetDefaultConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()->operand(0)));
  ASSERT_THAT(config, absl_testing::IsOk());
  ASSERT_TRUE((*config)->has_gemm());
  EXPECT_THAT((*config)->gemm().algorithm(), 0);
  EXPECT_NE((*config)->gemm().autotune_workspace_size(), 0);
}

TEST_F(HipblasLtBackendTest, GetDefaultConfigFailsWithoutAHipblasLtCustomCall) {
  std::string hlo = R"(
    HloModule module

    ENTRY main {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[1024,1024]{1,0} parameter(1)
      ROOT dot = f32[1024,1024]{1,0} dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(config,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HipblasLtBackendTest, ApplyConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  HipblasLtBackendConfig config;
  config.set_algorithm(2);
  config.set_autotune_workspace_size(42);
  BackendConfig backend_config;
  *backend_config.mutable_gemm() = config;
  EXPECT_OK(backend_.ApplyConfig(*hlo_module->entry_computation()
                                      ->root_instruction()
                                      ->mutable_operands()
                                      .at(0),
                                 backend_config));
  EXPECT_THAT(RunFileCheck(hlo_module->ToString(),
                           R"(CHECK: (f32[100,100]{1,0}, s8[42]{0}) custom-call
                              CHECK: "selected_algorithm":"2")"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(HipblasLtBackendTest, Compile) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()->operand(0))));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()->operand(0)), *config);
  EXPECT_THAT(executable, absl_testing::IsOk());
}

const char kScaledDotFp8FusionHlo[] = R"(
HloModule ScaledDotFusion

%fusion_dot (p0: f8e4m3fn[32,256], p1: f8e4m3fn[16,256], p2: f8e8m0fnu[32,8], p3: f8e8m0fnu[16,8]) -> f32[32,16] {
  %p0 = f8e4m3fn[32,256]{1,0} parameter(0)
  %p1 = f8e4m3fn[16,256]{1,0} parameter(1)
  %p2 = f8e8m0fnu[32,8]{1,0} parameter(2)
  %p3 = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %scaled_dot = f32[32,16]{1,0} scaled-dot(%p0, %p1, %p2, %p3), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY %main (lhs: f8e4m3fn[32,256], rhs: f8e4m3fn[16,256], lhs_scale: f8e8m0fnu[32,8], rhs_scale: f8e8m0fnu[16,8]) -> f32[32,16] {
  %lhs = f8e4m3fn[32,256]{1,0} parameter(0)
  %rhs = f8e4m3fn[16,256]{1,0} parameter(1)
  %lhs_scale = f8e8m0fnu[32,8]{1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %fusion = f32[32,16]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale), kind=kCustom, calls=%fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

const char kScaledDotFp4FusionHlo[] = R"(
HloModule ScaledDotFp4Fusion

%fusion_dot (p0: f4e2m1fn[32,256], p1: f4e2m1fn[16,256], p2: f8e8m0fnu[32,8], p3: f8e8m0fnu[16,8]) -> f32[32,16] {
  %p0 = f4e2m1fn[32,256]{1,0} parameter(0)
  %p1 = f4e2m1fn[16,256]{1,0} parameter(1)
  %p2 = f8e8m0fnu[32,8]{1,0} parameter(2)
  %p3 = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %scaled_dot = f32[32,16]{1,0} scaled-dot(%p0, %p1, %p2, %p3), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY %main (lhs: f4e2m1fn[32,256], rhs: f4e2m1fn[16,256], lhs_scale: f8e8m0fnu[32,8], rhs_scale: f8e8m0fnu[16,8]) -> f32[32,16] {
  %lhs = f4e2m1fn[32,256]{1,0} parameter(0)
  %rhs = f4e2m1fn[16,256]{1,0} parameter(1)
  %lhs_scale = f8e8m0fnu[32,8]{1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %fusion = f32[32,16]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale), kind=kCustom, calls=%fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

class HipblasLtScaledDotTest : public HipblasLtBackendTest {
 protected:
  void SetUp() override {
    const auto& gpu_cc =
        target_config_.device_description.gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "Scaled dot requires MX type support (gfx950+).";
    }
  }

  static constexpr const char* kScaledDotHlos[] = {kScaledDotFp8FusionHlo,
                                                   kScaledDotFp4FusionHlo};
};

TEST_F(HipblasLtScaledDotTest, GetSupportedConfigs) {
  for (const char* hlo : kScaledDotHlos) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    auto configs = backend_.GetSupportedConfigs(*fusion);
    EXPECT_THAT(configs,
                absl_testing::IsOkAndHolds(testing::SizeIs(testing::Gt(0))));
  }
}

TEST_F(HipblasLtScaledDotTest, ApplyConfig) {
  for (const char* hlo : kScaledDotHlos) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(auto config, backend_.GetDefaultConfig(*fusion));

    EXPECT_OK(backend_.ApplyConfig(*fusion, *config));

    EXPECT_THAT(RunFileCheck(module->ToString(),
                             R"(CHECK: custom-call
                        CHECK-SAME: custom_call_target="__cublas$lt$matmul$mx"
                        CHECK: "scale_mode":2)"),
                absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(HipblasLtScaledDotTest, Compile) {
  for (const char* hlo : kScaledDotHlos) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(auto config, backend_.GetDefaultConfig(*fusion));

    auto executable = backend_.Compile(*fusion, *config);
    EXPECT_THAT(executable, absl_testing::IsOk());
  }
}

// A hand-written FP8-output GEMM with an A of type <<A>> and an optional D
// scale. GemmRewriter passes no D scale with the DEFAULT epilogue, so a GEMM
// with one stands in for one created with a wrong SupportsF8OutputDScale.
const char kFp8OutputGemmHlo[] = R"(
HloModule module

ENTRY main {
  a = <<A>>[32,64]{1,0} parameter(0)
  b = f8e4m3fn[16,64]{1,0} parameter(1)
  a_scale = f32[] parameter(2)
  b_scale = f32[] parameter(3)
  d_scale = f32[] parameter(4)
  gemm = (f8e4m3fn[32,16]{1,0}, s8[67108864]{0}) custom-call(a, b, a_scale, b_scale<<D_SCALE>>),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","scale_mode":1<<HAS_D_SCALE>>}}
  ROOT out = f8e4m3fn[32,16]{1,0} get-tuple-element(gemm), index=0
})";

class HipblasLtFp8OutputTest : public HipblasLtBackendTest {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseFp8OutputGemm(
      absl::string_view a_type, bool has_d_scale) {
    return ParseAndReturnVerifiedModule(absl::StrReplaceAll(
        kFp8OutputGemmHlo,
        {{"<<A>>", a_type},
         {"<<D_SCALE>>", has_d_scale ? ", d_scale" : ""},
         {"<<HAS_D_SCALE>>", has_d_scale ? R"(,"has_d_scale":true)" : ""}}));
  }
};

// The output is unfused only if hipBLASLt has no algorithm for the FP8 output.
// It has D-scaled algorithms only for E4M3 x E4M3 -> E4M3 on gfx950.
TEST_F(HipblasLtFp8OutputTest, UnfusesOutputOnlyWithoutAlgorithms) {
  const auto* rocm_cc =
      target_config_.device_description.gpu_compute_capability()
          .rocm_compute_capability();
  if (rocm_cc == nullptr || !rocm_cc->has_ocp_fp8_support()) {
    GTEST_SKIP() << "Requires OCP FP8 support.";
  }
  // {A type, has a D scale, unfuses the output}
  for (auto [a_type, has_d_scale, unfused] :
       {std::tuple{"f8e5m2", false, false}, std::tuple{"f8e4m3fn", true, false},
        std::tuple{"f8e5m2", true, true}}) {
    SCOPED_TRACE(testing::Message() << a_type << " " << has_d_scale);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseFp8OutputGemm(a_type, has_d_scale));
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<BackendConfig>> configs,
        backend_.GetSupportedConfigs(
            *module->entry_computation()->root_instruction()->operand(0)));
    ASSERT_THAT(configs, testing::Not(testing::IsEmpty()));
    if (unfused && !configs[0]->gemm().unfuse_fp8_output()) {
      GTEST_SKIP() << "hipBLASLt has D-scaled algorithms for E5M2 x E4M3.";
    }
    for (const std::unique_ptr<BackendConfig>& config : configs) {
      EXPECT_EQ(config->gemm().unfuse_fp8_output(), unfused);
    }
  }
}

TEST_F(HipblasLtFp8OutputTest, ApplyUnfusedOutputConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFp8OutputGemm("f8e5m2", true));
  BackendConfig config;
  config.mutable_gemm()->set_algorithm(2);
  config.mutable_gemm()->set_autotune_workspace_size(42);
  config.mutable_gemm()->set_unfuse_fp8_output(true);
  EXPECT_OK(backend_.ApplyConfig(
      *module->entry_computation()->root_instruction()->mutable_operand(0),
      config));
  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
CHECK:      fused_computation
CHECK-DAG:    multiply(
CHECK-DAG:    f32[] constant(-448)
CHECK-DAG:    f32[] constant(448)
CHECK-DAG:    clamp(
CHECK:        ROOT {{.*}} = f8e4m3fn[32,16]{1,0} convert(
CHECK:      ENTRY
CHECK:        (f32[32,16]{1,0}, s8[42]{0}) custom-call({{[^)]*}} %b_scale)
CHECK-SAME:     custom_call_target="__cublas$lt$matmul$f8"
CHECK-SAME:     "selected_algorithm":"2"
CHECK-NOT:      has_d_scale
CHECK:        ROOT {{.*}} = f8e4m3fn[32,16]{1,0} fusion({{.*}}), kind=kLoop
)"),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace gpu
}  // namespace xla
