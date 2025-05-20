/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/transforms/gemm_fusion.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using HloExtractionTest = HloTestBase;

TEST_F(HloExtractionTest, InstructionExtractionIsCorrect) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
HloModule module

triton_gemm_dot {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  c0 = f32[10,10] convert(p0)
  ROOT dot.0 = f32[10,10] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  s = f32[10,10] sqrt(p1)
  d = f32[10,10] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot
  ROOT r = f32[10,10] add(d, s)
})")
                                                  .value();

  std::unique_ptr<HloModule> extracted_module = ExtractInstructionIntoNewModule(
      *module->entry_computation()->root_instruction()->operand(0));

  // Destroy the original module to be sure that the extracted one has no
  // dependency on it.
  module = nullptr;

  EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
  EXPECT_EQ(extracted_module->entry_computation()->instruction_count(), 3);
  TF_EXPECT_OK(VerifyHloModule(extracted_module.get(),
                               /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/false));
}

TEST_F(HloExtractionTest, ComputationExtractionIsCorrect) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
HloModule module

triton_gemm_dot {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  c0 = f32[10,10] convert(p0)
  ROOT dot.0 = f32[10,10] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  s = f32[10,10] sqrt(p1)
  d = f32[10,10] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot
  ROOT r = f32[10,10] add(d, s)
})")
                                                  .value();

  std::unique_ptr<HloModule> extracted_module =
      ExtractComputationIntoNewModule(*module->entry_computation()
                                           ->root_instruction()
                                           ->operand(0)
                                           ->fused_instructions_computation());

  // Destroy the original module to be sure that the extracted one has no
  // dependency on it.
  module = nullptr;

  EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Convert(m::Parameter()), m::Parameter())));
  EXPECT_EQ(extracted_module->entry_computation()->instruction_count(), 4);
  TF_EXPECT_OK(VerifyHloModule(extracted_module.get(),
                               /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/false));
}

class StatelessAutotunerTest : public HloTestBase {
 public:
  StatelessAutotunerTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

  se::SemanticVersion GetToolkitVersion() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .runtime_version();
  }

  void SetUp() override {
    AutotunerUtil::ClearAutotuneResults();
    HloTestBase::SetUp();
  }

  void TearDown() override {
    AutotunerUtil::ClearAutotuneResults();
    HloTestBase::TearDown();
  }

  absl::StatusOr<std::vector<GemmFusionAutotunerImpl::BackendConfig>>
  GetPossibleMatmulAutotuneConfigs(
      const HloModule& module,
      const se::GpuComputeCapability& compute_capability,
      const se::SemanticVersion& toolkit_version,
      const DebugOptions& debug_options) {
    const HloFusionInstruction& fusion = *Cast<HloFusionInstruction>(
        module.entry_computation()->root_instruction());
    if (!isRocm()) {
      auto cu_compute_capability =
          std::get<se::CudaComputeCapability>(compute_capability);
      se::GpuDeviceInfoProto deviceless_proto;
      auto ccc = deviceless_proto.mutable_cuda_compute_capability();
      ccc->set_major(cu_compute_capability.major);
      ccc->set_minor(cu_compute_capability.minor);
    }

    DeviceConfig test_config{backend().default_stream_executor(),
                             backend().memory_allocator()};
    AutotuneConfig autotune_config = AutotuneConfig::FromDebugOptions(
        DeviceOrDevicelessConfig{test_config}, debug_options);
    GemmFusionAutotunerImpl autotuner(autotune_config, toolkit_version,
                                      debug_options, nullptr);
    return autotuner.GenerateConfigs(fusion);
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  se::RocmComputeCapability GetRocmComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .rocm_compute_capability();
  }

  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }

  bool isRocm() {
    return std::holds_alternative<se::RocmComputeCapability>(GpuComputeComp());
  }

  // Returns the config for the current device.
  absl::StatusOr<std::vector<GemmFusionAutotunerImpl::BackendConfig>>
  GetPossibleMatmulAutotuneConfigs(const HloModule& module) {
    DeviceConfig device_config{backend().default_stream_executor(),
                               backend().memory_allocator()};
    AutotuneConfig autotune_config = AutotuneConfig::FromDebugOptions(
        DeviceOrDevicelessConfig{device_config}, GetDebugOptionsForTest());
    GemmFusionAutotunerImpl autotuner(autotune_config, GetToolkitVersion(),
                                      GetDebugOptionsForTest(), nullptr);
    const HloFusionInstruction& fusion = *Cast<HloFusionInstruction>(
        module.entry_computation()->root_instruction());
    return autotuner.GenerateConfigs(fusion);
  }

  bool hasCublasConfig(
      const std::vector<GemmFusionAutotunerImpl::BackendConfig>& configs) {
    return std::any_of(
        configs.begin(), configs.end(),
        [](const GemmFusionAutotunerImpl::BackendConfig& config) {
          return std::holds_alternative<GemmFusionAutotunerImpl::CuBlasConfig>(
              config);
        });
  }
};

constexpr absl::string_view kHloDotFusionWithAlgorithm = R"(
  HloModule module

  computation {
    p0 = f32[1024,1024] parameter(0)
    p1 = f32[1024,1024] parameter(1)
    ROOT r = f32[1024,1024] dot(p0, p1),
      algorithm=$0,
      lhs_contracting_dims={1},
      rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f32[1024,1024] parameter(0)
    p1 = f32[1024,1024] parameter(1)
    ROOT computation = f32[1024,1024] fusion(f32[1024,1024] p0,f32[1024,1024] p1),
      kind=kCustom,
      calls=computation
  }
)";

TEST_F(StatelessAutotunerTest, CublasFallbackForTf32Tf32F32X3Algorithm) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(absl::Substitute(
                       kHloDotFusionWithAlgorithm, "dot_tf32_tf32_f32_x3")));

  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          GetPossibleMatmulAutotuneConfigs(*module));
  EXPECT_TRUE(hasCublasConfig(configs))
      << "There is dot_algorithm_rewrite that supports fallback to cublas "
         "implementation for dot_tf32_tf32_f32_x3.";
}

TEST_F(StatelessAutotunerTest, CublasFallbackForBf16Bf16F32Algorithm) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(absl::Substitute(
                       kHloDotFusionWithAlgorithm, "dot_bf16_bf16_f32")));

  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          GetPossibleMatmulAutotuneConfigs(*module));
  if (!isRocm()) {
    switch (GetCudaComputeCapability().major) {
      case se::CudaComputeCapability::kAmpere:
        EXPECT_TRUE(hasCublasConfig(configs))
            << "There should be a cublas fallback for dot_bf16_bf16_f32 on "
               "Ampere";
        break;
      case se::CudaComputeCapability::kHopper:
        EXPECT_TRUE(hasCublasConfig(configs))
            << "There should be a cublas fallback for dot_bf16_bf16_f32 on "
               "Hopper";
        break;
      case se::CudaComputeCapability::kBlackwell:
        EXPECT_TRUE(hasCublasConfig(configs))
            << "There should be a cublas fallback for dot_bf16_bf16_f32 on "
               "Blackwell";
        break;
      default:
        // We don't know what to expect for other compute capabilities.
        EXPECT_FALSE(hasCublasConfig(configs));
    }
  } else {
    // ROCm
    EXPECT_TRUE(hasCublasConfig(configs));
  }
}

class GemmFusionAutotunerTest : public StatelessAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        StatelessAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    debug_options.set_xla_gpu_cublas_fallback(false);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
    return debug_options;
  }

  stream_executor::GpuComputeCapability CudaAmpereOrRocm() {
    if (isRocm()) {
      return GetRocmComputeCapability();
    } else {
      return stream_executor::GpuComputeCapability{
          stream_executor::CudaComputeCapability{
              stream_executor::CudaComputeCapability::kAmpere, 0}};
    }
  }

  void CheckTritonAutotuning(absl::string_view hlo,
                             absl::string_view expected) {
    HloPassPipeline pipeline("gemm_rewrite");
    pipeline.AddPass<GemmFusion>(backend()
                                     .default_stream_executor()
                                     ->GetDeviceDescription()
                                     .gpu_compute_capability());
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                        tsl::port::MaxParallelism());
    DebugOptions opts;
    MultiProcessKeyValueStore key_value_store;
    pipeline.AddPass<GemmFusionAutotuner>(
        AutotuneConfig::FromDebugOptions(
            DeviceOrDevicelessConfig{
                DeviceConfig{backend().default_stream_executor(),
                             backend().memory_allocator()}},
            opts),
        GetToolkitVersion(), &thread_pool, key_value_store);

    RunAndFilecheckHloRewrite(
        hlo, std::move(pipeline), expected, [](const HloModule* m) {
          VLOG(5) << m->ToString();
          const HloInstruction* dot_fusion =
              m->entry_computation()->root_instruction();
          // Split-K rewriting may introduce a convert and / or a reduce op.
          if (dot_fusion->opcode() == HloOpcode::kConvert) {
            dot_fusion = dot_fusion->operand(0);
          }
          if (dot_fusion->opcode() == HloOpcode::kReduce) {
            dot_fusion = dot_fusion->operand(0);
          }
          CHECK_EQ(dot_fusion->opcode(), HloOpcode::kFusion);
          if (!dot_fusion->backend_config<GpuBackendConfig>()
                   ->fusion_backend_config()
                   .has_cudnn_fusion_config()) {
            CHECK_GT(dot_fusion->backend_config<GpuBackendConfig>()
                         .value()
                         .fusion_backend_config()
                         .triton_gemm_config()
                         .block_m(),
                     0);
          }
        });
  }
};

// TODO: b/404470821 - Remove once this is enabled by default.
class DynamicSearchSpaceAutotunerTest : public GemmFusionAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        GemmFusionAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_dynamic_dot_search_space(
        true);
    return debug_options;
  }
};

// TODO: b/404470821 - Merge this with the autotuning levels test once dynamic
// search space is enabled by default.
class DynamicSearchSpaceAutotunerDisabledTest
    : public DynamicSearchSpaceAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        GemmFusionAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }
};

absl::StatusOr<std::vector<TritonGemmConfig>>
GetPossibleMatmulAutotuneTritonConfigs(
    const HloDotInstruction& dot,
    const se::CudaComputeCapability& compute_capability,
    const se::SemanticVersion& toolkit_version,
    const DebugOptions& debug_options) {
  se::DeviceDescription device_description(
      se::GpuDeviceInfoProto::default_instance());
  device_description.set_gpu_compute_capability(compute_capability);
  // Using H100 numbers as the most relevant example here.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
  // https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#nvidia_h100_gpu_architecture_in-depth
  device_description.set_registers_per_block_limit(64 * 1024);
  device_description.set_core_count(132);
  device_description.set_threads_per_block_limit(1024);
  device_description.set_threads_per_warp(32);
  device_description.set_shared_memory_per_block_optin(227 * 1024);
  DevicelessConfig test_config = {device_description};
  AutotuneConfig autotune_config = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{test_config}, debug_options);
  GemmFusionAutotunerImpl autotuner(autotune_config, toolkit_version,
                                    debug_options, nullptr);
  return autotuner.GenerateTritonConfigs(dot);
}

TEST_F(GemmFusionAutotunerTest, AmpereUsesMoreThanTwoStages) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetToolkitVersion(), GetDebugOptionsForTest()));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const TritonGemmConfig& config) { return config.num_stages > 2; }));
}

TEST_F(GemmFusionAutotunerTest, SmallOutputCanUseLargeSplitK) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetToolkitVersion(), GetDebugOptionsForTest()));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const TritonGemmConfig& config) { return config.split_k >= 4; }));
}

TEST_F(GemmFusionAutotunerTest, LargeOutputDoesNotUseLargeSplitK) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[20480,20480] parameter(0)
  p1 = f32[20480,20480] parameter(1)
  ROOT r = f32[20480,20480] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetToolkitVersion(), GetDebugOptionsForTest()));
  EXPECT_FALSE(std::any_of(
      configs.begin(), configs.end(),
      [](const TritonGemmConfig& config) { return config.split_k > 1; }));
}

TEST_F(GemmFusionAutotunerTest, Int8FusedGemm) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,64] parameter(0)
  c = f16[128,64] convert(x)

  y = f16[64,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  CheckTritonAutotuning(hlo, R"(
// CHECK: ENTRY
// CHECK: ROOT
// CHECK-SAME: kCustom
// CHECK-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

TEST_F(GemmFusionAutotunerTest, Int8FusedGemm256) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,256] parameter(0)
  c = f16[128,256] convert(x)

  y = f16[256,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  CheckTritonAutotuning(hlo, R"(
// CHECK: ENTRY
// CHECK: ROOT
// CHECK-SAME: kCustom
// CHECK-SAME: block_m
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(GemmFusionAutotunerTest, SelectsSplitK) {
  // Shapes with K >> M, N have to force split-K configurations.
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = s8[7,8192] parameter(0)
  p0c = f16[7,8192] convert(p0)
  p1 = f16[8192,18] parameter(1)
  ROOT dot.0 = f16[7,18] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: reduce
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: kCustom
; CHECK-NEXT: {{kLoop|kInput}}
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3}));
}

TEST_F(GemmFusionAutotunerTest, ApplySplitKWithoutAlteringTiling) {
  const std::string kHloText = R"(
triton_dot {
  p0 = f16[55,120] parameter(0)
  p1 = f16[120,20] parameter(1)
  ROOT dot = f16[55,20] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[55,120]{1,0} parameter(0)
  p1 = f16[120,20]{1,0} parameter(1)
  ROOT _ = f16[55,20] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config":{kind: "__triton_gemm", triton_gemm_config: {"block_m":16,"block_n":64,"block_k":32,"split_k":3,"num_stages":1,"num_warps":2,"num_ctas":1}}}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: f{{(16|32)}}[3,55,20]
; CHECK: {"block_m":16,"block_n":64,"block_k":32,"split_k":3,"num_stages":1,"num_warps":2,"num_ctas":1}
; CHECK: f16[55,20]{1,0} {{(reduce|fusion)}}
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/344770374): Make this test not fragile.
TEST_F(GemmFusionAutotunerTest, DoNotRunAutotuningKernelSpillingRegisters) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHloText = R"(
HloModule m

%triton_gemm_dot {
  %p1 = s8[4,12288]{1,0} parameter(1)
  %p0 = s8[12288,1536]{1,0} parameter(0)
  %convert.p0 = f16[12288,1536]{1,0} convert(s8[12288,1536]{1,0} %p0)
  %convert.p1 = f16[4,12288]{1,0} convert(s8[4,12288]{1,0} %p1)
  %dot = f16[4,1536]{1,0} dot(f16[4,12288]{1,0} %convert.p1, f16[12288,1536]{1,0} %convert.p0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %convert = s8[4,1536]{1,0} convert(f16[4,1536]{1,0} %dot)
}

ENTRY %e {
  %get-tuple-element.7020 = s8[12288,1536]{1,0} parameter(0)
  %convert = s8[4,12288]{1,0} parameter(1)
  ROOT %triton = s8[4,1536]{1,0} fusion(s8[12288,1536]{1,0} %get-tuple-element.7020, s8[4,12288]{1,0} %convert), kind=kCustom, calls=%triton_gemm_dot,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"16","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1"}}}
})";

  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  EXPECT_THAT(backend().compiler()->RunBackend(
                  std::move(module), backend().default_stream_executor(),
                  {/*device_allocator=*/nullptr,
                   /*thread_pool=*/nullptr,
                   /*layout_canonicalization_callback=*/{},
                   /*is_autotuning_compilation=*/true}),
              ::testing::AnyOf(
                  tsl::testing::StatusIs(
                      tsl::error::CANCELLED,
                      "Compilation result discarded due to register spilling"),
                  // Hopper can't spill registers since wgmma instructions are
                  // asynchronous, instead it just runs out of them.
                  tsl::testing::StatusIs(
                      tsl::error::RESOURCE_EXHAUSTED,
                      ::testing::HasSubstr("Register allocation failed")),
                  tsl::testing::StatusIs(
                      tsl::error::RESOURCE_EXHAUSTED,
                      ::testing::HasSubstr("Insufficient registers"))));
}

// TODO(b/344770374): Make this test not fragile.
TEST_F(GemmFusionAutotunerTest,
       DoNotFilterOutAutotuningKernelSpillingRegisters) {
  if (GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Hopper and newer runs out of registers for such HLOs";
  }
  const std::string kHloText = R"(
HloModule m

%triton_gemm_dot {
  %p1 = s8[4,12288]{1,0} parameter(1)
  %p0 = s8[12288,1536]{1,0} parameter(0)
  %convert.p0 = f16[12288,1536]{1,0} convert(s8[12288,1536]{1,0} %p0)
  %convert.p1 = f16[4,12288]{1,0} convert(s8[4,12288]{1,0} %p1)
  %dot = f16[4,1536]{1,0} dot(f16[4,12288]{1,0} %convert.p1, f16[12288,1536]{1,0} %convert.p0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %convert = s8[4,1536]{1,0} convert(f16[4,1536]{1,0} %dot)
}

ENTRY %e {
  %get-tuple-element.7020 = s8[12288,1536]{1,0} parameter(0)
  %convert = s8[4,12288]{1,0} parameter(1)
  ROOT %triton = s8[4,1536]{1,0} fusion(s8[12288,1536]{1,0} %get-tuple-element.7020, s8[4,12288]{1,0} %convert), kind=kCustom, calls=%triton_gemm_dot,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"16","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1"}}}
})";

  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  HloModuleConfig config = module->config();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      false);
  config.set_debug_options(debug_options);
  module->set_config(config);

  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/true})
          .value();
  EXPECT_NE(executable, nullptr);
}

TEST_F(GemmFusionAutotunerTest, RunAutotuningKernelNotSpillingRegisters) {
  const std::string kHloText = R"(
HloModule m

%triton_gemm_dot {
  %p1 = f16[4,12288]{1,0} parameter(1)
  %p0 = s8[12288,1536]{1,0} parameter(0)
  %convert.10406 = f16[12288,1536]{1,0} convert(s8[12288,1536]{1,0} %p0)
  ROOT %dot = f16[4,1536]{1,0} dot(f16[4,12288]{1,0} %p1, f16[12288,1536]{1,0} %convert.10406), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY %e {
  %p0 = s8[12288,1536]{1,0} parameter(0)
  %p1 = f16[4,12288]{1,0} parameter(1)
  ROOT %triton_dot = f16[4,1536]{1,0} fusion(s8[12288,1536]{1,0} %p0, f16[4,12288]{1,0} %p1), kind=kCustom, calls=%triton_gemm_dot,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"16","block_n":"32","block_k":"16","split_k":"1","num_stages":"1","num_warps":"2","num_ctas":"1"}}}
})";

  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/true})
          .value();
  EXPECT_NE(executable, nullptr);
}

using GemmFusionAutotunerDumpTest = GemmFusionAutotunerTest;

TEST_F(GemmFusionAutotunerDumpTest, Fp8CublasltFallbackSupport) {
  const std::string kHloText = R"(
HloModule o

gemm_fusion {
  p0 = f8e4m3fn[64,6144]{1,0} parameter(0)
  p1 = f8e4m3fn[64,6144]{1,0} parameter(1)
  ROOT %dot.0 = f32[64,64]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  p0 = f8e4m3fn[64,6144]{1,0} parameter(0)
  p1 = f8e4m3fn[64,6144]{1,0} parameter(1)
  ROOT %dot.0 = f32[64,64]{1,0} fusion(p0, p1), kind=kCustom, calls=gemm_fusion, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));

  DebugOptions opts;
  AutotuneConfig autotune_config = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{DeviceConfig{backend().default_stream_executor(),
                                            backend().memory_allocator()}},
      opts);
  AutotuneCacheKey cache_key(autotune_config.GetModelStr(),
                             *module->entry_computation()->root_instruction());

  TF_ASSERT_OK_AND_ASSIGN(AutotuneResults autotune_results_override,
                          ParseTextProto<AutotuneResults>(R"pb(
                            results {
                              device: "..."
                              hlo: "..."
                              result {
                                gemm { algorithm: -1 }
                                run_time { nanos: 14 }
                              }
                            })pb"));
  AddVersionToAutotuneResults(autotune_results_override);
  autotune_results_override.mutable_results(0)->set_device(
      std::string(cache_key.GetModelStr()));
  autotune_results_override.mutable_results(0)->set_hlo(
      std::string(cache_key.GetHlo()));
  CHECK_OK(AutotunerUtil::LoadAutotuneResults(autotune_results_override));

  HloPassPipeline pipeline("gemm_autotune");
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                      tsl::port::MaxParallelism());
  MultiProcessKeyValueStore key_value_store;
  pipeline.AddPass<GemmFusionAutotuner>(autotune_config, GetToolkitVersion(),
                                        &thread_pool, key_value_store);
  pipeline.AddPass<CallInliner>();
  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    pipeline.AddPass<GemmRewriter>(autotune_config.GetGpuComputeCapability(),
                                   GetToolkitVersion(),
                                   GemmRewriterOptions{dtype});
  }

  TF_EXPECT_OK(HloTestBase::RunHloPass(&pipeline, module.get()));
  const bool is_at_least_hopper =
      std::holds_alternative<se::CudaComputeCapability>(
          autotune_config.GetGpuComputeCapability()) &&
      std::get<se::CudaComputeCapability>(
          autotune_config.GetGpuComputeCapability())
          .IsAtLeastHopper();
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(module->ToString(), is_at_least_hopper
                                           ? "// CHECK: __cublas$lt"
                                           : "// CHECK: __cublas$gemm"));
  EXPECT_TRUE(filecheck_matches);
}

TEST_F(GemmFusionAutotunerDumpTest, DumpingWorks) {
  if (isRocm()) {
    GTEST_SKIP() << "cuBLAS not selected on ROCM.";
  }
  HloModuleConfig config;
  DebugOptions options = GetDebugOptionsForTest();
  options.set_xla_gpu_cublas_fallback(true);
  options.set_xla_gpu_dump_autotuned_gemm_fusions(true);
  std::string output_directory;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&output_directory)) {
    output_directory = tsl::testing::TmpDir();
  }
  options.set_xla_dump_to(output_directory);
  config.set_debug_options(options);
  // Computation is chosen such that relatively heavy math operations before the
  // GEMM are not worth fusing because they would get duplicated many times and
  // slow down execution. Therefore autotuning picks cuBLAS here.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
fusion1 {
  p0 = f32[333,333] parameter(0)
  s = f32[333,333] sine(p0)
  p1 = f32[333,333] parameter(1)
  c = f32[333,333] cosine(p1)
  ROOT dot = f32[333,333] dot(s, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[333,333] parameter(0)
  p1 = f32[333,333] parameter(1)
  ROOT rr = f32[333,333] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm"}}
})",
                                                       config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(module)));

  std::string dump;
  TF_EXPECT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(output_directory,
                        FilenameFor(*optimized_module, /*prefix=*/"",
                                    /*suffix=*/"gemm_fusion_0.rr.txt")),
      &dump));
  EXPECT_TRUE(*RunFileCheck(dump, R"(
CHECK: HloModule rr
CHECK-NOT: cublas
CHECK: __triton_gemm
CHECK-NOT: block_m
)"));

  dump.clear();

  TF_EXPECT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(
          output_directory,
          FilenameFor(*optimized_module, /*prefix=*/"",
                      /*suffix=*/"gemm_fusion_0.rr.optimized.txt")),
      &dump));
  EXPECT_TRUE(*RunFileCheck(dump, R"(
CHECK: HloModule rr
CHECK-NOT: triton
CHECK: cublas
)"));
}

TEST_F(GemmFusionAutotunerTest, AutotuneCuDnnFusion) {
  if (isRocm()) {
    GTEST_SKIP() << "No CuDnnFusion on ROCM.";
  }
  const std::string kHlo = R"(
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

  CheckTritonAutotuning(kHlo, R"(
// CHECK: "plan_id":
)");
}

// TODO(b/281489442): Write a testcase called
// `SkipConfigsProducingDeviantResults` or similar.

class GemmFusionAutotunerLevelTest : public StatelessAutotunerTest,
                                     public ::testing::WithParamInterface<int> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        StatelessAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(GetParam());
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_P(GemmFusionAutotunerLevelTest, AllAutotuningLevelsWorkCorrectly) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  p0 = pred[64,10] parameter(0)
  p0c = f32[64,10] convert(p0)
  p1 = f32[10,128] parameter(1)
  ROOT r = f32[64,128] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: kind=kCustom
; CHECK-SAME: block_m
      )");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(GemmFusionAutotunerLevelTest, Deviceless) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[16,16] parameter(0)
  c = f16[16,16] convert(x)
  y = f16[16,16] parameter(1)
  ROOT out = f16[16,16] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  HloPassPipeline pipeline("gemm_rewrite_deviceless");
  pipeline.AddPass<GemmFusion>(backend()
                                   .default_stream_executor()
                                   ->GetDeviceDescription()
                                   .gpu_compute_capability());
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                      tsl::port::MaxParallelism());
  DebugOptions opts;
  MultiProcessKeyValueStore key_value_store;
  pipeline.AddPass<GemmFusionAutotuner>(
      AutotuneConfig::FromDebugOptions(
          DeviceOrDevicelessConfig{DevicelessConfig{
              backend().default_stream_executor()->GetDeviceDescription()}},
          opts),
      GetToolkitVersion(), &thread_pool, key_value_store);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  if (GetDebugOptionsForTest().xla_gpu_autotune_level() == 0) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed,
                            HloTestBase::RunHloPass(&pipeline, module.get()));
    EXPECT_TRUE(changed);

    // Check default configuration.
    // TODO: b/407494653 - This is a bad test because it relies on particular
    // implementation details to succeed. Thus, it tests that there is no
    // autotuning happening in a brittle way. Fix this when refactoring the
    // autotuner.
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matches,
        RunFileCheck(
            module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
            R"(
// CHECK: backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"16","block_n":"16","block_k":"16","split_k":"1","num_stages":"1","num_warps":"2","num_ctas":"1"
            )"));
    EXPECT_TRUE(filecheck_matches);
  } else {
    EXPECT_THAT(HloTestBase::RunHloPass(&pipeline, module.get()),
                tsl::testing::StatusIs(
                    tsl::error::INTERNAL,
                    ::testing::HasSubstr(
                        "Expect autotune result cache hit for deviceless")));
  }
}

INSTANTIATE_TEST_SUITE_P(GemmFusionAutotunerLevelSweep,
                         GemmFusionAutotunerLevelTest, ::testing::Range(0, 5));

class GemmFusionAutotunerExhaustiveTest : public GemmFusionAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        GemmFusionAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_exhaustive_tiling_search(true);
    return debug_options;
  }
};

TEST_F(GemmFusionAutotunerExhaustiveTest, DISABLED_CompileOnly) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[16,16] parameter(0)
  c = f16[16,16] convert(x)
  y = f16[16,16] parameter(1)
  ROOT out = f16[16,16] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  CheckTritonAutotuning(hlo, R"(
// CHECK:   %triton_gemm_out_computation (
// CHECK:   ROOT %out.1 = f16[16,16]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:   ROOT %triton_gemm_out = f16[16,16]{1,0} fusion(%x, %y), kind=kCustom, calls=%triton_gemm_out_computation
// CHECK-SAME: "block_m":
)");
}

class GemmFusionAutotunerDisableSplitK : public GemmFusionAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        GemmFusionAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    return debug_options;
  }
};

TEST_F(GemmFusionAutotunerDisableSplitK, SplitKIsDisabled) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetToolkitVersion(), GetDebugOptionsForTest()));
  EXPECT_TRUE(std::all_of(
      configs.begin(), configs.end(),
      [](const TritonGemmConfig& config) { return config.split_k == 1; }));
}

TEST_F(GemmFusionAutotunerTest, SplitKFLoatNormalization) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "f8 types are only supported from Hopper onwards.";
  }
  const se::CudaComputeCapability compute_capability =
      GetCudaComputeCapability();
  se::GpuDeviceInfoProto deviceless_proto;
  auto ccc = deviceless_proto.mutable_cuda_compute_capability();
  ccc->set_major(compute_capability.major);
  ccc->set_minor(compute_capability.minor);
  DeviceConfig test_config{backend().default_stream_executor(),
                           backend().memory_allocator()};
  AutotuneConfig autotune_config = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{test_config}, GetDebugOptionsForTest());
  GemmFusionAutotunerImpl autotuner(autotune_config, GetToolkitVersion(),
                                    GetDebugOptionsForTest(), nullptr);
  TF_ASSERT_OK_AND_ASSIGN(
      AutotunerCompileUtil compile_util,
      AutotunerCompileUtil::Create(autotune_config.DeviceConfig(),
                                   GetDebugOptionsForTest()))

  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
HloModule module

%gemm_fusion_dot_computation (parameter_0: f8e5m2[256,256], parameter_1: f8e4m3fn[128,256]) -> f8e5m2[256,128] {
  %parameter_0 = f8e5m2[256,256]{1,0} parameter(0)
  %parameter_1 = f8e4m3fn[128,256]{1,0} parameter(1)
  %dot.1 = f32[256,128]{1,0} dot(f8e5m2[256,256]{1,0} %parameter_0, f8e4m3fn[128,256]{1,0} %parameter_1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  ROOT %convert.2 = f8e5m2[256,128]{1,0} convert(f32[256,128]{1,0} %dot.1)
}
ENTRY entry {
  %p0 = f8e5m2[256,256]{1,0} parameter(0)
  %p1 = f8e4m3fn[128,256]{1,0} parameter(1)
  ROOT r = f8e5m2[256,128]{1,0} fusion(f8e5m2[256,256]{1,0} %p0, f8e4m3fn[128,256]{1,0} %p1), kind=kCustom, calls=%gemm_fusion_dot_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
})")
                                                  .value();
  GemmFusionAutotunerImpl::BackendConfigs configs;
  configs.emplace_back(
      DynCast<HloFusionInstruction>(
          module->entry_computation()->root_instruction()),
      std::vector<GemmFusionAutotunerImpl::BackendConfig>{
          GemmFusionAutotunerImpl::BackendConfig(TritonGemmConfig(
              /*block_m=*/32,
              /*block_n=*/64,
              /*block_k=*/64,
              /*split_k=*/4,
              /*num_stages=*/1,
              /*num_warps=*/4,
              /*num_ctas=*/1))});
  CHECK_OK(autotuner.CompileAll(compile_util, configs));
}

TEST_F(GemmFusionAutotunerTest, CreatesCustomKernelFusionConfigs) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo = R"(
  HloModule module, entry_computation_layout={(bf16[1024,1024]{1,0}, bf16[1024,1024]{1,0})->f32[1024,1024]{1,0}}

  %gemm_fusion_r_computation {
    %parameter_0 = bf16[1024,1024]{1,0} parameter(0)
    %convert.2 = f32[1024,1024]{1,0} convert(%parameter_0)
    %parameter_1 = bf16[1024,1024]{1,0} parameter(1)
    %convert.3 = f32[1024,1024]{1,0} convert(%parameter_1)
    ROOT %r.1 = f32[1024,1024]{1,0} dot(%convert.2, %convert.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = bf16[1024,1024]{1,0} parameter(0)
    %p1 = bf16[1024,1024]{1,0} parameter(1)
    ROOT %gemm_fusion_r = f32[1024,1024]{1,0} fusion(%p0, %p1), kind=kCustom, calls=gemm_fusion_r_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  })";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();

  const se::GpuComputeCapability compute_capability{GpuComputeComp()};

  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<GemmFusionAutotunerImpl::BackendConfig> configs,
      GetPossibleMatmulAutotuneConfigs(*module, compute_capability,
                                       GetToolkitVersion(),
                                       GetDebugOptionsForTest()));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const GemmFusionAutotunerImpl::BackendConfig& config) {
        return std::holds_alternative<
            GemmFusionAutotunerImpl::CustomKernelFusionConfig>(config);
      }));
}

TEST_F(GemmFusionAutotunerTest, GeneratesTwoConfigsForUpcastGemmWithPrologue) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo = R"(
  HloModule module

  %gemm_fusion_r_computation (parameter_0.1: f32[1,256,4,16], parameter_1.1: bf16[1,4,16,4096]) -> f32[256,4096] {
    %parameter_0.1 = f32[1,256,4,16]{3,2,1,0} parameter(0)
    %bitcast.60 = f32[256,64]{1,0} bitcast(f32[1,256,4,16]{3,2,1,0} %parameter_0.1)
    %parameter_1.1 = bf16[1,4,16,4096]{3,2,1,0} parameter(1)
    %bitcast.61 = bf16[64,4096]{1,0} bitcast(bf16[1,4,16,4096]{3,2,1,0} %parameter_1.1)
    %convert.22 = f32[64,4096]{1,0} convert(bf16[64,4096]{1,0} %bitcast.61)
    ROOT r = f32[256,4096]{1,0} dot(f32[256,64]{1,0} %bitcast.60, f32[64,4096]{1,0} %convert.22), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = f32[1,256,4,16] parameter(0)
    %p1 = bf16[1,4,16,4096] parameter(1)
    ROOT %gemm_fusion_r = f32[256,4096] fusion(%p0, %p1), kind=kCustom,
    calls=gemm_fusion_r_computation,
    backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};

  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<GemmFusionAutotunerImpl::BackendConfig> configs,
      GetPossibleMatmulAutotuneConfigs(*module, compute_capability,
                                       GetToolkitVersion(),
                                       GetDebugOptionsForTest()));
  EXPECT_EQ(
      2, std::count_if(
             configs.begin(), configs.end(),
             [](const GemmFusionAutotunerImpl::BackendConfig& config) {
               return std::holds_alternative<
                   GemmFusionAutotunerImpl::CustomKernelFusionConfig>(config);
             }));
}

TEST_F(GemmFusionAutotunerTest, GeneratesOneConfigForUpcastGemmWithPrologue) {
  // Same as GeneratesTwoConfigsForUpcastGemmWithPrologue, but with contracting
  // dimension size = 128 which is not supported by the SplitK kernel.
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo = R"(
  HloModule module

  %gemm_fusion_r_computation (parameter_0.1: f32[1,256,4,32], parameter_1.1: bf16[1,4,32,4096]) -> f32[256,4096] {
    %parameter_0.1 = f32[1,256,4,32]{3,2,1,0} parameter(0)
    %bitcast.60 = f32[256,128]{1,0} bitcast(f32[1,256,4,32]{3,2,1,0} %parameter_0.1)
    %parameter_1.1 = bf16[1,4,32,4096]{3,2,1,0} parameter(1)
    %bitcast.61 = bf16[128,4096]{1,0} bitcast(bf16[1,4,32,4096]{3,2,1,0} %parameter_1.1)
    %convert.22 = f32[128,4096]{1,0} convert(bf16[128,4096]{1,0} %bitcast.61)
    ROOT r = f32[256,4096]{1,0} dot(f32[256,128]{1,0} %bitcast.60, f32[128,4096]{1,0} %convert.22), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = f32[1,256,4,32] parameter(0)
    %p1 = bf16[1,4,32,4096] parameter(1)
    ROOT %gemm_fusion_r = f32[256,4096] fusion(%p0, %p1), kind=kCustom,
    calls=gemm_fusion_r_computation,
    backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};

  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<GemmFusionAutotunerImpl::BackendConfig> configs,
      GetPossibleMatmulAutotuneConfigs(*module, compute_capability,
                                       GetToolkitVersion(),
                                       GetDebugOptionsForTest()));
  EXPECT_EQ(
      1, std::count_if(
             configs.begin(), configs.end(),
             [](const GemmFusionAutotunerImpl::BackendConfig& config) {
               return std::holds_alternative<
                   GemmFusionAutotunerImpl::CustomKernelFusionConfig>(config);
             }));
}

TEST_F(GemmFusionAutotunerTest,
       GeneratesConfigForUpcastGemmWithPrologueAndEpilogue) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo = R"(
  HloModule module

  %gemm_fusion_r_computation (parameter_0.1: f32[1,256,4,16], parameter_1.1: bf16[1,4,16,4096]) -> bf16[1048576] {
    %parameter_0.1 = f32[1,256,4,16]{3,2,1,0} parameter(0)
    %bitcast.60 = f32[256,64]{1,0} bitcast(f32[1,256,4,16]{3,2,1,0} %parameter_0.1)
    %parameter_1.1 = bf16[1,4,16,4096]{3,2,1,0} parameter(1)
    %bitcast.61 = bf16[64,4096]{1,0} bitcast(bf16[1,4,16,4096]{3,2,1,0} %parameter_1.1)
    %convert.22 = f32[64,4096]{1,0} convert(bf16[64,4096]{1,0} %bitcast.61)
    %dot.5 = f32[256,4096]{1,0} dot(f32[256,64]{1,0} %bitcast.60, f32[64,4096]{1,0} %convert.22), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    %convert.23 = bf16[256,4096]{1,0} convert(f32[256,4096]{1,0} %dot.5)
    %bitcast.62 = bf16[1,256,4096]{2,1,0} bitcast(bf16[256,4096]{1,0} %convert.23)
    %transpose.18 = bf16[1,4096,256]{2,1,0} transpose(bf16[1,256,4096]{2,1,0} %bitcast.62), dimensions={0,2,1}
    ROOT %bitcast.63 = bf16[1048576]{0} bitcast(bf16[1,4096,256]{2,1,0} %transpose.18)
  }

  ENTRY main {
    %p0 = f32[1,256,4,16] parameter(0)
    %p1 = bf16[1,4,16,4096] parameter(1)
    ROOT %gemm_fusion_r = bf16[1048576] fusion(%p0, %p1), kind=kCustom,
    calls=gemm_fusion_r_computation,
    backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};

  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<GemmFusionAutotunerImpl::BackendConfig> configs,
      GetPossibleMatmulAutotuneConfigs(*module, compute_capability,
                                       GetToolkitVersion(),
                                       GetDebugOptionsForTest()));
  EXPECT_EQ(
      2, std::count_if(
             configs.begin(), configs.end(),
             [](const GemmFusionAutotunerImpl::BackendConfig& config) {
               return std::holds_alternative<
                   GemmFusionAutotunerImpl::CustomKernelFusionConfig>(config);
             }));
}

// Implements KeyValueStoreInterface for testing. When attempting to get a key
// that is not present using `Get`, the key is inserted and we return a dummy
// value. This actually is a valid implementation of the interface, and
// convenient for testing.
//
// To fail gracefully if a key is not found, use `TryGet` instead.
class KeyValueStoreForTest : public KeyValueStoreInterface {
 public:
  explicit KeyValueStoreForTest(std::string dummy_value)
      : dummy_value_(dummy_value) {}

  absl::StatusOr<std::string> Get(absl::string_view key,
                                  absl::Duration timeout) override {
    if (auto v = storage_.find(key); v != storage_.end()) {
      return v->second;
    }

    TF_RETURN_IF_ERROR(Set(key, dummy_value_));
    return dummy_value_;
  }

  absl::StatusOr<std::string> TryGet(absl::string_view key) override {
    if (auto v = storage_.find(key); v != storage_.end()) {
      return v->second;
    }

    return absl::NotFoundError(absl::StrCat("Key not found: ", key));
  }

  absl::Status Set(absl::string_view key, absl::string_view value) override {
    if (storage_.contains(key)) {
      return absl::AlreadyExistsError(
          absl::StrCat("Key already exists: ", key));
    }

    storage_.insert({std::string(key), std::string(value)});
    return absl::OkStatus();
  }

  const absl::flat_hash_map<std::string, std::string>& storage() const {
    return storage_;
  }

 private:
  absl::flat_hash_map<std::string, std::string> storage_;
  std::string dummy_value_;
};

// Produces dummy autotuning results for the provided cache key.
absl::StatusOr<AutotuneResults> GetDummyAutotuneResultsForCacheKey(
    const AutotuneCacheKey& cache_key) {
  TF_ASSIGN_OR_RETURN(AutotuneResults autotune_results,
                      ParseTextProto<AutotuneResults>(R"pb(
                        version: 3
                        results {
                          device: "..."
                          hlo: "..."
                          result {
                            custom_kernel_fusion { kernel_index: 1 }
                            run_time { nanos: 14 }
                          }
                        })pb"));
  AddVersionToAutotuneResults(autotune_results);
  autotune_results.mutable_results(0)->set_device(
      std::string(cache_key.GetModelStr()));
  autotune_results.mutable_results(0)->set_hlo(std::string(cache_key.GetHlo()));

  return autotune_results;
}

// Produces a MultiProcessKeyValueStore from the provided autotuning results,
// and for the provided number of processes.
absl::StatusOr<MultiProcessKeyValueStore> KeyValueStoreFromAutotuneResults(
    const AutotuneResults& autotune_results, int process_count) {
  TF_ASSIGN_OR_RETURN(std::string autotune_results_str,
                      AutotuneResultsToString(autotune_results,
                                              /*as_textproto=*/true));

  MultiProcessKeyValueStore multi_process_key_value_store;
  multi_process_key_value_store.key_value_store =
      std::make_shared<KeyValueStoreForTest>(autotune_results_str);
  multi_process_key_value_store.process_count = process_count;

  return multi_process_key_value_store;
}

class GemmFusionShardedAutotunerTest : public GemmFusionAutotunerTest {
 protected:
  AutotuneConfig GetAutotuneConfigForTest() const {
    return AutotuneConfig::FromDebugOptions(
        DeviceOrDevicelessConfig{DeviceConfig{
            backend().default_stream_executor(), backend().memory_allocator()}},
        GetDebugOptionsForTest());
  }

  GemmFusionAutotuner GemmFusionAutotunerForKeyValueStore(
      MultiProcessKeyValueStore& multi_process_key_value_store) const {
    return GemmFusionAutotuner(GetAutotuneConfigForTest(), GetToolkitVersion(),
                               /*thread_pool=*/{},
                               multi_process_key_value_store);
  }
};

TEST_F(
    GemmFusionShardedAutotunerTest,
    AutotuningSucceedsWhenKeyValueStoreAlreadyContainsAutotuningResultsForTheInputModule) {  // NOLINT(whitespace/line_length)
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo = R"(
  HloModule module

  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  const int kProcessCount = 2;
  AutotuneConfig autotune_config = GetAutotuneConfigForTest();
  AutotuneCacheKey cache_key(autotune_config.GetModelStr(),
                             *module->entry_computation()->root_instruction());
  TF_ASSERT_OK_AND_ASSIGN(AutotuneResults autotune_results_override,
                          GetDummyAutotuneResultsForCacheKey(cache_key));
  TF_ASSERT_OK_AND_ASSIGN(
      MultiProcessKeyValueStore multi_process_key_value_store,
      KeyValueStoreFromAutotuneResults(autotune_results_override,
                                       kProcessCount));
  GemmFusionAutotuner autotuner =
      GemmFusionAutotunerForKeyValueStore(multi_process_key_value_store);

  // Run the autotuner once to populate the key-value store.
  EXPECT_THAT(autotuner.Run(module->Clone().get()),
              ::tsl::testing::IsOkAndHolds(true));

  auto& key_value_store = *static_cast<KeyValueStoreForTest*>(
      multi_process_key_value_store.key_value_store.get());

  // The key-value store should now contain a single entry for each process.
  ASSERT_THAT(key_value_store.storage(), ::testing::SizeIs(kProcessCount));

  // Running the autotuner a second time on the same module should succeed and
  // modify the HLO again, but we should hit the cache (i.e., the key-value
  // store should still contain a single entry for each process).
  EXPECT_THAT(autotuner.Run(module.get()), ::tsl::testing::IsOkAndHolds(true));
  ASSERT_THAT(key_value_store.storage(), ::testing::SizeIs(kProcessCount));
}

TEST_F(
    GemmFusionShardedAutotunerTest,
    AutotuningStoresDifferentResultsForTheSameFusionInDifferentModules) {  // NOLINT(whitespace/line_length)
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo1 = R"(
  HloModule module

  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

  // Like kHlo1, but with 'abs' at the ROOT.
  const std::string kHlo2 = R"(
  HloModule module

  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    ROOT abs = f32[1024,1024]{1,0} abs(fusion)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module1,
                          ParseAndReturnVerifiedModule(kHlo1));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module2,
                          ParseAndReturnVerifiedModule(kHlo2));

  const int kProcessCount = 2;
  AutotuneConfig autotune_config = GetAutotuneConfigForTest();
  AutotuneCacheKey cache_key(autotune_config.GetModelStr(),
                             *module1->entry_computation()->root_instruction());
  TF_ASSERT_OK_AND_ASSIGN(AutotuneResults autotune_results_override,
                          GetDummyAutotuneResultsForCacheKey(cache_key));
  TF_ASSERT_OK_AND_ASSIGN(
      MultiProcessKeyValueStore multi_process_key_value_store,
      KeyValueStoreFromAutotuneResults(autotune_results_override,
                                       kProcessCount));
  GemmFusionAutotuner autotuner =
      GemmFusionAutotunerForKeyValueStore(multi_process_key_value_store);

  // Run the autotuner on the first module.
  EXPECT_THAT(autotuner.Run(module1.get()), ::tsl::testing::IsOkAndHolds(true));

  auto& key_value_store = *static_cast<KeyValueStoreForTest*>(
      multi_process_key_value_store.key_value_store.get());

  // The key-value store should now contain a single entry for each process.
  ASSERT_THAT(key_value_store.storage(), ::testing::SizeIs(kProcessCount));

  // Running the autotuner on the second module should *not* hit the cached
  // results in the key-value store. I.e., the key-value store should now
  // contain a second entry for each process).
  EXPECT_THAT(autotuner.Run(module2.get()), ::tsl::testing::IsOkAndHolds(true));
  ASSERT_THAT(key_value_store.storage(), ::testing::SizeIs(2 * kProcessCount));
}

TEST_F(GemmFusionAutotunerTest, RewritesGemmFusionToCustomKernelFusion) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  const std::string kHlo = R"(
  HloModule module, entry_computation_layout={(bf16[1024,1024]{1,0}, bf16[1024,1024]{1,0})->f32[1024,1024]{1,0}}

  %gemm_fusion_r_computation {
    %parameter_0 = bf16[1024,1024]{1,0} parameter(0)
    %convert.2 = f32[1024,1024]{1,0} convert(%parameter_0)
    %parameter_1 = bf16[1024,1024]{1,0} parameter(1)
    %convert.3 = f32[1024,1024]{1,0} convert(%parameter_1)
    ROOT %r.1 = f32[1024,1024]{1,0} dot(%convert.2, %convert.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = bf16[1024,1024]{1,0} parameter(0)
    %p1 = bf16[1024,1024]{1,0} parameter(1)
    ROOT %gemm_fusion_r = f32[1024,1024]{1,0} fusion(%p0, %p1), kind=kCustom, calls=gemm_fusion_r_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();

  DebugOptions opts;
  AutotuneConfig autotune_config = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{DeviceConfig{backend().default_stream_executor(),
                                            backend().memory_allocator()}},
      opts);
  AutotuneCacheKey cache_key(autotune_config.GetModelStr(),
                             *module->entry_computation()->root_instruction());
  TF_ASSERT_OK_AND_ASSIGN(AutotuneResults autotune_results_override,
                          GetDummyAutotuneResultsForCacheKey(cache_key));

  GemmFusionAutotunerRewriterVisitor visitor(autotune_config);

  CHECK_OK(AutotunerUtil::LoadAutotuneResults(autotune_results_override));
  visitor.RunOnModule(module.get(), {}).value();
  std::string pattern = R"(
    CHECK: ROOT %cutlass_gemm_with_upcast
    CHECK-SAME: fusion
    CHECK-SAME: kind=kCustom
    CHECK-SAME: "kernel_index":1
  )";
  TF_ASSERT_OK_AND_ASSIGN(bool file_check_matches,
                          RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(file_check_matches);
}

TEST_F(DynamicSearchSpaceAutotunerTest, AutotunesSimpleDotFusion) {
  const std::string hlo = R"(
HloModule module
ENTRY e {
  x = s8[128,64] parameter(0)
  c = f16[128,64] convert(x)
  y = f16[64,6144] parameter(1)
  ROOT out = f16[128,6144] dot(c, y),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  CheckTritonAutotuning(hlo, R"(
// CHECK: ENTRY
// CHECK: ROOT
// CHECK-SAME: kCustom
// CHECK-SAME: block_m
)");
  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

TEST_F(DynamicSearchSpaceAutotunerTest, UsesSplitKForSmallOuterDimensions) {
  const std::string hlo = R"(
HloModule module
ENTRY e {
  x = s8[32,16384] parameter(0)
  c = f16[32,16384] convert(x)
  y = f16[16384,32] parameter(1)
  ROOT out = f16[32,32] dot(c, y),
                lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  CheckTritonAutotuning(hlo, R"(
// CHECK: ENTRY
// CHECK: __triton_gemm
// CHECK-NOT: "split_k":"1"
// CHECK: ROOT
)");
}

TEST_F(DynamicSearchSpaceAutotunerTest,
       FindsValidConfigForSlicedContractingDimension) {
  const std::string hlo = R"(
ENTRY e {
  p0 = f16[32,16400] parameter(0)
  s0 = f16[32,16384] slice(p0), slice={[0:32], [11:16395]}
  p1 = f16[16384,32] parameter(1)
  ROOT _ = f16[32,32] dot(s0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  CheckTritonAutotuning(hlo, R"(
// CHECK: ENTRY
// CHECK: __triton_gemm
)");
}

TEST_F(DynamicSearchSpaceAutotunerDisabledTest,
       ReturnsSingleConfigWhenAutotuningIsDisabled) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::kAmpere, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetToolkitVersion(), GetDebugOptionsForTest()));

  EXPECT_EQ(configs.size(), 1);
}

TEST_F(GemmFusionAutotunerTest, VerifyHopperConfigsAreDifferentFromBlackwell) {
  if (isRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT r = f32[1024,1024] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })")
                                                  .value();

  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> blackwell_configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          se::CudaComputeCapability(se::CudaComputeCapability::kBlackwell, 0),
          GetToolkitVersion(), GetDebugOptionsForTest()));
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> hopper_configs,
      GetPossibleMatmulAutotuneTritonConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          se::CudaComputeCapability(se::CudaComputeCapability::kHopper, 0),
          GetToolkitVersion(), GetDebugOptionsForTest()));

  std::set<TritonGemmConfig> blackwell_configs_set(blackwell_configs.begin(),
                                                   blackwell_configs.end());
  std::set<TritonGemmConfig> hopper_configs_set(hopper_configs.begin(),
                                                hopper_configs.end());

  EXPECT_GT(blackwell_configs_set.size(), 0);
  EXPECT_GT(hopper_configs_set.size(), 0);
  EXPECT_NE(blackwell_configs_set, hopper_configs_set);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
