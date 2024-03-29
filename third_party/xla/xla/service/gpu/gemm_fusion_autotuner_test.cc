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
#include "xla/service/gpu/gemm_fusion_autotuner.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gemm_fusion.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

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
  module.release();

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
  module.release();

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

  void SetUp() override {
    AutotunerUtil::ClearAutotuneResults();
    HloTestBase::SetUp();
  }

  void TearDown() override {
    AutotunerUtil::ClearAutotuneResults();
    HloTestBase::TearDown();
  }
};

class GemmFusionAutotunerTest : public StatelessAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        StatelessAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    debug_options.set_xla_gpu_cublas_fallback(false);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
    return debug_options;
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  void CheckTritonAutotuning(absl::string_view hlo,
                             absl::string_view expected) {
    HloPassPipeline pipeline("gemm_rewrite");
    pipeline.AddPass<GemmFusion>(backend()
                                     .default_stream_executor()
                                     ->GetDeviceDescription()
                                     .cuda_compute_capability());
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                        tsl::port::MaxParallelism());
    DebugOptions opts;
    pipeline.AddPass<GemmFusionAutotuner>(
        AutotuneConfig{DeviceConfig{backend().default_stream_executor(),
                                    backend().memory_allocator()},
                       opts},
        &thread_pool);

    RunAndFilecheckHloRewrite(
        hlo, std::move(pipeline), expected, [](const HloModule* m) {
          VLOG(5) << m->ToString();
          const HloInstruction* dot_fusion =
              m->entry_computation()->root_instruction();
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

class GemmFusionAutotunerTestWithMorePreciseReduction
    : public GemmFusionAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        GemmFusionAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_disable_reduced_precision_reduction(
        true);
    return debug_options;
  }
};

TEST_F(GemmFusionAutotunerTest, AmpereUsesMoreThanTwoStages) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  ROOT r = f32[1024,1024] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                                                  .value();
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::AMPERE, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetDebugOptionsForTest()));
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
      se::CudaComputeCapability::AMPERE, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetDebugOptionsForTest()));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const TritonGemmConfig& config) { return config.split_k >= 16; }));
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
      se::CudaComputeCapability::AMPERE, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetDebugOptionsForTest()));
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
  const std::string kHloText = R"(
HloModule t

ENTRY e {
  p0 = s8[7,8192] parameter(0)
  p0c = bf16[7,8192] convert(p0)
  p1 = bf16[8192,18] parameter(1)
  ROOT dot.0 = bf16[7,18] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: reduce
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: kCustom
; CHECK-NEXT: kLoop
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/4, /*arel=*/1e-1}));
}

TEST_F(GemmFusionAutotunerTestWithMorePreciseReduction, SelectsSplitK) {
  // Shapes with K >> M, N have to force split-K configurations.
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = s8[7,8192] parameter(0)
  p0c = bf16[7,8192] convert(p0)
  p1 = bf16[8192,18] parameter(1)
  ROOT dot.0 = bf16[7,18] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: reduce
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: kCustom
; CHECK-NEXT: kLoop
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
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
; CHECK: f16[3,55,20]
; CHECK: {"block_m":16,"block_n":64,"block_k":32,"split_k":3,"num_stages":1,"num_warps":2,"num_ctas":1}
; CHECK: f16[55,20]{1,0} {{(reduce|fusion)}}
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// Modify block_k back to 16 once b/331362083 is fixed.
TEST_F(GemmFusionAutotunerTest, DoNotRunAutotuningKernelSpillingRegisters) {
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
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"32","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1"}}}
})";

  auto module = ParseAndReturnVerifiedModule(kHloText).value();
  EXPECT_THAT(
      backend().compiler()->RunBackend(std::move(module),
                                       backend().default_stream_executor(),
                                       {/*device_allocator=*/nullptr,
                                        /*thread_pool=*/nullptr,
                                        /*layout_canonicalization_callback=*/{},
                                        /*is_autotuning_compilation=*/true}),
      tsl::testing::StatusIs(
          tsl::error::CANCELLED,
          absl::StrFormat(
              "Compilation result discarded due to register spilling")));
}

// Modify block_k back to 16 once b/331362083 is fixed.
TEST_F(GemmFusionAutotunerTest,
       DoNotFilterOutAutotuningKernelSpillingRegisters) {
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
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"32","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1"}}}
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

// Modify block_k back to 16 once b/331362083 is fixed.
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
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"16","block_n":"32","block_k":"32","split_k":"1","num_stages":"1","num_warps":"2","num_ctas":"1"}}}
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

class GemmFusionAutotunerDumpTest : public GemmFusionAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        GemmFusionAutotunerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_cublas_fallback(true);
    debug_options.set_xla_gpu_dump_autotuned_triton_fusions(true);
    return debug_options;
  }
};

TEST_F(GemmFusionAutotunerDumpTest, DumpingFusionsWorksWithFallback) {
  // Computation is chosen such that relatively heavy math operations before the
  // GEMM are not worth fusing because they would get duplicated many times and
  // slow down execution. Therefore autotuning picks cuBLAS here.
  const std::string kHloText = R"(
ENTRY e {
  p0 = f32[3333,3333] parameter(0)
  s = f32[3333,3333] sine(p0)
  p1 = f32[3333,3333] parameter(1)
  c = f32[3333,3333] cosine(p1)
  ROOT dot = f32[3333,3333] dot(s, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: cublas
; CHECK-NOT: triton
)");
}

TEST_F(GemmFusionAutotunerTest, AutotuneCuDnnFusion) {
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
  DebugOptions GetDebugOptionsForTest() override {
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
                                   .cuda_compute_capability());
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                      tsl::port::MaxParallelism());
  DebugOptions opts;
  pipeline.AddPass<GemmFusionAutotuner>(
      AutotuneConfig{DevicelessConfig{backend()
                                          .default_stream_executor()
                                          ->GetDeviceDescription()
                                          .model_str(),
                                      backend()
                                          .default_stream_executor()
                                          ->GetDeviceDescription()
                                          .cuda_compute_capability()},
                     opts},
      &thread_pool);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  if (GetDebugOptionsForTest().xla_gpu_autotune_level() == 0) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed,
                            HloTestBase::RunHloPass(&pipeline, module.get()));
    EXPECT_TRUE(changed);

    // Check default configuration.
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matches,
        RunFileCheck(
            module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
            R"(
// CHECK: backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"16","block_n":"16","block_k":"16","split_k":"1","num_stages":"1","num_warps":"4","num_ctas":"1"}},"force_earliest_schedule":false}
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
  DebugOptions GetDebugOptionsForTest() override {
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
  DebugOptions GetDebugOptionsForTest() override {
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
      se::CudaComputeCapability::AMPERE, /*minor=*/0};
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<TritonGemmConfig> configs,
      GetPossibleMatmulAutotuneConfigs(
          *Cast<HloDotInstruction>(
              module->entry_computation()->root_instruction()),
          compute_capability, GetDebugOptionsForTest()));
  EXPECT_TRUE(std::all_of(
      configs.begin(), configs.end(),
      [](const TritonGemmConfig& config) { return config.split_k == 1; }));
}

class GemmFusionAutotunerConfigTest
    : public StatelessAutotunerTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(GemmFusionAutotunerConfigTest, SparseDotDiscardsUnsupportedTiles) {
  const std::string kHloText = R"(
HloModule test
ENTRY wais {
  lhs = f16[5,1600] parameter(0)
  rhs = f16[3200,10] parameter(1)
  meta = u16[5,200] parameter(2)
  ROOT dot = f32[5,10] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  auto dot =
      Cast<HloDotInstruction>(module->entry_computation()->root_instruction());

  TF_ASSERT_OK_AND_ASSIGN(
      auto configs,
      GetPossibleMatmulAutotuneConfigs(
          *dot, se::CudaComputeCapability{8, 0}, GetDebugOptionsForTest(),
          /*exhaustive_tiling_search=*/GetParam()));
  for (const auto& config : configs) {
    int metadata_size = config.block_m * config.block_k / 16;
    EXPECT_LE(config.num_warps * WarpSize(), metadata_size);
    EXPECT_GT(config.block_k, 16);  // kMinTileSize
  }
}

INSTANTIATE_TEST_SUITE_P(GemmFusionAutotunerConfigSweep,
                         GemmFusionAutotunerConfigTest, ::testing::Bool());

}  // namespace
}  // namespace gpu
}  // namespace xla
