/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/triton_fusion_numerics_verifier.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class TritonFusionNumericsVerifierTest
    : public HloPjRtTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto options = HloPjRtTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_verify_triton_fusion_numerics(true);
    return options;
  }

 protected:
  std::unique_ptr<xla::HloModule> Module(absl::string_view hlo_text_template,
                                         absl::string_view type) {
    auto m = ParseAndReturnVerifiedModule(
        absl::Substitute(hlo_text_template, type), GetModuleConfigForTest());
    TF_EXPECT_OK(m);
    return std::move(m.value());
  }

  const HloFusionInstruction* TritonFusion(const xla::HloModule& module) {
    const HloFusionInstruction* fusion_result = nullptr;

    absl::Status res =
        triton_fusion_numerics_pass_internal::ForAllTritonFusions(
            module, /*execution_threads=*/{},
            [&](const HloFusionInstruction& fusion) -> absl::Status {
              EXPECT_EQ(fusion_result, nullptr);
              fusion_result = &fusion;
              return absl::OkStatus();
            });
    return fusion_result;
  }

  AutotuneConfig CreateAutotuneConfig() {
    se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
    auto executors_or = PlatformUtil::GetStreamExecutors(platform);
    TF_EXPECT_OK(executors_or);
    return AutotuneConfig{DeviceConfig{executors_or->at(0), nullptr},
                          GetDebugOptionsForTest()};
  }

  AutotunerCompileUtil CreateAutotunerCompileUtil(AutotuneConfig& config) {
    auto compile_util_or =
        AutotunerCompileUtil::Create(config, GetDebugOptionsForTest());
    TF_EXPECT_OK(compile_util_or);
    return std::move(compile_util_or).value();
  }
};

constexpr absl::string_view kSoftmaxHlo = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
triton_softmax_computation {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
ENTRY main{
  p = $0[127,125] parameter(0)
  ROOT triton_softmax = $0[127,125] fusion(p), kind=kCustom,
    calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";

TEST_P(TritonFusionNumericsVerifierTest, VerifyExactSoftmaxFusionNumerics) {
  auto module = Module(kSoftmaxHlo,
                       primitive_util::LowercasePrimitiveTypeName(GetParam()));

  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}

TEST_P(TritonFusionNumericsVerifierTest, VerifyMultiOutputFusionNumerics) {
  constexpr absl::string_view kMultiOutputFusionHloText = R"(
HloModule m
fusion_computation {
  param_0 = $0[127,125]{1,0} parameter(0)
  exponential = $0[127,125]{1,0} exponential(param_0)
  negate = $0[127,125]{1,0} negate(exponential)
  ROOT res = ($0[127,125]{1,0}, $0[127,125]{1,0}) tuple(exponential, negate)
}

ENTRY main{
  p = $0[127,125] parameter(0)
  ROOT result = ($0[127,125], $0[127,125]) fusion(p), kind=kCustom,
    calls=fusion_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}, {"sizes":["1","125"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";
  auto module = Module(kMultiOutputFusionHloText,
                       primitive_util::LowercasePrimitiveTypeName(GetParam()));

  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}

TEST_F(TritonFusionNumericsVerifierTest, CheckMismatch) {
  // This test intentionally compares two different Triton modules to each
  // other. This is to test that the verifier functions correctly catch and
  // report mismatches.
  //
  // Note that as part of computing the two modules below, the numerics verifier
  // pass also runs individually for each module. These runs compare the
  // modules to the corresponding emitters generated version, which matches. In
  // that sense this test covers what is being tested by
  // VerifyExactSoftmaxFusionNumerics. The reason to keep two tests is that
  // VerifyExactSoftmaxFusionNumerics is minimal and will be easier to debug if
  // it fails.

  auto module_f64 = Module(kSoftmaxHlo, "f64");
  auto fusion_f64 = TritonFusion(*module_f64);
  EXPECT_NE(fusion_f64, nullptr);

  auto module_f32 = Module(kSoftmaxHlo, "f32");
  auto fusion_f32 = TritonFusion(*module_f32);
  EXPECT_NE(fusion_f32, nullptr);

  AutotuneConfig autotune_config = CreateAutotuneConfig();
  AutotunerCompileUtil compile_util =
      CreateAutotunerCompileUtil(autotune_config);
  const DebugOptions& debug_options = GetDebugOptionsForTest();

  auto f64_result = triton_fusion_numerics_pass_internal::CompileAndRunFusion(
      compile_util, *fusion_f64, autotune_config, debug_options,
      /*clear_backend_config=*/false);
  TF_EXPECT_OK(f64_result);

  auto f32_result = triton_fusion_numerics_pass_internal::CompileAndRunFusion(
      compile_util, *fusion_f32, autotune_config, debug_options,
      /*clear_backend_config=*/false);
  TF_EXPECT_OK(f32_result);

  auto stream = autotune_config.GetStream();
  TF_EXPECT_OK(stream);

  // Intentionally compare the fusions from the different modules, triggering a
  // mismatch.
  auto cmp = triton_fusion_numerics_pass_internal::CompareBuffers(
      *f64_result, *f32_result, fusion_f64->shape(),
      fusion_f64->GetModule()->config(), *stream);

  EXPECT_FALSE(cmp.ok());
}

// By default, AutotunerCompileUtil filters out kernels that cause registers to
// spill. Verify that the numerics verifier still runs on those kernels.
TEST_F(TritonFusionNumericsVerifierTest,
       CompilationSucceedsEvenIfKernelWillSpillRegisters) {
  auto module = Module(R"(
HloModule m

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[16,256000] parameter(0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[16]{0} reduce(param_0, constant_0), dimensions={1}, to_apply=add
  broadcast_0 = f32[16,256000]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[16,256000]{1,0} multiply(param_0, broadcast_0)
}

ENTRY main {
  param_0 = f32[16,256000] parameter(0)
  ROOT triton_softmax = f32[16,256000]{1,0} fusion(param_0), kind=kCustom,
    calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","256000"]}],
          "num_warps":"32",
          "num_ctas":"1",
          "num_stages":"1"}}}
})",
                       "");

  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
  auto fusion = TritonFusion(*module);
  EXPECT_NE(fusion, nullptr);

  AutotuneConfig autotune_config = CreateAutotuneConfig();
  AutotunerCompileUtil compile_util =
      CreateAutotunerCompileUtil(autotune_config);
  auto compilation_result =
      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
          compile_util, *fusion, autotune_config, GetDebugOptionsForTest(),
          /*disable_triton=*/false);

  // Verify that the compilation with default flags fails. The compilation
  // fails, because the kernel will spill registers, but the error is
  // overwritten inside the autotuner utils and returns a generic error.
  EXPECT_FALSE(compilation_result.ok());
  EXPECT_THAT(compilation_result.status(),
              tsl::testing::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(compilation_result.status().message(),
              ::testing::HasSubstr("Failed to compile Triton fusion"));
}

TEST_F(TritonFusionNumericsVerifierTest, CacheIsUsed) {
  absl::string_view hlo_text = R"(
add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] maximum(p0, p1)
}

reduce_0 {
  p = f32[16,16] parameter(0)
  c = f32[] constant(0)
  ROOT reduce_0 = f32[16]{0} reduce(p, c), dimensions={1}, to_apply=add
}

reduce_1 {
  p = f32[16,16] parameter(0)
  c = f32[] constant(0)
  ROOT reduce_0 = f32[16]{0} reduce(p, c), dimensions={1}, to_apply=max
}

// Identical to reduce_0.
reduce_2 {
  p = f32[16,16] parameter(0)
  c = f32[] constant(0)
  ROOT reduce_0 = f32[16]{0} reduce(p, c), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[16,16] parameter(0)
  p1 = f32[16,16] parameter(1)
  p2 = f32[16,16] parameter(2)
  r0 = f32[16] fusion(p0), kind=kCustom, calls=reduce_0, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
  r1 = f32[16] fusion(p1), kind=kCustom, calls=reduce_1, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
  r2 = f32[16] fusion(p2), kind=kCustom, calls=reduce_2, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
  add_0_1 = f32[16] add(r0, r1)
  ROOT add_0_2 = f32[16] add(add_0_1, r2)
}
  )";

  std::unique_ptr<HloModule> module = Module(hlo_text, "");
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
  EXPECT_EQ(verifier.CacheHitsForTestingOnly(), 1);
}

TEST_F(TritonFusionNumericsVerifierTest, VerifyThatDisablingTritonIsFast) {
  // This computation results in a single Triton fusion. If that fusion is
  // compiled without Triton and without rerunning the fusion pass, the
  // resulting kernel is extremely slow and the test will timeout. This test
  // ensures that the fusion pass is rerun.
  absl::string_view hlo_text = R"(
max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

triton_softmax_computation {
  p0 = f32[16384,16384] parameter(0)
  reshape1 = f32[1,1,16384,16384] reshape(p0)
  reshape2 = f32[1,16384,16384] reshape(p0)
  constant3 = f32[] constant(-inf)
  reduce0 = f32[1,16384] reduce(reshape2, constant3), dimensions={2}, to_apply=max
  broadcast3 = f32[1,1,16384,16384] broadcast(reduce0), dimensions={1,2}
  sub = f32[1,1,16384,16384] subtract(reshape1, broadcast3)
  exp = f32[1,1,16384,16384] exponential(sub)
  reshape3 = f32[1,16384,16384] reshape(exp)
  constant4 = f32[] constant(0)
  reduce1 = f32[1,16384] reduce(reshape3, constant4), dimensions={2}, to_apply=add
  broadcast4 = f32[1,1,16384,16384] broadcast(reduce1), dimensions={1,2}
  ROOT div = f32[1,1,16384,16384] divide(exp, broadcast4)
}

ENTRY main {
  p = f32[16384,16384] parameter(0)
  ROOT triton_softmax = f32[1,1,16384,16384] fusion(p), kind=kCustom,
    calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1","1","16384"]}],
        "num_warps":"32",
        "num_ctas":"1",
        "num_stages":"1"}}}
}
  )";
  auto module = Module(hlo_text, "");
  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}

INSTANTIATE_TEST_SUITE_P(TritonFusionNumericsVerifierTestSuite,
                         TritonFusionNumericsVerifierTest,
                         ::testing::Values(F32, F64));

}  // namespace
}  // namespace xla::gpu
