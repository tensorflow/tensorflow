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

#include "xla/service/gpu/triton_fusion_numerics_verifier.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {

class TritonFusionNumericsVerifierTest
    : public HloTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    auto options = HloTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_enable_triton_softmax_fusion(true);
    options.set_xla_gpu_verify_triton_fusion_numerics(true);
    return options;
  }

 protected:
  std::unique_ptr<xla::HloModule> Module(absl::string_view hlo_text_template,
                                         absl::string_view type) {
    auto m = GetOptimizedModule(absl::Substitute(hlo_text_template, type));
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
    auto opt_compile_util_or =
        AutotunerCompileUtil::Create(config, GetDebugOptionsForTest());
    TF_EXPECT_OK(opt_compile_util_or);
    EXPECT_TRUE(opt_compile_util_or->has_value());
    return std::move(opt_compile_util_or->value());
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
ENTRY main {
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
)";

bool HloPassHasRun(const HloModule& module, absl::string_view pass_name) {
  for (const auto& pass_metadata : module.metadata().proto().pass_metadata()) {
    if (pass_metadata.pass_name() == pass_name) {
      return true;
    }
  }
  return false;
}

TEST_P(TritonFusionNumericsVerifierTest, VerifyExactSoftmaxFusionNumerics) {
  PrimitiveType data_type = GetParam();

  auto module = Module(kSoftmaxHlo,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  // At this point all HLO passes have been executed successfully, because the
  // Module() function hasn't failed. In particular the numerics verification
  // pass should have also run and **not** found any issues. Below we just
  // ensure that the pass has indeed been correctly enabled and that there are
  // Triton Fusions in the input module.

  EXPECT_TRUE(HloPassHasRun(*module, TritonFusionNumericsVerifier::Name()));
  auto fusion = TritonFusion(*module);
  EXPECT_NE(fusion, nullptr);
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

  auto module_f16 = Module(kSoftmaxHlo, "f16");
  auto fusion_f16 = TritonFusion(*module_f16);
  EXPECT_NE(fusion_f16, nullptr);

  auto module_f32 = Module(kSoftmaxHlo, "f32");
  auto fusion_f32 = TritonFusion(*module_f32);
  EXPECT_NE(fusion_f32, nullptr);

  AutotuneConfig autotune_config = CreateAutotuneConfig();
  AutotunerCompileUtil compile_util =
      CreateAutotunerCompileUtil(autotune_config);
  const DebugOptions& debug_options = GetDebugOptionsForTest();

  auto f16_result = triton_fusion_numerics_pass_internal::CompileAndRunFusion(
      compile_util, *fusion_f16, autotune_config, debug_options,
      /*clear_backend_config=*/false);
  TF_EXPECT_OK(f16_result);

  auto f32_result = triton_fusion_numerics_pass_internal::CompileAndRunFusion(
      compile_util, *fusion_f32, autotune_config, debug_options,
      /*clear_backend_config=*/false);
  TF_EXPECT_OK(f32_result);

  auto stream = autotune_config.GetStream();
  TF_EXPECT_OK(stream);

  // Intentionally compare the fusions from the different modules, triggering a
  // mismatch.
  auto cmp = triton_fusion_numerics_pass_internal::CompareBuffers(
      *f16_result, *f32_result, fusion_f16->shape(),
      fusion_f16->GetModule()->config(), *stream);

  EXPECT_FALSE(cmp.ok());
}

INSTANTIATE_TEST_SUITE_P(TritonFusionNumericsVerifierTestSuite,
                         TritonFusionNumericsVerifierTest,
                         ::testing::Values(F32, F16, BF16));

}  // namespace
}  // namespace xla::gpu
