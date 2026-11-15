/* Copyright 2026 The OpenXLA Authors.

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

// AOT compatibility of DynamicSliceFusionThunk, on 1 GPU: a dot on one slice
// of a stacked weight array per loop iteration, as in a `lax.scan` over layers.
// DynamicSliceFusionRewriterV2 wraps the cuBLASLt matmul so that it reads the
// weight slice and writes the output slice in place.

#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "dynamic_slice_fusion_aot_test";

class DynamicSliceFusionAotTest
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  DynamicSliceFusionAotTest() : AotCompatibilityTest(GetParam()) {}
};

// Iteration i computes out[i] = lhs . weights[i]. The rewriter only fuses
// slices whose size is a multiple of 256 bytes, so each slice is f32[8,8].
// The index is written the way JAX normalizes a possibly negative index,
// select(i < 0, i + 4, i), so the offset expressions in the golden include
// SELECT, COMPARE and ADD.
TEST_P(DynamicSliceFusionAotTest, DynamicSliceDotInWhileLoop) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    body {
      param = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      lhs = f32[8,8] get-tuple-element(param), index=1
      weights = f32[4,8,8] get-tuple-element(param), index=2
      out = f32[4,8,8] get-tuple-element(param), index=3
      zero = s32[] constant(0)
      four = s32[] constant(4)
      negative = pred[] compare(i, zero), direction=LT
      wrapped = s32[] add(i, four)
      idx = s32[] select(negative, wrapped, i)
      slice = f32[1,8,8] dynamic-slice(weights, idx, zero, zero),
              dynamic_slice_sizes={1,8,8}
      rhs = f32[8,8] reshape(slice)
      dot = f32[8,8] dot(lhs, rhs), lhs_contracting_dims={1},
                                    rhs_contracting_dims={0}
      update = f32[1,8,8] reshape(dot)
      updated = f32[4,8,8] dynamic-update-slice(out, update, idx, zero, zero)
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8])
                   tuple(next_i, lhs, weights, updated)
    }
    cond {
      param = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }
    ENTRY main {
      lhs = f32[8,8] parameter(0)
      weights = f32[4,8,8] parameter(1)
      zero = s32[] constant(0)
      init_out = f32[4,8,8] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8])
             tuple(zero, lhs, weights, init_out)
      while = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8]) while(init),
              condition=cond, body=body
      ROOT result = f32[4,8,8] get-tuple-element(while), index=3
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions& debug_options = config.mutable_debug_options();
  // Triton GEMM off and cuBLASLt on, so GemmRewriter lowers the dot to a
  // cuBLASLt matmul, the hero DynamicSliceFusionRewriterV2 wraps.
  debug_options.set_xla_gpu_autotune_level(0);
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  debug_options.set_xla_gpu_enable_cublaslt(true);
  debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);

  // Small integers, so the products and sums are exact in f32 and TF32.
  std::vector<float> lhs_values(8 * 8);
  for (int64_t k = 0; k < lhs_values.size(); ++k) {
    lhs_values[k] = static_cast<float>(k % 4);
  }
  std::vector<float> weight_values(4 * 8 * 8);
  std::iota(weight_values.begin(), weight_values.end(), 0.0f);
  std::vector<float> expected_values(4 * 8 * 8, 0.0f);
  for (int64_t layer = 0; layer < 4; ++layer) {
    for (int64_t m = 0; m < 8; ++m) {
      for (int64_t n = 0; n < 8; ++n) {
        for (int64_t k = 0; k < 8; ++k) {
          expected_values[(layer * 8 + m) * 8 + n] +=
              lhs_values[m * 8 + k] * weight_values[(layer * 8 + k) * 8 + n];
        }
      }
    }
  }
  ASSERT_OK_AND_ASSIGN(
      Literal lhs, LiteralUtil::CreateR1<float>(lhs_values).Reshape({8, 8}));
  ASSERT_OK_AND_ASSIGN(
      Literal weights,
      LiteralUtil::CreateR1<float>(weight_values).Reshape({4, 8, 8}));
  ASSERT_OK_AND_ASSIGN(
      Literal expected,
      LiteralUtil::CreateR1<float>(expected_values).Reshape({4, 8, 8}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&lhs, &weights}));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec{1e-5, 1e-5}));
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, DynamicSliceFusionAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, DynamicSliceFusionAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
