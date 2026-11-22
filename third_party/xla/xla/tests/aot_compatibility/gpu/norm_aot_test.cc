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

// AOT compatibility of NormThunk (cuDNN layer norm), on 1 GPU, H100 only.
//
// CudnnNormRewriter is off by default (xla_gpu_enable_cudnn_layer_norm) and
// only rewrites on Ampere and Hopper, so no Blackwell executable can contain a
// NormThunk, and the target runs on H100 only. The custom calls below are the
// ones the rewriter emits for a layer norm over dimension 1 of an f32[2,4]
// input, written by hand so the test does not depend on the flag. The workspace
// size is the rewriter's Hopper upper bound for 2 rows,
// 2 * (32 * 144) * (4 + 256) + 2 * 2 * 4 + 64 = 2396240, and the u8 scratch
// buffer in the result tuple must be exactly workspace_size bytes.

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "norm_aot_test";

constexpr float kEpsilon = 0.001f;

class NormAotTest : public AotCompatibilityTest,
                    public ::testing::WithParamInterface<AotTestParam> {
 public:
  NormAotTest() : AotCompatibilityTest(GetParam()) {}
};

// Inference: y = scale * (x - mean) * k + bias.
TEST_P(NormAotTest, Norm) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      x = f32[2,4,1,1] parameter(0)
      scale = f32[1,4,1,1] parameter(1)
      bias = f32[1,4,1,1] parameter(2)
      norm = (f32[2,4,1,1], u8[2396240]) custom-call(x, scale, bias),
          custom_call_target="__cudnn$norm",
          backend_config={"cudnn_norm_backend_config":{"epsilon":0.001,
            "kind":"LAYER_FWD_INFER","algorithm":{"algo_id":"0",
            "math_type":"TENSOR_OP_MATH","workspace_size":"2396240"}}}
      ROOT y = f32[2,4,1,1] get-tuple-element(norm), index=0
    }
  )hlo";
  // Row 0 normalizes to {-k, -k, k, k} and row 1 to {-k, k, -k, k}. Scale and
  // bias differ per feature, so a row or feature mix-up changes the result.
  Literal x = LiteralUtil::CreateR4<float>(
      {{{{-1}}, {{-1}}, {{1}}, {{1}}}, {{{0}}, {{2}}, {{0}}, {{2}}}});
  Literal scale = LiteralUtil::CreateR4<float>({{{{1}}, {{2}}, {{3}}, {{4}}}});
  Literal bias = LiteralUtil::CreateR4<float>({{{{0}}, {{1}}, {{2}}, {{3}}}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&x, &scale, &bias}));
  const float k = 1 / std::sqrt(1 + kEpsilon);
  LiteralTestUtil::ExpectR4Near<float>(
      {{{{-k}}, {{1 - 2 * k}}, {{2 + 3 * k}}, {{3 + 4 * k}}},
       {{{-k}}, {{1 + 2 * k}}, {{2 - 3 * k}}, {{3 + 4 * k}}}},
      result, ErrorSpec{1e-4, 1e-4});
}

// Backward pass. Covers the ExecuteOnStream branches that Norm does not: the
// optional dy, mean, inv_std, dscale and dbias buffers, and a non-default kind.
TEST_P(NormAotTest, NormBackward) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      x = f32[2,4,1,1] parameter(0)
      scale = f32[1,4,1,1] parameter(1)
      dy = f32[2,4,1,1] parameter(2)
      mean = f32[2,1,1,1] parameter(3)
      inv_std = f32[2,1,1,1] parameter(4)
      bwd = (f32[2,4,1,1], f32[1,4,1,1], f32[1,4,1,1], u8[2396240])
          custom-call(x, scale, dy, mean, inv_std),
          custom_call_target="__cudnn$norm",
          backend_config={"cudnn_norm_backend_config":{"kind":"LAYER_BWD",
            "algorithm":{"algo_id":"0","math_type":"TENSOR_OP_MATH",
            "workspace_size":"2396240"}}}
      dx = f32[2,4,1,1] get-tuple-element(bwd), index=0
      dscale = f32[1,4,1,1] get-tuple-element(bwd), index=1
      dbias = f32[1,4,1,1] get-tuple-element(bwd), index=2
      ROOT out = (f32[2,4,1,1], f32[1,4,1,1], f32[1,4,1,1]) tuple(dx, dscale,
                                                                  dbias)
    }
  )hlo";
  const float k = 1 / std::sqrt(1 + kEpsilon);
  Literal x = LiteralUtil::CreateR4<float>(
      {{{{-1}}, {{-1}}, {{1}}, {{1}}}, {{{0}}, {{2}}, {{0}}, {{2}}}});
  Literal scale = LiteralUtil::CreateR4<float>({{{{1}}, {{2}}, {{3}}, {{4}}}});
  Literal dy = LiteralUtil::CreateR4<float>(
      {{{{4}}, {{0}}, {{0}}, {{0}}}, {{{0}}, {{1}}, {{2}}, {{3}}}});
  Literal mean = LiteralUtil::CreateR4<float>({{{{0}}}, {{{1}}}});
  Literal inv_std = LiteralUtil::CreateR4<float>({{{{k}}}, {{{k}}}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(std::move(module), {&x, &scale, &dy, &mean, &inv_std}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 3);
  const float k2 = k * k;
  LiteralTestUtil::ExpectR4Near<float>({{{{k * (3 - k2)}},
                                         {{k * (-1 - k2)}},
                                         {{k * (k2 - 1)}},
                                         {{k * (k2 - 1)}}},
                                        {{{k * (2 * k2 - 5)}},
                                         {{k * (-3 - 2 * k2)}},
                                         {{k * (1 + 2 * k2)}},
                                         {{k * (7 - 2 * k2)}}}},
                                       outputs[0], ErrorSpec{1e-4, 1e-4});
  LiteralTestUtil::ExpectR4Near<float>(
      {{{{-4 * k}}, {{k}}, {{-2 * k}}, {{3 * k}}}}, outputs[1],
      ErrorSpec{1e-4, 1e-4});
  LiteralTestUtil::ExpectR4Near<float>({{{{4}}, {{1}}, {{2}}, {{3}}}},
                                       outputs[2], ErrorSpec{1e-4, 1e-4});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, NormAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, NormAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
