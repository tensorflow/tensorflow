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

// AOT compatibility of ConvolutionThunk (cuDNN convolution), on 1 GPU.
// Autotuning is off, so the golden holds cuDNN's first heuristic algorithm.

#include <memory>
#include <utility>

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

constexpr absl::string_view kTargetBase = "convolution_aot_test";

class ConvolutionAotTest : public AotCompatibilityTest,
                           public ::testing::WithParamInterface<AotTestParam> {
 public:
  ConvolutionAotTest() : AotCompatibilityTest(GetParam()) {}
};

// A sliding 2x2 window.
TEST_P(ConvolutionAotTest, Convolution) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      input = f32[1,1,3,3] parameter(0)
      filter = f32[1,1,2,2] parameter(1)
      ROOT conv = f32[1,1,2,2] convolution(input, filter), window={size=2x2},
                                       dim_labels=bf01_oi01->bf01
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_gpu_autotune_level(0);
  Literal input =
      LiteralUtil::CreateR4<float>({{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
  Literal filter = LiteralUtil::CreateR4<float>({{{{1, 2}, {3, 4}}}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&input, &filter}));
  // Cross-correlation (filter not flipped), e.g. 1*1 + 2*2 + 4*3 + 5*4 = 37.
  LiteralTestUtil::ExpectR4Near<float>({{{{37, 47}, {67, 77}}}}, result,
                                       ErrorSpec{1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, ConvolutionAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, ConvolutionAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
