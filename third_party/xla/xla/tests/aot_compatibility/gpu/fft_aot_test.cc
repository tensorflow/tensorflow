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

// AOT compatibility of FftThunk (cuFFT), on 1 GPU.

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "fft_aot_test";

class FftAotTest : public AotCompatibilityTest,
                   public ::testing::WithParamInterface<AotTestParam> {
 public:
  FftAotTest() : AotCompatibilityTest(GetParam()) {}
};

TEST_P(FftAotTest, Fft) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      p = c64[4] parameter(0)
      ROOT f = c64[4] fft(p), fft_type=FFT, fft_length={4}
    }
  )hlo";
  Literal p = LiteralUtil::CreateR1<complex64>({1, 2, 3, 4});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&p}));
  LiteralTestUtil::ExpectR1Near<complex64>(
      {{10, 0}, {-2, 2}, {-2, 0}, {-2, -2}}, result, ErrorSpec{1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, FftAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, FftAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
