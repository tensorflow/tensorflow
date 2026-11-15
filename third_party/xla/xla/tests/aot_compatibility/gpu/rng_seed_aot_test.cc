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

// AOT compatibility of RngSeedThunk, on 1 GPU.

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "rng_seed_aot_test";

class RngSeedAotTest : public AotCompatibilityTest,
                       public ::testing::WithParamInterface<AotTestParam> {
 public:
  RngSeedAotTest() : AotCompatibilityTest(GetParam()) {}
};

TEST_P(RngSeedAotTest, RngSeed) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      ROOT seed = u64[] custom-call(), custom_call_target="GetRngSeed"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK(Execute(std::move(module), {}).status());
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, RngSeedAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, RngSeedAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
