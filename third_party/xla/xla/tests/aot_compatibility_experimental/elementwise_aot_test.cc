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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility_experimental/test_lib.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility_experimental {

using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

namespace {

class ElementwiseAotTest : public AotCompatibilityTest,
                           public WithParamInterface<AotTestParam> {
 public:
  ElementwiseAotTest() : AotCompatibilityTest(GetParam()) {}
};

TEST_P(ElementwiseAotTest, SimpleAdd) {
  absl::string_view kModuleStr = R"hlo(
    HloModule SimpleAdd
    ENTRY SimpleAdd {
      c0 = f32[8] constant({1, 2, 3, 4, 5, 6, 7, 8})
      c1 = f32[8] constant({10, 11, 12, 13, 14, 15, 16, 17})
      ROOT add = f32[8] add(c0, c1)
    }
  )hlo";

  const int64_t kNumReplicas = 1;
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, config));

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));

  LiteralTestUtil::ExpectR1Near<float>(
      {11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0}, results[0],
      ErrorSpec{1e-5, 1e-5});
}

std::vector<AotTestParam> GetTestParamsOrDie(
    absl::StatusOr<std::vector<AotTestParam>> params) {
  CHECK(params.ok()) << params.status();
  return *std::move(params);
}

INSTANTIATE_TEST_SUITE_P(
    BackwardsCompatibility, ElementwiseAotTest,
    ValuesIn(GetTestParamsOrDie(GetAotTestParamsForBackwardsCompatibility(
        "elementwise_aot_test_cpu", AOTTestPlatform::kCpu))),
    [](const TestParamInfo<AotTestParam>& info) {
      return absl::StrCat("v", info.param.version);
    });

INSTANTIATE_TEST_SUITE_P(
    GoldenFileVerification, ElementwiseAotTest,
    ValuesIn(GetTestParamsOrDie(GetAotTestParamsForGoldenFileVerification(
        "elementwise_aot_test_cpu", AOTTestPlatform::kCpu))),
    [](const TestParamInfo<AotTestParam>& info) {
      return absl::StrCat("v", info.param.version);
    });

}  // namespace
}  // namespace aot_compatibility_experimental
}  // namespace xla
