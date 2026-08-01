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

#include "xla/tests/aot_compatibility_experimental/test_lib.h"

#include <stdlib.h>

#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility_experimental {
namespace {

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_With4Versions) {
  unsetenv("XLA_AOT_TEST_ALL_VERSIONS");
  std::vector<AotTestParam> params =
      GetAotTestParamsForBackwardsCompatibility("test_dummy_test");
  ASSERT_EQ(params.size(), 2);
  EXPECT_EQ(params[0].version, 1);
  EXPECT_EQ(params[0].mode, AOTTestMode::kBackwardsCompatibility);
  EXPECT_EQ(params[1].version, 3);
  EXPECT_EQ(params[1].mode, AOTTestMode::kBackwardsCompatibility);
}

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_AllVersions) {
  setenv("XLA_AOT_TEST_ALL_VERSIONS", "1", 1);
  std::vector<AotTestParam> params =
      GetAotTestParamsForBackwardsCompatibility("test_dummy_test");
  ASSERT_EQ(params.size(), 4);
  EXPECT_EQ(params[0].version, 1);
  EXPECT_EQ(params[1].version, 2);
  EXPECT_EQ(params[2].version, 3);
  EXPECT_EQ(params[3].version, 4);
  unsetenv("XLA_AOT_TEST_ALL_VERSIONS");
}

TEST(TestLibTest, GetAotTestParamsForGoldenFileVerification_With4Versions) {
  std::vector<AotTestParam> params =
      GetAotTestParamsForGoldenFileVerification("test_dummy_test");
  ASSERT_EQ(params.size(), 1);
  EXPECT_EQ(params[0].version, 4);
  EXPECT_EQ(params[0].mode, AOTTestMode::kGoldenVerification);
}

}  // namespace
}  // namespace aot_compatibility_experimental
}  // namespace xla
