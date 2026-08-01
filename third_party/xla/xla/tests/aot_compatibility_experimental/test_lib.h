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
#ifndef XLA_TESTS_AOT_COMPATIBILITY_EXPERIMENTAL_TEST_LIB_H_
#define XLA_TESTS_AOT_COMPATIBILITY_EXPERIMENTAL_TEST_LIB_H_

#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace aot_compatibility_experimental {

// Returns the path to the executables directory for the current test target.
std::string GetExecutablesDirectory(absl::string_view target_name);

// Returns all available artifact versions sorted in ascending order.
std::vector<int> GetExecutableVersions(absl::string_view target_name);

struct AotTestParam {
  AOTTestMode mode;
  int version;
  std::string target_name;
};

std::vector<AotTestParam> GetAotTestParamsForBackwardsCompatibility(
    absl::string_view target_name);
std::vector<AotTestParam> GetAotTestParamsForGoldenFileVerification(
    absl::string_view target_name);

// A parameterized test fixture base class for AOT compatibility tests.
class AotCompatibilityTest : public HloTestBase {
 public:
  explicit AotCompatibilityTest(AotTestParam param);
};

}  // namespace aot_compatibility_experimental
}  // namespace xla

#endif  // XLA_TESTS_AOT_COMPATIBILITY_EXPERIMENTAL_TEST_LIB_H_
