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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace aot_compatibility_experimental {

struct AotTestParam {
  AOTTestMode mode;
  int32_t version;
  std::string target_name;

  bool operator==(const AotTestParam& other) const {
    return mode == other.mode && version == other.version &&
           target_name == other.target_name;
  }
};

// Gets the list of AOT test parameters for testing backwards compatibility
// boundaries.
// By default we test only 2 versions for backwards compatibility: the minimum
// and the (maximum - 1) versions to verify the boundaries of our compatibility
// guarantees. Set XLA_AOT_TEST_ALL_VERSIONS to test all versions.
absl::StatusOr<std::vector<AotTestParam>>
GetAotTestParamsForBackwardsCompatibility(absl::string_view target_name);

// Returns the latest version of the AOT dumped artifact, wrapped in a list for
// test parameterization.
absl::StatusOr<std::vector<AotTestParam>>
GetAotTestParamsForGoldenFileVerification(absl::string_view target_name);

// A parameterized test fixture base class for AOT compatibility tests.
class AotCompatibilityTest : public HloTestBase {
 public:
  explicit AotCompatibilityTest(AotTestParam param);
};

}  // namespace aot_compatibility_experimental
}  // namespace xla

#endif  // XLA_TESTS_AOT_COMPATIBILITY_EXPERIMENTAL_TEST_LIB_H_
