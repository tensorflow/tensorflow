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

#ifndef XLA_TESTS_AOT_COMPATIBILITY_GPU_GPU_AOT_TEST_PARAMS_H_
#define XLA_TESTS_AOT_COMPATIBILITY_GPU_GPU_AOT_TEST_PARAMS_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility {

// Test parameters for the two suites every GPU AOT test target instantiates.
// `target_base` is the target name without the arch suffix, e.g.
// "p2p_collectives_aot_test_2gpu".
std::vector<AotTestParam> GpuBackwardsCompatibilityParams(
    absl::string_view target_base);
std::vector<AotTestParam> GpuGoldenFileVerificationParams(
    absl::string_view target_base);

std::string AotTestParamName(
    const ::testing::TestParamInfo<AotTestParam>& info);

}  // namespace aot_compatibility
}  // namespace xla

#endif  // XLA_TESTS_AOT_COMPATIBILITY_GPU_GPU_AOT_TEST_PARAMS_H_
