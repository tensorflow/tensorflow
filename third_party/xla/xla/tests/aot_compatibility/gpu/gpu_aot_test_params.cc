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

#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility {
namespace {

// Returns the golden target of the current GPU for a target base name, e.g.
// "p2p_collectives_aot_test_2gpu_h100" for "p2p_collectives_aot_test_2gpu".
// The result must equal the target name that xla_test generates, since
// update_goldens.py derives the golden directory from that name.
std::string GpuGoldenTarget(absl::string_view target_base) {
  return absl::StrCat(target_base, "_", DetectGpuArchToken());
}

// INSTANTIATE_TEST_SUITE_P cannot propagate errors, so crash on them.
std::vector<AotTestParam> ParamsOrDie(
    absl::StatusOr<std::vector<AotTestParam>> params) {
  CHECK(params.ok()) << params.status();
  return *std::move(params);
}

}  // namespace

std::vector<AotTestParam> GpuBackwardsCompatibilityParams(
    absl::string_view target_base) {
  return ParamsOrDie(GetAotTestParamsForBackwardsCompatibility(
      GpuGoldenTarget(target_base), AOTTestPlatform::kGpu));
}

std::vector<AotTestParam> GpuGoldenFileVerificationParams(
    absl::string_view target_base) {
  return ParamsOrDie(GetAotTestParamsForGoldenFileVerification(
      GpuGoldenTarget(target_base), AOTTestPlatform::kGpu));
}

std::string AotTestParamName(
    const ::testing::TestParamInfo<AotTestParam>& info) {
  return absl::StrCat("v", info.param.version);
}

}  // namespace aot_compatibility
}  // namespace xla
