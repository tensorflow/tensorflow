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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace aot_compatibility_experimental {

using ::testing::TestInfo;
using ::testing::UnitTest;

std::string GetExecutablesDirectory(absl::string_view target_name,
                                    AOTTestPlatform platform) {
  // We use the full target name as part of the path, including backend (e.g.
  // collective_ops_aot_test_2gpu). The platform selects the "gpu" or "cpu"
  // executables subdirectory via the shared PlatformSubdir mapping.
  return tsl::io::JoinPath(
      tsl::testing::TensorFlowSrcRoot(),
      absl::StrCat("compiler/xla/tests/aot_compatibility_experimental/",
                   AOTInterceptionPjrtClient::PlatformSubdir(platform),
                   "/executables"),
      target_name);
}

namespace {

// Returns all available artifact versions sorted in ascending order.
absl::StatusOr<std::vector<int32_t>> GetExecutableVersions(
    absl::string_view target_name, AOTTestPlatform platform) {
  std::string dir = GetExecutablesDirectory(target_name, platform);
  std::vector<std::string> children;
  auto* env = tsl::Env::Default();
  ABSL_RETURN_IF_ERROR(env->GetChildren(dir, &children));

  std::vector<int32_t> all_versions;
  all_versions.reserve(children.size());
  for (const std::string& child : children) {
    if (!absl::StartsWith(child, "v")) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse version: ", child));
    }
    std::string child_path = tsl::io::JoinPath(dir, child);
    ABSL_RETURN_IF_ERROR(env->IsDirectory(child_path));
    absl::string_view version_str = absl::string_view(child).substr(1);
    int32_t version;
    if (!absl::SimpleAtoi(version_str, &version)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse version: ", child));
    }
    all_versions.push_back(version);
  }

  std::sort(all_versions.begin(), all_versions.end());
  return all_versions;
}

}  // namespace

absl::StatusOr<std::vector<AotTestParam>>
GetAotTestParamsForBackwardsCompatibility(absl::string_view target_name,
                                          AOTTestPlatform platform) {
  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> versions,
                   GetExecutableVersions(target_name, platform));

  if (std::getenv("XLA_AOT_TEST_ALL_VERSIONS") == nullptr &&
      versions.size() > 2) {
    // For backwards compatibility testing, we only test the minimum and the
    // (maximum - 1) versions to verify the boundaries of our compatibility
    // guarantees. The maximum version is omitted here because it is already
    // covered by the golden file verification.
    versions = {versions.front(), versions[versions.size() - 2]};
  }

  std::vector<AotTestParam> params;
  params.reserve(versions.size());
  for (int32_t v : versions) {
    params.push_back(
        {AOTTestMode::kBackwardsCompatibility, v, std::string(target_name)});
  }
  return params;
}

absl::StatusOr<std::vector<AotTestParam>>
GetAotTestParamsForGoldenFileVerification(absl::string_view target_name,
                                          AOTTestPlatform platform) {
  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> versions,
                   GetExecutableVersions(target_name, platform));
  if (versions.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No artifacts found for target: ", target_name));
  }

  return std::vector<AotTestParam>{{AOTTestMode::kGoldenVerification,
                                    versions.back(), std::string(target_name)}};
}

AotCompatibilityTest::AotCompatibilityTest(AotTestParam param)
    : HloTestBase(
          [](AotTestParam param) {
            absl::StatusOr<std::unique_ptr<PjRtClient>> client =
                GetGlobalPjRtClientTestFactory().Get()();
            CHECK_OK(client.status())
                << "Failed to create PjRt client. " << client.status();
            absl::StatusOr<AOTTestPlatform> platform =
                AOTInterceptionPjrtClient::PlatformFromName(
                    (*client)->platform_name());
            CHECK_OK(platform.status())
                << "Failed to get platform from name: "
                << (*client)->platform_name() << ". " << platform.status();
            const TestInfo* test_info =
                UnitTest::GetInstance()->current_test_info();
            std::string test_name = "";
            if (test_info != nullptr) {
              absl::string_view name_view = test_info->name();
              size_t slash_pos = name_view.find('/');
              if (slash_pos != absl::string_view::npos) {
                test_name = std::string(name_view.substr(0, slash_pos));
              } else {
                test_name = std::string(name_view);
              }
            }
            std::string artifact_path = tsl::io::JoinPath(
                GetExecutablesDirectory(param.target_name, platform.value()),
                absl::StrCat("v", param.version),
                absl::StrCat(test_name, ".pbtxt"));
            return std::make_unique<AOTInterceptionPjrtClient>(
                std::move(*client), param.mode, artifact_path);
          }(param),
          HloTestBaseOptions()) {}

}  // namespace aot_compatibility_experimental
}  // namespace xla
