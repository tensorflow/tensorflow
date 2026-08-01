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
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

std::string GetExecutablesDirectory(absl::string_view target_name) {
  // We use the full target name as part of the path, including backend (e.g.
  // collective_ops_aot_test_2gpu)
  return tsl::io::JoinPath(
      tsl::testing::TensorFlowSrcRoot(),
      "compiler/xla/tests/aot_compatibility_experimental/gpu/executables",
      target_name);
}

std::vector<int> GetExecutableVersions(absl::string_view target_name) {
  std::string dir = GetExecutablesDirectory(target_name);
  std::vector<std::string> children;
  auto* env = tsl::Env::Default();
  CHECK_OK(env->GetChildren(dir, &children));

  std::vector<int> all_versions;
  all_versions.reserve(children.size());
  for (const std::string& child : children) {
    if (absl::StartsWith(child, "v")) {
      std::string child_path = tsl::io::JoinPath(dir, child);
      CHECK_OK(env->IsDirectory(child_path));
      absl::string_view version_str = absl::string_view(child).substr(1);
      int version;
      CHECK(absl::SimpleAtoi(version_str, &version))
          << "Failed to parse version: " << child;
      all_versions.push_back(version);
    }
  }

  std::sort(all_versions.begin(), all_versions.end());
  return all_versions;
}

std::vector<AotTestParam> GetAotTestParamsForBackwardsCompatibility(
    absl::string_view target_name) {
  std::vector<int> versions = GetExecutableVersions(target_name);

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
  for (int v : versions) {
    params.push_back(
        {AOTTestMode::kBackwardsCompatibility, v, std::string(target_name)});
  }
  return params;
}

std::vector<AotTestParam> GetAotTestParamsForGoldenFileVerification(
    absl::string_view target_name) {
  std::vector<int> versions = GetExecutableVersions(target_name);
  std::vector<AotTestParam> params;
  if (!versions.empty()) {
    params.push_back({AOTTestMode::kGoldenVerification, versions.back(),
                      std::string(target_name)});
  } else {
    LOG(FATAL) << "No artifacts found for target: " << target_name;
  }
  return params;
}

AotCompatibilityTest::AotCompatibilityTest(AotTestParam param)
    : HloTestBase(
          [](AotTestParam param) {
            absl::StatusOr<std::unique_ptr<PjRtClient>> client =
                GetGlobalPjRtClientTestFactory().Get()();
            CHECK_OK(client.status())
                << "Failed to create PjRt client. " << client.status();
            const ::testing::TestInfo* test_info =
                ::testing::UnitTest::GetInstance()->current_test_info();
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
            std::string artifact_path =
                tsl::io::JoinPath(GetExecutablesDirectory(param.target_name),
                                  absl::StrCat("v", param.version),
                                  absl::StrCat(test_name, ".pbtxt"));
            return std::make_unique<AOTInterceptionPjrtClient>(
                std::move(*client), param.mode, artifact_path);
          }(param)
              .release(),
          HloTestBaseOptions()) {}

}  // namespace aot_compatibility_experimental
}  // namespace xla
