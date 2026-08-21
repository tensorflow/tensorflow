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
#include "xla/tests/aot_compatibility_test_lib.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tests/aot_utils.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {

std::string AotTestConfigNameGenerator::operator()(
    const ::testing::TestParamInfo<AotTestConfig>& info) const {
  if (info.param.is_golden) {
    return "golden";
  }
  return absl::StrCat("v", info.param.version);
}

std::vector<AotTestConfig> GetAvailableAotVersions(
    const std::string& test_dir) {
  std::vector<AotTestConfig> configs;
  // Always include golden as the primary test mode.
  configs.push_back(AotTestConfig{/*is_golden=*/true, /*version=*/0});

  std::string pattern = tsl::io::JoinPath(test_dir, "v*.pbtxt");
  std::vector<std::string> paths;
  tsl::Env* env = tsl::Env::Default();

  if (!env->GetMatchingPaths(pattern, &paths).ok() || paths.empty()) {
    return configs;
  }

  std::vector<int> versions;
  for (const std::string& path : paths) {
    std::string basename = std::string(tsl::io::Basename(path));
    absl::string_view name(basename);
    if (absl::ConsumePrefix(&name, "v") &&
        absl::ConsumeSuffix(&name, ".pbtxt")) {
      int version;
      if (absl::SimpleAtoi(name, &version)) {
        versions.push_back(version);
      }
    }
  }

  if (versions.empty()) {
    return configs;
  }

  std::sort(versions.begin(), versions.end());

  int min_version = versions.front();
  int max_version = versions.back();

  // The user requested exactly: golden, <version_max - 1>, and version_min.
  // 'golden' is already added above (tests current baseline).
  // Now add version_max - 1 (if it exists and is >= min_version).
  if (max_version - 1 >= min_version) {
    configs.push_back(AotTestConfig{/*is_golden=*/false, max_version - 1});
  }

  // Add version_min (only if it doesn't collide with the max_version - 1 we
  // just added).
  if (min_version != max_version - 1) {
    configs.push_back(AotTestConfig{/*is_golden=*/false, min_version});
  }

  return configs;
}

namespace {

AotCompatibilityTestBase::ClientData CreateInterceptedClientData(
    const AotTestConfig& config, const std::string& artifact_dir) {
  CHECK(ShouldUsePjRt()) << "PjRt is required for AotCompatibilityTestBase.";
  absl::StatusOr<std::unique_ptr<PjRtClient>> client_or =
      GetGlobalPjRtClientTestFactory().Get()();
  CHECK_OK(client_or.status()) << "Failed to create PjRt client.";

  AOTTestMode mode = config.is_golden ? AOTTestMode::kGoldenVerification
                                      : AOTTestMode::kBackwardPrevious;

  std::string filename = absl::StrCat("v", config.version, ".pbtxt");
  std::string artifact_path = tsl::io::JoinPath(artifact_dir, filename);

  auto client = std::make_unique<AOTInterceptionPjrtClient>(
      *std::move(client_or), mode, artifact_path);

  auto rep_fn = GetGlobalPjRtClientTestFactory().GetDeviceShapeRepresentationFn(
      client.get());
  auto size_fn =
      GetGlobalPjRtClientTestFactory().GetDeviceShapeSizeFn(client.get());

  return AotCompatibilityTestBase::ClientData{
      std::move(client), std::move(rep_fn), std::move(size_fn)};
}

HloRunnerAgnosticTestBaseOptions BuildOptions() {
  HloRunnerAgnosticTestBaseOptions new_options;
  new_options.swallow_execution_errors =
      HasPjRtAotAwareSwallowExecutionErrors();
  return new_options;
}

}  // namespace

AotCompatibilityTestBase::AotCompatibilityTestBase(std::string artifact_dir)
    : AotCompatibilityTestBase(
          CreateInterceptedClientData(GetParam(), artifact_dir)) {}

AotCompatibilityTestBase::AotCompatibilityTestBase(ClientData data)
    : HloRunnerAgnosticTestBase(MakeAotAwareHloRunner(std::move(data.client)),
                                std::move(data.rep_fn), std::move(data.size_fn),
                                BuildOptions()) {}

}  // namespace xla
