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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
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

namespace {

absl::StatusOr<std::vector<int32_t>> GetExecutableVersions(
    absl::string_view target_name, AOTTestPlatform platform) {
  std::string dir = GetExecutablesDirectory(target_name, platform);
  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> versions,
                   test_lib_internal::GetExecutableVersionsInDir(dir));
  // On some filesystems a missing directory lists empty rather than failing, so
  // emptiness -- not the status -- is what identifies a bad or ungenerated
  // target.
  if (versions.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No AOT golden artifacts found for target '", target_name,
                     "' under ", dir));
  }
  return versions;
}

// Treats an environment variable as a boolean flag: `=0` must disable golden
// dumping, not enable it. An unparseable value counts as disabled.
bool IsEnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  bool enabled = false;
  return value != nullptr && absl::SimpleAtob(value, &enabled) && enabled;
}

// The undeclared-outputs directory exists only when the golden-update helper
// (`third_party/tensorflow/compiler/xla/tests/aot_compatibility_experimental/google/update_goldens.py`)
// drives the test. Shared so both callers enforce it with the same message.
absl::StatusOr<std::string> GetUndeclaredOutputsDir() {
  const char* out_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (out_dir == nullptr) {
    return absl::FailedPreconditionError(
        "TEST_UNDECLARED_OUTPUTS_DIR is unset. Golden updates must be driven "
        "by the golden-update helper for this package "
        "(Google-internal: third_party/tensorflow/compiler/xla/tests/"
        "aot_compatibility_experimental/google/update_goldens.py), which sets "
        "it; running the test target directly will not work.");
  }
  return std::string(out_dir);
}

}  // namespace

std::string DetectGpuArchToken() {
  if (const char* env_arch = std::getenv("XLA_AOT_GOLDEN_ARCH")) {
    return env_arch;
  }
  absl::StatusOr<stream_executor::Platform*> platform =
      stream_executor::PlatformManager::PlatformWithName("CUDA");
  CHECK_OK(platform.status())
      << "AOT golden arch detection failed to get the CUDA platform; set "
         "XLA_AOT_GOLDEN_ARCH to override.";
  absl::StatusOr<stream_executor::StreamExecutor*> executor =
      (*platform)->ExecutorForDevice(0);
  CHECK_OK(executor.status())
      << "AOT golden arch detection failed to get a device executor; set "
         "XLA_AOT_GOLDEN_ARCH to override.";
  const std::string name =
      absl::AsciiStrToLower((*executor)->GetDeviceDescription().name());

  // Longest token first: "gb200" contains "b200". gb300 is deliberately
  // absent -- it is listed in gpu/BUILD's disabled_backends (b/491194726), so
  // no gb300 goldens exist and no gb300 target is generated.
  for (const absl::string_view arch : {"gb200", "b200", "h100"}) {
    if (absl::StrContains(name, arch)) {
      return std::string(arch);
    }
  }

  LOG(FATAL) << "Unrecognized GPU arch for AOT goldens: " << name
             << "; set XLA_AOT_GOLDEN_ARCH to override.";
}

std::string GetExecutablesDirectory(absl::string_view target_name,
                                    AOTTestPlatform platform) {
  return tsl::io::JoinPath(
      tsl::testing::TensorFlowSrcRoot(),
      absl::StrCat("compiler/xla/tests/aot_compatibility_experimental/",
                   AOTInterceptionPjrtClient::PlatformSubdir(platform),
                   "/executables"),
      target_name);
}

namespace test_lib_internal {

absl::StatusOr<std::vector<int32_t>> GetExecutableVersionsInDir(
    absl::string_view dir) {
  std::vector<std::string> children;
  auto* env = tsl::Env::Default();
  ABSL_RETURN_IF_ERROR(env->GetChildren(std::string(dir), &children));

  std::vector<int32_t> versions;
  versions.reserve(children.size());
  for (const std::string& child : children) {
    // Skip anything that is not a `v<N>` directory holding a golden; a stray or
    // empty entry must not wedge test discovery for the whole target.
    int32_t version;
    if (!absl::StartsWith(child, "v") ||
        !absl::SimpleAtoi(absl::string_view(child).substr(1), &version)) {
      continue;
    }
    std::vector<std::string> files;
    if (!env->GetChildren(tsl::io::JoinPath(dir, child), &files).ok() ||
        std::none_of(files.begin(), files.end(), [](absl::string_view f) {
          return absl::EndsWith(f, ".pbtxt");
        })) {
      continue;
    }
    versions.push_back(version);
  }

  std::sort(versions.begin(), versions.end());
  return versions;
}

}  // namespace test_lib_internal

absl::StatusOr<std::vector<AotTestParam>>
GetAotTestParamsForBackwardsCompatibility(absl::string_view target_name,
                                          AOTTestPlatform platform) {
  if (IsEnvFlagEnabled("XLA_AOT_UPDATE_GOLDENS")) {
    return std::vector<AotTestParam>{};
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> versions,
                   GetExecutableVersions(target_name, platform));

  if (!IsEnvFlagEnabled("XLA_AOT_TEST_ALL_VERSIONS") && versions.size() > 2) {
    // The oldest and second-newest versions bound our compatibility guarantee.
    // This is positional, not arithmetic: for v1, v2, v5 the pair is {v1, v2}.
    // The newest is covered by golden file verification instead.
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
  if (IsEnvFlagEnabled("XLA_AOT_UPDATE_GOLDENS")) {
    ABSL_RETURN_IF_ERROR(GetUndeclaredOutputsDir().status());
    return std::vector<AotTestParam>{
        {AOTTestMode::kUpdateGolden, 0, std::string(target_name)}};
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> versions,
                   GetExecutableVersions(target_name, platform));

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
            // gtest names a parameterized case `<test>/<instantiation>`; the
            // golden is named after the test alone.
            absl::string_view name =
                test_info == nullptr ? "" : test_info->name();
            std::string test_name(name.substr(0, name.find('/')));
            std::string artifact_path;
            if (param.mode == AOTTestMode::kUpdateGolden) {
              absl::StatusOr<std::string> out_dir = GetUndeclaredOutputsDir();
              CHECK_OK(out_dir.status());
              artifact_path = tsl::io::JoinPath(
                  *out_dir, absl::StrCat(test_name, ".pbtxt"));
            } else {
              artifact_path = tsl::io::JoinPath(
                  GetExecutablesDirectory(param.target_name, platform.value()),
                  absl::StrCat("v", param.version),
                  absl::StrCat(test_name, ".pbtxt"));
            }
            return std::make_unique<AOTInterceptionPjrtClient>(
                std::move(*client), param.mode, artifact_path);
          }(param),
          HloTestBaseOptions()) {}

DebugOptions AotCompatibilityTest::GetDebugOptionsForTest() const {
  DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
  AOTInterceptionPjrtClient::ApplyAotDeterminismDebugOptions(debug_options);
  return debug_options;
}

}  // namespace aot_compatibility_experimental
}  // namespace xla
