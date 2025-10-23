/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/testing/temporary_directory.h"

#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace testing {

absl::StatusOr<TemporaryDirectory> TemporaryDirectory::CreateForTestcase(
    const ::testing::TestInfo& test_info) {
  std::string path =
      tsl::io::JoinPath(::testing::TempDir(), "xla_testing_tmp",
                        test_info.test_suite_name(), test_info.name());
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(path));
  return TemporaryDirectory(std::move(path));
}

absl::StatusOr<TemporaryDirectory>
TemporaryDirectory::CreateForCurrentTestcase() {
  auto unit_test = ::testing::UnitTest::GetInstance();
  if (unit_test->current_test_info() == nullptr) {
    return absl::FailedPreconditionError(
        "No current test info. Are you calling this from a non-test "
        "environment?");
  }
  return CreateForTestcase(*unit_test->current_test_info());
}

void TemporaryDirectory::RecursiveFilepathDeleter::operator()(
    std::string* path) const {
  if (path == nullptr) {
    return;
  }

  int64_t undeleted_files, undeleted_dirs;
  absl::Status status = tsl::Env::Default()->DeleteRecursively(
      *path, &undeleted_files, &undeleted_dirs);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to delete temporary directory " << path << ": "
                 << status;
  }

  delete path;
}
}  // namespace testing
}  // namespace tsl
