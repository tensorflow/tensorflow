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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/statusor.h"

namespace tsl::testing {
namespace {

TEST(TemporaryDirectoryTest, CreateForCurrentTestcase) {
  std::string path;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        TemporaryDirectory temp_dir,
        TemporaryDirectory::CreateForTestcase(
            *::testing::UnitTest::GetInstance()->current_test_info()));
    path = temp_dir.path();

    tsl::FileStatistics stat;
    ASSERT_OK(tsl::Env::Default()->Stat(temp_dir.path(), &stat));
    EXPECT_TRUE(stat.is_directory);
  }

  tsl::FileStatistics stat;
  EXPECT_THAT(tsl::Env::Default()->Stat(path, &stat),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace tsl::testing
