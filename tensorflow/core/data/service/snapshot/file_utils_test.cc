/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/file_utils.h"

#include <string>

#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

tsl::StatusOr<std::string> CreateTestDirectory() {
  std::string directory;
  if (!tsl::Env::Default()->LocalTempFilename(&directory)) {
    return tsl::errors::FailedPrecondition(
        "Failed to create local test directory.");
  }
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(directory));
  return directory;
}

using AtomicallyWriteStringToFileTest = ::testing::TestWithParam<std::string>;

TEST_P(AtomicallyWriteStringToFileTest, WriteString) {
  TF_ASSERT_OK_AND_ASSIGN(std::string directory, CreateTestDirectory());
  std::string test_file = tsl::io::JoinPath(directory, "test_file");
  std::string file_contents = GetParam();
  TF_ASSERT_OK(AtomicallyWriteStringToFile(test_file, file_contents,
                                           tsl::Env::Default()));

  std::string data;
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(test_file));
  TF_ASSERT_OK(ReadFileToString(tsl::Env::Default(), test_file, &data));
  EXPECT_EQ(data, file_contents);
}

INSTANTIATE_TEST_SUITE_P(FileContents, AtomicallyWriteStringToFileTest,
                         ::testing::ValuesIn<std::string>({"OK", ""}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
