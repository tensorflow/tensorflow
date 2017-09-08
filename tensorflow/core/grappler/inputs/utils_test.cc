/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class UtilsTest : public ::testing::Test {
 protected:
  string BaseDir() { return io::JoinPath(testing::TmpDir(), "base_dir"); }

  void SetUp() override {
    TF_CHECK_OK(env_->CreateDir(BaseDir()));
    non_existent_file_ = io::JoinPath(BaseDir(), "non_existent_file.txt");
    actual_file_ = io::JoinPath(BaseDir(), "test_file.txt");
    TF_CHECK_OK(WriteStringToFile(env_, actual_file_, "Some test data"));
  }

  void TearDown() override {
    int64 undeleted_files, undeleted_dirs;
    TF_CHECK_OK(
        env_->DeleteRecursively(BaseDir(), &undeleted_files, &undeleted_dirs));
  }

  string non_existent_file_;
  string actual_file_;
  Env* env_ = Env::Default();
};

TEST_F(UtilsTest, FilesExist) {
  EXPECT_FALSE(FilesExist(std::vector<string>{{non_existent_file_}}));
  EXPECT_FALSE(
      FilesExist(std::vector<string>{{non_existent_file_}, {actual_file_}}));
  EXPECT_TRUE(FilesExist(std::vector<string>{{actual_file_}}));

  std::vector<Status> status;
  EXPECT_FALSE(FilesExist(
      std::vector<string>{{non_existent_file_}, {actual_file_}}, &status));
  EXPECT_EQ(status.size(), 2);
  EXPECT_FALSE(status[0].ok());
  EXPECT_TRUE(status[1].ok());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
