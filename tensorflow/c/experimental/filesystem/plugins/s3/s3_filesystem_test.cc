/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_filesystem.h"

#include <random>

#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"

#define ASSERT_TF_OK(x) ASSERT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)

static std::string InitializeTmpDir() {
  // This env should be something like `s3://bucket/path`
  const char* test_dir = getenv("S3_TEST_TMPDIR");
  if (test_dir != nullptr) {
    Aws::String bucket, object;
    TF_Status* status = TF_NewStatus();
    ParseS3Path(test_dir, true, &bucket, &object, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_DeleteStatus(status);
      return "";
    }
    TF_DeleteStatus(status);

    // We add a random value into `test_dir` to ensures that two consecutive
    // runs are unlikely to clash.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution;
    std::string rng_val = std::to_string(distribution(gen));
    return tensorflow::io::JoinPath(std::string(test_dir), rng_val);
  } else {
    return "";
  }
}

static std::string* GetTmpDir() {
  static std::string tmp_dir = InitializeTmpDir();
  if (tmp_dir == "")
    return nullptr;
  else
    return &tmp_dir;
}

namespace tensorflow {
namespace {

class S3FilesystemTest : public ::testing::Test {
 public:
  void SetUp() override {
    root_dir_ = io::JoinPath(
        *GetTmpDir(),
        ::testing::UnitTest::GetInstance()->current_test_info()->name());
    status_ = TF_NewStatus();
    filesystem_ = new TF_Filesystem;
    tf_s3_filesystem::Init(filesystem_, status_);
    ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                          << TF_Message(status_);
  }
  void TearDown() override {
    TF_DeleteStatus(status_);
    tf_s3_filesystem::Cleanup(filesystem_);
    delete filesystem_;
  }

  std::string GetURIForPath(const std::string& path) {
    const std::string translated_name =
        tensorflow::io::JoinPath(root_dir_, path);
    return translated_name;
  }

 protected:
  TF_Filesystem* filesystem_;
  TF_Status* status_;

 private:
  std::string root_dir_;
};

TEST_F(S3FilesystemTest, Init) { ASSERT_TF_OK(status_); }

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  if (!GetTmpDir()) {
    std::cerr << "Could not read S3_TEST_TMPDIR env";
    return -1;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
