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
#include "tensorflow/c/experimental/filesystem/plugins/gcs/gcs_filesystem.h"

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"

#define ASSERT_TF_OK(x) ASSERT_EQ(TF_OK, TF_GetCode(x))

namespace tensorflow {
namespace {

class GCSFilesystemTest : public ::testing::Test {
 public:
  void SetUp() override {
    status_ = TF_NewStatus();
    filesystem_ = new TF_Filesystem;
    tf_gcs_filesystem::Init(filesystem_, status_);
    ASSERT_TF_OK(status_) << "Can not initialize filesystem. "
                          << TF_Message(status_);
  }
  void TearDown() override {
    TF_DeleteStatus(status_);
    tf_gcs_filesystem::Cleanup(filesystem_);
    delete filesystem_;
  }

 protected:
  TF_Filesystem* filesystem_;
  TF_Status* status_;
};

TEST_F(GCSFilesystemTest, ParseGCSPath) {
  std::string bucket, object;
  ParseGCSPath("gs://bucket/path/to/object", false, &bucket, &object, status_);
  ASSERT_TF_OK(status_);
  ASSERT_EQ(bucket, "bucket");
  ASSERT_EQ(object, "path/to/object");

  ParseGCSPath("gs://bucket/", true, &bucket, &object, status_);
  ASSERT_TF_OK(status_);
  ASSERT_EQ(bucket, "bucket");

  ParseGCSPath("bucket/path/to/object", false, &bucket, &object, status_);
  ASSERT_EQ(TF_GetCode(status_), TF_INVALID_ARGUMENT);

  // bucket name must end with "/"
  ParseGCSPath("gs://bucket", true, &bucket, &object, status_);
  ASSERT_EQ(TF_GetCode(status_), TF_INVALID_ARGUMENT);

  ParseGCSPath("gs://bucket/", false, &bucket, &object, status_);
  ASSERT_EQ(TF_GetCode(status_), TF_INVALID_ARGUMENT);
}

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
