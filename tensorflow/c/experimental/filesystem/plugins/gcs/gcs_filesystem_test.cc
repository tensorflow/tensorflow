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

#include <random>

#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"

#define ASSERT_TF_OK(x) ASSERT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)

static const char* content = "abcdefghijklmnopqrstuvwxyz1234567890";
// We will work with content_view instead of content.
static const absl::string_view content_view = content;

namespace gcs = google::cloud::storage;

namespace tensorflow {
namespace {

class GCSFilesystemTest : public ::testing::Test {
 public:
  void SetUp() override {
    root_dir_ = io::JoinPath(
        tmp_dir_,
        ::testing::UnitTest::GetInstance()->current_test_info()->name());
    status_ = TF_NewStatus();
    filesystem_ = new TF_Filesystem;
    tf_gcs_filesystem::Init(filesystem_, status_);
    ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                          << TF_Message(status_);
  }
  void TearDown() override {
    TF_DeleteStatus(status_);
    tf_gcs_filesystem::Cleanup(filesystem_);
    delete filesystem_;
  }

  static bool InitializeTmpDir() {
    // This env should be something like `gs://bucket/path`
    const char* test_dir = getenv("GCS_TEST_TMPDIR");
    if (test_dir != nullptr) {
      std::string bucket, object;
      TF_Status* status = TF_NewStatus();
      ParseGCSPath(test_dir, true, &bucket, &object, status);
      if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        return false;
      }
      TF_DeleteStatus(status);

      // We add a random value into `test_dir` to ensures that two consecutive
      // runs are unlikely to clash.
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distribution;
      std::string rng_val = std::to_string(distribution(gen));
      tmp_dir_ = io::JoinPath(string(test_dir), rng_val);
      return true;
    } else {
      return false;
    }
  }

  std::string GetURIForPath(absl::string_view path) {
    const std::string translated_name =
        tensorflow::io::JoinPath(root_dir_, path);
    return translated_name;
  }

 protected:
  TF_Filesystem* filesystem_;
  TF_Status* status_;

 private:
  std::string root_dir_;
  static std::string tmp_dir_;
};
std::string GCSFilesystemTest::tmp_dir_;

::testing::AssertionResult WriteToServer(const std::string& path, size_t length,
                                         gcs::Client* gcs_client,
                                         TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) {
    return ::testing::AssertionFailure() << TF_Message(status);
  }

  auto writer = gcs_client->WriteObject(bucket, object);
  writer.write(content, length);
  writer.Close();
  if (writer.metadata()) {
    return ::testing::AssertionSuccess();
  } else {
    return ::testing::AssertionFailure()
           << writer.metadata().status().message();
  }
}

::testing::AssertionResult CompareSubString(int64_t offset, size_t n,
                                            absl::string_view result,
                                            size_t read) {
  // Result isn't a null-terminated string so we have to wrap it inside a
  // `string_view`
  if (n == read && content_view.substr(offset, n) ==
                       absl::string_view(result).substr(0, read)) {
    return ::testing::AssertionSuccess();
  } else {
    return ::testing::AssertionFailure()
           << "Result: " << absl::string_view(result).substr(0, read)
           << " Read:" << read;
  }
}

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

TEST_F(GCSFilesystemTest, RandomAccessFile) {
  std::string filepath = GetURIForPath("a_file");
  TF_RandomAccessFile* file = new TF_RandomAccessFile;
  tf_gcs_filesystem::NewRandomAccessFile(filesystem_, filepath.c_str(), file,
                                         status_);
  ASSERT_TF_OK(status_);
  char* result = new char[content_view.length()];
  int64_t read = tf_random_access_file::Read(file, 0, 1, result, status_);
  ASSERT_EQ(read, -1) << "Read: " << read;
  ASSERT_EQ(TF_GetCode(status_), TF_NOT_FOUND) << TF_Message(status_);
  TF_SetStatus(status_, TF_OK, "");

  auto gcs_client = static_cast<gcs::Client*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(
      WriteToServer(filepath, content_view.length(), gcs_client, status_));

  read = tf_random_access_file::Read(file, 0, content_view.length(), result,
                                     status_);
  ASSERT_TF_OK(status_);
  ASSERT_TRUE(CompareSubString(0, content_view.length(), result, read));

  read = tf_random_access_file::Read(file, 0, 4, result, status_);
  ASSERT_TF_OK(status_);
  ASSERT_TRUE(CompareSubString(0, 4, result, read));

  read = tf_random_access_file::Read(file, content_view.length() - 2, 4, result,
                                     status_);
  ASSERT_EQ(TF_GetCode(status_), TF_OUT_OF_RANGE) << TF_Message(status_);
  ASSERT_TRUE(CompareSubString(content_view.length() - 2, 2, result, read));

  delete result;
  tf_random_access_file::Cleanup(file);
  delete file;
}

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  if (!tensorflow::GCSFilesystemTest::InitializeTmpDir()) {
    std::cerr << "Could not read GCS_TEST_TMPDIR env";
    return -1;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
