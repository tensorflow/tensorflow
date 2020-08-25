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
#include "tensorflow/c/experimental/filesystem/plugins/hadoop/hadoop_filesystem.h"

#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"

#define ASSERT_TF_OK(x) ASSERT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)
#define EXPECT_TF_OK(x) EXPECT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)

namespace tensorflow {
namespace {

class HadoopFileSystemTest : public ::testing::Test {
 public:
  void SetUp() override {
    status_ = TF_NewStatus();
    filesystem_ = new TF_Filesystem;
    tf_hadoop_filesystem::Init(filesystem_, status_);
    ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                          << TF_Message(status_);
  }
  void TearDown() override {
    TF_DeleteStatus(status_);
    tf_hadoop_filesystem::Cleanup(filesystem_);
    delete filesystem_;
  }

  std::string TmpDir(const std::string& path) {
    char* test_dir = getenv("HADOOP_TEST_TMPDIR");
    if (test_dir != nullptr) {
      return io::JoinPath(std::string(test_dir), path);
    } else {
      return "file://" + io::JoinPath(testing::TmpDir(), path);
    }
  }

  std::unique_ptr<TF_WritableFile, void (*)(TF_WritableFile* file)>
  GetWriter() {
    std::unique_ptr<TF_WritableFile, void (*)(TF_WritableFile * file)> writer(
        new TF_WritableFile, [](TF_WritableFile* file) {
          if (file != nullptr) {
            if (file->plugin_file != nullptr) tf_writable_file::Cleanup(file);
            delete file;
          }
        });
    writer->plugin_file = nullptr;
    return writer;
  }

  std::unique_ptr<TF_RandomAccessFile, void (*)(TF_RandomAccessFile* file)>
  GetReader() {
    std::unique_ptr<TF_RandomAccessFile, void (*)(TF_RandomAccessFile * file)>
        reader(new TF_RandomAccessFile, [](TF_RandomAccessFile* file) {
          if (file != nullptr) {
            if (file->plugin_file != nullptr)
              tf_random_access_file::Cleanup(file);
            delete file;
          }
        });
    reader->plugin_file = nullptr;
    return reader;
  }

 protected:
  TF_Filesystem* filesystem_;
  TF_Status* status_;
};

TEST_F(HadoopFileSystemTest, Init) { ASSERT_TF_OK(status_); }

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
