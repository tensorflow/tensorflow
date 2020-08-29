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

  void WriteString(const std::string& path, const std::string& content) {
    auto writer = GetWriter();
    tf_hadoop_filesystem::NewWritableFile(filesystem_, path.c_str(),
                                          writer.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return;
    tf_writable_file::Append(writer.get(), content.c_str(), content.length(),
                             status_);
    if (TF_GetCode(status_) != TF_OK) return;
    tf_writable_file::Close(writer.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return;
  }

  std::string ReadAll(const std::string& path) {
    auto reader = GetReader();
    tf_hadoop_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                              reader.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return "";

    auto file_size =
        tf_hadoop_filesystem::GetFileSize(filesystem_, path.c_str(), status_);
    if (TF_GetCode(status_) != TF_OK) return "";

    std::string content;
    content.resize(file_size);
    auto read = tf_random_access_file::Read(reader.get(), 0, file_size,
                                            &content[0], status_);
    if (TF_GetCode(status_) != TF_OK) return "";
    if (read >= 0) content.resize(read);
    if (file_size != content.size())
      TF_SetStatus(
          status_, TF_DATA_LOSS,
          std::string("expected " + std::to_string(file_size) + " got " +
                      std::to_string(content.size()) + " bytes")
              .c_str());
    return content;
  }

 protected:
  TF_Filesystem* filesystem_;
  TF_Status* status_;
};

TEST_F(HadoopFileSystemTest, RandomAccessFile) {
  const std::string path = TmpDir("RandomAccessFile");
  const std::string content = "abcdefghijklmn";

  WriteString(path, content);
  ASSERT_TF_OK(status_);

  auto reader = GetReader();
  tf_hadoop_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                            reader.get(), status_);
  EXPECT_TF_OK(status_);

  std::string result;
  result.resize(content.size());
  auto read = tf_random_access_file::Read(reader.get(), 0, content.size(),
                                          &result[0], status_);
  result.resize(read);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(content.size(), result.size());
  EXPECT_EQ(content, result);

  result.clear();
  result.resize(4);
  read = tf_random_access_file::Read(reader.get(), 2, 4, &result[0], status_);
  result.resize(read);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(4, result.size());
  EXPECT_EQ(content.substr(2, 4), result);
}

TEST_F(HadoopFileSystemTest, WritableFile) {
  auto writer = GetWriter();
  const std::string path = TmpDir("WritableFile");
  tf_hadoop_filesystem::NewWritableFile(filesystem_, path.c_str(), writer.get(),
                                        status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Append(writer.get(), "content1,", strlen("content1,"),
                           status_);
  EXPECT_TF_OK(status_);
  auto pos = tf_writable_file::Tell(writer.get(), status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(pos, 9);

  tf_writable_file::Append(writer.get(), "content2", strlen("content2"),
                           status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Flush(writer.get(), status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Sync(writer.get(), status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Close(writer.get(), status_);
  EXPECT_TF_OK(status_);

  auto content = ReadAll(path);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("content1,content2", content);
}

TEST_F(HadoopFileSystemTest, PathExists) {
  const std::string path = TmpDir("PathExists");
  tf_hadoop_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_EQ(TF_NOT_FOUND, TF_GetCode(status_)) << TF_Message(status_);
  TF_SetStatus(status_, TF_OK, "");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  tf_hadoop_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(HadoopFileSystemTest, GetChildren) {
  const std::string base = TmpDir("GetChildren");
  tf_hadoop_filesystem::CreateDir(filesystem_, base.c_str(), status_);
  EXPECT_TF_OK(status_);

  const std::string file = io::JoinPath(base, "TestFile.csv");
  WriteString(file, "test");
  EXPECT_TF_OK(status_);

  const std::string subdir = io::JoinPath(base, "SubDir");
  tf_hadoop_filesystem::CreateDir(filesystem_, subdir.c_str(), status_);
  EXPECT_TF_OK(status_);
  const std::string subfile = io::JoinPath(subdir, "TestSubFile.csv");
  WriteString(subfile, "test");
  EXPECT_TF_OK(status_);

  char** entries;
  auto num_entries = tf_hadoop_filesystem::GetChildren(
      filesystem_, base.c_str(), &entries, status_);
  EXPECT_TF_OK(status_);

  std::vector<std::string> childrens;
  for (int i = 0; i < num_entries; ++i) {
    childrens.push_back(entries[i]);
  }
  std::sort(childrens.begin(), childrens.end());
  EXPECT_EQ(std::vector<string>({"SubDir", "TestFile.csv"}), childrens);
}

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
