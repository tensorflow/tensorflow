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
#include "third_party/hadoop/hdfs.h"

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

TEST_F(HadoopFileSystemTest, DeleteFile) {
  const std::string path = TmpDir("DeleteFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  tf_hadoop_filesystem::DeleteFile(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(HadoopFileSystemTest, GetFileSize) {
  const std::string path = TmpDir("GetFileSize");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  auto file_size =
      tf_hadoop_filesystem::GetFileSize(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(4, file_size);
}

TEST_F(HadoopFileSystemTest, CreateDirStat) {
  const std::string path = TmpDir("CreateDirStat");
  tf_hadoop_filesystem::CreateDir(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
  TF_FileStatistics stat;
  tf_hadoop_filesystem::Stat(filesystem_, path.c_str(), &stat, status_);
  EXPECT_TF_OK(status_);
  EXPECT_TRUE(stat.is_directory);
}

TEST_F(HadoopFileSystemTest, DeleteDir) {
  const std::string path = TmpDir("DeleteDir");
  tf_hadoop_filesystem::DeleteDir(filesystem_, path.c_str(), status_);
  EXPECT_NE(TF_GetCode(status_), TF_OK);
  tf_hadoop_filesystem::CreateDir(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_hadoop_filesystem::DeleteDir(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
  TF_FileStatistics stat;
  tf_hadoop_filesystem::Stat(filesystem_, path.c_str(), &stat, status_);
  EXPECT_NE(TF_GetCode(status_), TF_OK);
}

TEST_F(HadoopFileSystemTest, RenameFile) {
  const std::string src = TmpDir("RenameFileSrc");
  const std::string dst = TmpDir("RenameFileDst");
  WriteString(src, "test");
  ASSERT_TF_OK(status_);

  tf_hadoop_filesystem::RenameFile(filesystem_, src.c_str(), dst.c_str(),
                                   status_);
  EXPECT_TF_OK(status_);
  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("test", result);
}

TEST_F(HadoopFileSystemTest, RenameFileOverwrite) {
  const std::string src = TmpDir("RenameFileOverwriteSrc");
  const std::string dst = TmpDir("RenameFileOverwriteDst");

  WriteString(src, "test_old");
  ASSERT_TF_OK(status_);
  WriteString(dst, "test_new");
  ASSERT_TF_OK(status_);

  tf_hadoop_filesystem::PathExists(filesystem_, dst.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_hadoop_filesystem::RenameFile(filesystem_, src.c_str(), dst.c_str(),
                                   status_);
  EXPECT_TF_OK(status_);

  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("test_old", result);
}

TEST_F(HadoopFileSystemTest, StatFile) {
  const std::string path = TmpDir("StatFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  TF_FileStatistics stat;
  tf_hadoop_filesystem::Stat(filesystem_, path.c_str(), &stat, status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(4, stat.length);
  EXPECT_FALSE(stat.is_directory);
}

TEST_F(HadoopFileSystemTest, WriteWhileReading) {
  const std::string path = TmpDir("WriteWhileReading");
  // Skip the test if we're not testing on HDFS. Hadoop's local filesystem
  // implementation makes no guarantees that writable files are readable while
  // being written.
  if (path.find_first_of("hdfs://") != 0) GTEST_SKIP();

  auto writer = GetWriter();
  tf_hadoop_filesystem::NewWritableFile(filesystem_, path.c_str(), writer.get(),
                                        status_);
  EXPECT_TF_OK(status_);

  const std::string content1 = "content1";
  tf_writable_file::Append(writer.get(), content1.c_str(), content1.size(),
                           status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Flush(writer.get(), status_);
  EXPECT_TF_OK(status_);

  auto reader = GetReader();
  tf_hadoop_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                            reader.get(), status_);
  EXPECT_TF_OK(status_);

  std::string result;
  result.resize(content1.size());
  auto read = tf_random_access_file::Read(reader.get(), 0, content1.size(),
                                          &result[0], status_);
  result.resize(read);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(content1, result);

  const std::string content2 = "content2";
  tf_writable_file::Append(writer.get(), content2.c_str(), content2.size(),
                           status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Flush(writer.get(), status_);
  EXPECT_TF_OK(status_);

  result.resize(content2.size());
  read = tf_random_access_file::Read(reader.get(), content1.size(),
                                     content2.size(), &result[0], status_);
  result.resize(read);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(content2, result);

  tf_writable_file::Close(writer.get(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(HadoopFileSystemTest, ReadWhileOverwriting) {
  static char set_disable_var[] = "HDFS_DISABLE_READ_EOF_RETRIED=1";
  putenv(set_disable_var);

  const std::string path = TmpDir("ReadWhileOverwriting");
  if (path.find_first_of("hdfs://") != 0) GTEST_SKIP();

  const string content1 = "content1";
  WriteString(path, content1);
  ASSERT_TF_OK(status_);

  auto reader = GetReader();
  tf_hadoop_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                            reader.get(), status_);
  EXPECT_TF_OK(status_);

  std::string result;
  result.resize(content1.size());
  auto read = tf_random_access_file::Read(reader.get(), 0, content1.size(),
                                          &result[0], status_);
  result.resize(read);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(content1, result);

  tf_hadoop_filesystem::DeleteFile(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);

  string content2 = "overwrite";
  WriteString(path, content1 + content2);
  ASSERT_TF_OK(status_);

  result.resize(content2.size());
  read = tf_random_access_file::Read(reader.get(), content1.size(),
                                     content2.size(), &result[0], status_);
  result.resize(read);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(0, result.size());

  static char set_enable_var[] = "HDFS_DISABLE_READ_EOF_RETRIED=0";
  putenv(set_enable_var);
}

TEST_F(HadoopFileSystemTest, HarSplit) {
  const std::string har_path =
      "har://hdfs-root/user/j.doe/my_archive.har/dir0/dir1/file.txt";
  std::string scheme, namenode, path;
  ParseHadoopPath(har_path, &scheme, &namenode, &path);
  EXPECT_EQ("har", scheme);
  EXPECT_EQ("hdfs-root", namenode);
  EXPECT_EQ("/user/j.doe/my_archive.har/dir0/dir1/file.txt", path);
  SplitArchiveNameAndPath(&path, &namenode, status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("har://hdfs-root/user/j.doe/my_archive.har", namenode);
  EXPECT_EQ("/dir0/dir1/file.txt", path);
}

TEST_F(HadoopFileSystemTest, NoHarExtension) {
  const std::string har_path =
      "har://hdfs-root/user/j.doe/my_archive/dir0/dir1/file.txt";
  std::string scheme, namenode, path;
  ParseHadoopPath(har_path, &scheme, &namenode, &path);
  EXPECT_EQ("har", scheme);
  EXPECT_EQ("hdfs-root", namenode);
  EXPECT_EQ("/user/j.doe/my_archive/dir0/dir1/file.txt", path);
  SplitArchiveNameAndPath(&path, &namenode, status_);
  EXPECT_EQ(TF_GetCode(status_), TF_INVALID_ARGUMENT) << TF_Message(status_);
}

TEST_F(HadoopFileSystemTest, HarRootPath) {
  const std::string har_path = "har://hdfs-root/user/j.doe/my_archive.har";
  std::string scheme, namenode, path;
  ParseHadoopPath(har_path, &scheme, &namenode, &path);
  EXPECT_EQ("har", scheme);
  EXPECT_EQ("hdfs-root", namenode);
  EXPECT_EQ("/user/j.doe/my_archive.har", path);
  SplitArchiveNameAndPath(&path, &namenode, status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("har://hdfs-root/user/j.doe/my_archive.har", namenode);
  EXPECT_EQ("/", path);
}

TEST_F(HadoopFileSystemTest, WriteLargeFile) {
  if (std::getenv("HADOOP_TEST_LARGE_FILE") != "1") GTEST_SKIP();
  const std::string path = TmpDir("WriteLargeFile");
  const size_t file_size =
      static_cast<size_t>(std::numeric_limits<tSize>::max()) + 1024;
  // Fake a test string.
  std::string source(file_size, {});
  for (size_t i = 0; i < file_size; ++i) source[i] = (i % 128);
  WriteString(path, source);
  ASSERT_TF_OK(status_);
  auto result = ReadAll(path);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(source, result);
}
// NewAppendableFile() is not testable. Local filesystem maps to
// ChecksumFileSystem in Hadoop, where appending is an unsupported operation.

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
