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

#include <fstream>
#include <random>

#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"

#define ASSERT_TF_OK(x) ASSERT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)
#define EXPECT_TF_OK(x) EXPECT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)

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

static std::string GetLocalLargeFile() {
  // This env is used when we want to test against a large file ( ~  50MB ).
  // `S3_TEST_LOCAL_LARGE_FILE` and `S3_TEST_SERVER_LARGE_FILE` must be the same
  // file.
  static std::string path;
  if (path.empty()) {
    const char* env = getenv("S3_TEST_LOCAL_LARGE_FILE");
    if (env == nullptr) return "";
    path = env;
  }
  return path;
}

static std::string GetServerLargeFile() {
  // This env is used when we want to test against a large file ( ~  50MB ).
  // `S3_TEST_LOCAL_LARGE_FILE` and `S3_TEST_SERVER_LARGE_FILE` must be the same
  // file.
  static std::string path;
  if (path.empty()) {
    const char* env = getenv("S3_TEST_SERVER_LARGE_FILE");
    if (env == nullptr) return "";
    Aws::String bucket, object;
    TF_Status* status = TF_NewStatus();
    ParseS3Path(env, false, &bucket, &object, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_DeleteStatus(status);
      return "";
    }
    TF_DeleteStatus(status);
    path = env;
  }
  return path;
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
    tf_s3_filesystem::NewWritableFile(filesystem_, path.c_str(), writer.get(),
                                      status_);
    if (TF_GetCode(status_) != TF_OK) return;
    tf_writable_file::Append(writer.get(), content.c_str(), content.length(),
                             status_);
    if (TF_GetCode(status_) != TF_OK) return;
    tf_writable_file::Close(writer.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return;
  }

  std::string ReadAll(const string& path) {
    auto reader = GetReader();
    tf_s3_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                          reader.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return "";

    auto file_size =
        tf_s3_filesystem::GetFileSize(filesystem_, path.c_str(), status_);
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

  std::string ReadAllInChunks(const string& path, size_t buffer_size,
                              bool use_multi_part_download) {
    auto reader = GetReader();
    auto s3_file =
        static_cast<tf_s3_filesystem::S3File*>(filesystem_->plugin_filesystem);
    s3_file->use_multi_part_download = use_multi_part_download;
    s3_file
        ->multi_part_chunk_sizes[Aws::Transfer::TransferDirection::DOWNLOAD] =
        buffer_size;
    tf_s3_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                          reader.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return "";
    auto file_size =
        tf_s3_filesystem::GetFileSize(filesystem_, path.c_str(), status_);
    if (TF_GetCode(status_) != TF_OK) return "";

    std::size_t part_count = (std::max)(
        static_cast<size_t>((file_size + buffer_size - 1) / buffer_size),
        static_cast<size_t>(1));
    std::unique_ptr<char[]> buffer{new char[buffer_size]};
    std::stringstream ss;

    uint64_t offset = 0;
    uint64_t server_size = 0;
    for (size_t i = 0; i < part_count; i++) {
      offset = i * buffer_size;
      buffer_size =
          (i == part_count - 1) ? file_size - server_size : buffer_size;
      auto read = tf_random_access_file::Read(reader.get(), offset, buffer_size,
                                              buffer.get(), status_);
      if (TF_GetCode(status_) != TF_OK) return "";
      if (read > 0) {
        ss.write(buffer.get(), read);
        server_size += static_cast<uint64_t>(read);
      }
      if (server_size == file_size) break;
      if (read != buffer_size) {
        if (read == 0)
          TF_SetStatus(status_, TF_OUT_OF_RANGE, "eof");
        else
          TF_SetStatus(
              status_, TF_DATA_LOSS,
              ("truncated record at " + std::to_string(offset)).c_str());
        return "";
      }
    }

    if (file_size != server_size) {
      TF_SetStatus(status_, TF_DATA_LOSS,
                   std::string("expected " + std::to_string(file_size) +
                               " got " + std::to_string(server_size) + " bytes")
                       .c_str());
      return "";
    }
    TF_SetStatus(status_, TF_OK, "");
    return ss.str();
  }

 protected:
  TF_Filesystem* filesystem_;
  TF_Status* status_;

 private:
  std::string root_dir_;
};

TEST_F(S3FilesystemTest, NewRandomAccessFile) {
  const std::string path = GetURIForPath("RandomAccessFile");
  const std::string content = "abcdefghijklmn";

  WriteString(path, content);
  ASSERT_TF_OK(status_);

  auto reader = GetReader();
  tf_s3_filesystem::NewRandomAccessFile(filesystem_, path.c_str(), reader.get(),
                                        status_);
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

TEST_F(S3FilesystemTest, NewWritableFile) {
  auto writer = GetWriter();
  const std::string path = GetURIForPath("WritableFile");
  tf_s3_filesystem::NewWritableFile(filesystem_, path.c_str(), writer.get(),
                                    status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Append(writer.get(), "content1,", strlen("content1,"),
                           status_);
  EXPECT_TF_OK(status_);
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

TEST_F(S3FilesystemTest, NewAppendableFile) {
  const std::string path = GetURIForPath("AppendableFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);

  auto writer = GetWriter();
  tf_s3_filesystem::NewAppendableFile(filesystem_, path.c_str(), writer.get(),
                                      status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Append(writer.get(), "content", strlen("content"), status_);
  EXPECT_TF_OK(status_);
  tf_writable_file::Close(writer.get(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(S3FilesystemTest, NewReadOnlyMemoryRegionFromFile) {
  const std::string path = GetURIForPath("MemoryFile");
  const std::string content = "content";
  WriteString(path, content);
  ASSERT_TF_OK(status_);

  std::unique_ptr<TF_ReadOnlyMemoryRegion,
                  void (*)(TF_ReadOnlyMemoryRegion * file)>
      region(new TF_ReadOnlyMemoryRegion, [](TF_ReadOnlyMemoryRegion* file) {
        if (file != nullptr) {
          if (file->plugin_memory_region != nullptr)
            tf_read_only_memory_region::Cleanup(file);
          delete file;
        }
      });
  region->plugin_memory_region = nullptr;
  tf_s3_filesystem::NewReadOnlyMemoryRegionFromFile(filesystem_, path.c_str(),
                                                    region.get(), status_);
  EXPECT_TF_OK(status_);
  std::string result(reinterpret_cast<const char*>(
                         tf_read_only_memory_region::Data(region.get())),
                     tf_read_only_memory_region::Length(region.get()));
  EXPECT_EQ(content, result);
}

TEST_F(S3FilesystemTest, PathExists) {
  const std::string path = GetURIForPath("PathExists");
  tf_s3_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_EQ(TF_NOT_FOUND, TF_GetCode(status_)) << TF_Message(status_);
  TF_SetStatus(status_, TF_OK, "");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  tf_s3_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(S3FilesystemTest, GetChildren) {
  const std::string base = GetURIForPath("GetChildren");
  tf_s3_filesystem::CreateDir(filesystem_, base.c_str(), status_);
  EXPECT_TF_OK(status_);

  const std::string file = io::JoinPath(base, "TestFile.csv");
  WriteString(file, "test");
  EXPECT_TF_OK(status_);

  const std::string subdir = io::JoinPath(base, "SubDir");
  tf_s3_filesystem::CreateDir(filesystem_, subdir.c_str(), status_);
  EXPECT_TF_OK(status_);
  const std::string subfile = io::JoinPath(subdir, "TestSubFile.csv");
  WriteString(subfile, "test");
  EXPECT_TF_OK(status_);

  char** entries;
  auto num_entries = tf_s3_filesystem::GetChildren(filesystem_, base.c_str(),
                                                   &entries, status_);
  EXPECT_TF_OK(status_);

  std::vector<std::string> childrens;
  for (int i = 0; i < num_entries; ++i) {
    childrens.push_back(entries[i]);
  }
  std::sort(childrens.begin(), childrens.end());
  EXPECT_EQ(std::vector<string>({"SubDir", "TestFile.csv"}), childrens);
}

TEST_F(S3FilesystemTest, DeleteFile) {
  const std::string path = GetURIForPath("DeleteFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  tf_s3_filesystem::DeleteFile(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(S3FilesystemTest, CreateDir) {
  // s3 object storage doesn't support empty directory, we create file in the
  // directory
  const std::string dir = GetURIForPath("CreateDir");
  tf_s3_filesystem::CreateDir(filesystem_, dir.c_str(), status_);
  EXPECT_TF_OK(status_);

  const std::string file = io::JoinPath(dir, "CreateDirFile.csv");
  WriteString(file, "test");
  ASSERT_TF_OK(status_);

  TF_FileStatistics stat;
  tf_s3_filesystem::Stat(filesystem_, dir.c_str(), &stat, status_);
  EXPECT_TF_OK(status_);
  EXPECT_TRUE(stat.is_directory);
}

TEST_F(S3FilesystemTest, DeleteDir) {
  // s3 object storage doesn't support empty directory, we create file in the
  // directory
  const std::string dir = GetURIForPath("DeleteDir");
  const std::string file = io::JoinPath(dir, "DeleteDirFile.csv");
  WriteString(file, "test");
  ASSERT_TF_OK(status_);
  tf_s3_filesystem::DeleteDir(filesystem_, dir.c_str(), status_);
  EXPECT_NE(TF_GetCode(status_), TF_OK);

  TF_SetStatus(status_, TF_OK, "");
  tf_s3_filesystem::DeleteFile(filesystem_, file.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_s3_filesystem::DeleteDir(filesystem_, dir.c_str(), status_);
  EXPECT_TF_OK(status_);
  TF_FileStatistics stat;
  tf_s3_filesystem::Stat(filesystem_, dir.c_str(), &stat, status_);
  EXPECT_EQ(TF_GetCode(status_), TF_NOT_FOUND) << TF_Message(status_);
}

TEST_F(S3FilesystemTest, StatFile) {
  const std::string path = GetURIForPath("StatFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);

  TF_FileStatistics stat;
  tf_s3_filesystem::Stat(filesystem_, path.c_str(), &stat, status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(4, stat.length);
  EXPECT_FALSE(stat.is_directory);
}

TEST_F(S3FilesystemTest, SimpleCopyFile) {
  const std::string src = GetURIForPath("SimpleCopySrc");
  const std::string dst = GetURIForPath("SimpleCopyDst");
  WriteString(src, "test");
  ASSERT_TF_OK(status_);

  tf_s3_filesystem::CopyFile(filesystem_, src.c_str(), dst.c_str(), status_);
  EXPECT_TF_OK(status_);
  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(result, "test");
}

TEST_F(S3FilesystemTest, RenameFile) {
  const std::string src = GetURIForPath("RenameFileSrc");
  const std::string dst = GetURIForPath("RenameFileDst");
  WriteString(src, "test");
  ASSERT_TF_OK(status_);

  tf_s3_filesystem::RenameFile(filesystem_, src.c_str(), dst.c_str(), status_);
  EXPECT_TF_OK(status_);
  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("test", result);
}

TEST_F(S3FilesystemTest, RenameFileOverwrite) {
  const std::string src = GetURIForPath("RenameFileOverwriteSrc");
  const std::string dst = GetURIForPath("RenameFileOverwriteDst");

  WriteString(src, "test_old");
  ASSERT_TF_OK(status_);
  WriteString(dst, "test_new");
  ASSERT_TF_OK(status_);

  tf_s3_filesystem::PathExists(filesystem_, dst.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_s3_filesystem::RenameFile(filesystem_, src.c_str(), dst.c_str(), status_);
  EXPECT_TF_OK(status_);

  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("test_old", result);
}

// Test against large file.
TEST_F(S3FilesystemTest, ReadLargeFile) {
  auto local_path = GetLocalLargeFile();
  auto server_path = GetServerLargeFile();
  if (local_path.empty() || server_path.empty()) GTEST_SKIP();
  std::ifstream in(local_path, std::ios::binary);
  std::string local_content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

  constexpr size_t buffer_size = 50 * 1024 * 1024;
  auto server_content = ReadAllInChunks(server_path, buffer_size, true);
  ASSERT_TF_OK(status_);
  EXPECT_EQ(local_content, server_content);

  server_content = ReadAllInChunks(server_path, buffer_size, false);
  ASSERT_TF_OK(status_);
  EXPECT_EQ(local_content, server_content);
}

TEST_F(S3FilesystemTest, CopyLargeFile) {
  auto server_path = GetServerLargeFile();
  if (server_path.empty()) GTEST_SKIP();

  auto path = GetURIForPath("CopyLargeFile");
  constexpr size_t buffer_size = 5 * 1024 * 1024;
  auto s3_file =
      static_cast<tf_s3_filesystem::S3File*>(filesystem_->plugin_filesystem);
  s3_file->multi_part_chunk_sizes[Aws::Transfer::TransferDirection::UPLOAD] =
      buffer_size;
  tf_s3_filesystem::CopyFile(filesystem_, server_path.c_str(), path.c_str(),
                             status_);
  EXPECT_TF_OK(status_);

  auto server_size =
      tf_s3_filesystem::GetFileSize(filesystem_, server_path.c_str(), status_);
  EXPECT_TF_OK(status_);
  auto actual_size =
      tf_s3_filesystem::GetFileSize(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(server_size, actual_size);
}

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
