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
#define EXPECT_TF_OK(x) EXPECT_EQ(TF_OK, TF_GetCode(x)) << TF_Message(x)

static const char* content = "abcdefghijklmnopqrstuvwxyz1234567890";
// We will work with content_view instead of content.
static const absl::string_view content_view = content;

namespace gcs = google::cloud::storage;

static std::string InitializeTmpDir() {
  // This env should be something like `gs://bucket/path`
  const char* test_dir = getenv("GCS_TEST_TMPDIR");
  if (test_dir != nullptr) {
    std::string bucket, object;
    TF_Status* status = TF_NewStatus();
    ParseGCSPath(test_dir, true, &bucket, &object, status);
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

// TODO(vnvo2409): Refactor `gcs_filesystem_test` to remove unnecessary tests
// after porting all tests from
// `//tensorflow/core/platform/cloud:gcs_file_system_test`.
class GCSFilesystemTest : public ::testing::Test {
 public:
  void SetUp() override {
    root_dir_ = io::JoinPath(
        *GetTmpDir(),
        ::testing::UnitTest::GetInstance()->current_test_info()->name());
    status_ = TF_NewStatus();
    filesystem_ = new TF_Filesystem;
    filesystem_->plugin_filesystem = nullptr;
    // Because different tests requires different setup for filesystem. We
    // initialize filesystem in each testcase.
  }
  void TearDown() override {
    TF_DeleteStatus(status_);
    if (filesystem_->plugin_filesystem != nullptr)
      tf_gcs_filesystem::Cleanup(filesystem_);
    delete filesystem_;
  }

  std::string GetURIForPath(absl::string_view path) {
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
    tf_gcs_filesystem::NewWritableFile(filesystem_, path.c_str(), writer.get(),
                                       status_);
    if (TF_GetCode(status_) != TF_OK) return;
    tf_writable_file::Append(writer.get(), content.c_str(), content.length(),
                             status_);
    if (TF_GetCode(status_) != TF_OK) return;
    tf_writable_file::Close(writer.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return;
  }

  std::string ReadAll(const std::string& path) {
    auto reader = GetReader();
    tf_gcs_filesystem::NewRandomAccessFile(filesystem_, path.c_str(),
                                           reader.get(), status_);
    if (TF_GetCode(status_) != TF_OK) return "";

    auto file_size =
        tf_gcs_filesystem::GetFileSize(filesystem_, path.c_str(), status_);
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

 private:
  std::string root_dir_;
};

::testing::AssertionResult WriteToServer(const std::string& path, size_t offset,
                                         size_t length, gcs::Client* gcs_client,
                                         TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK)
    return ::testing::AssertionFailure() << TF_Message(status);

  auto writer = gcs_client->WriteObject(bucket, object);
  writer.write(content + offset, length);
  writer.Close();
  if (writer.metadata()) {
    return ::testing::AssertionSuccess();
  } else {
    return ::testing::AssertionFailure()
           << writer.metadata().status().message();
  }
}

::testing::AssertionResult InsertObject(const std::string& path,
                                        const std::string& content,
                                        gcs::Client* gcs_client,
                                        TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK)
    return ::testing::AssertionFailure() << TF_Message(status);
  auto metadata = gcs_client->InsertObject(bucket, object, content);
  if (metadata)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << metadata.status().message();
}

::testing::AssertionResult CompareSubString(int64_t offset, size_t length,
                                            absl::string_view result,
                                            size_t read) {
  // Result isn't a null-terminated string so we have to wrap it inside a
  // `string_view`
  if (length == read && content_view.substr(offset, length) ==
                            absl::string_view(result).substr(0, read))
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure()
           << "Result: " << absl::string_view(result).substr(0, read)
           << " Read: " << read;
}

::testing::AssertionResult CompareWithServer(const std::string& path,
                                             size_t offset, size_t length,
                                             gcs::Client* gcs_client,
                                             TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK)
    return ::testing::AssertionFailure() << TF_Message(status);

  auto reader = gcs_client->ReadObject(bucket, object);
  if (!reader) {
    return ::testing::AssertionFailure() << reader.status().message();
  } else {
    std::string content{std::istreambuf_iterator<char>{reader}, {}};
    return CompareSubString(offset, length, content, content.length());
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
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
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

  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(WriteToServer(filepath, 0, content_view.length(),
                            &gcs_file->gcs_client, status_));

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

  delete[] result;
  tf_random_access_file::Cleanup(file);
  delete file;
}

TEST_F(GCSFilesystemTest, WritableFile) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
  std::string filepath = GetURIForPath("a_file");
  TF_WritableFile* file = new TF_WritableFile;
  tf_gcs_filesystem::NewWritableFile(filesystem_, filepath.c_str(), file,
                                     status_);
  ASSERT_TF_OK(status_);
  tf_writable_file::Append(file, content, 4, status_);
  ASSERT_TF_OK(status_);
  auto length = tf_writable_file::Tell(file, status_);
  ASSERT_EQ(length, 4);
  ASSERT_TF_OK(status_);
  tf_writable_file::Flush(file, status_);
  ASSERT_TF_OK(status_);

  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(
      CompareWithServer(filepath, 0, 4, &gcs_file->gcs_client, status_));

  tf_writable_file::Append(file, content + 4, 4, status_);
  ASSERT_TF_OK(status_);
  length = tf_writable_file::Tell(file, status_);
  ASSERT_EQ(length, 8);
  ASSERT_TF_OK(status_);
  tf_writable_file::Flush(file, status_);
  ASSERT_TF_OK(status_);
  ASSERT_TRUE(
      CompareWithServer(filepath, 0, 8, &gcs_file->gcs_client, status_));

  tf_writable_file::Close(file, status_);
  ASSERT_TF_OK(status_);
  tf_writable_file::Cleanup(file);

  // Testing for compose objects
  gcs_file->compose = true;
  filepath = GetURIForPath("b_file");
  tf_gcs_filesystem::NewWritableFile(filesystem_, filepath.c_str(), file,
                                     status_);
  ASSERT_TF_OK(status_);
  tf_writable_file::Append(file, content, 4, status_);
  ASSERT_TF_OK(status_);
  length = tf_writable_file::Tell(file, status_);
  ASSERT_EQ(length, 4);
  ASSERT_TF_OK(status_);
  tf_writable_file::Flush(file, status_);
  ASSERT_TF_OK(status_);
  ASSERT_TRUE(
      CompareWithServer(filepath, 0, 4, &gcs_file->gcs_client, status_));

  tf_writable_file::Append(file, content + 4, 4, status_);
  ASSERT_TF_OK(status_);
  length = tf_writable_file::Tell(file, status_);
  ASSERT_EQ(length, 8);
  ASSERT_TF_OK(status_);
  tf_writable_file::Flush(file, status_);
  ASSERT_TF_OK(status_);
  ASSERT_TRUE(
      CompareWithServer(filepath, 0, 8, &gcs_file->gcs_client, status_));

  tf_writable_file::Close(file, status_);
  ASSERT_TF_OK(status_);
  tf_writable_file::Cleanup(file);
  delete file;
}

TEST_F(GCSFilesystemTest, ReadOnlyMemoryRegion) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
  std::string path = GetURIForPath("a_file");
  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(WriteToServer(path, 0, 0, &gcs_file->gcs_client, status_));
  TF_ReadOnlyMemoryRegion* region = new TF_ReadOnlyMemoryRegion;
  tf_gcs_filesystem::NewReadOnlyMemoryRegionFromFile(filesystem_, path.c_str(),
                                                     region, status_);
  ASSERT_EQ(TF_GetCode(status_), TF_INVALID_ARGUMENT) << TF_Message(status_);

  TF_SetStatus(status_, TF_OK, "");
  ASSERT_TRUE(WriteToServer(path, 0, content_view.length(),
                            &gcs_file->gcs_client, status_));
  tf_gcs_filesystem::NewReadOnlyMemoryRegionFromFile(filesystem_, path.c_str(),
                                                     region, status_);
  ASSERT_TF_OK(status_);
  auto length = tf_read_only_memory_region::Length(region);
  ASSERT_EQ(length, content_view.length());
  auto data =
      static_cast<const char*>(tf_read_only_memory_region::Data(region));
  ASSERT_TRUE(CompareSubString(0, content_view.length(), data, length));

  tf_read_only_memory_region::Cleanup(region);
  delete region;
}

TEST_F(GCSFilesystemTest, PathExists) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string path = GetURIForPath("PathExists");
  tf_gcs_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_EQ(TF_NOT_FOUND, TF_GetCode(status_)) << TF_Message(status_);
  TF_SetStatus(status_, TF_OK, "");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  tf_gcs_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
}

TEST_F(GCSFilesystemTest, GetChildren) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string base = GetURIForPath("GetChildren");
  tf_gcs_filesystem::CreateDir(filesystem_, base.c_str(), status_);
  EXPECT_TF_OK(status_);

  const std::string file = io::JoinPath(base, "TestFile.csv");
  WriteString(file, "test");
  EXPECT_TF_OK(status_);

  const std::string subdir = io::JoinPath(base, "SubDir");
  tf_gcs_filesystem::CreateDir(filesystem_, subdir.c_str(), status_);
  EXPECT_TF_OK(status_);
  const std::string subfile = io::JoinPath(subdir, "TestSubFile.csv");
  WriteString(subfile, "test");
  EXPECT_TF_OK(status_);

  char** entries;
  auto num_entries = tf_gcs_filesystem::GetChildren(filesystem_, base.c_str(),
                                                    &entries, status_);
  EXPECT_TF_OK(status_);

  std::vector<std::string> childrens;
  for (int i = 0; i < num_entries; ++i) {
    childrens.push_back(entries[i]);
  }
  std::sort(childrens.begin(), childrens.end());
  EXPECT_EQ(std::vector<string>({"SubDir/", "TestFile.csv"}), childrens);
}

TEST_F(GCSFilesystemTest, DeleteFile) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string path = GetURIForPath("DeleteFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);
  tf_gcs_filesystem::DeleteFile(filesystem_, path.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_gcs_filesystem::PathExists(filesystem_, path.c_str(), status_);
  EXPECT_EQ(TF_GetCode(status_), TF_NOT_FOUND);
}

TEST_F(GCSFilesystemTest, CreateDir) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string dir = GetURIForPath("CreateDir");
  tf_gcs_filesystem::CreateDir(filesystem_, dir.c_str(), status_);
  EXPECT_TF_OK(status_);

  TF_FileStatistics stat;
  tf_gcs_filesystem::Stat(filesystem_, dir.c_str(), &stat, status_);
  EXPECT_TF_OK(status_);
  EXPECT_TRUE(stat.is_directory);
}

TEST_F(GCSFilesystemTest, DeleteDir) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string dir = GetURIForPath("DeleteDir");
  const std::string file = io::JoinPath(dir, "DeleteDirFile.csv");
  WriteString(file, "test");
  ASSERT_TF_OK(status_);
  tf_gcs_filesystem::DeleteDir(filesystem_, dir.c_str(), status_);
  EXPECT_EQ(TF_GetCode(status_), TF_FAILED_PRECONDITION);

  TF_SetStatus(status_, TF_OK, "");
  tf_gcs_filesystem::DeleteFile(filesystem_, file.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_gcs_filesystem::DeleteDir(filesystem_, dir.c_str(), status_);
  EXPECT_TF_OK(status_);
  TF_FileStatistics stat;
  tf_gcs_filesystem::Stat(filesystem_, dir.c_str(), &stat, status_);
  EXPECT_EQ(TF_GetCode(status_), TF_NOT_FOUND) << TF_Message(status_);
}

TEST_F(GCSFilesystemTest, StatFile) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string path = GetURIForPath("StatFile");
  WriteString(path, "test");
  ASSERT_TF_OK(status_);

  TF_FileStatistics stat;
  tf_gcs_filesystem::Stat(filesystem_, path.c_str(), &stat, status_);
  EXPECT_TF_OK(status_);
  EXPECT_EQ(4, stat.length);
  EXPECT_FALSE(stat.is_directory);
}

TEST_F(GCSFilesystemTest, RenameFile) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string src = GetURIForPath("RenameFileSrc");
  const std::string dst = GetURIForPath("RenameFileDst");
  WriteString(src, "test");
  ASSERT_TF_OK(status_);

  tf_gcs_filesystem::RenameFile(filesystem_, src.c_str(), dst.c_str(), status_);
  EXPECT_TF_OK(status_);
  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("test", result);
}

TEST_F(GCSFilesystemTest, RenameFileOverwrite) {
  tf_gcs_filesystem::Init(filesystem_, status_);
  ASSERT_TF_OK(status_);
  const std::string src = GetURIForPath("RenameFileOverwriteSrc");
  const std::string dst = GetURIForPath("RenameFileOverwriteDst");

  WriteString(src, "test_old");
  ASSERT_TF_OK(status_);
  WriteString(dst, "test_new");
  ASSERT_TF_OK(status_);

  tf_gcs_filesystem::PathExists(filesystem_, dst.c_str(), status_);
  EXPECT_TF_OK(status_);
  tf_gcs_filesystem::RenameFile(filesystem_, src.c_str(), dst.c_str(), status_);
  EXPECT_TF_OK(status_);

  auto result = ReadAll(dst);
  EXPECT_TF_OK(status_);
  EXPECT_EQ("test_old", result);
}

// These tests below are ported from
// `//tensorflow/core/platform/cloud:gcs_file_system_test`
TEST_F(GCSFilesystemTest, NewRandomAccessFile_NoBlockCache) {
  tf_gcs_filesystem::InitTest(filesystem_, false, 0, 0, 0, 0, 0, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
  std::string path = GetURIForPath("a_file");
  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(InsertObject(path, "0123456789", &gcs_file->gcs_client, status_));

  TF_RandomAccessFile* file = new TF_RandomAccessFile;
  tf_gcs_filesystem::NewRandomAccessFile(filesystem_, path.c_str(), file,
                                         status_);
  ASSERT_TF_OK(status_);

  std::string result;
  result.resize(6);
  int64_t read = tf_random_access_file::Read(file, 0, 6, &result[0], status_);
  ASSERT_EQ(read, 6) << "Read: " << read << "\n";
  ASSERT_TF_OK(status_);
  ASSERT_EQ(result, "012345") << "Result: " << result << "\n";

  read = tf_random_access_file::Read(file, 6, 6, &result[0], status_);
  ASSERT_EQ(read, 4) << "Read: " << read << "\n";
  ASSERT_EQ(TF_GetCode(status_), TF_OUT_OF_RANGE) << TF_Message(status_);
  result.resize(read);
  ASSERT_EQ(result, "6789") << "Result: " << result << "\n";
}

TEST_F(GCSFilesystemTest, NewRandomAccessFile_Buffered) {
  tf_gcs_filesystem::InitTest(filesystem_, false, 10, 0, 0, 0, 0, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
  std::string path = GetURIForPath("a_file");
  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(InsertObject(path, "0123456789", &gcs_file->gcs_client, status_));

  TF_RandomAccessFile* file = new TF_RandomAccessFile;
  tf_gcs_filesystem::NewRandomAccessFile(filesystem_, path.c_str(), file,
                                         status_);
  ASSERT_TF_OK(status_);

  std::string result;
  result.resize(6);
  int64_t read = tf_random_access_file::Read(file, 0, 6, &result[0], status_);
  ASSERT_EQ(read, 6) << "Read: " << read << "\n";
  ASSERT_TF_OK(status_);
  ASSERT_EQ(result, "012345") << "Result: " << result << "\n";

  read = tf_random_access_file::Read(file, 6, 6, &result[0], status_);
  ASSERT_EQ(read, 4) << "Read: " << read << "\n";
  ASSERT_EQ(TF_GetCode(status_), TF_OUT_OF_RANGE) << TF_Message(status_);
  result.resize(read);
  ASSERT_EQ(result, "6789") << "Result: " << result << "\n";
}

TEST_F(GCSFilesystemTest, NewRandomAccessFile_Buffered_ReadAtEOF) {
  tf_gcs_filesystem::InitTest(filesystem_, false, 10, 0, 0, 0, 0, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
  std::string path = GetURIForPath("a_file");
  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(InsertObject(path, "0123456789", &gcs_file->gcs_client, status_));

  TF_RandomAccessFile* file = new TF_RandomAccessFile;
  tf_gcs_filesystem::NewRandomAccessFile(filesystem_, path.c_str(), file,
                                         status_);
  ASSERT_TF_OK(status_);

  std::string result;
  result.resize(10);
  int64_t read = tf_random_access_file::Read(file, 0, result.length(),
                                             &result[0], status_);
  ASSERT_EQ(read, 10) << "Read: " << read << "\n";
  ASSERT_TF_OK(status_);
  ASSERT_EQ(result, "0123456789") << "Result: " << result << "\n";

  read = tf_random_access_file::Read(file, result.length(), result.length(),
                                     &result[0], status_);
  ASSERT_EQ(read, 0) << "Read: " << read << "\n";
  ASSERT_EQ(TF_GetCode(status_), TF_OUT_OF_RANGE) << TF_Message(status_);
  result.resize(read);
  ASSERT_EQ(result, "") << "Result: " << result << "\n";
}

TEST_F(GCSFilesystemTest, NewRandomAccessFile_Buffered_CachedOutOfRange) {
  tf_gcs_filesystem::InitTest(filesystem_, false, 10, 0, 0, 0, 0, status_);
  ASSERT_TF_OK(status_) << "Could not initialize filesystem. "
                        << TF_Message(status_);
  std::string path = GetURIForPath("a_file");
  auto gcs_file =
      static_cast<tf_gcs_filesystem::GCSFile*>(filesystem_->plugin_filesystem);
  ASSERT_TRUE(InsertObject(path, "012345678", &gcs_file->gcs_client, status_));

  TF_RandomAccessFile* file = new TF_RandomAccessFile;
  tf_gcs_filesystem::NewRandomAccessFile(filesystem_, path.c_str(), file,
                                         status_);
  ASSERT_TF_OK(status_);

  std::string result;
  result.resize(5);
  int64_t read = tf_random_access_file::Read(file, 0, result.length(),
                                             &result[0], status_);
  ASSERT_EQ(read, 5) << "Read: " << read << "\n";
  ASSERT_TF_OK(status_);
  ASSERT_EQ(result, "01234") << "Result: " << result << "\n";

  read = tf_random_access_file::Read(file, 4, result.length(), &result[0],
                                     status_);
  ASSERT_EQ(read, 5) << "Read: " << read << "\n";
  ASSERT_TF_OK(status_);
  result.resize(read);
  ASSERT_EQ(result, "45678") << "Result: " << result << "\n";

  read = tf_random_access_file::Read(file, 5, result.length(), &result[0],
                                     status_);
  ASSERT_EQ(read, 4) << "Read: " << read << "\n";
  ASSERT_EQ(TF_GetCode(status_), TF_OUT_OF_RANGE) << TF_Message(status_);
  result.resize(read);
  ASSERT_EQ(result, "5678") << "Result: " << result << "\n";
}

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  if (!GetTmpDir()) {
    std::cerr << "Could not read GCS_TEST_TMPDIR env";
    return -1;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
