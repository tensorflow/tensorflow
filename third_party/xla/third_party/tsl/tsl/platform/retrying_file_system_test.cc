/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/retrying_file_system.h"

#include <fstream>

#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace {

typedef std::vector<std::tuple<string, absl::Status>> ExpectedCalls;

ExpectedCalls CreateRetriableErrors(const string& method, int n) {
  ExpectedCalls expected_calls;
  expected_calls.reserve(n);
  for (int i = 0; i < n; i++) {
    expected_calls.emplace_back(std::make_tuple(
        method, errors::Unavailable(strings::StrCat("Retriable error #", i))));
  }
  return expected_calls;
}

// A class to manage call expectations on mock implementations.
class MockCallSequence {
 public:
  explicit MockCallSequence(const ExpectedCalls& calls) : calls_(calls) {}

  ~MockCallSequence() {
    EXPECT_TRUE(calls_.empty())
        << "Not all expected calls have been made, "
        << "the next expected call: " << std::get<0>(calls_.front());
  }

  absl::Status ConsumeNextCall(const string& method) {
    EXPECT_FALSE(calls_.empty()) << "No more calls were expected.";
    auto call = calls_.front();
    calls_.erase(calls_.begin());
    EXPECT_EQ(std::get<0>(call), method) << "Unexpected method called.";
    return std::get<1>(call);
  }

 private:
  ExpectedCalls calls_;
};

class MockRandomAccessFile : public RandomAccessFile {
 public:
  explicit MockRandomAccessFile(const ExpectedCalls& calls) : calls_(calls) {}
  absl::Status Name(StringPiece* result) const override {
    return calls_.ConsumeNextCall("Name");
  }
  absl::Status Read(uint64 offset, size_t n, StringPiece* result,
                    char* scratch) const override {
    return calls_.ConsumeNextCall("Read");
  }

 private:
  mutable MockCallSequence calls_;
};

class MockWritableFile : public WritableFile {
 public:
  explicit MockWritableFile(const ExpectedCalls& calls) : calls_(calls) {}
  absl::Status Append(StringPiece data) override {
    return calls_.ConsumeNextCall("Append");
  }
  absl::Status Close() override { return calls_.ConsumeNextCall("Close"); }
  absl::Status Flush() override { return calls_.ConsumeNextCall("Flush"); }
  absl::Status Name(StringPiece* result) const override {
    return calls_.ConsumeNextCall("Name");
  }
  absl::Status Sync() override { return calls_.ConsumeNextCall("Sync"); }
  absl::Status Tell(int64_t* position) override {
    return calls_.ConsumeNextCall("Tell");
  }

 private:
  mutable MockCallSequence calls_;
};

class MockFileSystem : public FileSystem {
 public:
  explicit MockFileSystem(const ExpectedCalls& calls, bool* flushed = nullptr)
      : calls_(calls), flushed_(flushed) {}

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
    *result = std::move(random_access_file_to_return);
    return calls_.ConsumeNextCall("NewRandomAccessFile");
  }

  absl::Status NewWritableFile(const string& fname, TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override {
    *result = std::move(writable_file_to_return);
    return calls_.ConsumeNextCall("NewWritableFile");
  }

  absl::Status NewAppendableFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override {
    *result = std::move(writable_file_to_return);
    return calls_.ConsumeNextCall("NewAppendableFile");
  }

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return calls_.ConsumeNextCall("NewReadOnlyMemoryRegionFromFile");
  }

  absl::Status FileExists(const string& fname,
                          TransactionToken* token) override {
    return calls_.ConsumeNextCall("FileExists");
  }

  absl::Status GetChildren(const string& dir, TransactionToken* token,
                           std::vector<string>* result) override {
    return calls_.ConsumeNextCall("GetChildren");
  }

  absl::Status GetMatchingPaths(const string& dir, TransactionToken* token,
                                std::vector<string>* result) override {
    return calls_.ConsumeNextCall("GetMatchingPaths");
  }

  absl::Status Stat(const string& fname, TransactionToken* token,
                    FileStatistics* stat) override {
    return calls_.ConsumeNextCall("Stat");
  }

  absl::Status DeleteFile(const string& fname,
                          TransactionToken* token) override {
    return calls_.ConsumeNextCall("DeleteFile");
  }

  absl::Status CreateDir(const string& dirname,
                         TransactionToken* token) override {
    return calls_.ConsumeNextCall("CreateDir");
  }

  absl::Status DeleteDir(const string& dirname,
                         TransactionToken* token) override {
    return calls_.ConsumeNextCall("DeleteDir");
  }

  absl::Status GetFileSize(const string& fname, TransactionToken* token,
                           uint64* file_size) override {
    return calls_.ConsumeNextCall("GetFileSize");
  }

  absl::Status RenameFile(const string& src, const string& target,
                          TransactionToken* token) override {
    return calls_.ConsumeNextCall("RenameFile");
  }

  absl::Status IsDirectory(const string& dirname,
                           TransactionToken* token) override {
    return calls_.ConsumeNextCall("IsDirectory");
  }

  absl::Status DeleteRecursively(const string& dirname, TransactionToken* token,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) override {
    return calls_.ConsumeNextCall("DeleteRecursively");
  }

  void FlushCaches(TransactionToken* token) override {
    if (flushed_) {
      *flushed_ = true;
    }
  }

  std::unique_ptr<WritableFile> writable_file_to_return;
  std::unique_ptr<RandomAccessFile> random_access_file_to_return;

 private:
  MockCallSequence calls_;
  bool* flushed_ = nullptr;
};

TEST(RetryingFileSystemTest, NewRandomAccessFile_ImmediateSuccess) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Name", absl::OkStatus()),
       std::make_tuple("Read", absl::OkStatus())});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  TF_EXPECT_OK(random_access_file->Name(&result));
  EXPECT_EQ(result, "");

  char scratch[10];
  TF_EXPECT_OK(random_access_file->Read(0, 10, &result, scratch));
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Read", errors::Unavailable("Something is wrong")),
       std::make_tuple("Read", errors::Unavailable("Wrong again")),
       std::make_tuple("Read", absl::OkStatus())});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  TF_EXPECT_OK(random_access_file->Read(0, 10, &result, scratch));
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_AllRetriesFailed) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls = CreateRetriableErrors("Read", 11);
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  const auto& status = random_access_file->Read(0, 10, &result, scratch);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_NoRetriesForSomeErrors) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls({
      std::make_tuple("Read",
                      errors::FailedPrecondition("Failed precondition")),
  });
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  EXPECT_EQ("Failed precondition",
            random_access_file->Read(0, 10, &result, scratch).message());
}

TEST(RetryingFileSystemTest, NewWritableFile_ImmediateSuccess) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Name", absl::OkStatus()),
       std::make_tuple("Sync", absl::OkStatus()),
       std::make_tuple("Close", absl::OkStatus())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  StringPiece result;
  TF_EXPECT_OK(writable_file->Name(&result));
  EXPECT_EQ(result, "");

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", absl::OkStatus()),
       std::make_tuple("Close", absl::OkStatus())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_SuccessWith3rdTry_ViaDestructor) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Close", errors::Unavailable("Something is wrong")),
       std::make_tuple("Close",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("Close", absl::OkStatus())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  writable_file.reset();  // Trigger Close() via destructor.
}

TEST(RetryingFileSystemTest, NewAppendableFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", absl::OkStatus()),
       std::make_tuple("Close", absl::OkStatus())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewAppendableFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped appendable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewAppendableFile("filename.txt", nullptr, &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_AllRetriesFailed) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls = CreateRetriableErrors("Sync", 11);
  expected_file_calls.emplace_back(std::make_tuple("Close", absl::OkStatus()));
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  // Use it and check the results.
  const auto& status = writable_file->Sync();
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest,
     NewReadOnlyMemoryRegionFromFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("NewReadOnlyMemoryRegionFromFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::unique_ptr<ReadOnlyMemoryRegion> result;
  TF_EXPECT_OK(
      fs.NewReadOnlyMemoryRegionFromFile("filename.txt", nullptr, &result));
}

TEST(RetryingFileSystemTest, NewReadOnlyMemoryRegionFromFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls =
      CreateRetriableErrors("NewReadOnlyMemoryRegionFromFile", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::unique_ptr<ReadOnlyMemoryRegion> result;
  const auto& status =
      fs.NewReadOnlyMemoryRegionFromFile("filename.txt", nullptr, &result);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, GetChildren_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetChildren",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetChildren", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetChildren("gs://path", nullptr, &result));
}

TEST(RetryingFileSystemTest, GetChildren_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("GetChildren", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  const auto& status = fs.GetChildren("gs://path", nullptr, &result);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, GetMatchingPaths_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetMatchingPaths",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetMatchingPaths", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://path/dir", nullptr, &result));
}

TEST(RetryingFileSystemTest, GetMatchingPaths_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls =
      CreateRetriableErrors("GetMatchingPaths", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  const auto& status = fs.GetMatchingPaths("gs://path/dir", nullptr, &result);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, DeleteFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.DeleteFile("gs://path/file.txt", nullptr));
}

TEST(RetryingFileSystemTest, DeleteFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("DeleteFile", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.DeleteFile("gs://path/file.txt", nullptr);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, CreateDir_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("CreateDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("CreateDir", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.CreateDir("gs://path/newdir", nullptr));
}

TEST(RetryingFileSystemTest, CreateDir_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("CreateDir", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.CreateDir("gs://path/newdir", nullptr);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, DeleteDir_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteDir", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.DeleteDir("gs://path/dir", nullptr));
}

TEST(RetryingFileSystemTest, DeleteDir_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("DeleteDir", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.DeleteDir("gs://path/dir", nullptr);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, GetFileSize_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetFileSize",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetFileSize", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://path/file.txt", nullptr, &size));
}

TEST(RetryingFileSystemTest, GetFileSize_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("GetFileSize", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  uint64 size;
  const auto& status = fs.GetFileSize("gs://path/file.txt", nullptr, &size);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, RenameFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("RenameFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("RenameFile", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.RenameFile("old_name", "new_name", nullptr));
}

TEST(RetryingFileSystemTest, RenameFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("RenameFile", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.RenameFile("old_name", "new_name", nullptr);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, Stat_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("Stat", errors::Unavailable("Something is wrong")),
       std::make_tuple("Stat", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("file_name", nullptr, &stat));
}

TEST(RetryingFileSystemTest, Stat_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("Stat", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  FileStatistics stat;
  const auto& status = fs.Stat("file_name", nullptr, &stat);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, FileExists_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("FileExists", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.FileExists("file_name", nullptr);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, FileExists_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("FileExists", errors::Unavailable("Something is wrong")),
       std::make_tuple("FileExists", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.FileExists("gs://path/dir", nullptr));
}

TEST(RetryingFileSystemTest, IsDirectory_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("IsDirectory",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("IsDirectory", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.IsDirectory("gs://path/dir", nullptr));
}

TEST(RetryingFileSystemTest, IsDirectory_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("IsDirectory", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.IsDirectory("gs://path/dir", nullptr);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, DeleteRecursively_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteRecursively",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteRecursively", absl::OkStatus())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));
  int64_t undeleted_files, undeleted_dirs;

  TF_EXPECT_OK(fs.DeleteRecursively("gs://path/dir", nullptr, &undeleted_files,
                                    &undeleted_dirs));
}

TEST(RetryingFileSystemTest, DeleteRecursively_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls =
      CreateRetriableErrors("DeleteRecursively", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));
  int64_t undeleted_files, undeleted_dirs;

  const auto& status = fs.DeleteRecursively("gs://path/dir", nullptr,
                                            &undeleted_files, &undeleted_dirs);
  EXPECT_TRUE(absl::StrContains(status.message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, FlushCaches) {
  ExpectedCalls none;
  bool flushed = false;
  std::unique_ptr<MockFileSystem> base_fs(new MockFileSystem(none, &flushed));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));
  fs.FlushCaches(nullptr);
  EXPECT_TRUE(flushed);
}

}  // namespace
}  // namespace tsl
