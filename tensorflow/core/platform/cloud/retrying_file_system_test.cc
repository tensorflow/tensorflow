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

#include "tensorflow/core/platform/cloud/retrying_file_system.h"
#include <fstream>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

typedef std::vector<std::tuple<string, Status>> ExpectedCalls;

// A class to manage call expectations on mock implementations.
class MockCallSequence {
 public:
  MockCallSequence(const ExpectedCalls& calls) : calls_(calls) {}

  ~MockCallSequence() {
    EXPECT_TRUE(calls_.empty())
        << "Not all expected calls have been made, "
        << "the next expected call: " << std::get<0>(calls_.front());
  }

  Status ConsumeNextCall(const string& method) {
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
  MockRandomAccessFile(const ExpectedCalls& calls) : calls_(calls) {}
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    return calls_.ConsumeNextCall("Read");
  }

 private:
  mutable MockCallSequence calls_;
};

class MockWritableFile : public WritableFile {
 public:
  MockWritableFile(const ExpectedCalls& calls) : calls_(calls) {}
  Status Append(const StringPiece& data) override {
    return calls_.ConsumeNextCall("Append");
  }
  Status Close() override { return calls_.ConsumeNextCall("Close"); }
  Status Flush() override { return calls_.ConsumeNextCall("Flush"); }
  Status Sync() override { return calls_.ConsumeNextCall("Sync"); }

 private:
  mutable MockCallSequence calls_;
};

class MockFileSystem : public FileSystem {
 public:
  MockFileSystem(const ExpectedCalls& calls) : calls_(calls) {}

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    *result = std::move(random_access_file_to_return);
    return calls_.ConsumeNextCall("NewRandomAccessFile");
  }

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    *result = std::move(writable_file_to_return);
    return calls_.ConsumeNextCall("NewWritableFile");
  }

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    *result = std::move(writable_file_to_return);
    return calls_.ConsumeNextCall("NewAppendableFile");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return calls_.ConsumeNextCall("NewReadOnlyMemoryRegionFromFile");
  }

  bool FileExists(const string& fname) override { return true; }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    return calls_.ConsumeNextCall("GetChildren");
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    return calls_.ConsumeNextCall("Stat");
  }

  Status DeleteFile(const string& fname) override {
    return calls_.ConsumeNextCall("DeleteFile");
  }

  Status CreateDir(const string& dirname) override {
    return calls_.ConsumeNextCall("CreateDir");
  }

  Status DeleteDir(const string& dirname) override {
    return calls_.ConsumeNextCall("DeleteDir");
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    return calls_.ConsumeNextCall("GetFileSize");
  }

  Status RenameFile(const string& src, const string& target) override {
    return calls_.ConsumeNextCall("RenameFile");
  }

  std::unique_ptr<WritableFile> writable_file_to_return;
  std::unique_ptr<RandomAccessFile> random_access_file_to_return;

 private:
  MockCallSequence calls_;
};

TEST(RetryingFileSystemTest, NewRandomAccessFile_ImmediateSuccess) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls({std::make_tuple("Read", Status::OK())});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("filename.txt", &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  TF_EXPECT_OK(random_access_file->Read(0, 10, &result, scratch));
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Read", errors::Unavailable("Something is wrong")),
       std::make_tuple("Read", errors::Unavailable("Wrong again")),
       std::make_tuple("Read", Status::OK())});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("filename.txt", &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  TF_EXPECT_OK(random_access_file->Read(0, 10, &result, scratch));
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_AllRetriesFailed) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Read", errors::Unavailable("Something is wrong")),
       std::make_tuple("Read", errors::Unavailable("Wrong again")),
       std::make_tuple("Read", errors::Unavailable("And again")),
       std::make_tuple("Read", errors::Unavailable("Last error"))});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("filename.txt", &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  EXPECT_EQ("Last error",
            random_access_file->Read(0, 10, &result, scratch).error_message());
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
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("filename.txt", &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  EXPECT_EQ("Failed precondition",
            random_access_file->Read(0, 10, &result, scratch).error_message());
}

TEST(RetryingFileSystemTest, NewWritableFile_ImmediateSuccess) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls({std::make_tuple("Sync", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewAppendableFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewAppendableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped appendable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewAppendableFile("filename.txt", &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_AllRetriesFailed) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", errors::Unavailable("...and again")),
       std::make_tuple("Sync", errors::Unavailable("And again"))});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem fs(std::move(base_fs));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", &writable_file));

  // Use it and check the results.
  EXPECT_EQ("And again", writable_file->Sync().error_message());
}

TEST(RetryingFileSystemTest,
     NewReadOnlyMemoryRegionFromFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("NewReadOnlyMemoryRegionFromFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::unique_ptr<ReadOnlyMemoryRegion> result;
  TF_EXPECT_OK(fs.NewReadOnlyMemoryRegionFromFile("filename.txt", &result));
}

TEST(RetryingFileSystemTest, NewReadOnlyMemoryRegionFromFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("and again")),
       std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("Last error"))});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::unique_ptr<ReadOnlyMemoryRegion> result;
  EXPECT_EQ("Last error",
            fs.NewReadOnlyMemoryRegionFromFile("filename.txt", &result)
                .error_message());
}

TEST(RetryingFileSystemTest, GetChildren_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetChildren",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetChildren", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetChildren("gs://path", &result));
}

TEST(RetryingFileSystemTest, GetChildren_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetChildren",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetChildren",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("GetChildren", errors::Unavailable("And again")),
       std::make_tuple("GetChildren", errors::Unavailable("Last error"))});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  EXPECT_EQ("Last error", fs.GetChildren("gs://path", &result).error_message());
}

TEST(RetryingFileSystemTest, DeleteFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  TF_EXPECT_OK(fs.DeleteFile("gs://path/file.txt"));
}

TEST(RetryingFileSystemTest, DeleteFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteFile",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("DeleteFile", errors::Unavailable("And again")),
       std::make_tuple("DeleteFile", errors::Unavailable("Last error"))});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  EXPECT_EQ("Last error", fs.DeleteFile("gs://path/file.txt").error_message());
}

TEST(RetryingFileSystemTest, CreateDir_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("CreateDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("CreateDir", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  TF_EXPECT_OK(fs.CreateDir("gs://path/newdir"));
}

TEST(RetryingFileSystemTest, CreateDir_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("CreateDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("CreateDir",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("CreateDir", errors::Unavailable("And again")),
       std::make_tuple("CreateDir", errors::Unavailable("Last error"))});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  EXPECT_EQ("Last error", fs.CreateDir("gs://path/newdir").error_message());
}

TEST(RetryingFileSystemTest, DeleteDir_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteDir", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  TF_EXPECT_OK(fs.DeleteDir("gs://path/dir"));
}

TEST(RetryingFileSystemTest, DeleteDir_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteDir",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("DeleteDir", errors::Unavailable("And again")),
       std::make_tuple("DeleteDir", errors::Unavailable("Last error"))});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  std::vector<string> result;
  EXPECT_EQ("Last error", fs.DeleteDir("gs://path/dir").error_message());
}

TEST(RetryingFileSystemTest, GetFileSize_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetFileSize",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetFileSize", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://path/file.txt", &size));
}

TEST(RetryingFileSystemTest, GetFileSize_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetFileSize",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetFileSize",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("GetFileSize", errors::Unavailable("And again")),
       std::make_tuple("GetFileSize", errors::Unavailable("Last error"))});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  uint64 size;
  EXPECT_EQ("Last error",
            fs.GetFileSize("gs://path/file.txt", &size).error_message());
}

TEST(RetryingFileSystemTest, RenameFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("RenameFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("RenameFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  TF_EXPECT_OK(fs.RenameFile("old_name", "new_name"));
}

TEST(RetryingFileSystemTest, RenameFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls({
      std::make_tuple("RenameFile", errors::Unavailable("Something is wrong")),
      std::make_tuple("RenameFile",
                      errors::Unavailable("Something is wrong again")),
      std::make_tuple("RenameFile", errors::Unavailable("And again")),
      std::make_tuple("RenameFile", errors::Unavailable("Last error")),
  });
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  EXPECT_EQ("Last error",
            fs.RenameFile("old_name", "new_name").error_message());
}

TEST(RetryingFileSystemTest, Stat_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("Stat", errors::Unavailable("Something is wrong")),
       std::make_tuple("Stat", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("file_name", &stat));
}

TEST(RetryingFileSystemTest, Stat_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls({
      std::make_tuple("Stat", errors::Unavailable("Something is wrong")),
      std::make_tuple("Stat", errors::Unavailable("Something is wrong again")),
      std::make_tuple("Stat", errors::Unavailable("And again")),
      std::make_tuple("Stat", errors::Unavailable("Last error")),
  });
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem fs(std::move(base_fs));

  FileStatistics stat;
  EXPECT_EQ("Last error", fs.Stat("file_name", &stat).error_message());
}

}  // namespace
}  // namespace tensorflow
