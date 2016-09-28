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
#include <functional>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

namespace {

// In case of failure, every call will be retried kMaxAttempts-1 times.
constexpr int kMaxAttempts = 4;

bool IsRetriable(Status status) {
  switch (status.code()) {
    case error::UNAVAILABLE:
    case error::DEADLINE_EXCEEDED:
    case error::UNKNOWN:
      return true;
    default:
      // OK also falls here.
      return false;
  }
}

Status CallWithRetries(const std::function<Status()>& f) {
  int attempts = 0;
  while (true) {
    attempts++;
    auto status = f();
    if (!IsRetriable(status) || attempts >= kMaxAttempts) {
      return status;
    }
    LOG(ERROR) << "The operation resulted in an error and will be retried: "
               << status.ToString();
  }
}

class RetryingRandomAccessFile : public RandomAccessFile {
 public:
  RetryingRandomAccessFile(std::unique_ptr<RandomAccessFile> base_file)
      : base_file_(std::move(base_file)) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    return CallWithRetries(std::bind(&RandomAccessFile::Read, base_file_.get(),
                                     offset, n, result, scratch));
  }

 private:
  std::unique_ptr<RandomAccessFile> base_file_;
};

class RetryingWritableFile : public WritableFile {
 public:
  RetryingWritableFile(std::unique_ptr<WritableFile> base_file)
      : base_file_(std::move(base_file)) {}

  Status Append(const StringPiece& data) override {
    return CallWithRetries(
        std::bind(&WritableFile::Append, base_file_.get(), data));
  }
  Status Close() override {
    return CallWithRetries(std::bind(&WritableFile::Close, base_file_.get()));
  }
  Status Flush() override {
    return CallWithRetries(std::bind(&WritableFile::Flush, base_file_.get()));
  }
  Status Sync() override {
    return CallWithRetries(std::bind(&WritableFile::Sync, base_file_.get()));
  }

 private:
  std::unique_ptr<WritableFile> base_file_;
};

}  // namespace

Status RetryingFileSystem::NewRandomAccessFile(
    const string& filename, std::unique_ptr<RandomAccessFile>* result) {
  std::unique_ptr<RandomAccessFile> base_file;
  TF_RETURN_IF_ERROR(CallWithRetries(std::bind(&FileSystem::NewRandomAccessFile,
                                               base_file_system_.get(),
                                               filename, &base_file)));
  result->reset(new RetryingRandomAccessFile(std::move(base_file)));
  return Status::OK();
}

Status RetryingFileSystem::NewWritableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(CallWithRetries(std::bind(&FileSystem::NewWritableFile,
                                               base_file_system_.get(),
                                               filename, &base_file)));
  result->reset(new RetryingWritableFile(std::move(base_file)));
  return Status::OK();
}

Status RetryingFileSystem::NewAppendableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(CallWithRetries(std::bind(&FileSystem::NewAppendableFile,
                                               base_file_system_.get(),
                                               filename, &base_file)));
  result->reset(new RetryingWritableFile(std::move(base_file)));
  return Status::OK();
}

Status RetryingFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& filename, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  return CallWithRetries(std::bind(&FileSystem::NewReadOnlyMemoryRegionFromFile,
                                   base_file_system_.get(), filename, result));
}

bool RetryingFileSystem::FileExists(const string& fname) {
  // No status -- no retries.
  return base_file_system_->FileExists(fname);
}

Status RetryingFileSystem::Stat(const string& fname, FileStatistics* stat) {
  return CallWithRetries(
      std::bind(&FileSystem::Stat, base_file_system_.get(), fname, stat));
}

Status RetryingFileSystem::GetChildren(const string& dir,
                                       std::vector<string>* result) {
  return CallWithRetries(std::bind(&FileSystem::GetChildren,
                                   base_file_system_.get(), dir, result));
}

Status RetryingFileSystem::GetMatchingPaths(const string& pattern,
                                            std::vector<string>* result) {
  return CallWithRetries(std::bind(&FileSystem::GetMatchingPaths,
                                   base_file_system_.get(), pattern, result));
}

Status RetryingFileSystem::DeleteFile(const string& fname) {
  return CallWithRetries(
      std::bind(&FileSystem::DeleteFile, base_file_system_.get(), fname));
}

Status RetryingFileSystem::CreateDir(const string& dirname) {
  return CallWithRetries(
      std::bind(&FileSystem::CreateDir, base_file_system_.get(), dirname));
}

Status RetryingFileSystem::DeleteDir(const string& dirname) {
  return CallWithRetries(
      std::bind(&FileSystem::DeleteDir, base_file_system_.get(), dirname));
}

Status RetryingFileSystem::GetFileSize(const string& fname, uint64* file_size) {
  return CallWithRetries(std::bind(&FileSystem::GetFileSize,
                                   base_file_system_.get(), fname, file_size));
}

Status RetryingFileSystem::RenameFile(const string& src, const string& target) {
  return CallWithRetries(
      std::bind(&FileSystem::RenameFile, base_file_system_.get(), src, target));
}

Status RetryingFileSystem::IsDirectory(const string& dirname) {
  return CallWithRetries(
      std::bind(&FileSystem::IsDirectory, base_file_system_.get(), dirname));
}

}  // namespace tensorflow
