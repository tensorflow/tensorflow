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
#include <chrono>
#include <functional>
#include <random>
#include <thread>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

namespace {

// In case of failure, every call will be retried kMaxRetries times.
constexpr int kMaxRetries = 3;
// Maximum backoff time in seconds.
constexpr int kMaximumBackoffMilliSeconds = 32000;

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

void WaitBeforeRetry(const int delay_seconds) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, 999);

  const int delay_milliseconds = delay_seconds * 1000;
  const int random_milliseconds = dist(gen);
  const int delay = std::min(delay_milliseconds + random_milliseconds,
                             kMaximumBackoffMilliSeconds);

  std::this_thread::sleep_for(std::chrono::milliseconds(delay));
}

Status CallWithRetries(const std::function<Status()>& f,
                       const int initial_delay_seconds) {
  int retries = 0;
  while (true) {
    auto status = f();
    if (!IsRetriable(status) || retries >= kMaxRetries) {
      return status;
    }
    LOG(ERROR) << "The operation resulted in an error and will be retried: "
               << status.ToString();
    WaitBeforeRetry(initial_delay_seconds << retries);
    retries++;
  }
}

class RetryingRandomAccessFile : public RandomAccessFile {
 public:
  RetryingRandomAccessFile(std::unique_ptr<RandomAccessFile> base_file,
                           int delay_seconds = 1)
      : base_file_(std::move(base_file)),
        initial_delay_seconds_(delay_seconds) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    return CallWithRetries(std::bind(&RandomAccessFile::Read, base_file_.get(),
                                     offset, n, result, scratch),
                           initial_delay_seconds_);
  }

 private:
  std::unique_ptr<RandomAccessFile> base_file_;
  const int initial_delay_seconds_;
};

class RetryingWritableFile : public WritableFile {
 public:
  RetryingWritableFile(std::unique_ptr<WritableFile> base_file,
                       int delay_seconds = 1)
      : base_file_(std::move(base_file)),
        initial_delay_seconds_(delay_seconds) {}

  Status Append(const StringPiece& data) override {
    return CallWithRetries(
        std::bind(&WritableFile::Append, base_file_.get(), data),
        initial_delay_seconds_);
  }
  Status Close() override {
    return CallWithRetries(std::bind(&WritableFile::Close, base_file_.get()),
                           initial_delay_seconds_);
  }
  Status Flush() override {
    return CallWithRetries(std::bind(&WritableFile::Flush, base_file_.get()),
                           initial_delay_seconds_);
  }
  Status Sync() override {
    return CallWithRetries(std::bind(&WritableFile::Sync, base_file_.get()),
                           initial_delay_seconds_);
  }

 private:
  std::unique_ptr<WritableFile> base_file_;
  const int initial_delay_seconds_;
};

}  // namespace

Status RetryingFileSystem::NewRandomAccessFile(
    const string& filename, std::unique_ptr<RandomAccessFile>* result) {
  std::unique_ptr<RandomAccessFile> base_file;
  TF_RETURN_IF_ERROR(CallWithRetries(std::bind(&FileSystem::NewRandomAccessFile,
                                               base_file_system_.get(),
                                               filename, &base_file),
                                     initial_delay_seconds_));
  result->reset(new RetryingRandomAccessFile(std::move(base_file)));
  return Status::OK();
}

Status RetryingFileSystem::NewWritableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(CallWithRetries(std::bind(&FileSystem::NewWritableFile,
                                               base_file_system_.get(),
                                               filename, &base_file),
                                     initial_delay_seconds_));
  result->reset(new RetryingWritableFile(std::move(base_file)));
  return Status::OK();
}

Status RetryingFileSystem::NewAppendableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(CallWithRetries(std::bind(&FileSystem::NewAppendableFile,
                                               base_file_system_.get(),
                                               filename, &base_file),
                                     initial_delay_seconds_));
  result->reset(new RetryingWritableFile(std::move(base_file)));
  return Status::OK();
}

Status RetryingFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& filename, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  return CallWithRetries(std::bind(&FileSystem::NewReadOnlyMemoryRegionFromFile,
                                   base_file_system_.get(), filename, result),
                         initial_delay_seconds_);
}

bool RetryingFileSystem::FileExists(const string& fname) {
  // No status -- no retries.
  return base_file_system_->FileExists(fname);
}

Status RetryingFileSystem::Stat(const string& fname, FileStatistics* stat) {
  return CallWithRetries(
      std::bind(&FileSystem::Stat, base_file_system_.get(), fname, stat),
      initial_delay_seconds_);
}

Status RetryingFileSystem::GetChildren(const string& dir,
                                       std::vector<string>* result) {
  return CallWithRetries(std::bind(&FileSystem::GetChildren,
                                   base_file_system_.get(), dir, result),
                         initial_delay_seconds_);
}

Status RetryingFileSystem::GetChildrenRecursively(const string& dir,
                                                  std::vector<string>* result) {
  return CallWithRetries(std::bind(&FileSystem::GetChildrenRecursively,
                                   base_file_system_.get(), dir, result),
                         initial_delay_seconds_);
}

Status RetryingFileSystem::DeleteFile(const string& fname) {
  return CallWithRetries(
      std::bind(&FileSystem::DeleteFile, base_file_system_.get(), fname),
      initial_delay_seconds_);
}

Status RetryingFileSystem::CreateDir(const string& dirname) {
  return CallWithRetries(
      std::bind(&FileSystem::CreateDir, base_file_system_.get(), dirname),
      initial_delay_seconds_);
}

Status RetryingFileSystem::DeleteDir(const string& dirname) {
  return CallWithRetries(
      std::bind(&FileSystem::DeleteDir, base_file_system_.get(), dirname),
      initial_delay_seconds_);
}

Status RetryingFileSystem::GetFileSize(const string& fname, uint64* file_size) {
  return CallWithRetries(std::bind(&FileSystem::GetFileSize,
                                   base_file_system_.get(), fname, file_size),
                         initial_delay_seconds_);
}

Status RetryingFileSystem::RenameFile(const string& src, const string& target) {
  return CallWithRetries(
      std::bind(&FileSystem::RenameFile, base_file_system_.get(), src, target),
      initial_delay_seconds_);
}

Status RetryingFileSystem::IsDirectory(const string& dirname) {
  return CallWithRetries(
      std::bind(&FileSystem::IsDirectory, base_file_system_.get(), dirname),
      initial_delay_seconds_);
}

}  // namespace tensorflow
