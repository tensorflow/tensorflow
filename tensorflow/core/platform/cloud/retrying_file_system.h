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

#ifndef TENSORFLOW_CORE_PLATFORM_RETRYING_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_RETRYING_FILE_SYSTEM_H_

#include <string>
#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

/// A wrapper to add retry logic to another file system.
class RetryingFileSystem : public FileSystem {
 public:
  RetryingFileSystem(std::unique_ptr<FileSystem> base_file_system,
                     int64 delay_microseconds = 1000000)
      : base_file_system_(std::move(base_file_system)),
        initial_delay_microseconds_(delay_microseconds) {}

  Status NewRandomAccessFile(
      const string& filename,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status GetMatchingPaths(const string& dir,
                          std::vector<string>* result) override;

  Status Stat(const string& fname, FileStatistics* stat) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& dirname) override;

  Status DeleteDir(const string& dirname) override;

  Status GetFileSize(const string& fname, uint64* file_size) override;

  Status RenameFile(const string& src, const string& target) override;

  Status IsDirectory(const string& dir) override;

  Status DeleteRecursively(const string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs) override;

  void FlushCaches() override;

 private:
  std::unique_ptr<FileSystem> base_file_system_;
  const int64 initial_delay_microseconds_;

  TF_DISALLOW_COPY_AND_ASSIGN(RetryingFileSystem);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RETRYING_FILE_SYSTEM_H_
