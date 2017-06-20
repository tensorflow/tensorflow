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

#ifndef TENSORFLOW_CORE_PLATFORM_GCS_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_GCS_FILE_SYSTEM_H_

#include <string>
#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cloud/auth_provider.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/cloud/retrying_file_system.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

/// Google Cloud Storage implementation of a file system.
///
/// The clients should use RetryingGcsFileSystem defined below,
/// which adds retry logic to GCS operations.
class GcsFileSystem : public FileSystem {
 public:
  GcsFileSystem();
  GcsFileSystem(std::unique_ptr<AuthProvider> auth_provider,
                std::unique_ptr<HttpRequest::Factory> http_request_factory,
                size_t read_ahead_bytes, int64 initial_retry_delay_usec);

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

  Status Stat(const string& fname, FileStatistics* stat) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& dirname) override;

  Status DeleteDir(const string& dirname) override;

  Status GetFileSize(const string& fname, uint64* file_size) override;

  Status RenameFile(const string& src, const string& target) override;

  Status IsDirectory(const string& fname) override;

  Status DeleteRecursively(const string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs) override;
  size_t get_readahead_buffer_size() const { return read_ahead_bytes_; }

 private:
  /// \brief Checks if the bucket exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  Status BucketExists(const string& bucket, bool* result);

  /// \brief Checks if the object exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  Status ObjectExists(const string& bucket, const string& object, bool* result);

  /// \brief Checks if the folder exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  Status FolderExists(const string& dirname, bool* result);

  /// \brief Internal version of GetChildren with more knobs.
  ///
  /// If 'recursively' is true, returns all objects in all subfolders.
  /// Otherwise only returns the immediate children in the directory.
  ///
  /// If 'include_self_directory_marker' is true and there is a GCS directory
  /// marker at the path 'dir', GetChildrenBound will return an empty string
  /// as one of the children that represents this marker.
  Status GetChildrenBounded(const string& dir, uint64 max_results,
                            std::vector<string>* result, bool recursively,
                            bool include_self_directory_marker);
  /// Retrieves file statistics assuming fname points to a GCS object.
  Status StatForObject(const string& bucket, const string& object,
                       FileStatistics* stat);
  Status RenameObject(const string& src, const string& target);

  std::unique_ptr<AuthProvider> auth_provider_;
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;

  // The number of bytes to read ahead for buffering purposes in the
  // RandomAccessFile implementation. Defaults to 256Mb.
  size_t read_ahead_bytes_ = 256 * 1024 * 1024;

  // The initial delay for exponential backoffs when retrying failed calls.
  const int64 initial_retry_delay_usec_ = 1000000L;

  TF_DISALLOW_COPY_AND_ASSIGN(GcsFileSystem);
};

/// Google Cloud Storage implementation of a file system with retry on failures.
class RetryingGcsFileSystem : public RetryingFileSystem {
 public:
  RetryingGcsFileSystem()
      : RetryingFileSystem(std::unique_ptr<FileSystem>(new GcsFileSystem)) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_GCS_FILE_SYSTEM_H_
