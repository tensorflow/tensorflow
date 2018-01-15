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
#include "tensorflow/core/platform/cloud/expiring_lru_cache.h"
#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/cloud/gcs_dns_cache.h"
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
  struct TimeoutConfig;

  GcsFileSystem();
  GcsFileSystem(std::unique_ptr<AuthProvider> auth_provider,
                std::unique_ptr<HttpRequest::Factory> http_request_factory,
                size_t block_size, size_t max_bytes, uint64 max_staleness,
                uint64 stat_cache_max_age, size_t stat_cache_max_entries,
                uint64 matching_paths_cache_max_age,
                size_t matching_paths_cache_max_entries,
                int64 initial_retry_delay_usec, TimeoutConfig timeouts);

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

  void FlushCaches() override;

  /// These accessors are mainly for testing purposes, to verify that the
  /// environment variables that control these parameters are handled correctly.
  size_t block_size() const { return file_block_cache_->block_size(); }
  size_t max_bytes() const { return file_block_cache_->max_bytes(); }
  uint64 max_staleness() const { return file_block_cache_->max_staleness(); }
  TimeoutConfig timeouts() const { return timeouts_; }

  uint64 stat_cache_max_age() const { return stat_cache_->max_age(); }
  size_t stat_cache_max_entries() const { return stat_cache_->max_entries(); }

  uint64 matching_paths_cache_max_age() const {
    return matching_paths_cache_->max_age();
  }
  size_t matching_paths_cache_max_entries() const {
    return matching_paths_cache_->max_entries();
  }

  /// Structure containing the information for timeouts related to accessing the
  /// GCS APIs.
  ///
  /// All values are in seconds.
  struct TimeoutConfig {
    // The request connection timeout. If a connection cannot be established
    // within `connect` seconds, abort the request.
    uint32 connect = 120;  // 2 minutes

    // The request idle timeout. If a request has seen no activity in `idle`
    // seconds, abort the request.
    uint32 idle = 60;  // 1 minute

    // The maximum total time a metadata request can take. If a request has not
    // completed within `metadata` seconds, the request is aborted.
    uint32 metadata = 3600;  // 1 hour

    // The maximum total time a block read request can take. If a request has
    // not completed within `read` seconds, the request is aborted.
    uint32 read = 3600;  // 1 hour

    // The maximum total time an upload request can take. If a request has not
    // completed within `write` seconds, the request is aborted.
    uint32 write = 3600;  // 1 hour

    TimeoutConfig() {}
    TimeoutConfig(uint32 connect, uint32 idle, uint32 metadata, uint32 read,
                  uint32 write)
        : connect(connect),
          idle(idle),
          metadata(metadata),
          read(read),
          write(write) {}
  };

  Status CreateHttpRequest(std::unique_ptr<HttpRequest>* request);

 private:
  /// \brief Checks if the bucket exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  Status BucketExists(const string& bucket, bool* result);

  /// \brief Checks if the object exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  Status ObjectExists(const string& fname, const string& bucket,
                      const string& object, bool* result);

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
  Status StatForObject(const string& fname, const string& bucket,
                       const string& object, FileStatistics* stat);
  Status RenameObject(const string& src, const string& target);

  std::unique_ptr<FileBlockCache> MakeFileBlockCache(size_t block_size,
                                                     size_t max_bytes,
                                                     uint64 max_staleness);

  /// Loads file contents from GCS for a given filename, offset, and length.
  Status LoadBufferFromGCS(const string& filename, size_t offset, size_t n,
                           char* buffer, size_t* bytes_transferred);

  std::unique_ptr<AuthProvider> auth_provider_;
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;
  std::unique_ptr<FileBlockCache> file_block_cache_;
  std::unique_ptr<GcsDnsCache> dns_cache_;

  using StatCache = ExpiringLRUCache<FileStatistics>;
  std::unique_ptr<StatCache> stat_cache_;

  using MatchingPathsCache = ExpiringLRUCache<std::vector<string>>;
  std::unique_ptr<MatchingPathsCache> matching_paths_cache_;

  TimeoutConfig timeouts_;

  /// The initial delay for exponential backoffs when retrying failed calls.
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
