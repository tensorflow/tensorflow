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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cloud/auth_provider.h"
#include "tensorflow/core/platform/cloud/compute_engine_metadata_client.h"
#include "tensorflow/core/platform/cloud/compute_engine_zone_provider.h"
#include "tensorflow/core/platform/cloud/expiring_lru_cache.h"
#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/cloud/gcs_dns_cache.h"
#include "tensorflow/core/platform/cloud/gcs_throttle.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/cloud/retrying_file_system.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

class GcsFileSystem;

// The environment variable that overrides the block size for aligned reads from
// GCS. Specified in MB (e.g. "16" = 16 x 1024 x 1024 = 16777216 bytes).
constexpr char kBlockSize[] = "GCS_READ_CACHE_BLOCK_SIZE_MB";
constexpr size_t kDefaultBlockSize = 64 * 1024 * 1024;
// The environment variable that overrides the max size of the LRU cache of
// blocks read from GCS. Specified in MB.
constexpr char kMaxCacheSize[] = "GCS_READ_CACHE_MAX_SIZE_MB";
constexpr size_t kDefaultMaxCacheSize = 0;
// The environment variable that overrides the maximum staleness of cached file
// contents. Once any block of a file reaches this staleness, all cached blocks
// will be evicted on the next read.
constexpr char kMaxStaleness[] = "GCS_READ_CACHE_MAX_STALENESS";
constexpr uint64 kDefaultMaxStaleness = 0;

// Helper function to extract an environment variable and convert it into a
// value of type T.
template <typename T>
bool GetEnvVar(const char* varname, bool (*convert)(StringPiece, T*),
               T* value) {
  const char* env_value = std::getenv(varname);
  if (env_value == nullptr) {
    return false;
  }
  return convert(env_value, value);
}

/// GcsStatsInterface allows for instrumentation of the GCS file system.
///
/// GcsStatsInterface and its subclasses must be safe to use from multiple
/// threads concurrently.
///
/// WARNING! This is an experimental interface that may change or go away at any
/// time.
class GcsStatsInterface {
 public:
  /// Configure is called by the GcsFileSystem to provide instrumentation hooks.
  ///
  /// Note: Configure can be called multiple times (e.g. if the block cache is
  /// re-initialized).
  virtual void Configure(GcsFileSystem* fs, GcsThrottle* throttle,
                         const FileBlockCache* block_cache) = 0;

  /// RecordBlockLoadRequest is called to record a block load request is about
  /// to be made.
  virtual void RecordBlockLoadRequest(const string& file, size_t offset) = 0;

  /// RecordBlockRetrieved is called once a block within the file has been
  /// retrieved.
  virtual void RecordBlockRetrieved(const string& file, size_t offset,
                                    size_t bytes_transferred) = 0;

  // RecordStatObjectRequest is called once a statting object request over GCS
  // is about to be made.
  virtual void RecordStatObjectRequest() = 0;

  /// HttpStats is called to optionally provide a RequestStats listener
  /// to be annotated on every HTTP request made to the GCS API.
  ///
  /// HttpStats() may return nullptr.
  virtual HttpRequest::RequestStats* HttpStats() = 0;

  virtual ~GcsStatsInterface() = default;
};

/// Google Cloud Storage implementation of a file system.
///
/// The clients should use RetryingGcsFileSystem defined below,
/// which adds retry logic to GCS operations.
class GcsFileSystem : public FileSystem {
 public:
  struct TimeoutConfig;

  // Main constructor used (via RetryingFileSystem) throughout Tensorflow
  explicit GcsFileSystem(bool make_default_cache = true);
  // Used mostly for unit testing or use cases which need to customize the
  // filesystem from defaults
  GcsFileSystem(std::unique_ptr<AuthProvider> auth_provider,
                std::unique_ptr<HttpRequest::Factory> http_request_factory,
                std::unique_ptr<ZoneProvider> zone_provider, size_t block_size,
                size_t max_bytes, uint64 max_staleness,
                uint64 stat_cache_max_age, size_t stat_cache_max_entries,
                uint64 matching_paths_cache_max_age,
                size_t matching_paths_cache_max_entries,
                RetryConfig retry_config, TimeoutConfig timeouts,
                const std::unordered_set<string>& allowed_locations,
                std::pair<const string, const string>* additional_header);

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
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

  /// Set an object to collect runtime statistics from the GcsFilesystem.
  void SetStats(GcsStatsInterface* stats);

  /// These accessors are mainly for testing purposes, to verify that the
  /// environment variables that control these parameters are handled correctly.
  size_t block_size() {
    tf_shared_lock l(block_cache_lock_);
    return file_block_cache_->block_size();
  }
  size_t max_bytes() {
    tf_shared_lock l(block_cache_lock_);
    return file_block_cache_->max_bytes();
  }
  uint64 max_staleness() {
    tf_shared_lock l(block_cache_lock_);
    return file_block_cache_->max_staleness();
  }
  TimeoutConfig timeouts() const { return timeouts_; }
  std::unordered_set<string> allowed_locations() const {
    return allowed_locations_;
  }
  string additional_header_name() const {
    return additional_header_ ? additional_header_->first : "";
  }
  string additional_header_value() const {
    return additional_header_ ? additional_header_->second : "";
  }

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

  /// \brief Sets a new AuthProvider on the GCS FileSystem.
  ///
  /// The new auth provider will be used for all subsequent requests.
  void SetAuthProvider(std::unique_ptr<AuthProvider> auth_provider);

  /// \brief Resets the block cache and re-instantiates it with the new values.
  ///
  /// This method can be used to clear the existing block cache and/or to
  /// re-configure the block cache for different values.
  ///
  /// Note: the existing block cache is not cleaned up until all existing files
  /// have been closed.
  void ResetFileBlockCache(size_t block_size_bytes, size_t max_bytes,
                           uint64 max_staleness_secs);

 protected:
  virtual std::unique_ptr<FileBlockCache> MakeFileBlockCache(
      size_t block_size, size_t max_bytes, uint64 max_staleness);

  /// Loads file contents from GCS for a given filename, offset, and length.
  Status LoadBufferFromGCS(const string& fname, size_t offset, size_t n,
                           char* buffer, size_t* bytes_transferred);

 private:
  // GCS file statistics.
  struct GcsFileStat {
    FileStatistics base;
    int64 generation_number = 0;
  };

  /// \brief Checks if the bucket exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  Status BucketExists(const string& bucket, bool* result);

  /// \brief Retrieves the GCS bucket location. Returns OK if the location was
  /// retrieved.
  ///
  /// Given a string bucket the GCS bucket metadata API will be called and the
  /// location string filled with the location of the bucket.
  ///
  /// This requires the bucket metadata permission.
  /// Repeated calls for the same bucket are cached so this function can be
  /// called frequently without causing an extra API call
  Status GetBucketLocation(const string& bucket, string* location);

  /// \brief Check if the GCS buckets location is allowed with the current
  /// constraint configuration
  Status CheckBucketLocationConstraint(const string& bucket);

  /// \brief Given the input bucket `bucket`, fills `result_buffer` with the
  /// results of the metadata. Returns OK if the API call succeeds without
  /// error.
  Status GetBucketMetadata(const string& bucket,
                           std::vector<char>* result_buffer);

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

  /// Retrieves file statistics assuming fname points to a GCS object. The data
  /// may be read from cache or from GCS directly.
  Status StatForObject(const string& fname, const string& bucket,
                       const string& object, GcsFileStat* stat);
  /// Retrieves file statistics of file fname directly from GCS.
  Status UncachedStatForObject(const string& fname, const string& bucket,
                               const string& object, GcsFileStat* stat);

  Status RenameObject(const string& src, const string& target);

  // Clear all the caches related to the file with name `filename`.
  void ClearFileCaches(const string& fname);

  mutex mu_;
  std::unique_ptr<AuthProvider> auth_provider_ GUARDED_BY(mu_);
  std::shared_ptr<HttpRequest::Factory> http_request_factory_;
  std::unique_ptr<ZoneProvider> zone_provider_;

  // Reads smaller than block_size_ will trigger a read of block_size_.
  uint64 block_size_;

  // block_cache_lock_ protects the file_block_cache_ pointer (Note that
  // FileBlockCache instances are themselves threadsafe).
  mutex block_cache_lock_;
  std::unique_ptr<FileBlockCache> file_block_cache_
      GUARDED_BY(block_cache_lock_);
  std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client_;
  std::unique_ptr<GcsDnsCache> dns_cache_;
  GcsThrottle throttle_;

  using StatCache = ExpiringLRUCache<GcsFileStat>;
  std::unique_ptr<StatCache> stat_cache_;

  using MatchingPathsCache = ExpiringLRUCache<std::vector<string>>;
  std::unique_ptr<MatchingPathsCache> matching_paths_cache_;

  using BucketLocationCache = ExpiringLRUCache<string>;
  std::unique_ptr<BucketLocationCache> bucket_location_cache_;
  std::unordered_set<string> allowed_locations_;

  TimeoutConfig timeouts_;

  GcsStatsInterface* stats_ = nullptr;  // Not owned.

  /// The initial delay for exponential backoffs when retrying failed calls.
  RetryConfig retry_config_;

  // Additional header material to be transmitted with all GCS requests
  std::unique_ptr<std::pair<const string, const string>> additional_header_;

  TF_DISALLOW_COPY_AND_ASSIGN(GcsFileSystem);
};

/// Google Cloud Storage implementation of a file system with retry on failures.
class RetryingGcsFileSystem : public RetryingFileSystem<GcsFileSystem> {
 public:
  RetryingGcsFileSystem()
      : RetryingFileSystem(std::unique_ptr<GcsFileSystem>(new GcsFileSystem),
                           RetryConfig(100000 /* init_delay_time_us */)) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_
