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

#ifndef TENSORFLOW_TSL_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_
#define TENSORFLOW_TSL_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tsl/platform/cloud/auth_provider.h"
#include "tsl/platform/cloud/compute_engine_metadata_client.h"
#include "tsl/platform/cloud/compute_engine_zone_provider.h"
#include "tsl/platform/cloud/expiring_lru_cache.h"
#include "tsl/platform/cloud/file_block_cache.h"
#include "tsl/platform/cloud/gcs_dns_cache.h"
#include "tsl/platform/cloud/gcs_throttle.h"
#include "tsl/platform/cloud/http_request.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/retrying_file_system.h"
#include "tsl/platform/status.h"

namespace tsl {

class GcsFileSystem;

// The environment variable that overrides the block size for aligned reads from
// GCS. Specified in MB (e.g. "16" = 16 x 1024 x 1024 = 16777216 bytes).
constexpr char kBlockSize[] = "GCS_READ_CACHE_BLOCK_SIZE_MB";
#if defined(LIBTPU_ON_GCE)
// Overwrite the default max block size for `libtpu` BUILDs which do not
// offer a mechanism to override the default through environment variable.
constexpr size_t kDefaultBlockSize = 512 * 1024 * 1024;
#else
constexpr size_t kDefaultBlockSize = 64 * 1024 * 1024;
#endif
// The environment variable that overrides the max size of the LRU cache of
// blocks read from GCS. Specified in MB.
constexpr char kMaxCacheSize[] = "GCS_READ_CACHE_MAX_SIZE_MB";
#if defined(LIBTPU_ON_GCE)
// Overwrite the default max cache size for `libtpu` BUILDs which do not
// offer a mechanism to override the default through environment variable.
constexpr size_t kDefaultMaxCacheSize = 163840LL * 1024LL * 1024LL;
#else
constexpr size_t kDefaultMaxCacheSize = 0;
#endif
// The environment variable that overrides the maximum staleness of cached file
// contents. Once any block of a file reaches this staleness, all cached blocks
// will be evicted on the next read.
constexpr char kMaxStaleness[] = "GCS_READ_CACHE_MAX_STALENESS";
constexpr uint64 kDefaultMaxStaleness = 0;

// Helper function to extract an environment variable and convert it into a
// value of type T.
template <typename T>
bool GetEnvVar(const char* varname, bool (*convert)(absl::string_view, T*),
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

struct UploadSessionHandle {
  std::string session_uri;
  bool resumable;
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
                std::pair<const string, const string>* additional_header,
                bool compose_append);

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  absl::Status NewWritableFile(const string& fname, TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override;

  absl::Status NewAppendableFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override;

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  absl::Status FileExists(const string& fname,
                          TransactionToken* token) override;

  absl::Status Stat(const string& fname, TransactionToken* token,
                    FileStatistics* stat) override;

  absl::Status GetChildren(const string& dir, TransactionToken* token,
                           std::vector<string>* result) override;

  absl::Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                                std::vector<string>* results) override;

  absl::Status DeleteFile(const string& fname,
                          TransactionToken* token) override;

  absl::Status CreateDir(const string& dirname,
                         TransactionToken* token) override;

  absl::Status DeleteDir(const string& dirname,
                         TransactionToken* token) override;

  absl::Status GetFileSize(const string& fname, TransactionToken* token,
                           uint64* file_size) override;

  absl::Status RenameFile(const string& src, const string& target,
                          TransactionToken* token) override;

  absl::Status IsDirectory(const string& fname,
                           TransactionToken* token) override;

  absl::Status DeleteRecursively(const string& dirname, TransactionToken* token,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) override;

  void FlushCaches(TransactionToken* token) override;

  /// Set an object to collect runtime statistics from the GcsFilesystem.
  void SetStats(GcsStatsInterface* stats);

  /// Set an object to collect file block cache stats.
  void SetCacheStats(FileBlockCacheStatsInterface* cache_stats);

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

  bool compose_append() const { return compose_append_; }
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

  absl::Status CreateHttpRequest(std::unique_ptr<HttpRequest>* request);

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
  virtual absl::Status LoadBufferFromGCS(const string& fname, size_t offset,
                                         size_t n, char* buffer,
                                         size_t* bytes_transferred);

  // Creates an upload session for an upcoming GCS object upload.
  virtual absl::Status CreateNewUploadSession(
      uint64 start_offset, const std::string& object_to_upload,
      const std::string& bucket, uint64 file_size, const std::string& gcs_path,
      UploadSessionHandle* session_handle);

  // Uploads object data to session.
  virtual absl::Status UploadToSession(const std::string& session_uri,
                                       uint64 start_offset,
                                       uint64 already_uploaded,
                                       const std::string& tmp_content_filename,
                                       uint64 file_size,
                                       const std::string& file_path);

  /// \brief Requests status of a previously initiated upload session.
  ///
  /// If the upload has already succeeded, sets 'completed' to true.
  /// Otherwise sets 'completed' to false and 'uploaded' to the currently
  /// uploaded size in bytes.
  virtual absl::Status RequestUploadSessionStatus(const string& session_uri,
                                                  uint64 file_size,
                                                  const std::string& gcs_path,
                                                  bool* completed,
                                                  uint64* uploaded);

  absl::Status ParseGcsPathForScheme(absl::string_view fname, string scheme,
                                     bool empty_object_ok, string* bucket,
                                     string* object);

  /// \brief Splits a GCS path to a bucket and an object.
  ///
  /// For example, "gs://bucket-name/path/to/file.txt" gets split into
  /// "bucket-name" and "path/to/file.txt".
  /// If fname only contains the bucket and empty_object_ok = true, the returned
  /// object is empty.
  virtual absl::Status ParseGcsPath(absl::string_view fname,
                                    bool empty_object_ok, string* bucket,
                                    string* object);

  std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client_;

  // Used by a subclass.
  TimeoutConfig timeouts_;

  /// The retry configuration used for retrying failed calls.
  RetryConfig retry_config_;

 private:
  // GCS file statistics.
  struct GcsFileStat {
    FileStatistics base;
    int64_t generation_number = 0;
  };

  /// \brief Checks if the bucket exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  absl::Status BucketExists(const string& bucket, bool* result);

  /// \brief Retrieves the GCS bucket location. Returns OK if the location was
  /// retrieved.
  ///
  /// Given a string bucket the GCS bucket metadata API will be called and the
  /// location string filled with the location of the bucket.
  ///
  /// This requires the bucket metadata permission.
  /// Repeated calls for the same bucket are cached so this function can be
  /// called frequently without causing an extra API call
  absl::Status GetBucketLocation(const string& bucket, string* location);

  /// \brief Check if the GCS buckets location is allowed with the current
  /// constraint configuration
  absl::Status CheckBucketLocationConstraint(const string& bucket);

  /// \brief Given the input bucket `bucket`, fills `result_buffer` with the
  /// results of the metadata. Returns OK if the API call succeeds without
  /// error.
  absl::Status GetBucketMetadata(const string& bucket,
                                 std::vector<char>* result_buffer);

  /// \brief Checks if the object exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  absl::Status ObjectExists(const string& fname, const string& bucket,
                            const string& object, bool* result);

  /// \brief Checks if the folder exists. Returns OK if the check succeeded.
  ///
  /// 'result' is set if the function returns OK. 'result' cannot be nullptr.
  absl::Status FolderExists(const string& dirname, bool* result);

  /// \brief Internal version of GetChildren with more knobs.
  ///
  /// If 'recursively' is true, returns all objects in all subfolders.
  /// Otherwise only returns the immediate children in the directory.
  ///
  /// If 'include_self_directory_marker' is true and there is a GCS directory
  /// marker at the path 'dir', GetChildrenBound will return an empty string
  /// as one of the children that represents this marker.
  absl::Status GetChildrenBounded(const string& dir, uint64 max_results,
                                  std::vector<string>* result, bool recursively,
                                  bool include_self_directory_marker);

  /// Retrieves file statistics assuming fname points to a GCS object. The data
  /// may be read from cache or from GCS directly.
  absl::Status StatForObject(const string& fname, const string& bucket,
                             const string& object, GcsFileStat* stat);
  /// Retrieves file statistics of file fname directly from GCS.
  absl::Status UncachedStatForObject(const string& fname, const string& bucket,
                                     const string& object, GcsFileStat* stat);

  absl::Status RenameObject(const string& src, const string& target);

  // Clear all the caches related to the file with name `filename`.
  void ClearFileCaches(const string& fname);

  mutex mu_;
  std::unique_ptr<AuthProvider> auth_provider_ TF_GUARDED_BY(mu_);
  std::shared_ptr<HttpRequest::Factory> http_request_factory_;
  std::unique_ptr<ZoneProvider> zone_provider_;

  // Reads smaller than block_size_ will trigger a read of block_size_.
  uint64 block_size_;

  // block_cache_lock_ protects the file_block_cache_ pointer (Note that
  // FileBlockCache instances are themselves threadsafe).
  mutex block_cache_lock_;
  std::unique_ptr<FileBlockCache> file_block_cache_
      TF_GUARDED_BY(block_cache_lock_);

  bool cache_enabled_;
  std::unique_ptr<GcsDnsCache> dns_cache_;
  GcsThrottle throttle_;

  using StatCache = ExpiringLRUCache<GcsFileStat>;
  std::unique_ptr<StatCache> stat_cache_;

  using MatchingPathsCache = ExpiringLRUCache<std::vector<string>>;
  std::unique_ptr<MatchingPathsCache> matching_paths_cache_;

  using BucketLocationCache = ExpiringLRUCache<string>;
  std::unique_ptr<BucketLocationCache> bucket_location_cache_;
  std::unordered_set<string> allowed_locations_;
  bool compose_append_;

  GcsStatsInterface* stats_ = nullptr;  // Not owned.

  // Additional header material to be transmitted with all GCS requests
  std::unique_ptr<std::pair<const string, const string>> additional_header_;

  GcsFileSystem(const GcsFileSystem&) = delete;
  void operator=(const GcsFileSystem&) = delete;
};

/// Google Cloud Storage implementation of a file system with retry on failures.
class RetryingGcsFileSystem : public RetryingFileSystem<GcsFileSystem> {
 public:
  RetryingGcsFileSystem();
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_CLOUD_GCS_FILE_SYSTEM_H_
