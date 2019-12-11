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

#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#ifdef _WIN32
#include <io.h>  // for _mktemp
#endif
#include "absl/base/macros.h"
#include "include/json/json.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/cloud/google_auth_provider.h"
#include "tensorflow/core/platform/cloud/ram_file_block_cache.h"
#include "tensorflow/core/platform/retrying_utils.h"
#include "tensorflow/core/platform/cloud/time_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/thread_annotations.h"

#ifdef _WIN32
#ifdef DeleteFile
#undef DeleteFile
#endif
#endif

namespace tensorflow {
namespace {

constexpr char kGcsUriBase[] = "https://www.googleapis.com/storage/v1/";
constexpr char kGcsUploadUriBase[] =
    "https://www.googleapis.com/upload/storage/v1/";
constexpr char kStorageHost[] = "storage.googleapis.com";
constexpr char kBucketMetadataLocationKey[] = "location";
constexpr size_t kReadAppendableFileBufferSize = 1024 * 1024;  // In bytes.
constexpr int kGetChildrenDefaultPageSize = 1000;
// The HTTP response code "308 Resume Incomplete".
constexpr uint64 HTTP_CODE_RESUME_INCOMPLETE = 308;
// The environment variable that overrides the size of the readahead buffer.
ABSL_DEPRECATED("Use GCS_READ_CACHE_BLOCK_SIZE_MB instead.")
constexpr char kReadaheadBufferSize[] = "GCS_READAHEAD_BUFFER_SIZE_BYTES";
// The environment variable that overrides the maximum age of entries in the
// Stat cache. A value of 0 (the default) means nothing is cached.
constexpr char kStatCacheMaxAge[] = "GCS_STAT_CACHE_MAX_AGE";
constexpr uint64 kStatCacheDefaultMaxAge = 5;
// The environment variable that overrides the maximum number of entries in the
// Stat cache.
constexpr char kStatCacheMaxEntries[] = "GCS_STAT_CACHE_MAX_ENTRIES";
constexpr size_t kStatCacheDefaultMaxEntries = 1024;
// The environment variable that overrides the maximum age of entries in the
// GetMatchingPaths cache. A value of 0 (the default) means nothing is cached.
constexpr char kMatchingPathsCacheMaxAge[] = "GCS_MATCHING_PATHS_CACHE_MAX_AGE";
constexpr uint64 kMatchingPathsCacheDefaultMaxAge = 0;
// The environment variable that overrides the maximum number of entries in the
// GetMatchingPaths cache.
constexpr char kMatchingPathsCacheMaxEntries[] =
    "GCS_MATCHING_PATHS_CACHE_MAX_ENTRIES";
constexpr size_t kMatchingPathsCacheDefaultMaxEntries = 1024;
// Number of bucket locations cached, most workloads wont touch more than one
// bucket so this limit is set fairly low
constexpr size_t kBucketLocationCacheMaxEntries = 10;
// ExpiringLRUCache doesnt support any "cache forever" option
constexpr size_t kCacheNeverExpire = std::numeric_limits<uint64>::max();
// The file statistics returned by Stat() for directories.
const FileStatistics DIRECTORY_STAT(0, 0, true);
// Some environments exhibit unreliable DNS resolution. Set this environment
// variable to a positive integer describing the frequency used to refresh the
// userspace DNS cache.
constexpr char kResolveCacheSecs[] = "GCS_RESOLVE_REFRESH_SECS";
// The environment variable to configure the http request's connection timeout.
constexpr char kRequestConnectionTimeout[] =
    "GCS_REQUEST_CONNECTION_TIMEOUT_SECS";
// The environment variable to configure the http request's idle timeout.
constexpr char kRequestIdleTimeout[] = "GCS_REQUEST_IDLE_TIMEOUT_SECS";
// The environment variable to configure the overall request timeout for
// metadata requests.
constexpr char kMetadataRequestTimeout[] = "GCS_METADATA_REQUEST_TIMEOUT_SECS";
// The environment variable to configure the overall request timeout for
// block reads requests.
constexpr char kReadRequestTimeout[] = "GCS_READ_REQUEST_TIMEOUT_SECS";
// The environment variable to configure the overall request timeout for
// upload requests.
constexpr char kWriteRequestTimeout[] = "GCS_WRITE_REQUEST_TIMEOUT_SECS";
// The environment variable to configure an additional header to send with
// all requests to GCS (format HEADERNAME:HEADERCONTENT)
constexpr char kAdditionalRequestHeader[] = "GCS_ADDITIONAL_REQUEST_HEADER";
// The environment variable to configure the throttle (format: <int64>)
constexpr char kThrottleRate[] = "GCS_THROTTLE_TOKEN_RATE";
// The environment variable to configure the token bucket size (format: <int64>)
constexpr char kThrottleBucket[] = "GCS_THROTTLE_BUCKET_SIZE";
// The environment variable that controls the number of tokens per request.
// (format: <int64>)
constexpr char kTokensPerRequest[] = "GCS_TOKENS_PER_REQUEST";
// The environment variable to configure the initial tokens (format: <int64>)
constexpr char kInitialTokens[] = "GCS_INITIAL_TOKENS";

// The environment variable to customize which GCS bucket locations are allowed,
// if the list is empty defaults to using the region of the zone (format, comma
// delimited list). Requires 'storage.buckets.get' permission.
constexpr char kAllowedBucketLocations[] = "GCS_ALLOWED_BUCKET_LOCATIONS";
// When this value is passed as an allowed location detects the zone tensorflow
// is running in and restricts to buckets in that region.
constexpr char kDetectZoneSentinalValue[] = "auto";

// TODO: DO NOT use a hardcoded path
Status GetTmpFilename(string* filename) {
#ifndef _WIN32
  char buffer[] = "/tmp/gcs_filesystem_XXXXXX";
  int fd = mkstemp(buffer);
  if (fd < 0) {
    return errors::Internal("Failed to create a temporary file.");
  }
  close(fd);
#else
  char buffer[] = "/tmp/gcs_filesystem_XXXXXX";
  char* ret = _mktemp(buffer);
  if (ret == nullptr) {
    return errors::Internal("Failed to create a temporary file.");
  }
#endif
  *filename = buffer;
  return Status::OK();
}

/// \brief Splits a GCS path to a bucket and an object.
///
/// For example, "gs://bucket-name/path/to/file.txt" gets split into
/// "bucket-name" and "path/to/file.txt".
/// If fname only contains the bucket and empty_object_ok = true, the returned
/// object is empty.
Status ParseGcsPath(StringPiece fname, bool empty_object_ok, string* bucket,
                    string* object) {
  StringPiece scheme, bucketp, objectp;
  io::ParseURI(fname, &scheme, &bucketp, &objectp);
  if (scheme != "gs") {
    return errors::InvalidArgument("GCS path doesn't start with 'gs://': ",
                                   fname);
  }
  *bucket = string(bucketp);
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("GCS path doesn't contain a bucket name: ",
                                   fname);
  }
  absl::ConsumePrefix(&objectp, "/");
  *object = string(objectp);
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("GCS path doesn't contain an object name: ",
                                   fname);
  }
  return Status::OK();
}

/// Appends a trailing slash if the name doesn't already have one.
string MaybeAppendSlash(const string& name) {
  if (name.empty()) {
    return "/";
  }
  if (name.back() != '/') {
    return strings::StrCat(name, "/");
  }
  return name;
}

// io::JoinPath() doesn't work in cases when we want an empty subpath
// to result in an appended slash in order for directory markers
// to be processed correctly: "gs://a/b" + "" should give "gs://a/b/".
string JoinGcsPath(const string& path, const string& subpath) {
  return strings::StrCat(MaybeAppendSlash(path), subpath);
}

/// \brief Returns the given paths appending all their subfolders.
///
/// For every path X in the list, every subfolder in X is added to the
/// resulting list.
/// For example:
///  - for 'a/b/c/d' it will append 'a', 'a/b' and 'a/b/c'
///  - for 'a/b/c/' it will append 'a', 'a/b' and 'a/b/c'
std::set<string> AddAllSubpaths(const std::vector<string>& paths) {
  std::set<string> result;
  result.insert(paths.begin(), paths.end());
  for (const string& path : paths) {
    StringPiece subpath = io::Dirname(path);
    while (!subpath.empty()) {
      result.emplace(string(subpath));
      subpath = io::Dirname(subpath);
    }
  }
  return result;
}

Status ParseJson(StringPiece json, Json::Value* result) {
  Json::Reader reader;
  if (!reader.parse(json.data(), json.data() + json.size(), *result)) {
    return errors::Internal("Couldn't parse JSON response from GCS.");
  }
  return Status::OK();
}

Status ParseJson(const std::vector<char>& json, Json::Value* result) {
  return ParseJson(StringPiece{json.data(), json.size()}, result);
}

/// Reads a JSON value with the given name from a parent JSON value.
Status GetValue(const Json::Value& parent, const char* name,
                Json::Value* result) {
  *result = parent.get(name, Json::Value::null);
  if (result->isNull()) {
    return errors::Internal("The field '", name,
                            "' was expected in the JSON response.");
  }
  return Status::OK();
}

/// Reads a string JSON value with the given name from a parent JSON value.
Status GetStringValue(const Json::Value& parent, const char* name,
                      string* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isString()) {
    return errors::Internal(
        "The field '", name,
        "' in the JSON response was expected to be a string.");
  }
  *result = result_value.asString();
  return Status::OK();
}

/// Reads a long JSON value with the given name from a parent JSON value.
Status GetInt64Value(const Json::Value& parent, const char* name,
                     int64* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (result_value.isNumeric()) {
    *result = result_value.asInt64();
    return Status::OK();
  }
  if (result_value.isString() &&
      strings::safe_strto64(result_value.asCString(), result)) {
    return Status::OK();
  }
  return errors::Internal(
      "The field '", name,
      "' in the JSON response was expected to be a number.");
}

/// Reads a boolean JSON value with the given name from a parent JSON value.
Status GetBoolValue(const Json::Value& parent, const char* name, bool* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isBool()) {
    return errors::Internal(
        "The field '", name,
        "' in the JSON response was expected to be a boolean.");
  }
  *result = result_value.asBool();
  return Status::OK();
}

/// A GCS-based implementation of a random access file with an LRU block cache.
class GcsRandomAccessFile : public RandomAccessFile {
 public:
  using ReadFn =
      std::function<Status(const string& filename, uint64 offset, size_t n,
                           StringPiece* result, char* scratch)>;

  GcsRandomAccessFile(const string& filename, ReadFn read_fn)
      : filename_(filename), read_fn_(std::move(read_fn)) {}

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  /// The implementation of reads with an LRU block cache. Thread safe.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    return read_fn_(filename_, offset, n, result, scratch);
  }

 private:
  /// The filename of this file.
  const string filename_;
  /// The implementation of the read operation (provided by the GCSFileSystem).
  const ReadFn read_fn_;
};

/// A GCS-based implementation of a random access file with a read buffer.
class BufferedGcsRandomAccessFile : public RandomAccessFile {
 public:
  using ReadFn =
      std::function<Status(const string& filename, uint64 offset, size_t n,
                           StringPiece* result, char* scratch)>;

  // Initialize the reader. Provided read_fn should be thread safe.
  BufferedGcsRandomAccessFile(const string& filename, uint64 buffer_size,
                              ReadFn read_fn)
      : filename_(filename),
        read_fn_(std::move(read_fn)),
        buffer_size_(buffer_size),
        buffer_start_(0) {}

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  /// The implementation of reads with an read buffer. Thread safe.
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  /// because of EOF.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    if (n > buffer_size_) {
      return read_fn_(filename_, offset, n, result, scratch);
    }
    {
      mutex_lock l(buffer_mutex_);
      size_t buffer_end = buffer_start_ + buffer_.size();
      size_t copy_size = 0;
      if (offset < buffer_end && offset >= buffer_start_) {
        copy_size = std::min(n, static_cast<size_t>(buffer_end - offset));
        memcpy(scratch, buffer_.data() + (offset - buffer_start_), copy_size);
        *result = StringPiece(scratch, copy_size);
      }
      if (copy_size < n) {
        // Try reading from the file regardless of previous read status.
        // The file might have grown since the last read.
        Status status = FillBuffer(offset + copy_size);
        if (!status.ok() && status.code() != errors::Code::OUT_OF_RANGE) {
          // Empty the buffer to avoid caching bad reads.
          buffer_.resize(0);
          return status;
        }
        size_t remaining_copy = std::min(n - copy_size, buffer_.size());
        memcpy(scratch + copy_size, buffer_.data(), remaining_copy);
        copy_size += remaining_copy;
        *result = StringPiece(scratch, copy_size);
      }
      if (copy_size < n) {
        return errors::OutOfRange("EOF reached. Requested to read ", n,
                                  " bytes from ", offset, " but only got ",
                                  copy_size, " bytes.");
      }
    }
    return Status::OK();
  }

 private:
  Status FillBuffer(uint64 start) const
      EXCLUSIVE_LOCKS_REQUIRED(buffer_mutex_) {
    buffer_start_ = start;
    buffer_.resize(buffer_size_);
    StringPiece str_piece;
    Status status = read_fn_(filename_, buffer_start_, buffer_size_, &str_piece,
                             &(buffer_[0]));
    buffer_.resize(str_piece.size());
    return status;
  }

  // The filename of this file.
  const string filename_;

  // The implementation of the read operation (provided by the GCSFileSystem).
  const ReadFn read_fn_;

  // Size of buffer that we read from GCS each time we send a request.
  const uint64 buffer_size_;

  // Mutex for buffering operations that can be accessed from multiple threads.
  // The following members are mutable in order to provide a const Read.
  mutable mutex buffer_mutex_;

  // Offset of buffer from start of the file.
  mutable uint64 buffer_start_ GUARDED_BY(buffer_mutex_);

  mutable string buffer_ GUARDED_BY(buffer_mutex_);
};

/// \brief GCS-based implementation of a writeable file.
///
/// Since GCS objects are immutable, this implementation writes to a local
/// tmp file and copies it to GCS on flush/close.
class GcsWritableFile : public WritableFile {
 public:
  GcsWritableFile(const string& bucket, const string& object,
                  GcsFileSystem* filesystem,
                  GcsFileSystem::TimeoutConfig* timeouts,
                  std::function<void()> file_cache_erase,
                  RetryConfig retry_config)
      : bucket_(bucket),
        object_(object),
        filesystem_(filesystem),
        timeouts_(timeouts),
        file_cache_erase_(std::move(file_cache_erase)),
        sync_needed_(true),
        retry_config_(retry_config) {
    // TODO: to make it safer, outfile_ should be constructed from an FD
    if (GetTmpFilename(&tmp_content_filename_).ok()) {
      outfile_.open(tmp_content_filename_,
                    std::ofstream::binary | std::ofstream::app);
    }
  }

  /// \brief Constructs the writable file in append mode.
  ///
  /// tmp_content_filename should contain a path of an existing temporary file
  /// with the content to be appended. The class takes onwnership of the
  /// specified tmp file and deletes it on close.
  GcsWritableFile(const string& bucket, const string& object,
                  GcsFileSystem* filesystem, const string& tmp_content_filename,
                  GcsFileSystem::TimeoutConfig* timeouts,
                  std::function<void()> file_cache_erase,
                  RetryConfig retry_config)
      : bucket_(bucket),
        object_(object),
        filesystem_(filesystem),
        timeouts_(timeouts),
        file_cache_erase_(std::move(file_cache_erase)),
        sync_needed_(true),
        retry_config_(retry_config) {
    tmp_content_filename_ = tmp_content_filename;
    outfile_.open(tmp_content_filename_,
                  std::ofstream::binary | std::ofstream::app);
  }

  ~GcsWritableFile() override { Close().IgnoreError(); }

  Status Append(StringPiece data) override {
    TF_RETURN_IF_ERROR(CheckWritable());
    sync_needed_ = true;
    outfile_ << data;
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not append to the internal temporary file.");
    }
    return Status::OK();
  }

  Status Close() override {
    if (outfile_.is_open()) {
      TF_RETURN_IF_ERROR(Sync());
      outfile_.close();
      std::remove(tmp_content_filename_.c_str());
    }
    return Status::OK();
  }

  Status Flush() override { return Sync(); }

  Status Name(StringPiece* result) const override {
    return errors::Unimplemented("GCSWritableFile does not support Name()");
  }

  Status Sync() override {
    TF_RETURN_IF_ERROR(CheckWritable());
    if (!sync_needed_) {
      return Status::OK();
    }
    Status status = SyncImpl();
    if (status.ok()) {
      sync_needed_ = false;
    }
    return status;
  }

  Status Tell(int64* position) override {
    *position = outfile_.tellp();
    if (*position == -1) {
      return errors::Internal("tellp on the internal temporary file failed");
    }
    return Status::OK();
  }

 private:
  /// Copies the current version of the file to GCS.
  ///
  /// This SyncImpl() uploads the object to GCS.
  /// In case of a failure, it resumes failed uploads as recommended by the GCS
  /// resumable API documentation. When the whole upload needs to be
  /// restarted, Sync() returns UNAVAILABLE and relies on RetryingFileSystem.
  Status SyncImpl() {
    outfile_.flush();
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not write to the internal temporary file.");
    }
    string session_uri;
    TF_RETURN_IF_ERROR(CreateNewUploadSession(&session_uri));
    uint64 already_uploaded = 0;
    bool first_attempt = true;
    const Status upload_status = RetryingUtils::CallWithRetries(
        [&first_attempt, &already_uploaded, &session_uri, this]() {
          if (!first_attempt) {
            bool completed;
            TF_RETURN_IF_ERROR(RequestUploadSessionStatus(
                session_uri, &completed, &already_uploaded));
            if (completed) {
              // Erase the file from the file cache on every successful write.
              file_cache_erase_();
              // It's unclear why UploadToSession didn't return OK in the
              // previous attempt, but GCS reports that the file is fully
              // uploaded, so succeed.
              return Status::OK();
            }
          }
          first_attempt = false;
          return UploadToSession(session_uri, already_uploaded);
        },
        retry_config_);
    if (upload_status.code() == errors::Code::NOT_FOUND) {
      // GCS docs recommend retrying the whole upload. We're relying on the
      // RetryingFileSystem to retry the Sync() call.
      return errors::Unavailable(
          strings::StrCat("Upload to gs://", bucket_, "/", object_,
                          " failed, caused by: ", upload_status.ToString()));
    }
    return upload_status;
  }

  Status CheckWritable() const {
    if (!outfile_.is_open()) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    return Status::OK();
  }

  Status GetCurrentFileSize(uint64* size) {
    const auto tellp = outfile_.tellp();
    if (tellp == static_cast<std::streampos>(-1)) {
      return errors::Internal(
          "Could not get the size of the internal temporary file.");
    }
    *size = tellp;
    return Status::OK();
  }

  /// Initiates a new resumable upload session.
  Status CreateNewUploadSession(string* session_uri) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

    std::vector<char> output_buffer;
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(filesystem_->CreateHttpRequest(&request));

    request->SetUri(strings::StrCat(
        kGcsUploadUriBase, "b/", bucket_,
        "/o?uploadType=resumable&name=", request->EscapeString(object_)));
    request->AddHeader("X-Upload-Content-Length", std::to_string(file_size));
    request->SetPostEmptyBody();
    request->SetResultBuffer(&output_buffer);
    request->SetTimeouts(timeouts_->connect, timeouts_->idle,
                         timeouts_->metadata);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        request->Send(), " when initiating an upload to ", GetGcsPath());
    *session_uri = request->GetResponseHeader("Location");
    if (session_uri->empty()) {
      return errors::Internal("Unexpected response from GCS when writing to ",
                              GetGcsPath(),
                              ": 'Location' header not returned.");
    }
    return Status::OK();
  }

  /// \brief Requests status of a previously initiated upload session.
  ///
  /// If the upload has already succeeded, sets 'completed' to true.
  /// Otherwise sets 'completed' to false and 'uploaded' to the currently
  /// uploaded size in bytes.
  Status RequestUploadSessionStatus(const string& session_uri, bool* completed,
                                    uint64* uploaded) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(filesystem_->CreateHttpRequest(&request));
    request->SetUri(session_uri);
    request->SetTimeouts(timeouts_->connect, timeouts_->idle,
                         timeouts_->metadata);
    request->AddHeader("Content-Range", strings::StrCat("bytes */", file_size));
    request->SetPutEmptyBody();
    const Status& status = request->Send();
    if (status.ok()) {
      *completed = true;
      return Status::OK();
    }
    *completed = false;
    if (request->GetResponseCode() != HTTP_CODE_RESUME_INCOMPLETE) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(status, " when resuming upload ",
                                      GetGcsPath());
    }
    const string& received_range = request->GetResponseHeader("Range");
    if (received_range.empty()) {
      // This means GCS doesn't have any bytes of the file yet.
      *uploaded = 0;
    } else {
      StringPiece range_piece(received_range);
      absl::ConsumePrefix(&range_piece,
                          "bytes=");  // May or may not be present.

      auto return_error = [this](string error_message) {
        return errors::Internal("Unexpected response from GCS when writing ",
                                GetGcsPath(), ": ", error_message);
      };

      std::vector<string> range_strs = str_util::Split(range_piece, '-');
      std::vector<int64> range_parts;
      for (const string& range_str : range_strs) {
        int64 tmp;
        if (strings::safe_strto64(range_str, &tmp)) {
          range_parts.push_back(tmp);
        } else {
          return return_error("Range header '" + received_range +
                              "' could not be parsed.");
        }
      }
      if (range_parts.size() != 2) {
        return return_error("Range header '" + received_range +
                            "' could not be parsed.");
      }

      if (range_parts[0] != 0) {
        return return_error("The returned range '" + received_range +
                            "' does not start at zero.");
      }
      // If GCS returned "Range: 0-10", this means 11 bytes were uploaded.
      *uploaded = range_parts[1] + 1;
    }
    return Status::OK();
  }

  Status UploadToSession(const string& session_uri, uint64 start_offset) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(filesystem_->CreateHttpRequest(&request));
    request->SetUri(session_uri);
    if (file_size > 0) {
      request->AddHeader("Content-Range",
                         strings::StrCat("bytes ", start_offset, "-",
                                         file_size - 1, "/", file_size));
    }
    request->SetTimeouts(timeouts_->connect, timeouts_->idle, timeouts_->write);

    TF_RETURN_IF_ERROR(
        request->SetPutFromFile(tmp_content_filename_, start_offset));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when uploading ",
                                    GetGcsPath());
    // Erase the file from the file cache on every successful write.
    file_cache_erase_();
    return Status::OK();
  }

  string GetGcsPath() const {
    return strings::StrCat("gs://", bucket_, "/", object_);
  }

  string bucket_;
  string object_;
  GcsFileSystem* const filesystem_;  // Not owned.
  string tmp_content_filename_;
  std::ofstream outfile_;
  GcsFileSystem::TimeoutConfig* timeouts_;
  std::function<void()> file_cache_erase_;
  bool sync_needed_;  // whether there is buffered data that needs to be synced
  RetryConfig retry_config_;
};

class GcsReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  GcsReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

bool StringPieceIdentity(StringPiece str, StringPiece* value) {
  *value = str;
  return true;
}

/// \brief Utility function to split a comma delimited list of strings to an
/// unordered set, lowercasing all values.
bool SplitByCommaToLowercaseSet(StringPiece list,
                                std::unordered_set<string>* set) {
  std::vector<string> vector = absl::StrSplit(absl::AsciiStrToLower(list), ',');
  *set = std::unordered_set<string>(vector.begin(), vector.end());
  return true;
}

// \brief Convert Compute Engine zone to region
string ZoneToRegion(string* zone) {
  return zone->substr(0, zone->find_last_of('-'));
}

}  // namespace

GcsFileSystem::GcsFileSystem(bool make_default_cache) {
  uint64 value;
  block_size_ = kDefaultBlockSize;
  size_t max_bytes = kDefaultMaxCacheSize;
  uint64 max_staleness = kDefaultMaxStaleness;

  http_request_factory_ = std::make_shared<CurlHttpRequest::Factory>();
  compute_engine_metadata_client_ =
      std::make_shared<ComputeEngineMetadataClient>(http_request_factory_);
  auth_provider_ = std::unique_ptr<AuthProvider>(
      new GoogleAuthProvider(compute_engine_metadata_client_));
  zone_provider_ = std::unique_ptr<ZoneProvider>(
      new ComputeEngineZoneProvider(compute_engine_metadata_client_));

  // Apply the sys env override for the readahead buffer size if it's provided.
  if (GetEnvVar(kReadaheadBufferSize, strings::safe_strtou64, &value)) {
    block_size_ = value;
  }

  // Apply the overrides for the block size (MB), max bytes (MB), and max
  // staleness (seconds) if provided.
  if (GetEnvVar(kBlockSize, strings::safe_strtou64, &value)) {
    block_size_ = value * 1024 * 1024;
  }

  if (GetEnvVar(kMaxCacheSize, strings::safe_strtou64, &value)) {
    max_bytes = value * 1024 * 1024;
  }

  if (GetEnvVar(kMaxStaleness, strings::safe_strtou64, &value)) {
    max_staleness = value;
  }
  if (!make_default_cache) {
    max_bytes = 0;
  }
  VLOG(1) << "GCS cache max size = " << max_bytes << " ; "
          << "block size = " << block_size_ << " ; "
          << "max staleness = " << max_staleness;
  file_block_cache_ = MakeFileBlockCache(block_size_, max_bytes, max_staleness);
  // Apply overrides for the stat cache max age and max entries, if provided.
  uint64 stat_cache_max_age = kStatCacheDefaultMaxAge;
  size_t stat_cache_max_entries = kStatCacheDefaultMaxEntries;
  if (GetEnvVar(kStatCacheMaxAge, strings::safe_strtou64, &value)) {
    stat_cache_max_age = value;
  }
  if (GetEnvVar(kStatCacheMaxEntries, strings::safe_strtou64, &value)) {
    stat_cache_max_entries = value;
  }
  stat_cache_.reset(new ExpiringLRUCache<GcsFileStat>(stat_cache_max_age,
                                                      stat_cache_max_entries));
  // Apply overrides for the matching paths cache max age and max entries, if
  // provided.
  uint64 matching_paths_cache_max_age = kMatchingPathsCacheDefaultMaxAge;
  size_t matching_paths_cache_max_entries =
      kMatchingPathsCacheDefaultMaxEntries;
  if (GetEnvVar(kMatchingPathsCacheMaxAge, strings::safe_strtou64, &value)) {
    matching_paths_cache_max_age = value;
  }
  if (GetEnvVar(kMatchingPathsCacheMaxEntries, strings::safe_strtou64,
                &value)) {
    matching_paths_cache_max_entries = value;
  }
  matching_paths_cache_.reset(new ExpiringLRUCache<std::vector<string>>(
      matching_paths_cache_max_age, matching_paths_cache_max_entries));

  bucket_location_cache_.reset(new ExpiringLRUCache<string>(
      kCacheNeverExpire, kBucketLocationCacheMaxEntries));

  int64 resolve_frequency_secs;
  if (GetEnvVar(kResolveCacheSecs, strings::safe_strto64,
                &resolve_frequency_secs)) {
    dns_cache_.reset(new GcsDnsCache(resolve_frequency_secs));
    VLOG(1) << "GCS DNS cache is enabled.  " << kResolveCacheSecs << " = "
            << resolve_frequency_secs;
  } else {
    VLOG(1) << "GCS DNS cache is disabled, because " << kResolveCacheSecs
            << " = 0 (or is not set)";
  }

  // Get the additional header
  StringPiece add_header_contents;
  if (GetEnvVar(kAdditionalRequestHeader, StringPieceIdentity,
                &add_header_contents)) {
    size_t split = add_header_contents.find(':', 0);

    if (split != StringPiece::npos) {
      StringPiece header_name = add_header_contents.substr(0, split);
      StringPiece header_value = add_header_contents.substr(split + 1);

      if (!header_name.empty() && !header_value.empty()) {
        additional_header_.reset(new std::pair<const string, const string>(
            string(header_name), string(header_value)));

        VLOG(1) << "GCS additional header ENABLED. "
                << "Name: " << additional_header_->first << ", "
                << "Value: " << additional_header_->second;
      } else {
        LOG(ERROR) << "GCS additional header DISABLED. Invalid contents: "
                   << add_header_contents;
      }
    } else {
      LOG(ERROR) << "GCS additional header DISABLED. Invalid contents: "
                 << add_header_contents;
    }
  } else {
    VLOG(1) << "GCS additional header DISABLED. No environment variable set.";
  }

  // Apply the overrides for request timeouts
  uint32 timeout_value;
  if (GetEnvVar(kRequestConnectionTimeout, strings::safe_strtou32,
                &timeout_value)) {
    timeouts_.connect = timeout_value;
  }
  if (GetEnvVar(kRequestIdleTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.idle = timeout_value;
  }
  if (GetEnvVar(kMetadataRequestTimeout, strings::safe_strtou32,
                &timeout_value)) {
    timeouts_.metadata = timeout_value;
  }
  if (GetEnvVar(kReadRequestTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.read = timeout_value;
  }
  if (GetEnvVar(kWriteRequestTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.write = timeout_value;
  }

  int64 token_value;
  if (GetEnvVar(kThrottleRate, strings::safe_strto64, &token_value)) {
    GcsThrottleConfig config;
    config.enabled = true;
    config.token_rate = token_value;

    if (GetEnvVar(kThrottleBucket, strings::safe_strto64, &token_value)) {
      config.bucket_size = token_value;
    }

    if (GetEnvVar(kTokensPerRequest, strings::safe_strto64, &token_value)) {
      config.tokens_per_request = token_value;
    }

    if (GetEnvVar(kInitialTokens, strings::safe_strto64, &token_value)) {
      config.initial_tokens = token_value;
    }
    throttle_.SetConfig(config);
  }

  GetEnvVar(kAllowedBucketLocations, SplitByCommaToLowercaseSet,
            &allowed_locations_);
}

GcsFileSystem::GcsFileSystem(
    std::unique_ptr<AuthProvider> auth_provider,
    std::unique_ptr<HttpRequest::Factory> http_request_factory,
    std::unique_ptr<ZoneProvider> zone_provider, size_t block_size,
    size_t max_bytes, uint64 max_staleness, uint64 stat_cache_max_age,
    size_t stat_cache_max_entries, uint64 matching_paths_cache_max_age,
    size_t matching_paths_cache_max_entries, RetryConfig retry_config,
    TimeoutConfig timeouts, const std::unordered_set<string>& allowed_locations,
    std::pair<const string, const string>* additional_header)
    : auth_provider_(std::move(auth_provider)),
      http_request_factory_(std::move(http_request_factory)),
      zone_provider_(std::move(zone_provider)),
      block_size_(block_size),
      file_block_cache_(
          MakeFileBlockCache(block_size, max_bytes, max_staleness)),
      stat_cache_(new StatCache(stat_cache_max_age, stat_cache_max_entries)),
      matching_paths_cache_(new MatchingPathsCache(
          matching_paths_cache_max_age, matching_paths_cache_max_entries)),
      bucket_location_cache_(new BucketLocationCache(
          kCacheNeverExpire, kBucketLocationCacheMaxEntries)),
      allowed_locations_(allowed_locations),
      timeouts_(timeouts),
      retry_config_(retry_config),
      additional_header_(additional_header) {}

Status GcsFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  TF_RETURN_IF_ERROR(CheckBucketLocationConstraint(bucket));
  bool cache_enabled;
  {
    mutex_lock l(block_cache_lock_);
    cache_enabled = file_block_cache_->IsCacheEnabled();
  }
  if (cache_enabled) {
    result->reset(new GcsRandomAccessFile(fname, [this, bucket, object](
                                                     const string& fname,
                                                     uint64 offset, size_t n,
                                                     StringPiece* result,
                                                     char* scratch) {
      tf_shared_lock l(block_cache_lock_);
      GcsFileStat stat;
      TF_RETURN_IF_ERROR(stat_cache_->LookupOrCompute(
          fname, &stat,
          [this, bucket, object](const string& fname, GcsFileStat* stat) {
            return UncachedStatForObject(fname, bucket, object, stat);
          }));
      if (!file_block_cache_->ValidateAndUpdateFileSignature(
              fname, stat.generation_number)) {
        VLOG(1)
            << "File signature has been changed. Refreshing the cache. Path: "
            << fname;
      }
      *result = StringPiece();
      size_t bytes_transferred;
      TF_RETURN_IF_ERROR(file_block_cache_->Read(fname, offset, n, scratch,
                                                 &bytes_transferred));
      *result = StringPiece(scratch, bytes_transferred);
      if (bytes_transferred < n) {
        return errors::OutOfRange("EOF reached, ", result->size(),
                                  " bytes were read out of ", n,
                                  " bytes requested.");
      }
      return Status::OK();
    }));
  } else {
    result->reset(new BufferedGcsRandomAccessFile(
        fname, block_size_,
        [this, bucket, object](const string& fname, uint64 offset, size_t n,
                               StringPiece* result, char* scratch) {
          *result = StringPiece();
          size_t bytes_transferred;
          TF_RETURN_IF_ERROR(
              LoadBufferFromGCS(fname, offset, n, scratch, &bytes_transferred));
          *result = StringPiece(scratch, bytes_transferred);
          if (bytes_transferred < n) {
            return errors::OutOfRange("EOF reached, ", result->size(),
                                      " bytes were read out of ", n,
                                      " bytes requested.");
          }
          return Status::OK();
        }));
  }
  return Status::OK();
}

void GcsFileSystem::ResetFileBlockCache(size_t block_size_bytes,
                                        size_t max_bytes,
                                        uint64 max_staleness_secs) {
  mutex_lock l(block_cache_lock_);
  file_block_cache_ =
      MakeFileBlockCache(block_size_bytes, max_bytes, max_staleness_secs);
  if (stats_ != nullptr) {
    stats_->Configure(this, &throttle_, file_block_cache_.get());
  }
}

// A helper function to build a FileBlockCache for GcsFileSystem.
std::unique_ptr<FileBlockCache> GcsFileSystem::MakeFileBlockCache(
    size_t block_size, size_t max_bytes, uint64 max_staleness) {
  std::unique_ptr<FileBlockCache> file_block_cache(new RamFileBlockCache(
      block_size, max_bytes, max_staleness,
      [this](const string& filename, size_t offset, size_t n, char* buffer,
             size_t* bytes_transferred) {
        return LoadBufferFromGCS(filename, offset, n, buffer,
                                 bytes_transferred);
      }));
  return file_block_cache;
}

// A helper function to actually read the data from GCS.
Status GcsFileSystem::LoadBufferFromGCS(const string& fname, size_t offset,
                                        size_t n, char* buffer,
                                        size_t* bytes_transferred) {
  *bytes_transferred = 0;

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(CreateHttpRequest(&request),
                                  "when reading gs://", bucket, "/", object);

  request->SetUri(strings::StrCat("https://", kStorageHost, "/", bucket, "/",
                                  request->EscapeString(object)));
  request->SetRange(offset, offset + n - 1);
  request->SetResultBufferDirect(buffer, n);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.read);

  if (stats_ != nullptr) {
    stats_->RecordBlockLoadRequest(fname, offset);
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading gs://",
                                  bucket, "/", object);

  size_t bytes_read = request->GetResultBufferDirectBytesTransferred();
  *bytes_transferred = bytes_read;
  VLOG(1) << "Successful read of gs://" << bucket << "/" << object << " @ "
          << offset << " of size: " << bytes_read;

  if (stats_ != nullptr) {
    stats_->RecordBlockRetrieved(fname, offset, bytes_read);
  }

  throttle_.RecordResponse(bytes_read);

  if (bytes_read < n) {
    // Check stat cache to see if we encountered an interrupted read.
    GcsFileStat stat;
    if (stat_cache_->Lookup(fname, &stat)) {
      if (offset + bytes_read < stat.base.length) {
        return errors::Internal(strings::Printf(
            "File contents are inconsistent for file: %s @ %lu.", fname.c_str(),
            offset));
      }
      VLOG(2) << "Successful integrity check for: gs://" << bucket << "/"
              << object << " @ " << offset;
    }
  }

  return Status::OK();
}

void GcsFileSystem::ClearFileCaches(const string& fname) {
  tf_shared_lock l(block_cache_lock_);
  file_block_cache_->RemoveFile(fname);
  stat_cache_->Delete(fname);
  // TODO(rxsang): Remove the patterns that matche the file in
  // MatchingPathsCache as well.
}

Status GcsFileSystem::NewWritableFile(const string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  result->reset(new GcsWritableFile(bucket, object, this, &timeouts_,
                                    [this, fname]() { ClearFileCaches(fname); },
                                    retry_config_));
  return Status::OK();
}

// Reads the file from GCS in chunks and stores it in a tmp file,
// which is then passed to GcsWritableFile.
Status GcsFileSystem::NewAppendableFile(const string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<RandomAccessFile> reader;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &reader));
  std::unique_ptr<char[]> buffer(new char[kReadAppendableFileBufferSize]);
  Status status;
  uint64 offset = 0;
  StringPiece read_chunk;

  // Read the file from GCS in chunks and save it to a tmp file.
  string old_content_filename;
  TF_RETURN_IF_ERROR(GetTmpFilename(&old_content_filename));
  std::ofstream old_content(old_content_filename, std::ofstream::binary);
  while (true) {
    status = reader->Read(offset, kReadAppendableFileBufferSize, &read_chunk,
                          buffer.get());
    if (status.ok()) {
      old_content << read_chunk;
      offset += kReadAppendableFileBufferSize;
    } else if (status.code() == error::OUT_OF_RANGE) {
      // Expected, this means we reached EOF.
      old_content << read_chunk;
      break;
    } else {
      return status;
    }
  }
  old_content.close();

  // Create a writable file and pass the old content to it.
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  result->reset(new GcsWritableFile(
      bucket, object, this, old_content_filename, &timeouts_,
      [this, fname]() { ClearFileCaches(fname); }, retry_config_));
  return Status::OK();
}

Status GcsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(fname, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new GcsReadOnlyMemoryRegion(std::move(data), size));
  return Status::OK();
}

Status GcsFileSystem::FileExists(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool result;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &result));
    if (result) {
      return Status::OK();
    }
  }

  // Check if the object exists.
  GcsFileStat stat;
  const Status status = StatForObject(fname, bucket, object, &stat);
  if (status.code() != errors::Code::NOT_FOUND) {
    return status;
  }

  // Check if the folder exists.
  bool result;
  TF_RETURN_IF_ERROR(FolderExists(fname, &result));
  if (result) {
    return Status::OK();
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}

Status GcsFileSystem::ObjectExists(const string& fname, const string& bucket,
                                   const string& object, bool* result) {
  GcsFileStat stat;
  const Status status = StatForObject(fname, bucket, object, &stat);
  switch (status.code()) {
    case errors::Code::OK:
      *result = !stat.base.is_directory;
      return Status::OK();
    case errors::Code::NOT_FOUND:
      *result = false;
      return Status::OK();
    default:
      return status;
  }
}

Status GcsFileSystem::UncachedStatForObject(const string& fname,
                                            const string& bucket,
                                            const string& object,
                                            GcsFileStat* stat) {
  std::vector<char> output_buffer;
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(CreateHttpRequest(&request),
                                  " when reading metadata of gs://", bucket,
                                  "/", object);

  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                  request->EscapeString(object),
                                  "?fields=size%2Cgeneration%2Cupdated"));
  request->SetResultBuffer(&output_buffer);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);

  if (stats_ != nullptr) {
    stats_->RecordStatObjectRequest();
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      request->Send(), " when reading metadata of gs://", bucket, "/", object);

  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));

  // Parse file size.
  TF_RETURN_IF_ERROR(GetInt64Value(root, "size", &stat->base.length));

  // Parse generation number.
  TF_RETURN_IF_ERROR(
      GetInt64Value(root, "generation", &stat->generation_number));

  // Parse file modification time.
  string updated;
  TF_RETURN_IF_ERROR(GetStringValue(root, "updated", &updated));
  TF_RETURN_IF_ERROR(ParseRfc3339Time(updated, &(stat->base.mtime_nsec)));

  VLOG(1) << "Stat of: gs://" << bucket << "/" << object << " -- "
          << " length: " << stat->base.length
          << " generation: " << stat->generation_number
          << "; mtime_nsec: " << stat->base.mtime_nsec
          << "; updated: " << updated;

  if (str_util::EndsWith(fname, "/")) {
    // In GCS a path can be both a directory and a file, both it is uncommon for
    // other file systems. To avoid the ambiguity, if a path ends with "/" in
    // GCS, we always regard it as a directory mark or a virtual directory.
    stat->base.is_directory = true;
  } else {
    stat->base.is_directory = false;
  }
  return Status::OK();
}

Status GcsFileSystem::StatForObject(const string& fname, const string& bucket,
                                    const string& object, GcsFileStat* stat) {
  if (object.empty()) {
    return errors::InvalidArgument(strings::Printf(
        "'object' must be a non-empty string. (File: %s)", fname.c_str()));
  }

  TF_RETURN_IF_ERROR(stat_cache_->LookupOrCompute(
      fname, stat,
      [this, &bucket, &object](const string& fname, GcsFileStat* stat) {
        return UncachedStatForObject(fname, bucket, object, stat);
      }));
  return Status::OK();
}

Status GcsFileSystem::BucketExists(const string& bucket, bool* result) {
  const Status status = GetBucketMetadata(bucket, nullptr);
  switch (status.code()) {
    case errors::Code::OK:
      *result = true;
      return Status::OK();
    case errors::Code::NOT_FOUND:
      *result = false;
      return Status::OK();
    default:
      return status;
  }
}

Status GcsFileSystem::CheckBucketLocationConstraint(const string& bucket) {
  if (allowed_locations_.empty()) {
    return Status::OK();
  }

  // Avoid calling external API's in the constructor
  if (allowed_locations_.erase(kDetectZoneSentinalValue) == 1) {
    string zone;
    TF_RETURN_IF_ERROR(zone_provider_->GetZone(&zone));
    allowed_locations_.insert(ZoneToRegion(&zone));
  }

  string location;
  TF_RETURN_IF_ERROR(GetBucketLocation(bucket, &location));
  if (allowed_locations_.find(location) != allowed_locations_.end()) {
    return Status::OK();
  }

  return errors::FailedPrecondition(strings::Printf(
      "Bucket '%s' is in '%s' location, allowed locations are: (%s).",
      bucket.c_str(), location.c_str(),
      absl::StrJoin(allowed_locations_, ", ").c_str()));
}

Status GcsFileSystem::GetBucketLocation(const string& bucket,
                                        string* location) {
  auto compute_func = [this](const string& bucket, string* location) {
    std::vector<char> result_buffer;
    Status status = GetBucketMetadata(bucket, &result_buffer);
    Json::Value result;
    TF_RETURN_IF_ERROR(ParseJson(result_buffer, &result));
    string bucket_location;
    TF_RETURN_IF_ERROR(
        GetStringValue(result, kBucketMetadataLocationKey, &bucket_location));
    // Lowercase the GCS location to be case insensitive for allowed locations.
    *location = absl::AsciiStrToLower(bucket_location);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(
      bucket_location_cache_->LookupOrCompute(bucket, location, compute_func));

  return Status::OK();
}

Status GcsFileSystem::GetBucketMetadata(const string& bucket,
                                        std::vector<char>* result_buffer) {
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket));

  if (result_buffer != nullptr) {
    request->SetResultBuffer(result_buffer);
  }

  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  return request->Send();
}

Status GcsFileSystem::FolderExists(const string& dirname, bool* result) {
  StatCache::ComputeFunc compute_func = [this](const string& dirname,
                                               GcsFileStat* stat) {
    std::vector<string> children;
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(dirname, 1, &children, true /* recursively */,
                           true /* include_self_directory_marker */));
    if (!children.empty()) {
      stat->base = DIRECTORY_STAT;
      return Status::OK();
    } else {
      return errors::InvalidArgument("Not a directory!");
    }
  };
  GcsFileStat stat;
  Status s = stat_cache_->LookupOrCompute(MaybeAppendSlash(dirname), &stat,
                                          compute_func);
  if (s.ok()) {
    *result = stat.base.is_directory;
    return Status::OK();
  }
  if (errors::IsInvalidArgument(s)) {
    *result = false;
    return Status::OK();
  }
  return s;
}

Status GcsFileSystem::GetChildren(const string& dirname,
                                  std::vector<string>* result) {
  return GetChildrenBounded(dirname, UINT64_MAX, result,
                            false /* recursively */,
                            false /* include_self_directory_marker */);
}

Status GcsFileSystem::GetMatchingPaths(const string& pattern,
                                       std::vector<string>* results) {
  MatchingPathsCache::ComputeFunc compute_func =
      [this](const string& pattern, std::vector<string>* results) {
        results->clear();
        // Find the fixed prefix by looking for the first wildcard.
        const string& fixed_prefix =
            pattern.substr(0, pattern.find_first_of("*?[\\"));
        const string dir(io::Dirname(fixed_prefix));
        if (dir.empty()) {
          return errors::InvalidArgument(
              "A GCS pattern doesn't have a bucket name: ", pattern);
        }
        std::vector<string> all_files;
        TF_RETURN_IF_ERROR(GetChildrenBounded(
            dir, UINT64_MAX, &all_files, true /* recursively */,
            false /* include_self_directory_marker */));

        const auto& files_and_folders = AddAllSubpaths(all_files);

        // Match all obtained paths to the input pattern.
        for (const auto& path : files_and_folders) {
          const string& full_path = io::JoinPath(dir, path);
          if (Env::Default()->MatchPath(full_path, pattern)) {
            results->push_back(full_path);
          }
        }
        return Status::OK();
      };
  TF_RETURN_IF_ERROR(
      matching_paths_cache_->LookupOrCompute(pattern, results, compute_func));
  return Status::OK();
}

Status GcsFileSystem::GetChildrenBounded(const string& dirname,
                                         uint64 max_results,
                                         std::vector<string>* result,
                                         bool recursive,
                                         bool include_self_directory_marker) {
  if (!result) {
    return errors::InvalidArgument("'result' cannot be null");
  }
  string bucket, object_prefix;
  TF_RETURN_IF_ERROR(
      ParseGcsPath(MaybeAppendSlash(dirname), true, &bucket, &object_prefix));

  string nextPageToken;
  uint64 retrieved_results = 0;
  while (true) {  // A loop over multiple result pages.
    std::vector<char> output_buffer;
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
    auto uri = strings::StrCat(kGcsUriBase, "b/", bucket, "/o");
    if (recursive) {
      uri = strings::StrCat(uri, "?fields=items%2Fname%2CnextPageToken");
    } else {
      // Set "/" as a delimiter to ask GCS to treat subfolders as children
      // and return them in "prefixes".
      uri = strings::StrCat(uri,
                            "?fields=items%2Fname%2Cprefixes%2CnextPageToken");
      uri = strings::StrCat(uri, "&delimiter=%2F");
    }
    if (!object_prefix.empty()) {
      uri = strings::StrCat(uri,
                            "&prefix=", request->EscapeString(object_prefix));
    }
    if (!nextPageToken.empty()) {
      uri = strings::StrCat(
          uri, "&pageToken=", request->EscapeString(nextPageToken));
    }
    if (max_results - retrieved_results < kGetChildrenDefaultPageSize) {
      uri =
          strings::StrCat(uri, "&maxResults=", max_results - retrieved_results);
    }
    request->SetUri(uri);
    request->SetResultBuffer(&output_buffer);
    request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading ", dirname);
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
    const auto items = root.get("items", Json::Value::null);
    if (!items.isNull()) {
      if (!items.isArray()) {
        return errors::Internal(
            "Expected an array 'items' in the GCS response.");
      }
      for (size_t i = 0; i < items.size(); i++) {
        const auto item = items.get(i, Json::Value::null);
        if (!item.isObject()) {
          return errors::Internal(
              "Unexpected JSON format: 'items' should be a list of objects.");
        }
        string name;
        TF_RETURN_IF_ERROR(GetStringValue(item, "name", &name));
        // The names should be relative to the 'dirname'. That means the
        // 'object_prefix', which is part of 'dirname', should be removed from
        // the beginning of 'name'.
        StringPiece relative_path(name);
        if (!absl::ConsumePrefix(&relative_path, object_prefix)) {
          return errors::Internal(strings::StrCat(
              "Unexpected response: the returned file name ", name,
              " doesn't match the prefix ", object_prefix));
        }
        if (!relative_path.empty() || include_self_directory_marker) {
          result->emplace_back(relative_path);
        }
        if (++retrieved_results >= max_results) {
          return Status::OK();
        }
      }
    }
    const auto prefixes = root.get("prefixes", Json::Value::null);
    if (!prefixes.isNull()) {
      // Subfolders are returned for the non-recursive mode.
      if (!prefixes.isArray()) {
        return errors::Internal(
            "'prefixes' was expected to be an array in the GCS response.");
      }
      for (size_t i = 0; i < prefixes.size(); i++) {
        const auto prefix = prefixes.get(i, Json::Value::null);
        if (prefix.isNull() || !prefix.isString()) {
          return errors::Internal(
              "'prefixes' was expected to be an array of strings in the GCS "
              "response.");
        }
        const string& prefix_str = prefix.asString();
        StringPiece relative_path(prefix_str);
        if (!absl::ConsumePrefix(&relative_path, object_prefix)) {
          return errors::Internal(
              "Unexpected response: the returned folder name ", prefix_str,
              " doesn't match the prefix ", object_prefix);
        }
        result->emplace_back(relative_path);
        if (++retrieved_results >= max_results) {
          return Status::OK();
        }
      }
    }
    const auto token = root.get("nextPageToken", Json::Value::null);
    if (token.isNull()) {
      return Status::OK();
    }
    if (!token.isString()) {
      return errors::Internal(
          "Unexpected response: nextPageToken is not a string");
    }
    nextPageToken = token.asString();
  }
}

Status GcsFileSystem::Stat(const string& fname, FileStatistics* stat) {
  if (!stat) {
    return errors::Internal("'stat' cannot be nullptr.");
  }
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    if (is_bucket) {
      *stat = DIRECTORY_STAT;
      return Status::OK();
    }
    return errors::NotFound("The specified bucket ", fname, " was not found.");
  }

  GcsFileStat gcs_stat;
  const Status status = StatForObject(fname, bucket, object, &gcs_stat);
  if (status.ok()) {
    *stat = gcs_stat.base;
    return Status::OK();
  }
  if (status.code() != errors::Code::NOT_FOUND) {
    return status;
  }
  bool is_folder;
  TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
  if (is_folder) {
    *stat = DIRECTORY_STAT;
    return Status::OK();
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}

Status GcsFileSystem::DeleteFile(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                  request->EscapeString(object)));
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  request->SetDeleteRequest();

  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when deleting ", fname);
  ClearFileCaches(fname);
  return Status::OK();
}

Status GcsFileSystem::CreateDir(const string& dirname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(dirname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    return is_bucket ? Status::OK()
                     : errors::NotFound("The specified bucket ", dirname,
                                        " was not found.");
  }

  const string dirname_with_slash = MaybeAppendSlash(dirname);

  if (FileExists(dirname_with_slash).ok()) {
    return errors::AlreadyExists(dirname);
  }

  // Create a zero-length directory marker object.
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(NewWritableFile(dirname_with_slash, &file));
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

// Checks that the directory is empty (i.e no objects with this prefix exist).
// Deletes the GCS directory marker if it exists.
Status GcsFileSystem::DeleteDir(const string& dirname) {
  std::vector<string> children;
  // A directory is considered empty either if there are no matching objects
  // with the corresponding name prefix or if there is exactly one matching
  // object and it is the directory marker. Therefore we need to retrieve
  // at most two children for the prefix to detect if a directory is empty.
  TF_RETURN_IF_ERROR(
      GetChildrenBounded(dirname, 2, &children, true /* recursively */,
                         true /* include_self_directory_marker */));

  if (children.size() > 1 || (children.size() == 1 && !children[0].empty())) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }
  if (children.size() == 1 && children[0].empty()) {
    // This is the directory marker object. Delete it.
    return DeleteFile(MaybeAppendSlash(dirname));
  }
  return Status::OK();
}

Status GcsFileSystem::GetFileSize(const string& fname, uint64* file_size) {
  if (!file_size) {
    return errors::Internal("'file_size' cannot be nullptr.");
  }

  // Only validate the name.
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, &stat));
  *file_size = stat.length;
  return Status::OK();
}

Status GcsFileSystem::RenameFile(const string& src, const string& target) {
  if (!IsDirectory(src).ok()) {
    return RenameObject(src, target);
  }
  // Rename all individual objects in the directory one by one.
  std::vector<string> children;
  TF_RETURN_IF_ERROR(
      GetChildrenBounded(src, UINT64_MAX, &children, true /* recursively */,
                         true /* include_self_directory_marker */));
  for (const string& subpath : children) {
    TF_RETURN_IF_ERROR(
        RenameObject(JoinGcsPath(src, subpath), JoinGcsPath(target, subpath)));
  }
  return Status::OK();
}

// Uses a GCS API command to copy the object and then deletes the old one.
Status GcsFileSystem::RenameObject(const string& src, const string& target) {
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseGcsPath(src, false, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(
      ParseGcsPath(target, false, &target_bucket, &target_object));

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", src_bucket, "/o/",
                                  request->EscapeString(src_object),
                                  "/rewriteTo/b/", target_bucket, "/o/",
                                  request->EscapeString(target_object)));
  request->SetPostEmptyBody();
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  std::vector<char> output_buffer;
  request->SetResultBuffer(&output_buffer);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when renaming ", src,
                                  " to ", target);
  // Flush the target from the caches.  The source will be flushed in the
  // DeleteFile call below.
  ClearFileCaches(target);
  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
  bool done;
  TF_RETURN_IF_ERROR(GetBoolValue(root, "done", &done));
  if (!done) {
    // If GCS didn't complete rewrite in one call, this means that a large file
    // is being copied to a bucket with a different storage class or location,
    // which requires multiple rewrite calls.
    // TODO(surkov): implement multi-step rewrites.
    return errors::Unimplemented(
        "Couldn't rename ", src, " to ", target,
        ": moving large files between buckets with different "
        "locations or storage classes is not supported.");
  }

  // In case the delete API call failed, but the deletion actually happened
  // on the server side, we can't just retry the whole RenameFile operation
  // because the source object is already gone.
  return RetryingUtils::DeleteWithRetries(
      [this, &src]() { return DeleteFile(src); }, retry_config_);
}

Status GcsFileSystem::IsDirectory(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    if (is_bucket) {
      return Status::OK();
    }
    return errors::NotFound("The specified bucket gs://", bucket,
                            " was not found.");
  }
  bool is_folder;
  TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
  if (is_folder) {
    return Status::OK();
  }
  bool is_object;
  TF_RETURN_IF_ERROR(ObjectExists(fname, bucket, object, &is_object));
  if (is_object) {
    return errors::FailedPrecondition("The specified path ", fname,
                                      " is not a directory.");
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}

Status GcsFileSystem::DeleteRecursively(const string& dirname,
                                        int64* undeleted_files,
                                        int64* undeleted_dirs) {
  if (!undeleted_files || !undeleted_dirs) {
    return errors::Internal(
        "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
  }
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  if (!IsDirectory(dirname).ok()) {
    *undeleted_dirs = 1;
    return Status(
        error::NOT_FOUND,
        strings::StrCat(dirname, " doesn't exist or not a directory."));
  }
  std::vector<string> all_objects;
  // Get all children in the directory recursively.
  TF_RETURN_IF_ERROR(GetChildrenBounded(
      dirname, UINT64_MAX, &all_objects, true /* recursively */,
      true /* include_self_directory_marker */));
  for (const string& object : all_objects) {
    const string& full_path = JoinGcsPath(dirname, object);
    // Delete all objects including directory markers for subfolders.
    // Since DeleteRecursively returns OK if individual file deletions fail,
    // and therefore RetryingFileSystem won't pay attention to the failures,
    // we need to make sure these failures are properly retried.
    const auto& delete_file_status = RetryingUtils::DeleteWithRetries(
        [this, &full_path]() { return DeleteFile(full_path); }, retry_config_);
    if (!delete_file_status.ok()) {
      if (IsDirectory(full_path).ok()) {
        // The object is a directory marker.
        (*undeleted_dirs)++;
      } else {
        (*undeleted_files)++;
      }
    }
  }
  return Status::OK();
}

// Flushes all caches for filesystem metadata and file contents. Useful for
// reclaiming memory once filesystem operations are done (e.g. model is loaded),
// or for resetting the filesystem to a consistent state.
void GcsFileSystem::FlushCaches() {
  tf_shared_lock l(block_cache_lock_);
  file_block_cache_->Flush();
  stat_cache_->Clear();
  matching_paths_cache_->Clear();
  bucket_location_cache_->Clear();
}

void GcsFileSystem::SetStats(GcsStatsInterface* stats) {
  CHECK(stats_ == nullptr) << "SetStats() has already been called.";
  CHECK(stats != nullptr);
  mutex_lock l(block_cache_lock_);
  stats_ = stats;
  stats_->Configure(this, &throttle_, file_block_cache_.get());
}

void GcsFileSystem::SetCacheStats(FileBlockCacheStatsInterface* cache_stats) {
  tf_shared_lock l(block_cache_lock_);
  if (file_block_cache_ == nullptr) {
    LOG(ERROR) << "Tried to set cache stats of non-initialized file block "
                  "cache object. This may result in not exporting the intended "
                  "monitoring data";
    return;
  }
  file_block_cache_->SetStats(cache_stats);
}

void GcsFileSystem::SetAuthProvider(
    std::unique_ptr<AuthProvider> auth_provider) {
  mutex_lock l(mu_);
  auth_provider_ = std::move(auth_provider);
}

// Creates an HttpRequest and sets several parameters that are common to all
// requests.  All code (in GcsFileSystem) that creates an HttpRequest should
// go through this method, rather than directly using http_request_factory_.
Status GcsFileSystem::CreateHttpRequest(std::unique_ptr<HttpRequest>* request) {
  std::unique_ptr<HttpRequest> new_request{http_request_factory_->Create()};
  if (dns_cache_) {
    dns_cache_->AnnotateRequest(new_request.get());
  }

  string auth_token;
  {
    tf_shared_lock l(mu_);
    TF_RETURN_IF_ERROR(
        AuthProvider::GetToken(auth_provider_.get(), &auth_token));
  }

  new_request->AddAuthBearerHeader(auth_token);

  if (additional_header_) {
    new_request->AddHeader(additional_header_->first,
                           additional_header_->second);
  }

  if (stats_ != nullptr) {
    new_request->SetRequestStats(stats_->HttpStats());
  }

  if (!throttle_.AdmitRequest()) {
    return errors::Unavailable("Request throttled");
  }

  *request = std::move(new_request);
  return Status::OK();
}

}  // namespace tensorflow

// The TPU_GCS_FS option sets a TPU-on-GCS optimized file system that allows
// TPU pods to function more optimally. When TPU_GCS_FS is enabled then
// gcs_file_system will not be registered as a file system since the
// tpu_gcs_file_system is going to take over its responsibilities. The tpu file
// system is a child of gcs file system with TPU-pod on GCS optimizations.
// This option is set ON/OFF in the GCP TPU tensorflow config.
// Initialize gcs_file_system
REGISTER_FILE_SYSTEM("gs", ::tensorflow::RetryingGcsFileSystem);
