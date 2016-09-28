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
#include "include/json/json.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/cloud/google_auth_provider.h"
#include "tensorflow/core/platform/cloud/time_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

namespace {

constexpr char kGcsUriBase[] = "https://www.googleapis.com/storage/v1/";
constexpr char kGcsUploadUriBase[] =
    "https://www.googleapis.com/upload/storage/v1/";
constexpr char kStorageHost[] = "storage.googleapis.com";
constexpr size_t kBufferSize = 1024 * 1024;  // In bytes.
constexpr int kGetChildrenDefaultPageSize = 1000;
// Initial delay before retrying a GCS upload.
// Subsequent delays can be larger due to exponential back-off.
constexpr uint64 kUploadRetryDelayMicros = 1000000L;
// The HTTP response code "308 Resume Incomplete".
constexpr uint64 HTTP_CODE_RESUME_INCOMPLETE = 308;

Status GetTmpFilename(string* filename) {
  if (!filename) {
    return errors::Internal("'filename' cannot be nullptr.");
  }
  char buffer[] = "/tmp/gcs_filesystem_XXXXXX";
  int fd = mkstemp(buffer);
  if (fd < 0) {
    return errors::Internal("Failed to create a temporary file.");
  }
  close(fd);
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
  if (!bucket || !object) {
    return errors::Internal("bucket and object cannot be null.");
  }
  StringPiece scheme, bucketp, objectp;
  ParseURI(fname, &scheme, &bucketp, &objectp);
  if (scheme != "gs") {
    return errors::InvalidArgument(
        strings::StrCat("GCS path doesn't start with 'gs://': ", fname));
  }
  *bucket = bucketp.ToString();
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument(
        strings::StrCat("GCS path doesn't contain a bucket name: ", fname));
  }
  objectp.Consume("/");
  *object = objectp.ToString();
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument(
        strings::StrCat("GCS path doesn't contain an object name: ", fname));
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

Status ParseJson(StringPiece json, Json::Value* result) {
  Json::Reader reader;
  if (!reader.parse(json.ToString(), *result)) {
    return errors::Internal("Couldn't parse JSON response from GCS.");
  }
  return Status::OK();
}

/// Reads a JSON value with the given name from a parent JSON value.
Status GetValue(const Json::Value& parent, const string& name,
                Json::Value* result) {
  *result = parent.get(name, Json::Value::null);
  if (*result == Json::Value::null) {
    return errors::Internal(strings::StrCat(
        "The field '", name, "' was expected in the JSON response."));
  }
  return Status::OK();
}

/// Reads a string JSON value with the given name from a parent JSON value.
Status GetStringValue(const Json::Value& parent, const string& name,
                      string* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isString()) {
    return errors::Internal(
        strings::StrCat("The field '", name,
                        "' in the JSON response was expected to be a string."));
  }
  *result = result_value.asString();
  return Status::OK();
}

/// Reads a long JSON value with the given name from a parent JSON value.
Status GetInt64Value(const Json::Value& parent, const string& name,
                     int64* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (result_value.isNumeric()) {
    *result = result_value.asInt64();
    return Status::OK();
  }
  if (result_value.isString() &&
      strings::safe_strto64(result_value.asString().c_str(), result)) {
    return Status::OK();
  }
  return errors::Internal(
      strings::StrCat("The field '", name,
                      "' in the JSON response was expected to be a number."));
}

/// Reads a boolean JSON value with the given name from a parent JSON value.
Status GetBoolValue(const Json::Value& parent, const string& name,
                    bool* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isBool()) {
    return errors::Internal(strings::StrCat(
        "The field '", name,
        "' in the JSON response was expected to be a boolean."));
  }
  *result = result_value.asBool();
  return Status::OK();
}

/// A GCS-based implementation of a random access file with a read-ahead buffer.
class GcsRandomAccessFile : public RandomAccessFile {
 public:
  GcsRandomAccessFile(const string& bucket, const string& object,
                      AuthProvider* auth_provider,
                      HttpRequest::Factory* http_request_factory,
                      size_t read_ahead_bytes)
      : bucket_(bucket),
        object_(object),
        auth_provider_(auth_provider),
        http_request_factory_(http_request_factory),
        read_ahead_bytes_(read_ahead_bytes) {}

  /// The implementation of reads with a read-ahead buffer. Thread-safe.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    mutex_lock lock(mu_);
    const bool range_start_included = offset >= buffer_start_offset_;
    const bool range_end_included =
        offset + n <= buffer_start_offset_ + buffer_content_size_;
    if (range_start_included && (range_end_included || buffer_reached_eof_)) {
      // The requested range can be filled from the buffer.
      const size_t offset_in_buffer =
          std::min<uint64>(offset - buffer_start_offset_, buffer_content_size_);
      const auto copy_size =
          std::min(n, buffer_content_size_ - offset_in_buffer);
      std::memcpy(scratch, buffer_.get() + offset_in_buffer, copy_size);
      *result = StringPiece(scratch, copy_size);
    } else {
      // Update the buffer content based on the new requested range.
      const size_t desired_buffer_size = n + read_ahead_bytes_;
      if (n > buffer_size_ || desired_buffer_size > 2 * buffer_size_) {
        // Re-allocate only if buffer size increased significantly.
        buffer_.reset(new char[desired_buffer_size]);
        buffer_size_ = desired_buffer_size;
      }

      buffer_start_offset_ = offset;
      buffer_content_size_ = 0;
      StringPiece buffer_content;
      TF_RETURN_IF_ERROR(
          ReadFromGCS(offset, buffer_size_, &buffer_content, buffer_.get()));
      buffer_content_size_ = buffer_content.size();
      buffer_reached_eof_ = buffer_content_size_ < buffer_size_;

      // Set the results.
      *result = StringPiece(scratch, std::min(buffer_content_size_, n));
      std::memcpy(scratch, buffer_.get(), result->size());
    }

    if (result->size() < n) {
      // This is not an error per se. The RandomAccessFile interface expects
      // that Read returns OutOfRange if fewer bytes were read than requested.
      return errors::OutOfRange(strings::StrCat("EOF reached, ", result->size(),
                                                " bytes were read out of ", n,
                                                " bytes requested."));
    }
    return Status::OK();
  }

 private:
  /// A helper function to actually read the data from GCS.
  Status ReadFromGCS(uint64 offset, size_t n, StringPiece* result,
                     char* scratch) const {
    string auth_token;
    TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_, &auth_token));

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(
        request->SetUri(strings::StrCat("https://", bucket_, ".", kStorageHost,
                                        "/", request->EscapeString(object_))));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(request->SetRange(offset, offset + n - 1));
    TF_RETURN_IF_ERROR(request->SetResultBuffer(scratch, n, result));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading gs://",
                                    bucket_, "/", object_);
    return Status::OK();
  }

  string bucket_;
  string object_;
  AuthProvider* auth_provider_;
  HttpRequest::Factory* http_request_factory_;
  const size_t read_ahead_bytes_;

  // The buffer-related members need to be mutable, because they are modified
  // by the const Read() method.
  mutable mutex mu_;
  mutable std::unique_ptr<char[]> buffer_ GUARDED_BY(mu_);
  mutable size_t buffer_size_ GUARDED_BY(mu_) = 0;
  // The original file offset of the first byte in the buffer.
  mutable size_t buffer_start_offset_ GUARDED_BY(mu_) = 0;
  mutable size_t buffer_content_size_ GUARDED_BY(mu_) = 0;
  mutable bool buffer_reached_eof_ GUARDED_BY(mu_) = false;
};

/// \brief GCS-based implementation of a writeable file.
///
/// Since GCS objects are immutable, this implementation writes to a local
/// tmp file and copies it to GCS on flush/close.
class GcsWritableFile : public WritableFile {
 public:
  GcsWritableFile(const string& bucket, const string& object,
                  AuthProvider* auth_provider,
                  HttpRequest::Factory* http_request_factory,
                  int32 max_upload_attempts)
      : bucket_(bucket),
        object_(object),
        auth_provider_(auth_provider),
        http_request_factory_(http_request_factory),
        max_upload_attempts_(max_upload_attempts) {
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
                  AuthProvider* auth_provider,
                  const string& tmp_content_filename,
                  HttpRequest::Factory* http_request_factory,
                  int32 max_upload_attempts)
      : bucket_(bucket),
        object_(object),
        auth_provider_(auth_provider),
        http_request_factory_(http_request_factory),
        max_upload_attempts_(max_upload_attempts) {
    tmp_content_filename_ = tmp_content_filename;
    outfile_.open(tmp_content_filename_,
                  std::ofstream::binary | std::ofstream::app);
  }

  ~GcsWritableFile() { Close(); }

  Status Append(const StringPiece& data) override {
    TF_RETURN_IF_ERROR(CheckWritable());
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

  /// Copies the current version of the file to GCS.
  ///
  /// This Sync() uploads the object to GCS.
  /// In case of a failure, it resumes failed uploads as recommended by the GCS
  /// resumable API documentation. When the whole upload needs to be
  /// restarted, Sync() returns UNAVAILABLE and relies on RetryingFileSystem.
  Status Sync() override {
    TF_RETURN_IF_ERROR(CheckWritable());
    outfile_.flush();
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not write to the internal temporary file.");
    }
    string session_uri;
    TF_RETURN_IF_ERROR(CreateNewUploadSession(&session_uri));
    uint64 already_uploaded = 0;
    for (int attempt = 0; attempt < max_upload_attempts_; attempt++) {
      if (attempt > 0) {
        bool completed;
        TF_RETURN_IF_ERROR(RequestUploadSessionStatus(session_uri, &completed,
                                                      &already_uploaded));
        if (completed) {
          // It's unclear why UploadToSession didn't return OK in the previous
          // attempt, but GCS reports that the file is fully uploaded,
          // so succeed.
          return Status::OK();
        }
      }
      const Status upload_status =
          UploadToSession(session_uri, already_uploaded);
      if (upload_status.ok()) {
        return Status::OK();
      }
      switch (upload_status.code()) {
        case errors::Code::NOT_FOUND:
          // GCS docs recommend retrying the whole upload. We're relying on the
          // RetryingFileSystem to retry the Sync() call.
          return errors::Unavailable(
              strings::StrCat("Could not upload gs://", bucket_, "/", object_));
        case errors::Code::UNAVAILABLE:
          // The upload can be resumed, but GCS docs recommend an exponential
          // back-off.
          Env::Default()->SleepForMicroseconds(kUploadRetryDelayMicros
                                               << attempt);
          break;
        default:
          // Something unexpected happen, fail.
          return upload_status;
      }
    }
    return errors::Aborted(
        strings::StrCat("Upload gs://", bucket_, "/", object_, " failed."));
  }

 private:
  Status CheckWritable() const {
    if (!outfile_.is_open()) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    return Status::OK();
  }

  Status GetCurrentFileSize(uint64* size) {
    if (size == nullptr) {
      return errors::Internal("'size' cannot be nullptr");
    }
    const auto tellp = outfile_.tellp();
    if (tellp == -1) {
      return errors::Internal(
          "Could not get the size of the internal temporary file.");
    }
    *size = tellp;
    return Status::OK();
  }

  /// Initiates a new resumable upload session.
  Status CreateNewUploadSession(string* session_uri) {
    if (session_uri == nullptr) {
      return errors::Internal("'session_uri' cannot be nullptr.");
    }
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

    string auth_token;
    TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_, &auth_token));

    std::unique_ptr<char[]> scratch(new char[kBufferSize]);
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
        kGcsUploadUriBase, "b/", bucket_, "/o?uploadType=resumable&name=",
        request->EscapeString(object_))));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(request->AddHeader("X-Upload-Content-Length",
                                          std::to_string(file_size)));
    TF_RETURN_IF_ERROR(request->SetPostEmptyBody());
    StringPiece response_piece;
    TF_RETURN_IF_ERROR(
        request->SetResultBuffer(scratch.get(), kBufferSize, &response_piece));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        request->Send(), " when initiating an upload to ", GetGcsPath());
    *session_uri = request->GetResponseHeader("Location");
    if (session_uri->empty()) {
      return errors::Internal(
          strings::StrCat("Unexpected response from GCS when writing to ",
                          GetGcsPath(), ": 'Location' header not returned."));
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
    if (completed == nullptr || uploaded == nullptr) {
      return errors::Internal("'completed' and 'uploaded' cannot be nullptr.");
    }
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

    string auth_token;
    TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_, &auth_token));

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(request->SetUri(session_uri));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(request->AddHeader(
        "Content-Range", strings::StrCat("bytes */", file_size)));
    TF_RETURN_IF_ERROR(request->SetPutEmptyBody());
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
      std::vector<int32> range_parts;
      if (!str_util::SplitAndParseAsInts(received_range, '-', &range_parts) ||
          range_parts.size() != 2) {
        return errors::Internal(strings::StrCat(
            "Unexpected response from GCS when writing ", GetGcsPath(),
            ": Range header '", received_range, "' could not be parsed."));
      }
      if (range_parts[0] != 0) {
        return errors::Internal(
            strings::StrCat("Unexpected response from GCS when writing to ",
                            GetGcsPath(), ": the returned range '",
                            received_range, "' does not start at zero."));
      }
      // If GCS returned "Range: 0-10", this means 11 bytes were uploaded.
      *uploaded = range_parts[1] + 1;
    }
    return Status::OK();
  }

  Status UploadToSession(const string& session_uri, uint64 start_offset) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));

    string auth_token;
    TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_, &auth_token));

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(request->SetUri(session_uri));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    if (file_size > 0) {
      TF_RETURN_IF_ERROR(request->AddHeader(
          "Content-Range", strings::StrCat("bytes ", start_offset, "-",
                                           file_size - 1, "/", file_size)));
    }
    TF_RETURN_IF_ERROR(
        request->SetPutFromFile(tmp_content_filename_, start_offset));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when uploading ",
                                    GetGcsPath());
    return Status::OK();
  }

  string GetGcsPath() const {
    return strings::StrCat("gs://", bucket_, "/", object_);
  }

  string bucket_;
  string object_;
  AuthProvider* auth_provider_;
  string tmp_content_filename_;
  std::ofstream outfile_;
  HttpRequest::Factory* http_request_factory_;
  int32 max_upload_attempts_;
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
}  // namespace

GcsFileSystem::GcsFileSystem()
    : auth_provider_(new GoogleAuthProvider()),
      http_request_factory_(new HttpRequest::Factory()) {}

GcsFileSystem::GcsFileSystem(
    std::unique_ptr<AuthProvider> auth_provider,
    std::unique_ptr<HttpRequest::Factory> http_request_factory,
    size_t read_ahead_bytes, int32 max_upload_attempts)
    : auth_provider_(std::move(auth_provider)),
      http_request_factory_(std::move(http_request_factory)),
      read_ahead_bytes_(read_ahead_bytes),
      max_upload_attempts_(max_upload_attempts) {}

Status GcsFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  result->reset(new GcsRandomAccessFile(bucket, object, auth_provider_.get(),
                                        http_request_factory_.get(),
                                        read_ahead_bytes_));
  return Status::OK();
}

Status GcsFileSystem::NewWritableFile(const string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  result->reset(new GcsWritableFile(bucket, object, auth_provider_.get(),
                                    http_request_factory_.get(),
                                    max_upload_attempts_));
  return Status::OK();
}

// Reads the file from GCS in chunks and stores it in a tmp file,
// which is then passed to GcsWritableFile.
Status GcsFileSystem::NewAppendableFile(const string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<RandomAccessFile> reader;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &reader));
  std::unique_ptr<char[]> buffer(new char[kBufferSize]);
  Status status;
  uint64 offset = 0;
  StringPiece read_chunk;

  // Read the file from GCS in chunks and save it to a tmp file.
  string old_content_filename;
  TF_RETURN_IF_ERROR(GetTmpFilename(&old_content_filename));
  std::ofstream old_content(old_content_filename, std::ofstream::binary);
  while (true) {
    status = reader->Read(offset, kBufferSize, &read_chunk, buffer.get());
    if (status.ok()) {
      old_content << read_chunk;
      offset += kBufferSize;
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
      bucket, object, auth_provider_.get(), old_content_filename,
      http_request_factory_.get(), max_upload_attempts_));
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

bool GcsFileSystem::FileExists(const string& fname) {
  string bucket, object;
  if (!ParseGcsPath(fname, true, &bucket, &object).ok()) {
    LOG(ERROR) << "Could not parse GCS file name " << fname;
    return false;
  }
  if (object.empty()) {
    return BucketExists(bucket).ok();
  }
  return ObjectExists(bucket, object).ok() || FolderExists(fname).ok();
}

Status GcsFileSystem::ObjectExists(const string& bucket, const string& object) {
  FileStatistics stat;
  return StatForObject(bucket, object, &stat);
}

Status GcsFileSystem::StatForObject(const string& bucket, const string& object,
                                    FileStatistics* stat) {
  if (!stat) {
    return errors::Internal("'stat' cannot be nullptr.");
  }
  if (object.empty()) {
    return errors::InvalidArgument("'object' must be a non-empty string.");
  }

  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<char[]> scratch(new char[kBufferSize]);
  StringPiece response_piece;

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
      kGcsUriBase, "b/", bucket, "/o/", request->EscapeString(object),
      "?fields=size%2Cupdated")));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(
      request->SetResultBuffer(scratch.get(), kBufferSize, &response_piece));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      request->Send(), " when reading metadata of gs://", bucket, "/", object);

  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));

  // Parse file size.
  TF_RETURN_IF_ERROR(GetInt64Value(root, "size", &(stat->length)));

  // Parse file modification time.
  string updated;
  TF_RETURN_IF_ERROR(GetStringValue(root, "updated", &updated));
  TF_RETURN_IF_ERROR(ParseRfc3339Time(updated, &(stat->mtime_nsec)));

  stat->is_directory = false;

  return Status::OK();
}

Status GcsFileSystem::BucketExists(const string& bucket) {
  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket));
  request->AddAuthBearerHeader(auth_token);
  return request->Send();
}

Status GcsFileSystem::FolderExists(const string& dirname) {
  std::vector<string> children;
  TF_RETURN_IF_ERROR(GetChildrenBounded(dirname, 1, &children, true));
  if (children.empty()) {
    return errors::NotFound("Folder does not exist.");
  }
  return Status::OK();
}

Status GcsFileSystem::GetChildren(const string& dirname,
                                  std::vector<string>* result) {
  return GetChildrenBounded(dirname, UINT64_MAX, result, false);
}

Status GcsFileSystem::GetMatchingPaths(const string& pattern,
                                       std::vector<string>* results) {
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  const string& fixed_prefix =
      pattern.substr(0, pattern.find_first_of("*?[\\"));
  const string& dir = io::Dirname(fixed_prefix).ToString();
  if (dir.empty()) {
    return errors::InvalidArgument(
        strings::StrCat("A GCS pattern doesn't have a bucket name: ", pattern));
  }
  std::vector<string> all_files;
  TF_RETURN_IF_ERROR(GetChildrenBounded(dir, UINT64_MAX, &all_files, true));

  // Match all obtained files to the input pattern.
  for (const auto& f : all_files) {
    const string& full_path = io::JoinPath(dir, f);
    if (Env::Default()->MatchPath(full_path, pattern)) {
      results->push_back(full_path);
    }
  }
  return Status::OK();
}

Status GcsFileSystem::GetChildrenBounded(const string& dirname,
                                         uint64 max_results,
                                         std::vector<string>* result,
                                         bool recursive) {
  if (!result) {
    return errors::InvalidArgument("'result' cannot be null");
  }
  string bucket, object_prefix;
  TF_RETURN_IF_ERROR(
      ParseGcsPath(MaybeAppendSlash(dirname), true, &bucket, &object_prefix));

  string nextPageToken;
  uint64 retrieved_results = 0;
  while (true) {  // A loop over multiple result pages.
    string auth_token;
    TF_RETURN_IF_ERROR(
        AuthProvider::GetToken(auth_provider_.get(), &auth_token));

    std::unique_ptr<char[]> scratch(new char[kBufferSize]);
    StringPiece response_piece;
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
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
      uri = strings::StrCat(uri, "&prefix=",
                            request->EscapeString(object_prefix));
    }
    if (!nextPageToken.empty()) {
      uri = strings::StrCat(uri, "&pageToken=",
                            request->EscapeString(nextPageToken));
    }
    if (max_results - retrieved_results < kGetChildrenDefaultPageSize) {
      uri =
          strings::StrCat(uri, "&maxResults=", max_results - retrieved_results);
    }
    TF_RETURN_IF_ERROR(request->SetUri(uri));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(
        request->SetResultBuffer(scratch.get(), kBufferSize, &response_piece));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading ", dirname);
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));
    const auto items = root.get("items", Json::Value::null);
    if (items == Json::Value::null) {
      // Empty results.
      return Status::OK();
    }
    if (!items.isArray()) {
      return errors::Internal("Expected an array 'items' in the GCS response.");
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
      // 'object_prefix', which is part of 'dirname', should be removed from the
      // beginning of 'name'.
      StringPiece relative_path(name);
      if (!relative_path.Consume(object_prefix)) {
        return errors::Internal(
            strings::StrCat("Unexpected response: the returned file name ",
                            name, " doesn't match the prefix ", object_prefix));
      }
      result->emplace_back(relative_path.ToString());
      if (++retrieved_results >= max_results) {
        return Status::OK();
      }
    }
    const auto prefixes = root.get("prefixes", Json::Value::null);
    if (prefixes != Json::Value::null) {
      // Subfolders are returned for the non-recursive mode.
      if (!prefixes.isArray()) {
        return errors::Internal(
            "'prefixes' was expected to be an array in the GCS response.");
      }
      for (size_t i = 0; i < prefixes.size(); i++) {
        const auto prefix = prefixes.get(i, Json::Value::null);
        if (prefix == Json::Value::null || !prefix.isString()) {
          return errors::Internal(
              "'prefixes' was expected to be an array of strings in the GCS "
              "response.");
        }
        const string& prefix_str = prefix.asString();
        StringPiece relative_path(prefix_str);
        if (!relative_path.Consume(object_prefix)) {
          return errors::Internal(strings::StrCat(
              "Unexpected response: the returned folder name ", prefix_str,
              " doesn't match the prefix ", object_prefix));
        }
        result->emplace_back(relative_path.ToString());
        if (++retrieved_results >= max_results) {
          return Status::OK();
        }
      }
    }
    const auto token = root.get("nextPageToken", Json::Value::null);
    if (token == Json::Value::null) {
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
  if (StatForObject(bucket, object, stat).ok()) {
    return Status::OK();
  }
  if ((object.empty() && BucketExists(bucket).ok()) ||
      (!object.empty() && FolderExists(fname).ok())) {
    stat->length = 0;
    stat->mtime_nsec = 0;
    stat->is_directory = true;
    return Status::OK();
  }
  return errors::NotFound(
      strings::StrCat("The specified path ", fname, " was not found."));
}

Status GcsFileSystem::DeleteFile(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));

  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
      kGcsUriBase, "b/", bucket, "/o/", request->EscapeString(object))));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(request->SetDeleteRequest());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when deleting ", fname);
  return Status::OK();
}

Status GcsFileSystem::CreateDir(const string& dirname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(dirname, true, &bucket, &object));
  if (object.empty()) {
    if (BucketExists(bucket).ok()) {
      return Status::OK();
    }
    return errors::NotFound(
        strings::StrCat("The specified bucket ", dirname, " was not found."));
  }
  // Create a zero-length directory marker object.
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(NewWritableFile(MaybeAppendSlash(dirname), &file));
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

// Checks that the directory is empty (i.e no objects with this prefix exist).
// If it is, does nothing, because directories are not entities in GCS.
Status GcsFileSystem::DeleteDir(const string& dirname) {
  std::vector<string> children;
  // A directory is considered empty either if there are no matching objects
  // with the corresponding name prefix or if there is exactly one matching
  // object and it is the directory marker. Therefore we need to retrieve
  // at most two children for the prefix to detect if a directory is empty.
  TF_RETURN_IF_ERROR(GetChildrenBounded(dirname, 2, &children, true));

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
  TF_RETURN_IF_ERROR(GetChildrenBounded(src, UINT64_MAX, &children, true));
  for (const string& subpath : children) {
    // io::JoinPath() wouldn't work here, because we want an empty subpath
    // to result in an appended slash in order for directory markers
    // to be processed correctly: "gs://a/b" + "" should give "gs:/a/b/".
    TF_RETURN_IF_ERROR(
        RenameObject(strings::StrCat(MaybeAppendSlash(src), subpath),
                     strings::StrCat(MaybeAppendSlash(target), subpath)));
  }
  return Status::OK();
}

// Uses a GCS API command to copy the object and then deletes the old one.
Status GcsFileSystem::RenameObject(const string& src, const string& target) {
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseGcsPath(src, false, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(
      ParseGcsPath(target, false, &target_bucket, &target_object));

  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
      kGcsUriBase, "b/", src_bucket, "/o/", request->EscapeString(src_object),
      "/rewriteTo/b/", target_bucket, "/o/",
      request->EscapeString(target_object))));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(request->SetPostEmptyBody());
  std::unique_ptr<char[]> scratch(new char[kBufferSize]);
  StringPiece response_piece;
  TF_RETURN_IF_ERROR(
      request->SetResultBuffer(scratch.get(), kBufferSize, &response_piece));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when renaming ", src,
                                  " to ", target);

  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));
  bool done;
  TF_RETURN_IF_ERROR(GetBoolValue(root, "done", &done));
  if (!done) {
    // If GCS didn't complete rewrite in one call, this means that a large file
    // is being copied to a bucket with a different storage class or location,
    // which requires multiple rewrite calls.
    // TODO(surkov): implement multi-step rewrites.
    return errors::Unimplemented(
        strings::StrCat("Couldn't rename ", src, " to ", target,
                        ": moving large files between buckets with different "
                        "locations or storage classes is not supported."));
  }

  TF_RETURN_IF_ERROR(DeleteFile(src));
  return Status::OK();
}

Status GcsFileSystem::IsDirectory(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    if (BucketExists(bucket).ok()) {
      return Status::OK();
    }
    return errors::NotFound(strings::StrCat("The specified bucket gs://",
                                            bucket, " was not found."));
  }
  if (FolderExists(fname).ok()) {
    return Status::OK();
  }
  if (ObjectExists(bucket, object).ok()) {
    return errors::FailedPrecondition(
        strings::StrCat("The specified path ", fname, " is not a directory."));
  }
  return errors::NotFound(
      strings::StrCat("The specified path ", fname, " was not found."));
}

REGISTER_FILE_SYSTEM("gs", RetryingGcsFileSystem);

}  // namespace tensorflow
