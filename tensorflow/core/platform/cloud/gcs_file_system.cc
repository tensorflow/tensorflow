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
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/cloud/google_auth_provider.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {

namespace {

constexpr char kGcsUriBase[] = "https://www.googleapis.com/storage/v1/";
constexpr char kGcsUploadUriBase[] =
    "https://www.googleapis.com/upload/storage/v1/";
constexpr char kStorageHost[] = "storage.googleapis.com";
constexpr size_t kBufferSize = 1024 * 1024;  // In bytes.

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
/// If fname only contains the bucket, the returned object is empty.
Status ParseGcsPath(StringPiece fname, string* bucket, string* object) {
  if (!bucket || !object) {
    return errors::Internal("bucket and object cannot be null.");
  }
  if (!fname.Consume("gs://")) {
    return errors::InvalidArgument("GCS path must start with gs://");
  }
  auto first_slash = fname.find('/');
  if (first_slash == -1) {
    *bucket = fname.ToString();
    *object = string();
  } else {
    *bucket = fname.substr(0, first_slash).ToString();
    fname.remove_prefix(first_slash + 1);
    *object = fname.ToString();
  }
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
        http_request_factory_(std::move(http_request_factory)),
        read_ahead_bytes_(read_ahead_bytes) {}

  /// The implementation of reads with a read-ahead buffer.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    if (offset >= buffer_start_offset_ &&
        offset + n <= buffer_start_offset_ + buffer_content_size_) {
      // If the requested range is fully in the buffer, just return it.
      std::memcpy(scratch, buffer_.get() + offset - buffer_start_offset_, n);
      *result = StringPiece(scratch, n);
      return Status::OK();
    }

    // Update the buffer content based on the new requested range.
    auto buffer_size = n + read_ahead_bytes_;
    buffer_.reset(new char[buffer_size]);
    buffer_start_offset_ = offset;
    buffer_content_size_ = 0;
    StringPiece buffer_content;
    TF_RETURN_IF_ERROR(
        ReadFromGCS(offset, buffer_size, &buffer_content, buffer_.get()));
    buffer_content_size_ = buffer_content.size();

    // Set the results.
    *result = StringPiece(scratch, std::min(buffer_content_size_, n));
    std::memcpy(scratch, buffer_.get(), result->size());

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
    TF_RETURN_IF_ERROR(request->Send());
    return Status::OK();
  }

  string bucket_;
  string object_;
  AuthProvider* auth_provider_;
  HttpRequest::Factory* http_request_factory_;
  const size_t read_ahead_bytes_;

  // The buffer-related members need to be mutable, because they are modified
  // by the const Read() method.
  mutable std::unique_ptr<char[]> buffer_;
  // The original file offset of the first byte in the buffer.
  mutable size_t buffer_start_offset_ = 0;
  mutable size_t buffer_content_size_ = 0;
};

/// \brief GCS-based implementation of a writeable file.
///
/// Since GCS objects are immutable, this implementation writes to a local
/// tmp file and copies it to GCS on flush/close.
class GcsWritableFile : public WritableFile {
 public:
  GcsWritableFile(const string& bucket, const string& object,
                  AuthProvider* auth_provider,
                  HttpRequest::Factory* http_request_factory)
      : bucket_(bucket),
        object_(object),
        auth_provider_(auth_provider),
        http_request_factory_(std::move(http_request_factory)) {
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
                  HttpRequest::Factory* http_request_factory)
      : bucket_(bucket),
        object_(object),
        auth_provider_(auth_provider),
        http_request_factory_(std::move(http_request_factory)) {
    tmp_content_filename_ = tmp_content_filename;
    outfile_.open(tmp_content_filename_,
                  std::ofstream::binary | std::ofstream::app);
  }

  ~GcsWritableFile() { Close(); }

  Status Append(const StringPiece& data) override {
    TF_RETURN_IF_ERROR(CheckWritable());
    outfile_ << data;
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
  Status Sync() override {
    TF_RETURN_IF_ERROR(CheckWritable());
    outfile_.flush();
    string auth_token;
    TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_, &auth_token));

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
        kGcsUploadUriBase, "b/", bucket_, "/o?uploadType=media&name=",
        request->EscapeString(object_))));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(request->SetPostRequest(tmp_content_filename_));
    TF_RETURN_IF_ERROR(request->Send());
    return Status::OK();
  }

 private:
  Status CheckWritable() const {
    if (!outfile_.is_open()) {
      return errors::FailedPrecondition(
          "The underlying tmp file is not writable.");
    }
    return Status::OK();
  }

  string bucket_;
  string object_;
  AuthProvider* auth_provider_;
  string tmp_content_filename_;
  std::ofstream outfile_;
  HttpRequest::Factory* http_request_factory_;
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
    size_t read_ahead_bytes)
    : auth_provider_(std::move(auth_provider)),
      http_request_factory_(std::move(http_request_factory)),
      read_ahead_bytes_(read_ahead_bytes) {}

Status GcsFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));
  result->reset(new GcsRandomAccessFile(bucket, object, auth_provider_.get(),
                                        http_request_factory_.get(),
                                        read_ahead_bytes_));
  return Status::OK();
}

Status GcsFileSystem::NewWritableFile(const string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));
  result->reset(new GcsWritableFile(bucket, object, auth_provider_.get(),
                                    http_request_factory_.get()));
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
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));
  result->reset(new GcsWritableFile(bucket, object, auth_provider_.get(),
                                    old_content_filename,
                                    http_request_factory_.get()));
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
  string bucket, object_prefix;
  if (!ParseGcsPath(fname, &bucket, &object_prefix).ok()) {
    LOG(ERROR) << "Could not parse GCS file name " << fname;
    return false;
  }

  string auth_token;
  if (!AuthProvider::GetToken(auth_provider_.get(), &auth_token).ok()) {
    LOG(ERROR) << "Could not get an auth token.";
    return false;
  }

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  if (!request->Init().ok()) {
    LOG(ERROR) << "Could not initialize the HTTP request.";
    return false;
  }
  if (!object_prefix.empty()) {
    request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                    request->EscapeString(object_prefix),
                                    "?fields=size"));
  } else {
    request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket));
  }
  request->AddAuthBearerHeader(auth_token);
  return request->Send().ok();
}

Status GcsFileSystem::GetChildren(const string& dirname,
                                  std::vector<string>* result) {
  if (!result) {
    return errors::InvalidArgument("'result' cannot be null");
  }
  string sanitized_dirname = dirname;
  if (!dirname.empty() && dirname.back() != '/') {
    sanitized_dirname += "/";
  }
  string bucket, object_prefix;
  TF_RETURN_IF_ERROR(ParseGcsPath(sanitized_dirname, &bucket, &object_prefix));

  string nextPageToken;
  while (true) {  // A loop over multiple result pages.
    string auth_token;
    TF_RETURN_IF_ERROR(
        AuthProvider::GetToken(auth_provider_.get(), &auth_token));

    std::unique_ptr<char[]> scratch(new char[kBufferSize]);
    StringPiece response_piece;
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    TF_RETURN_IF_ERROR(request->Init());
    auto uri = strings::StrCat(kGcsUriBase, "b/", bucket,
                               "/o?fields=items%2Fname%2CnextPageToken");
    if (!object_prefix.empty()) {
      uri = strings::StrCat(uri, "&prefix=",
                            request->EscapeString(object_prefix));
    }
    if (!nextPageToken.empty()) {
      uri = strings::StrCat(uri, "&pageToken=",
                            request->EscapeString(nextPageToken));
    }
    TF_RETURN_IF_ERROR(request->SetUri(uri));
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(
        request->SetResultBuffer(scratch.get(), kBufferSize, &response_piece));
    TF_RETURN_IF_ERROR(request->Send());
    std::stringstream response_stream;
    response_stream << response_piece;
    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(response_stream.str(), root)) {
      return errors::Internal("Couldn't parse JSON response from GCS.");
    }
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
      const auto name = item.get("name", Json::Value::null);
      if (name == Json::Value::null || !name.isString()) {
        return errors::Internal(
            "Unexpected JSON format: 'items.name' is missing or not a string.");
      }
      // The names should be relative to the 'dirname'. That means the
      // 'object_prefix', which is part of 'dirname', should be removed from the
      // beginning of 'name'.
      string name_str(name.asString());
      result->emplace_back(name_str.begin() + object_prefix.size(),
                           name_str.end());
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
  return errors::Unimplemented("Stat unimplemented");
}

Status GcsFileSystem::DeleteFile(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));

  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
      kGcsUriBase, "b/", bucket, "/o/", request->EscapeString(object))));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(request->SetDeleteRequest());
  TF_RETURN_IF_ERROR(request->Send());
  return Status::OK();
}

// Does nothing, because directories are not entities in GCS.
Status GcsFileSystem::CreateDir(const string& dirname) { return Status::OK(); }

// Checks that the directory is empty (i.e no objects with this prefix exist).
// If it is, does nothing, because directories are not entities in GCS.
Status GcsFileSystem::DeleteDir(const string& dirname) {
  std::vector<string> children;
  TF_RETURN_IF_ERROR(GetChildren(dirname, &children));
  if (!children.empty()) {
    return errors::InvalidArgument("Cannot delete a non-empty directory.");
  }
  return Status::OK();
}

Status GcsFileSystem::GetFileSize(const string& fname, uint64* file_size) {
  string bucket, object_prefix;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object_prefix));

  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<char[]> scratch(new char[kBufferSize]);
  StringPiece response_piece;

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(
      strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                      request->EscapeString(object_prefix), "?fields=size")));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(
      request->SetResultBuffer(scratch.get(), kBufferSize, &response_piece));
  TF_RETURN_IF_ERROR(request->Send());
  std::stringstream response_stream;
  response_stream << response_piece;

  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(response_stream.str(), root)) {
    return errors::Internal("Couldn't parse JSON response from GCS.");
  }
  const auto size = root.get("size", Json::Value::null);
  if (size == Json::Value::null) {
    return errors::Internal("'size' was expected in the JSON response.");
  }
  if (size.isNumeric()) {
    *file_size = size.asUInt64();
  } else if (size.isString()) {
    if (!strings::safe_strtou64(size.asString().c_str(), file_size)) {
      return errors::Internal("'size' couldn't be parsed as a nubmer.");
    }
  } else {
    return errors::Internal("'size' is not a number in the JSON response.");
  }
  return Status::OK();
}

// Uses a GCS API command to copy the object and then deletes the old one.
Status GcsFileSystem::RenameFile(const string& src, const string& target) {
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseGcsPath(src, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(ParseGcsPath(target, &target_bucket, &target_object));

  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
      kGcsUriBase, "b/", src_bucket, "/o/", request->EscapeString(src_object),
      "/rewriteTo/b/", target_bucket, "/o/",
      request->EscapeString(target_object))));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(request->SetPostRequest());
  TF_RETURN_IF_ERROR(request->Send());

  TF_RETURN_IF_ERROR(DeleteFile(src));
  return Status::OK();
}

REGISTER_FILE_SYSTEM("gs", RetryingGcsFileSystem);

}  // namespace tensorflow
