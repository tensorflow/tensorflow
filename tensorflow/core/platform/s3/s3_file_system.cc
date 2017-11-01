/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/s3/s3_file_system.h"
#include "tensorflow/core/platform/s3/s3_crypto.h"

#include <aws/core/Aws.h>
#include <aws/core/utils/FileSystemUtils.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>

#include <cstdlib>

namespace tensorflow {

static const char* kS3FileSystemAllocationTag = "S3FileSystemAllocation";
static const size_t kS3ReadAppendableFileBufferSize = 1024 * 1024;
static const int kS3GetChildrenMaxKeys = 100;

Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  static mutex cfg_lock;
  static bool init(false);
  static Aws::Client::ClientConfiguration cfg;

  std::lock_guard<mutex> lock(cfg_lock);

  if (!init) {
    const char* endpoint = getenv("S3_ENDPOINT");
    if (endpoint) {
      cfg.endpointOverride = Aws::String(endpoint);
    }
    const char* region = getenv("S3_REGION");
    if (region) {
      cfg.region = Aws::String(region);
    }
    const char* use_https = getenv("S3_USE_HTTPS");
    if (use_https) {
      if (use_https[0] == '0') {
        cfg.scheme = Aws::Http::Scheme::HTTP;
      } else {
        cfg.scheme = Aws::Http::Scheme::HTTPS;
      }
    }
    const char* verify_ssl = getenv("S3_VERIFY_SSL");
    if (verify_ssl) {
      if (verify_ssl[0] == '0') {
        cfg.verifySSL = false;
      } else {
        cfg.verifySSL = true;
      }
    }

    init = true;
  }

  return cfg;
};

Status ParseS3Path(const string& fname, bool empty_object_ok, string* bucket,
                   string* object) {
  if (!bucket || !object) {
    return errors::Internal("bucket and object cannot be null.");
  }
  StringPiece scheme, bucketp, objectp;
  io::ParseURI(fname, &scheme, &bucketp, &objectp);
  if (scheme != "s3") {
    return errors::InvalidArgument("S3 path doesn't start with 's3://': ",
                                   fname);
  }
  *bucket = bucketp.ToString();
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("S3 path doesn't contain a bucket name: ",
                                   fname);
  }
  objectp.Consume("/");
  *object = objectp.ToString();
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("S3 path doesn't contain an object name: ",
                                   fname);
  }
  return Status::OK();
}

class S3RandomAccessFile : public RandomAccessFile {
 public:
  S3RandomAccessFile(const string& bucket, const string& object)
      : bucket_(bucket), object_(object) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Aws::S3::S3Client s3Client(GetDefaultClientConfig());
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.WithBucket(bucket_.c_str()).WithKey(object_.c_str());
    string bytes = strings::StrCat("bytes=", offset, "-", offset + n - 1);
    getObjectRequest.SetRange(bytes.c_str());
    getObjectRequest.SetResponseStreamFactory([]() {
      return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag);
    });
    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
      n = 0;
      *result = StringPiece(scratch, n);
      return Status(error::OUT_OF_RANGE, "Read less bytes than requested");
    }
    n = getObjectOutcome.GetResult().GetContentLength();
    std::stringstream ss;
    ss << getObjectOutcome.GetResult().GetBody().rdbuf();
    ss.read(scratch, n);

    *result = StringPiece(scratch, n);
    return Status::OK();
  }

 private:
  string bucket_;
  string object_;
};

class S3WritableFile : public WritableFile {
 public:
  S3WritableFile(const string& bucket, const string& object)
      : bucket_(bucket),
        object_(object),
        sync_needed_(true),
        outfile_(Aws::MakeShared<Aws::Utils::TempFile>(
            kS3FileSystemAllocationTag, "/tmp/s3_filesystem_XXXXXX",
            std::ios_base::binary | std::ios_base::trunc | std::ios_base::in |
                std::ios_base::out)) {}

  Status Append(const StringPiece& data) override {
    if (!outfile_) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    sync_needed_ = true;
    outfile_->write(data.data(), data.size());
    if (!outfile_->good()) {
      return errors::Internal(
          "Could not append to the internal temporary file.");
    }
    return Status::OK();
  }

  Status Close() override {
    if (outfile_) {
      TF_RETURN_IF_ERROR(Sync());
      outfile_.reset();
    }
    return Status::OK();
  }

  Status Flush() override { return Sync(); }

  Status Sync() override {
    if (!outfile_) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    if (!sync_needed_) {
      return Status::OK();
    }
    Aws::Client::ClientConfiguration clientConfig = GetDefaultClientConfig();
    clientConfig.connectTimeoutMs = 300000;
    clientConfig.requestTimeoutMs = 600000;
    Aws::S3::S3Client s3Client(clientConfig);
    Aws::S3::Model::PutObjectRequest putObjectRequest;
    putObjectRequest.WithBucket(bucket_.c_str()).WithKey(object_.c_str());
    long offset = outfile_->tellp();
    outfile_->seekg(0);
    putObjectRequest.SetBody(outfile_);
    putObjectRequest.SetContentLength(offset);
    auto putObjectOutcome = s3Client.PutObject(putObjectRequest);
    outfile_->clear();
    outfile_->seekp(offset);
    if (!putObjectOutcome.IsSuccess()) {
      string error = strings::StrCat(
          putObjectOutcome.GetError().GetExceptionName().c_str(), ": ",
          putObjectOutcome.GetError().GetMessage().c_str());
      return errors::Internal(error);
    }
    return Status::OK();
  }

 private:
  string bucket_;
  string object_;
  bool sync_needed_;
  std::shared_ptr<Aws::Utils::TempFile> outfile_;
};

class S3ReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  S3ReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

S3FileSystem::S3FileSystem() {
  Aws::SDKOptions options;
  options.cryptoOptions.sha256Factory_create_fn = []() {
    return Aws::MakeShared<S3SHA256Factory>(S3CryptoAllocationTag);
  };
  options.cryptoOptions.sha256HMACFactory_create_fn = []() {
    return Aws::MakeShared<S3SHA256HmacFactory>(S3CryptoAllocationTag);
  };
  Aws::InitAPI(options);
}

S3FileSystem::~S3FileSystem() {
  Aws::SDKOptions options;
  Aws::ShutdownAPI(options);
}

Status S3FileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3RandomAccessFile(bucket, object));
  return Status::OK();
}

Status S3FileSystem::NewWritableFile(const string& fname,
                                     std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3WritableFile(bucket, object));
  return Status::OK();
}

Status S3FileSystem::NewAppendableFile(const string& fname,
                                       std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<RandomAccessFile> reader;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &reader));
  std::unique_ptr<char[]> buffer(new char[kS3ReadAppendableFileBufferSize]);
  Status status;
  uint64 offset = 0;
  StringPiece read_chunk;

  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3WritableFile(bucket, object));

  while (true) {
    status = reader->Read(offset, kS3ReadAppendableFileBufferSize, &read_chunk,
                          buffer.get());
    if (status.ok()) {
      (*result)->Append(read_chunk);
      offset += kS3ReadAppendableFileBufferSize;
    } else if (status.code() == error::OUT_OF_RANGE) {
      (*result)->Append(read_chunk);
      break;
    } else {
      (*result).reset();
      return status;
    }
  }

  return Status::OK();
}

Status S3FileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(fname, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new S3ReadOnlyMemoryRegion(std::move(data), size));
  return Status::OK();
}

Status S3FileSystem::FileExists(const string& fname) {
  FileStatistics stats;
  TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
  return Status::OK();
}

Status S3FileSystem::GetChildren(const string& dir,
                                 std::vector<string>* result) {
  string bucket, prefix;
  TF_RETURN_IF_ERROR(ParseS3Path(dir, false, &bucket, &prefix));

  if (prefix.back() != '/') {
    prefix.push_back('/');
  }

  Aws::S3::S3Client s3Client(GetDefaultClientConfig());
  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys)
      .WithDelimiter("/");
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  do {
    auto listObjectsOutcome = s3Client.ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      string error = strings::StrCat(
          listObjectsOutcome.GetError().GetExceptionName().c_str(), ": ",
          listObjectsOutcome.GetError().GetMessage().c_str());
      return errors::Internal(error);
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetCommonPrefixes()) {
      Aws::String s = object.GetPrefix();
      s.erase(s.length() - 1);
      Aws::String entry = s.substr(strlen(prefix.c_str()));
      if (entry.length() > 0) {
        result->push_back(entry.c_str());
      }
    }
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String s = object.GetKey();
      Aws::String entry = s.substr(strlen(prefix.c_str()));
      if (entry.length() > 0) {
        result->push_back(entry.c_str());
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  return Status::OK();
}

Status S3FileSystem::Stat(const string& fname, FileStatistics* stats) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, true, &bucket, &object));

  Aws::S3::S3Client s3Client(GetDefaultClientConfig());
  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket.c_str());
    auto headBucketOutcome = s3Client.HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      string error = strings::StrCat(
          headBucketOutcome.GetError().GetExceptionName().c_str(), ": ",
          headBucketOutcome.GetError().GetMessage().c_str());
      return errors::Internal(error);
    }
    stats->length = 0;
    stats->is_directory = 1;
    return Status::OK();
  }

  bool found = false;

  Aws::S3::Model::HeadObjectRequest headObjectRequest;
  headObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
  headObjectRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto headObjectOutcome = s3Client.HeadObject(headObjectRequest);
  if (headObjectOutcome.IsSuccess()) {
    stats->length = headObjectOutcome.GetResult().GetContentLength();
    stats->is_directory = 0;
    stats->mtime_nsec =
        headObjectOutcome.GetResult().GetLastModified().Millis() * 1e6;
    found = true;
  }
  string prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }
  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(1);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto listObjectsOutcome = s3Client.ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    if (listObjectsOutcome.GetResult().GetContents().size() > 0) {
      stats->length = 0;
      stats->is_directory = 1;
      found = true;
    }
  }
  if (!found) {
    return errors::NotFound("Object ", fname, " does not exist");
  }
  return Status::OK();
}

Status S3FileSystem::DeleteFile(const string& fname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));

  Aws::S3::S3Client s3Client(GetDefaultClientConfig());
  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());

  auto deleteObjectOutcome = s3Client.DeleteObject(deleteObjectRequest);
  if (!deleteObjectOutcome.IsSuccess()) {
    string error = strings::StrCat(
        deleteObjectOutcome.GetError().GetExceptionName().c_str(), ": ",
        deleteObjectOutcome.GetError().GetMessage().c_str());
    return errors::Internal(error);
  }
  return Status::OK();
}

Status S3FileSystem::CreateDir(const string& dirname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, true, &bucket, &object));

  if (object.empty()) {
    Aws::S3::S3Client s3Client(GetDefaultClientConfig());
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket.c_str());
    auto headBucketOutcome = s3Client.HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      return errors::NotFound("The bucket ", bucket, " was not found.");
    }
    return Status::OK();
  }
  string filename = dirname;
  if (filename.back() != '/') {
    filename.push_back('/');
  }
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(NewWritableFile(filename, &file));
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

Status S3FileSystem::DeleteDir(const string& dirname) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, false, &bucket, &object));

  Aws::S3::S3Client s3Client(GetDefaultClientConfig());
  string prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }
  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(2);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto listObjectsOutcome = s3Client.ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto contents = listObjectsOutcome.GetResult().GetContents();
    if (contents.size() > 1 ||
        (contents.size() == 1 && contents[0].GetKey() != prefix.c_str())) {
      return errors::FailedPrecondition("Cannot delete a non-empty directory.");
    }
    if (contents.size() == 1 && contents[0].GetKey() == prefix.c_str()) {
      string filename = dirname;
      if (filename.back() != '/') {
        filename.push_back('/');
      }
      return DeleteFile(filename);
    }
  }
  return Status::OK();
}

Status S3FileSystem::GetFileSize(const string& fname, uint64* file_size) {
  FileStatistics stats;
  TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
  *file_size = stats.length;
  return Status::OK();
}

Status S3FileSystem::RenameFile(const string& src, const string& target) {
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseS3Path(src, false, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(
      ParseS3Path(target, false, &target_bucket, &target_object));
  if (src_object.back() == '/') {
    if (target_object.back() != '/') {
      target_object.push_back('/');
    }
  } else {
    if (target_object.back() == '/') {
      target_object.pop_back();
    }
  }

  Aws::S3::S3Client s3Client(GetDefaultClientConfig());

  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;

  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(src_bucket.c_str())
      .WithPrefix(src_object.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys);
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  do {
    auto listObjectsOutcome = s3Client.ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      string error = strings::StrCat(
          listObjectsOutcome.GetError().GetExceptionName().c_str(), ": ",
          listObjectsOutcome.GetError().GetMessage().c_str());
      return errors::Internal(error);
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String src_key = object.GetKey();
      Aws::String target_key = src_key;
      target_key.replace(0, src_object.length(), target_object.c_str());
      Aws::String source = Aws::String(src_bucket.c_str()) + "/" + src_key;

      copyObjectRequest.SetBucket(target_bucket.c_str());
      copyObjectRequest.SetKey(target_key);
      copyObjectRequest.SetCopySource(source);

      auto copyObjectOutcome = s3Client.CopyObject(copyObjectRequest);
      if (!copyObjectOutcome.IsSuccess()) {
        string error = strings::StrCat(
            copyObjectOutcome.GetError().GetExceptionName().c_str(), ": ",
            copyObjectOutcome.GetError().GetMessage().c_str());
        return errors::Internal(error);
      }

      deleteObjectRequest.SetBucket(src_bucket.c_str());
      deleteObjectRequest.SetKey(src_key.c_str());

      auto deleteObjectOutcome = s3Client.DeleteObject(deleteObjectRequest);
      if (!deleteObjectOutcome.IsSuccess()) {
        string error = strings::StrCat(
            deleteObjectOutcome.GetError().GetExceptionName().c_str(), ": ",
            deleteObjectOutcome.GetError().GetMessage().c_str());
        return errors::Internal(error);
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  return Status::OK();
}

REGISTER_FILE_SYSTEM("s3", S3FileSystem);

}  // namespace tensorflow
