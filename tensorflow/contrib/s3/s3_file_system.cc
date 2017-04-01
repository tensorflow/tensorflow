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
#include "tensorflow/core/platform/env.h"

#include "tensorflow/contrib/s3/s3_crypto.h"

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>

namespace tensorflow {

static const char* S3FileSystemAllocationTag = "S3FileSystemAllocation";

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
  S3RandomAccessFile(const string& fname) : fname_(fname) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseS3Path(fname_, false, &bucket, &object));

    Aws::S3::S3Client s3Client;
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
    char buffer[50];
    memset(buffer, 0x00, sizeof(buffer));
    snprintf(buffer, sizeof(buffer) - 1, "bytes=%lld-%lld", offset,
             offset + n - 1);
    getObjectRequest.SetRange(buffer);
    getObjectRequest.SetResponseStreamFactory([]() {
      return Aws::New<Aws::StringStream>(S3FileSystemAllocationTag);
    });
    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
      std::stringstream ss;
      ss << getObjectOutcome.GetError().GetExceptionName() << ": "
         << getObjectOutcome.GetError().GetMessage();
      return Status(error::INVALID_ARGUMENT, StringPiece(ss.str()));
    }
    n = getObjectOutcome.GetResult().GetContentLength();
    std::stringstream ss;
    ss << getObjectOutcome.GetResult().GetBody().rdbuf();
    ss.read(scratch, n);

    *result = StringPiece(scratch, n);
    return Status::OK();
  }
  string fname_;
};

class S3FileSystem : public FileSystem {
 public:
  S3FileSystem() {
    Aws::SDKOptions options;
    options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Info;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<S3SHA256Factory>(S3CryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<S3SHA256HmacFactory>(S3CryptoAllocationTag);
    };
    Aws::InitAPI(options);
  }
  ~S3FileSystem() {
    Aws::SDKOptions options;
    options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Info;
    Aws::ShutdownAPI(options);
  }
  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    result->reset(new S3RandomAccessFile(fname));
    return Status::OK();
  }
  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewWritableFile unimplemented");
  }

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewAppendableFile unimplemented");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented(
        "NewReadOnlyMemoryRegionFromFile unimplemented");
  }

  Status FileExists(const string& fname) override {
    FileStatistics stats;
    TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
    return Status::OK();
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    return GetChildrenBounded(dir, UINT64_MAX, result);
  }

  Status GetChildrenBounded(const string& dir, uint64 max_results,
                            std::vector<string>* result) {
    string bucket, prefix;
    TF_RETURN_IF_ERROR(ParseS3Path(dir, false, &bucket, &prefix));
    if (prefix.back() != '/') {
      prefix.push_back('/');
    }

    Aws::S3::S3Client s3Client;
    Aws::S3::Model::ListObjectsV2Request listObjectsV2Request;
    listObjectsV2Request.WithBucket(bucket.c_str())
        .WithPrefix(prefix.c_str())
        .WithMaxKeys(1)
        .WithDelimiter("/");
    listObjectsV2Request.SetResponseStreamFactory([]() {
      return Aws::New<Aws::StringStream>(S3FileSystemAllocationTag);
    });

    Aws::S3::Model::ListObjectsV2Result listObjectsV2Result;

    uint64 results = 0;
    do {
      auto listObjectsV2Outcome = s3Client.ListObjectsV2(listObjectsV2Request);
      if (!listObjectsV2Outcome.IsSuccess()) {
        std::stringstream ss;
        ss << listObjectsV2Outcome.GetError().GetExceptionName() << ": "
           << listObjectsV2Outcome.GetError().GetMessage();
        return Status(error::INVALID_ARGUMENT, StringPiece(ss.str()));
      }

      listObjectsV2Result = listObjectsV2Outcome.GetResult();
      for (const auto& object : listObjectsV2Result.GetCommonPrefixes()) {
        Aws::String s = object.GetPrefix();
        s.erase(s.length() - 1);
        result->push_back(s.substr(strlen(prefix.c_str())).c_str());
        if (++results >= max_results) {
          return Status::OK();
        }
      }
      for (const auto& object : listObjectsV2Result.GetContents()) {
        Aws::String s = object.GetKey();
        result->push_back(s.substr(strlen(prefix.c_str())).c_str());
        if (++results >= max_results) {
          return Status::OK();
        }
      }
      listObjectsV2Request.SetContinuationToken(
          listObjectsV2Result.GetNextContinuationToken());
    } while (listObjectsV2Result.GetIsTruncated());

    return Status::OK();
  }

  Status DeleteFile(const string& fname) override {
    return errors::Unimplemented("DeleteFile unimplemented");
  }

  Status CreateDir(const string& dirname) override {
    return errors::Unimplemented("CreateDir unimplemented");
  }

  Status DeleteDir(const string& dirname) override {
    return errors::Unimplemented("DeleteDir unimplemented");
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    FileStatistics stats;
    TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
    *file_size = stats.length;
    return Status::OK();
  }

  Status RenameFile(const string& src, const string& target) override {
    return errors::Unimplemented("RenameFile unimplemented");
  }

  Status Stat(const string& fname, FileStatistics* stats) override {
    string bucket, object;
    TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));

    Aws::S3::S3Client s3Client;

    Aws::S3::Model::HeadObjectRequest headObjectRequest;
    headObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
    headObjectRequest.SetResponseStreamFactory([]() {
      return Aws::New<Aws::StringStream>(S3FileSystemAllocationTag);
    });
    auto headObjectOutcome = s3Client.HeadObject(headObjectRequest);
    if (!headObjectOutcome.IsSuccess()) {
      std::vector<string> result;
      TF_RETURN_IF_ERROR(GetChildrenBounded(fname, 1, &result));
      if (result.size() > 0) {
        stats->is_directory = 1;
      }
    }
    stats->length = headObjectOutcome.GetResult().GetContentLength();
    stats->mtime_nsec =
        headObjectOutcome.GetResult().GetLastModified().Millis() * 1e6;
    return Status::OK();
  }
};

REGISTER_FILE_SYSTEM("s3", S3FileSystem);

}  // namespace tensorflow
