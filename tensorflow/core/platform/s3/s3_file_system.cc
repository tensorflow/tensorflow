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
#include "tensorflow/core/platform/s3/s3_file_system.h"

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/utils/FileSystemUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/UploadPartCopyRequest.h>

#include <cmath>
#include <cstdlib>

#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/s3/aws_crypto.h"
#include "tensorflow/core/platform/s3/aws_logging.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

namespace {
#ifdef PLATFORM_WINDOWS
// On Windows, `Aws::FileSystem::CreateTempFilePath()` return
// `C:\Users\username\AppData\Local\Temp\`. Adding template will cause an error.
static const char* kS3TempFileTemplate = nullptr;
#else
static const char* kS3TempFileTemplate = "/tmp/s3_filesystem_XXXXXX";
#endif
static const char* kS3FileSystemAllocationTag = "S3FileSystemAllocation";
static const size_t kS3ReadAppendableFileBufferSize = 1024 * 1024;
static const int64 kS3TimeoutMsec = 300000;                       // 5 min
static const uint64 kS3MultiPartUploadChunkSize = 50 * 1024 * 1024;  // 50 MB
static const uint64 kS3MultiPartDownloadChunkSize = 2 * 1024 * 1024;  // 50 MB
static const int kS3GetChildrenMaxKeys = 100;

// With this change multiple threads are used in one single download.
// Increasing the thread pool size since multiple downloads
// and uploads can occur in parallel.
static const int kExecutorPoolSize = 25;
static const int kUploadRetries = 3;
static const int kDownloadRetries = 3;
static const char* kExecutorTag = "TransferManagerExecutor";

Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  static mutex cfg_lock(LINKER_INITIALIZED);
  static bool init(false);
  static Aws::Client::ClientConfiguration cfg;

  std::lock_guard<mutex> lock(cfg_lock);

  if (!init) {
    const char* endpoint = getenv("S3_ENDPOINT");
    if (endpoint) {
      cfg.endpointOverride = Aws::String(endpoint);
    }
    const char* region = getenv("AWS_REGION");
    if (!region) {
      // TODO (yongtang): `S3_REGION` should be deprecated after 2.0.
      region = getenv("S3_REGION");
    }
    if (region) {
      cfg.region = Aws::String(region);
    } else {
      // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
      // is set with a truthy value.
      const char* load_config_env = getenv("AWS_SDK_LOAD_CONFIG");
      string load_config =
          load_config_env ? absl::AsciiStrToLower(load_config_env) : "";
      if (load_config == "true" || load_config == "1") {
        Aws::String config_file;
        // If AWS_CONFIG_FILE is set then use it, otherwise use ~/.aws/config.
        const char* config_file_env = getenv("AWS_CONFIG_FILE");
        if (config_file_env) {
          config_file = config_file_env;
        } else {
          const char* home_env = getenv("HOME");
          if (home_env) {
            config_file = home_env;
            config_file += "/.aws/config";
          }
        }
        Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
        loader.Load();
        auto profiles = loader.GetProfiles();
        if (!profiles["default"].GetRegion().empty()) {
          cfg.region = profiles["default"].GetRegion();
        }
      }
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
    // if these timeouts are low, you may see an error when
    // uploading/downloading large files: Unable to connect to endpoint
    const char* connect_timeout_str = getenv("S3_CONNECT_TIMEOUT_MSEC");
    int64 connect_timeout = kS3TimeoutMsec;
    if (connect_timeout_str) {
      // if conversion is unsafe, below method doesn't modify connect_timeout
      strings::safe_strto64(connect_timeout_str, &connect_timeout);
    }
    cfg.connectTimeoutMs = connect_timeout;

    const char* request_timeout_str = getenv("S3_REQUEST_TIMEOUT_MSEC");
    int64 request_timeout = kS3TimeoutMsec;
    if (request_timeout_str) {
      strings::safe_strto64(request_timeout_str, &request_timeout);
    }
    cfg.requestTimeoutMs = request_timeout;

    const char* ca_file = getenv("S3_CA_FILE");
    if (ca_file) {
      cfg.caFile = Aws::String(ca_file);
    }
    const char* ca_path = getenv("S3_CA_PATH");
    if (ca_path) {
      cfg.caPath = Aws::String(ca_path);
    }

    init = true;
  }

  return cfg;
};

void ShutdownClient(Aws::S3::S3Client* s3_client) {
  if (s3_client != nullptr) {
    delete s3_client;
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
    AWSLogSystem::ShutdownAWSLogging();
  }
}

void ShutdownTransferManager(Aws::Transfer::TransferManager* transfer_manager) {
  if (transfer_manager != nullptr) {
    delete transfer_manager;
  }
}

void ShutdownExecutor(Aws::Utils::Threading::PooledThreadExecutor* executor) {
  if (executor != nullptr) {
    delete executor;
  }
}

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
  *bucket = string(bucketp);
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("S3 path doesn't contain a bucket name: ",
                                   fname);
  }
  absl::ConsumePrefix(&objectp, "/");
  *object = string(objectp);
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("S3 path doesn't contain an object name: ",
                                   fname);
  }
  return Status::OK();
}

static Status CheckForbiddenError(
    const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  if (error.GetResponseCode() == Aws::Http::HttpResponseCode::FORBIDDEN) {
    return errors::FailedPrecondition(
        "AWS Credentials have not been set properly. "
        "Unable to access the specified S3 location");
  } else {
    return Status::OK();
  }
}

static Status CreateStatusFromAwsError(
    const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  TF_RETURN_IF_ERROR(CheckForbiddenError(error));
  return errors::Unknown(error.GetExceptionName(), ": ", error.GetMessage());
}

class S3RandomAccessFile : public RandomAccessFile {
 public:
  S3RandomAccessFile(const string& bucket, const string& object, 
                     const bool use_multi_part_download, 
                     std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager,
                     std::shared_ptr<Aws::S3::S3Client> s3_client)
                    : bucket_(bucket), object_(object), 
                      use_multi_part_download_(use_multi_part_download),
                      transfer_manager_(transfer_manager),
                      s3_client_(s3_client) {}
  
  Status Name(StringPiece* result) const override {
    return errors::Unimplemented("S3RandomAccessFile does not support Name()");
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    VLOG(1) << "ReadFilefromS3 s3://" << bucket_ << "/" << object_ << " from "
            << offset << " for n:" << n;
    if (use_multi_part_download_) {
      return ReadS3TransferManager(offset, n, result, scratch);
    } else {
      return ReadS3Client(offset, n, result, scratch);
    }
  }

  Status ReadS3TransferManager(uint64 offset, size_t n, StringPiece* result,
                               char* scratch) const {
    VLOG(3) << "Using TransferManager";
    
    auto create_stream_fn = [&]() {  // create stream lambda fn
       return Aws::New<TFS3UnderlyingStream>(
           "S3ReadStream",
           Aws::New<Aws::Utils::Stream::PreallocatedStreamBuf>(
             "S3ReadStream", reinterpret_cast<unsigned char*>(scratch), n));
    };
    
    VLOG(3) << "Created stream to read with transferManager";

    std::shared_ptr<Aws::Transfer::TransferHandle> handle =
      transfer_manager_.get()->DownloadFile(
        bucket_.c_str(), object_.c_str(), offset, n, create_stream_fn);
    handle->WaitUntilFinished();

    // todo change this
    int retries = 0;
    
    while (
      handle->GetStatus() == Aws::Transfer::TransferStatus::FAILED &&
      handle->GetLastError().GetResponseCode() != Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE && 
      retries++ < kDownloadRetries) {
      // only failed parts will be downloaded again
      VLOG(1) << "Retrying read of s3://" << bucket_ << "/" << object_
              << " after failure. Current retry count:" << retries;
      transfer_manager_.get()->RetryDownload(handle);
      handle->WaitUntilFinished();
    }

    if (handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED) {
      auto error = handle->GetLastError();
      if (error.GetResponseCode() ==
          Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE) {
        // expected when end of file is reached
        n = 0;
        *result = StringPiece(scratch, n);
        return Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      }
      return CreateStatusFromAwsError(error);
    } else {
      n = handle->GetBytesTotalSize();
      *result = StringPiece(scratch, handle->GetBytesTransferred());
      return Status::OK();
    }       
  }

  Status ReadS3Client(uint64 offset, size_t n, StringPiece* result,
                      char* scratch) const {
    VLOG(3) << "ReadFile using S3Client s3://" << bucket_ << "/" << object_;
      
    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.WithBucket(bucket_.c_str()).WithKey(object_.c_str());
    string bytes = strings::StrCat("bytes=", offset, "-", offset + n - 1);
    getObjectRequest.SetRange(bytes.c_str());
    getObjectRequest.SetResponseStreamFactory([]() {
      return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag);
    });
    
    auto getObjectOutcome = this->s3_client_->GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
      auto error = getObjectOutcome.GetError();
      if (error.GetResponseCode() ==
          Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE) {
        n = 0;
        *result = StringPiece(scratch, n);
        return Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      }
      return CreateStatusFromAwsError(error);
    } else {
      n = getObjectOutcome.GetResult().GetContentLength();
      getObjectOutcome.GetResult().GetBody().read(scratch, n);

      *result = StringPiece(scratch, n);
      return Status::OK();
    }
  }

 private:
  string bucket_;
  string object_;
  std::shared_ptr<Aws::S3::S3Client> s3_client_;
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;
  bool use_multi_part_download_;
};

class S3WritableFile : public WritableFile {
 public:
  S3WritableFile(
      const string& bucket, const string& object,
      std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager,
      std::shared_ptr<Aws::S3::S3Client> s3_client)
      : bucket_(bucket),
        object_(object),
        s3_client_(s3_client),
        transfer_manager_(transfer_manager),
        sync_needed_(true),
        outfile_(Aws::MakeShared<Aws::Utils::TempFile>(
            kS3FileSystemAllocationTag, kS3TempFileTemplate,
            std::ios_base::binary | std::ios_base::trunc | std::ios_base::in |
                std::ios_base::out)) {}

  Status Append(StringPiece data) override {
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

  Status Name(StringPiece* result) const override {
    return errors::Unimplemented("S3WritableFile does not support Name()");
  }

  Status Sync() override {
    if (!outfile_) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    if (!sync_needed_) {
      return Status::OK();
    }
    VLOG(1) << "WriteFileToS3: s3://" << bucket_ << "/" << object_;
    long offset = outfile_->tellp();
    std::shared_ptr<Aws::Transfer::TransferHandle> handle =
        transfer_manager_.get()->UploadFile(
            outfile_, bucket_.c_str(), object_.c_str(),
            "application/octet-stream", Aws::Map<Aws::String, Aws::String>());
    handle->WaitUntilFinished();
    int retries = 0;

    while (handle->GetStatus() == Aws::Transfer::TransferStatus::FAILED &&
           retries++ < kUploadRetries) {
      // if multipart upload was used, only the failed parts will be re-sent
      VLOG(1) << "Retrying Upload of s3://" << bucket_ << "/" << object_
              << " after failure. Current retry count:" << retries;
      transfer_manager_.get()->RetryUpload(outfile_, handle);
      handle->WaitUntilFinished();
    }

    if (handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED) {
      auto error = handle->GetLastError();
      TF_RETURN_IF_ERROR(CheckForbiddenError(error));
      return errors::Unknown(error.GetExceptionName(), ": ",
                             handle->GetFailedParts().size(), " failed parts. ",
                             handle->GetLastError().GetMessage());
    }
    outfile_->clear();
    outfile_->seekp(offset);
    sync_needed_ = false;
    return Status::OK();
  }

 private:
  string bucket_;
  string object_;
  std::shared_ptr<Aws::S3::S3Client> s3_client_;
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;
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

}  // namespace

S3FileSystem::S3FileSystem()
    : s3_client_(nullptr, ShutdownClient),
      initialization_lock_(),
      executor_(nullptr, ShutdownExecutor) {
  
  const char* part_size_str = getenv("S3_MULTI_PART_UPLOAD_CHUNK_SIZE");
  multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD] = kS3MultiPartUploadChunkSize;
  if (part_size_str) {
    uint64 part_size_num;
    if (strings::safe_strtou64(part_size_str, &part_size_num)) {
      multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD] = part_size_num;
    }
  }

  // Different TensorFlow APIs call the download API with different
  // buffer size. Download performance depends on that size and this chunk size.
  part_size_str = getenv("S3_MULTI_PART_DOWNLOAD_CHUNK_SIZE");
  multi_part_chunk_size_[Aws::Transfer::TransferDirection::DOWNLOAD] = kS3MultiPartDownloadChunkSize;
  if (part_size_str) {
    uint64 part_size_num;
    if (strings::safe_strtou64(part_size_str, &part_size_num)) {
      multi_part_chunk_size_[Aws::Transfer::TransferDirection::DOWNLOAD] = part_size_num;
    }
  }

  use_multi_part_download_ = true;
  const char* disable_transfer_mgr = getenv("S3_DISABLE_MULTI_PART_DOWNLOAD");
  if (disable_transfer_mgr) {
   if (disable_transfer_mgr[0] == '1') {
     use_multi_part_download_ = false;
   }
  }
  
  auto upload_pair = 
    std::pair<Aws::Transfer::TransferDirection, 
              std::shared_ptr<Aws::Transfer::TransferManager> > 
             (Aws::Transfer::TransferDirection::UPLOAD,
              std::shared_ptr<Aws::Transfer::TransferManager>
              (nullptr, ShutdownTransferManager));
  auto download_pair = 
    std::pair<Aws::Transfer::TransferDirection, 
              std::shared_ptr<Aws::Transfer::TransferManager> > 
             (Aws::Transfer::TransferDirection::DOWNLOAD,
              std::shared_ptr<Aws::Transfer::TransferManager>
              (nullptr, ShutdownTransferManager));
  
  this->transfer_managers_.insert(upload_pair);
  this->transfer_managers_.insert(download_pair);
}

S3FileSystem::~S3FileSystem() {}

// Initializes s3_client_, if needed, and returns it.
std::shared_ptr<Aws::S3::S3Client> S3FileSystem::GetS3Client() {
  std::lock_guard<mutex> lock(this->initialization_lock_);

  if (this->s3_client_.get() == nullptr) {
    AWSLogSystem::InitializeAWSLogging();

    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256Factory>(AWSCryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256HmacFactory>(AWSCryptoAllocationTag);
    };
    options.cryptoOptions.secureRandomFactory_create_fn = []() {
      return Aws::MakeShared<AWSSecureRandomFactory>(AWSCryptoAllocationTag);
    };
    Aws::InitAPI(options);

    // The creation of S3Client disables virtual addressing:
    //   S3Client(clientConfiguration, signPayloads, useVirtualAddressing =
    //   true)
    // The purpose is to address the issue encountered when there is an `.`
    // in the bucket name. Due to TLS hostname validation or DNS rules,
    // the bucket may not be resolved. Disabling of virtual addressing
    // should address the issue. See GitHub issue 16397 for details.
    this->s3_client_ = std::shared_ptr<Aws::S3::S3Client>(new Aws::S3::S3Client(
        GetDefaultClientConfig(),
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false));
  }

  return this->s3_client_;
}

std::shared_ptr<Aws::Transfer::TransferManager>
S3FileSystem::GetTransferManager(const Aws::Transfer::TransferDirection& direction) {
  std::shared_ptr<Aws::S3::S3Client> s3_client = this->GetS3Client();
  std::lock_guard<mutex> lock(this->initialization_lock_);
  if (this->transfer_managers_[direction].get() == nullptr) {
    Aws::Transfer::TransferManagerConfiguration config(this->GetExecutor().get());
    config.s3Client = s3_client;
    config.bufferSize = this->multi_part_chunk_size_[direction];
    // must be larger than pool size * multi part chunk size
    config.transferBufferMaxHeapSize =
      (kExecutorPoolSize + 1) * this->multi_part_chunk_size_[direction];
    this->transfer_managers_[direction] = Aws::Transfer::TransferManager::Create(config);
  }
  return this->transfer_managers_[direction];
}

std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
S3FileSystem::GetExecutor() {
  if (this->executor_.get() == nullptr) {
    this->executor_ =
        Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
            kExecutorTag, kExecutorPoolSize);
  }
  return this->executor_;
}

Status S3FileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  return NewRandomAccessFile(fname, result, true);
}

Status S3FileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result,
    bool use_multi_part_download) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));

  // check if an override was defined for this file. used for testing
  bool use_mpd = this->use_multi_part_download_ && use_multi_part_download;
  result->reset(new S3RandomAccessFile(
                      bucket, object, use_mpd,
                      this->GetTransferManager(
                        Aws::Transfer::TransferDirection::DOWNLOAD),
                      this->GetS3Client()));
  return Status::OK();
}

Status S3FileSystem::NewWritableFile(const string& fname,
                                     std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3WritableFile(
                      bucket, object, 
                      this->GetTransferManager(
                        Aws::Transfer::TransferDirection::UPLOAD),
                      this->GetS3Client()));

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
  result->reset(new S3WritableFile(
                      bucket, object, 
                      this->GetTransferManager(
                        Aws::Transfer::TransferDirection::UPLOAD),
                      this->GetS3Client()));

  while (true) {
    status = reader->Read(offset, kS3ReadAppendableFileBufferSize, &read_chunk,
                          buffer.get());
    if (status.ok()) {
      (void)(*result)->Append(read_chunk);
      offset += kS3ReadAppendableFileBufferSize;
    } else if (status.code() == error::OUT_OF_RANGE) {
      (void)(*result)->Append(read_chunk);
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
  VLOG(1) << "GetChildren for path: " << dir;
  string bucket, prefix;
  TF_RETURN_IF_ERROR(ParseS3Path(dir, true, &bucket, &prefix));

  if (!prefix.empty() && prefix.back() != '/') {
    prefix.push_back('/');
  }

  Aws::S3::Model::ListObjectsRequest listObjectsRequest;
  listObjectsRequest.WithBucket(bucket.c_str())
      .WithPrefix(prefix.c_str())
      .WithMaxKeys(kS3GetChildrenMaxKeys)
      .WithDelimiter("/");
  listObjectsRequest.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult listObjectsResult;
  do {
    auto listObjectsOutcome =
        this->GetS3Client()->ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      return CreateStatusFromAwsError(listObjectsOutcome.GetError());
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
  VLOG(1) << "Stat on path: " << fname;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, true, &bucket, &object));

  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket.c_str());
    auto headBucketOutcome = this->GetS3Client()->HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      return CreateStatusFromAwsError(headBucketOutcome.GetError());
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
  auto headObjectOutcome = this->GetS3Client()->HeadObject(headObjectRequest);
  if (headObjectOutcome.IsSuccess()) {
    stats->length = headObjectOutcome.GetResult().GetContentLength();
    stats->is_directory = 0;
    stats->mtime_nsec =
        headObjectOutcome.GetResult().GetLastModified().Millis() * 1e6;
    found = true;
  } else {
    TF_RETURN_IF_ERROR(CheckForbiddenError(headObjectOutcome.GetError()));
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
  auto listObjectsOutcome =
      this->GetS3Client()->ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto listObjects = listObjectsOutcome.GetResult().GetContents();
    if (listObjects.size() > 0) {
      stats->length = 0;
      stats->is_directory = 1;
      stats->mtime_nsec = listObjects[0].GetLastModified().Millis() * 1e6;
      found = true;
    }
  } else {
    TF_RETURN_IF_ERROR(CheckForbiddenError(listObjectsOutcome.GetError()));
  }
  if (!found) {
    return errors::NotFound("Object ", fname, " does not exist");
  }
  return Status::OK();
}

Status S3FileSystem::GetMatchingPaths(const string& pattern,
                                      std::vector<string>* results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status S3FileSystem::DeleteFile(const string& fname) {
  VLOG(1) << "DeleteFile: " << fname;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));

  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());

  auto deleteObjectOutcome =
      this->GetS3Client()->DeleteObject(deleteObjectRequest);
  if (!deleteObjectOutcome.IsSuccess()) {
    return CreateStatusFromAwsError(deleteObjectOutcome.GetError());
  }
  return Status::OK();
}

Status S3FileSystem::CreateDir(const string& dirname) {
  VLOG(1) << "CreateDir: " << dirname;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, true, &bucket, &object));

  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest headBucketRequest;
    headBucketRequest.WithBucket(bucket.c_str());
    auto headBucketOutcome = this->GetS3Client()->HeadBucket(headBucketRequest);
    if (!headBucketOutcome.IsSuccess()) {
      TF_RETURN_IF_ERROR(CheckForbiddenError(headBucketOutcome.GetError()));
      return errors::NotFound("The bucket ", bucket, " was not found.");
    }
    return Status::OK();
  }
  string filename = dirname;
  if (filename.back() != '/') {
    filename.push_back('/');
  }
  if (!this->FileExists(filename).ok()) {
    std::unique_ptr<WritableFile> file;
    TF_RETURN_IF_ERROR(NewWritableFile(filename, &file));
    TF_RETURN_IF_ERROR(file->Close());
  }
  return Status::OK();
}

Status S3FileSystem::DeleteDir(const string& dirname) {
  VLOG(1) << "DeleteDir: " << dirname;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, false, &bucket, &object));

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
  auto listObjectsOutcome =
      this->GetS3Client()->ListObjects(listObjectsRequest);
  if (listObjectsOutcome.IsSuccess()) {
    auto contents = listObjectsOutcome.GetResult().GetContents();
    if (contents.size() > 1 ||
        (contents.size() == 1 && contents[0].GetKey() != prefix.c_str())) {
      return errors::Unknown(
          "Cannot delete a non-empty directory. "
          "This operation will be retried in case this "
          "is due to S3's eventual consistency.");
    }
    if (contents.size() == 1 && contents[0].GetKey() == prefix.c_str()) {
      string filename = dirname;
      if (filename.back() != '/') {
        filename.push_back('/');
      }
      return DeleteFile(filename);
    }
  } else {
    TF_RETURN_IF_ERROR(CheckForbiddenError(listObjectsOutcome.GetError()));
  }
  return Status::OK();
}

Status S3FileSystem::GetFileSize(const string& fname, uint64* file_size) {
  FileStatistics stats;
  TF_RETURN_IF_ERROR(this->Stat(fname, &stats));
  *file_size = stats.length;
  return Status::OK();
}

void S3FileSystem::MultiPartCopyCallback(
    const Aws::S3::Model::UploadPartCopyRequest& request,
    const Aws::S3::Model::UploadPartCopyOutcome& uploadPartCopyOutcome,
    const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) {
  std::shared_ptr<tensorflow::MultiPartCopyAsyncContext> multiPartContext =
      std::const_pointer_cast<tensorflow::MultiPartCopyAsyncContext>(
          std::static_pointer_cast<const tensorflow::MultiPartCopyAsyncContext>(
              context));

  {
    std::unique_lock<std::mutex> lock(*multiPartContext->multi_part_copy_mutex);

    Status status;
    if (uploadPartCopyOutcome.IsSuccess()) {
      // success
      Aws::String eTag =
          uploadPartCopyOutcome.GetResult().GetCopyPartResult().GetETag();
      multiPartContext->eTag = eTag;
      status = Status::OK();
    } else {
      LOG(ERROR) << "Error when copying part " << multiPartContext->partNumber
                 << " " << uploadPartCopyOutcome.GetError().GetMessage();
      status =
          errors::Unknown(uploadPartCopyOutcome.GetError().GetExceptionName(),
                          ": ", uploadPartCopyOutcome.GetError().GetMessage());
    }

    (*multiPartContext->finishedPartStates)[multiPartContext->partNumber] =
        multiPartContext->incompletePartStates->at(
            multiPartContext->partNumber);
    multiPartContext->finishedPartStates->at(multiPartContext->partNumber)
        .status = status;
    multiPartContext->incompletePartStates->erase(multiPartContext->partNumber);
    // Notify the thread that started the operation
    multiPartContext->multi_part_copy_cv->notify_one();
  }
}

Status S3FileSystem::CopyFile(const Aws::String& source_bucket,
                              const Aws::String& source_key,
                              const Aws::String& target_bucket,
                              const Aws::String& target_key) {
  Aws::String source = Aws::String((source_bucket + "/" + source_key).c_str());
  Aws::String source_full_path = Aws::String("s3://") + source;
  uint64 file_length;
  TF_RETURN_IF_ERROR(
      this->GetFileSize(string(source_full_path.c_str()), &file_length));
  int num_parts;
  if (file_length <= multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD]) {
    num_parts = 1;
  } else {
    num_parts = ceil((float)file_length / multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD]);
  }

  if (num_parts == 1) {
    return SimpleCopy(source, target_bucket, target_key);
  } else if (num_parts > 10000) {
    string message = strings::StrCat(
        "MultiPartCopy with number of parts more than 10000 is not supported. "
        "Your object ",
        source, " required ", num_parts,
        " as multi_part_copy_part_size is set to ", 
        multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD],
        ". You can control this part size using the environment variable ",
        "S3_MULTI_PART_COPY_PART_SIZE to increase it.");
    return tensorflow::errors::Unimplemented(message);
  } else {
    return MultiPartCopy(source, target_bucket, target_key, num_parts,
                         file_length);
  }
}

Status S3FileSystem::SimpleCopy(const Aws::String& source,
                                const Aws::String& target_bucket,
                                const Aws::String& target_key) {
  VLOG(1) << "SimpleCopy from " << source << " to: " << target_bucket << "/"
          << target_key;
  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  copyObjectRequest.SetBucket(target_bucket.c_str());
  copyObjectRequest.SetKey(target_key);
  copyObjectRequest.SetCopySource(source);
  auto copyObjectOutcome = this->GetS3Client()->CopyObject(copyObjectRequest);
  if (!copyObjectOutcome.IsSuccess()) {
    return CreateStatusFromAwsError(copyObjectOutcome.GetError());
  }
  return Status::OK();
}

Status S3FileSystem::MultiPartCopy(const Aws::String& source,
                                   const Aws::String& target_bucket,
                                   const Aws::String& target_key,
                                   const int num_parts,
                                   const uint64 file_length) {
  VLOG(1) << "MultiPartCopy from " << source << " to: " << target_bucket << "/"
          << target_key;
  Aws::S3::Model::CreateMultipartUploadRequest multipartUploadRequest;
  multipartUploadRequest.SetBucket(target_bucket);
  multipartUploadRequest.SetKey(target_key);

  auto multipartUploadOutcome =
      this->GetS3Client()->CreateMultipartUpload(multipartUploadRequest);
  if (!multipartUploadOutcome.IsSuccess()) {
    return CreateStatusFromAwsError(multipartUploadOutcome.GetError());
  }

  Aws::String uploadID = multipartUploadOutcome.GetResult().GetUploadId();
  VLOG(1) << "Copying from " << source << " in " << num_parts
          << " parts of size "
          << multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD]
          << " each";
  Aws::S3::Model::CompletedMultipartUpload completedMPURequest;

  // passed to each callback keyed by partNumber
  std::map<int, std::shared_ptr<tensorflow::MultiPartCopyAsyncContext>>
      partContexts;
  // keeps track of incompleteParts keyed by partNumber
  std::map<int, PartState> incompletePartStates;
  // S3 API partNumber starts from 1
  for (int partNumber = 1; partNumber <= num_parts; partNumber++) {
    PartState ps;
    ps.partNumber = partNumber;
    incompletePartStates[partNumber] = ps;
  }

  // keeps track of completed parts keyed by partNumber
  std::map<int, PartState> finishedPartStates;
  // mutex which protects access of the partStates map
  std::mutex multi_part_copy_mutex;
  // condition variable to be used with above mutex for synchronization
  std::condition_variable multi_part_copy_cv;

  int retry_count_ = 3;
  while (retry_count_-- > 0) {
    // queue up parts
    for (std::map<int, PartState>::iterator it = incompletePartStates.begin();
         it != incompletePartStates.end(); it++) {
      int partNumber = it->first;
      uint64 startPos = (partNumber - 1) * multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD];
      uint64 endPos = startPos + multi_part_chunk_size_[Aws::Transfer::TransferDirection::UPLOAD] - 1;
      if (endPos >= file_length) {
        endPos = file_length - 1;
      }

      string range = strings::StrCat("bytes=", startPos, "-", endPos);

      Aws::S3::Model::UploadPartCopyRequest uploadPartCopyRequest;
      uploadPartCopyRequest.SetBucket(target_bucket);
      uploadPartCopyRequest.SetKey(target_key);
      uploadPartCopyRequest.SetCopySource(source.c_str());
      uploadPartCopyRequest.SetCopySourceRange(range.c_str());
      uploadPartCopyRequest.SetPartNumber(partNumber);
      uploadPartCopyRequest.SetUploadId(uploadID);

      auto multiPartContext =
          Aws::MakeShared<tensorflow::MultiPartCopyAsyncContext>(
              "MultiPartCopyContext");

      multiPartContext->partNumber = partNumber;
      multiPartContext->incompletePartStates = &incompletePartStates;
      multiPartContext->finishedPartStates = &finishedPartStates;
      multiPartContext->multi_part_copy_mutex = &multi_part_copy_mutex;
      multiPartContext->multi_part_copy_cv = &multi_part_copy_cv;

      // replace with current context
      partContexts[partNumber] = multiPartContext;

      auto callback =
          [this](const Aws::S3::S3Client* client,
                 const Aws::S3::Model::UploadPartCopyRequest& request,
                 const Aws::S3::Model::UploadPartCopyOutcome& outcome,
                 const std::shared_ptr<const Aws::Client::AsyncCallerContext>&
                     context) {
            this->MultiPartCopyCallback(request, outcome, context);
          };

      this->GetS3Client()->UploadPartCopyAsync(uploadPartCopyRequest, callback,
                                               multiPartContext);
    }
    // wait till they finish
    {
      std::unique_lock<std::mutex> lock(multi_part_copy_mutex);
      // wait on the mutex until notify is called
      // then check the finished parts as there could be false notifications
      multi_part_copy_cv.wait(lock, [&finishedPartStates, num_parts] {
        return finishedPartStates.size() == num_parts;
      });
    }
    // check if there was any error for any part
    for (int partNumber = 1; partNumber <= num_parts; partNumber++) {
      if (finishedPartStates[partNumber].status != Status::OK()) {
        if (retry_count_ <= 0) {
          if (finishedPartStates[partNumber].status != Status::OK()) {
            TF_RETURN_IF_ERROR(
                AbortMultiPartCopy(target_bucket, target_key, uploadID));
            return finishedPartStates[partNumber].status;
          }
        } else {
          // retry part
          LOG(ERROR) << "Retrying failed copy of part " << partNumber
                     << " due to an error with S3. ";
          PartState ps;
          ps.partNumber = partNumber;
          incompletePartStates[partNumber] = ps;
          finishedPartStates.erase(partNumber);
        }
      }
    }
  }

  // if there was an error still in any part, it would abort and return in the
  // above loop set the eTag of completed Part to the final CompletedMPURequest
  // note these parts have to be added in order
  for (int partNumber = 1; partNumber <= num_parts; partNumber++) {
    Aws::S3::Model::CompletedPart completedPart;
    completedPart.SetPartNumber(partNumber);
    completedPart.SetETag(partContexts[partNumber]->eTag);
    completedMPURequest.AddParts(completedPart);
  }

  Status finalStatus = CompleteMultiPartCopy(target_bucket, target_key,
                                             uploadID, completedMPURequest);
  if (finalStatus != Status::OK()) {
    TF_RETURN_IF_ERROR(AbortMultiPartCopy(target_bucket, target_key, uploadID));
  }
  return finalStatus;
}

Status S3FileSystem::AbortMultiPartCopy(Aws::String target_bucket,
                                        Aws::String target_key,
                                        Aws::String uploadID) {
  Aws::S3::Model::AbortMultipartUploadRequest abortRequest;
  abortRequest.WithBucket(target_bucket)
      .WithKey(target_key)
      .WithUploadId(uploadID);
  auto abortOutcome = this->GetS3Client()->AbortMultipartUpload(abortRequest);
  if (!abortOutcome.IsSuccess()) {
    return CreateStatusFromAwsError(abortOutcome.GetError());
  }
  return Status::OK();
}

Status S3FileSystem::CompleteMultiPartCopy(
    Aws::String target_bucket, Aws::String target_key, Aws::String uploadID,
    Aws::S3::Model::CompletedMultipartUpload completedMPURequest) {
  Aws::S3::Model::CompleteMultipartUploadRequest completeRequest;
  completeRequest.SetBucket(target_bucket);
  completeRequest.SetKey(target_key);
  completeRequest.SetUploadId(uploadID);
  completeRequest.SetMultipartUpload(completedMPURequest);
  auto completeOutcome =
      this->GetS3Client()->CompleteMultipartUpload(completeRequest);
  if (!completeOutcome.IsSuccess()) {
    return CreateStatusFromAwsError(completeOutcome.GetError());
  }
  return Status::OK();
}

Status S3FileSystem::RenameFile(const string& src, const string& target) {
  VLOG(1) << "RenameFile from: " << src << " to: " << target;
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
    auto listObjectsOutcome =
        this->GetS3Client()->ListObjects(listObjectsRequest);
    if (!listObjectsOutcome.IsSuccess()) {
      return CreateStatusFromAwsError(listObjectsOutcome.GetError());
    }

    listObjectsResult = listObjectsOutcome.GetResult();
    for (const auto& object : listObjectsResult.GetContents()) {
      Aws::String src_key = object.GetKey();
      Aws::String target_key = src_key;
      target_key.replace(0, src_object.length(), target_object.c_str());

      TF_RETURN_IF_ERROR(CopyFile(Aws::String(src_bucket.c_str()), src_key,
                                  Aws::String(target_bucket.c_str()),
                                  target_key));

      deleteObjectRequest.SetBucket(src_bucket.c_str());
      deleteObjectRequest.SetKey(src_key.c_str());

      auto deleteObjectOutcome =
          this->GetS3Client()->DeleteObject(deleteObjectRequest);
      if (!deleteObjectOutcome.IsSuccess()) {
        return CreateStatusFromAwsError(deleteObjectOutcome.GetError());
      }
    }
    listObjectsRequest.SetMarker(listObjectsResult.GetNextMarker());
  } while (listObjectsResult.GetIsTruncated());

  return Status::OK();
}

Status S3FileSystem::HasAtomicMove(const string& path, bool* has_atomic_move) {
  *has_atomic_move = false;
  return Status::OK();
}

REGISTER_FILE_SYSTEM("s3", RetryingS3FileSystem);

}  // namespace tensorflow
