/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_filesystem.h"

#include <aws/core/client/AsyncCallerContext.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/utils/FileSystemUtils.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/UploadPartCopyRequest.h>
#include <stdlib.h>
#include <string.h>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/aws_crypto.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/aws_logging.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for S3 environments.
// This filesystem will support `s3://` URI schemes.
constexpr char kS3FileSystemAllocationTag[] = "S3FileSystemAllocation";
constexpr char kS3ClientAllocationTag[] = "S3ClientAllocation";
constexpr int64_t kS3TimeoutMsec = 300000;  // 5 min
constexpr int kS3GetChildrenMaxKeys = 100;

constexpr char kExecutorTag[] = "TransferManagerExecutorAllocation";
constexpr int kExecutorPoolSize = 25;

constexpr uint64_t kS3MultiPartUploadChunkSize = 50 * 1024 * 1024;    // 50 MB
constexpr uint64_t kS3MultiPartDownloadChunkSize = 50 * 1024 * 1024;  // 50 MB
constexpr size_t kDownloadRetries = 3;
constexpr size_t kUploadRetries = 3;

constexpr size_t kS3ReadAppendableFileBufferSize = 1024 * 1024;  // 1 MB

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

static inline void TF_SetStatusFromAWSError(
    const Aws::Client::AWSError<Aws::S3::S3Errors>& error, TF_Status* status) {
  switch (error.GetResponseCode()) {
    case Aws::Http::HttpResponseCode::FORBIDDEN:
      TF_SetStatus(status, TF_FAILED_PRECONDITION,
                   "AWS Credentials have not been set properly. "
                   "Unable to access the specified S3 location");
      break;
    case Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE:
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read less bytes than requested");
      break;
    case Aws::Http::HttpResponseCode::NOT_FOUND:
      TF_SetStatus(status, TF_NOT_FOUND, error.GetMessage().c_str());
      break;
    default:
      TF_SetStatus(
          status, TF_UNKNOWN,
          (error.GetExceptionName() + ": " + error.GetMessage()).c_str());
      break;
  }
}

void ParseS3Path(const Aws::String& fname, bool object_empty_ok,
                 Aws::String* bucket, Aws::String* object, TF_Status* status) {
  size_t scheme_end = fname.find("://") + 2;
  if (fname.substr(0, scheme_end + 1) != "s3://") {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't start with 's3://'.");
    return;
  }

  size_t bucket_end = fname.find("/", scheme_end + 1);
  if (bucket_end == std::string::npos) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't contain a bucket name.");
    return;
  }

  *bucket = fname.substr(scheme_end + 1, bucket_end - scheme_end - 1);
  *object = fname.substr(bucket_end + 1);

  if (object->empty() && !object_empty_ok) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't contain an object name.");
  }
}

static Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  ABSL_CONST_INIT static absl::Mutex cfg_lock(absl::kConstInit);
  static bool init(false);
  static Aws::Client::ClientConfiguration cfg;

  absl::MutexLock l(&cfg_lock);

  if (!init) {
    const char* endpoint = getenv("S3_ENDPOINT");
    if (endpoint) cfg.endpointOverride = Aws::String(endpoint);
    const char* region = getenv("AWS_REGION");
    // TODO (yongtang): `S3_REGION` should be deprecated after 2.0.
    if (!region) region = getenv("S3_REGION");
    if (region) {
      cfg.region = Aws::String(region);
    } else {
      // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
      // is set with a truthy value.
      const char* load_config_env = getenv("AWS_SDK_LOAD_CONFIG");
      std::string load_config =
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
        if (!profiles["default"].GetRegion().empty())
          cfg.region = profiles["default"].GetRegion();
      }
    }
    const char* use_https = getenv("S3_USE_HTTPS");
    if (use_https) {
      if (use_https[0] == '0')
        cfg.scheme = Aws::Http::Scheme::HTTP;
      else
        cfg.scheme = Aws::Http::Scheme::HTTPS;
    }
    const char* verify_ssl = getenv("S3_VERIFY_SSL");
    if (verify_ssl) {
      if (verify_ssl[0] == '0')
        cfg.verifySSL = false;
      else
        cfg.verifySSL = true;
    }
    // if these timeouts are low, you may see an error when
    // uploading/downloading large files: Unable to connect to endpoint
    int64_t timeout;
    cfg.connectTimeoutMs =
        absl::SimpleAtoi(getenv("S3_CONNECT_TIMEOUT_MSEC"), &timeout)
            ? timeout
            : kS3TimeoutMsec;
    cfg.requestTimeoutMs =
        absl::SimpleAtoi(getenv("S3_REQUEST_TIMEOUT_MSEC"), &timeout)
            ? timeout
            : kS3TimeoutMsec;
    const char* ca_file = getenv("S3_CA_FILE");
    if (ca_file) cfg.caFile = Aws::String(ca_file);
    const char* ca_path = getenv("S3_CA_PATH");
    if (ca_path) cfg.caPath = Aws::String(ca_path);
    init = true;
  }
  return cfg;
};

static void GetS3Client(tf_s3_filesystem::S3File* s3_file) {
  absl::MutexLock l(&s3_file->initialization_lock);

  if (s3_file->s3_client.get() == nullptr) {
    tf_s3_filesystem::AWSLogSystem::InitializeAWSLogging();

    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<tf_s3_filesystem::AWSSHA256Factory>(
          tf_s3_filesystem::AWSCryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<tf_s3_filesystem::AWSSHA256HmacFactory>(
          tf_s3_filesystem::AWSCryptoAllocationTag);
    };
    options.cryptoOptions.secureRandomFactory_create_fn = []() {
      return Aws::MakeShared<tf_s3_filesystem::AWSSecureRandomFactory>(
          tf_s3_filesystem::AWSCryptoAllocationTag);
    };
    Aws::InitAPI(options);

    // The creation of S3Client disables virtual addressing:
    //   S3Client(clientConfiguration, signPayloads, useVirtualAddressing =
    //   true)
    // The purpose is to address the issue encountered when there is an `.`
    // in the bucket name. Due to TLS hostname validation or DNS rules,
    // the bucket may not be resolved. Disabling of virtual addressing
    // should address the issue. See GitHub issue 16397 for details.
    s3_file->s3_client = Aws::MakeShared<Aws::S3::S3Client>(
        kS3ClientAllocationTag, GetDefaultClientConfig(),
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);
  }
}

static void GetExecutor(tf_s3_filesystem::S3File* s3_file) {
  absl::MutexLock l(&s3_file->initialization_lock);

  if (s3_file->executor.get() == nullptr) {
    s3_file->executor =
        Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
            kExecutorTag, kExecutorPoolSize);
  }
}

static void GetTransferManager(
    const Aws::Transfer::TransferDirection& direction,
    tf_s3_filesystem::S3File* s3_file) {
  // These functions should be called before holding `initialization_lock`.
  GetS3Client(s3_file);
  GetExecutor(s3_file);

  absl::MutexLock l(&s3_file->initialization_lock);

  if (s3_file->transfer_managers[direction].get() == nullptr) {
    Aws::Transfer::TransferManagerConfiguration config(s3_file->executor.get());
    config.s3Client = s3_file->s3_client;
    config.bufferSize = s3_file->multi_part_chunk_sizes[direction];
    // must be larger than pool size * multi part chunk size
    config.transferBufferMaxHeapSize =
        (kExecutorPoolSize + 1) * s3_file->multi_part_chunk_sizes[direction];
    s3_file->transfer_managers[direction] =
        Aws::Transfer::TransferManager::Create(config);
  }
}

static void ShutdownClient(Aws::S3::S3Client* s3_client) {
  if (s3_client != nullptr) {
    delete s3_client;
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
    tf_s3_filesystem::AWSLogSystem::ShutdownAWSLogging();
  }
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
typedef struct S3File {
  Aws::String bucket;
  Aws::String object;
  std::shared_ptr<Aws::S3::S3Client> s3_client;
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager;
  bool use_multi_part_download;
} S3File;

// AWS Streams destroy the buffer (buf) passed, so creating a new
// IOStream that retains the buffer so the calling function
// can control it's lifecycle
class TFS3UnderlyingStream : public Aws::IOStream {
 public:
  using Base = Aws::IOStream;
  TFS3UnderlyingStream(std::streambuf* buf) : Base(buf) {}
  virtual ~TFS3UnderlyingStream() = default;
};

void Cleanup(TF_RandomAccessFile* file) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  delete s3_file;
}

static int64_t ReadS3Client(S3File* s3_file, uint64_t offset, size_t n,
                            char* buffer, TF_Status* status) {
  TF_VLog(3, "ReadFile using S3Client\n");
  Aws::S3::Model::GetObjectRequest get_object_request;
  get_object_request.WithBucket(s3_file->bucket).WithKey(s3_file->object);
  Aws::String bytes =
      absl::StrCat("bytes=", offset, "-", offset + n - 1).c_str();
  get_object_request.SetRange(bytes);
  get_object_request.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  auto get_object_outcome = s3_file->s3_client->GetObject(get_object_request);
  if (!get_object_outcome.IsSuccess())
    TF_SetStatusFromAWSError(get_object_outcome.GetError(), status);
  else
    TF_SetStatus(status, TF_OK, "");
  if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_OUT_OF_RANGE)
    return -1;

  int64_t read = get_object_outcome.GetResult().GetContentLength();
  if (read < n)
    TF_SetStatus(status, TF_OUT_OF_RANGE, "Read less bytes than requested");
  get_object_outcome.GetResult().GetBody().read(buffer, read);
  return read;
}

static int64_t ReadS3TransferManager(S3File* s3_file, uint64_t offset, size_t n,
                                     char* buffer, TF_Status* status) {
  TF_VLog(3, "Using TransferManager\n");
  auto create_download_stream = [&]() {
    return Aws::New<TFS3UnderlyingStream>(
        "S3ReadStream",
        Aws::New<Aws::Utils::Stream::PreallocatedStreamBuf>(
            "S3ReadStream", reinterpret_cast<unsigned char*>(buffer), n));
  };
  TF_VLog(3, "Created stream to read with transferManager\n");
  auto handle = s3_file->transfer_manager->DownloadFile(
      s3_file->bucket, s3_file->object, offset, n, create_download_stream);
  handle->WaitUntilFinished();

  size_t retries = 0;
  while (handle->GetStatus() == Aws::Transfer::TransferStatus::FAILED &&
         handle->GetLastError().GetResponseCode() !=
             Aws::Http::HttpResponseCode::REQUESTED_RANGE_NOT_SATISFIABLE &&
         retries++ < kDownloadRetries) {
    // Only failed parts will be downloaded again.
    TF_VLog(
        1,
        "Retrying read of s3://%s/%s after failure. Current retry count: %u\n",
        s3_file->bucket.c_str(), s3_file->object.c_str(), retries);
    s3_file->transfer_manager->RetryDownload(handle);
    handle->WaitUntilFinished();
  }

  if (handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED)
    TF_SetStatusFromAWSError(handle->GetLastError(), status);
  else
    TF_SetStatus(status, TF_OK, "");
  if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_OUT_OF_RANGE)
    return -1;
  int64_t read = handle->GetBytesTransferred();
  if (read < n)
    TF_SetStatus(status, TF_OUT_OF_RANGE, "Read less bytes than requested");
  return read;
}

int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
             char* buffer, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  TF_VLog(1, "ReadFilefromS3 s3://%s/%s from %u for n: %u\n",
          s3_file->bucket.c_str(), s3_file->object.c_str(), offset, n);
  if (s3_file->use_multi_part_download)
    return ReadS3TransferManager(s3_file, offset, n, buffer, status);
  else
    return ReadS3Client(s3_file, offset, n, buffer, status);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
typedef struct S3File {
  Aws::String bucket;
  Aws::String object;
  std::shared_ptr<Aws::S3::S3Client> s3_client;
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager;
  bool sync_needed;
  std::shared_ptr<Aws::Utils::TempFile> outfile;
  S3File(Aws::String bucket, Aws::String object,
         std::shared_ptr<Aws::S3::S3Client> s3_client,
         std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager)
      : bucket(bucket),
        object(object),
        s3_client(s3_client),
        transfer_manager(transfer_manager),
        outfile(Aws::MakeShared<Aws::Utils::TempFile>(
            kS3FileSystemAllocationTag, nullptr, "_s3_filesystem_XXXXXX",
            std::ios_base::binary | std::ios_base::trunc | std::ios_base::in |
                std::ios_base::out)) {}
} S3File;

void Cleanup(TF_WritableFile* file) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  delete s3_file;
}

void Append(const TF_WritableFile* file, const char* buffer, size_t n,
            TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (!s3_file->outfile) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "The internal temporary file is not writable.");
    return;
  }
  s3_file->sync_needed = true;
  s3_file->outfile->write(buffer, n);
  if (!s3_file->outfile->good())
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not append to the internal temporary file.");
  else
    TF_SetStatus(status, TF_OK, "");
}

int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  auto position = static_cast<int64_t>(s3_file->outfile->tellp());
  if (position == -1)
    TF_SetStatus(status, TF_INTERNAL,
                 "tellp on the internal temporary file failed");
  else
    TF_SetStatus(status, TF_OK, "");
  return position;
}

void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (!s3_file->outfile) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "The internal temporary file is not writable.");
    return;
  }
  if (!s3_file->sync_needed) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  TF_VLog(1, "WriteFileToS3: s3://%s/%s\n", s3_file->bucket.c_str(),
          s3_file->object.c_str());
  auto position = static_cast<int64_t>(s3_file->outfile->tellp());
  auto handle = s3_file->transfer_manager->UploadFile(
      s3_file->outfile, s3_file->bucket, s3_file->object,
      "application/octet-stream", Aws::Map<Aws::String, Aws::String>());
  handle->WaitUntilFinished();

  size_t retries = 0;
  while (handle->GetStatus() == Aws::Transfer::TransferStatus::FAILED &&
         retries++ < kUploadRetries) {
    // if multipart upload was used, only the failed parts will be re-sent
    TF_VLog(1,
            "Retrying upload of s3://%s/%s after failure. Current retry count: "
            "%u\n",
            s3_file->bucket.c_str(), s3_file->object.c_str(), retries);
    s3_file->transfer_manager->RetryUpload(s3_file->outfile, handle);
    handle->WaitUntilFinished();
  }
  if (handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED)
    return TF_SetStatusFromAWSError(handle->GetLastError(), status);
  s3_file->outfile->clear();
  s3_file->outfile->seekp(position);
  s3_file->sync_needed = false;
  TF_SetStatus(status, TF_OK, "");
}

void Flush(const TF_WritableFile* file, TF_Status* status) {
  Sync(file, status);
}

void Close(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (s3_file->outfile) {
    Sync(file, status);
    if (TF_GetCode(status) != TF_OK) return;
    s3_file->outfile.reset();
  }
  TF_SetStatus(status, TF_OK, "");
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {
typedef struct S3MemoryRegion {
  std::unique_ptr<char[]> data;
  uint64_t length;
} S3MemoryRegion;

void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  delete r;
}

const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  return reinterpret_cast<const void*>(r->data.get());
}

uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_s3_filesystem {
S3File::S3File()
    : s3_client(nullptr, ShutdownClient),
      executor(nullptr),
      transfer_managers(),
      multi_part_chunk_sizes(),
      use_multi_part_download(true),
      initialization_lock() {
  uint64_t temp_value;
  multi_part_chunk_sizes[Aws::Transfer::TransferDirection::UPLOAD] =
      absl::SimpleAtoi(getenv("S3_MULTI_PART_UPLOAD_CHUNK_SIZE"), &temp_value)
          ? temp_value
          : kS3MultiPartUploadChunkSize;
  multi_part_chunk_sizes[Aws::Transfer::TransferDirection::DOWNLOAD] =
      absl::SimpleAtoi(getenv("S3_MULTI_PART_DOWNLOAD_CHUNK_SIZE"), &temp_value)
          ? temp_value
          : kS3MultiPartDownloadChunkSize;
  use_multi_part_download =
      absl::SimpleAtoi(getenv("S3_DISABLE_MULTI_PART_DOWNLOAD"), &temp_value)
          ? (temp_value != 1)
          : use_multi_part_download;
  transfer_managers.emplace(Aws::Transfer::TransferDirection::UPLOAD, nullptr);
  transfer_managers.emplace(Aws::Transfer::TransferDirection::DOWNLOAD,
                            nullptr);
}
void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new S3File();
  TF_SetStatus(status, TF_OK, "");
}

void Cleanup(TF_Filesystem* filesystem) {
  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  delete s3_file;
}

void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status) {
  Aws::String bucket, object;
  ParseS3Path(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);
  GetTransferManager(Aws::Transfer::TransferDirection::DOWNLOAD, s3_file);
  file->plugin_file = new tf_random_access_file::S3File(
      {bucket, object, s3_file->s3_client,
       s3_file->transfer_managers[Aws::Transfer::TransferDirection::DOWNLOAD],
       s3_file->use_multi_part_download});
  TF_SetStatus(status, TF_OK, "");
}

void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status) {
  Aws::String bucket, object;
  ParseS3Path(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);
  GetTransferManager(Aws::Transfer::TransferDirection::UPLOAD, s3_file);
  file->plugin_file = new tf_writable_file::S3File(
      bucket, object, s3_file->s3_client,
      s3_file->transfer_managers[Aws::Transfer::TransferDirection::UPLOAD]);
  TF_SetStatus(status, TF_OK, "");
}

void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status) {
  Aws::String bucket, object;
  ParseS3Path(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);
  GetTransferManager(Aws::Transfer::TransferDirection::UPLOAD, s3_file);

  // We need to delete `file->plugin_file` in case of errors. We set
  // `file->plugin_file` to `nullptr` in order to avoid segment fault when
  // calling deleter of `unique_ptr`.
  file->plugin_file = nullptr;
  std::unique_ptr<TF_WritableFile, void (*)(TF_WritableFile*)> writer(
      file, [](TF_WritableFile* file) {
        if (file != nullptr && file->plugin_file != nullptr) {
          tf_writable_file::Cleanup(file);
        }
      });
  writer->plugin_file = new tf_writable_file::S3File(
      bucket, object, s3_file->s3_client,
      s3_file->transfer_managers[Aws::Transfer::TransferDirection::UPLOAD]);
  TF_SetStatus(status, TF_OK, "");

  // Wraping inside a `std::unique_ptr` to prevent memory-leaking.
  std::unique_ptr<TF_RandomAccessFile, void (*)(TF_RandomAccessFile*)> reader(
      new TF_RandomAccessFile, [](TF_RandomAccessFile* file) {
        if (file != nullptr) {
          if (file->plugin_file != nullptr)
            tf_random_access_file::Cleanup(file);
          delete file;
        }
      });
  // We set `reader->plugin_file` to `nullptr` in order to avoid segment fault
  // when calling deleter of `unique_ptr`
  reader->plugin_file = nullptr;
  NewRandomAccessFile(filesystem, path, reader.get(), status);
  if (TF_GetCode(status) != TF_OK) return;

  uint64_t offset = 0;
  std::string buffer(kS3ReadAppendableFileBufferSize, {});
  while (true) {
    auto read = tf_random_access_file::Read(reader.get(), offset,
                                            kS3ReadAppendableFileBufferSize,
                                            &buffer[0], status);
    if (TF_GetCode(status) == TF_NOT_FOUND) {
      break;
    } else if (TF_GetCode(status) == TF_OK) {
      offset += read;
      tf_writable_file::Append(file, buffer.c_str(), read, status);
      if (TF_GetCode(status) != TF_OK) return;
    } else if (TF_GetCode(status) == TF_OUT_OF_RANGE) {
      offset += read;
      tf_writable_file::Append(file, buffer.c_str(), read, status);
      if (TF_GetCode(status) != TF_OK) return;
      break;
    } else {
      return;
    }
  }
  writer.release();
  TF_SetStatus(status, TF_OK, "");
}

void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status) {
  TF_VLog(1, "Stat on path: %s\n", path);
  Aws::String bucket, object;
  ParseS3Path(path, true, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;
  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);

  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest head_bucket_request;
    head_bucket_request.WithBucket(bucket);
    auto head_bucket_outcome =
        s3_file->s3_client->HeadBucket(head_bucket_request);
    if (!head_bucket_outcome.IsSuccess())
      return TF_SetStatusFromAWSError(head_bucket_outcome.GetError(), status);
    stats->length = 0;
    stats->is_directory = 1;
    stats->mtime_nsec = 0;
    return TF_SetStatus(status, TF_OK, "");
  }

  bool found = false;
  Aws::S3::Model::HeadObjectRequest head_object_request;
  head_object_request.WithBucket(bucket).WithKey(object);
  head_object_request.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto head_object_outcome =
      s3_file->s3_client->HeadObject(head_object_request);
  if (head_object_outcome.IsSuccess()) {
    stats->length = head_object_outcome.GetResult().GetContentLength();
    stats->is_directory = 0;
    stats->mtime_nsec =
        head_object_outcome.GetResult().GetLastModified().Millis() * 1e6;
    found = true;
  } else {
    TF_SetStatusFromAWSError(head_object_outcome.GetError(), status);
    if (TF_GetCode(status) == TF_FAILED_PRECONDITION) return;
  }

  auto prefix = object;
  if (prefix.back() != '/') {
    prefix.push_back('/');
  }
  Aws::S3::Model::ListObjectsRequest list_objects_request;
  list_objects_request.WithBucket(bucket).WithPrefix(prefix).WithMaxKeys(1);
  list_objects_request.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto list_objects_outcome =
      s3_file->s3_client->ListObjects(list_objects_request);
  if (list_objects_outcome.IsSuccess()) {
    auto objects = list_objects_outcome.GetResult().GetContents();
    if (objects.size() > 0) {
      stats->length = 0;
      stats->is_directory = 1;
      stats->mtime_nsec = objects[0].GetLastModified().Millis() * 1e6;
      found = true;
    }
  } else {
    TF_SetStatusFromAWSError(list_objects_outcome.GetError(), status);
    if (TF_GetCode(status) == TF_FAILED_PRECONDITION) return;
  }
  if (!found)
    return TF_SetStatus(
        status, TF_NOT_FOUND,
        absl::StrCat("Object ", path, " does not exist").c_str());
  TF_SetStatus(status, TF_OK, "");
}

void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
}

int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
  return stats.length;
}

void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                     const char* path,
                                     TF_ReadOnlyMemoryRegion* region,
                                     TF_Status* status) {
  Aws::String bucket, object;
  ParseS3Path(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);
  GetTransferManager(Aws::Transfer::TransferDirection::UPLOAD, s3_file);

  auto size = GetFileSize(filesystem, path, status);
  if (TF_GetCode(status) != TF_OK) return;
  if (size == 0)
    return TF_SetStatus(status, TF_INVALID_ARGUMENT, "File is empty");

  std::unique_ptr<char[]> data(new char[size]);
  // Wraping inside a `std::unique_ptr` to prevent memory-leaking.
  std::unique_ptr<TF_RandomAccessFile, void (*)(TF_RandomAccessFile*)> reader(
      new TF_RandomAccessFile, [](TF_RandomAccessFile* file) {
        if (file != nullptr) {
          if (file->plugin_file != nullptr)
            tf_random_access_file::Cleanup(file);
          delete file;
        }
      });
  // We set `reader->plugin_file` to `nullptr` in order to avoid segment fault
  // when calling deleter of `unique_ptr`
  reader->plugin_file = nullptr;
  NewRandomAccessFile(filesystem, path, reader.get(), status);
  if (TF_GetCode(status) != TF_OK) return;
  auto read =
      tf_random_access_file::Read(reader.get(), 0, size, data.get(), status);
  if (TF_GetCode(status) != TF_OK) return;

  region->plugin_memory_region = new tf_read_only_memory_region::S3MemoryRegion(
      {std::move(data), static_cast<uint64_t>(read)});
  TF_SetStatus(status, TF_OK, "");
}

static void SimpleCopyFile(const Aws::String& source,
                           const Aws::String& bucket_dst,
                           const Aws::String& object_dst, S3File* s3_file,
                           TF_Status* status) {
  TF_VLog(1, "SimpleCopyFile from %s to %s/%s\n", bucket_dst.c_str(),
          object_dst.c_str());
  Aws::S3::Model::CopyObjectRequest copy_object_request;
  copy_object_request.WithCopySource(source)
      .WithBucket(bucket_dst)
      .WithKey(object_dst);
  auto copy_object_outcome =
      s3_file->s3_client->CopyObject(copy_object_request);
  if (!copy_object_outcome.IsSuccess())
    TF_SetStatusFromAWSError(copy_object_outcome.GetError(), status);
  else
    TF_SetStatus(status, TF_OK, "");
};

using EtagOutcome =
    Aws::Utils::Outcome<Aws::String, Aws::Client::AWSError<Aws::S3::S3Errors>>;
typedef struct MultipartCopyAsyncContext
    : public Aws::Client::AsyncCallerContext {
  int part_number;
  int* num_finished_parts;
  Aws::Vector<EtagOutcome>* etag_outcomes;

  // lock and cv for multi part copy
  absl::Mutex* multi_part_copy_mutex;
  absl::CondVar* multi_part_copy_cv;
} MultipartCopyAsyncContext;

static void AbortMultiPartCopy(const Aws::String& bucket_dst,
                               const Aws::String& object_dst,
                               const Aws::String& upload_id, S3File* s3_file,
                               TF_Status* status) {
  Aws::S3::Model::AbortMultipartUploadRequest request;
  request.WithBucket(bucket_dst).WithKey(object_dst).WithUploadId(upload_id);
  auto outcome = s3_file->s3_client->AbortMultipartUpload(request);
  if (!outcome.IsSuccess())
    TF_SetStatusFromAWSError(outcome.GetError(), status);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void MultiPartCopyCallback(
    const Aws::S3::Model::UploadPartCopyRequest& request,
    const Aws::S3::Model::UploadPartCopyOutcome& outcome,
    const std::shared_ptr<const MultipartCopyAsyncContext>& context) {
  // Access to `etag_outcomes` should be thread-safe because of distinct
  // `part_number`.
  auto part_number = context->part_number;
  auto etag_outcomes = context->etag_outcomes;
  if (outcome.IsSuccess()) {
    (*etag_outcomes)[part_number] =
        outcome.GetResult().GetCopyPartResult().GetETag();
  } else {
    (*etag_outcomes)[part_number] = outcome.GetError();
  }
  {
    absl::MutexLock l(context->multi_part_copy_mutex);
    (*context->num_finished_parts)++;
    context->multi_part_copy_cv->Signal();
  }
}

static void MultiPartCopy(const Aws::String& source,
                          const Aws::String& bucket_dst,
                          const Aws::String& object_dst, const size_t num_parts,
                          const uint64_t file_size, S3File* s3_file,
                          TF_Status* status) {
  TF_VLog(1, "MultiPartCopy from %s to %s/%s\n", bucket_dst.c_str(),
          object_dst.c_str());
  Aws::S3::Model::CreateMultipartUploadRequest create_multipart_upload_request;
  create_multipart_upload_request.WithBucket(bucket_dst).WithKey(object_dst);

  GetS3Client(s3_file);
  GetTransferManager(Aws::Transfer::TransferDirection::UPLOAD, s3_file);

  auto create_multipart_upload_outcome =
      s3_file->s3_client->CreateMultipartUpload(
          create_multipart_upload_request);
  if (!create_multipart_upload_outcome.IsSuccess())
    return TF_SetStatusFromAWSError(create_multipart_upload_outcome.GetError(),
                                    status);

  auto upload_id = create_multipart_upload_outcome.GetResult().GetUploadId();

  int num_finished_parts = 0;
  // Keep track of `Outcome` of each upload part.
  Aws::Vector<EtagOutcome> etag_outcomes(num_parts);
  // Mutex which protects access of the part_states map.
  absl::Mutex multi_part_copy_mutex;
  // Condition variable to be used with above mutex for synchronization.
  absl::CondVar multi_part_copy_cv;

  auto chunk_size =
      s3_file->multi_part_chunk_sizes[Aws::Transfer::TransferDirection::UPLOAD];

  TF_VLog(1, "Copying from %s in %u parts of size %u each\n", source.c_str(),
          num_parts, chunk_size);
  size_t retries = 0;
  while (retries++ < 3) {
    // Queue up parts.
    for (auto part_number = 0; part_number < num_parts; ++part_number) {
      if (etag_outcomes[part_number].IsSuccess()) continue;
      uint64_t start_pos = part_number * chunk_size;
      uint64_t end_pos = start_pos + chunk_size - 1;
      if (end_pos >= file_size) end_pos = file_size - 1;

      Aws::String range =
          absl::StrCat("bytes=", start_pos, "-", end_pos).c_str();
      Aws::S3::Model::UploadPartCopyRequest upload_part_copy_request;
      upload_part_copy_request.WithBucket(bucket_dst)
          .WithKey(object_dst)
          .WithCopySource(source)
          .WithCopySourceRange(range)
          // S3 API partNumber starts from 1.
          .WithPartNumber(part_number + 1)
          .WithUploadId(upload_id);

      auto multi_part_context =
          Aws::MakeShared<MultipartCopyAsyncContext>("MultiPartCopyContext");
      multi_part_context->part_number = part_number;
      multi_part_context->num_finished_parts = &num_finished_parts;
      multi_part_context->etag_outcomes = &etag_outcomes;
      multi_part_context->multi_part_copy_mutex = &multi_part_copy_mutex;
      multi_part_context->multi_part_copy_cv = &multi_part_copy_cv;
      auto callback =
          [](const Aws::S3::S3Client* client,
             const Aws::S3::Model::UploadPartCopyRequest& request,
             const Aws::S3::Model::UploadPartCopyOutcome& outcome,
             const std::shared_ptr<const Aws::Client::AsyncCallerContext>&
                 context) {
            auto multipart_context =
                std::static_pointer_cast<const MultipartCopyAsyncContext>(
                    context);
            MultiPartCopyCallback(request, outcome, multipart_context);
          };

      std::shared_ptr<const Aws::Client::AsyncCallerContext> context =
          multi_part_context;
      s3_file->s3_client->UploadPartCopyAsync(upload_part_copy_request,
                                              callback, context);
    }
    // Wait till they finish.
    {
      absl::MutexLock l(&multi_part_copy_mutex);
      // Wait on the mutex until notify is called then check the finished parts
      // as there could be false notifications.
      while (num_finished_parts != num_parts) {
        multi_part_copy_cv.Wait(&multi_part_copy_mutex);
      }
    }
    // check if there was any error for any part.
    for (auto part_number = 0; part_number < num_parts; ++part_number) {
      if (!etag_outcomes[part_number].IsSuccess()) {
        if (retries >= 3) {
          AbortMultiPartCopy(bucket_dst, object_dst, upload_id, s3_file,
                             status);
          if (TF_GetCode(status) != TF_OK) return;
          return TF_SetStatusFromAWSError(etag_outcomes[part_number].GetError(),
                                          status);
        } else {
          // Retry.
          TF_Log(TF_ERROR,
                 "Retrying failed copy of part %u due to an error with S3\n",
                 part_number);
          num_finished_parts--;
        }
      }
    }
  }

  Aws::S3::Model::CompletedMultipartUpload completed_multipart_upload;
  // If there was an error still in any part, it would abort and return in the
  // above loop. We set the eTag of completed parts to the final
  // `completed_multipart_upload`. Note these parts have to be added in order.
  for (int part_number = 0; part_number < num_parts; ++part_number) {
    Aws::S3::Model::CompletedPart completed_part;
    completed_part.SetPartNumber(part_number + 1);
    completed_part.SetETag(etag_outcomes[part_number].GetResult());
    completed_multipart_upload.AddParts(completed_part);
  }

  Aws::S3::Model::CompleteMultipartUploadRequest
      complete_multipart_upload_request;
  complete_multipart_upload_request.WithBucket(bucket_dst)
      .WithKey(object_dst)
      .WithUploadId(upload_id)
      .WithMultipartUpload(completed_multipart_upload);
  auto complete_multipart_upload_outcome =
      s3_file->s3_client->CompleteMultipartUpload(
          complete_multipart_upload_request);
  if (!complete_multipart_upload_outcome.IsSuccess())
    AbortMultiPartCopy(bucket_dst, object_dst, upload_id, s3_file, status);
  else
    return TF_SetStatus(status, TF_OK, "");
  if (TF_GetCode(status) == TF_OK)
    return TF_SetStatusFromAWSError(
        complete_multipart_upload_outcome.GetError(), status);
};

void CopyFile(const TF_Filesystem* filesystem, const char* src, const char* dst,
              TF_Status* status) {
  auto file_size = GetFileSize(filesystem, src, status);
  if (TF_GetCode(status) != TF_OK) return;
  if (file_size == 0)
    return TF_SetStatus(status, TF_FAILED_PRECONDITION,
                        "Source is a directory or empty file");

  Aws::String bucket_src, object_src;
  ParseS3Path(src, false, &bucket_src, &object_src, status);
  if (TF_GetCode(status) != TF_OK) return;
  Aws::String copy_src = bucket_src + "/" + object_src;

  Aws::String bucket_dst, object_dst;
  ParseS3Path(dst, false, &bucket_dst, &object_dst, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  auto chunk_size =
      s3_file->multi_part_chunk_sizes[Aws::Transfer::TransferDirection::UPLOAD];
  size_t num_parts = 1;
  if (file_size > chunk_size) num_parts = ceil((float)file_size / chunk_size);
  if (num_parts == 1)
    SimpleCopyFile(copy_src, bucket_dst, object_dst, s3_file, status);
  else if (num_parts > 10000)
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        absl::StrCat("MultiPartCopy with number of parts more than 10000 is "
                     "not supported. Your object ",
                     src, " required ", num_parts,
                     " as multi_part_copy_part_size is set to ", chunk_size,
                     ". You can control this part size using the environment "
                     "variable S3_MULTI_PART_COPY_PART_SIZE to increase it.")
            .c_str());
  else
    MultiPartCopy(copy_src, bucket_dst, object_dst, num_parts, file_size,
                  s3_file, status);
}

void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  TF_VLog(1, "DeleteFile: %s\n", path);
  Aws::String bucket, object;
  ParseS3Path(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;
  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);

  Aws::S3::Model::DeleteObjectRequest delete_object_request;
  delete_object_request.WithBucket(bucket).WithKey(object);
  auto delete_object_outcome =
      s3_file->s3_client->DeleteObject(delete_object_request);
  if (!delete_object_outcome.IsSuccess())
    TF_SetStatusFromAWSError(delete_object_outcome.GetError(), status);
  else
    TF_SetStatus(status, TF_OK, "");
}

void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  TF_VLog(1, "CreateDir: %s\n", path);
  Aws::String bucket, object;
  ParseS3Path(path, true, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;
  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);

  if (object.empty()) {
    Aws::S3::Model::HeadBucketRequest head_bucket_request;
    head_bucket_request.WithBucket(bucket);
    auto head_bucket_outcome =
        s3_file->s3_client->HeadBucket(head_bucket_request);
    if (!head_bucket_outcome.IsSuccess())
      TF_SetStatusFromAWSError(head_bucket_outcome.GetError(), status);
    else
      TF_SetStatus(status, TF_OK, "");
    return;
  }

  Aws::String dir_path = path;
  if (dir_path.back() != '/') dir_path.push_back('/');

  PathExists(filesystem, dir_path.c_str(), status);
  if (TF_GetCode(status) == TF_OK) {
    std::unique_ptr<TF_WritableFile, void (*)(TF_WritableFile * file)> file(
        new TF_WritableFile, [](TF_WritableFile* file) {
          if (file != nullptr) {
            if (file->plugin_file != nullptr) tf_writable_file::Cleanup(file);
            delete file;
          }
        });
    file->plugin_file = nullptr;
    NewWritableFile(filesystem, dir_path.c_str(), file.get(), status);
    if (TF_GetCode(status) != TF_OK) return;
    tf_writable_file::Close(file.get(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  TF_VLog(1, "DeleteDir: %s\n", path);
  Aws::String bucket, object;
  ParseS3Path(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;
  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);

  if (object.back() != '/') object.push_back('/');
  Aws::S3::Model::ListObjectsRequest list_objects_request;
  list_objects_request.WithBucket(bucket).WithPrefix(object).WithMaxKeys(2);
  list_objects_request.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });
  auto list_objects_outcome =
      s3_file->s3_client->ListObjects(list_objects_request);
  if (list_objects_outcome.IsSuccess()) {
    auto contents = list_objects_outcome.GetResult().GetContents();
    if (contents.size() > 1 ||
        (contents.size() == 1 && contents[0].GetKey() != object)) {
      TF_SetStatus(status, TF_UNKNOWN,
                   "Cannot delete a non-empty directory. "
                   "This operation will be retried in case this "
                   "is due to S3's eventual consistency.");
    }
    if (contents.size() == 1 && contents[0].GetKey() == object) {
      Aws::String dir_path = path;
      if (dir_path.back() != '/') dir_path.push_back('/');
      DeleteFile(filesystem, dir_path.c_str(), status);
    }
  } else {
    TF_SetStatusFromAWSError(list_objects_outcome.GetError(), status);
  }
}

void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status) {
  TF_VLog(1, "RenameFile from: %s to %s\n", src, dst);
  Aws::String bucket_src, object_src;
  ParseS3Path(src, false, &bucket_src, &object_src, status);
  if (TF_GetCode(status) != TF_OK) return;
  Aws::String copy_src = bucket_src + "/" + object_src;

  Aws::String bucket_dst, object_dst;
  ParseS3Path(dst, false, &bucket_dst, &object_dst, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);

  if (object_src.back() == '/') {
    if (object_dst.back() != '/') {
      object_dst.push_back('/');
    }
  } else {
    if (object_dst.back() == '/') {
      object_dst.pop_back();
    }
  }

  Aws::S3::Model::DeleteObjectRequest delete_object_request;
  Aws::S3::Model::ListObjectsRequest list_objects_request;
  list_objects_request.WithBucket(bucket_src)
      .WithPrefix(object_src)
      .WithMaxKeys(kS3GetChildrenMaxKeys);
  list_objects_request.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult list_objects_result;
  do {
    auto list_objects_outcome =
        s3_file->s3_client->ListObjects(list_objects_request);
    if (!list_objects_outcome.IsSuccess())
      return TF_SetStatusFromAWSError(list_objects_outcome.GetError(), status);

    list_objects_result = list_objects_outcome.GetResult();
    for (const auto& object : list_objects_result.GetContents()) {
      Aws::String key_src = object.GetKey();
      Aws::String key_dst = key_src;
      key_dst.replace(0, object_src.length(), object_dst);
      CopyFile(filesystem, ("s3://" + bucket_src + "/" + key_src).c_str(),
               ("s3://" + bucket_dst + "/" + key_dst).c_str(), status);
      if (TF_GetCode(status) != TF_OK) return;

      delete_object_request.WithBucket(bucket_src).WithKey(key_src);
      auto delete_object_outcome =
          s3_file->s3_client->DeleteObject(delete_object_request);
      if (!delete_object_outcome.IsSuccess())
        return TF_SetStatusFromAWSError(delete_object_outcome.GetError(),
                                        status);
    }
    list_objects_request.SetMarker(list_objects_result.GetNextMarker());
  } while (list_objects_result.GetIsTruncated());
  TF_SetStatus(status, TF_OK, "");
}

int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status) {
  TF_VLog(1, "GetChildren for path: %s\n", path);
  Aws::String bucket, prefix;
  ParseS3Path(path, true, &bucket, &prefix, status);
  if (TF_GetCode(status) != TF_OK) return -1;
  if (!prefix.empty() && prefix.back() != '/') prefix.push_back('/');

  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  GetS3Client(s3_file);

  Aws::S3::Model::ListObjectsRequest list_objects_request;
  list_objects_request.WithBucket(bucket)
      .WithPrefix(prefix)
      .WithMaxKeys(kS3GetChildrenMaxKeys)
      .WithDelimiter("/");
  list_objects_request.SetResponseStreamFactory(
      []() { return Aws::New<Aws::StringStream>(kS3FileSystemAllocationTag); });

  Aws::S3::Model::ListObjectsResult list_objects_result;
  std::vector<Aws::String> result;
  do {
    auto list_objects_outcome =
        s3_file->s3_client->ListObjects(list_objects_request);
    if (!list_objects_outcome.IsSuccess()) {
      TF_SetStatusFromAWSError(list_objects_outcome.GetError(), status);
      return -1;
    }

    list_objects_result = list_objects_outcome.GetResult();
    for (const auto& object : list_objects_result.GetCommonPrefixes()) {
      Aws::String s = object.GetPrefix();
      s.erase(s.length() - 1);
      Aws::String entry = s.substr(prefix.length());
      if (entry.length() > 0) {
        result.push_back(entry);
      }
    }
    for (const auto& object : list_objects_result.GetContents()) {
      Aws::String s = object.GetKey();
      Aws::String entry = s.substr(prefix.length());
      if (entry.length() > 0) {
        result.push_back(entry);
      }
    }
    list_objects_result.SetMarker(list_objects_result.GetNextMarker());
  } while (list_objects_result.GetIsTruncated());

  int num_entries = result.size();
  *entries = static_cast<char**>(
      plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
  for (int i = 0; i < num_entries; i++)
    (*entries)[i] = strdup(result[i].c_str());
  TF_SetStatus(status, TF_OK, "");
  return num_entries;
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

}  // namespace tf_s3_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      plugin_memory_allocate(TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;

  ops->writable_file_ops = static_cast<TF_WritableFileOps*>(
      plugin_memory_allocate(TF_WRITABLE_FILE_OPS_SIZE));
  ops->writable_file_ops->cleanup = tf_writable_file::Cleanup;
  ops->writable_file_ops->append = tf_writable_file::Append;
  ops->writable_file_ops->tell = tf_writable_file::Tell;
  ops->writable_file_ops->flush = tf_writable_file::Flush;
  ops->writable_file_ops->sync = tf_writable_file::Sync;
  ops->writable_file_ops->close = tf_writable_file::Close;

  ops->read_only_memory_region_ops = static_cast<TF_ReadOnlyMemoryRegionOps*>(
      plugin_memory_allocate(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
  ops->read_only_memory_region_ops->cleanup =
      tf_read_only_memory_region::Cleanup;
  ops->read_only_memory_region_ops->data = tf_read_only_memory_region::Data;
  ops->read_only_memory_region_ops->length = tf_read_only_memory_region::Length;

  ops->filesystem_ops = static_cast<TF_FilesystemOps*>(
      plugin_memory_allocate(TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_s3_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_s3_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_s3_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_s3_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_s3_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_s3_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_s3_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_s3_filesystem::DeleteFile;
  ops->filesystem_ops->delete_dir = tf_s3_filesystem::DeleteDir;
  ops->filesystem_ops->copy_file = tf_s3_filesystem::CopyFile;
  ops->filesystem_ops->rename_file = tf_s3_filesystem::RenameFile;
  ops->filesystem_ops->path_exists = tf_s3_filesystem::PathExists;
  ops->filesystem_ops->get_file_size = tf_s3_filesystem::GetFileSize;
  ops->filesystem_ops->stat = tf_s3_filesystem::Stat;
  ops->filesystem_ops->get_children = tf_s3_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_s3_filesystem::TranslateName;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "s3");
}
