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
#include <aws/core/Aws.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_shared.h"
#include "tensorflow/c/tf_status.h"

#define TF_S3_RETURN_IF_ERROR(status)        \
  do {                                       \
    if (TF_GetCode(status) != TF_OK) return; \
  } while (0)

#define S3_ALLOCATION_TAG_TEMP_FILE "S3_TENSORFLOW_TEMP_FILE"

// Implementation of a filesystem for S3 environments.
// This filesystem will support `s3://` URI scheme.

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

class FstreamWithName : public std::fstream {
 public:
  FstreamWithName() : name(CreateTempPath()) {
    if (name == NULL) {
      std::fstream::clear(std::ios::badbit);
    }
  };
  ~FstreamWithName() {
    if (name == NULL) return;
    std::error_code errorCode;
    if (std::fstream::is_open()) std::fstream::close();
    if (std::filesystem::exists(name)) std::filesystem::remove(name, errorCode);
    if (errorCode)
      std::cout << errorCode.value() << ": " << errorCode.message();
    plugin_memory_free(const_cast<char*>(name));
  }
  const char* getName() const { return name; }

 private:
  const char* name;
  static const char* CreateTempPath();
};

const char* FstreamWithName::CreateTempPath() {
  uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::error_code errorCode;
  auto path = std::filesystem::temp_directory_path(errorCode);
  if (errorCode) {
    std::cout << errorCode.value() << ": " << errorCode.message() << "\n";
    return NULL;
  }
  path /= "tensorflow_tmp_" + std::to_string(now) + "_tmp_s3_filesystem";
  std::string path_str = path.string();
  auto temp_path_ =
      static_cast<char*>(plugin_memory_allocate(path_str.length() + 1));
  strcpy(temp_path_, path_str.c_str());
  return temp_path_;
}

static void inline TF_SetStatusFromAWSError(
    TF_Status* status, const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  switch (error.GetResponseCode()) {
    case Aws::Http::HttpResponseCode::PRECONDITION_FAILED:
      TF_SetStatus(status, TF_FAILED_PRECONDITION, error.GetMessage().c_str());
      break;
    case Aws::Http::HttpResponseCode::NOT_FOUND:
      TF_SetStatus(status, TF_NOT_FOUND, error.GetMessage().c_str());
      break;
    case Aws::Http::HttpResponseCode::NOT_IMPLEMENTED:
      TF_SetStatus(status, TF_UNIMPLEMENTED, error.GetMessage().c_str());
      break;
    case Aws::Http::HttpResponseCode::UNAUTHORIZED:
      TF_SetStatus(status, TF_UNAUTHENTICATED, error.GetMessage().c_str());
      break;
    case Aws::Http::HttpResponseCode::FORBIDDEN:
      TF_SetStatus(status, TF_PERMISSION_DENIED, error.GetMessage().c_str());
      break;
    default:
      TF_SetStatus(
          status, TF_UNKNOWN,
          (error.GetExceptionName() + ": " + error.GetMessage()).c_str());
      break;
  }
}

static void ParseS3Path(const char* fname, bool object_empty_ok, char** bucket,
                        char** object, TF_Status* status) {
  std::string_view fname_view{fname};
  size_t scheme_end = fname_view.find("://") + 2;
  if (fname_view.substr(0, scheme_end + 1) != "s3://") {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't start with 's3://'.");
    return;
  }

  size_t bucket_end = fname_view.find("/", scheme_end + 1);
  if (bucket_end == std::string_view::npos) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't contain a bucket name.");
    return;
  }
  std::string_view bucket_view =
      fname_view.substr(scheme_end + 1, bucket_end - scheme_end - 1);
  *bucket =
      static_cast<char*>(plugin_memory_allocate(bucket_view.length() + 1));
  memcpy(*bucket, bucket_view.data(), bucket_view.length());
  (*bucket)[bucket_view.length()] = '\0';

  std::string_view object_view = fname_view.substr(bucket_end + 1);
  if (object_view == "") {
    if (object_empty_ok) {
      *object = nullptr;
      return;
    } else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "S3 path doesn't contain an object name.");
      return;
    }
  }
  *object =
      static_cast<char*>(plugin_memory_allocate(object_view.length() + 1));
  // object_view.data() is a null-terminated string_view because fname is.
  strcpy(*object, object_view.data());
}

static int64_t ReadObjectImpl(
    const char* bucket, const char* object,
    std::shared_ptr<FstreamWithName>* temp_file,
    const std::shared_ptr<Aws::Transfer::TransferManager>& transfer_manager,
    uint64_t offset, size_t n, char** buffer, bool read_to_buffer,
    TF_Status* status) {
  int64_t read = 0;
  if (*temp_file == nullptr) {
    *temp_file = Aws::MakeShared<FstreamWithName>(S3_ALLOCATION_TAG_TEMP_FILE);
    if ((*temp_file)->fail()) return -1;
    auto transfer_handle =
        transfer_manager->DownloadFile(bucket, object, (*temp_file)->getName());
    transfer_handle->WaitUntilFinished();
    if (transfer_handle->GetStatus() !=
        Aws::Transfer::TransferStatus::COMPLETED) {
      TF_SetStatusFromAWSError(status, transfer_handle->GetLastError());
      return -1;
    }
    (*temp_file)
        ->open((*temp_file)->getName(),
               std::ios::binary | std::ios::in | std::ios::out | std::ios::ate);
  }
  if (!(*temp_file)->is_open()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Could not open downloaded temporary file.");
    return -1;
  }

  std::error_code errorCode;
  read =
      std::filesystem::file_size((*temp_file)->getName(), errorCode) - offset;
  if (errorCode) {
    TF_SetStatus(
        status, TF_UNKNOWN,
        (std::to_string(errorCode.value()) + ": " + errorCode.message())
            .c_str());
    return -1;
  }

  if (read < n)
    TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested.");

  if (!read_to_buffer || buffer == nullptr) return read;

  if (!(*buffer)) *buffer = static_cast<char*>(plugin_memory_allocate(read));
  if (*buffer == NULL) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not allocate buffer to save data from downloaded "
                 "temporary file.");
    return -1;
  }
  (*temp_file)->seekg(offset);
  (*temp_file)->read(*buffer, read);
  if ((*temp_file)->fail()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested.");
    read = (*temp_file)->gcount();
  }
  return read;
}

static void SyncObjectImpl(
    const char* bucket, const char* object, const char* temp_path,
    const std::shared_ptr<Aws::Transfer::TransferManager>& transfer_manager,
    TF_Status* status) {
  auto transfer_handle = transfer_manager->UploadFile(
      temp_path, bucket, object, "application/octet-stream",
      Aws::Map<Aws::String, Aws::String>());
  transfer_handle->WaitUntilFinished();
  if (transfer_handle->GetStatus() != Aws::Transfer::TransferStatus::COMPLETED)
    TF_SetStatusFromAWSError(status, transfer_handle->GetLastError());
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

typedef struct S3File {
  const char* bucket;
  const char* object;
  const std::shared_ptr<Aws::Transfer::TransferManager>& transfer_manager;
  std::shared_ptr<FstreamWithName> temp_file;
} S3File;

static void Cleanup(TF_RandomAccessFile* file) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  plugin_memory_free(const_cast<char*>(s3_file->bucket));
  plugin_memory_free(const_cast<char*>(s3_file->object));
  delete s3_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  return ReadObjectImpl(s3_file->bucket, s3_file->object, &s3_file->temp_file,
                        s3_file->transfer_manager, offset, n, &buffer, true,
                        status);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

typedef struct S3File {
  const char* bucket;
  const char* object;
  const std::shared_ptr<Aws::Transfer::TransferManager>& transfer_manager;
  std::shared_ptr<FstreamWithName> temp_file;
  bool sync_need;
} S3File;

static void Cleanup(TF_WritableFile* file) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  plugin_memory_free(const_cast<char*>(s3_file->bucket));
  plugin_memory_free(const_cast<char*>(s3_file->object));
  delete s3_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (s3_file->temp_file == nullptr) {
    s3_file->temp_file =
        Aws::MakeShared<FstreamWithName>(S3_ALLOCATION_TAG_TEMP_FILE);
  }

  if (!s3_file->temp_file->is_open()) {
    s3_file->temp_file->open(
        s3_file->temp_file->getName(),
        std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  }

  if (!s3_file->temp_file->is_open()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Could not open the internal temporary file.");
  }

  if (s3_file->temp_file->fail()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "The internal temporary file is not writable.");
  }

  s3_file->temp_file->seekp(0, std::ios::end);
  s3_file->temp_file->write(buffer, n);
  if (s3_file->temp_file->fail()) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not append to temporary internal file.");
  }

  s3_file->sync_need = true;
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (s3_file->temp_file == nullptr) return 0;
  int64_t position = int64_t(s3_file->temp_file->tellp());
  if (position == -1) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not tellp on the internal temporary file");
  }
  return position;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (s3_file->sync_need) {
    s3_file->temp_file->operator<<(std::flush);
    SyncObjectImpl(s3_file->bucket, s3_file->object,
                   s3_file->temp_file->getName(), s3_file->transfer_manager,
                   status);
  }
  s3_file->sync_need = false;
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  Append(file, "", 0, status);
  Flush(file, status);
  s3_file->sync_need = false;
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto s3_file = static_cast<S3File*>(file->plugin_file);
  if (s3_file->sync_need) {
    Flush(file, status);
    TF_S3_RETURN_IF_ERROR(status);
  }
  if (s3_file->temp_file != nullptr) {
    if (s3_file->temp_file->is_open()) s3_file->temp_file->close();
    if (s3_file->temp_file->fail()) {
      TF_SetStatus(status, TF_INTERNAL, "Could not close temporary file.");
    }
  }
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

typedef struct S3MemoryRegion {
  const void* const address;
  const uint64_t length;
} S3MemoryRegion;

static void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  plugin_memory_free(const_cast<void*>(r->address));
  delete r;
}

static const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  return r->address;
}

static uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<S3MemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_s3_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = plugin_memory_allocate(sizeof(S3Shared));
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  char* bucket;
  char* object;
  ParseS3Path(path, false, &bucket, &object, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetTransferManager(s3_shared);
  file->plugin_file = new tf_random_access_file::S3File(
      {bucket, object, s3_shared->transfer_manager, nullptr});
  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  char* bucket;
  char* object;
  ParseS3Path(path, false, &bucket, &object, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetTransferManager(s3_shared);
  file->plugin_file = new tf_writable_file::S3File(
      {bucket, object, s3_shared->transfer_manager, nullptr, false});

  TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  char* bucket;
  char* object;
  ParseS3Path(path, false, &bucket, &object, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetTransferManager(s3_shared);
  std::shared_ptr<FstreamWithName> fstream_with_name = nullptr;

  int64_t read =
      ReadObjectImpl(bucket, object, &fstream_with_name,
                     s3_shared->transfer_manager, 0, 0, nullptr, false, status);
  if (read < 0 && TF_GetCode(status) != TF_NOT_FOUND) {
    return;
  }
  file->plugin_file = new tf_writable_file::S3File(
      {bucket, object, s3_shared->transfer_manager, fstream_with_name, false});
  TF_SetStatus(status, TF_OK, "");
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
  char* bucket;
  char* object;
  ParseS3Path(path, false, &bucket, &object, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetTransferManager(s3_shared);
  std::shared_ptr<FstreamWithName> fstream_with_name = nullptr;
  char* buffer = nullptr;
  int64_t read =
      ReadObjectImpl(bucket, object, &fstream_with_name,
                     s3_shared->transfer_manager, 0, 0, &buffer, true, status);
  if (read > 0 && buffer) {
    region->plugin_memory_region =
        new tf_read_only_memory_region::S3MemoryRegion(
            {buffer, static_cast<uint64_t>(read)});
    return;
  }
  if (read == 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "File is empty");
    return;
  }
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  char* bucket;
  char* object_temporary;
  ParseS3Path(path, false, &bucket, &object_temporary, status);
  TF_S3_RETURN_IF_ERROR(status);

  char* object = nullptr;
  if (object_temporary[strlen(object_temporary) - 1] != '/') {
    object = static_cast<char*>(
        plugin_memory_allocate(strlen(object_temporary) + 2));
    strcpy(object, object_temporary);
    object[strlen(object_temporary)] = '/';
    object[strlen(object_temporary) + 1] = '\0';
    free(object_temporary);
  } else {
    object = object_temporary;
  }

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetS3Client(s3_shared);

  Aws::S3::Model::PutObjectRequest putObjectRequest;
  putObjectRequest.WithBucket(bucket).WithKey(object);
  auto outcome = s3_shared->s3_client->PutObject(putObjectRequest);
  if (!outcome.IsSuccess())
    TF_SetStatusFromAWSError(status, outcome.GetError());
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  char* bucket;
  char* object;
  ParseS3Path(path, false, &bucket, &object, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetS3Client(s3_shared);

  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket).WithKey(object);
  auto outcome = s3_shared->s3_client->DeleteObject(deleteObjectRequest);
  if (!outcome.IsSuccess())
    TF_SetStatusFromAWSError(status, outcome.GetError());
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  char* bucket;
  char* object_temporary;
  ParseS3Path(path, false, &bucket, &object_temporary, status);
  TF_S3_RETURN_IF_ERROR(status);

  char* object = nullptr;
  if (object_temporary[strlen(object_temporary) - 1] != '/') {
    object = static_cast<char*>(
        plugin_memory_allocate(strlen(object_temporary) + 2));
    strcpy(object, object_temporary);
    object[strlen(object_temporary)] = '/';
    object[strlen(object_temporary) + 1] = '\0';
    free(object_temporary);
  } else {
    object = object_temporary;
  }

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetS3Client(s3_shared);

  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket).WithKey(object);
  auto outcome = s3_shared->s3_client->DeleteObject(deleteObjectRequest);
  if (!outcome.IsSuccess())
    TF_SetStatusFromAWSError(status, outcome.GetError());
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  char* bucket_src;
  char* object_src;
  ParseS3Path(src, false, &bucket_src, &object_src, status);
  TF_S3_RETURN_IF_ERROR(status);

  char* bucket_dst;
  char* object_dst;
  ParseS3Path(dst, false, &bucket_dst, &object_dst, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetS3Client(s3_shared);

  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  copyObjectRequest.WithCopySource(Aws::String(bucket_src) + "/" + object_src);
  copyObjectRequest.WithBucket(bucket_dst).WithKey(object_dst);
  auto copyObjectOutcome = s3_shared->s3_client->CopyObject(copyObjectRequest);
  if (!copyObjectOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, copyObjectOutcome.GetError());
    return;
  }

  Aws::S3::Model::DeleteObjectRequest deleteObjectRequest;
  deleteObjectRequest.WithBucket(bucket_src).WithKey(object_src);
  auto deleteObjectOutcome =
      s3_shared->s3_client->DeleteObject(deleteObjectRequest);
  if (!deleteObjectOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, deleteObjectOutcome.GetError());
    return;
  }
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
  char* bucket_src;
  char* object_src;
  ParseS3Path(src, false, &bucket_src, &object_src, status);
  TF_S3_RETURN_IF_ERROR(status);

  char* bucket_dst;
  char* object_dst;
  ParseS3Path(dst, false, &bucket_dst, &object_dst, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetS3Client(s3_shared);

  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  copyObjectRequest.WithCopySource(Aws::String(bucket_src) + "/" + object_src);
  copyObjectRequest.WithBucket(bucket_dst).WithKey(object_dst);
  auto copyObjectOutcome = s3_shared->s3_client->CopyObject(copyObjectRequest);
  if (!copyObjectOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, copyObjectOutcome.GetError());
    return;
  }
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  char* bucket;
  char* object;
  ParseS3Path(path, false, &bucket, &object, status);
  TF_S3_RETURN_IF_ERROR(status);

  auto s3_shared = static_cast<S3Shared*>(filesystem->plugin_filesystem);
  GetS3Client(s3_shared);

  Aws::S3::Model::HeadObjectRequest headObjectRequest;
  headObjectRequest.WithBucket(bucket).WithKey(object);
  auto outcome = s3_shared->s3_client->HeadObject(headObjectRequest);
  if (!outcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, outcome.GetError());
    return;
  }
  stats->is_directory = false;
  stats->length = outcome.GetResult().GetContentLength();
  stats->mtime_nsec = outcome.GetResult().GetLastModified().Millis();
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  char* name = static_cast<char*>(plugin_memory_allocate(strlen(uri) + 1));
  strcpy(name, uri);
  return name;
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
  ops->writable_file_ops->flush = tf_writable_file::Flush;
  ops->writable_file_ops->sync = tf_writable_file::Sync;
  ops->writable_file_ops->close = tf_writable_file::Close;
  ops->writable_file_ops->tell = tf_writable_file::Tell;

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
  ops->filesystem_ops->recursively_create_dir = tf_s3_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_s3_filesystem::DeleteFile;
  ops->filesystem_ops->rename_file = tf_s3_filesystem::RenameFile;
  ops->filesystem_ops->copy_file = tf_s3_filesystem::CopyFile;
  ops->filesystem_ops->path_exists = tf_s3_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_s3_filesystem::Stat;
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
