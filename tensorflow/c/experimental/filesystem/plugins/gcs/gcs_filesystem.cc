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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "google/cloud/storage/client.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for GCS environments.
// This filesystem will support `gs://` URI scheme.

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
  path /= "tensorflow_tmp_" + std::to_string(now) + "_tmp_gcs_filesystem";
  std::string path_str = path.string();
  auto temp_path_ =
      static_cast<char*>(plugin_memory_allocate(path_str.length() + 1));
  strcpy(temp_path_, path_str.c_str());
  return temp_path_;
}

// We can cast `google::cloud::StatusCode` to `TF_Code` because they have the
// same integer values. See
// https://github.com/googleapis/google-cloud-cpp/blob/6c09cbfa0160bc046e5509b4dd2ab4b872648b4a/google/cloud/status.h#L32-L52
static inline void TF_SetStatusFromGCSStatus(
    const google::cloud::Status& gcs_status, TF_Status* status) {
  TF_SetStatus(status, static_cast<TF_Code>(gcs_status.code()),
               gcs_status.message().c_str());
}

static void ParseGCSPath(const char* fname, bool object_empty_ok, char** bucket,
                         char** object, TF_Status* status) {
  std::string_view fname_view{fname};
  size_t scheme_end = fname_view.find("://") + 2;
  if (fname_view.substr(0, scheme_end + 1) != "gs://") {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "GCS path doesn't start with 'gs://'.");
    return;
  }

  size_t bucket_end = fname_view.find("/", scheme_end + 1);
  if (bucket_end == std::string_view::npos) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "GCS path doesn't contain a bucket name.");
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
                   "GCS path doesn't contain an object name.");
      return;
    }
  }
  *object =
      static_cast<char*>(plugin_memory_allocate(object_view.length() + 1));
  // object_view.data() is a null-terminated string_view because fname is.
  strcpy(*object, object_view.data());
}

static int64_t ReadObjectImpl(const char* bucket, const char* object,
                              std::shared_ptr<FstreamWithName>* temp_file,
                              google::cloud::storage::Client* gcs_client,
                              uint64_t offset, size_t n, char** buffer,
                              bool read_to_buffer, TF_Status* status) {
  int64_t read = 0;
  if (*temp_file == nullptr) {
    // We have to download and save the whole file from GCS,
    // to make sure that we can read any range of this file later.
    *temp_file = std::make_shared<FstreamWithName>();
    if ((*temp_file)->fail()) return -1;
    auto gcs_status = gcs_client->DownloadToFile(bucket, object, (*temp_file)->getName());
    TF_SetStatusFromGCSStatus(gcs_status, status);
    if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_OUT_OF_RANGE)
      return -1;
    (*temp_file)->open((*temp_file)->getName(),
                        std::ios::binary | std::ios::in |
                        std::ios::out | std::ios::ate);
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

static void SyncObjectImpl(const char* bucket, const char* object,
                           const char* temp_path,
                           google::cloud::storage::Client* gcs_client,
                           TF_Status* status) {
  auto metadata = gcs_client->UploadFile(temp_path, bucket, object);
  if (!metadata) TF_SetStatusFromGCSStatus(metadata.status(), status);
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
namespace gcs = google::cloud::storage;

typedef struct GCSFile {
  const char* bucket;
  const char* object;
  gcs::Client* gcs_client;
  std::shared_ptr<FstreamWithName> temp_file;
} GCSFile;

static void Cleanup(TF_RandomAccessFile* file) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  plugin_memory_free(const_cast<char*>(gcs_file->bucket));
  plugin_memory_free(const_cast<char*>(gcs_file->object));
  delete gcs_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  return ReadObjectImpl(gcs_file->bucket, gcs_file->object,
                        &gcs_file->temp_file, gcs_file->gcs_client, offset, n,
                        &buffer, true, status);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
namespace gcs = google::cloud::storage;

typedef struct GCSFile {
  const char* bucket;
  const char* object;
  gcs::Client* gcs_client;
  std::shared_ptr<FstreamWithName> temp_file;
  bool sync_need;
} GCSFile;

static void Cleanup(TF_WritableFile* file) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  plugin_memory_free(const_cast<char*>(gcs_file->bucket));
  plugin_memory_free(const_cast<char*>(gcs_file->object));
  delete gcs_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (gcs_file->temp_file == nullptr) {
    gcs_file->temp_file = std::make_shared<FstreamWithName>();
  }
  if (!gcs_file->temp_file->is_open()) {
    gcs_file->temp_file->open(
        gcs_file->temp_file->getName(),
        std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  }
  if (!gcs_file->temp_file->is_open()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "Could not open the internal temporary file.");
  }

  if (gcs_file->temp_file->fail()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "The internal temporary file is not writable.");
  }
  gcs_file->temp_file->seekp(0, std::ios::end);
  gcs_file->temp_file->write(buffer, n);
  if (gcs_file->temp_file->fail()) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not append to temporary internal file.");
  }
  gcs_file->sync_need = true;
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (gcs_file->temp_file == nullptr) return 0;
  int64_t position = int64_t(gcs_file->temp_file->tellp());
  if (position == -1) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not tellp on the internal temporary file");
  }
  return position;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (gcs_file->sync_need) {
    gcs_file->temp_file->operator<<(std::flush);
    SyncObjectImpl(gcs_file->bucket, gcs_file->object,
                   gcs_file->temp_file->getName(), gcs_file->gcs_client,
                   status);
  }
  gcs_file->sync_need = false;
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  Append(file, "", 0, status);
  Flush(file, status);
  gcs_file->sync_need = false;
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (gcs_file->sync_need) {
    Flush(file, status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (gcs_file->temp_file != nullptr) {
    if (gcs_file->temp_file->is_open()) gcs_file->temp_file->close();
    if (gcs_file->temp_file->fail()) {
      TF_SetStatus(status, TF_INTERNAL, "Could not close temporary file.");
    }
  }
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

typedef struct GCSMemoryRegion {
  const void* const address;
  const uint64_t length;
} GCSMemoryRegion;

static void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  plugin_memory_free(const_cast<void*>(r->address));
  delete r;
}

static const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  return r->address;
}

static uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------

namespace tf_gcs_filesystem {
namespace gcs = google::cloud::storage;

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  google::cloud::StatusOr<gcs::Client> client =
      gcs::Client::CreateDefaultClient();
  if (!client) {
    TF_SetStatusFromGCSStatus(client.status(), status);
    return;
  }
  filesystem->plugin_filesystem = plugin_memory_allocate(sizeof(gcs::Client));
  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  (*gcs_client) = client.value();
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {
  plugin_memory_free(filesystem->plugin_filesystem);
}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  char* bucket;
  char* object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  file->plugin_file =
      new tf_random_access_file::GCSFile({bucket, object, gcs_client, nullptr});
  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  char* bucket;
  char* object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  file->plugin_file = new tf_writable_file::GCSFile(
      {bucket, object, gcs_client, nullptr, false});

  TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  char* bucket;
  char* object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  std::shared_ptr<FstreamWithName> fstream_with_name = nullptr;
  int64_t read = ReadObjectImpl(bucket, object, &fstream_with_name, gcs_client,
                                0, 0, nullptr, false, status);
  if (read < 0 && TF_GetCode(status) != TF_NOT_FOUND) {
    return;
  }

  file->plugin_file = new tf_writable_file::GCSFile(
      {bucket, object, gcs_client, fstream_with_name, false});
  TF_SetStatus(status, TF_OK, "");
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
  char* bucket;
  char* object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  std::shared_ptr<FstreamWithName> fstream_with_name = nullptr;
  char* buffer = nullptr;
  int64_t read = ReadObjectImpl(bucket, object, &fstream_with_name, gcs_client,
                                0, 0, &buffer, true, status);
  if (read > 0 && buffer) {
    region->plugin_memory_region =
        new tf_read_only_memory_region::GCSMemoryRegion(
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
  ParseGCSPath(path, false, &bucket, &object_temporary, status);
  if (TF_GetCode(status) != TF_OK) return;
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

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  auto inserter = gcs_client->InsertObject(bucket, object, "");
  if (!inserter) {
    TF_SetStatusFromGCSStatus(inserter.status(), status);
    return;
  }
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  char* bucket;
  char* object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  auto gcs_status = gcs_client->DeleteObject(bucket, object);
  TF_SetStatusFromGCSStatus(gcs_status, status);
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  char* bucket;
  char* object_temporary;
  ParseGCSPath(path, false, &bucket, &object_temporary, status);
  if (TF_GetCode(status) != TF_OK) return;
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

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  auto gcs_status = gcs_client->DeleteObject(bucket, object);
  TF_SetStatusFromGCSStatus(gcs_status, status);
}

static void DeleteRecursively(const TF_Filesystem* filesystem, const char* path,
                              uint64_t* undeleted_files,
                              uint64_t* undeleted_dirs, TF_Status* status) {
  *undeleted_dirs = 0;
  *undeleted_files = 0;
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "DeleteRecursively is not implemented");
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  char* bucket_src;
  char* object_src;
  ParseGCSPath(src, false, &bucket_src, &object_src, status);
  if (TF_GetCode(status) != TF_OK) return;

  char* bucket_dst;
  char* object_dst;
  ParseGCSPath(dst, false, &bucket_dst, &object_dst, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  auto metadata = gcs_client->RewriteObjectBlocking(bucket_src, object_src,
                                                    bucket_dst, object_dst);
  if (!metadata) {
    TF_SetStatusFromGCSStatus(metadata.status(), status);
    return;
  }
  auto gcs_status = gcs_client->DeleteObject(bucket_src, object_src);
  TF_SetStatusFromGCSStatus(metadata.status(), status);
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
  char* bucket_src;
  char* object_src;
  ParseGCSPath(src, false, &bucket_src, &object_src, status);
  if (TF_GetCode(status) != TF_OK) return;

  char* bucket_dst;
  char* object_dst;
  ParseGCSPath(dst, false, &bucket_dst, &object_dst, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  auto metadata = gcs_client->RewriteObjectBlocking(bucket_src, object_src,
                                                    bucket_dst, object_dst);
  if (!metadata) {
    TF_SetStatusFromGCSStatus(metadata.status(), status);
    return;
  }
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  char* bucket;
  char* object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_client = static_cast<gcs::Client*>(filesystem->plugin_filesystem);
  auto metadata = gcs_client->GetObjectMetadata(bucket, object);
  if (!metadata) {
    TF_SetStatusFromGCSStatus(metadata.status(), status);
    return;
  }
  stats->length = metadata.value().size();
  stats->mtime_nsec =
      metadata.value().time_storage_class_updated().time_since_epoch().count();
  if (path[strlen(path) - 1] == '/') {
    stats->is_directory = true;
  } else {
    stats->is_directory = false;
  }
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  TF_FileStatistics stats;
  Stat(filesystem, path, &stats, status);
}

static bool IsDirectory(const TF_Filesystem* filesystem, const char* path,
                        TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED, "IsDirectory is not implemented");
  return true;
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  char* name = static_cast<char*>(plugin_memory_allocate(strlen(uri) + 1));
  strcpy(name, uri);
  return name;
}

}  // namespace tf_gcs_filesystem

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
  ops->filesystem_ops->init = tf_gcs_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_gcs_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_gcs_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->new_writable_file = tf_gcs_filesystem::NewWritableFile;
  ops->filesystem_ops->new_appendable_file =
      tf_gcs_filesystem::NewAppendableFile;
  ops->filesystem_ops->new_read_only_memory_region_from_file =
      tf_gcs_filesystem::NewReadOnlyMemoryRegionFromFile;
  ops->filesystem_ops->create_dir = tf_gcs_filesystem::CreateDir;
  ops->filesystem_ops->delete_file = tf_gcs_filesystem::DeleteFile;
  ops->filesystem_ops->rename_file = tf_gcs_filesystem::RenameFile;
  ops->filesystem_ops->copy_file = tf_gcs_filesystem::CopyFile;
  ops->filesystem_ops->path_exists = tf_gcs_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_gcs_filesystem::Stat;
  ops->filesystem_ops->translate_name = tf_gcs_filesystem::TranslateName;
  ops->filesystem_ops->recursively_create_dir = tf_gcs_filesystem::CreateDir;
  ops->filesystem_ops->delete_recursively =
      tf_gcs_filesystem::DeleteRecursively;
  ops->filesystem_ops->is_directory = tf_gcs_filesystem::IsDirectory;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "gs");
}