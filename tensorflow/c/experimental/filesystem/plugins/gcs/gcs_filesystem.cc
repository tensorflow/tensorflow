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
#include "tensorflow/c/experimental/filesystem/plugins/gcs/gcs_filesystem.h"

#include <stdlib.h>
#include <string.h>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "google/cloud/storage/client.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/experimental/filesystem/plugins/gcs/gcs_helper.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for GCS environments.
// This filesystem will support `gs://` URI schemes.
namespace gcs = google::cloud::storage;

// The environment variable that overrides the block size for aligned reads from
// GCS. Specified in MB (e.g. "16" = 16 x 1024 x 1024 = 16777216 bytes).
constexpr char kBlockSize[] = "GCS_READ_CACHE_BLOCK_SIZE_MB";
constexpr size_t kDefaultBlockSize = 64 * 1024 * 1024;
// The environment variable that overrides the max size of the LRU cache of
// blocks read from GCS. Specified in MB.
constexpr char kMaxCacheSize[] = "GCS_READ_CACHE_MAX_SIZE_MB";
constexpr size_t kDefaultMaxCacheSize = 0;
// The environment variable that overrides the maximum staleness of cached file
// contents. Once any block of a file reaches this staleness, all cached blocks
// will be evicted on the next read.
constexpr char kMaxStaleness[] = "GCS_READ_CACHE_MAX_STALENESS";
constexpr uint64_t kDefaultMaxStaleness = 0;

constexpr char kStatCacheMaxAge[] = "GCS_STAT_CACHE_MAX_AGE";
constexpr uint64_t kStatCacheDefaultMaxAge = 5;
// The environment variable that overrides the maximum number of entries in the
// Stat cache.
constexpr char kStatCacheMaxEntries[] = "GCS_STAT_CACHE_MAX_ENTRIES";
constexpr size_t kStatCacheDefaultMaxEntries = 1024;

// How to upload new data when Flush() is called multiple times.
// By default the entire file is reuploaded.
constexpr char kAppendMode[] = "GCS_APPEND_MODE";
// If GCS_APPEND_MODE=compose then instead the new data is uploaded to a
// temporary object and composed with the original object. This is disabled by
// default as the multiple API calls required add a risk of stranding temporary
// objects.
constexpr char kComposeAppend[] = "compose";

// We can cast `google::cloud::StatusCode` to `TF_Code` because they have the
// same integer values. See
// https://github.com/googleapis/google-cloud-cpp/blob/6c09cbfa0160bc046e5509b4dd2ab4b872648b4a/google/cloud/status.h#L32-L52
static inline void TF_SetStatusFromGCSStatus(
    const google::cloud::Status& gcs_status, TF_Status* status) {
  TF_SetStatus(status, static_cast<TF_Code>(gcs_status.code()),
               gcs_status.message().c_str());
}

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

void ParseGCSPath(const std::string& fname, bool object_empty_ok,
                  std::string* bucket, std::string* object, TF_Status* status) {
  size_t scheme_end = fname.find("://") + 2;
  if (fname.substr(0, scheme_end + 1) != "gs://") {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "GCS path doesn't start with 'gs://'.");
    return;
  }

  size_t bucket_end = fname.find('/', scheme_end + 1);
  if (bucket_end == std::string::npos) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "GCS path doesn't contain a bucket name.");
    return;
  }

  *bucket = fname.substr(scheme_end + 1, bucket_end - scheme_end - 1);
  *object = fname.substr(bucket_end + 1);

  if (object->empty() && !object_empty_ok) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "GCS path doesn't contain an object name.");
  }
}

/// Appends a trailing slash if the name doesn't already have one.
static void MaybeAppendSlash(std::string* name) {
  if (name->empty())
    *name = "/";
  else if (name->back() != '/')
    name->push_back('/');
}

// A helper function to actually read the data from GCS.
static int64_t LoadBufferFromGCS(const std::string& path, size_t offset,
                                 size_t buffer_size, char* buffer,
                                 tf_gcs_filesystem::GCSFile* gcs_file,
                                 TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return -1;
  auto stream = gcs_file->gcs_client.ReadObject(
      bucket, object, gcs::ReadRange(offset, offset + buffer_size));
  TF_SetStatusFromGCSStatus(stream.status(), status);
  if ((TF_GetCode(status) != TF_OK) &&
      (TF_GetCode(status) != TF_OUT_OF_RANGE)) {
    return -1;
  }
  int64_t read;
  auto content_length = stream.headers().find("content-length");
  if (content_length == stream.headers().end()) {
    // When we read a file with offset that is bigger than the actual file size.
    // GCS will return an empty header (e.g no `content-length` header). In this
    // case, we will set read to `0` and continue.
    read = 0;
  } else if (!absl::SimpleAtoi(content_length->second, &read)) {
    TF_SetStatus(status, TF_UNKNOWN, "Could not get content-length header");
    return -1;
  }
  // `TF_OUT_OF_RANGE` isn't considered as an error. So we clear it here.
  TF_SetStatus(status, TF_OK, "");
  TF_VLog(1, "Successful read of %s @ %u of size: %u", path.c_str(), offset,
          read);
  stream.read(buffer, read);
  read = stream.gcount();
  if (read < buffer_size) {
    // Check stat cache to see if we encountered an interrupted read.
    tf_gcs_filesystem::GcsFileStat stat;
    if (gcs_file->stat_cache->Lookup(path, &stat)) {
      if (offset + read < stat.base.length) {
        TF_SetStatus(status, TF_INTERNAL,
                     absl::StrCat("File contents are inconsistent for file: ",
                                  path, " @ ", offset)
                         .c_str());
      }
      TF_VLog(2, "Successful integrity check for: %s @ %u", path.c_str(),
              offset);
    }
  }
  return read;
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
using ReadFn =
    std::function<int64_t(const std::string& path, uint64_t offset, size_t n,
                          char* buffer, TF_Status* status)>;
typedef struct GCSFile {
  const std::string path;
  const bool is_cache_enable;
  const uint64_t buffer_size;
  ReadFn read_fn;
  absl::Mutex buffer_mutex;
  uint64_t buffer_start ABSL_GUARDED_BY(buffer_mutex);
  bool buffer_end_is_past_eof ABSL_GUARDED_BY(buffer_mutex);
  std::string buffer ABSL_GUARDED_BY(buffer_mutex);

  GCSFile(std::string path, bool is_cache_enable, uint64_t buffer_size,
          ReadFn read_fn)
      : path(path),
        is_cache_enable(is_cache_enable),
        buffer_size(buffer_size),
        read_fn(std::move(read_fn)),
        buffer_mutex(),
        buffer_start(0),
        buffer_end_is_past_eof(false),
        buffer() {}
} GCSFile;

void Cleanup(TF_RandomAccessFile* file) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  delete gcs_file;
}

// `google-cloud-cpp` is working on a feature that we may want to use.
// See https://github.com/googleapis/google-cloud-cpp/issues/4013.
int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
             char* buffer, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (gcs_file->is_cache_enable || n > gcs_file->buffer_size) {
    return gcs_file->read_fn(gcs_file->path, offset, n, buffer, status);
  } else {
    absl::MutexLock l(&gcs_file->buffer_mutex);
    size_t buffer_end = gcs_file->buffer_start + gcs_file->buffer.size();
    size_t copy_size = 0;
    if (offset < buffer_end && gcs_file->buffer_start) {
      copy_size = (std::min)(n, static_cast<size_t>(buffer_end - offset));
      memcpy(buffer,
             gcs_file->buffer.data() + (offset - gcs_file->buffer_start),
             copy_size);
    }
    bool consumed_buffer_to_eof =
        offset + copy_size >= buffer_end && gcs_file->buffer_end_is_past_eof;
    if (copy_size < n && !consumed_buffer_to_eof) {
      gcs_file->buffer_start = offset + copy_size;
      gcs_file->buffer.resize(gcs_file->buffer_size);
      auto read_fill_buffer = gcs_file->read_fn(
          gcs_file->path, gcs_file->buffer_start, gcs_file->buffer_size,
          &(gcs_file->buffer[0]), status);
      gcs_file->buffer_end_is_past_eof =
          (TF_GetCode(status) == TF_OUT_OF_RANGE);
      if (read_fill_buffer >= 0) gcs_file->buffer.resize(read_fill_buffer);
      if (TF_GetCode(status) != TF_OK &&
          TF_GetCode(status) != TF_OUT_OF_RANGE) {
        // Empty the buffer to avoid caching bad reads.
        gcs_file->buffer.resize(0);
        return -1;
      }
      size_t remaining_copy =
          (std::min)(n - copy_size, gcs_file->buffer.size());
      memcpy(buffer + copy_size, gcs_file->buffer.data(), remaining_copy);
      copy_size += remaining_copy;
    }
    if (copy_size < n) {
      // Forget the end-of-file flag to allow for clients that poll on the
      // same file.
      gcs_file->buffer_end_is_past_eof = false;
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read less bytes than requested");
      return copy_size;
    }
    TF_SetStatus(status, TF_OK, "");
    return copy_size;
  }
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
typedef struct GCSFile {
  const std::string bucket;
  const std::string object;
  gcs::Client* gcs_client;  // not owned
  TempFile outfile;
  bool sync_need;
  // `offset` tells us how many bytes of this file are already uploaded to
  // server. If `offset == -1`, we always upload the entire temporary file.
  int64_t offset;
} GCSFile;

static void SyncImpl(const std::string& bucket, const std::string& object,
                     int64_t* offset, TempFile* outfile,
                     gcs::Client* gcs_client, TF_Status* status) {
  outfile->flush();
  // `*offset == 0` means this file does not exist on the server.
  if (*offset == -1 || *offset == 0) {
    // UploadFile will automatically switch to resumable upload based on Client
    // configuration.
    auto metadata = gcs_client->UploadFile(outfile->getName(), bucket, object,
                                           gcs::Fields("size"));
    if (!metadata) {
      TF_SetStatusFromGCSStatus(metadata.status(), status);
      return;
    }
    if (*offset == 0) {
      if (!outfile->truncate()) {
        TF_SetStatus(status, TF_INTERNAL,
                     "Could not truncate internal temporary file.");
        return;
      }
      *offset = static_cast<int64_t>(metadata->size());
    }
    outfile->clear();
    outfile->seekp(0, std::ios::end);
    TF_SetStatus(status, TF_OK, "");
  } else {
    std::string temporary_object =
        gcs::CreateRandomPrefixName("tf_writable_file_gcs");
    auto metadata = gcs_client->UploadFile(outfile->getName(), bucket,
                                           temporary_object, gcs::Fields(""));
    if (!metadata) {
      TF_SetStatusFromGCSStatus(metadata.status(), status);
      return;
    }
    TF_VLog(3, "AppendObject: gs://%s/%s to gs://%s/%s", bucket.c_str(),
            temporary_object.c_str(), bucket.c_str(), object.c_str());
    const std::vector<gcs::ComposeSourceObject> source_objects = {
        {object, {}, {}}, {temporary_object, {}, {}}};
    metadata = gcs_client->ComposeObject(bucket, source_objects, object,
                                         gcs::Fields("size"));
    if (!metadata) {
      TF_SetStatusFromGCSStatus(metadata.status(), status);
      return;
    }
    // We have to delete the temporary object after composing.
    auto delete_status = gcs_client->DeleteObject(bucket, temporary_object);
    if (!delete_status.ok()) {
      TF_SetStatusFromGCSStatus(delete_status, status);
      return;
    }
    // We truncate the data that are already uploaded.
    if (!outfile->truncate()) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Could not truncate internal temporary file.");
      return;
    }
    *offset = static_cast<int64_t>(metadata->size());
    TF_SetStatus(status, TF_OK, "");
  }
}

void Cleanup(TF_WritableFile* file) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  delete gcs_file;
}

void Append(const TF_WritableFile* file, const char* buffer, size_t n,
            TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (!gcs_file->outfile.is_open()) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION,
                 "The internal temporary file is not writable.");
    return;
  }
  TF_VLog(3, "Append: gs://%s/%s size %u", gcs_file->bucket.c_str(),
          gcs_file->object.c_str(), n);
  gcs_file->sync_need = true;
  gcs_file->outfile.write(buffer, n);
  if (!gcs_file->outfile)
    TF_SetStatus(status, TF_INTERNAL,
                 "Could not append to the internal temporary file.");
  else
    TF_SetStatus(status, TF_OK, "");
}

int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  int64_t position = int64_t(gcs_file->outfile.tellp());
  if (position == -1)
    TF_SetStatus(status, TF_INTERNAL,
                 "tellp on the internal temporary file failed");
  else
    TF_SetStatus(status, TF_OK, "");
  return position == -1
             ? -1
             : position + (gcs_file->offset == -1 ? 0 : gcs_file->offset);
}

void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  if (gcs_file->sync_need) {
    TF_VLog(3, "Flush started: gs://%s/%s", gcs_file->bucket.c_str(),
            gcs_file->object.c_str());
    if (!gcs_file->outfile) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Could not append to the internal temporary file.");
      return;
    }
    SyncImpl(gcs_file->bucket, gcs_file->object, &gcs_file->offset,
             &gcs_file->outfile, gcs_file->gcs_client, status);
    TF_VLog(3, "Flush finished: gs://%s/%s", gcs_file->bucket.c_str(),
            gcs_file->object.c_str());
    if (TF_GetCode(status) != TF_OK) return;
    gcs_file->sync_need = false;
  } else {
    TF_SetStatus(status, TF_OK, "");
  }
}

void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  TF_VLog(3, "Sync: gs://%s/%s", gcs_file->bucket.c_str(),
          gcs_file->object.c_str());
  Flush(file, status);
}

void Close(const TF_WritableFile* file, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(file->plugin_file);
  TF_VLog(3, "Close: gs://%s/%s", gcs_file->bucket.c_str(),
          gcs_file->object.c_str());
  if (gcs_file->sync_need) {
    Flush(file, status);
  }
  gcs_file->outfile.close();
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {
typedef struct GCSMemoryRegion {
  const void* const address;
  const uint64_t length;
} GCSMemoryRegion;

void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  plugin_memory_free(const_cast<void*>(r->address));
  delete r;
}

const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  return r->address;
}

uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<GCSMemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_gcs_filesystem {
// TODO(vnvo2409): Use partial reponse for better performance.
// TODO(vnvo2409): We could do some cleanups like `return TF_SetStatus`.
// TODO(vnvo2409): Refactor the filesystem implementation when
// https://github.com/googleapis/google-cloud-cpp/issues/4482 is done.
GCSFile::GCSFile(google::cloud::storage::Client&& gcs_client)
    : gcs_client(gcs_client), block_cache_lock() {
  const char* append_mode = std::getenv(kAppendMode);
  compose = (append_mode != nullptr) && (!strcmp(kAppendMode, append_mode));

  uint64_t value;
  block_size = kDefaultBlockSize;
  size_t max_bytes = kDefaultMaxCacheSize;
  uint64_t max_staleness = kDefaultMaxStaleness;

  // Apply the overrides for the block size (MB), max bytes (MB), and max
  // staleness (seconds) if provided.
  if (absl::SimpleAtoi(std::getenv(kBlockSize), &value)) {
    block_size = value * 1024 * 1024;
  }
  if (absl::SimpleAtoi(std::getenv(kMaxCacheSize), &value)) {
    max_bytes = static_cast<size_t>(value * 1024 * 1024);
  }
  if (absl::SimpleAtoi(std::getenv(kMaxStaleness), &value)) {
    max_staleness = value;
  }
  TF_VLog(1, "GCS cache max size = %u ; block size = %u ; max staleness = %u",
          max_bytes, block_size, max_staleness);

  file_block_cache = std::make_unique<RamFileBlockCache>(
      block_size, max_bytes, max_staleness,
      [this](const std::string& filename, size_t offset, size_t buffer_size,
             char* buffer, TF_Status* status) {
        return LoadBufferFromGCS(filename, offset, buffer_size, buffer, this,
                                 status);
      });

  uint64_t stat_cache_max_age = kStatCacheDefaultMaxAge;
  size_t stat_cache_max_entries = kStatCacheDefaultMaxEntries;
  if (absl::SimpleAtoi(std::getenv(kStatCacheMaxAge), &value)) {
    stat_cache_max_age = value;
  }
  if (absl::SimpleAtoi(std::getenv(kStatCacheMaxEntries), &value)) {
    stat_cache_max_entries = static_cast<size_t>(value);
  }
  stat_cache = std::make_unique<ExpiringLRUCache<GcsFileStat>>(
      stat_cache_max_age, stat_cache_max_entries);
}

GCSFile::GCSFile(google::cloud::storage::Client&& gcs_client, bool compose,
                 uint64_t block_size, size_t max_bytes, uint64_t max_staleness,
                 uint64_t stat_cache_max_age, size_t stat_cache_max_entries)
    : gcs_client(gcs_client),
      compose(compose),
      block_cache_lock(),
      block_size(block_size) {
  file_block_cache = std::make_unique<RamFileBlockCache>(
      block_size, max_bytes, max_staleness,
      [this](const std::string& filename, size_t offset, size_t buffer_size,
             char* buffer, TF_Status* status) {
        return LoadBufferFromGCS(filename, offset, buffer_size, buffer, this,
                                 status);
      });
  stat_cache = std::make_unique<ExpiringLRUCache<GcsFileStat>>(
      stat_cache_max_age, stat_cache_max_entries);
}

void InitTest(TF_Filesystem* filesystem, bool compose, uint64_t block_size,
              size_t max_bytes, uint64_t max_staleness,
              uint64_t stat_cache_max_age, size_t stat_cache_max_entries,
              TF_Status* status) {
  google::cloud::StatusOr<gcs::Client> client =
      gcs::Client::CreateDefaultClient();
  if (!client) {
    TF_SetStatusFromGCSStatus(client.status(), status);
    return;
  }

  filesystem->plugin_filesystem =
      new GCSFile(std::move(client.value()), compose, block_size, max_bytes,
                  max_staleness, stat_cache_max_age, stat_cache_max_entries);
  TF_SetStatus(status, TF_OK, "");
}

void Init(TF_Filesystem* filesystem, TF_Status* status) {
  google::cloud::StatusOr<gcs::Client> client =
      gcs::Client::CreateDefaultClient();
  if (!client) {
    TF_SetStatusFromGCSStatus(client.status(), status);
    return;
  }

  filesystem->plugin_filesystem = new GCSFile(std::move(client.value()));
  TF_SetStatus(status, TF_OK, "");
}

void Cleanup(TF_Filesystem* filesystem) {
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  delete gcs_file;
}

static void UncachedStatForObject(const std::string& bucket,
                                  const std::string& object, GcsFileStat* stat,
                                  gcs::Client* gcs_client, TF_Status* status) {
  auto metadata = gcs_client->GetObjectMetadata(
      bucket, object, gcs::Fields("generation,size,timeStorageClassUpdated"));
  if (!metadata) return TF_SetStatusFromGCSStatus(metadata.status(), status);
  stat->generation_number = metadata->generation();
  stat->base.length = metadata->size();
  stat->base.mtime_nsec =
      metadata->time_storage_class_updated().time_since_epoch().count();
  stat->base.is_directory = object.back() == '/';
  TF_VLog(1,
          "Stat of: gs://%s/%s --  length: %u generation: %u; mtime_nsec: %u;",
          bucket.c_str(), object.c_str(), stat->base.length,
          stat->generation_number, stat->base.mtime_nsec);
  return TF_SetStatus(status, TF_OK, "");
}

// TODO(vnvo2409): Implement later
void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  bool is_cache_enabled;
  {
    absl::MutexLock l(&gcs_file->block_cache_lock);
    is_cache_enabled = gcs_file->file_block_cache->IsCacheEnabled();
  }
  auto read_fn = [gcs_file, is_cache_enabled, bucket, object](
                     const std::string& path, uint64_t offset, size_t n,
                     char* buffer, TF_Status* status) -> int64_t {
    int64_t read = 0;
    if (is_cache_enabled) {
      absl::ReaderMutexLock l(&gcs_file->block_cache_lock);
      GcsFileStat stat;
      gcs_file->stat_cache->LookupOrCompute(
          path, &stat,
          [gcs_file, bucket, object](const std::string& path, GcsFileStat* stat,
                                     TF_Status* status) {
            UncachedStatForObject(bucket, object, stat, &gcs_file->gcs_client,
                                  status);
          },
          status);
      if (TF_GetCode(status) != TF_OK) return -1;
      if (!gcs_file->file_block_cache->ValidateAndUpdateFileSignature(
              path, stat.generation_number)) {
        TF_VLog(
            1,
            "File signature has been changed. Refreshing the cache. Path: %s",
            path.c_str());
      }
      read = gcs_file->file_block_cache->Read(path, offset, n, buffer, status);
    } else {
      read = LoadBufferFromGCS(path, offset, n, buffer, gcs_file, status);
    }
    if (TF_GetCode(status) != TF_OK) return -1;
    if (read < n)
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read less bytes than requested");
    else
      TF_SetStatus(status, TF_OK, "");
    return read;
  };
  file->plugin_file = new tf_random_access_file::GCSFile(
      std::move(path), is_cache_enabled, gcs_file->block_size, read_fn);
  TF_SetStatus(status, TF_OK, "");
}

void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  char* temp_file_name = TF_GetTempFileName("");
  file->plugin_file = new tf_writable_file::GCSFile(
      {std::move(bucket), std::move(object), &gcs_file->gcs_client,
       TempFile(temp_file_name, std::ios::binary | std::ios::out), true,
       (gcs_file->compose ? 0 : -1)});
  // We are responsible for freeing the pointer returned by TF_GetTempFileName
  free(temp_file_name);
  TF_VLog(3, "GcsWritableFile: %s", path);
  TF_SetStatus(status, TF_OK, "");
}

void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  char* temp_file_name_c_str = TF_GetTempFileName("");
  std::string temp_file_name(temp_file_name_c_str);  // To prevent memory-leak
  free(temp_file_name_c_str);

  if (!gcs_file->compose) {
    auto gcs_status =
        gcs_file->gcs_client.DownloadToFile(bucket, object, temp_file_name);
    TF_SetStatusFromGCSStatus(gcs_status, status);
    auto status_code = TF_GetCode(status);
    if (status_code != TF_OK && status_code != TF_NOT_FOUND) return;
    // If this file does not exist on server, we will need to sync it.
    bool sync_need = (status_code == TF_NOT_FOUND);
    file->plugin_file = new tf_writable_file::GCSFile(
        {std::move(bucket), std::move(object), &gcs_file->gcs_client,
         TempFile(temp_file_name, std::ios::binary | std::ios::app), sync_need,
         -1});
  } else {
    // If compose is true, we do not download anything.
    // Instead we only check if this file exists on server or not.
    auto metadata = gcs_file->gcs_client.GetObjectMetadata(bucket, object,
                                                           gcs::Fields("size"));
    TF_SetStatusFromGCSStatus(metadata.status(), status);
    if (TF_GetCode(status) == TF_OK) {
      file->plugin_file = new tf_writable_file::GCSFile(
          {std::move(bucket), std::move(object), &gcs_file->gcs_client,
           TempFile(temp_file_name, std::ios::binary | std::ios::trunc), false,
           static_cast<int64_t>(metadata->size())});
    } else if (TF_GetCode(status) == TF_NOT_FOUND) {
      file->plugin_file = new tf_writable_file::GCSFile(
          {std::move(bucket), std::move(object), &gcs_file->gcs_client,
           TempFile(temp_file_name, std::ios::binary | std::ios::trunc), true,
           0});
    } else {
      return;
    }
  }
  TF_VLog(3, "GcsWritableFile: %s with existing file %s", path,
          temp_file_name.c_str());
  TF_SetStatus(status, TF_OK, "");
}

// TODO(vnvo2409): We could download into a local temporary file and use
// memory-mapping.
void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                     const char* path,
                                     TF_ReadOnlyMemoryRegion* region,
                                     TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  auto metadata = gcs_file->gcs_client.GetObjectMetadata(bucket, object,
                                                         gcs::Fields("size"));
  if (!metadata) {
    TF_SetStatusFromGCSStatus(metadata.status(), status);
    return;
  }

  TF_RandomAccessFile reader;
  NewRandomAccessFile(filesystem, path, &reader, status);
  if (TF_GetCode(status) != TF_OK) return;
  char* buffer = static_cast<char*>(plugin_memory_allocate(metadata->size()));
  int64_t read =
      tf_random_access_file::Read(&reader, 0, metadata->size(), buffer, status);
  tf_random_access_file::Cleanup(&reader);
  if (TF_GetCode(status) != TF_OK) return;

  if (read > 0 && buffer) {
    region->plugin_memory_region =
        new tf_read_only_memory_region::GCSMemoryRegion(
            {buffer, static_cast<uint64_t>(read)});
    TF_SetStatus(status, TF_OK, "");
  } else if (read == 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "File is empty");
  }
}

static void StatForObject(GCSFile* gcs_file, const std::string& path,
                          const std::string& bucket, const std::string& object,
                          GcsFileStat* stat, TF_Status* status) {
  if (object.empty())
    return TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        absl::StrCat("'object' must be a non-empty string. (File: ", path, ")")
            .c_str());
  TF_SetStatus(status, TF_OK, "");
  gcs_file->stat_cache->LookupOrCompute(
      path, stat,
      [gcs_file, bucket, object](const std::string& path, GcsFileStat* stat,
                                 TF_Status* status) {
        UncachedStatForObject(bucket, object, stat, &gcs_file->gcs_client,
                              status);
      },
      status);
}

static bool ObjectExists(GCSFile* gcs_file, const std::string& path,
                         const std::string& bucket, const std::string& object,
                         TF_Status* status) {
  GcsFileStat stat;
  StatForObject(gcs_file, path, bucket, object, &stat, status);
  if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_NOT_FOUND)
    return false;
  if (TF_GetCode(status) == TF_NOT_FOUND) {
    TF_SetStatus(status, TF_OK, "");
    return false;
  }
  return !stat.base.is_directory;
}

static bool BucketExists(GCSFile* gcs_file, const std::string& bucket,
                         TF_Status* status) {
  auto metadata =
      gcs_file->gcs_client.GetBucketMetadata(bucket, gcs::Fields(""));
  TF_SetStatusFromGCSStatus(metadata.status(), status);
  if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_NOT_FOUND)
    return false;
  if (TF_GetCode(status) == TF_NOT_FOUND) {
    TF_SetStatus(status, TF_OK, "");
    return false;
  }
  return true;
}

static std::vector<std::string> GetChildrenBounded(
    GCSFile* gcs_file, std::string dir, uint64_t max_results, bool recursive,
    bool include_self_directory_marker, TF_Status* status) {
  std::string bucket, prefix;
  MaybeAppendSlash(&dir);
  ParseGCSPath(dir, true, &bucket, &prefix, status);

  std::vector<std::string> result;
  uint64_t count = 0;
  std::string delimiter = recursive ? "" : "/";

  for (auto&& item : gcs_file->gcs_client.ListObjectsAndPrefixes(
           bucket, gcs::Prefix(prefix), gcs::Delimiter(delimiter),
           gcs::Fields("items(name),prefixes"))) {
    if (count == max_results) {
      TF_SetStatus(status, TF_OK, "");
      return result;
    }
    if (!item) {
      TF_SetStatusFromGCSStatus(item.status(), status);
      return result;
    }
    auto value = *std::move(item);
    std::string children = absl::holds_alternative<std::string>(value)
                               ? absl::get<std::string>(value)
                               : absl::get<gcs::ObjectMetadata>(value).name();
    auto pos = children.find(prefix);
    if (pos != 0) {
      TF_SetStatus(status, TF_INTERNAL,
                   absl::StrCat("Unexpected response: the returned file name ",
                                children, " doesn't match the prefix ", prefix)
                       .c_str());
      return result;
    }
    children.erase(0, prefix.length());
    if (!children.empty() || include_self_directory_marker) {
      result.emplace_back(children);
    }
    ++count;
  }

  return result;
}

static bool FolderExists(GCSFile* gcs_file, std::string dir,
                         TF_Status* status) {
  ExpiringLRUCache<GcsFileStat>::ComputeFunc compute_func =
      [gcs_file](const std::string& dir, GcsFileStat* stat, TF_Status* status) {
        auto children =
            GetChildrenBounded(gcs_file, dir, 1, true, true, status);
        if (TF_GetCode(status) != TF_OK) return;
        if (!children.empty()) {
          stat->base = {0, 0, true};
          return TF_SetStatus(status, TF_OK, "");
        } else {
          return TF_SetStatus(status, TF_INVALID_ARGUMENT, "Not a directory!");
        }
      };
  GcsFileStat stat;
  MaybeAppendSlash(&dir);
  gcs_file->stat_cache->LookupOrCompute(dir, &stat, compute_func, status);
  if (TF_GetCode(status) != TF_OK && TF_GetCode(status) != TF_INVALID_ARGUMENT)
    return false;
  if (TF_GetCode(status) == TF_INVALID_ARGUMENT) {
    TF_SetStatus(status, TF_OK, "");
    return false;
  }
  return true;
}

static void ClearFileCaches(GCSFile* gcs_file, const std::string& path) {
  absl::ReaderMutexLock l(&gcs_file->block_cache_lock);
  gcs_file->file_block_cache->RemoveFile(path);
  gcs_file->stat_cache->Delete(path);
}

void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, true, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  if (object.empty()) {
    bool result = BucketExists(gcs_file, bucket, status);
    if (result) return TF_SetStatus(status, TF_OK, "");
  }

  GcsFileStat stat;
  StatForObject(gcs_file, path, bucket, object, &stat, status);
  if (TF_GetCode(status) != TF_NOT_FOUND) return;

  bool result = FolderExists(gcs_file, path, status);
  if (TF_GetCode(status) != TF_OK || (TF_GetCode(status) == TF_OK && result))
    return;
  return TF_SetStatus(
      status, TF_NOT_FOUND,
      absl::StrCat("The path ", path, " does not exist.").c_str());
}

void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  std::string dir = path;
  MaybeAppendSlash(&dir);
  TF_VLog(3,
          "CreateDir: creating directory with path: %s and "
          "path_with_slash: %s",
          path, dir.c_str());
  std::string bucket, object;
  ParseGCSPath(dir, true, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  if (object.empty()) {
    bool is_directory = BucketExists(gcs_file, bucket, status);
    if (TF_GetCode(status) != TF_OK) return;
    if (!is_directory)
      TF_SetStatus(status, TF_NOT_FOUND,
                   absl::StrCat("The specified bucket ", dir, " was not found.")
                       .c_str());
    return;
  }

  PathExists(filesystem, dir.c_str(), status);
  if (TF_GetCode(status) == TF_OK) {
    // Use the original name for a correct error here.
    TF_VLog(3, "CreateDir: directory already exists, not uploading %s", path);
    return TF_SetStatus(status, TF_ALREADY_EXISTS, path);
  }

  auto metadata = gcs_file->gcs_client.InsertObject(
      bucket, object, "",
      // Adding this parameter means HTTP_CODE_PRECONDITION_FAILED
      // will be returned if the object already exists, so avoid reuploading.
      gcs::IfGenerationMatch(0), gcs::Fields(""));
  TF_SetStatusFromGCSStatus(metadata.status(), status);
  if (TF_GetCode(status) == TF_FAILED_PRECONDITION)
    TF_SetStatus(status, TF_ALREADY_EXISTS, path);
}

// TODO(vnvo2409): `RecursivelyCreateDir` should use `CreateDir` instead of the
// default implementation. Because we could create an empty object whose
// key is equal to the `path` and Google Cloud Console will automatically
// display it as a directory tree.

void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  auto gcs_status = gcs_file->gcs_client.DeleteObject(bucket, object);
  TF_SetStatusFromGCSStatus(gcs_status, status);
  if (TF_GetCode(status) == TF_OK) ClearFileCaches(gcs_file, path);
}

// Checks that the directory is empty (i.e no objects with this prefix exist).
// Deletes the GCS directory marker if it exists.
void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  // A directory is considered empty either if there are no matching objects
  // with the corresponding name prefix or if there is exactly one matching
  // object and it is the directory marker. Therefore we need to retrieve
  // at most two children for the prefix to detect if a directory is empty.
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  auto childrens = GetChildrenBounded(gcs_file, path, 2, true, true, status);
  if (TF_GetCode(status) != TF_OK) return;
  if (childrens.size() > 1 || (childrens.size() == 1 && !childrens[0].empty()))
    return TF_SetStatus(status, TF_FAILED_PRECONDITION,
                        "Cannot delete a non-empty directory.");
  if (childrens.size() == 1 && childrens[0].empty()) {
    // This is the directory marker object. Delete it.
    std::string dir = path;
    MaybeAppendSlash(&dir);
    DeleteFile(filesystem, dir.c_str(), status);
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void CopyFile(const TF_Filesystem* filesystem, const char* src, const char* dst,
              TF_Status* status) {
  std::string bucket_src, object_src;
  ParseGCSPath(src, false, &bucket_src, &object_src, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string bucket_dst, object_dst;
  ParseGCSPath(dst, false, &bucket_dst, &object_dst, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  auto metadata = gcs_file->gcs_client.RewriteObjectBlocking(
      bucket_src, object_src, bucket_dst, object_dst,
      gcs::Fields("done,rewriteToken"));
  TF_SetStatusFromGCSStatus(metadata.status(), status);
}

bool IsDirectory(const TF_Filesystem* filesystem, const char* path,
                 TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, true, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return false;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  if (object.empty()) {
    bool result = BucketExists(gcs_file, bucket, status);
    if (TF_GetCode(status) != TF_OK) return false;
    if (!result)
      TF_SetStatus(
          status, TF_NOT_FOUND,
          absl::StrCat("The specified bucket gs://", bucket, " was not found.")
              .c_str());
    return result;
  }

  bool is_folder = FolderExists(gcs_file, path, status);
  if (TF_GetCode(status) != TF_OK) return false;
  if (is_folder) return true;

  bool is_object = ObjectExists(gcs_file, path, bucket, object, status);
  if (TF_GetCode(status) != TF_OK) return false;
  if (is_object) {
    TF_SetStatus(
        status, TF_FAILED_PRECONDITION,
        absl::StrCat("The specified path ", path, " is not a directory.")
            .c_str());
    return false;
  }
  TF_SetStatus(status, TF_NOT_FOUND,
               absl::StrCat("The path ", path, " does not exist.").c_str());
  return false;
}

static void RenameObject(const TF_Filesystem* filesystem,
                         const std::string& src, const std::string& dst,
                         TF_Status* status) {
  TF_VLog(3, "RenameObject: started %s to %s", src.c_str(), dst.c_str());
  std::string bucket_src, object_src;
  ParseGCSPath(src, false, &bucket_src, &object_src, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string bucket_dst, object_dst;
  ParseGCSPath(dst, false, &bucket_dst, &object_dst, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  auto metadata = gcs_file->gcs_client.RewriteObjectBlocking(
      bucket_src, object_src, bucket_dst, object_dst,
      gcs::Fields("done,rewriteToken"));
  TF_SetStatusFromGCSStatus(metadata.status(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_VLog(3, "RenameObject: finished %s to %s", src.c_str(), dst.c_str());

  ClearFileCaches(gcs_file, dst);
  DeleteFile(filesystem, src.c_str(), status);
}

void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status) {
  if (!IsDirectory(filesystem, src, status)) {
    if (TF_GetCode(status) == TF_FAILED_PRECONDITION) {
      TF_SetStatus(status, TF_OK, "");
      RenameObject(filesystem, src, dst, status);
    }
    return;
  }

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  std::vector<std::string> childrens =
      GetChildrenBounded(gcs_file, src, UINT64_MAX, true, true, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string src_dir = src;
  std::string dst_dir = dst;
  MaybeAppendSlash(&src_dir);
  MaybeAppendSlash(&dst_dir);
  for (const std::string& children : childrens) {
    RenameObject(filesystem, src_dir + children, dst_dir + children, status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void DeleteRecursively(const TF_Filesystem* filesystem, const char* path,
                       uint64_t* undeleted_files, uint64_t* undeleted_dirs,
                       TF_Status* status) {
  if (!undeleted_files || !undeleted_dirs)
    return TF_SetStatus(
        status, TF_INTERNAL,
        "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  if (!IsDirectory(filesystem, path, status)) {
    *undeleted_dirs = 1;
    return;
  }
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  std::vector<std::string> childrens =
      GetChildrenBounded(gcs_file, path, UINT64_MAX, true, true, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string dir = path;
  MaybeAppendSlash(&dir);
  for (const std::string& children : childrens) {
    const std::string& full_path = dir + children;
    DeleteFile(filesystem, full_path.c_str(), status);
    if (TF_GetCode(status) != TF_OK) {
      if (IsDirectory(filesystem, full_path.c_str(), status))
        // The object is a directory marker.
        (*undeleted_dirs)++;
      else
        (*undeleted_files)++;
    }
  }
}

int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status) {
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  std::vector<std::string> childrens =
      GetChildrenBounded(gcs_file, path, UINT64_MAX, false, false, status);
  if (TF_GetCode(status) != TF_OK) return -1;

  int num_entries = childrens.size();
  *entries = static_cast<char**>(
      plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
  for (int i = 0; i < num_entries; i++)
    (*entries)[i] = strdup(childrens[i].c_str());
  TF_SetStatus(status, TF_OK, "");
  return num_entries;
}

void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status) {
  std::string bucket, object;
  ParseGCSPath(path, true, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return;

  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  if (object.empty()) {
    auto bucket_metadata =
        gcs_file->gcs_client.GetBucketMetadata(bucket, gcs::Fields(""));
    TF_SetStatusFromGCSStatus(bucket_metadata.status(), status);
    if (TF_GetCode(status) == TF_OK) {
      stats->is_directory = true;
      stats->length = 0;
      stats->mtime_nsec = 0;
    }
    return;
  }
  if (IsDirectory(filesystem, path, status)) {
    stats->is_directory = true;
    stats->length = 0;
    stats->mtime_nsec = 0;
    return TF_SetStatus(status, TF_OK, "");
  }
  if (TF_GetCode(status) == TF_FAILED_PRECONDITION) {
    auto metadata = gcs_file->gcs_client.GetObjectMetadata(
        bucket, object, gcs::Fields("size,timeStorageClassUpdated"));
    if (metadata) {
      stats->is_directory = false;
      stats->length = metadata.value().size();
      stats->mtime_nsec = metadata.value()
                              .time_storage_class_updated()
                              .time_since_epoch()
                              .count();
    }
    TF_SetStatusFromGCSStatus(metadata.status(), status);
  }
}

int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status) {
  // Only validate the name.
  std::string bucket, object;
  ParseGCSPath(path, false, &bucket, &object, status);
  if (TF_GetCode(status) != TF_OK) return -1;

  TF_FileStatistics stat;
  Stat(filesystem, path, &stat, status);
  return stat.length;
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

static void FlushCaches(const TF_Filesystem* filesystem) {
  auto gcs_file = static_cast<GCSFile*>(filesystem->plugin_filesystem);
  absl::ReaderMutexLock l(&gcs_file->block_cache_lock);
  gcs_file->file_block_cache->Flush();
  gcs_file->stat_cache->Clear();
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
  ops->filesystem_ops->delete_dir = tf_gcs_filesystem::DeleteDir;
  ops->filesystem_ops->delete_recursively =
      tf_gcs_filesystem::DeleteRecursively;
  ops->filesystem_ops->copy_file = tf_gcs_filesystem::CopyFile;
  ops->filesystem_ops->path_exists = tf_gcs_filesystem::PathExists;
  ops->filesystem_ops->is_directory = tf_gcs_filesystem::IsDirectory;
  ops->filesystem_ops->stat = tf_gcs_filesystem::Stat;
  ops->filesystem_ops->get_children = tf_gcs_filesystem::GetChildren;
  ops->filesystem_ops->translate_name = tf_gcs_filesystem::TranslateName;
  ops->filesystem_ops->flush_caches = tf_gcs_filesystem::FlushCaches;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "gs");
}
