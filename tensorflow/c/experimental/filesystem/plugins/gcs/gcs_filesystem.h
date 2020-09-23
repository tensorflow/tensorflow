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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_GCS_FILESYSTEM_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_GCS_FILESYSTEM_H_

#include "google/cloud/storage/client.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h"
#include "tensorflow/c/experimental/filesystem/plugins/gcs/ram_file_block_cache.h"
#include "tensorflow/c/tf_status.h"

void ParseGCSPath(const std::string& fname, bool object_empty_ok,
                  std::string* bucket, std::string* object, TF_Status* status);

namespace tf_random_access_file {
void Cleanup(TF_RandomAccessFile* file);
int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
             char* buffer, TF_Status* status);
}  // namespace tf_random_access_file

namespace tf_writable_file {
void Cleanup(TF_WritableFile* file);
void Append(const TF_WritableFile* file, const char* buffer, size_t n,
            TF_Status* status);
int64_t Tell(const TF_WritableFile* file, TF_Status* status);
void Flush(const TF_WritableFile* file, TF_Status* status);
void Sync(const TF_WritableFile* file, TF_Status* status);
void Close(const TF_WritableFile* file, TF_Status* status);
}  // namespace tf_writable_file

namespace tf_read_only_memory_region {
void Cleanup(TF_ReadOnlyMemoryRegion* region);
const void* Data(const TF_ReadOnlyMemoryRegion* region);
uint64_t Length(const TF_ReadOnlyMemoryRegion* region);
}  // namespace tf_read_only_memory_region

namespace tf_gcs_filesystem {
typedef struct GcsFileStat {
  TF_FileStatistics base;
  int64_t generation_number;
} GcsFileStat;

typedef struct GCSFile {
  google::cloud::storage::Client gcs_client;  // owned
  bool compose;
  absl::Mutex block_cache_lock;
  std::shared_ptr<RamFileBlockCache> file_block_cache
      ABSL_GUARDED_BY(block_cache_lock);
  uint64_t block_size;  // Reads smaller than block_size will trigger a read
                        // of block_size.
  std::unique_ptr<ExpiringLRUCache<GcsFileStat>> stat_cache;
  GCSFile(google::cloud::storage::Client&& gcs_client);
  // This constructor is used for testing purpose only.
  GCSFile(google::cloud::storage::Client&& gcs_client, bool compose,
          uint64_t block_size, size_t max_bytes, uint64_t max_staleness,
          uint64_t stat_cache_max_age, size_t stat_cache_max_entries);
} GCSFile;

// This function is used to initialize a filesystem without the need of setting
// manually environement variables.
void InitTest(TF_Filesystem* filesystem, bool compose, uint64_t block_size,
              size_t max_bytes, uint64_t max_staleness,
              uint64_t stat_cache_max_age, size_t stat_cache_max_entries,
              TF_Status* status);

void Init(TF_Filesystem* filesystem, TF_Status* status);
void Cleanup(TF_Filesystem* filesystem);
void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status);
void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status);
void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status);
void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                     const char* path,
                                     TF_ReadOnlyMemoryRegion* region,
                                     TF_Status* status);
int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status);
void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status);
void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status);
int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status);
void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status);
void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status);
void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status);
void CopyFile(const TF_Filesystem* filesystem, const char* src, const char* dst,
              TF_Status* status);
void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status);
}  // namespace tf_gcs_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_GCS_FILESYSTEM_H_
