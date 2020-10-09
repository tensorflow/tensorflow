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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_HADOOP_HADOOP_FILESYSTEM_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_HADOOP_HADOOP_FILESYSTEM_H_

#include <map>
#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"
#include "third_party/hadoop/hdfs.h"

void ParseHadoopPath(const std::string& fname, std::string* scheme,
                     std::string* namenode, std::string* path);
void SplitArchiveNameAndPath(std::string* path, std::string* nn,
                             TF_Status* status);
class LibHDFS;

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
void Sync(const TF_WritableFile* file, TF_Status* status);
void Flush(const TF_WritableFile* file, TF_Status* status);
void Close(const TF_WritableFile* file, TF_Status* status);
}  // namespace tf_writable_file

namespace tf_hadoop_filesystem {
typedef struct HadoopFile {
  LibHDFS* libhdfs;
  absl::Mutex connection_cache_lock;
  std::map<std::string, hdfsFS> connection_cache
      ABSL_GUARDED_BY(connection_cache_lock);
  HadoopFile(TF_Status* status);
} HadoopFile;

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
void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status);
void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status);
int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status);
void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status);
void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status);
void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status);
void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status);
int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status);
}  // namespace tf_hadoop_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_HADOOP_HADOOP_FILESYSTEM_H_
