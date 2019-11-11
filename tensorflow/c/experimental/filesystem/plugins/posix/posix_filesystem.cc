/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <vector>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for POSIX environments.
// This filesystem will support `file://` and empty (local) URI schemes.

// TODO(mihaimaruseac): More implementations to follow in subsequent changes.

// SECTION 1. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

typedef struct PosixFile {
  const char* filename;
  FILE* handle;
} PosixFile;

static void Cleanup(TF_WritableFile* file) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  if (posix_file->handle != nullptr) fclose(posix_file->handle);
  free(const_cast<char*>(posix_file->filename));
  delete posix_file;
}

}  // namespace tf_writable_file

// SECTION 2. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_posix_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  FILE* f = fopen(path, "w");
  if (f == nullptr) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  file->plugin_file = new tf_writable_file::PosixFile({strdup(path), f});
  TF_SetStatus(status, TF_OK, "");
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  if (strlen(path) == 0)
    TF_SetStatus(status, TF_ALREADY_EXISTS, "already exists");
  else if (mkdir(path, /*mode=*/0755) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

}  // namespace tf_posix_filesystem

void TF_InitPlugin(TF_Status* status) {
  TF_WritableFileOps writable_file_ops = {tf_writable_file::Cleanup, nullptr};
  TF_FilesystemOps filesystem_ops = {
      tf_posix_filesystem::Init,
      tf_posix_filesystem::Cleanup,
      /*new_random_access_file=*/nullptr,
      tf_posix_filesystem::NewWritableFile,
      /*new_appendable_file=*/nullptr,
      /*new_read_only_memory_region_from_file=*/nullptr,
      tf_posix_filesystem::CreateDir,
      nullptr};

  for (const char* scheme : {"", "file"})
    TF_REGISTER_FILESYSTEM_PLUGIN(
        scheme, &filesystem_ops, /*pluginRandomAccessFileOps=*/nullptr,
        &writable_file_ops, /*pluginReadOnlyMemoryRegionOps=*/nullptr, status);
}
