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
#include <stdlib.h>
#include <string.h>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for POSIX environments.
// This filesystem will support `file://` and empty (local) URI schemes.

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

// TODO(mihaimaruseac): Implement later

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

// TODO(mihaimaruseac): Implement later

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

// TODO(mihaimaruseac): Implement later

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_windows_filesystem {

// TODO(mihaimaruseac): Implement later

}  // namespace tf_windows_filesystem

int TF_InitPlugin(void* (*allocator)(size_t), TF_FilesystemPluginInfo** info) {
  const int num_schemes = 2;
  *info = static_cast<TF_FilesystemPluginInfo*>(
      allocator(num_schemes * sizeof((*info)[0])));

  for (int i = 0; i < num_schemes; i++) {
    TF_FilesystemPluginInfo* current_info = &((*info)[i]);
    TF_SetFilesystemVersionMetadata(current_info);
  }

  (*info)[0].scheme = strdup("");
  (*info)[1].scheme = strdup("file");

  return num_schemes;
}
