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
#include <vector>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for POSIX environments.
// This filesystem will support `file://` and empty (local) URI schemes.

// TODO(mihaimaruseac): More implementations to follow in subsequent changes.

namespace posix_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

}  // namespace posix_filesystem

void TF_InitPlugin(TF_Status* status) {
  TF_FilesystemOps filesystem_ops = {posix_filesystem::Init,
                                     posix_filesystem::Cleanup, nullptr};

  for (const char* scheme : {"", "file"})
    TF_REGISTER_FILESYSTEM_PLUGIN(scheme, &filesystem_ops,
                                  /*pluginRandomAccessFileOps=*/nullptr,
                                  /*pluginWritableFileOps=*/nullptr,
                                  /*pluginReadOnlyMemoryRegionOps=*/nullptr,
                                  status);
}
