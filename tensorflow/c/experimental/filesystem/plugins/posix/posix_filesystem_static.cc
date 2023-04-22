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

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/modular_filesystem_registration.h"
#include "tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem.h"

namespace tensorflow {

// Register the POSIX filesystems statically.
// Return value will be unused
bool StaticallyRegisterLocalFilesystems() {
  TF_FilesystemPluginInfo info;
  TF_InitPlugin(&info);
  Status status = filesystem_registration::RegisterFilesystemPluginImpl(&info);
  if (!status.ok()) {
    VLOG(0) << "Static POSIX filesystem could not be registered: " << status;
    return false;
  }
  return true;
}

// Perform the actual registration
static bool unused = StaticallyRegisterLocalFilesystems();

}  // namespace tensorflow
