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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_COPY_FILE_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_COPY_FILE_H_

#include <sys/stat.h>

namespace tf_posix_filesystem {

// Transfers up to `size` bytes from `dst_fd` to `src_fd`.
//
// This method uses `sendfile` if available (i.e., linux 2.6.33 or later) or an
// intermediate buffer if not.
//
// Returns number of bytes transferred or -1 on failure.
int CopyFileContents(int dst_fd, int src_fd, off_t size);

}  // namespace tf_posix_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_COPY_FILE_H_
