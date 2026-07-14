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
#include <limits.h>
#include <stdint.h>
#include <sys/sendfile.h>

#include <cstddef>

#include "tensorflow/c/experimental/filesystem/plugins/posix/copy_file.h"

namespace tf_posix_filesystem {

// Transfers up to `size` bytes from `dst_fd` to `src_fd`.
//
// This method uses `sendfile` specific to linux after 2.6.33.
int CopyFileContents(int dst_fd, int src_fd, off_t size) {
  off_t offset = 0;
  int bytes_transferred = 0;
  int rc = 1;
  // When `sendfile` returns 0 we stop copying and let callers handle this.
  while (offset < size && rc > 0) {
    // Use uint64 for safe compare SSIZE_MAX
    uint64_t chunk = size - offset;
    if (chunk > SSIZE_MAX) chunk = SSIZE_MAX;

    rc = sendfile(dst_fd, src_fd, &offset, static_cast<size_t>(chunk));
    if (rc < 0) return -1;
    bytes_transferred += rc;
  }

  return bytes_transferred;
}

}  // namespace tf_posix_filesystem
