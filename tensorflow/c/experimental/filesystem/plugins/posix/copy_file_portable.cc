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
#include <stdint.h>
#include <unistd.h>

#include <memory>

#include "tensorflow/c/experimental/filesystem/plugins/posix/copy_file.h"

namespace tf_posix_filesystem {

// Transfers up to `size` bytes from `dst_fd` to `src_fd`.
//
// This method uses a temporary buffer to hold contents.
int CopyFileContents(int dst_fd, int src_fd, off_t size) {
  // Use a copy buffer of 128KB but don't store it on the stack.
  constexpr static size_t kPosixCopyFileBufferSize = 128 * 1024;
  std::unique_ptr<char[]> buffer(new char[kPosixCopyFileBufferSize]);

  off_t offset = 0;
  int bytes_transferred = 0;
  int rc = 1;
  // When `sendfile` returns 0 we stop copying and let callers handle this.
  while (offset < size && rc > 0) {
    size_t chunk = size - offset;
    if (chunk > kPosixCopyFileBufferSize) chunk = kPosixCopyFileBufferSize;

    rc = read(src_fd, buffer.get(), chunk);
    if (rc < 0) return -1;

    int total_write = 0;
    int total_read = rc;
    while (total_write < total_read && rc > 0) {
      rc = write(dst_fd, buffer.get() + total_write, chunk - total_write);
      if (rc < 0) return -1;

      total_write += rc;
      bytes_transferred += rc;
      offset += rc;
    }
  }

  return bytes_transferred;
}

}  // namespace tf_posix_filesystem
