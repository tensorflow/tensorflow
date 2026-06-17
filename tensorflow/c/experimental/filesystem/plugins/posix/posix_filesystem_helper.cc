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
#include "tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem_helper.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tensorflow/c/experimental/filesystem/plugins/posix/copy_file.h"

namespace tf_posix_filesystem {

int TransferFileContents(const char* src, const char* dst, mode_t mode,
                         off_t size) {
  int src_fd = open(src, O_RDONLY);
  if (src_fd < 0) return -1;

  // When creating file, use the same permissions as original
  mode_t open_mode = mode & (S_IRWXU | S_IRWXG | S_IRWXO);

  // O_WRONLY | O_CREAT | O_TRUNC:
  //   Open file for write and if file does not exist, create the file.
  //   If file exists, truncate its size to 0.
  int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, open_mode);
  if (dst_fd < 0) {
    close(src_fd);
    return -1;
  }

  // Both files have been opened, do the transfer.
  // Since errno would be overridden by `close` below, save it here.
  int error_code = 0;
  if (CopyFileContents(dst_fd, src_fd, size) < 0) error_code = errno;

  close(src_fd);
  close(dst_fd);
  if (error_code != 0) {
    errno = error_code;
    return -1;
  } else {
    return 0;
  }
}

int RemoveSpecialDirectoryEntries(const struct dirent* entry) {
  return strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0;
}

}  // namespace tf_posix_filesystem
