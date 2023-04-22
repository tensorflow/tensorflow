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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_HELPER_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_HELPER_H_

#include <dirent.h>
#include <sys/stat.h>

namespace tf_posix_filesystem {

// Copies up to `size` of `src` to `dst`, creating destination if needed.
//
// Callers should pass size of `src` in `size` and the permissions of `src` in
// `mode`. The later is only used if `dst` needs to be created.
int TransferFileContents(const char* src, const char* dst, mode_t mode,
                         off_t size);

// Returns true only if `entry` points to an entry other than `.` or `..`.
//
// This is a filter for `scandir`.
int RemoveSpecialDirectoryEntries(const struct dirent* entry);

}  // namespace tf_posix_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_HELPER_H_
