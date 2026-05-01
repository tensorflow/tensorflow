/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TSL_UTIL_MEMFILE_BUILTIN_H_
#define XLA_TSL_UTIL_MEMFILE_BUILTIN_H_

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "xla/tsl/util/file_toc.h"

namespace tsl::memfile {

// Registers all entries in `toc` at embed://name/<filename>.
absl::Status RegisterBuiltInFiles(const char* absl_nonnull name,
                                  const FileToc toc[absl_nonnull]);

// Version of the above for global registration. Do not use.
bool GlobalRegisterFiles(const char* absl_nonnull name,
                         const FileToc toc[absl_nonnull]);

}  // namespace tsl::memfile

#define REGISTER_BUILTIN_FILES_WITH_DIRECTORY_HELPER(ctr, name, directory) \
  static bool memfile_register_##ctr ABSL_ATTRIBUTE_UNUSED =               \
      tsl::memfile::GlobalRegisterFiles(directory, name##_create());

#define REGISTER_BUILTIN_FILES_WITH_DIRECTORY(name, directory) \
  REGISTER_BUILTIN_FILES_WITH_DIRECTORY_HELPER(__COUNTER__, name, directory);

#endif  // XLA_TSL_UTIL_MEMFILE_BUILTIN_H_
