/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_ELF_SECTION_EXTRACTION_H_
#define XLA_STREAM_EXECUTOR_GPU_ELF_SECTION_EXTRACTION_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

// Takes a static library archive or object (ELF) file contents (.a file, .so
// file, or .o file), iterates over all the files in the archive (if applicable)
// and searches for a section that starts with the name `section_name`. Returns
// the contents of that section as a string in the first found ELF file.
absl::StatusOr<std::string> GetSectionContentsFromLibraryDump(
    absl::string_view library_dump, absl::string_view section_name);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ELF_SECTION_EXTRACTION_H_
