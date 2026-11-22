/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FILE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FILE_UTILS_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"

// Small file helpers for the converter, built only on LLVM's support library.
// They operate on the local file system.
namespace mlir::TFL {

// Joins `dir` and `name` with exactly one '/' between them. Returns the other
// argument if either one is empty.
std::string JoinPath(absl::string_view dir, absl::string_view name);

// Creates `path` and any missing parent directories. Succeeds if the directory
// already exists.
absl::Status CreateDirectories(absl::string_view path);

// Opens `path` for writing, truncating it if it exists. Returns
// InvalidArgument for "-", which `llvm::raw_fd_ostream` would treat as stdout.
//
// The stream is buffered. Callers must call `close()`, then check
// `has_error()` and call `clear_error()` before destroying it: LLVM aborts if
// a stream is destroyed with a pending error.
absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> OpenFileForWriting(
    absl::string_view path);

// Writes `contents` to `path`, replacing any existing file.
absl::Status WriteStringToFile(absl::string_view path,
                               absl::string_view contents);

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FILE_UTILS_H_
