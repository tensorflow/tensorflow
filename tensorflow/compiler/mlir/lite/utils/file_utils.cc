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

#include "tensorflow/compiler/mlir/lite/utils/file_utils.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::TFL {
namespace {

absl::Status FileError(absl::string_view action, absl::string_view path,
                       absl::string_view message) {
  return absl::UnknownError(
      absl::StrCat("Failed to ", action, " '", path, "': ", message));
}

}  // namespace

std::string JoinPath(absl::string_view dir, absl::string_view name) {
  if (dir.empty()) return std::string(name);
  if (name.empty()) return std::string(dir);
  const bool dir_has_slash = dir.back() == '/';
  const bool name_has_slash = name.front() == '/';
  if (dir_has_slash && name_has_slash) {
    return absl::StrCat(dir, name.substr(1));
  }
  if (dir_has_slash || name_has_slash) {
    return absl::StrCat(dir, name);
  }
  return absl::StrCat(dir, "/", name);
}

absl::Status CreateDirectories(absl::string_view path) {
  if (auto error = llvm::sys::fs::create_directories(
          llvm::StringRef(path.data(), path.size()))) {
    return FileError("create directory", path, error.message());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> OpenFileForWriting(
    absl::string_view path) {
  // `llvm::raw_fd_ostream` maps "-" to stdout, which must not be closed.
  if (path == "-") {
    return absl::InvalidArgumentError(
        "Writing to \"-\" (stdout) is not supported");
  }
  std::error_code error;
  auto os = std::make_unique<llvm::raw_fd_ostream>(
      llvm::StringRef(path.data(), path.size()), error);
  if (error) return FileError("open", path, error.message());
  return os;
}

absl::Status WriteStringToFile(absl::string_view path,
                               absl::string_view contents) {
  absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> os =
      OpenFileForWriting(path);
  if (!os.ok()) return os.status();
  (*os)->write(contents.data(), contents.size());
  (*os)->close();
  if ((*os)->has_error()) {
    std::string message = (*os)->error().message();
    (*os)->clear_error();
    return FileError("write", path, message);
  }
  return absl::OkStatus();
}

}  // namespace mlir::TFL
