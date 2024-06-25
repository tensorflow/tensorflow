/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_IO_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_IO_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"

namespace stablehlo::quantization::io {

// Generates a unique local tmp file name. This function only generates the name
// (path) and doesn't actually creates the file.
absl::StatusOr<std::string> GetLocalTmpFileName(tsl::Env* env);

// Generates a unique local tmp file name. This function only generates the name
// (path) and doesn't actually creates the file. The default environment
// `tsl::Env::Default` is used to generate the name.
absl::StatusOr<std::string> GetLocalTmpFileName();

// Creates a temporary directory on an environment defined by the implementation
// of `tsl::Env` and returns its path. Returns an InternalError status if
// failed.
absl::StatusOr<std::string> CreateTmpDir(tsl::Env* env);

// Creates a temporary directory and returns its path. Returns an InternalError
// status if failed. The file system used will be the default environment
// returned by `tsl::Env::Default`.
absl::StatusOr<std::string> CreateTmpDir();

// Convenience function for writing string `data` to file without the need to
// pass `tsl::Env` instance. Internally it uses the default `tsl::Env::Default`.
absl::Status WriteStringToFile(absl::string_view file_path,
                               absl::string_view data);

// Convenience function for reading string data from file at `file_path` without
// the need to pass `tsl::Env` instance. Internally it uses the default
// `tsl::Env::Default`. Returns an OK status with string data containing file
// contents. Returns non-ok status upon error, e.g. file doesn't exist.
absl::StatusOr<std::string> ReadFileToString(absl::string_view file_path);

// Lists all files and directories under the given directory.
absl::StatusOr<std::vector<std::string>> ListDirectory(
    absl::string_view directory);

template <class MessageT>
absl::StatusOr<MessageT> ReadBinaryProto(const std::string& binary_file_path) {
  MessageT message;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), binary_file_path, &message));
  return message;
}

}  // namespace stablehlo::quantization::io

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_IO_H_
