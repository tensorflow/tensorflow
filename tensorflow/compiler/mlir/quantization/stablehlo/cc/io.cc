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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stablehlo::quantization::io {

absl::StatusOr<std::string> GetLocalTmpFileName(tsl::Env* const env) {
  std::string tmp_fname{};
  if (!env->LocalTempFilename(&tmp_fname)) {
    return absl::InternalError("Failed to create tmp file name.");
  }

  return tmp_fname;
}

absl::StatusOr<std::string> GetLocalTmpFileName() {
  return GetLocalTmpFileName(tsl::Env::Default());
}

absl::StatusOr<std::string> CreateTmpDir(tsl::Env* const env) {
  TF_ASSIGN_OR_RETURN(std::string tmp_dir, GetLocalTmpFileName(env));

  if (!env->RecursivelyCreateDir(tmp_dir).ok()) {
    return absl::InternalError(
        absl::StrFormat("Failed to create tmp dir: '%s'", tmp_dir));
  }

  return tmp_dir;
}

absl::StatusOr<std::string> CreateTmpDir() {
  // The overloaded function uses the default env.
  return CreateTmpDir(tsl::Env::Default());
}

absl::Status WriteStringToFile(const absl::string_view file_path,
                               const absl::string_view data) {
  auto* env = tsl::Env::Default();
  return WriteStringToFile(env, std::string(file_path), data);
}

absl::StatusOr<std::string> ReadFileToString(
    const absl::string_view file_path) {
  auto* env = tsl::Env::Default();
  std::string data{};
  absl::Status read_status =
      ReadFileToString(env, std::string(file_path), &data);

  if (read_status.ok()) {
    return data;
  } else {
    return read_status;
  }
}

absl::StatusOr<std::vector<std::string>> ListDirectory(
    absl::string_view directory) {
  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetChildren(std::string(directory), &children));
  return children;
}

}  // namespace stablehlo::quantization::io
