/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/asm_compiler.h"

#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/subprocess.h"

namespace stream_executor {

static std::string findRocmExecutable(const std::string& binary_relative_path,
                                      const std::string& rocm_root_dir) {
  auto env = tsl::Env::Default();
  std::string binary_path =
      tsl::io::JoinPath(rocm_root_dir, binary_relative_path);
  VLOG(2) << "Looking for " << binary_relative_path << " at " << rocm_root_dir;
  if (!env->FileExists(binary_path).ok()) {
    binary_path = absl::StrCat("<", binary_path, " - NOT FOUND>");
  }
  return binary_path;
}

absl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<HsacoImage> images, const std::string rocm_root_dir) {
  std::string clang_offload_bundler_path =
      findRocmExecutable("llvm/bin/clang-offload-bundler", rocm_root_dir);

  // Initialise the "--inputs" / "--targets" arguments for the
  // clang-offload-bundler with a dummy file / host target triple...
  // clang-offload-bundler requires 1 and only 1 host target triple
  std::ostringstream inputs_list;
  std::ostringstream targets_list;

  inputs_list << "/dev/null";
  targets_list << "host-x86_64-unknown-linux";

  // Write images to temporary files.
  std::vector<std::string> image_paths;
  auto env = tsl::Env::Default();
  for (const HsacoImage& img : images) {
    std::string img_path;
    if (!env->LocalTempFilename(&img_path)) {
      return tsl::errors::Internal(
          "Could not get temporary filenames for images.");
    }
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
        env, img_path, std::string(img.bytes.begin(), img.bytes.end())));
    VLOG(2) << "image written to " << img_path;
    inputs_list << "," << img_path;
    targets_list << ",hip-amdgcn-amd-amdhsa-" << img.gfx_arch;
    image_paths.push_back(std::move(img_path));
  }
  absl::Cleanup image_files_cleaner = [&image_paths] {
    for (const auto& path : image_paths) {
      TF_CHECK_OK(tsl::Env::Default()->DeleteFile(path));
    }
  };

  // Prepare temorary result file.
  std::string result_path;
  if (!env->LocalTempFilename(&result_path)) {
    return tsl::errors::Internal(
        "Could not get temporary filename for fatbin result.");
  }
  absl::Cleanup result_file_cleaner = [&result_path] {
    // This file may never be created, so the failure to delete it should not
    // propagate to TF.
    tsl::Env::Default()->DeleteFile(result_path).IgnoreError();
  };

  // Invoke clang_offload_bundler and collect its output.
  tsl::SubProcess clang_offload_bundler;
  std::vector<std::string> clang_offload_bundler_args = {
      clang_offload_bundler_path, absl::StrCat("--inputs=", inputs_list.str()),
      absl::StrCat("--targets=", targets_list.str()), "--type=o",
      absl::StrCat("--outputs=", result_path)};
  if (VLOG_IS_ON(3)) {
    VLOG(3) << absl::StrJoin(clang_offload_bundler_args, " ");
  }
  clang_offload_bundler.SetProgram(clang_offload_bundler_path,
                                   clang_offload_bundler_args);
  clang_offload_bundler.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!clang_offload_bundler.Start()) {
    return tsl::errors::Internal("Failed to launch clang_offload_bundler.");
  }
  std::string stderr_output;
  int exit_status = clang_offload_bundler.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  if (exit_status != 0) {
    return tsl::errors::Internal(
        absl::StrFormat("clang_offload_bundler exited with non-zero error "
                        "code %d, output: %s",
                        exit_status, stderr_output));
  }
  if (!stderr_output.empty()) {
    VLOG(2) << stderr_output;
  }

  // Read in the result and return it as a byte vector.
  std::string result_blob;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), result_path, &result_blob));
  return std::vector<uint8_t>(result_blob.begin(), result_blob.end());
}

}  // namespace stream_executor
