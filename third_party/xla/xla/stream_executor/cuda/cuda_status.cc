/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_status.h"

#include <dirent.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"

namespace stream_executor::cuda::internal {

namespace {
void LogLdLibraryPathContents() {
  char* ld_library_path;
  if ((ld_library_path = getenv("LD_LIBRARY_PATH")) == nullptr) {
    return;
  }
  LOG(ERROR) << "LD_LIBRARY_PATH: " << ld_library_path;
  std::vector<absl::string_view> paths = absl::StrSplit(ld_library_path, ':');
  for (absl::string_view path : paths) {
    LOG(ERROR) << "path: " << path;
    // Print dir content of the path using standard os api.
    DIR* dir = opendir(std::string(path).c_str());
    if (dir == nullptr) {
      LOG(ERROR) << "Failed to open directory: " << path << ": "
                 << strerror(errno);
      continue;
    }
    LOG(ERROR) << "Contents of directory: " << path;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string file_name = entry->d_name;
      if (file_name == "." || file_name == "..") {
        continue;
      }
      std::string full_path = absl::StrCat(path, "/", file_name);
      struct stat sb;
      if (stat(full_path.c_str(), &sb) == -1) {
        LOG(ERROR) << "  " << file_name << ": stat failed: " << strerror(errno);
      } else {
        LOG(ERROR) << "  " << file_name << " size: " << sb.st_size
                   << " mode: " << sb.st_mode;
      }
    }
    LOG(ERROR) << "Closing directory: " << path;
    closedir(dir);
  }
  LOG(ERROR) << "Done with LD_LIBRARY_PATH: " << ld_library_path;
}
}  // namespace

absl::Status ToStatusSlow(CUresult result, absl::string_view detail) {
  const char* error_name;
  std::string error_detail;
  LOG(ERROR) << "ToStatusSlow: " << detail;
  if (cuGetErrorName(result, &error_name)) {
    error_detail = absl::StrCat(detail, ": UNKNOWN ERROR (",
                                static_cast<int>(result), ")");
  } else {
    const char* error_string;
    if (cuGetErrorString(result, &error_string)) {
      error_detail = absl::StrCat(detail, ": ", error_name);
    } else {
      error_detail = absl::StrCat(detail, ": ", error_name, ": ", error_string);
    }
  }

  LogLdLibraryPathContents();
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    LOG(ERROR) << "CUDA_ERROR_OUT_OF_MEMORY";
    return absl::ResourceExhaustedError(error_detail);
  }
  if (result == CUDA_ERROR_NOT_FOUND) {
    LOG(ERROR) << "CUDA_ERROR_NOT_FOUND";
    // Look at all the LD_LIBRARY directory paths that are searched for the
    // libcuda.so file.
    return absl::NotFoundError(error_detail);
  }
  LOG(ERROR) << "Not CUDA_ERROR_NOT_FOUND";
  return absl::InternalError(absl::StrCat("CUDA error: ", error_detail));
}

absl::Status ToStatusSlow(cudaError_t result, absl::string_view detail) {
  std::string error_detail(detail);
  const char* error_name = cudaGetErrorName(result);
  const char* error_string = cudaGetErrorString(result);
  if (error_name == nullptr) {
    absl::StrAppend(&error_detail, ": UNKNOWN ERROR (",
                    static_cast<int>(result), ")");
  } else {
    absl::StrAppend(&error_detail, ": ", error_name);
  }

  if (error_string != nullptr) {
    absl::StrAppend(&error_detail, ": ", error_string);
  }

  LogLdLibraryPathContents();

  return absl::InternalError(
      absl::StrCat("CUDA Runtime error: ", error_detail));
}

}  // namespace stream_executor::cuda::internal
