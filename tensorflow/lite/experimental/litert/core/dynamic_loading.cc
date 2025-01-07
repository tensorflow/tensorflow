// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/dynamic_loading.h"

#include <dlfcn.h>

#ifndef __ANDROID__
#if __has_include(<link.h>)
#include <link.h>
#endif
#endif

#include <cstddef>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert::internal {

LiteRtStatus OpenLib(absl::string_view so_path, void** lib_handle) {
#ifdef RTLD_DEEPBIND
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
#else
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL);
#endif

  if (res == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to load .so at path: %s\n",
               so_path.data());
    LogDlError();

    return kLiteRtStatusErrorDynamicLoading;
  }
  *lib_handle = res;
  return kLiteRtStatusOk;
}

LiteRtStatus CloseLib(void* lib_handle) {
  if (0 != ::dlclose(lib_handle)) {
    LITERT_LOG(LITERT_ERROR, "Failed to close .so with error: %s", ::dlerror());
    return kLiteRtStatusErrorDynamicLoading;
  }
  return kLiteRtStatusOk;
}

namespace {

LiteRtStatus FindLiteRtSharedLibsHelper(const std::string& search_path,
                                        std::vector<std::string>& results) {
  if (!std::filesystem::exists(search_path)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const std::string compiler_plugin_lib_pattern =
      absl::StrFormat("%s%s", kLiteRtSharedLibPrefix, "CompilerPlugin");
  for (const auto& entry : std::filesystem::directory_iterator(search_path)) {
    const auto& path = entry.path();
    if (entry.is_regular_file()) {
      auto stem = path.stem().string();
      auto ext = path.extension().string();
      if (stem.find(compiler_plugin_lib_pattern) == 0 && ext == ".so") {
        results.push_back(path);
      }
    } else if (entry.is_directory()) {
      FindLiteRtSharedLibsHelper(path, results);
    }
  }

  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus FindLiteRtSharedLibs(absl::string_view search_path,
                                  std::vector<std::string>& results) {
  std::string root(search_path.data());
  return FindLiteRtSharedLibsHelper(root, results);
}

}  // namespace litert::internal
