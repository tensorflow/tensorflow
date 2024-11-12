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
#include <glob.h>
#include <link.h>
#endif

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert::internal {

LiteRtStatus OpenLib(absl::string_view so_path, void** lib_handle) {
#ifdef __ANDROID__
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL);
#else
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
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

LiteRtStatus MakePluginLibGlobPattern(absl::string_view search_path,
                                      std::string& pattern) {
  LITERT_ENSURE(!search_path.ends_with("/"), kLiteRtStatusErrorInvalidArgument,
                "Search paths must not have trailing slash");

  // NOTE: Compiler plugin shared libraries also have "Plugin" somewhere after
  // the standard prefix.
  constexpr absl::string_view kGlobPluginLibTemplate = "%s/%s*Plugin*.so";
  pattern = absl::StrFormat(kGlobPluginLibTemplate, search_path,
                            kLiteRtSharedLibPrefix);
  return kLiteRtStatusOk;
}

LiteRtStatus FindLiteRtSharedLibs(absl::string_view search_path,
                                  std::vector<std::string>& results) {
#ifndef __ANDROID__
  std::string glob_pattern;
  LITERT_RETURN_STATUS_IF_NOT_OK(
      MakePluginLibGlobPattern(search_path, glob_pattern));

  glob_t glob_result = {};
  const int glob_status =
      glob(glob_pattern.c_str(), GLOB_ERR, nullptr, &glob_result);
  if (glob_status == GLOB_NOMATCH || glob_status == GLOB_ABORTED) {
    LITERT_LOG(LITERT_WARNING, "%s", "Didn't find any plugin libs to load\n");
    globfree(&glob_result);
    return kLiteRtStatusOk;
  } else if (glob_status != 0) {
    LITERT_LOG(LITERT_ERROR, "Glob failed with code: %d\n", glob_status);
    globfree(&glob_result);
    return kLiteRtStatusErrorNotFound;
  }

  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    results.emplace_back().assign(glob_result.gl_pathv[i]);
    LITERT_LOG(LITERT_INFO, "Glob matched: %s\n", results.back().c_str());
  }

  globfree(&glob_result);
  return kLiteRtStatusOk;
#endif
  // TODO: Glob is not supported on android.
  return kLiteRtStatusErrorUnsupported;
}

}  // namespace litert::internal
