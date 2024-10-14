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

#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"

#include <dlfcn.h>

#ifndef __ANDROID__
#include <glob.h>
#include <link.h>
#endif

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"

namespace lrt {

LrtStatus OpenLib(absl::string_view so_path, void** lib_handle) {
#ifdef __ANDROID__
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL);
#else
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
#endif

  if (res == nullptr) {
    LITE_RT_LOG(LRT_ERROR,
                "Failed to load .so at path: %s, with error:\n\t %s\n", so_path,
                ::dlerror());

    return kLrtStatusDynamicLoadErr;
  }
  *lib_handle = res;
  return kLrtStatusOk;
}

LrtStatus CloseLib(void* lib_handle) {
  if (0 != ::dlclose(lib_handle)) {
    LITE_RT_LOG(LRT_ERROR, "Failed to close .so with error: %s", ::dlerror());
    return kLrtStatusDynamicLoadErr;
  }
  return kLrtStatusOk;
}

void DumpLibInfo(void* lib_handle, std::ostream& out) {
#ifndef __ANDROID__
  out << "\n--- Lib Info ---\n";
  if (lib_handle == nullptr) {
    out << "Handle is nullptr\n";
    return;
  }

  Lmid_t dl_ns_idx;
  if (0 != ::dlinfo(lib_handle, RTLD_DI_LMID, &dl_ns_idx)) {
    return;
  }

  std::string dl_origin;
  dl_origin.resize(512);
  if (0 != ::dlinfo(lib_handle, RTLD_DI_ORIGIN, dl_origin.data())) {
    return;
  }

  link_map* lm;
  if (0 != ::dlinfo(lib_handle, RTLD_DI_LINKMAP, &lm)) {
    return;
  }

  out << "Lib Namespace: " << dl_ns_idx << "\n";
  out << "Lib Origin: " << dl_origin << "\n";

  out << "loaded objects:\n";

  auto* forward = lm->l_next;
  auto* backward = lm->l_prev;

  while (forward != nullptr) {
    out << "  " << forward->l_name << "\n";
    forward = forward->l_next;
  }

  out << "***" << lm->l_name << "\n";

  while (backward != nullptr) {
    out << "  " << backward->l_name << "\n";
    backward = backward->l_prev;
  }

  out << "\n";
#endif
}

LrtStatus MakePluginLibGlobPattern(absl::string_view search_path,
                                   std::string& pattern) {
  LRT_ENSURE(!search_path.ends_with("/"), kLrtStatusErrorInvalidArgument,
             "Search paths must not have trailing slash");

  // NOTE: Compiler plugin shared libraries also have "Plugin" appended
  // to the standard prefix.
  constexpr absl::string_view kGlobPluginLibTemplate = "%s/%sPlugin*.so";
  pattern =
      absl::StrFormat(kGlobPluginLibTemplate, search_path, kLrtSharedLibPrefix);
  return kLrtStatusOk;
}

LrtStatus FindLrtSharedLibs(absl::string_view search_path,
                            std::vector<std::string>& results) {
#ifndef __ANDROID__
  std::string glob_pattern;
  LRT_RETURN_STATUS_IF_NOT_OK(
      MakePluginLibGlobPattern(search_path, glob_pattern));

  glob_t glob_result = {};
  const int glob_status =
      glob(glob_pattern.c_str(), GLOB_ERR, nullptr, &glob_result);
  if (glob_status == GLOB_NOMATCH) {
    LITE_RT_LOG(LRT_WARNING, "%s", "Didn't find any plugin libs to load\n");
    globfree(&glob_result);
    return kLrtStatusOk;
  } else if (glob_status != 0) {
    LITE_RT_LOG(LRT_ERROR, "Glob failed with code: %d\n", glob_status);
    globfree(&glob_result);
    return kLrtStatusErrorNotFound;
  }

  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    results.emplace_back().assign(glob_result.gl_pathv[i]);
    LITE_RT_LOG(LRT_INFO, "Glob matched: %s\n", results.back().c_str());
  }

  globfree(&glob_result);
  return kLrtStatusOk;
#endif
  // TODO: Glob is not supported on android.
  return kLrtStatusErrorUnsupported;
}

}  // namespace lrt
