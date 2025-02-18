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

// clang-format off
#ifndef __ANDROID__
#if __has_include(<link.h>)
#include <link.h>
#endif
#endif

#ifndef NDEBUG
#include <iostream>
#endif
// clang-format on

#include <filesystem>  // NOLINT
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"

namespace litert::internal {

LiteRtStatus OpenLib(const std::vector<std::string>& so_paths,
                     void** lib_handle, bool log_failure) {
  for (const auto& so_path : so_paths) {
    if (OpenLib(so_path, lib_handle, log_failure) == kLiteRtStatusOk) {
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorDynamicLoading;
}

LiteRtStatus OpenLib(absl::string_view so_path, void** lib_handle,
                     bool log_failure) {
  LITERT_LOG(LITERT_VERBOSE, "Loading shared library: %s", so_path.data());
#ifdef RTLD_DEEPBIND
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
#else
  void* res = ::dlopen(so_path.data(), RTLD_NOW | RTLD_LOCAL);
#endif

  if (res == nullptr) {
    if (log_failure) {
      LITERT_LOG(LITERT_WARNING, "Failed to load .so at path: %s",
                 so_path.data());
      LogDlError();
    }
    return kLiteRtStatusErrorDynamicLoading;
  }
  *lib_handle = res;
  LITERT_LOG(LITERT_INFO, "Successfully loaded shared library: %s",
             so_path.data());

#ifndef NDEBUG
  DLLInfo(*lib_handle, std::cerr);
#endif

  return kLiteRtStatusOk;
}

LiteRtStatus CloseLib(void* lib_handle) {
  if (0 != ::dlclose(lib_handle)) {
    LITERT_LOG(LITERT_ERROR, "Failed to close .so with error: %s", ::dlerror());
    return kLiteRtStatusErrorDynamicLoading;
  }
  return kLiteRtStatusOk;
}

static constexpr absl::string_view kSo = ".so";

LiteRtStatus FindLiteRtSharedLibsHelper(const std::string& search_path,
                                        const std::string& lib_pattern,
                                        bool full_match,
                                        std::vector<std::string>& results) {
  if (!Exists(search_path)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // TODO implement path glob in core/filesystem.h and remove filesystem
  // include from this file.
  for (const auto& entry : std::filesystem::directory_iterator(search_path)) {
    const auto& path = entry.path();
    if (entry.is_regular_file()) {
      if (full_match) {
        if (path.string().find(lib_pattern) != -1) {
          LITERT_LOG(LITERT_VERBOSE, "Found shared library: %s", path.c_str());
          results.push_back(path);
        }
      } else {
        const auto stem = path.stem().string();
        const auto ext = path.extension().string();
        if (stem.find(lib_pattern) == 0 && kSo == ext) {
          LITERT_LOG(LITERT_VERBOSE, "Found shared library: %s", path.c_str());
          results.push_back(path);
        }
      }
    } else if (entry.is_directory()) {
      FindLiteRtSharedLibsHelper(path, lib_pattern, full_match, results);
    }
  }

  return kLiteRtStatusOk;
}

static const char kCompilerPluginLibPatternFmt[] = "CompilerPlugin";

LiteRtStatus FindLiteRtCompilerPluginSharedLibs(
    absl::string_view search_path, std::vector<std::string>& results) {
  std::string root(search_path);
  const std::string lib_pattern =
      absl::StrCat(kLiteRtSharedLibPrefix, kCompilerPluginLibPatternFmt);
  return FindLiteRtSharedLibsHelper(root, lib_pattern, /*full_match=*/false,
                                    results);
}

static const char kDispatchLibPatternFmt[] = "Dispatch";

LiteRtStatus FindLiteRtDispatchSharedLibs(absl::string_view search_path,
                                          std::vector<std::string>& results) {
  std::string root(search_path.data());
  const std::string lib_pattern =
      absl::StrCat(kLiteRtSharedLibPrefix, kDispatchLibPatternFmt);
  return FindLiteRtSharedLibsHelper(root, lib_pattern, /*full_match=*/false,
                                    results);
}

void DLLInfo(void* lib_handle, std::ostream& out) {
  static constexpr absl::string_view kHeader = "/// DLL Info ///\n";
  static constexpr absl::string_view kFooter = "////////////////\n";

  out << absl::StreamFormat("%s", kHeader);
  if (lib_handle == nullptr) {
    out << "Handle is nullptr\n";
    out << absl::StreamFormat("%s", kFooter);
    return;
  }

#if !defined(__ANDROID__) && !defined(__APPLE__)
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

  out << "LIB NAMESPACE INDEX: " << dl_ns_idx << "\n";
  out << "LIB ORIGIN: " << dl_origin << "\n";

  out << "LINKED OBJECTS: \n";

  auto* forward = lm->l_next;
  auto* backward = lm->l_prev;

  while (forward != nullptr) {
    out << "   " << forward->l_name << "\n";
    forward = forward->l_next;
  }

  out << "***" << lm->l_name << "\n";

  while (backward != nullptr) {
    out << "   " << backward->l_name << "\n";
    backward = backward->l_prev;
  }

#else
  out << "Unsupported platform\n";
#endif
  out << absl::StreamFormat("%s", kFooter);
}

}  // namespace litert::internal
