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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DYNAMIC_LOADING_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DYNAMIC_LOADING_H_

#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"

namespace litert::internal {

constexpr absl::string_view kLiteRtSharedLibPrefix = "libLiteRt";

// Check for null and print the last dlerror.
inline void LogDlError() {
  char* err = ::dlerror();
  if (err == nullptr) {
    return;
  }
  LITERT_LOG(LITERT_WARNING, "::dlerror() : %s", err);
}

// Probes for a list of shared library at given paths and returns when the first
// one is found. Returns kLiteRtStatusErrorDynamicLoading if none of the shared
// libraries are found.
LiteRtStatus OpenLib(const std::vector<std::string>& so_paths,
                     void** lib_handle, bool log_failure = false);

// Loads shared library at given path. Logging can be disabled to probe for
// shared libraries.
LiteRtStatus OpenLib(absl::string_view so_path, void** lib_handle,
                     bool log_failure = true);

// Closes reference to loaded shared library held by lib_handle.
LiteRtStatus CloseLib(void* lib_handle);

// Resolves a named symbol from given lib handle of type Sym.
template <class Sym>
inline static LiteRtStatus ResolveLibSymbol(void* lib_handle,
                                            absl::string_view sym_name,
                                            Sym* sym_handle) {
  Sym ptr = (Sym)::dlsym(lib_handle, sym_name.data());
  if (ptr == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Faild to resolve symbol: %s\n", sym_name.data());
    LogDlError();
    return kLiteRtStatusErrorDynamicLoading;
  }
  *sym_handle = ptr;
  return kLiteRtStatusOk;
}

// Find all litert shared libraries in "search_path" and return
// kLiteRtStatusErrorInvalidArgument if the provided search_path doesn't
// exist. All internal dynamically linked dependencies for litert should be
// prefixed with "libLiteRtCompilerPlugin".
LiteRtStatus FindLiteRtCompilerPluginSharedLibs(
    absl::string_view search_path, std::vector<std::string>& results);

// Find all litert shared libraries in "search_path" and return
// kLiteRtStatusErrorInvalidArgument if the provided search_path doesn't
// exist. All internal dynamically linked dependencies for litert should be
// prefixed with "libLiteRtDispatch".
LiteRtStatus FindLiteRtDispatchSharedLibs(absl::string_view search_path,
                                          std::vector<std::string>& results);

// Find shared libraries for a given pattern in "search_path" and return
// kLiteRtStatusErrorInvalidArgument if the provided search_path doesn't
// exist.
LiteRtStatus FindLiteRtSharedLibsHelper(const std::string& search_path,
                                        const std::string& lib_pattern,
                                        bool full_match,
                                        std::vector<std::string>& results);

// Get details about the dynamic library including its .so dependencies.
void DLLInfo(void* lib_handle, std::ostream& out = std::cerr);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DYNAMIC_LOADING_H_
