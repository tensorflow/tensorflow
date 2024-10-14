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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_DYNAMIC_LOADING_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_DYNAMIC_LOADING_H_

#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>
#include <ostream>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"

namespace lrt {

constexpr absl::string_view kLrtSharedLibPrefix = "libLrt";

// Loads shared library at given path.
LrtStatus OpenLib(absl::string_view so_path, void** lib_handle);

// Closes reference to loaded shared library held by lib_handle.
LrtStatus CloseLib(void* lib_handle);

// Dumps loading details of given lib handle.
void DumpLibInfo(void* lib_handle, std::ostream& out = std::cerr);

// Resolves a named symbol from given lib handle of type Sym.
template <class Sym>
inline static LrtStatus ResolveLibSymbol(void* lib_handle,
                                         absl::string_view sym_name,
                                         Sym* sym_handle) {
  Sym ptr = (Sym)::dlsym(lib_handle, sym_name.data());
  if (ptr == nullptr) {
    LITE_RT_LOG(LRT_ERROR, "Faild to resolve symbol: %s, with err: %s\n",
                sym_name, ::dlerror());
    return kLrtStatusDynamicLoadErr;
  }
  *sym_handle = ptr;
  return kLrtStatusOk;
}

// All internal dynamically linked dependencies for lite_rt should be prefixed
// "libLrt". Find all lite_rt shared libraries in "search_path"
LrtStatus FindLrtSharedLibs(absl::string_view search_path,
                            std::vector<std::string>& results);

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_DYNAMIC_LOADING_H_
