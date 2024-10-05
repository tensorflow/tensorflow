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

#include "absl/strings/string_view.h"

namespace lrt {

// Loads shared library at given path, returning handle on success
// and nullptr on failure.
void* OpenLib(absl::string_view so_path);

// Dumps loading details of given lib handle.
void DumpLibInfo(void* lib_handle);

// Resolves a named symbol from given lib handle of type Sym. Returns
// nullptr on failure.
template <class Sym>
inline static Sym ResolveLibSymbol(void* lib_handle,
                                   absl::string_view sym_name) {
  Sym ptr = (Sym)::dlsym(lib_handle, sym_name.data());
  if (ptr == nullptr) {
    std::cerr << "Failed to resolve symbol: " << sym_name << " with err "
              << ::dlerror() << "\n";
  }
  return ptr;
}

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_DYNAMIC_LOADING_H_
