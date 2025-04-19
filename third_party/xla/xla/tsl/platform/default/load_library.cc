/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/load_library.h"

#include <dlfcn.h>

#include <string>

#include "absl/status/status.h"

namespace tsl {

namespace internal {

absl::Status LoadDynamicLibrary(const char* library_filename, void** handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    // Note that in C++17 std::string_view(nullptr) gives segfault!
    const char* const error_msg = dlerror();
    return absl::NotFoundError(error_msg ? error_msg : "(null error message)");
  }
  return absl::OkStatus();
}

absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                  void** symbol) {
  // Check that the handle is not NULL to avoid dlsym's RTLD_DEFAULT behavior.
  if (!handle) {
    *symbol = nullptr;
  } else {
    *symbol = dlsym(handle, symbol_name);
  }
  if (!*symbol) {
    // Note that in C++17 absl::string_view(nullptr) gives segfault!
    const char* error_msg = dlerror();
    return absl::NotFoundError(error_msg ? error_msg : "(null error message)");
  }
  return absl::OkStatus();
}

std::string FormatLibraryFileName(const std::string& name,
                                  const std::string& version) {
  std::string filename;
#if defined(__APPLE__)
  if (version.size() == 0) {
    filename = "lib" + name + ".dylib";
  } else {
    filename = "lib" + name + "." + version + ".dylib";
  }
#else
  if (version.empty()) {
    filename = "lib" + name + ".so";
  } else {
    filename = "lib" + name + ".so" + "." + version;
  }
#endif
  return filename;
}

}  // namespace internal

}  // namespace tsl
