/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <dlfcn.h>

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace internal {

Status LoadLibrary(const char* library_filename, void** handle) {
#if !defined(__ANDROID__)
  // Check to see if the library has been already loaded by the process, if so
  // return an error status. Note: dlopen with RTLD_NOLOAD flag returns a
  // non-null pointer if the library has already been loaded, and null
  // otherwise.
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (*handle) {
    return errors::AlreadyExists(library_filename, " has already been loaded");
  }
#endif  // !defined(__ANDROID__)

  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    return errors::NotFound(dlerror());
  }
  return Status::OK();
}

Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol) {
  *symbol = dlsym(handle, symbol_name);
  if (!*symbol) {
    return errors::NotFound(dlerror());
  }
  return Status::OK();
}

}  // namespace internal

}  // namespace tensorflow
