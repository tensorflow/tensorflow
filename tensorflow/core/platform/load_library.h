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

#ifndef TENSORFLOW_PLATFORM_LOAD_LIBRARY_H_
#define TENSORFLOW_PLATFORM_LOAD_LIBRARY_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace internal {

Status LoadLibrary(const char* library_filename, void** handle);
Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol);
// Return the filename of a dynamically linked library formatted according to
// platform naming conventions
string FormatLibraryFileName(const string& name, const string& version);

}  // namespace internal

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_LOAD_LIBRARY_H_
