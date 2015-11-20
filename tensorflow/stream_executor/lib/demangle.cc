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

#include "tensorflow/stream_executor/lib/demangle.h"

#if (__GNUC__ >= 4 || (__GNUC__ >= 3 && __GNUC_MINOR__ >= 4)) && \
    !defined(__mips__)
#  define HAS_CXA_DEMANGLE 1
#else
#  define HAS_CXA_DEMANGLE 0
#endif

#include <stdlib.h>
#if HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace perftools {
namespace gputools {
namespace port {

// The API reference of abi::__cxa_demangle() can be found in
// libstdc++'s manual.
// https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html
string Demangle(const char *mangled) {
  string demangled;
  int status = 0;
  char *result = NULL;
#if HAS_CXA_DEMANGLE
  result = abi::__cxa_demangle(mangled, NULL, NULL, &status);
#endif
  if (status == 0 && result != NULL) {  // Demangling succeeeded.
    demangled.append(result);
    free(result);
  }
  return demangled;
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools
