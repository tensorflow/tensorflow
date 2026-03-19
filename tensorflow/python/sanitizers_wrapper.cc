/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/pybind11.h"  // from @pybind11

// Check if specific santizers are enabled.
PYBIND11_MODULE(_pywrap_sanitizers, m) {
  m.def("is_asan_enabled", []() -> bool {
#if defined(ADDRESS_SANITIZER)
    return true;
#else
    return false;
#endif
  });

  m.def("is_msan_enabled", []() -> bool {
#if defined(MEMORY_SANITIZER)
    return true;
#else
    return false;
#endif
  });

  m.def("is_tsan_enabled", []() -> bool {
#if defined(THREAD_SANITIZER)
    return true;
#else
    return false;
#endif
  });

  m.def("is_ubsan_enabled", []() -> bool {
#if defined(UNDEFINED_BEHAVIOR_SANITIZER)
    return true;
#else
    return false;
#endif
  });
}
