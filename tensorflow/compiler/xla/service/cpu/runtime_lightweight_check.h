/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_LIGHTWEIGHT_CHECK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_LIGHTWEIGHT_CHECK_H_

#include <cstdlib>
#include <iostream>

// Aborts the program if the condition is false.
//
// This is like QCHECK, except it doesn't pull in the TF/XLA logging framework.
// This makes it suitable for use from within the XLA:CPU runtime files, which
// need to be lightweight.
#define XLA_LIGHTWEIGHT_CHECK(cond)                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::cerr << __FILE__ << ":" << __LINE__                              \
                << " Failed XLA_LIGHTWEIGHT_QCHECK " << #cond << std::endl; \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_LIGHTWEIGHT_CHECK_H_
