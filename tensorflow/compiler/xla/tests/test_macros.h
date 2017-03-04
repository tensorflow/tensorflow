/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Macros for use in enabling/disabling tests on particular
// platforms. Marking a gunit test as disabled still ensures that it
// compiles.
//
// Implementation note: the macros are structured as follows:
// * Define the disabled macro to just pass the test name through (which, in
//   effect, does not disable it at all)
// * If a XLA_TEST_BACKEND_$TARGET macro indicates we're compiling for
//   $TARGET platform, make the disabled macro truly disable the test; i.e. by
//   redefining the DISABLED_ON_$TARGET macro to prepend "DISABLED_" to the test
//   name.

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_TEST_MACROS_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_TEST_MACROS_H_

#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/test.h"

// Use this macro instead of directly using TEST_P for parameterized tests,
// otherwise DISABLED_ON_* macros nested in TEST_P will not get expanded since
// TEST_P stringifies its argument. That makes the test disabled for all targets
// when any one of the DISABLED_ON_* macro is used, and the test will just pass.
// TODO(b/29122096): Remove this once TEST_P fixes this problem.
#define XLA_TEST_P(test_case_name, test_name) TEST_P(test_case_name, test_name)

#define DISABLED_ON_CPU(X) X
#define DISABLED_ON_CPU_PARALLEL(X) X
#define DISABLED_ON_GPU(X) X

// We need this macro instead of pasting directly to support nesting
// the DISABLED_ON_FOO macros, as in the definition of DISABLED_ON_CPU.
// Otherwise the pasting is applied before macro expansion completes.
#define XLA_TEST_PASTE(A, B) A##B

// We turn off clang-format so we can indent the macros for readability.
// clang-format off

#ifdef XLA_TEST_BACKEND_CPU
# undef DISABLED_ON_CPU
# define DISABLED_ON_CPU(X) XLA_TEST_PASTE(DISABLED_, X)
#endif  // XLA_TEST_BACKEND_CPU

#ifdef XLA_TEST_BACKEND_CPU_PARALLEL
# undef DISABLED_ON_CPU
# define DISABLED_ON_CPU(X) XLA_TEST_PASTE(DISABLED_, X)
# undef DISABLED_ON_CPU_PARALLEL
# define DISABLED_ON_CPU_PARALLEL(X) XLA_TEST_PASTE(DISABLED_, X)
#endif  // XLA_TEST_BACKEND_CPU_PARALLEL

#ifdef XLA_TEST_BACKEND_GPU
# undef DISABLED_ON_GPU
# define DISABLED_ON_GPU(X) XLA_TEST_PASTE(DISABLED_, X)
#endif  // XLA_TEST_BACKEND_GPU

// clang-format on

#define XLA_TEST_F(test_fixture, test_name) TEST_F(test_fixture, test_name)

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_MACROS_H_
