/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_TESTS_TEST_MACROS_H_
#define XLA_TESTS_TEST_MACROS_H_

#define DISABLED_ON_TPU(X) X

// We need this macro instead of pasting directly to support nesting
// the DISABLED_ON_FOO macros, as in the definition of DISABLED_ON_CPU.
// Otherwise the pasting is applied before macro expansion completes.
#define XLA_TEST_PASTE(A, B) A##B

// We turn off clang-format so we can indent the macros for readability.
// clang-format off


#ifdef XLA_TEST_BACKEND_TPU
# undef DISABLED_ON_TPU
# define DISABLED_ON_TPU(X) XLA_TEST_PASTE(DISABLED_, X)
#endif  // XLA_TEST_BACKEND_TPU

// clang-format on

#endif  // XLA_TESTS_TEST_MACROS_H_
