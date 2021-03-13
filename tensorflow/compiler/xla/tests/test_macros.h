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

#define DISABLED_ON_CPU(X) X
#define DISABLED_ON_GPU(X) X
#define DISABLED_ON_GPU_ROCM(X) X
#define DISABLED_ON_INTERPRETER(X) X
#define DISABLED_ON_INTERPRETER_TSAN(X) X

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

#ifdef XLA_TEST_BACKEND_GPU
# undef DISABLED_ON_GPU
# define DISABLED_ON_GPU(X) XLA_TEST_PASTE(DISABLED_, X)

#if TENSORFLOW_USE_ROCM
# undef DISABLED_ON_GPU_ROCM
# define DISABLED_ON_GPU_ROCM(X) XLA_TEST_PASTE(DISABLED_, X)
#endif  // TENSORFLOW_USE_ROCM

#endif  // XLA_TEST_BACKEND_GPU

#ifdef XLA_TEST_BACKEND_INTERPRETER
# undef DISABLED_ON_INTERPRETER
# define DISABLED_ON_INTERPRETER(X) XLA_TEST_PASTE(DISABLED_, X)

#ifdef THREAD_SANITIZER
# undef DISABLED_ON_INTERPRETER_TSAN
# define DISABLED_ON_INTERPRETER_TSAN(X) XLA_TEST_PASTE(DISABLED_, X)
#endif  // THREAD_SANITIZER

#endif  // XLA_TEST_BACKEND_INTERPRETER

// clang-format on

namespace xla {

inline const char** DisabledManifestPath() {
  static const char* disabled_manifest_path = nullptr;
  return &disabled_manifest_path;
}

inline const char** TestPlatform() {
  static const char* test_platform = nullptr;
  return &test_platform;
}

}  // namespace xla

#define XLA_TEST_F(test_fixture, test_name) TEST_F(test_fixture, test_name)

#define XLA_TEST_P(test_case_name, test_name) TEST_P(test_case_name, test_name)

#define XLA_TYPED_TEST(CaseName, TestName) TYPED_TEST(CaseName, TestName)

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_MACROS_H_
