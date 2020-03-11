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

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/test.h"

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

// Reads a disabled manifest file to resolve whether test cases should be
// disabled on a particular platform. For a test that should be disabled,
// returns DISABLED_ prepended to its name; otherwise returns the test name
// unmodified.
std::string PrependDisabledIfIndicated(absl::string_view test_case_name,
                                       absl::string_view test_name);

}  // namespace xla

// This is the internal "gtest" class instantiation -- it is identical to the
// GTEST_TEST_ macro, except that we intercept the test name for potential
// modification by PrependDisabledIfIndicated. That file can use an arbitrary
// heuristic to decide whether the test case should be disabled, and we
// determine whether the test case should be disabled by resolving the (test
// case name, test name) in a manifest file.
#define XLA_GTEST_TEST_(test_case_name, test_name, parent_class)             \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                    \
      : public parent_class {                                                \
   public:                                                                   \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                   \
                                                                             \
   private:                                                                  \
    virtual void TestBody();                                                 \
    static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;    \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name,   \
                                                           test_name));      \
  };                                                                         \
                                                                             \
  ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name,          \
                                                    test_name)::test_info_ = \
      ::testing::RegisterTest(                                               \
          #test_case_name,                                                   \
          ::xla::PrependDisabledIfIndicated(#test_case_name, #test_name)     \
              .c_str(),                                                      \
          nullptr, nullptr, __FILE__, __LINE__, []() -> parent_class* {      \
            return new GTEST_TEST_CLASS_NAME_(test_case_name, test_name)();  \
          });                                                                \
  void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

// This is identical to the TEST_F macro from "gtest", but it potentially
// disables the test based on an external manifest file, DISABLED_MANIFEST.
//
// Per usual, you can see what tests are available via --gunit_list_tests and
// choose to run tests that have been disabled via the manifest via
// --gunit_also_run_disabled_tests.
#define XLA_TEST_F(test_fixture, test_name) \
  XLA_GTEST_TEST_(test_fixture, test_name, test_fixture)

// Likewise, this is identical to the TEST_P macro from "gtest", but
// potentially disables the test based on the DISABLED_MANIFEST file.
//
// We have to wrap this in an outer layer so that any DISABLED_ON_* macros will
// be properly expanded before the stringification occurs.
#define XLA_TEST_P_IMPL_(test_case_name, test_name)                            \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                      \
      : public test_case_name {                                                \
   public:                                                                     \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                     \
    virtual void TestBody();                                                   \
                                                                               \
   private:                                                                    \
    static int AddToRegistry() {                                               \
      ::testing::UnitTest::GetInstance()                                       \
          ->parameterized_test_registry()                                      \
          .GetTestCasePatternHolder<test_case_name>(                           \
              #test_case_name,                                                 \
              ::testing::internal::CodeLocation(__FILE__, __LINE__))           \
          ->AddTestPattern(                                                    \
              #test_case_name,                                                 \
              ::xla::PrependDisabledIfIndicated(#test_case_name, #test_name)   \
                  .c_str(),                                                    \
              new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_( \
                  test_case_name, test_name)>());                              \
      return 0;                                                                \
    }                                                                          \
    static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;               \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name,     \
                                                           test_name));        \
  };                                                                           \
  int GTEST_TEST_CLASS_NAME_(test_case_name,                                   \
                             test_name)::gtest_registering_dummy_ =            \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry();      \
  void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

#define XLA_TEST_P(test_case_name, test_name) \
  XLA_TEST_P_IMPL_(test_case_name, test_name)

// This is identical to the TEST_F macro from "gtest", but it potentially
// disables the test based on an external manifest file, DISABLED_MANIFEST.
#define XLA_TYPED_TEST(CaseName, TestName)                                     \
  template <typename gtest_TypeParam_>                                         \
  class GTEST_TEST_CLASS_NAME_(CaseName, TestName)                             \
      : public CaseName<gtest_TypeParam_> {                                    \
   private:                                                                    \
    typedef CaseName<gtest_TypeParam_> TestFixture;                            \
    typedef gtest_TypeParam_ TypeParam;                                        \
    virtual void TestBody();                                                   \
  };                                                                           \
  bool gtest_##CaseName##_##TestName##_registered_ GTEST_ATTRIBUTE_UNUSED_ =   \
      ::testing::internal::TypeParameterizedTest<                              \
          CaseName,                                                            \
          ::testing::internal::TemplateSel<GTEST_TEST_CLASS_NAME_(CaseName,    \
                                                                  TestName)>,  \
          GTEST_TYPE_PARAMS_(CaseName)>::                                      \
          Register(                                                            \
              "", ::testing::internal::CodeLocation(__FILE__, __LINE__),       \
              #CaseName,                                                       \
              ::xla::PrependDisabledIfIndicated(#CaseName, #TestName).c_str(), \
              0);                                                              \
  template <typename gtest_TypeParam_>                                         \
  void GTEST_TEST_CLASS_NAME_(CaseName,                                        \
                              TestName)<gtest_TypeParam_>::TestBody()

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_MACROS_H_
