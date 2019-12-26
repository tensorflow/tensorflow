//===- Test.h - Simple macros for API unit tests ----------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file define simple macros for declaring test functions and running them.
// The actual checking must be performed on the outputs with FileCheck.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_TEST_H_
#define MLIR_TEST_TEST_H_

#include <functional>
#include <vector>

namespace test_detail {
// Returns a mutable list of known test functions.  Used internally by test
// macros to add and run tests.  This function is static to ensure it creates a
// new list in each test file.
static std::vector<std::function<void()>> &tests() {
  static std::vector<std::function<void()>> list;
  return list;
}

// Test registration class.  Used internally by test macros to register tests
// during static allocation.
struct TestRegistration {
  explicit TestRegistration(std::function<void()> func) {
    test_detail::tests().push_back(func);
  }
};
} // end namespace test_detail

/// Declares a test function with the given name and adds it to the list of
/// known tests.  The body of the function must follow immediately.  Example:
///
/// TEST_FUNC(mytest) {
///   // CHECK: expected-output-here
///   emitSomethingToStdOut();
/// }
///
#define TEST_FUNC(name)                                                        \
  void name();                                                                 \
  static test_detail::TestRegistration name##Registration(name);               \
  void name()

/// Runs all registered tests.  Example:
///
/// int main() {
///   RUN_TESTS();
///   return 0;
/// }
#define RUN_TESTS                                                              \
  []() {                                                                       \
    for (auto f : test_detail::tests())                                        \
      f();                                                                     \
  }

#endif // MLIR_TEST_TEST_H_
