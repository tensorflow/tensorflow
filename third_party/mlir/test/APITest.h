//===- Test.h - Simple macros for API unit tests ----------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
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
