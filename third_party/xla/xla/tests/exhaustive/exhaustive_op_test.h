/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_H_
#define XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_H_

#include <cstddef>

#include "xla/tests/exhaustive/exhaustive_op_test_base.h"
#include "xla/tests/exhaustive/platform.h"

namespace xla {
namespace exhaustive_op_test {

// openXLA-specific ExhaustiveOpTestBase subclass.
//
// Holds utility functions related to determining the execution platform.
//
// Type Parameters:
// - T: The primitive type being tested.
// - N: The number of operands that the function being tested takes.
//
// Pure Virtual Functions:
// - GetInputSize
// - FillInput
template <PrimitiveType T, size_t N>
class ExhaustiveOpTest : public ExhaustiveOpTestBase<T, N> {
 public:
  using Traits = ExhaustiveOpTestBase<T, 1>::Traits;

  ExhaustiveOpTest() : platform_(*this->client_->platform()) {}

  bool RelaxedDenormalSigns() const override {
    return !platform_.IsNvidiaGpu();
  }

  const Platform& Platform() { return platform_; }

  // DEPRECATED: Only kept until exhaustive_unary_complex_test is merged into
  // exhaustive_unary_test. Use the new TestOp framework for
  // exhaustive_unary_test.
  bool IsGpu() const { return platform_.IsGpu(); }
  bool IsCpu() const { return platform_.IsCpu(); }

  static typename Traits::ErrorSpecGen GetDefaultSpecGenerator() {
    return exhaustive_op_test::GetDefaultSpecGenerator<T, N>();
  }

 protected:
  const class Platform platform_;
};

template <PrimitiveType T>
using ExhaustiveUnaryTest = ExhaustiveOpTest<T, 1>;

template <PrimitiveType T>
using ExhaustiveBinaryTest = ExhaustiveOpTest<T, 2>;

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_OP_TEST_H_
