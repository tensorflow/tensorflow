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

#include "tensorflow/compiler/xla/client/lib/comparators.h"

#include <limits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class ComparatorsTest : public ClientLibraryTestBase {
 public:
  ComparatorsTest() : builder_(TestName()) {}
  XlaBuilder* builder() { return &builder_; }

 private:
  XlaBuilder builder_;
};

template <
    PrimitiveType type,
    typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
void BuildComparatorAndComparisons(ComparatorsTest* test,
                                   bool compare_less_than,
                                   absl::InlinedVector<bool, 10>* expected) {
  auto compare = compare_less_than
                     ? CreateScalarLtComputation({type}, test->builder())
                     : CreateScalarGtComputation({type}, test->builder());

  auto negative_nan = ConstantR0<T>(
      test->builder(), -T(std::numeric_limits<float>::quiet_NaN()));
  auto positive_nan = ConstantR0<T>(test->builder(),
                                    T(std::numeric_limits<float>::quiet_NaN()));
  auto negative_zero = ConstantR0<T>(test->builder(), T(-0.));
  auto positive_zero = ConstantR0<T>(test->builder(), T(0.));
  auto negative_infinity = MinValue(test->builder(), type);
  auto positive_infinity = MaxValue(test->builder(), type);

  // List the values in the expected sorting order from smallest to largest.
  std::vector<XlaOp> all_constants{negative_nan,      negative_infinity,
                                   negative_zero,     positive_zero,
                                   positive_infinity, positive_nan};

  // Do pairwise comparisons.
  std::vector<XlaOp> all_comparisons;
  for (const XlaOp& lhs_constant : all_constants) {
    for (const XlaOp& rhs_constant : all_constants) {
      all_comparisons.push_back(Broadcast(
          Call(test->builder(), compare, {lhs_constant, rhs_constant}), {1}));
    }
  }

  // Concantenate the comparison results.
  ConcatInDim(test->builder(), all_comparisons, 0);

  // If we use less-than comparisons, we expect the comparison to result in true
  // if the lhs value to be compared appears earlier in 'all_constants' than the
  // rhs value. Likewise, if we use greater-than comparisons, we expect the
  // comparison to return true if the rhs value appears earlier in
  // 'all_constants' than the lhs value.
  expected->clear();
  for (int i = 0; i < all_constants.size(); ++i) {
    for (int j = 0; j < all_constants.size(); ++j) {
      expected->push_back(compare_less_than ? i < j : i > j);
    }
  }
}

XLA_TEST_F(ComparatorsTest, CompareLtBF16) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<BF16>(this, /*compare_less_than=*/true,
                                      &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtBF16) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<BF16>(this, /*compare_less_than=*/false,
                                      &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

// The interpreter doesn't support S16, and F16 results in comparisons with S16.
XLA_TEST_F(ComparatorsTest, DISABLED_ON_INTERPRETER(CompareLtF16)) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F16>(this, /*compare_less_than=*/true,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, DISABLED_ON_INTERPRETER(CompareGtF16)) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F16>(this, /*compare_less_than=*/false,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareLtF32) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F32>(this, /*compare_less_than=*/true,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtF32) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F32>(this, /*compare_less_than=*/false,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareLtF64) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F64>(this, /*compare_less_than=*/true,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtF64) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F64>(this, /*compare_less_than=*/false,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

}  // namespace
}  // namespace xla
