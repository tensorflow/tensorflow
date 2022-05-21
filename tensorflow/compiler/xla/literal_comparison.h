/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Library for comparing literals without taking a dependency on testing
// libraries.

#ifndef TENSORFLOW_COMPILER_XLA_LITERAL_COMPARISON_H_
#define TENSORFLOW_COMPILER_XLA_LITERAL_COMPARISON_H_

#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace literal_comparison {

// Returns ok if the given shapes have the same rank, dimension sizes, and
// primitive types.
Status EqualShapes(const Shape& expected, const Shape& actual);

// Returns ok if the expected and actual literals are (bitwise) equal for all
// elements in the literal. Also, asserts that the rank, dimensions sizes, and
// primitive type are equal.
Status Equal(const LiteralSlice& expected, const LiteralSlice& actual);

// Structure that contains the distribution of absolute and relative errors,
// bucketized into five buckets: [0.0001, 0.001, 0.01, 0.1, 1].
// Useful to understand the distribution of errors and set the permissible
// error bounds in an ErrorSpec.
struct ErrorBuckets {
  explicit ErrorBuckets(const std::vector<int64_t>& absolute_error_buckets = {},
                        const std::vector<int64_t>& rel_error_buckets = {})
      : abs_error_buckets(absolute_error_buckets),
        rel_error_buckets(rel_error_buckets) {}

  const std::vector<int64_t> abs_error_buckets;
  const std::vector<int64_t> rel_error_buckets;
};

using MiscompareCallback = std::function<void(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const LiteralSlice& mismatches, const ShapeIndex& shape_index,
    const ErrorBuckets& error_buckets)>;

// Inspects whether the expected and actual literals are within the given error
// bound for all elements. Also, inspects whether the rank, dimensions sizes,
// and dimension bounds are equivalent.
//
// Tuples are matched recursively.
//
// When comparing tensors of non-floating-point type, this inspects for exact
// equality, ignoring the ErrorSpec.
//
// If the shape of the literals is neither a complex/floating-point tensor nor a
// tuple which contains a complex/floating-point tensor, Near() is equivalent to
// Equal(). We don't raise an error in this case, because we want to allow
// callers to call Near() even if they have no preconceptions about the shapes
// being compared.
//
// If detailed_message is true, then the error message in the assertion result
// will contain a more detailed breakdown of mismatches.  By default, we display
// a detailed message only for "large" inputs.
//
// If miscompare_callback is nullptr, Near will return an error on the first
// detected mismatch.
Status Near(const LiteralSlice& expected, const LiteralSlice& actual,
            const ErrorSpec& error, absl::optional<bool> detailed_message,
            const MiscompareCallback& miscompare_callback);

// Calling ToString on a literal with over 100 million elements takes around
// 3 minutes.  The utility of printing a literal with >1000 elements is
// questionable, especially when writing the Literal proto to disk is orders
// of magnitude faster.
std::string ToStringTruncated(const LiteralSlice& literal);

}  // namespace literal_comparison
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_COMPARISON_H_
