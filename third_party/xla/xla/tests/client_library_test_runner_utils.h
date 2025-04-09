/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TESTS_CLIENT_LIBRARY_TEST_RUNNER_UTILS_H_
#define XLA_TESTS_CLIENT_LIBRARY_TEST_RUNNER_UTILS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/array2d.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/tests/test_utils.h"

namespace xla {
// Create scalar operations for use in reductions.
XlaComputation CreateScalarReluF32();
XlaComputation CreateScalarMax(PrimitiveType test_type);

// Special case convenience functions for creating filled arrays.

// Creates an array of pseudorandom values lying between the given minimum and
// maximum values.
template <typename NativeT>
std::vector<NativeT> CreatePseudorandomR1(const int width, NativeT min_value,
                                          NativeT max_value, uint32_t seed) {
  std::vector<NativeT> result(width);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int i = 0; i < width; ++i) {
    result[i] = generator.get();
  }
  return result;
}

template <typename NativeT>
std::unique_ptr<Array2D<NativeT>> CreatePseudorandomR2(const int rows,
                                                       const int cols,
                                                       const NativeT min_value,
                                                       const NativeT max_value,
                                                       const uint32_t seed) {
  auto result = std::make_unique<Array2D<NativeT>>(rows, cols);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      (*result)(y, x) = generator.get();
    }
  }
  return result;
}

std::unique_ptr<Array2D<float>> CreatePatternedMatrix(int rows, int cols,
                                                      float offset = 0.0f);

// Creates a (rows x cols) array as above, padded out to
// (rows_padded x cols_padded) with zeroes.  Requires rows_padded >= rows
// and cols_padded > cols.
std::unique_ptr<Array2D<float>> CreatePatternedMatrixWithZeroPadding(
    int rows, int cols, int rows_padded, int cols_padded);

}  // namespace xla

#endif  // XLA_TESTS_CLIENT_LIBRARY_TEST_RUNNER_UTILS_H_
