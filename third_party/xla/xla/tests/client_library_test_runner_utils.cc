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

#include "xla/tests/client_library_test_runner_utils.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "xla/array2d.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status.h"

namespace xla {

XlaComputation CreateScalarReluF32() {
  XlaBuilder builder("relu");
  Shape shape = ShapeUtil::MakeValidatedShape(F32, {}).value();
  XlaOp z_value = Parameter(&builder, 0, std::move(shape), "z_value");
  XlaOp zero = ConstantR0<float>(&builder, 0.0f);
  Max(std::move(z_value), std::move(zero));
  absl::StatusOr<XlaComputation> computation = builder.Build();
  TF_CHECK_OK(computation.status());
  return *std::move(computation);
}

XlaComputation CreateScalarMax(const PrimitiveType test_type) {
  XlaBuilder builder("max");
  Shape shape = ShapeUtil::MakeValidatedShape(test_type, {}).value();
  XlaOp x = Parameter(&builder, 0, shape, "x");
  XlaOp y = Parameter(&builder, 1, shape, "y");
  Max(std::move(x), std::move(y));
  absl::StatusOr<XlaComputation> computation = builder.Build();
  TF_CHECK_OK(computation.status());
  return *std::move(computation);
}

// Creates a (rows x cols) array filled in the following form:
//
//  [      0              1 ...                   cols-1]
//  [  1,000          1,001 ...          1000.0 + cols-1]
//  [    ...            ... ...                      ...]
//  [(rows-1)*1000.0    ... ... (rows-1)*1000.0 + cols-1]
//
// If provided, offset is added uniformly to every element (e.g. an offset of
// 64 would cause 0 in the above to be 64, 1 to be 65, 1000 to be 1064, etc.)
std::unique_ptr<Array2D<float>> CreatePatternedMatrix(const int rows,
                                                      const int cols,
                                                      float offset) {
  auto array = std::make_unique<Array2D<float>>(rows, cols);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f) + offset;
    }
  }
  return array;
}

std::unique_ptr<Array2D<float>> CreatePatternedMatrixWithZeroPadding(
    const int rows, const int cols, const int rows_padded,
    const int cols_padded) {
  CHECK_GE(rows_padded, rows);
  CHECK_GE(cols_padded, cols);
  auto array = std::make_unique<Array2D<float>>(rows_padded, cols_padded, 0.0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f);
    }
  }
  return array;
}
}  // namespace xla
