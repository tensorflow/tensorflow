/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_SHAPE_AND_SIZE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_SHAPE_AND_SIZE_UTILS_H_

#include <cstdint>

#include "mlir/Support/LLVM.h"
#include "absl/status/statusor.h"

namespace mlir {
namespace TFL {

// Converts a TF size (64-bit) to TFLite (32-bit) and properly converts TF's
// value for dynamic size (`std::numeric_limits<int64_t>::min()`) to the
// TFLite-specific value.
int32_t ConvertToTfliteSize(int64_t size);

// Returns the quantization dimension after a reshape operation.
//
// TFL Reshape Op can fold multiple dimensions into one, or split one
// dimension into multiple dimensions. This function will return an error if the
// input quantization dimension is part of the folded/split dimensions.
absl::StatusOr<int32_t> GetQuantDimensionAfterReshape(
    mlir::ArrayRef<int64_t> input_shape, mlir::ArrayRef<int64_t> output_shape,
    int32_t input_quant_dim);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_SHAPE_AND_SIZE_UTILS_H_
