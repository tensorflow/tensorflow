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

#include "tensorflow/compiler/mlir/lite/utils/shape_and_size_utils.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

int32_t ConvertToTfliteSize(int64_t size) {
  return mlir::ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size);
}

absl::StatusOr<int32_t> GetQuantDimensionAfterReshape(
    const mlir::ArrayRef<int64_t> input_shape,
    const mlir::ArrayRef<int64_t> output_shape, int32_t input_quant_dim) {
  if (input_quant_dim >= input_shape.size()) {
    return absl::InvalidArgumentError(
        "Input quantization dimension is not found in the output shape.");
  }

  // We might be able to handle the folded case but not the split case because
  // MLIR Quant only supports single axis quantization. So, leaving both the
  // cases out-of-scope/unsupported for now.
  int32_t input_dim = -1, output_dim = -1;
  while (input_quant_dim > input_dim) {
    ++input_dim;
    ++output_dim;
    if (input_shape[input_dim] > output_shape[output_dim]) {
      // This means that the input dimension is being reduced/split. Multiply
      // the output dims until it matches the input dim.
      size_t split_output_shape_product = output_shape[output_dim];
      while (split_output_shape_product < input_shape[input_dim]) {
        split_output_shape_product *= output_shape[++output_dim];
      }
      // If the split output shape product doesn't match the input shape, we
      // return an error. This is a safety check.
      if (split_output_shape_product != input_shape[input_dim]) {
        return absl::InvalidArgumentError(
            "Input quantization dimension is not found in the output shape.");
      }
    } else if (input_shape[input_dim] < output_shape[output_dim]) {
      // This means that the input dimension is being folded. We need to
      // multiply the input dims until it matches the output dim.
      size_t folded_input_shape_product = input_shape[input_dim];
      while (folded_input_shape_product < output_shape[output_dim]) {
        folded_input_shape_product *= input_shape[++input_dim];
      }
      // If the folded input shape product doesn't match the output shape, we
      // return an error. This is a safety check.
      if (folded_input_shape_product != output_shape[output_dim]) {
        return absl::InvalidArgumentError(
            "Input quantization dimension is not found in the output shape.");
      }
    }
  }

  // Safety check to ensure that the input quantization dimension was not folded
  // or split. We should land exactly at input_quant_dim == input_dim, here.
  if (input_quant_dim < input_dim) {
    return absl::InvalidArgumentError(
        "Input quantization dimension is not found in the output shape.");
  }

  // If the input quantization dimension is the same as the input dimension,
  // but the input and output shapes are different, we return an error.
  if (input_quant_dim == input_dim &&
      input_shape[input_dim] != output_shape[output_dim]) {
    return absl::InvalidArgumentError(
        "Input quantization dimension is not found in the output shape.");
  }

  return output_dim;
}

}  // namespace TFL
}  // namespace mlir
