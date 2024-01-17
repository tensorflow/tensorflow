/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONST_TENSOR_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONST_TENSOR_UTILS_H_

#include <stdbool.h>

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace TFL {

bool IsQuantized(const tflite::TensorT& tensor);

absl::StatusOr<mlir::quant::QuantizedType> GetQuantizedType(
    const tflite::TensorT& tensor, mlir::Builder builder,
    bool is_constant = false, mlir::Type storage_type = {});

// Imports float tensor with calibration value into calibrated quantized type.
absl::StatusOr<mlir::quant::QuantizedType> GetCalibratedQuantizedType(
    const tflite::TensorT& tensor, mlir::Builder builder);

absl::StatusOr<mlir::TensorType> GetTensorType(const tflite::TensorT& tensor,
                                               mlir::Builder builder,
                                               bool is_constant = false,
                                               bool is_intermediate = false,
                                               bool get_storage = false);

// Gets a constant splat for the given value of type. Requires value to be of
// type static shaped RankedTensorType. `unique_index` is used to get the unique
// value for the attribute.
mlir::ElementsAttr GetSplat(mlir::RankedTensorType type, int unique_index,
                            mlir::Builder builder);

absl::StatusOr<mlir::ElementsAttr> ConvertIntBuffer(
    mlir::RankedTensorType shaped_type, const std::vector<uint8_t>& buffer,
    bool truncate = false);

absl::StatusOr<mlir::ElementsAttr> ConvertFloatBuffer(
    mlir::RankedTensorType shaped_type, const std::vector<uint8_t>& buffer);

tensorflow::TensorProto ConvertTfliteConstTensor(
    const tflite::TensorT& tensor, const std::vector<uint8_t>& buffer);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONST_TENSOR_UTILS_H_
