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

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/meta/type_traits.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

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

// Get the size of the type in bits. The type can be ComplexType, FloatType,
// IntegerType, QuantizedType, or ShapeType of other supported types.
//
// Sub-byte types, e.g. qu4 and i2, are treated as a full i8.
int64_t GetSizeInBits(mlir::ShapedType shaped_type);
int64_t GetSizeInBits(mlir::Type type);
int64_t GetSizeInBits(mlir::quant::QuantizedType quant_type);

// Get the size of the type in bytes.
//
// Sub-byte element types, e.g. qu4 and i2, are treated as a full i8.
// e.g. GetSizeInBytes(tensor<4xi2>) == 4, instead of 1.
int64_t GetSizeInBytes(mlir::Type type);

// Performs an integer divide and checks that the remainder is zero.
// It supports int64 version as well.
template <typename Integer,
          typename = std::enable_if_t<std::is_same<int32_t, Integer>::value ||
                                      std::is_same<uint32_t, Integer>::value ||
                                      std::is_same<int64_t, Integer>::value ||
                                      std::is_same<uint64_t, Integer>::value>>
ABSL_ATTRIBUTE_ALWAYS_INLINE Integer ExactIntegerDivide(Integer numerator,
                                                        int64_t denominator) {
  const Integer ratio = numerator / denominator;
  assert((numerator % denominator) == 0);
  return ratio;
}

template <typename IntType,
          absl::enable_if_t<!std::is_unsigned<IntType>::value, int> = 0>
ABSL_ATTRIBUTE_ALWAYS_INLINE bool IsPowerOfTwo(IntType n) {
  static_assert(std::is_integral<IntType>::value, "");
  return n > 0 && (n & (n - 1)) == 0;
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONST_TENSOR_UTILS_H_
