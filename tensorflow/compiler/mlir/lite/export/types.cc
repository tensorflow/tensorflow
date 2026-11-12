/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/export/types.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace mlir::TFL::fb_export {
namespace {

absl::Status UnsupportedType(Type type) {
  std::string type_str;
  llvm::raw_string_ostream os(type_str);
  os << type;
  return absl::InvalidArgumentError(
      absl::StrCat("unsupported element type: ", type_str));
}

absl::StatusOr<tflite::TensorType> GetIntegerType(Type type, unsigned width,
                                                  bool is_unsigned) {
  switch (width) {
    case 1:
      return tflite::TensorType_BOOL;
    case 2:
      if (is_unsigned) return UnsupportedType(type);
      return tflite::TensorType_INT2;
    case 4:
      return is_unsigned ? tflite::TensorType_UINT4 : tflite::TensorType_INT4;
    case 8:
      return is_unsigned ? tflite::TensorType_UINT8 : tflite::TensorType_INT8;
    case 16:
      return is_unsigned ? tflite::TensorType_UINT16 : tflite::TensorType_INT16;
    case 32:
      return is_unsigned ? tflite::TensorType_UINT32 : tflite::TensorType_INT32;
    case 64:
      return is_unsigned ? tflite::TensorType_UINT64 : tflite::TensorType_INT64;
    default:
      return UnsupportedType(type);
  }
}

}  // namespace

absl::StatusOr<tflite::TensorType> GetTensorType(Type element_type) {
  if (element_type.isF32()) return tflite::TensorType_FLOAT32;
  if (element_type.isF16()) return tflite::TensorType_FLOAT16;
  if (element_type.isBF16()) return tflite::TensorType_BFLOAT16;
  if (element_type.isF64()) return tflite::TensorType_FLOAT64;
  if (isa<Float8E4M3FNType>(element_type)) {
    return tflite::TensorType_FLOAT8_E4M3FN;
  }
  if (isa<Float8E5M2Type>(element_type)) return tflite::TensorType_FLOAT8_E5M2;
  if (auto complex = dyn_cast<ComplexType>(element_type)) {
    if (complex.getElementType().isF32()) return tflite::TensorType_COMPLEX64;
    if (complex.getElementType().isF64()) return tflite::TensorType_COMPLEX128;
    return UnsupportedType(element_type);
  }
  if (auto integer = dyn_cast<IntegerType>(element_type)) {
    return GetIntegerType(element_type, integer.getWidth(),
                          integer.isUnsigned());
  }
  if (isa<quant::UniformQuantizedType, quant::UniformQuantizedPerAxisType>(
          element_type)) {
    auto quantized = cast<quant::QuantizedType>(element_type);
    return GetIntegerType(element_type, quantized.getStorageTypeIntegralWidth(),
                          !quantized.isSigned());
  }
  if (auto blockwise = dyn_cast<BlockwiseQuantType>(element_type)) {
    return GetTensorType(blockwise.getStorageType());
  }
  return UnsupportedType(element_type);
}

}  // namespace mlir::TFL::fb_export
