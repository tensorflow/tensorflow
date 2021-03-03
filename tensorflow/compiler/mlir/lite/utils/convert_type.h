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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONVERT_TYPE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONVERT_TYPE_H_

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
class Builder;
}  // namespace mlir

namespace tflite {
// Convert the MLIR type to the corresponding TFLite tensor.
tflite::TensorType ConvertTypeToTensorType(mlir::Type type);

// Convert the scalar type of a TFlite tensor to the corresponding MLIR type.
mlir::Type ConvertElementType(tflite::TensorType type, mlir::Builder builder);

// Convert the scalar type of a TFLite tensor to the corresponding
// Tensorflow type
tensorflow::DataType TflTypeToTfType(tflite::TensorType type);

// Convert the Tensorflow scalar type to the corresponding TFLite type
xla::StatusOr<tflite::TensorType> TfTypeToTflType(tensorflow::DataType type);

// Returns element type from attribute Type 'type_attr'.
mlir::Type GetShapeStrippedType(mlir::TypeAttr type_attr);

// Returns true if 'val' is not from Quantize op or
// from Quantize Op with same quant type as 'qtype_attr'
bool NotFromQuantOpOrSameQuantType(mlir::Value val, mlir::TypeAttr qtype_attr);

}  // namespace tflite
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONVERT_TYPE_H_
