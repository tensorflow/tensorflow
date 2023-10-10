/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {
namespace tfg {
// Converts the TensorFlow DataType 'dtype' into an MLIR (scalar) type.
tensorflow::Status ConvertDataType(tensorflow::DataType dtype, Builder& builder,
                                   Type* type);

// Converts a scalar MLIR type to a TensorFlow Datatype.
tensorflow::Status ConvertScalarTypeToDataType(Type type,
                                               tensorflow::DataType* dtype);

// Converts an MLIR type to TensorFlow DataType. If 'type' is a scalar type, it
// is converted directly. If it is a shaped type, the element type is converted.
tensorflow::Status ConvertToDataType(Type type, tensorflow::DataType* dtype);

// Converts an TensorFlow shape to the one used in MLIR.
void ConvertToMlirShape(const tensorflow::TensorShape& input_shape,
                        SmallVectorImpl<int64_t>* shape);

// Converts an TensorFlow shape proto to the one used in MLIR.
tensorflow::Status ConvertToMlirShape(
    const tensorflow::TensorShapeProto& input_shape,
    SmallVectorImpl<int64_t>* shape);

// Given a tensor shape and dtype, get the corresponding MLIR tensor type.
tensorflow::StatusOr<Type> ConvertToMlirTensorType(
    const tensorflow::TensorShapeProto& shape, tensorflow::DataType dtype,
    Builder* builder);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_CONVERT_TYPES_H_
