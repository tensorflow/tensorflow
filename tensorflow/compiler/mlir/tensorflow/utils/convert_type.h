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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TYPE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TYPE_H_

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

using tsl::StatusOr;

// Converts the TensorFlow DataType 'dtype' into an MLIR (scalar) type.
absl::Status ConvertDataType(DataType dtype, mlir::Builder builder,
                             mlir::Type* type);

// Converts a scalar MLIR type to a TensorFlow Datatype.
absl::Status ConvertScalarTypeToDataType(mlir::Type type, DataType* dtype);

// Converts an MLIR type to TensorFlow DataType. If 'type' is a scalar type, it
// is converted directly. If it is a shaped type, the element type is converted.
absl::Status ConvertToDataType(mlir::Type type, DataType* dtype);

// Converts an TensorFlow shape to the one used in MLIR.
void ConvertToMlirShape(const TensorShape& input_shape,
                        llvm::SmallVectorImpl<int64_t>* shape);

// Converts an TensorFlow shape proto to the one used in MLIR.
absl::Status ConvertToMlirShape(const TensorShapeProto& input_shape,
                                llvm::SmallVectorImpl<int64_t>* shape);

// Given a tensor shape and dtype, get the corresponding MLIR tensor type.
absl::StatusOr<mlir::Type> ConvertToMlirTensorType(
    const TensorShapeProto& shape, DataType dtype, mlir::Builder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TYPE_H_
