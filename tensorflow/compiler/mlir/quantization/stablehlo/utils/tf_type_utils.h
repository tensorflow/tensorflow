/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_TF_TYPE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_TF_TYPE_UTILS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::quant::tensorflow {

// GetDenseAttrFromTensorProtoAttr returns DenseElementsAttr from tensor proto.
FailureOr<mlir::DenseElementsAttr> GetDenseAttrFromTensorProtoAttr(
    llvm::StringRef mangled_tensor_proto, TensorType result_tensor_type);

// Check if a type is TF qint type.
bool IsTFQintType(Type type);

// Convert qint type to the corresponding int type. Return original type if it
// is not qint type.
Type GetIntTypeFromTFQint(Type type);

// Check if an op is TF UniformQuantized op.
bool IsTFUniformQuantizedOp(Operation *op);

}  // namespace mlir::quant::tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_TF_TYPE_UTILS_H_
