/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_MLIR_VALUE_UTILS_H_
#define TENSORFLOW_DTENSOR_MLIR_VALUE_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {

int ValueRank(mlir::Value operand_value);

// Creates a effective scalar type as rank 1 with a single element.
mlir::RankedTensorType EffectivelyScalarR1Type(mlir::Type element_type);

// Reshapes a value of size type tensor<i32> to scalar.
mlir::Value ReshapeSizeTypeToScalar(mlir::OpBuilder builder, mlir::Location loc,
                                    mlir::Value tensor);

// Retuns a int64 array representing the TensorFlow shape given the MLIR type.
// If the type is a resource, returns the underlying shape of the resource
// instead. Returns an error if the type is not a RankedTensorType.
StatusOr<llvm::SmallVector<int64_t>> GetTFShapeFromType(mlir::Type type);

// Return a 1-D int32 constant array with the given values.
mlir::Value IntConst(mlir::OpBuilder& builder, mlir::Location loc,
                     llvm::ArrayRef<int32> values);
// Return a 1-D int64 constant array with the given values.
mlir::Value Int64Const(mlir::OpBuilder& builder, mlir::Location loc,
                       llvm::ArrayRef<int64_t> values);
// Return a 1-D float32 constant array with the given values.
mlir::Value FloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                       llvm::ArrayRef<float> values);
// Returns a 1-D tf.string constant array with given values.
mlir::Value StringConst(mlir::OpBuilder& builder, mlir::Location loc,
                        llvm::ArrayRef<llvm::StringRef> values);
// Returns a tf.string scalar constant with given value.
mlir::Value StringScalarConst(mlir::OpBuilder& builder, mlir::Location loc,
                              llvm::StringRef value);
StatusOr<int64_t> ExtractConstIntFromValue(mlir::Value value);
Status ExtractConstVectorFromValue(mlir::Value value,
                                   llvm::SmallVector<int64_t, 4>* out_vector);

// Returns a int64 scalar constant with `value`.
mlir::Value CreateIntScalarConst(const int64_t value, mlir::OpBuilder builder,
                                 mlir::Location loc, bool use_int64 = true);

// Returns a scalar constant with 'value' of 'type'.
StatusOr<mlir::Value> CreateZeroScalarConst(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            mlir::Type type);

// Selects a scalar tensor value from a 1D array in specified index.
StatusOr<mlir::Value> SelectScalarValueFromArray(mlir::OpBuilder& builder,
                                                 int index,
                                                 mlir::Location location,
                                                 mlir::Value array);

// Returns the type that value holds. If value holds a Type that has a subtype,
// then it returns the subtype.
mlir::Type GetSubtypeOrSelf(mlir::Value value);

}  // namespace dtensor
}  // namespace tensorflow
#endif  // TENSORFLOW_DTENSOR_MLIR_VALUE_UTILS_H_
