/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_COLLECTION_OPS_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_COLLECTION_OPS_UTIL_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace collection_ops_util {

// This file includes utilities for decomposing collection ops (stack, tensor
// list, tensor array) in TF. We represent such a data structure as a buffer of
// shape [max_element_count, element_shape].

// Creates an i32 scalar tf.Const.
Value CreateScalarConst(int value, OpBuilder builder, Location loc);

// Creates an i32 vector tf.Const.
Value GetR1Const(ArrayRef<int64_t> r1, OpBuilder builder, Location loc);

// Returns the type of the size tensor used to track a data structure's element
// count. It is a tensor<1xi32>, and we use R1 instead of a scalar because it is
// easier to concat it with other offsets.
TensorType GetSizeType(OpBuilder builder);

// Reshapes a scalar value to match the size type tensor<i32>.
Value ReshapeScalarToSizeType(OpBuilder builder, Value scalar, Location loc);

// Creates ops that represent the indices of the slice for an element in the
// buffer. Requires `index` to have tensor<1xi32> type.
Value GetIndicesForElement(Value index, Value buffer, OpBuilder builder,
                           Location loc);

// Creates ops that slice the element out of a buffer at the given index.
// Requires `index` to have tensor<1xi32> type.
Value GetElement(Value index, Value buffer, OpBuilder builder, Location loc);

// Creates ops that copy the buffer and update an element at the given index.
// Requires `index` to have tensor<1xi32> type.
Value SetElement(Value index, Value buffer, Value element, OpBuilder builder,
                 Location loc);

// Creates the buffer for the data structure with given element shape, type and
// maximum size.
LogicalResult CreateInitBufferValue(ArrayRef<int64_t> element_shape,
                                    Value max_size, Operation* op,
                                    Type element_dtype, OpBuilder builder,
                                    Value* buffer);
}  // namespace collection_ops_util
}  // namespace TF
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_COLLECTION_OPS_UTIL_H_
