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
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace collection_ops_util {

// This file includes utilities for decomposing collection ops (stack, tensor
// list, tensor array) in TF. We represent such a data structure as a buffer of
// shape [max_element_count, element_shape].

// Creates an i32 scalar tf.Const.
Value CreateScalarConst(int32_t value, OpBuilder builder, Location loc);

// Creates an integer vector tf.Const.
Value GetR1Const(ArrayRef<int64_t> r1, OpBuilder builder, Location loc,
                 int bitwidth = 32);

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
Value GetElement(Value index, Value buffer, OpBuilder builder, Location loc,
                 bool keep_slice_shape = false);

// Creates ops that copy the buffer and update an element at the given index.
// Requires `index` to have tensor<1xi32> type.
Value SetElement(Value index, Value buffer, Value element, OpBuilder builder,
                 Location loc);

// Creates the buffer for the data structure with given element shape, type and
// maximum size.
LogicalResult CreateInitBufferValue(ArrayRef<int64_t> element_shape,
                                    int64_t max_size, Operation* op,
                                    Type element_dtype, OpBuilder builder,
                                    Value* buffer);

// Same as above, but uses a Value as max_size and check if it is a constant.
LogicalResult CreateInitBufferValue(ArrayRef<int64_t> element_shape,
                                    Value max_size, Operation* op,
                                    Type element_dtype, OpBuilder builder,
                                    Value* buffer);

// Tries to infer the element type with full shape based its write accesses.
// `infer_from_user` should check if the provided op is an accessing op that
// could be used to infer the type.
llvm::Optional<RankedTensorType> GetElementTypeFromAccess(
    Value collection, ModuleOp module,
    llvm::function_ref<llvm::Optional<Type>(Operation*)> infer_from_op);

// Creates a ReadVariableOp on a local variable.
Value ReadLocalVariable(Value local_var, OpBuilder builder, Location loc);

// Creates an AssignVariableOp on a local variable.
TF::AssignVariableOp WriteLocalVariable(Value local_var, Value value,
                                        OpBuilder builder, Location loc);

// Adds two values, or creates a logical-or if they are boolean type.
Value AccumulateBuffers(Value a, Value b, OpBuilder builder, Location loc);

// Gathers elements in buffer with the indices.
Value GatherElements(Value indices, Value buffer, OpBuilder builder,
                     Location loc);

// Scatters elements into buffer, where each scattered element is accumulated
// with the old value in buffer.
Value ScatterAccumulateElements(Value indices, Value updates, Value buffer,
                                OpBuilder builder, Location loc);

}  // namespace collection_ops_util
}  // namespace TF
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_COLLECTION_OPS_UTIL_H_
