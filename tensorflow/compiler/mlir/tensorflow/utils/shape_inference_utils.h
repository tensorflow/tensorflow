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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SHAPE_INFERENCE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SHAPE_INFERENCE_UTILS_H_

#include <cstdint>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/shape_inference.h"

namespace mlir {
namespace TF {

// Function that takes in a value and extracts a constant from it, if available.
// If the value cannot be resolved as a constant, a nullptr will be returned.
// Certain shape functions require constant values as arguments.
using OperandAsConstantFn = llvm::function_ref<Attribute(Value)>;

// Function that takes in an operation result and computes a shape (can be
// partial) value. Certain shape functions require shape values as arguments.
using OpResultAsShapeFn =
    llvm::function_ref<tensorflow::shape_inference::ShapeHandle(
        tensorflow::shape_inference::InferenceContext&, OpResult)>;

// Function that takes a result index and returns the element type. Element
// types are necessary for handle types (resource, variant).
using ResultElementTypeFn = llvm::function_ref<Type(int)>;

// Runs TensorFlow shape inference associated to the op type registered in the
// TensorFlow op registry based on the Graph version, operands, and attributes.
// Invoking this shape function will create conversions of parameters to the
// TensorFlow Graph equivalent data structures and back to MLIR equivalent data
// structures. This does not use a natively implemented shape inference in MLIR,
// and instead is temporary until shape functions are reimplemented/migrated to
// being in MLIR instead of the TensorFlow op registry.
LogicalResult InferReturnTypeComponentsForTFOp(
    Optional<Location> location, Operation* op, int64_t graph_version,
    OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes);

// Runs TensorFlow shape inference for an operation for a given Graph version.
// If an operation implements the `InferTypeOpInterface` or
// `InferShapedTypeOpInterface` interfaces, those are used instead but with
// derived attributes populated. Otherwise the above function is used but with
// default `operand_as_constant_fn` and `op_result_as_shape_fn` that only
// extracts a value if the operands are constant (no partial evaluation, and an
// empty `result_element_type_fn`. Element types with subtypes (DT_RESOURCE,
// DT_VARIANT) are not supported.
LogicalResult InferReturnTypeComponentsForTFOp(
    Optional<Location> location, Operation* op, int64_t graph_version,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SHAPE_INFERENCE_UTILS_H_
