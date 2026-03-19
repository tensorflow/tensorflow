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

#include <optional>

#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/utils/shape_inference_utils.h"

namespace mlir {

class Operation;

namespace TF {

// Runs TensorFlow shape inference associated to the op type registered in the
// TensorFlow op registry based on the Graph version, operands, and attributes.
// Invoking this shape function will create conversions of parameters to the
// TensorFlow Graph equivalent data structures and back to MLIR equivalent data
// structures. This does not use a natively implemented shape inference in MLIR,
// and instead is temporary until shape functions are reimplemented/migrated to
// being in MLIR instead of the TensorFlow op registry.
LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, int64_t graph_version,
    tfg::OperandAsConstantFn operand_as_constant_fn,
    tfg::OpResultAsShapeFn op_result_as_shape_fn,
    tfg::ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SHAPE_INFERENCE_UTILS_H_
