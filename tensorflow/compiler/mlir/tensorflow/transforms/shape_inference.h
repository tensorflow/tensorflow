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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SHAPE_INFERENCE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SHAPE_INFERENCE_H_

#include <cstdint>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Returns whether type can be further refined.
bool CanBeRefined(Type type);

// Refines all the shapes in a module.
// Returns a failure() on error, otherwise returns true to indicate that it
// reached convergence, false otherwise.
FailureOr<bool> InferModuleShape(ModuleOp module, int64_t max_iterations = 10);

// Given a list of refined shapes matching the function arguments of func, runs
// shape inference over the function to propagate this updated information.
// If arg_shapes are empty, then argument shapes will be left unchanged.
// Note: This affects the entire module, and changes are not just scoped to the
// function being inferred.
// Returns a failure() on error, otherwise returns true to indicate that it
// reached convergence, false otherwise.
FailureOr<bool> InferShapeForFunction(FuncOp func,
                                      ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                      int64_t graph_version,
                                      int64_t max_iterations = 10);

}  // namespace TF

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SHAPE_INFERENCE_H_
