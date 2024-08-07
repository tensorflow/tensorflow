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
#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
namespace TF {

inline constexpr char kMLIRContextSingleThreadVar[] =
    "TF_USE_SINGLE_THREAD_MLIR_CONTEXT";

// Returns whether type can be further refined.
bool CanBeRefined(Type type);

// Returns a new arg type based on the shape and element type. If there are
// dynamic bounds attribute to the arg, update the bounds based on the shape
// as well.
Type GetNewArgType(Type old_arg_type, ArrayRef<int64_t> shape,
                   Type element_type, mlir::MLIRContext* context);

// Refines all the shapes in a module, skipping the inference for all ops
// whose type is in ops_to_skip.
// Returns a failure() on error, otherwise returns true to indicate that it
// reached convergence, false otherwise.
// If input shapes are provided, first refines the `main` function using
// InferShapeForFunction.
FailureOr<bool> InferModuleShape(ModuleOp module, int64_t max_iterations = 10,
                                 ArrayRef<TypeID> ops_to_skip = {},
                                 ArrayRef<ArrayRef<int64_t>> input_shapes = {});

// Given a tensorflow NodeShape string, returns a vector of argument shapes
// that can be used with InferShapeForFunction.
// TF NodeShape uses `,` to separate dimensions, and `:` to separate arguments.
// Ex: 1,2:3,4,5:6,? --> [[1, 2], [3, 4, 5], [6, ?]]
absl::StatusOr<SmallVector<SmallVector<int64_t>>> ParseArgumentShapes(
    absl::string_view input_shapes);

// Given a list of refined shapes matching the function arguments of func, runs
// shape inference over the function to propagate this updated information,
// skipping the inference for all ops whose type is in ops_to_skip.
// If arg_shapes are empty, then argument shapes will be left unchanged.
// Note: This affects the entire module, and changes are not just scoped to the
// function being inferred.
// Returns a failure() on error, otherwise returns true to indicate that it
// reached convergence, false otherwise.
FailureOr<bool> InferShapeForFunction(func::FuncOp func,
                                      ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                      int64_t graph_version,
                                      int64_t max_iterations = 10,
                                      ArrayRef<TypeID> ops_to_skip = {});

// Create a MLIRContext based on the threading setup in the env var.
std::unique_ptr<MLIRContext> MakeMLIRContextWithThreading();

}  // namespace TF

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SHAPE_INFERENCE_H_
