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

#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Region.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir

namespace mlir {

namespace TF {

// Performs shape inference on the provided op and return true if the type of
// at least one result has been changed.
// A tf.Cast() is inserted for any uses that isn't in the TensorFlow dialect.
// `graph_version` indicates the current GraphDef compatibility versions
// (the versions field in graph.proto).
bool InferShapeForSingleOperation(Operation* op, Dialect* tf_dialect,
                                  int64_t graph_version);

// Infers shape on the provided region, including nested ones, iterate until fix
// point with a limit of max_iteration. Returns success if fix point is reached
// before max_iteration.
LogicalResult InferShapeUntilFixPoint(Region* region, int64_t graph_version,
                                      int64_t max_iteration = 10);

// Given a list of refined shapes matching the function arguments of func, runs
// shape inference over the function to propagate this updated information.
LogicalResult InferShapeForFunction(FuncOp func,
                                    ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                    int64_t graph_version);

// Refines the return type of the given function by folding tf.Cast that
// precedes the return instruction.
LogicalResult InferShapeForFunctionType(FuncOp func);

}  // namespace TF

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SHAPE_INFERENCE_H_
