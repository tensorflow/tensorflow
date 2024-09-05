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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_MLIR_BRIDGE_PASS_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_MLIR_BRIDGE_PASS_UTIL_H_

#include <optional>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/framework/function.h"

namespace tensorflow {

// Checks if a graph or reachable functions in the library have any
// StatefulPartitionedOps with _XlaMustCompile=true. The function library will
// be skipped if nullptr is provided.
bool IsSupportedByNonReplicatedBridge(
    const Graph& graph, const FunctionLibraryDefinition* function_library);

// Checks if a graph or reachable functions in the library have any ops with
// _tpu_replicate or _xla_compile_device_type=TPU. The function library will be
// skipped if nullptr is provided.

bool IsSupportedByReplicatedBridge(
    const Graph& graph, const FunctionLibraryDefinition* function_library);

// Check if an MLIR module has any ops with _tpu_replicate or
// _xla_compile_device_type=TPU.
bool IsSupportedByReplicatedBridge(mlir::ModuleOp module);

// Check if an MLIR module contains TPUPartitionedCall op. If so, we define
// such graph as an inference graph. Otherwise, it is non inference graph.
bool HasTPUPartitionedCallOpInModule(mlir::ModuleOp module);

// Check if a graph contains TPUPartitionedCall op, including its reachable
// functions. The function library is used to store the functions that are
// defined in a TensorFlow program
bool IsInferenceGraph(const Graph& graph,
                      const FunctionLibraryDefinition* function_library);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_MLIR_BRIDGE_PASS_UTIL_H_
