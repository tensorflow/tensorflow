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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_TF_EXECUTOR_TO_GRAPH_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_TF_EXECUTOR_TO_GRAPH_H_

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

// Converts an MLIR module to TensorFlow graph and FunctionLibraryDefinition.
// The "main" function of the module is stored in the graph and the rest of
// functions are stored in the library. Control ret nodes are stored separately
// in `control_ret_nodes`.
absl::Status ConvertTfExecutorToGraph(
    mlir::ModuleOp module, const GraphExportConfig& configs,
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    absl::flat_hash_set<Node*>* control_ret_nodes);

// Converts an MLIR function and adds it to a FunctionLibraryDefinition.
absl::Status ConvertMlirFunctionToFunctionLibraryDef(
    mlir::func::FuncOp func, const GraphExportConfig& configs,
    FunctionDef* function_def);

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_TF_EXECUTOR_TO_GRAPH_H_
