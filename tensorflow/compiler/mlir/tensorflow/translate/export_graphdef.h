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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_EXPORT_GRAPHDEF_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_EXPORT_GRAPHDEF_H_

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
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
// Given an MLIR module, returns a GraphDef.
stream_executor::port::StatusOr<std::unique_ptr<GraphDef>>
ConvertMlirToGraphdef(mlir::ModuleOp module, const GraphExportConfig& configs);

// Converts an MLIR module to TensorFlow graph and FunctionLibraryDefinition.
// The "main" function of the module is stored in the graph and the rest of
// functions are stored in the library. Control ret nodes are stored separately
// in `control_ret_nodes`.
Status ConvertMlirToGraph(mlir::ModuleOp module,
                          const GraphExportConfig& configs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def,
                          absl::flat_hash_set<Node*>* control_ret_nodes);

// Converts an MLIR module to TensorFlow graph and FunctionLibraryDefinition.
// The "main" function of the module is stored in the graph and the rest of
// functions are stored in the library.
Status ConvertMlirToGraph(mlir::ModuleOp module,
                          const GraphExportConfig& configs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def);

// Converts an MLIR function and adds it to a FunctionLibraryDefinition.
Status ConvertMlirFunctionToFunctionLibraryDef(mlir::func::FuncOp func,
                                               const GraphExportConfig& configs,
                                               FunctionDef* function_def);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_EXPORT_GRAPHDEF_H_
