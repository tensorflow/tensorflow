/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_GRAPH_TO_TF_EXECUTOR_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_GRAPH_TO_TF_EXECUTOR_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/graph_to_tf_executor_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

inline constexpr absl::string_view kImportModelDefaultGraphFuncName = "main";

// Given a Graph, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToTfExecutor(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context,
    std::unordered_map<std::string, std::string>* tf_name_to_mlir_name =
        nullptr,
    const ConfigProto& config_proto = {},
    tensorflow::TF2XLABridgeVersion bridge_version =
        tensorflow::TF2XLABridgeVersion::kNotBridgeUseCase);

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_GRAPH_TO_TF_EXECUTOR_H_
