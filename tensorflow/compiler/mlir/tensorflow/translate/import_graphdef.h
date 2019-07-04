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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_GRAPHDEF_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_GRAPHDEF_H_

#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Given a GraphDef, returns a MLIR module containing the graph in control-flow
// form.
stream_executor::port::StatusOr<mlir::OwningModuleRef> ConvertGraphdefToMlir(
    const GraphDef& graphdef, const GraphDebugInfo& debug_info,
    const NodeSpecs& specs, mlir::MLIRContext* context,
    bool add_default_attributes = true);

// Given a Graph, returns a MLIR module containing the graph in control-flow
// form.
stream_executor::port::StatusOr<mlir::OwningModuleRef> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const NodeSpecs& specs,
    mlir::MLIRContext* context);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_GRAPHDEF_H_
