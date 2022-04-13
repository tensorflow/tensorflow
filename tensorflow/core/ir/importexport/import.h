/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_IMPORT_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_IMPORT_H_

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
namespace tfg {

// Converts a Graph and function libs to a MLIR module containing the graph and
// expressed in TFG dialect.
tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportGraphAndFunctionsToMlir(
    MLIRContext* context, const tensorflow::Graph& graph,
    const tensorflow::GraphDebugInfo& debug_info,
    const tensorflow::FunctionLibraryDefinition& flib_def);

// Converts a GraphDef to a MLIR module containing the graph and expressed in
// TFG dialect.
tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportGraphDefToMlir(
    MLIRContext* context, const tensorflow::GraphDebugInfo& debug_info,
    const tensorflow::GraphDef& graphdef);

// Converts a saved model to a MLIR module expressed in TFG dialect.
// Only the root graph and function library of the saved model gets imported
// into MLIR TFG dialect.
// TODO(b/218882780): Consider importing SignatureDefs from the SavedModel.
tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportSavedModelToMlir(
    mlir::MLIRContext* context, const tensorflow::GraphDebugInfo& debug_info,
    const tensorflow::SavedModel& saved_model);

// Converts an array of "handle_data" (a DType and a Shape) to an MLIR array
// attribute. Each entry will be itself an ArrayAttribute containing a TypeAttr
// and a ShapeAttr
tensorflow::StatusOr<ArrayAttr> ConvertHandleData(
    Builder builder,
    const tensorflow::protobuf::RepeatedPtrField<
        tensorflow::ResourceHandleProto_DtypeAndShape>& handle_data);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_IMPORT_H_
