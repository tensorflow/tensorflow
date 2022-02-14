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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_H_

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
namespace tfg {
// Exports a GraphFunc operation as a new entry in the function library,
// overwriting any existing functions.
tensorflow::Status ExportFunction(mlir::tfg::GraphFuncOp func_op,
                                  tensorflow::FunctionLibraryDefinition &flib);
}  // namespace tfg
}  // namespace mlir
namespace tensorflow {

// Given an MLIR module, returns a `output_graph` GraphDef. The module must
// contain at most a single Graph operation and zero or more TFFunc operations.
Status ExportMlirToGraphdef(mlir::ModuleOp module, GraphDef *output_graph);

// Given an MLIR module, returns a `output_saved_model` SavedModel.
// The module must contain at most a single Graph operation and zero or more
// TFFunc operations. `original_saved_model` is used as only a GraphDef portion
// of a saved model represented in the MLIR module.
Status ExportMlirToSavedModel(mlir::ModuleOp module,
                              const SavedModel &original_saved_model,
                              SavedModel *output_saved_model);

// Callback type for `ConvertOperationToNode`.
using GetValueNameFn = llvm::function_ref<Status(
    mlir::Value /*operand*/, std::string & /*output_name*/)>;

// Converts an Operation to a NodeDef. The provided `get_value_name` callback
// computes the name to use in GraphDef for a given Value (either the result of
// an operation or a block operand if a function argument) and stores the result
// in the provided `output_name` string.
Status ConvertOperationToNode(mlir::Operation &op, NodeDef *node,
                              GetValueNameFn get_value_name);
Status ConvertOperationToNode(mlir::Operation &op, NodeDef *node);

// Convert the handle_data_arr to the `handle_data` field of the provided arg.
// Each entry of the array is itself an array with two entries: a Type and a
// ShapeAttr.
Status ConvertHandleData(mlir::ArrayAttr handle_data_arr, OpDef::ArgDef *arg);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_H_
