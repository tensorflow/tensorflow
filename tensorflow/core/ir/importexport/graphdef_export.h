/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {
namespace tfg {

// Get the name of a value as if it were an edge in a graph.
tensorflow::StatusOr<std::string> GetValueName(Value value,
                                               TFGraphDialect *dialect);

// Convert a TFG graph directly to GraphDef. Graph functions in the module are
// added to the GraphDef's function library.
tensorflow::Status ConvertToGraphDef(ModuleOp module,
                                     tensorflow::GraphDef *graph);

// Convert a single TFG op to NodeDef. This utliity function requires a callback
// `get_value_name` that returns the edge name of the given operand.
tensorflow::Status ConvertToNodeDef(
    Operation *op, tensorflow::NodeDef *node, TFGraphDialect *dialect,
    function_ref<tensorflow::StatusOr<std::string>(Value)> get_value_name);

// Convert a single TFG function to a FunctionDef and add it to the function
// library. If a function with the same name already exists, replace it.
tensorflow::Status ConvertToFunctionDef(
    GraphFuncOp func, tensorflow::FunctionLibraryDefinition &library);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_EXPORT_H_
