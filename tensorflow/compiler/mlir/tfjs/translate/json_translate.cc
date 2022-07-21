/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfjs/translate/json_translate.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

using mlir::ModuleOp;
using mlir::TranslateFromMLIRRegistration;
using std::string;
using tensorflow::Status;

// Translates the given MLIR module in the TFJS dialect to TFJS JSON
// format. Returns false on success.
//
bool tfjs::MlirToJSONTranslateFunction(ModuleOp module,
                                       std::string* serialized_json) {
  string json_output;
  // Allow TF to treat TFJS ops as TF ops.
  if (!tensorflow::AddTensorFlowOpPrefix("tfjs.").ok()) {
    LOG(ERROR) << "Failed to add tfjs op prefix.";
    return false;
  }
  tensorflow::GraphExportConfig confs;
  confs.export_shapes = true;
  confs.export_library = true;
  tensorflow::FunctionLibraryDefinition flib_def(
      tensorflow::OpRegistry::Global(), tensorflow::FunctionDefLibrary());
  absl::flat_hash_set<tensorflow::Node*> control_ret_nodes;
  auto graph = std::make_unique<tensorflow::Graph>(flib_def);
  auto status = tensorflow::ConvertMlirToGraph(module, confs, &graph, &flib_def,
                                               &control_ret_nodes);
  if (!status.ok()) {
    LOG(ERROR) << "Graph export failed: " << status;
    return false;
  }
  auto graphdef = std::make_unique<tensorflow::GraphDef>();
  graph->ToGraphDef(graphdef.get());

  // Replace the _Arg nodes of the main function with Placeholder op.
  auto nodes = graphdef->mutable_node();
  for (const auto& node : llvm::enumerate(*nodes)) {
    if (node.value().op() == "_Arg") {
      nodes->Mutable(node.index())->set_op("Placeholder");
    }
  }

  tensorflow::protobuf::util::JsonPrintOptions json_options;
  json_options.add_whitespace = true;
  auto jsonStatus = tensorflow::protobuf::util::MessageToJsonString(
      *graphdef, &json_output, json_options);
  if (!jsonStatus.ok()) {
    LOG(ERROR) << "Proto2Json failed: " << status;
    return false;
  }
  *serialized_json = std::move(json_output);
  return true;
}

static mlir::LogicalResult MlirToJSONFileTranslateFunction(
    ModuleOp module, llvm::raw_ostream& output) {
  std::string serialized_json;
  if (!tfjs::MlirToJSONTranslateFunction(module, &serialized_json))
    return mlir::failure();

  output << serialized_json;
  return mlir::success();
}

static TranslateFromMLIRRegistration MLIRToJSONFileTranslate(
    "mlir-to-tfjs-json", MlirToJSONFileTranslateFunction);
