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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/debugger.h"

#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/graph_def.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace stablehlo::quantization {
namespace {

using ::tensorflow::NodeDef;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::DebuggerOptions;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;

}  // namespace

void EnableDebugging(
    ExportedModel& exported_model, const DebuggerOptions& debugger_options,
    const PyFunctionLibrary& py_function_library,
    const absl::string_view src_saved_model_path,
    const std::unordered_set<std::string>& tags,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map) {
  // Enable `DumpTensor` nodes in `graph_def`. DumpTensor is disabled by
  // default to avoid logging data during calibration.
  MutateNodeDefs(*exported_model.mutable_graph_def(), [](NodeDef& node_def) {
    if (node_def.op() == "DumpTensor") {
      (*node_def.mutable_attr())["enabled"].set_b(true);
    }
  });

  if (debugger_options.debugger_type() ==
      DebuggerOptions::DEBUGGER_TYPE_WHOLE_MODEL) {
    // TODO: b/295139417 - Remove CustomAggregator op in unquantized dump model.
    // TODO: b/296916287 - Create a separate function for saving unquantized
    // dump model.
    py_function_library.SaveExportedModel(
        debugger_options.unquantized_dump_model_path(), exported_model,
        src_saved_model_path, tags, signature_def_map);

    // Update the `DumpTensor` ops' file name in `graph_def`.
    MutateNodeDefs(*exported_model.mutable_graph_def(), [](NodeDef& node_def) {
      if (node_def.op() == "DumpTensor") {
        (*node_def.mutable_attr())["file_name"].set_s(
            "quantized_tensor_data.pb");
      }
    });
  }
}

}  // namespace stablehlo::quantization
