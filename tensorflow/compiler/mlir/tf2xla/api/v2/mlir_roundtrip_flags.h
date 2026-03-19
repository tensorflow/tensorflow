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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_MLIR_ROUNDTRIP_FLAGS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_MLIR_ROUNDTRIP_FLAGS_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

struct ArrayInfoBase {
  // The node type when the input node is imported. Typically needs to be
  // specified when passing arbitrary nodes (some node attributes are removed).
  DataType imported_dtype;

  // Node "shape" attribute value.
  TensorShapeProto shape;
};

struct ArrayInfo : public ArrayInfoBase {
  using SubTypeInfo = ArrayInfoBase;
  // DT_RESOURCE and DT_VARIANT have subtypes
  std::vector<SubTypeInfo> subtypes;
};

struct GraphImportConfig {
  // Returns string representation of config.
  std::string str() const;

  using InputArrays =
      llvm::MapVector<std::string, ArrayInfo, llvm::StringMap<unsigned>>;
  // The name assigned to the function which is the import result of the given
  // graph. If empty, a default one will be used.
  std::string graph_func_name;
  // Maps input node names to node data types and shapes.
  InputArrays inputs;
  // name:index strings for the data outputs.
  std::vector<std::string> outputs;
  // name strings for the control outputs.
  std::vector<std::string> control_outputs;
  // Setting prune_unused_nodes to true, would prune unreachable nodes if
  // output_arrays is specified.
  bool prune_unused_nodes = false;
  // If true, inputs of type LegacyFedInput are replaced with Placeholder ops.
  // LegacyFedInput ops have two outputs unlike Placeholder which has only one
  // output, so if both outputs of the LegacyFedInput ops are used then returns
  // an error.
  bool convert_legacy_fed_inputs = false;
  // If true, the main graph will be treated as a function.
  bool graph_as_function = false;
  // If true, upgrade legacy features of the graph (for instance, functionalize
  // control-flow).
  bool upgrade_legacy = false;
  // If true, functionalization is restricted to nodes that will be
  // XLA-compiled. This is only needed if
  // - `upgrade_legacy` is true
  // - upgrading legacy features of the graph (which includes functionalization)
  //   runs before compilation cluster extraction (as for MLIR-based TPU bridge)
  // - session runtime is used (session runtime has issues with function names
  //   rewritten by functionalization).
  // Otherwise, this parameter should be set to false.
  bool restrict_functionalization_to_compiled_nodes = false;
  // If true, enables shape inference on input.
  // TODO(jpienaar): This will be removed shortly.
  bool enable_shape_inference = true;
  // _output_shapes is an unregistered attribute which is used during
  // GraphConstructor::ConvertGraph to override shapes. It is unfortunately
  // not always set correctly (which is undesirable and should be addressed)
  // so make it opt-in to consider it unconditionally also when importing the
  // graph.
  bool unconditionally_use_set_output_shapes = false;
  // If set, use the value as the device type and mark the function graph for
  // XLA compilation.
  std::string xla_compile_device_type;
  // If true, enables moving ops to different devices or moving unsupported ops
  // out of a compilation cluster.
  bool enable_soft_placement = false;
  // If true, a function attribute, `tf._original_func_name`, will be set in
  // functions which contains the corresponding original TF function name.
  bool set_original_tf_func_name = false;

  // If true, all functions in the graph will be converted to MLIR regardless of
  // whether the functions are referenced by the nodes. This is needed if
  // aliases and saved model object graph function matching is needed.
  bool convert_all_functions_to_mlir = false;
};

struct GraphExportConfig {
  // Whether to export the entry function to function library instead of the
  // graph.
  bool export_entry_func_to_flib = false;
  // Whether to export functions using the name set in the attribute
  // `tf._original_func_name` if it exists.
  bool export_original_tf_func_name = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_MLIR_ROUNDTRIP_FLAGS_H_
