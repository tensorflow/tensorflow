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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_ROUNDTRIP_FLAGS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_ROUNDTRIP_FLAGS_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

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
  std::vector<string> outputs;
  // name strings for the control outputs.
  std::vector<string> control_outputs;
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
};

struct GraphExportConfig {
  // Whether to export shape attribute for the NodeDefs in the GraphDef.
  bool export_shapes = true;
  // Whether to export library field in the GraphDef.
  bool export_library = true;
  // Whether to export debug original node name in the GraphDef.
  bool export_debug_info = true;
  // Whether to export the entry function to function library instead of the
  // graph.
  bool export_entry_func_to_flib = false;
};

// Parses the command line flag strings to the specification of nodes in
// the Graph.
Status ParseOutputArrayInfo(absl::string_view array_names,
                            std::vector<string>* outputs);

Status ParseOutputArrayInfo(const std::vector<string>& output_names,
                            std::vector<string>* outputs);

// Parses the command line flag strings to the specification of nodes in
// the Graph. `data_types` input string can be empty since the flag is optional.
Status ParseInputArrayInfo(absl::string_view array_names,
                           absl::string_view data_types,
                           absl::string_view shapes,
                           GraphImportConfig::InputArrays* inputs);

Status ParseInputArrayInfo(
    const std::vector<string>& node_names,
    const std::vector<string>& node_dtypes,
    const std::vector<llvm::Optional<std::vector<int>>>& node_shapes,
    GraphImportConfig::InputArrays* inputs);

// Parses shapes from the given string into shapes_vector which is a structured
// format.
// NOTE: If shapes_str is empty, shapes_vector will also be empty.
Status ParseNodeShapes(
    absl::string_view shapes_str,
    std::vector<llvm::Optional<std::vector<int>>>& shapes_vector);

// Parses names from the given string into the names_vector.
// NOTE: If names_str is empty, names_vector will also be empty.
Status ParseNodeNames(absl::string_view names_str,
                      std::vector<std::string>& names_vector);

// Parses data types from the given string into the data_type_vector.
// NOTE: If data_types_str is empty, data_type_vector will also be empty.
Status ParseNodeDataTypes(absl::string_view data_types_str,
                          std::vector<std::string>& data_type_vector);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_ROUNDTRIP_FLAGS_H_
