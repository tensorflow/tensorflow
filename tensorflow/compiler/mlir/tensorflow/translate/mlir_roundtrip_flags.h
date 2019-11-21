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

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

struct ArrayInfo {
  // The node type when the input node is imported. Typically needs to be
  // specified when passing arbitrary nodes (some node attributes are removed).
  DataType imported_dtype;

  // Node "shape" attribute value.
  TensorShapeProto shape;
};

struct GraphImportConfig {
  using InputArrays =
      llvm::MapVector<string, ArrayInfo, llvm::StringMap<unsigned>>;
  // Maps input node names to node data types and shapes.
  InputArrays inputs;
  // Output node names.
  absl::flat_hash_set<string> output_arrays;
  // nodes:index strings for the output as specified on the command line.
  std::vector<string> output_arrays_order;
  // setting prune_unused_nodes to true, would prune unreachable nodes if
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
};

struct GraphExportConfig {
  // Whether to export shape attribute for the NodeDefs in the GraphDef.
  bool export_shapes = true;
  // Whether to export library field in the GraphDef.
  bool export_library = true;
  // Whether to export debug original node name in the GraphDef.
  bool export_debug_info = true;
  // If true, the main graph will be treated as a function.
  bool graph_as_function = false;
};

// Parses the command line flag strings to the specification of nodes in
// the Graph.
Status ParseOutputArrayInfo(absl::string_view array_names,
                            absl::flat_hash_set<string>* array,
                            std::vector<string>* order);

Status ParseOutputArrayInfo(const std::vector<string>& output_names,
                            absl::flat_hash_set<string>* array,
                            std::vector<string>* order);

// Parses the command line flag strings to the specification of nodes in
// the Graph. `data_types` input string can be empty since the flag is optional.
Status ParseInputArrayInfo(absl::string_view array_names,
                           absl::string_view data_types,
                           absl::string_view shapes,
                           GraphImportConfig::InputArrays* inputs);

Status ParseInputArrayInfo(const std::vector<string>& node_names,
                           const std::vector<string>& node_dtypes,
                           const std::vector<std::vector<int>>& node_shapes,
                           GraphImportConfig::InputArrays* inputs);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_MLIR_ROUNDTRIP_FLAGS_H_
