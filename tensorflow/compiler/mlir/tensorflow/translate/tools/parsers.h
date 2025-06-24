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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TOOLS_PARSERS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TOOLS_PARSERS_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Parses the command line flag strings to the specification of nodes in
// the Graph.
absl::Status ParseOutputArrayInfo(absl::string_view array_names,
                                  std::vector<string>* outputs);

absl::Status ParseOutputArrayInfo(const std::vector<string>& output_names,
                                  std::vector<string>* outputs);

// Parses the command line flag strings to the specification of nodes in
// the Graph. `data_types` input string can be empty since the flag is optional.
absl::Status ParseInputArrayInfo(absl::string_view array_names,
                                 absl::string_view data_types,
                                 absl::string_view shapes,
                                 GraphImportConfig::InputArrays* inputs);

absl::Status ParseInputArrayInfo(
    const std::vector<string>& node_names,
    const std::vector<string>& node_dtypes,
    const std::vector<std::optional<std::vector<int>>>& node_shapes,
    GraphImportConfig::InputArrays* inputs);

// Parses shapes from the given string into shapes_vector which is a structured
// format.
// NOTE: If shapes_str is empty, shapes_vector will also be empty.
absl::Status ParseNodeShapes(
    absl::string_view shapes_str,
    std::vector<std::optional<std::vector<int>>>& shapes_vector);

// Parses names from the given string into the names_vector.
// NOTE: If names_str is empty, names_vector will also be empty.
absl::Status ParseNodeNames(absl::string_view names_str,
                            std::vector<std::string>& names_vector);

// Parses data types from the given string into the data_type_vector.
// NOTE: If data_types_str is empty, data_type_vector will also be empty.
absl::Status ParseNodeDataTypes(absl::string_view data_types_str,
                                std::vector<std::string>& data_type_vector);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TOOLS_PARSERS_H_
