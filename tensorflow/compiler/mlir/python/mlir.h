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

// Functions for getting information about kernels registered in the binary.
// Migrated from previous SWIG file (mlir.i) authored by aminim@.
#ifndef TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_H_
#define TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {

// Simple wrapper to support tf.mlir.experimental.convert_graph_def.
// Load a GraphDef (binary or textual proto format), convert to MLIR, and
// (optionally) optimize the module before returning it as a string.
// This is an early experimental API, ideally we should return a wrapper object
// around a Python binding to the MLIR module.
std::string ImportGraphDef(const std::string &proto,
                           const std::string &pass_pipeline,
                           bool show_debug_info, TF_Status *status);

// Simple wrapper to support tf.mlir.experimental.convert_function.
// Load FunctionDef (binary or textual proto format), convert to MLIR, and
// (optionally) optimize the module before returning it as a string.
// This is an early experimental API, ideally we should return a wrapper object
// around a Python binding to the MLIR module.
std::string ImportFunction(const std::string &functiondef_proto,
                           const std::string &pass_pipeline,
                           bool show_debug_info, TFE_Context *context,
                           TF_Status *status);

// This wrapper passes the graph_def taking names of input nodes, the shapes and
// types of its inputs and the output nodes as parameters to MLIR.
std::string ImportGraphDef(const std::string &proto,
                           const std::string &pass_pipeline,
                           bool show_debug_info, absl::string_view(input_names),
                           absl::string_view(input_data_types),
                           absl::string_view(input_data_shapes),
                           absl::string_view(output_names), TF_Status *status);

// Load a SavedModel and return a textual MLIR string corresponding to it.
//
// Args:
//   saved_model_path: File path from which to load the SavedModel.
//   exported_names_str: Comma-separated list of names to export.
//                       Empty means "export all".
//
// Returns:
//   A string of textual MLIR representing the raw imported SavedModel.
std::string ExperimentalConvertSavedModelToMlir(
    const std::string &saved_model_path, const std::string &exported_names_str,
    bool show_debug_info, TF_Status *status);

// Load a SavedModel V1 and return a textual MLIR string corresponding to it
// without any MLIR graph transformation.
//
// Args:
//   saved_model_path: File path from which to load the SavedModel.
//   tags: Tags to identify MetaGraphDef that need to be loaded.
//   upgrade_legacy: Boolean flag that indicates whether to upgrade legacy
//                   graphs
//
// Returns:
//   A string of textual MLIR representing the raw imported SavedModel.
std::string ExperimentalConvertSavedModelV1ToMlirLite(
    const std::string &saved_model_path, const std::string &exported_names_str,
    const std::string &tags, bool upgrade_legacy, bool show_debug_info,
    TF_Status *status);

// Load a SavedModel V1 and return a textual MLIR string corresponding to it.
//
// Args:
//   saved_model_path: File path from which to load the SavedModel.
//   tags: Tags to identify MetaGraphDef that need to be loaded.
//   lift_variables: Boolean flag that indicates whether to hoist variables
//                   after loading the SavedModel.
//
// Returns:
//   A string of textual MLIR representing the raw imported SavedModel.
std::string ExperimentalConvertSavedModelV1ToMlir(
    const std::string &saved_model_path, const std::string &exported_names_str,
    const std::string &tags, bool lift_variables,
    bool include_variables_in_initializers, bool upgrade_legacy,
    bool show_debug_info, TF_Status *status);

std::string ExperimentalRunPassPipeline(const std::string &mlir_txt,
                                        const std::string &pass_pipeline,
                                        bool show_debug_info,
                                        TF_Status *status);

// Writes the input textual MLIR as bytecode to output file.
void ExperimentalWriteBytecode(const std::string &filename,
                               const std::string &mlir_txt, TF_Status *status);

// Loads a TFLite flatbuffer, convert to TOSA for backend compilation and
// produce an MLIR bytecode file as output.
// TODO(jpienaar): Refactor this when we use more implicit module passing
// between calls to avoid serialization overhead.
void ExperimentalTFLiteToTosaBytecode(
    const std::string &flatbuffer_file, const std::string &tosa_bytecode_file,
    bool use_external_constant,
    const std::vector<std::string> &ordered_input_arrays,
    const std::vector<std::string> &ordered_output_arrays, TF_Status *status);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_H_
