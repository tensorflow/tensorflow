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

#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"

// These command-line options are following LLVM conventions because we also
// need to register the TF Graph(Def) to MLIR conversion with mlir-translate,
// which expects command-line options of such style.

using llvm::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_arrays(
    "tf-input-arrays", llvm::cl::desc("Input tensor names, separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> input_dtypes(
    "tf-input-data-types",
    llvm::cl::desc("(Optional) Input tensor data types, separated by ','. Use "
                   "'' if a single data type is skipped. The data type from "
                   "the import graph is used if it is skipped."),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> input_shapes(
    "tf-input-shapes",
    llvm::cl::desc(
        "Input tensor shapes. Shapes for different tensors are separated by "
        "':', and dimension sizes for the same tensor are separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> output_arrays(
    "tf-output-arrays", llvm::cl::desc("Output tensor names, separated by ','"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> inference_type(
    "tf-inference-type",
    llvm::cl::desc(
        "Sets the type of real-number arrays in the output file. Only allows "
        "float and quantized types"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> min_values(
    "tf-input-min-values",
    llvm::cl::desc(
        "Sets the lower bound of the input data. Separated by ','; Each entry "
        "in the list should match an entry in -tf-input-arrays. This is "
        "used when -tf-inference-type is a quantized type."),
    llvm::cl::Optional, llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> max_values(
    "tf-input-max-values",
    llvm::cl::desc(
        "Sets the upper bound of the input data. Separated by ','; Each entry "
        "in the list should match an entry in -tf-input-arrays. This is "
        "used when -tf-inference-type is a quantized type."),
    llvm::cl::Optional, llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> debug_info_file(
    "tf-debug-info",
    llvm::cl::desc("Path to the debug info file of the input graph def"),
    llvm::cl::init(""));

// TODO(b/134792656): If pruning is moved into TF dialect as a pass
// we should remove this.
// NOLINTNEXTLINE
opt<bool> prune_unused_nodes(
    "tf-prune-unused-nodes",
    llvm::cl::desc("Prune unused nodes in the input graphdef"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> convert_legacy_fed_inputs(
    "tf-convert-legacy-fed-inputs",
    llvm::cl::desc(
        "Eliminate LegacyFedInput nodes by replacing them with Placeholder"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> graph_as_function("tf-graph-as-function",
                            llvm::cl::desc("Treat main graph as a function"),
                            llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> upgrade_legacy("tf-upgrade-legacy",
                         llvm::cl::desc("Upgrade legacy TF graph behavior"),
                         llvm::cl::init(false));
