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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_CL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_CL_H_

// This file contains command-line options aimed to provide the parameters
// required by the TensorFlow Graph(Def) to MLIR module conversion. It is only
// intended to be included by binaries.

#include <string>

#include "llvm/Support/CommandLine.h"

// Please see the implementation file for documentation of these options.

// Import options.
extern llvm::cl::opt<std::string> input_arrays;
extern llvm::cl::opt<std::string> input_dtypes;
extern llvm::cl::opt<std::string> input_shapes;
extern llvm::cl::opt<std::string> output_arrays;
extern llvm::cl::opt<std::string> control_output_arrays;
extern llvm::cl::opt<std::string> inference_type;
extern llvm::cl::opt<std::string> min_values;
extern llvm::cl::opt<std::string> max_values;
extern llvm::cl::opt<std::string> debug_info_file;
extern llvm::cl::opt<bool> prune_unused_nodes;
extern llvm::cl::opt<bool> convert_legacy_fed_inputs;
extern llvm::cl::opt<bool> graph_as_function;
extern llvm::cl::opt<bool> upgrade_legacy;
// TODO(jpienaar): Temporary flag, flip default and remove.
extern llvm::cl::opt<bool> enable_shape_inference;

// Export options.
extern llvm::cl::opt<bool> export_entry_func_to_flib;

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_CL_H_
