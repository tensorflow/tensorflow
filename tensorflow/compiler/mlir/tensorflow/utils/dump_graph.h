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

// Helper functions for dumping Graphs in IR form. This is the miror of
// core/util/dump_graph.h.
// TODO(jpienaar): This is currently in separate location to avoid cyclic
// dependencies given the build rule dump_graph is part of.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_GRAPH_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_GRAPH_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

struct MlirDumpConfig;

// Dumps 'graph_def' to a file, as textual IR. Returns the file name chosen.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is "-", then instead the
// GraphDef will be logged (using the LOG() macro).
//
// Automatically picks a file name. Prefixes 'config.name' with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if 'config.dirname' is empty, and
// suffixes 'config.name' with ".mlir" to form a filename. If a graph with the
// same name has already been dumped by this process then a sequence number will
// be appended.
//
// Note: This is for debugging use and is not optimized for performance.
string DumpTextualIRToFile(const MlirDumpConfig& config, const Graph& graph,
                           const FunctionLibraryDefinition* flib_def = nullptr);

// Config of the textual dump.
struct MlirDumpConfig {
  MlirDumpConfig& with_name(string name) {
    this->name = name;
    return *this;
  }

  MlirDumpConfig& with_dirname(string dirname) {
    this->dirname = dirname;
    return *this;
  }

  MlirDumpConfig& with_standard_pipeline(bool enabled = true) {
    this->run_standard_pipeline = enabled;
    return *this;
  }

  MlirDumpConfig& with_op_printing_flags(
      mlir::OpPrintingFlags op_printing_flags) {
    this->op_printing_flags = op_printing_flags;
    return *this;
  }

  // Name of the graph which is used as the prefix of the filename chosen to
  // dump the graph to.
  string name = "dump";

  // The filename will be prefixed by dirname, if set, else read from the
  // TF_DUMP_GRAPH_PREFIX environment variable.
  string dirname = "";

  // If run_standard_pipeline is true, then the standard TF pipeline will be run
  // on the converted module before printing. The standard pipeline does basic
  // cleanup to simplify
  // TODO(jpienaar): An alternative would be to be able to specify a pipeline
  // (potentially as a callback or a string to avoid build dependency) for
  // more flexibility.
  bool run_standard_pipeline = false;

  // Op printing flags.
  mlir::OpPrintingFlags op_printing_flags = llvm::None;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DUMP_GRAPH_H_
