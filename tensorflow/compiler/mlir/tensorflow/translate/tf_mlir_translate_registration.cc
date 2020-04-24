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

// This file wraps TensorFlow Graph(Def) to MLIR module conversion into passes
// to satisfy the API of MLIR pass registration. In order to do this, the
// command-line option header is pulled in.

#include <memory>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
using stream_executor::port::Status;
using stream_executor::port::StatusOr;

namespace {
inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}
}  // namespace

static OwningModuleRef GraphdefToMlirTranslateFunction(llvm::StringRef input,
                                                       MLIRContext* context) {
  return tensorflow::GraphdefToMlirTranslateFunction(
      input, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, upgrade_legacy,
      enable_shape_inference, context);
}

static TranslateToMLIRRegistration GraphdefToMlirTranslate(
    "graphdef-to-mlir", GraphdefToMlirTranslateFunction);

static OwningModuleRef GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, MLIRContext* context) {
  return tensorflow::GraphdefToSplattedMlirTranslateFunction(
      input, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, upgrade_legacy,
      enable_shape_inference, context);
}

static TranslateToMLIRRegistration GraphdefToSplattedMlirTranslate(
    "graphdef-to-splatted-mlir", GraphdefToSplattedMlirTranslateFunction);

static LogicalResult MlirToGraphdefTranslateFunction(
    ModuleOp module, llvm::raw_ostream& output) {
  if (!module) return failure();

  // TODO(fengliuai): Add exporter flags.
  tensorflow::GraphExportConfig confs;
  StatusOr<std::unique_ptr<tensorflow::GraphDef>> graphdef_or(
      tensorflow::ConvertMlirToGraphdef(module, confs));
  if (!graphdef_or.status().ok()) {
    LOG(ERROR) << "Graph export failed: " << graphdef_or.status();
    return mlir::failure();
  }

  output << graphdef_or.ValueOrDie()->DebugString();
  return success();
}

static TranslateFromMLIRRegistration mlir_to_graphdef_translate(
    "mlir-to-graphdef", MlirToGraphdefTranslateFunction);

}  // namespace mlir
