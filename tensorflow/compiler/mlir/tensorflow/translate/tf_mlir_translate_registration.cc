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
#include <utility>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace mlir {
using stream_executor::port::Status;
using stream_executor::port::StatusOr;

namespace {
inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}
}  // namespace

static OwningOpRef<mlir::ModuleOp> GraphdefToMlirTranslateFunction(
    llvm::StringRef input, MLIRContext* context) {
  auto module_or = tensorflow::GraphdefToMlirTranslateFunction(
      input, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes, context);
  if (!module_or.status().ok()) return nullptr;
  return std::move(module_or).value();
}

static TranslateToMLIRRegistration GraphdefToMlirTranslate(
    "graphdef-to-mlir", GraphdefToMlirTranslateFunction);

static OwningOpRef<mlir::ModuleOp> GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, MLIRContext* context) {
  auto module_or = tensorflow::GraphdefToSplattedMlirTranslateFunction(
      input, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, control_output_arrays, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes, context);
  if (!module_or.status().ok()) return nullptr;
  return std::move(module_or).value();
}

static TranslateToMLIRRegistration GraphdefToSplattedMlirTranslate(
    "graphdef-to-splatted-mlir", GraphdefToSplattedMlirTranslateFunction);

static LogicalResult MlirToGraphdefTranslateFunction(
    ModuleOp module, llvm::raw_ostream& output) {
  if (!module) return failure();

  // TODO(fengliuai): Add exporter flags.
  tensorflow::GraphExportConfig confs;
  confs.export_entry_func_to_flib = export_entry_func_to_flib;
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
    "mlir-to-graphdef", MlirToGraphdefTranslateFunction,
    [](DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

}  // namespace mlir
