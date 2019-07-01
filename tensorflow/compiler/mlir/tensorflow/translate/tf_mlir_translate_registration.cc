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
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using stream_executor::port::Status;
using stream_executor::port::StatusOr;

namespace {
inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}
}  // namespace

static std::unique_ptr<mlir::Module> GraphdefToMlirTranslateFunction(
    llvm::StringRef input_filename, mlir::MLIRContext* context) {
  return tensorflow::GraphdefToMlirTranslateFunction(
      StringRefToView(input_filename), debug_info_file, input_arrays,
      input_dtypes, input_shapes, output_arrays, inference_type, min_values,
      max_values, prune_unused_nodes, context);
}

static mlir::TranslateToMLIRRegistration GraphdefToMlirTranslate(
    "graphdef-to-mlir", GraphdefToMlirTranslateFunction);

static std::unique_ptr<mlir::Module> GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input_filename, mlir::MLIRContext* context) {
  return tensorflow::GraphdefToSplattedMlirTranslateFunction(
      StringRefToView(input_filename), debug_info_file, input_arrays,
      input_dtypes, input_shapes, output_arrays, inference_type, min_values,
      max_values, prune_unused_nodes, context);
}

static mlir::TranslateToMLIRRegistration GraphdefToSplattedMlirTranslate(
    "graphdef-to-splatted-mlir", GraphdefToSplattedMlirTranslateFunction);

static bool MlirToGraphdefTranslateFunction(mlir::Module* module,
                                            llvm::StringRef output_filename) {
  if (!module) return true;

  std::error_code error;
  auto result = llvm::make_unique<llvm::ToolOutputFile>(output_filename, error,
                                                        llvm::sys::fs::F_None);
  if (error) {
    LOG(ERROR) << error.message();
    return true;
  }

  // TODO(fengliuai): Add exporter flags.
  tensorflow::ExporterConfigs confs;
  StatusOr<std::unique_ptr<tensorflow::GraphDef>> graphdef_or(
      tensorflow::ConvertMlirToGraphdef(*module, confs));
  if (!graphdef_or.status().ok()) {
    LOG(ERROR) << "Graph export failed: " << graphdef_or.status();
    return true;
  }

  result->os() << graphdef_or.ValueOrDie()->DebugString();
  result->keep();
  return false;
}

static mlir::TranslateFromMLIRRegistration MlirToGraphdefTranslate(
    "mlir-to-graphdef", MlirToGraphdefTranslateFunction);
