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

#include <utility>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/tools/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace mlir {
using tsl::Status;
using tsl::StatusOr;


static OwningOpRef<mlir::ModuleOp> GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, MLIRContext* context) {
  tensorflow::GraphdefToMlirOptions options{
      debug_info_file,        xla_compile_device_type,
      prune_unused_nodes,     convert_legacy_fed_inputs,
      graph_as_function,      upgrade_legacy,
      enable_shape_inference, unconditionally_use_set_output_shapes};
  auto module_or = tensorflow::GraphdefToSplattedMlirTranslateFunction(
      input, input_arrays, input_dtypes, input_shapes, output_arrays,
      control_output_arrays, options, context);
  if (!module_or.status().ok()) return nullptr;
  return std::move(module_or).value();
}

static TranslateToMLIRRegistration GraphdefToSplattedMlirTranslate(
    "graphdef-to-splatted-mlir", "graphdef-to-splatted-mlir",
    GraphdefToSplattedMlirTranslateFunction);

}  // namespace mlir
