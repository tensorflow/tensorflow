/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_EXPORT_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace stablehlo::quantization {

// Suffix string for the module export step. Used for debugging.
constexpr absl::string_view kExportStepSuffix = "_export";

// Options when running passes for exporting an MLIR ModuleOp.
struct ExportOptions {
  // If set to `true`, it runs `DuplicateShapeDeterminingConstantsPass` before
  // lowering to tf_executor dialect.
  bool duplicate_shape_determining_constants = true;

  // If set to `true`, unfreezes constants into variables and saves them to a
  // checkpoint file. Setting this to `true` is an experimental feature that has
  // no stability guarantees.
  bool unfreeze_constants = false;

  // Path to the directory where checkpoint files are saved.
  std::string checkpoint_dir = "";

  // Name used to identify the ModuleOp this is exporting. Only used for
  // debugging and does not modify the behavior of the export.
  std::string debug_name = "stablehlo_quant";
};

// Factory function for `ExportedModel`.
[[nodiscard]] tensorflow::quantization::ExportedModel CreateExportedModel(
    tensorflow::GraphDef&& graph_def, absl::string_view init_node_name,
    absl::string_view checkpoint_dir,
    std::optional<tensorflow::SaverDef> saver_def,
    const absl::flat_hash_map<std::string, std::string>& function_aliases,
    const std::vector<tensorflow::AssetFileDef>& asset_file_defs);

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_EXPORT_H_
