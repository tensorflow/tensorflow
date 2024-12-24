/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/savedmodel_export.h"

#include <utility>

#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir {
namespace tfg {

absl::Status ExportMlirToSavedModel(
    mlir::ModuleOp module, const tensorflow::SavedModel &original_saved_model,
    tensorflow::SavedModel *output_saved_model) {
  if (original_saved_model.meta_graphs_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "Original saved model has no meta graphs");
  }

  tensorflow::GraphDef new_graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ConvertToGraphDef(module, &new_graphdef),
                                  "while converting TFG to GraphDef");

  // Overwrite the graph def portion of the saved model with the new one.
  tensorflow::MetaGraphDef meta_graph_def = original_saved_model.meta_graphs(0);
  *(meta_graph_def.mutable_graph_def()) = std::move(new_graphdef);
  *output_saved_model = original_saved_model;
  *(output_saved_model->mutable_meta_graphs(0)) = std::move(meta_graph_def);

  return absl::OkStatus();
}

}  // namespace tfg
}  // namespace mlir
