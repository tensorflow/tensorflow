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

#include "tensorflow/core/ir/importexport/savedmodel_import.h"

#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir {
namespace tfg {

absl::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportSavedModelToMlir(
    mlir::MLIRContext *context, const tensorflow::GraphDebugInfo &debug_info,
    const tensorflow::SavedModel &saved_model) {
  if (saved_model.meta_graphs_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "Input saved model has no meta graphs");
  }

  if (saved_model.meta_graphs_size() > 1) {
    return tensorflow::errors::InvalidArgument(
        "Input saved model has more than one meta graph, currently not "
        "supported");
  }

  const auto &graphdef = saved_model.meta_graphs(0).graph_def();
  return ImportGraphDef(context, debug_info, graphdef);
}

}  // namespace tfg
}  // namespace mlir
