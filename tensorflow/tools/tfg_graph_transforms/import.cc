/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/tfg_graph_transforms/import.h"

#include <string>

#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/tfg_graph_transforms/utils.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

tensorflow::StatusOr<mlir::OwningModuleRef> ImportSavedModel(
    mlir::MLIRContext* context, const std::string& saved_model_file) {
  tensorflow::SavedModel saved_model;

  TF_RETURN_IF_ERROR(ReadSavedModelProto(saved_model_file, saved_model));
  if (saved_model.meta_graphs_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "Input saved model has no meta graphs");
  }

  const auto& graphdef = saved_model.meta_graphs(0).graph_def();
  tensorflow::GraphDebugInfo debug_info;
  return mlir::tfg::ImportGraphDefToMlir(context, debug_info, graphdef);
}

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir
