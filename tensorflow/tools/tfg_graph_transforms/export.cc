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

#include "tensorflow/tools/tfg_graph_transforms/export.h"

#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/tools/tfg_graph_transforms/utils.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

tensorflow::Status ExportTFGToSavedModel(mlir::ModuleOp module,
                                         const std::string& input_file,
                                         const std::string& output_file) {
  tensorflow::GraphDef new_graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tensorflow::ExportMlirToGraphdef(module, &new_graphdef),
      "while converting TFG to GraphDef");

  tensorflow::SavedModel saved_model_proto;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ReadSavedModelProto(input_file, saved_model_proto),
      "loading saved model");

  if (saved_model_proto.meta_graphs_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "Original saved model has no meta graphs");
  }

  // Overwrite the graph def with the new one.
  // SavedModelV2 always has a single MetaGraph.
  // Use the first meta graph in case of the SavedModelV1.
  tensorflow::MetaGraphDef meta_graph_def =
      std::move(saved_model_proto.meta_graphs(0));
  *(meta_graph_def.mutable_graph_def()) = std::move(new_graphdef);
  *(saved_model_proto.mutable_meta_graphs(0)) = std::move(meta_graph_def);

  VLOG(1) << "Serializing resulting SavedModel to " << output_file;
  auto output_dir = tensorflow::io::Dirname(output_file);

  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->RecursivelyCreateDir(
      {output_dir.data(), output_dir.length()}));
  if (IsTextProto(output_file)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tensorflow::WriteTextProto(tensorflow::Env::Default(), output_file,
                                   saved_model_proto),
        "Error while writing the resulting saved model proto");
  } else {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tensorflow::WriteBinaryProto(tensorflow::Env::Default(), output_file,
                                     saved_model_proto),
        "Error while writing the resulting saved model proto");
  }
  return tensorflow::Status::OK();
}

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir
