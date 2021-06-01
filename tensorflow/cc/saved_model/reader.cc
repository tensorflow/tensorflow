/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/reader.h"

#include <unordered_set>

#include "absl/memory/memory.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

namespace tensorflow {
namespace {

Status ReadSavedModel(const string& export_dir, SavedModel* saved_model_proto) {
  LOG(INFO) << "Reading SavedModel from: " << export_dir;

  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    return ReadBinaryProto(Env::Default(), saved_model_pb_path,
                           saved_model_proto);
  }
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    return ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                         saved_model_proto);
  }
  return Status(error::Code::NOT_FOUND,
                "Could not find SavedModel .pb or .pbtxt at supplied export "
                "directory path: " +
                    export_dir);
}

// Swap tensor_content field of Const Op Tensors in the named functions
static Status SwapTensorContent(MetaGraphDef* meta_graph_def) {
  GraphDef graph_def = *meta_graph_def->mutable_graph_def();
  for (auto& function : *meta_graph_def->mutable_graph_def()
                             ->mutable_library()
                             ->mutable_function()) {
    for (auto& node : (*function.mutable_node_def())) {
      if (node.op() != "Const") continue;
      auto node_iterator = node.mutable_attr()->find("value");
      if (node_iterator == node.mutable_attr()->end()) continue;
      AttrValue node_value = node_iterator->second;
      if (!node_value.has_tensor()) continue;

      auto tsize = node_value.mutable_tensor()->tensor_content().size();
      auto p_type = node_value.mutable_tensor()->dtype();
      // Swap only when there is something in tensor_content field
      if (tsize != 0 && DataTypeCanUseMemcpy(p_type)) {
        Tensor parsed(p_type);
        DCHECK(parsed.FromProto(*node_value.mutable_tensor()));
        TF_RETURN_IF_ERROR(ByteSwapTensor(&parsed));
        (*node.mutable_attr())["value"].mutable_tensor()->set_tensor_content(
            string(reinterpret_cast<const char*>(parsed.tensor_data().data()),
                   parsed.tensor_data().size()));
      }
    }
  }
  return Status::OK();
}

Status FindMetaGraphDef(const std::unordered_set<string>& tags,
                        SavedModel* saved_model_proto,
                        MetaGraphDef* meta_graph_def) {
  LOG(INFO) << "Reading meta graph with tags { " << absl::StrJoin(tags, " ")
            << " }";
  for (MetaGraphDef& graph_def : *saved_model_proto->mutable_meta_graphs()) {
    // Get tags from the graph_def.
    std::unordered_set<string> graph_tags;
    for (const string& tag : graph_def.meta_info_def().tags()) {
      graph_tags.insert(tag);
    }
    // Match with the set of tags provided.
    if (graph_tags == tags) {
      *meta_graph_def = std::move(graph_def);
      // Correct the endiness of Tensor content on big-endian system
      if (!port::kLittleEndian) {
        TF_RETURN_IF_ERROR(SwapTensorContent(meta_graph_def));
      }
      return Status::OK();
    }
  }
  return Status(
      error::Code::NOT_FOUND,
      strings::StrCat(
          "Could not find meta graph def matching supplied tags: { ",
          absl::StrJoin(tags, " "),
          " }. To inspect available tag-sets in the SavedModel, please "
          "use the SavedModel CLI: `saved_model_cli`"));
}

}  // namespace

Status ReadMetaGraphDefFromSavedModel(const string& export_dir,
                                      const std::unordered_set<string>& tags,
                                      MetaGraphDef* const meta_graph_def) {
  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModel(export_dir, &saved_model_proto));
  TF_RETURN_IF_ERROR(
      FindMetaGraphDef(tags, &saved_model_proto, meta_graph_def));
  return Status::OK();
}

Status ReadSavedModelDebugInfoIfPresent(
    const string& export_dir,
    std::unique_ptr<GraphDebugInfo>* debug_info_proto) {
  LOG(INFO) << "Reading SavedModel debug info (if present) from: "
            << export_dir;

  const string debug_info_pb_path =
      io::JoinPath(export_dir, "debug", "saved_model_debug_info.pb");
  if (Env::Default()->FileExists(debug_info_pb_path).ok()) {
    GraphDebugInfo debug_info;
    TF_RETURN_IF_ERROR(
        ReadBinaryProto(Env::Default(), debug_info_pb_path, &debug_info));
    *debug_info_proto =
        absl::make_unique<GraphDebugInfo>(std::move(debug_info));
  }
  return Status::OK();
}

}  // namespace tensorflow
