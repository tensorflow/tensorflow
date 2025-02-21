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

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_tensor.h"
// Placeholder for protosplitter merger include.

#define IS_OSS true

namespace tensorflow {

absl::StatusOr<MetaGraphDef*> FindMetaGraphDef(
    const std::unordered_set<string>& tags, SavedModel* saved_model_proto) {
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
      MetaGraphDef* meta_graph_def = &graph_def;
      // Correct the endiness of Tensor content on big-endian system
      if (!port::kLittleEndian) {
        TF_RETURN_IF_ERROR(ByteSwapTensorContentInMetaGraphDef(meta_graph_def));
      }
      return meta_graph_def;
    }
  }
  return Status(
      absl::StatusCode::kNotFound,
      strings::StrCat(
          "Could not find meta graph def matching supplied tags: { ",
          absl::StrJoin(tags, " "),
          " }. To inspect available tag-sets in the SavedModel, please "
          "use the SavedModel CLI: `saved_model_cli`"));
}

// Reads the SavedModel proto from saved_model.pb in `export_dir`.
// Returns a failure status when the SavedModel file does not exist.
Status ReadSavedModel(absl::string_view export_dir,
                      SavedModel* saved_model_proto) {
  LOG(INFO) << "Reading SavedModel from: " << export_dir;

  if (IS_OSS) {
    const std::string saved_model_pb_path =
        io::JoinPath(export_dir, kSavedModelFilenamePb);
    TF_ASSIGN_OR_RETURN(
        bool saved_model_pb_exists,
        internal::FileExists(Env::Default(), saved_model_pb_path));
    if (saved_model_pb_exists) {
      Status result = ReadBinaryProto(Env::Default(), saved_model_pb_path,
                                      saved_model_proto);
      if (result.ok()) {
        metrics::SavedModelReadCount(
            saved_model::GetWriteVersion(*saved_model_proto))
            .IncrementBy(1);
      }
      return result;
    }
  }

  const std::string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  auto saved_model_pbtxt_exists =
      internal::FileExists(Env::Default(), saved_model_pbtxt_path);
  if (saved_model_pbtxt_exists.value_or(false)) {
    absl::Status result = ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                        saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelReadCount(
          saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }

  if (!IS_OSS) {
    // Only use Merger outside of OSS.
    // Placeholder for protosplitter merger call.
  }

  return absl::Status(
      absl::StatusCode::kNotFound,
      strings::StrCat("Could not find SavedModel .pb or .pbtxt at supplied "
                      "export directory path: ",
                      export_dir,
                      ". Check that "
                      "the directory exists and that you have the right "
                      "permissions for accessing it."));
}

Status ReadMetaGraphDefFromSavedModel(absl::string_view export_dir,
                                      const std::unordered_set<string>& tags,
                                      MetaGraphDef* const meta_graph_def) {
  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModel(export_dir, &saved_model_proto));
  TF_ASSIGN_OR_RETURN(MetaGraphDef * m,
                      FindMetaGraphDef(tags, &saved_model_proto));
  *meta_graph_def = std::move(*m);
  return absl::OkStatus();
}

absl::Status ReadSavedModelDebugInfoIfPresent(
    absl::string_view export_dir,
    std::unique_ptr<GraphDebugInfo>* debug_info_proto) {
  LOG(INFO) << "Reading SavedModel debug info (if present) from: "
            << export_dir;

  const string debug_info_pb_path =
      io::JoinPath(export_dir, "debug", "saved_model_debug_info.pb");
  TF_ASSIGN_OR_RETURN(bool debug_info_pb_exists,
                      internal::FileExists(Env::Default(), debug_info_pb_path));
  if (debug_info_pb_exists) {
    GraphDebugInfo debug_info;
    TF_RETURN_IF_ERROR(
        ReadBinaryProto(Env::Default(), debug_info_pb_path, &debug_info));
    *debug_info_proto = std::make_unique<GraphDebugInfo>(std::move(debug_info));
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
