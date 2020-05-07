/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/bundle_v2.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"

namespace tensorflow {

namespace {

Status ReadSavedModelProto(const string& export_dir,
                           SavedModel* saved_model_proto) {
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

Status ReadCheckpointObjectGraph(BundleReader* bundle_reader,
                                 TrackableObjectGraph* object_graph) {
  Tensor object_graph_tensor;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      bundle_reader->Lookup(kObjectGraphProtoKey, &object_graph_tensor),
      "SavedModel checkpoint does not contain object graph.");
  if (object_graph_tensor.dtype() != DT_STRING ||
      object_graph_tensor.dims() != 0 ||
      object_graph_tensor.NumElements() != 1) {
    return Status(
        error::Code::FAILED_PRECONDITION,
        "SavedModel checkpoint object graph was not the correct type.");
  }

  const tstring* object_graph_string = reinterpret_cast<const tstring*>(
      object_graph_tensor.tensor_data().data());
  if (!object_graph->ParseFromString(*object_graph_string)) {
    return Status(
        error::Code::FAILED_PRECONDITION,
        "SavedModel checkpoint object graph could not be deserialized.");
  }
  return Status::OK();
}

}  // namespace

Status SavedModelV2Bundle::Load(const std::string& export_dir,
                                SavedModelV2Bundle* const bundle) {
  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModelProto(export_dir, &saved_model_proto));

  // Load MetaGraphDef.
  // In version 2 SavedModels, there is only one MetaGraphDef.
  if (saved_model_proto.meta_graphs_size() != 1) {
    return Status(
        error::Code::INVALID_ARGUMENT,
        strings::StrCat(
            "SavedModelV2 should have exactly one MetaGraphDef but actually ",
            "contains ", saved_model_proto.meta_graphs_size()));
  }
  bundle->meta_graph_def_ =
      std::move(*saved_model_proto.mutable_meta_graphs(0));

  // Load GraphDebugInfo.
  TF_RETURN_IF_ERROR(
      ReadSavedModelDebugInfoIfPresent(export_dir, &bundle->debug_info_));

  // Load the variables checkpoint reader.
  const std::string variables_prefix = io::JoinPath(
      export_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename);
  bundle->variable_reader_.reset(
      new BundleReader(Env::Default(), variables_prefix));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      bundle->variable_reader_->status(),
      "Unable to load SavedModel variables checkpoint from ", variables_prefix);

  // Deserialize the object graph proto from the tensor bundle.
  TF_RETURN_IF_ERROR(ReadCheckpointObjectGraph(
      bundle->variable_reader_.get(), &bundle->trackable_object_graph_));
  return Status::OK();
}

Status SavedModelV2Bundle::VisitObjectsToRestore(
    RestoreObjectsCallback callback) {
  if (saved_object_graph().nodes_size() == 0 ||
      trackable_object_graph().nodes_size() == 0) {
    return Status::OK();
  }

  // Start from root nodes of both the SavedObjectGraph and TrackableObjectGraph
  // and descend to leaves. Note that the TrackableObjectGraph can have cycles
  // (as can the SavedObjectGraph).
  // This is detected and cycle edges are skipped.
  const SavedObject* root_saved_object = &saved_object_graph().nodes(0);
  const TrackableObjectGraph::TrackableObject* root_trackable_object =
      &trackable_object_graph().nodes(0);
  absl::flat_hash_set<int> trackable_node_ids;
  return RecurseObjectsToRestore(root_saved_object, 0, root_trackable_object,
                                 std::string(), &trackable_node_ids,
                                 std::move(callback));
}

Status SavedModelV2Bundle::RecurseObjectsToRestore(
    const SavedObject* saved_object, int saved_object_node_id,
    const TrackableObjectGraph::TrackableObject* trackable_object,
    std::string object_name, absl::flat_hash_set<int>* seen_trackable_node_ids,
    RestoreObjectsCallback callback) {
  // Callback if any attributes or slot variables.
  // Note that the root is always excluded from the search (it can never
  // be a restorable object). This matches some logic on the Python side.
  if (saved_object_node_id != 0 &&
      (trackable_object->attributes_size() > 0 ||
       trackable_object->slot_variables_size() > 0)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        callback(saved_object_node_id, *trackable_object), "Unable to restore ",
        object_name);
  }

  for (const auto& trackable_child_ref : trackable_object->children()) {
    const auto& local_name = trackable_child_ref.local_name();

    // Compute the full child name.
    std::string child_name;
    if (object_name.empty()) {
      child_name = local_name;
    } else {
      child_name = strings::StrCat(object_name, ".", local_name);
    }

    // Descend down the trackable graph.
    int trackable_child_node_id = trackable_child_ref.node_id();
    if (!seen_trackable_node_ids->insert(trackable_child_node_id).second) {
      // Cycle or duplicate detected - ignore this branch.
      continue;
    }
    if (trackable_child_node_id < 0 ||
        trackable_child_node_id >= trackable_object_graph().nodes_size()) {
      return Status(
          errors::Code::FAILED_PRECONDITION,
          strings::StrCat("Illegal trackable child node id for ", child_name));
    }
    const auto* trackable_child =
        &trackable_object_graph().nodes(trackable_child_node_id);

    // Descend down the saved object graph.
    int saved_child_node_id = -1;
    const SavedObject* saved_child = nullptr;
    for (const auto& saved_child_ref : saved_object->children()) {
      if (saved_child_ref.local_name() == local_name) {
        // Found.
        saved_child_node_id = saved_child_ref.node_id();
        if (saved_child_node_id >= 0 &&
            saved_child_node_id < saved_object_graph().nodes_size()) {
          saved_child = &saved_object_graph().nodes(saved_child_node_id);
        }
        break;
      }
    }

    if (!saved_child) {
      return Status(
          errors::Code::FAILED_PRECONDITION,
          strings::StrCat("Could not find saved object to restore for ",
                          child_name));
    }

    TF_RETURN_IF_ERROR(RecurseObjectsToRestore(
        saved_child, saved_child_node_id, trackable_child, child_name,
        seen_trackable_node_ids, callback));
  }
  return Status::OK();
}

}  // namespace tensorflow
