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

#include <string>
#include <utility>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/fingerprinting.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_tensor.h"

namespace tensorflow {
namespace {

using error::Code::NOT_FOUND;
using strings::StrCat;

// `tensorflow::SavedModelV2Bundle::Load` API label.
constexpr char kCCLoadBundleV2Label[] = "cc_load_bundle_v2";

Status ReadSavedModelProto(const string& export_dir,
                           SavedModel* saved_model_proto) {
  LOG(INFO) << "Reading SavedModel from: " << export_dir;

  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  Status found_pb = Env::Default()->FileExists(saved_model_pb_path);
  if (found_pb.ok()) {
    Status result =
        ReadBinaryProto(Env::Default(), saved_model_pb_path, saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelRead(saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }

  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  Status found_pbtxt = Env::Default()->FileExists(saved_model_pbtxt_path);
  if (found_pbtxt.ok()) {
    Status result = ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                  saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelRead(saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }

  Status err;
  if (found_pb.code() == found_pbtxt.code()) {
    err = Status(found_pb.code(), StrCat(found_pb.error_message(), "\n",
                                         found_pbtxt.error_message()));
  } else if (found_pb.code() == NOT_FOUND) {
    err = found_pbtxt;
  } else if (found_pbtxt.code() == NOT_FOUND) {
    err = found_pb;
  } else {
    // found_pb and found_pbtxt both errored, w/ different codes, neither being
    // NOT_FOUND.
    err = Status(
        error::Code::INTERNAL,
        StrCat("Different errors encountered while looking for saved_model.pb "
               "and saved_model.pbtxt in the export directory path \"",
               export_dir, "\": \n", found_pb.ToString(), "\n",
               found_pbtxt.ToString()));
  }

  return err;
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
  return OkStatus();
}

}  // namespace

Status SavedModelV2Bundle::Load(const std::string& export_dir,
                                SavedModelV2Bundle* const bundle) {
  metrics::SavedModelReadApi(kCCLoadBundleV2Label).IncrementBy(1);
  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModelProto(export_dir, &saved_model_proto));
  metrics::SavedModelReadPath().Set(export_dir);

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

  // Correct the endiness of Tensor content on big-endian system
  if (!port::kLittleEndian) {
    TF_RETURN_IF_ERROR(
        ByteSwapTensorContentInMetaGraphDef(&(bundle->meta_graph_def_)));
  }

  // Load GraphDebugInfo.
  TF_RETURN_IF_ERROR(
      ReadSavedModelDebugInfoIfPresent(export_dir, &bundle->debug_info_));

  const std::string variables_dir =
      io::JoinPath(export_dir, kSavedModelVariablesDirectory);
  if (!Env::Default()->FileExists(variables_dir).ok()) {
    LOG(INFO)
        << "No checkpoint found, assuming this is a program-only SavedModel";
  } else {
    // Load the variables checkpoint reader.
    const std::string variables_prefix =
        io::JoinPath(variables_dir, kSavedModelVariablesFilename);
    bundle->variable_reader_.reset(
        new BundleReader(Env::Default(), variables_prefix));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        bundle->variable_reader_->status(),
        "Unable to load SavedModel variables checkpoint from ",
        variables_prefix);

    // Deserialize the object graph proto from the tensor bundle.
    TF_RETURN_IF_ERROR(ReadCheckpointObjectGraph(
        bundle->variable_reader_.get(), &bundle->trackable_object_graph_));
  }
  // Read the fingerprint.
  auto fingerprint_proto =
      saved_model::fingerprinting::ReadSavedModelFingerprint(export_dir);
  if (fingerprint_proto.ok()) {
    // Set gauge cell with saved_model_checksum.
    metrics::SavedModelReadFingerprint().Set(
        std::to_string(fingerprint_proto->saved_model_checksum()));
  }
  return OkStatus();
}

Status SavedModelV2Bundle::VisitObjectsToRestore(
    RestoreObjectsCallback callback) {
  if (saved_object_graph().nodes_size() == 0 ||
      trackable_object_graph().nodes_size() == 0) {
    return OkStatus();
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
      return errors::FailedPrecondition(
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
  return OkStatus();
}

}  // namespace tensorflow
