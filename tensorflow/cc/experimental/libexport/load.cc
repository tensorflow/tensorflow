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
#include "tensorflow/cc/experimental/libexport/load.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace libexport {

using protobuf::RepeatedPtrField;

absl::StatusOr<TFPackage> TFPackage::Load(const std::string& path) {
  // Load the proto
  TFPackage tf_package;
  const string saved_model_pb_path = io::JoinPath(path, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(path, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), saved_model_pb_path,
                                       &tf_package.saved_model_proto_));
  } else if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                     &tf_package.saved_model_proto_));
  } else {
    return absl::Status(
        absl::StatusCode::kNotFound,
        "Could not find SavedModel .pb or .pbtxt at supplied export "
        "directory path: " +
            path);
  }

  // Load the trackable object graph for restoring checkpoint values
  const std::string variables_dir =
      tensorflow::io::JoinPath(path, tensorflow::kSavedModelVariablesDirectory);
  // TODO(b/228181641): revisit non-explicit-checkpoint-loading behavior when
  // MLAs come along
  if (Env::Default()->FileExists(variables_dir).ok()) {
    tf_package.has_checkpoint_ = true;
    tf_package.variables_filepath_ = tensorflow::io::JoinPath(
        variables_dir, tensorflow::kSavedModelVariablesFilename);
    tf_package.variable_reader_ = std::make_unique<tensorflow::BundleReader>(
        tensorflow::Env::Default(), tf_package.variables_filepath_);
    tensorflow::Tensor object_graph_tensor;
    TF_RETURN_IF_ERROR(tf_package.variable_reader_->Lookup(
        tensorflow::kObjectGraphProtoKey, &object_graph_tensor));
    const auto* object_graph_string =
        reinterpret_cast<const tensorflow::tstring*>(
            object_graph_tensor.tensor_data().data());
    // TODO(danielellis): make sure parse was successful
    tf_package.trackable_object_graph_.ParseFromString(*object_graph_string);
  } else {
    tf_package.has_checkpoint_ = false;
    LOG(INFO)
        << "No checkpoint found, assuming this is a program-only SavedModel";
  }

  // Build a map of node names to their corresponding nodes.
  //
  // See `GetGraphDefNode` for more details.
  const auto& nodes =
      tf_package.saved_model_proto_.meta_graphs(0).graph_def().node();
  for (const auto& node : nodes) {
    tf_package.graph_def_nodes_by_name_[node.name()] = &node;
  }
  return tf_package;
}

absl::StatusOr<std::string> TFPackage::GetVariableCheckpointKey(int index) {
  // TODO(danielellis): make sure valid index
  const auto& trackable_object = trackable_object_graph_.nodes(index);
  const TrackableObjectGraph::TrackableObject::SerializedTensor*
      serialized_tensor = nullptr;
  for (auto& maybe_serialized_tensor : trackable_object.attributes()) {
    if (maybe_serialized_tensor.name() == "VARIABLE_VALUE") {
      serialized_tensor = &maybe_serialized_tensor;
    }
  }
  if (serialized_tensor == nullptr) {
    return absl::Status(absl::StatusCode::kInternal,
                        "Failed to find variable value field.");
  }
  return serialized_tensor->checkpoint_key();
}

const SavedObjectGraph& TFPackage::GetObjectGraph() {
  return saved_model_proto_.mutable_meta_graphs(0)->object_graph_def();
}

absl::StatusOr<const tensorflow::NodeDef*> TFPackage::GetGraphDefNode(
    std::string name) {
  const auto& iter = graph_def_nodes_by_name_.find(name);
  if (iter == graph_def_nodes_by_name_.end()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrCat("Failed to find node named ", name));
  }
  return iter->second;
}

const RepeatedPtrField<FunctionDef>& TFPackage::GetFunctionDefs() {
  auto& function_library =
      saved_model_proto_.mutable_meta_graphs(0)->graph_def().library();
  return function_library.function();
}

}  // namespace libexport
}  // namespace tensorflow
