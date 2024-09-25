/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/saved_model/core/tf_saved_model_api.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/ops/restore_ops.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function.h"
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"

namespace tensorflow {

// Maps from a FunctionDef's name to FunctionDef, for a given FunctionDefLibrary
using FunctionDefMap = gtl::FlatMap<StringPiece, const tensorflow::FunctionDef*,
                                    StringPieceHasher>;

// Maps from a functiondef's name to the corresponding "TFConcreteFunction"
using FlatTensorFunctionMap =
    gtl::FlatMap<std::string, std::unique_ptr<FlatTensorFunction>>;

namespace {

const TrackableObjectGraph::TrackableObject::SerializedTensor*
FindSerializedTensorInTrackable(
    const TrackableObjectGraph::TrackableObject& trackable_object,
    absl::string_view name) {
  for (const auto& maybe_serialized_tensor : trackable_object.attributes()) {
    if (maybe_serialized_tensor.name() == name) {
      return &maybe_serialized_tensor;
    }
  }
  return nullptr;
}

// This function reads the Checkpoint embedded in the SavedModel, and calls the
// appropriate Restore ops on each of the variables.
// Note(bmzhao): Conceptually, objects that contain checkpointable state
// implement the "_gather_saveables_for_checkpoint" method
// https://github.com/tensorflow/tensorflow/blob/ddc1bbad3dfd4a089eb96014f26cc16664b1b2f8/tensorflow/python/training/tracking/base.py#L953-L983
// which returns a dict of string key -> EITHER:
// 1. python callable (taking a checkpoint key) returning SaveableObject OR
// 2. variable (partitioned/resource/reference or otherwise)
// https://github.com/tensorflow/tensorflow/blob/ddc1bbad3dfd4a089eb96014f26cc16664b1b2f8/tensorflow/python/training/saving/saveable_object.py#L58.
// The string key becomes the "name" attribute of the SerializedTensor proto
// in the TrackableObjectGraph,
// https://github.com/tensorflow/tensorflow/blob/ddc1bbad3dfd4a089eb96014f26cc16664b1b2f8/tensorflow/core/protobuf/trackable_object_graph.proto#L26
// And the checkpoint_key is a globally unique string derived from this name:
// https://github.com/tensorflow/tensorflow/blob/842df9e6b516e42578a8d23b35d41176b9a6cf1d/tensorflow/python/training/tracking/graph_view.py#L236-L241
// SaveableObjects model the information needed to pass to the SaveV2/RestoreV2
// ops via their SaveSpec members
// https://github.com/tensorflow/tensorflow/blob/ddc1bbad3dfd4a089eb96014f26cc16664b1b2f8/tensorflow/python/training/saving/saveable_object.py#L21,
// which contain the "real" checkpoint keys into the TensorBundle SSTable.
// They also contain the logic needed to take the restored tensors from
// RestoreV2 and load them back into the "object" they came from via their
// overridden "restore" method:
// https://github.com/tensorflow/tensorflow/blob/ddc1bbad3dfd4a089eb96014f26cc16664b1b2f8/tensorflow/python/training/saving/saveable_object.py#L85
Status RestoreCheckpoint(SavedModelV2Bundle* bundle,
                         const RevivedObjects& revived_objects,
                         const std::string& directory,
                         ImmediateExecutionContext* context) {
  // TODO(bmzhao): Batch up all the restores into a single restore op per
  // device, following logic in MultiDeviceSaver.
  TF_RETURN_IF_ERROR(bundle->VisitObjectsToRestore(
      [&revived_objects, &directory, context, bundle](
          int node, const TrackableObjectGraph::TrackableObject& trackable) {
        if (bundle->saved_object_graph().nodes(node).kind_case() !=
            SavedObject::kVariable) {
          // TODO(bmzhao): This requires using the newly added Save/Restore
          // functions from
          // https://github.com/tensorflow/tensorflow/commit/df6b21c13c82b5d0981642cfe18f10e60f78ea5c
          LOG(WARNING) << "Restoring non-variable objects has not been "
                          "implemented yet. (Kind="
                       << bundle->saved_object_graph().nodes(node).kind_case()
                       << ")";
          return absl::OkStatus();
        }

        Variable* variable = revived_objects.variables.at(node).get();

        // Restore the tensor's value from the checkpoint
        const TrackableObjectGraph::TrackableObject::SerializedTensor*
            attribute =
                FindSerializedTensorInTrackable(trackable, "VARIABLE_VALUE");
        if (attribute == nullptr) {
          return errors::FailedPrecondition(
              "Could not find SerializedTensor with name VARIABLE_VALUE for "
              "saved variable");
        }

        const std::string& checkpoint_key = attribute->checkpoint_key();
        if (!bundle->variable_reader()->Contains(checkpoint_key)) {
          LOG(WARNING) << "No checkpoint entry found for " << checkpoint_key
                       << ". Variable will be uninitialized.";
          return Status();
        }

        std::string variables_path_prefix =
            io::JoinPath(directory, kSavedModelVariablesDirectory,
                         kSavedModelVariablesFilename);
        ImmediateTensorHandlePtr restored_output;
        TF_RETURN_IF_ERROR(internal::SingleRestore(
            context, variables_path_prefix, checkpoint_key, variable->dtype(),
            &restored_output));

        // Assign the restored tensor's value to the variable
        return variable->Assign(restored_output.get());
      }));

  return Status();
}

Status InitializeAllResources(const RevivedObjects& revived) {
  for (const auto& node_and_resource : revived.restored_resources) {
    const RestoredResource& resource = node_and_resource.second;
    TF_RETURN_IF_ERROR(resource.Initialize());
  }
  return Status();
}

}  // namespace

Status TFSavedModelAPI::GetFunction(const std::string& function_path,
                                    ConcreteFunction** function) {
  absl::optional<int> node =
      internal::FindNodeAtPath(function_path, bundle_.saved_object_graph());
  if (!node.has_value()) {
    return errors::NotFound("No saved object found at path ", function_path);
  }

  *function = revived_objects_.concrete_functions.Find(*node);
  if (*function == nullptr) {
    return errors::NotFound("No function found at path ", function_path);
  }

  return Status();
}

Status TFSavedModelAPI::GetFunctions(
    int node_id,
    absl::flat_hash_map<std::string, ConcreteFunction*>* functions) {
  const auto& nodes = bundle_.saved_object_graph().nodes();
  if (node_id >= nodes.size()) {
    return errors::OutOfRange(
        "node_id ", node_id,
        " not found.  Maximum node ID: ", nodes.size() - 1);
  }
  const SavedObject* current_node = &nodes.Get(node_id);
  for (const auto& child : current_node->children()) {
    ConcreteFunction* concrete_fn;
    Status status = GetFunction(child.local_name(), &concrete_fn);
    if (status.ok()) {
      (*functions)[child.local_name()] = concrete_fn;
    }
  }
  return Status();
}

Status TFSavedModelAPI::GetSignatureDefFunction(
    const std::string& signature_def_key, SignatureDefFunction** function) {
  auto signatures_iter =
      revived_objects_.signatures_map.find(signature_def_key);
  if (signatures_iter == revived_objects_.signatures_map.end()) {
    return errors::NotFound("No signature with key ", signature_def_key,
                            " was found");
  }
  int node = signatures_iter->second;

  auto function_iter = revived_objects_.signature_def_functions.find(node);
  if (function_iter == revived_objects_.signature_def_functions.end()) {
    return errors::Internal(
        "Unable to find SignatureDefFunction associated with key ",
        signature_def_key, " despite key being valid.");
  }

  *function = function_iter->second.get();
  return Status();
}

Status TFSavedModelAPI::GetVariable(const std::string& variable_path,
                                    Variable** variable) {
  absl::optional<int> node =
      internal::FindNodeAtPath(variable_path, bundle_.saved_object_graph());
  if (!node.has_value()) {
    return errors::NotFound("No saved object found at path ", variable_path);
  }

  auto variables_iter = revived_objects_.variables.find(*node);
  if (variables_iter == revived_objects_.variables.end()) {
    return errors::NotFound("No variable found at path ", variable_path);
  }

  *variable = variables_iter->second.get();
  return Status();
}

SavedModelV2Bundle* TFSavedModelAPI::GetBundle() { return &this->bundle_; }

TFSavedModelAPI::TFSavedModelAPI(const std::string& directory,
                                 SavedModelV2Bundle bundle,
                                 RevivedObjects revived_objects)
    : directory_(directory),
      bundle_(std::move(bundle)),
      revived_objects_(std::move(revived_objects)) {}

Status TFSavedModelAPI::Load(
    const std::string& directory,
    const absl::optional<std::unordered_set<std::string>>& tags,
    ImmediateExecutionContext* context, std::unique_ptr<TFSavedModelAPI>* out) {
  // TODO(bmzhao): Add support for loading a TF1 SavedModel.
  if (tags) {
    return errors::Unimplemented(
        "Loading saved models with explicit tags will be supported in the "
        "future");
  }

  SavedModelV2Bundle bundle;
  TF_RETURN_IF_ERROR(SavedModelV2Bundle::Load(directory, &bundle));

  // TODO(bmzhao): Mangle loaded function names so that different
  // models loaded in the same runtime Context don't clobber eachother.
  // This occurs in python here:
  // https://github.com/tensorflow/tensorflow/blob/285b5fa15405c5e2c084080f52a1818be8648079/tensorflow/python/saved_model/function_deserialization.py#L438-L454

  // For each node in the graph, we should initialize an object of the
  // corresponding type. For objects that depend on the initialization of other
  // objects (like functions which capture resources), we will initialize them
  // later.
  PartiallyRevivedObjects partially_revived_objects;
  TF_RETURN_IF_ERROR(internal::PartiallyReviveSavedModelObjects(
      bundle.meta_graph_def(), context, directory, &partially_revived_objects));

  RevivedObjects revived_objects;
  TF_RETURN_IF_ERROR(partially_revived_objects.Build(
      context, bundle.saved_object_graph(), &revived_objects));

  // Revive function library functions as concrete functions without captures.
  // This is necessary because object graph functions may refer to functions
  // _not_ in the object graph: A while loop, for example, will create two
  // auxiliary `while_cond` and `while_body` functions that are only present in
  // the graph def function library.
  for (const FunctionDef& function :
       bundle.meta_graph_def().graph_def().library().function()) {
    std::unique_ptr<TFConcreteFunction> concrete_function;
    TF_RETURN_IF_ERROR(TFConcreteFunction::Create(/*function_def=*/&function,
                                                  /*captures=*/{},
                                                  /*metadata=*/{},
                                                  /*ctx=*/context,
                                                  /*out=*/&concrete_function));
    revived_objects.concrete_functions.Insert(std::move(concrete_function));
  }

  TF_RETURN_IF_ERROR(
      RestoreCheckpoint(&bundle, revived_objects, directory, context));

  TF_RETURN_IF_ERROR(InitializeAllResources(revived_objects));

  out->reset(new TFSavedModelAPI(directory, std::move(bundle),
                                 std::move(revived_objects)));
  return Status();
}

}  // namespace tensorflow
