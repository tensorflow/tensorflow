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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/ops/restore_ops.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function.h"
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
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
using FunctionDefMap =
    std::unordered_map<StringPiece, const tensorflow::FunctionDef*,
                       StringPieceHasher>;

// Maps from a Nodedef's name to its corresponding AttrValues, for a given
// Graphdef
using NodeAttrMap =
    std::unordered_map<StringPiece, const AttrValueMap*, StringPieceHasher>;

// Maps from Node ID to an "Revived Object" implementing
// "TensorHandleConvertible"
using RevivedObjectMap =
    std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>>;

// Maps from a functiondef's name to the corresponding "TFConcreteFunction"
using ConcreteFunctionMap =
    std::unordered_map<std::string, std::unique_ptr<TFConcreteFunction>>;

namespace {

Status ConstantFromSavedConstant(
    ImmediateExecutionContext* ctx,
    const tensorflow::SavedConstant& saved_constant,
    const NodeAttrMap& node_attr_map, std::unique_ptr<Constant>* output) {
  const std::string& const_op_name = saved_constant.operation();
  const auto& node_name_and_attrs = node_attr_map.find(const_op_name);
  if (node_name_and_attrs == node_attr_map.end()) {
    return errors::FailedPrecondition(
        "Unable to find Const operation with name'", const_op_name,
        "' in SavedModel graphdef");
  }
  const AttrValueMap* attrs = node_name_and_attrs->second;
  const auto& attr_name_and_value = attrs->find("value");
  if (attr_name_and_value == attrs->end()) {
    return errors::FailedPrecondition("Unable to find Const operation '",
                                      const_op_name, "'s value attribute");
  }
  const TensorProto& tensor_proto = attr_name_and_value->second.tensor();
  return internal::TensorProtoToConstant(ctx, tensor_proto, output);
}

// Restores all non-function objects in the SavedModel's object graph.
// This function walks through the metagraph's saved object graph, and
// constructs revived versions of SavedVariable, SavedConstant, SavedAsset, and
// SavedResources. These are returned via the `out` parameter.
Status ReviveObjects(
    const MetaGraphDef& metagraph, ImmediateExecutionContext* context,
    const std::string& directory,
    std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>>*
        revived_objects) {
  // This is needed to restore "Constant" nodes by looking up their
  // "Value" attribute.
  NodeAttrMap node_attr_map = internal::NodeToAttrMap(metagraph.graph_def());

  // These are needed for creating "Assets", by looking up their filenames.
  std::vector<AssetFileDef> assets;
  TF_RETURN_IF_ERROR(internal::GetAssetFileDefs(metagraph, &assets));

  // Iterate through all the saved objects, restoring objects as we go.
  // We don't recreate functions until all other objects have been created.
  for (int i = 0; i < metagraph.object_graph_def().nodes_size(); ++i) {
    const SavedObject& node = metagraph.object_graph_def().nodes(i);
    if (node.kind_case() == SavedObject::kVariable) {
      std::unique_ptr<Variable> variable;
      TF_RETURN_IF_ERROR(
          internal::LoadSavedVariable(context, node.variable(), &variable));
      (*revived_objects)[i] = std::move(variable);
    } else if (node.kind_case() == SavedObject::kConstant) {
      std::unique_ptr<Constant> constant;
      TF_RETURN_IF_ERROR(ConstantFromSavedConstant(context, node.constant(),
                                                   node_attr_map, &constant));
      (*revived_objects)[i] = std::move(constant);
    } else if (node.kind_case() == SavedObject::kAsset) {
      std::unique_ptr<Asset> asset;
      TF_RETURN_IF_ERROR(internal::LoadSavedAsset(context, node.asset(),
                                                  directory, assets, &asset));
      (*revived_objects)[i] = std::move(asset);
    } else if (node.kind_case() == SavedObject::kResource) {
      // TODO(bmzhao): Figure out how resource loading works and implement it
      return errors::Unimplemented(
          "SavedResource loading is not implemented yet");
    }
  }
  return Status();
}

Status ReviveFunctions(const MetaGraphDef& metagraph,
                       const RevivedObjectMap& revived_objects,
                       ImmediateExecutionContext* context,
                       ConcreteFunctionMap* restored_functions) {
  const FunctionDefMap function_def_map =
      internal::FunctionNameToFunctionDefMap(metagraph.graph_def().library());

  // Iterate through all objects, only examining functions.
  for (const SavedObject& node : metagraph.object_graph_def().nodes()) {
    if (node.kind_case() == SavedObject::kBareConcreteFunction) {
      const std::string& function_name =
          node.bare_concrete_function().concrete_function_name();

      const SavedConcreteFunction& saved_concrete_function =
          metagraph.object_graph_def().concrete_functions().at(function_name);

      const FunctionDef* function_def = function_def_map.at(function_name);
      std::unique_ptr<TFConcreteFunction> concrete_function;
      TF_RETURN_IF_ERROR(internal::LoadTFConcreteFunction(
          saved_concrete_function, function_def, revived_objects, context,
          &concrete_function));
      (*restored_functions)[function_name] = std::move(concrete_function);
    } else if (node.kind_case() == SavedObject::kFunction) {
      // We only allow loading functions that have an annotated input signature,
      // which means there is 1:1 correspondence between tf.function
      // <=> SavedFunction <=> SavedConcreteFunction <=> FunctionDef. This is
      // the same restriction that MLIR has:
      // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L2677-L2707
      const SavedFunction& saved_function = node.function();
      if (saved_function.concrete_functions_size() != 1) {
        return errors::FailedPrecondition(
            "Only tf.functions annotated with an input signature are supported "
            "by SavedModelAPI. This means that there should only be a single "
            "ConcreteFunction per tf.function");
      }
      const std::string& function_name = saved_function.concrete_functions(0);
      const SavedConcreteFunction& saved_concrete_function =
          metagraph.object_graph_def().concrete_functions().at(function_name);

      const FunctionDef* function_def = function_def_map.at(function_name);

      std::unique_ptr<TFConcreteFunction> concrete_function;
      TF_RETURN_IF_ERROR(internal::LoadTFConcreteFunction(
          saved_concrete_function, function_def, revived_objects, context,
          &concrete_function));
      (*restored_functions)[function_name] = std::move(concrete_function);
    }
  }
  return Status();
}

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
                         const RevivedObjectMap& revived_objects,
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
          return Status::OK();
        }

        Variable* variable =
            down_cast<Variable*>(revived_objects.at(node).get());

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

}  // namespace

Status TFSavedModelAPI::GetFunction(const std::string& function_path,
                                    ConcreteFunction** function) {
  const SavedObject* object =
      internal::FindNodeAtPath(function_path, bundle_.saved_object_graph());
  if (object == nullptr) {
    return errors::NotFound("No saved object found at path ", function_path);
  }

  if (object->kind_case() == SavedObject::kBareConcreteFunction) {
    *function =
        concrete_functions_
            .at(object->bare_concrete_function().concrete_function_name())
            .get();
  } else if (object->kind_case() == SavedObject::kFunction) {
    *function =
        concrete_functions_.at(object->function().concrete_functions(0)).get();
  } else {
    return errors::InvalidArgument(function_path,
                                   " is not a path to a Function.");
  }

  return Status();
}

Status TFSavedModelAPI::GetSignatureDefFunction(
    const std::string& signature_def_key, SignatureDefFunction** function) {
  // TODO(bmzhao): Add support for retrieving a signaturedef function.
  return errors::Unimplemented(
      "Retrieving SignatureDef functions is unimplemented currently");
}

std::vector<ConcreteFunction*> TFSavedModelAPI::ListFunctions() {
  std::vector<ConcreteFunction*> result;
  result.reserve(concrete_functions_.size());
  for (auto& index_and_function : concrete_functions_) {
    result.push_back(index_and_function.second.get());
  }
  return result;
}

Status TFSavedModelAPI::GetVariable(const std::string& variable_path,
                                    Variable** variable) {
  int node_id;
  const SavedObject* object = internal::FindNodeAtPath(
      variable_path, bundle_.saved_object_graph(), &node_id);
  if (object == nullptr) {
    return errors::NotFound("No saved object found at path ", variable_path);
  }

  if (object->kind_case() == SavedObject::kVariable) {
    auto iter = revived_objects_.find(node_id);
    if (iter == revived_objects_.end()) {
      return errors::Internal("Variable ", variable_path,
                              " was not properly revived.");
    }
    *variable = static_cast<Variable*>(iter->second.get());
    return Status();
  }

  *variable = nullptr;
  return errors::InvalidArgument(
      variable_path, " is not a path to a Variable (kind=", object->kind_case(),
      ")");
}

TFSavedModelAPI::TFSavedModelAPI(
    const std::string& directory, SavedModelV2Bundle bundle,
    std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>>
        revived_objects,
    std::unordered_map<std::string, std::unique_ptr<TFConcreteFunction>>
        concrete_functions)
    : directory_(directory),
      bundle_(std::move(bundle)),
      revived_objects_(std::move(revived_objects)),
      concrete_functions_(std::move(concrete_functions)) {}

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

  RevivedObjectMap revived_objects;
  TF_RETURN_IF_ERROR(ReviveObjects(bundle.meta_graph_def(), context, directory,
                                   &revived_objects));

  // TODO(bmzhao): When we later add support for loading resources, we need to
  // handle the case where materializing a function's captures requires invoking
  // other functions. This occurs when retrieving the resource handle for a
  // TrackableResource:
  // https://github.com/tensorflow/tensorflow/blob/f19c6efb4a8ba60e2492eedc98ef5375abb39dc7/tensorflow/python/saved_model/load.py#L240
  // https://github.com/tensorflow/tensorflow/blob/f19c6efb4a8ba60e2492eedc98ef5375abb39dc7/tensorflow/python/training/tracking/tracking.py#L233
  // This requires restoring functions in a topological sort order by capture
  // dependencies.
  ConcreteFunctionMap function_map;
  TF_RETURN_IF_ERROR(ReviveFunctions(bundle.meta_graph_def(), revived_objects,
                                     context, &function_map));

  TF_RETURN_IF_ERROR(
      RestoreCheckpoint(&bundle, revived_objects, directory, context));

  out->reset(new TFSavedModelAPI(directory, std::move(bundle),
                                 std::move(revived_objects),
                                 std::move(function_map)));
  return Status();
}

}  // namespace tensorflow
