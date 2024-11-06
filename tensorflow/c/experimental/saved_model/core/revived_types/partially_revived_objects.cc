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

#include "tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function_metadata.h"
#include "tensorflow/c/experimental/saved_model/core/tensor_spec.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

namespace {

using StructuredValueDictEntry =
    protobuf::MapPair<std::string, StructuredValue>;

using NamedParamMap =
    gtl::FlatMap<StringPiece, const TensorSpecProto*, StringPieceHasher>;

absl::Status AssertAllCreateResourceFunctionsHaveNoCaptures(
    const PartiallyRevivedObjects& objects) {
  for (const auto& id_and_resource : objects.restored_resources) {
    int node_id = id_and_resource.first;
    const RestoredResourceRevivalState& resource = id_and_resource.second;
    const TFConcreteFunctionRevivalState* create_resource_fn =
        resource.create_resource;
    if (create_resource_fn == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Resource at node ", node_id,
                       " did not have a create_resource() function"));
    }
    const SavedConcreteFunction* saved_create_resource_fn =
        create_resource_fn->saved_concrete_func;
    if (!saved_create_resource_fn->bound_inputs().empty()) {
      // TODO(b/124045874): Support loading resource functions via a top sort
      return absl::UnimplementedError(
          "Create Resource functions with captures are currently unsupported.");
    }
  }
  return absl::Status();
}

// Retrieves the TensorHandle associated with `node_id` from `obj_graph`, and
// set `*handle` to point to it.
absl::Status TensorHandleFromNode(int node_id,
                                  const SavedObjectGraph& obj_graph,
                                  const PartiallyRevivedObjects& objects,
                                  ImmediateExecutionTensorHandle** handle) {
  const SavedObject& node = obj_graph.nodes(node_id);
  SavedObject::KindCase kind = node.kind_case();
  switch (kind) {
    case SavedObject::kVariable: {
      const auto& variables_iter = objects.variables.find(node_id);
      if (variables_iter == objects.variables.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Tried to convert node id ", node_id,
            " of type variable to tensor but the variable wasn't initialized"));
      }
      *handle = variables_iter->second->handle();
      return absl::Status();
    }
    case SavedObject::kConstant: {
      const auto& constants_iter = objects.constants.find(node_id);
      if (constants_iter == objects.constants.end()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Tried to convert node id ", node_id,
                         " of type constant to tensor but the "
                         "constant wasn't initialized"));
      }
      *handle = constants_iter->second->handle();
      return absl::Status();
    }
    case SavedObject::kAsset: {
      const auto& assets_iter = objects.assets.find(node_id);
      if (assets_iter == objects.assets.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Tried to convert node id ", node_id,
            " of type asset to tensor but the asset wasn't initialized"));
      }
      *handle = assets_iter->second->handle();
      return absl::Status();
    }
    case SavedObject::kResource: {
      const auto& resource_iter = objects.restored_resources.find(node_id);
      if (resource_iter == objects.restored_resources.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Tried to convert node id ", node_id,
            " of type Resource to tensor but the Resource wasn't initialized"));
      }
      const RestoredResourceRevivalState& resource = resource_iter->second;
      if (resource.resource_handle == nullptr) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Resource with node id ", node_id,
            " should have its resource_handle created, but was nullptr."));
      }
      *handle = resource.resource_handle.get();
      return absl::Status();
    }
    default: {
      return absl::FailedPreconditionError(absl::StrCat(
          "Only objects of type variable, constant, asset, and resources have "
          "capturable tensorhandles. Encountered object of kind ",
          node.kind_case(), " at node id: ", node_id));
    }
  }
}

std::vector<SignatureDefParam> SignatureDefParamsFromNamedParamMap(
    const NamedParamMap& params) {
  // The underlying functiondef associated with the SignatureDef has
  // nest.flattened inputs and outputs, which are sorted by string key.
  std::vector<SignatureDefParam> result;
  result.reserve(params.size());
  for (const auto& named_param : params) {
    result.push_back(SignatureDefParam(std::string(named_param.first),
                                       TensorSpec(*named_param.second)));
  }
  std::sort(result.begin(), result.end(),
            [](const SignatureDefParam& x, const SignatureDefParam& y) {
              return x.name() < y.name();
            });

  return result;
}

// SignatureDefArgsFromInputs takes the "canonicalized_input_signature"
// field of a SavedConcreteFunction, ensures it conforms to the structure of
// tuple(tuple(), dict<string,TensorSpec>()), and "returns" a list of
// SignatureDefParams of the SignatureDefFunction's arguments.
absl::Status SignatureDefArgsFromInputs(
    const StructuredValue& canonicalized_input_signature,
    std::vector<SignatureDefParam>* out) {
  // Note(bmzhao): canonicalized_input_signature should be a tuple of
  // (args, kwargs), where args is an empty tuple, and kwargs is a dictionary of
  // string keys to TensorSpecs.
  if (!canonicalized_input_signature.has_tuple_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SignatureDefFunction's canonicalized_input_signature should be "
        "of form tuple(tuple(), dict()), but was instead: \n",
        canonicalized_input_signature.DebugString()));
  }

  const TupleValue& args_kwargs_tuple =
      canonicalized_input_signature.tuple_value();
  if (args_kwargs_tuple.values_size() != 2) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SignatureDefFunction's canonicalized_input_signature should be "
        "a tuple of two elements (args, kwargs), but was instead: \n",
        args_kwargs_tuple.DebugString()));
  }

  const StructuredValue& args = args_kwargs_tuple.values(0);
  if (!args.has_tuple_value() || !args.tuple_value().values().empty()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SignatureDefFunction's canonicalized_input_signature's args"
        "should be an empty tuple, but instead got: \n",
        args.DebugString()));
  }

  const StructuredValue& kwargs = args_kwargs_tuple.values(1);
  if (!kwargs.has_dict_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SignatureDefFunction's canonicalized_input_signature's kwargs"
        "should be a dictionary, but instead got: \n",
        kwargs.DebugString()));
  }

  const DictValue& kwargs_dict = kwargs.dict_value();
  NamedParamMap result;
  result.reserve(kwargs_dict.fields_size());

  for (const auto& key_value : kwargs_dict.fields()) {
    const std::string& key = key_value.first;
    const StructuredValue& value = key_value.second;
    if (!value.has_tensor_spec_value()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "SignatureDefFunction's canonicalized_input_signature's kwargs"
          "dictionary contained a non-tensorspec value for key-value pair: \n",
          "Key: ", key, "Value: \n", value.DebugString()));
    }
    result[key] = &value.tensor_spec_value();
  }

  *out = SignatureDefParamsFromNamedParamMap(result);

  return absl::Status();
}

// SignatureDefReturnsFromOutputs takes the "output_signature" field of a
// SavedConcreteFunction, ensures it conforms to the structure of
// dict<string,TensorSpec>(), and "returns" a list of SignatureDefParams of the
// SignatureDefFunction's returns.
absl::Status SignatureDefReturnsFromOutputs(
    const StructuredValue& output_signature,
    std::vector<SignatureDefParam>* out) {
  if (!output_signature.has_dict_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SignatureDefFunction's output_signature must be a dictionary, but "
        "instead got: ",
        output_signature.DebugString()));
  }

  const DictValue& output_dict = output_signature.dict_value();
  NamedParamMap result;
  result.reserve(output_dict.fields_size());

  for (const auto& key_value : output_dict.fields()) {
    const std::string& key = key_value.first;
    const StructuredValue& value = key_value.second;
    if (!value.has_tensor_spec_value()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "SignatureDefFunction's output_signature dictionary contained a "
          "non-tensorspec value for key-value pair: \n",
          "Key: ", key, "Value: \n", value.DebugString()));
    }
    result[key] = &value.tensor_spec_value();
  }
  *out = SignatureDefParamsFromNamedParamMap(result);

  return absl::Status();
}

// The implementation takes advantage of the fact that SignatureDefFunction's
// "traced" Signature wrapper function always has inputs/outputs of dictionaries
// https://github.com/tensorflow/tensorflow/blob/53cdd5e87c423b195f33775753273286fd5a1a65/tensorflow/python/saved_model/signature_serialization.py#L119-L126
// https://github.com/tensorflow/tensorflow/blob/53cdd5e87c423b195f33775753273286fd5a1a65/tensorflow/python/saved_model/signature_serialization.py#L153-L178
// Additionally, we take advantage of the fact that the SignatureDefFunction's
// associated functiondef has lexicographically ordered inputs/outputs due to
// nest.flatten.
absl::Status LoadSignatureDefFunctionMetadata(
    const SavedConcreteFunction& saved_concrete_function,
    SignatureDefFunctionMetadata* out) {
  std::vector<SignatureDefParam> args;
  TF_RETURN_IF_ERROR(SignatureDefArgsFromInputs(
      saved_concrete_function.canonicalized_input_signature(), &args));

  std::vector<SignatureDefParam> rets;
  TF_RETURN_IF_ERROR(SignatureDefReturnsFromOutputs(
      saved_concrete_function.output_signature(), &rets));

  *out = SignatureDefFunctionMetadata(std::move(args), std::move(rets));
  return absl::Status();
}

// This function finds the necessary captures, then forwards to the builder
// method
absl::Status CreateConcreteFunction(
    ImmediateExecutionContext* ctx,
    const TFConcreteFunctionRevivalState& builder,
    const SavedObjectGraph& obj_graph, const PartiallyRevivedObjects& objects,
    std::unique_ptr<TFConcreteFunction>* out) {
  const auto& capture_node_ids = builder.saved_concrete_func->bound_inputs();
  std::vector<ImmediateExecutionTensorHandle*> captures;
  captures.reserve(capture_node_ids.size());
  for (int capture_node_id : capture_node_ids) {
    ImmediateExecutionTensorHandle* capture_handle;
    TF_RETURN_IF_ERROR(TensorHandleFromNode(capture_node_id, obj_graph, objects,
                                            &capture_handle));
    captures.push_back(capture_handle);
  }
  // TODO(bmzhao): Create Metadata here
  return TFConcreteFunction::Create(/*function_def=*/builder.fdef,
                                    /*captures=*/std::move(captures),
                                    /*metadata=*/{},
                                    /*ctx=*/ctx,
                                    /*out=*/out);
}

absl::Status CreateSignatureDefFunction(
    ImmediateExecutionContext* ctx,
    const TFSignatureDefFunctionRevivalState& builder,
    const SavedObjectGraph& obj_graph, const PartiallyRevivedObjects& objects,
    std::unique_ptr<TFSignatureDefFunction>* out) {
  const auto& capture_node_ids = builder.saved_concrete_func->bound_inputs();
  std::vector<ImmediateExecutionTensorHandle*> captures;
  captures.reserve(capture_node_ids.size());
  for (int capture_node_id : capture_node_ids) {
    ImmediateExecutionTensorHandle* capture_handle;
    TF_RETURN_IF_ERROR(TensorHandleFromNode(capture_node_id, obj_graph, objects,
                                            &capture_handle));
    captures.push_back(capture_handle);
  }

  SignatureDefFunctionMetadata metadata;
  TF_RETURN_IF_ERROR(LoadSignatureDefFunctionMetadata(
      *builder.saved_concrete_func, &metadata));

  return TFSignatureDefFunction::Create(/*function_def=*/builder.fdef,
                                        /*captures=*/std::move(captures),
                                        /*metadata=*/std::move(metadata),
                                        /*ctx=*/ctx,
                                        /*out=*/out);
}

absl::Status InitializeCreateResourceFunctions(
    ImmediateExecutionContext* ctx, const SavedObjectGraph& obj_graph,
    const PartiallyRevivedObjects& objects, RevivedObjects* revived) {
  for (const auto& id_and_resource : objects.restored_resources) {
    const RestoredResourceRevivalState& resource = id_and_resource.second;
    const TFConcreteFunctionRevivalState* create_resource_fn =
        resource.create_resource;

    const SavedConcreteFunction* saved_create_resource_fn =
        create_resource_fn->saved_concrete_func;
    if (!saved_create_resource_fn->bound_inputs().empty()) {
      // TODO(b/124045874): Load resource functions via a topological sort
      return absl::UnimplementedError(
          "Create Resource functions with captures are currently unsupported.");
    }
    std::unique_ptr<TFConcreteFunction> out;
    TF_RETURN_IF_ERROR(CreateConcreteFunction(ctx, *create_resource_fn,
                                              obj_graph, objects, &out));
    revived->concrete_functions.Insert(std::move(out),
                                       create_resource_fn->node_id);
  }
  return absl::Status();
}

absl::Status InitializeAllFunctions(ImmediateExecutionContext* ctx,
                                    const SavedObjectGraph& obj_graph,
                                    const PartiallyRevivedObjects& objects,
                                    RevivedObjects* revived) {
  gtl::FlatMap<int, std::unique_ptr<TFSignatureDefFunction>>*
      destination_sig_map = &revived->signature_def_functions;

  for (const auto& id_and_func : objects.concrete_functions) {
    int node_id = id_and_func.first;
    const TFConcreteFunctionRevivalState& func = id_and_func.second;

    if (revived->concrete_functions.Find(node_id)) {
      // The function has already been initialized in the destination_map,
      // so we can skip this node. This can occur because we initialize
      // CreateResource functions before calling this function.
      continue;
    }

    std::unique_ptr<TFConcreteFunction> out;
    TF_RETURN_IF_ERROR(
        CreateConcreteFunction(ctx, func, obj_graph, objects, &out));
    revived->concrete_functions.Insert(std::move(out), node_id);
  }

  for (const auto& id_and_func : objects.signature_def_functions) {
    int node_id = id_and_func.first;
    const TFSignatureDefFunctionRevivalState& func = id_and_func.second;

    if (destination_sig_map->find(node_id) != destination_sig_map->end()) {
      continue;
    }

    std::unique_ptr<TFSignatureDefFunction> out;
    TF_RETURN_IF_ERROR(
        CreateSignatureDefFunction(ctx, func, obj_graph, objects, &out));
    (*destination_sig_map)[node_id] = std::move(out);
  }

  return absl::Status();
}

absl::Status CreateAllResourceHandles(ImmediateExecutionContext* ctx,
                                      const SavedObjectGraph& obj_graph,
                                      PartiallyRevivedObjects* objects,
                                      RevivedObjects* revived) {
  for (auto& id_and_resource : objects->restored_resources) {
    RestoredResourceRevivalState& resource = id_and_resource.second;
    int create_resource_fn_node = resource.create_resource->node_id;

    const TFConcreteFunction* create_resource_fn =
        revived->concrete_functions.Find(create_resource_fn_node);
    if (create_resource_fn == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("ConcreteFunction at node ", create_resource_fn_node,
                       " should have been initialized prior to being called."));
    }
    ImmediateOpPtr function_op;
    TF_RETURN_IF_ERROR(create_resource_fn->MakeCallOp({}, &function_op));
    TF_RETURN_IF_ERROR(function_op->SetDeviceName(resource.device.c_str()));

    AbstractTensorHandle* resource_handle = nullptr;
    int num_retvals = 1;
    TF_RETURN_IF_ERROR(function_op->Execute(
        absl::MakeSpan(&resource_handle, num_retvals), &num_retvals));
    AbstractTensorHandlePtr owned_resource_handle(resource_handle);
    if (!tensorflow::isa<ImmediateExecutionTensorHandle>(
            owned_resource_handle.get())) {
      return absl::InternalError("Unexpected tensor handle kind.");
    }
    ImmediateTensorHandlePtr result(
        reinterpret_cast<ImmediateExecutionTensorHandle*>(
            owned_resource_handle.release()));
    resource.resource_handle = std::move(result);
  }
  return absl::Status();
}

absl::Status BuildResources(ImmediateExecutionContext* ctx,
                            const SavedObjectGraph& obj_graph,
                            PartiallyRevivedObjects* objects,
                            RevivedObjects* revived) {
  for (auto& id_and_resource : objects->restored_resources) {
    int node_id = id_and_resource.first;
    RestoredResourceRevivalState& resource_revival_state =
        id_and_resource.second;

    TFConcreteFunction* create_resource = nullptr;

    // Check all the functions associated with the resource have already been
    // initialized in `revived`
    if (resource_revival_state.create_resource != nullptr) {
      create_resource = revived->concrete_functions.Find(
          resource_revival_state.create_resource->node_id);
      if (create_resource == nullptr) {
        return absl::FailedPreconditionError(absl::StrCat(
            "'create_resource' function with node id ",
            resource_revival_state.create_resource->node_id, " not found"));
      }
    }

    TFConcreteFunction* initialize = nullptr;
    if (resource_revival_state.initialize != nullptr) {
      initialize = revived->concrete_functions.Find(
          resource_revival_state.initialize->node_id);
      if (initialize == nullptr) {
        return absl::FailedPreconditionError(absl::StrCat(
            "'initialize' function with node id ",
            resource_revival_state.initialize->node_id, " not found"));
      }
    }

    TFConcreteFunction* destroy_resource = nullptr;
    if (resource_revival_state.destroy_resource != nullptr) {
      destroy_resource = revived->concrete_functions.Find(
          resource_revival_state.destroy_resource->node_id);
      if (destroy_resource == nullptr) {
        return absl::FailedPreconditionError(absl::StrCat(
            "'destroy_resource' function with node id ",
            resource_revival_state.destroy_resource->node_id, " not found"));
      }
    }

    if (resource_revival_state.resource_handle == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Resource at node id ", node_id,
                       " does not have a resource handle."));
    }

    revived->restored_resources.emplace(
        node_id, RestoredResource(
                     /*device=*/resource_revival_state.device,
                     /*create_resource=*/create_resource,
                     /*initialize=*/initialize,
                     /*destroy_resource=*/destroy_resource,
                     /*resource_handle=*/
                     std::move(resource_revival_state.resource_handle)));
  }
  return absl::Status();
}

}  // namespace

absl::Status PartiallyRevivedObjects::Build(ImmediateExecutionContext* ctx,
                                            const SavedObjectGraph& obj_graph,
                                            RevivedObjects* revived) {
  // Step 1: We would like to initialize all functions; this requires setting up
  // their captured tensorhandles, which may come from variables, assets,
  // constants, or resources. The first three are trivial; However,
  // tensorhandles that correspond to resources must be created by invoking
  // their "create_resource" function.
  // https://github.com/tensorflow/tensorflow/blob/f19c6efb4a8ba60e2492eedc98ef5375abb39dc7/tensorflow/python/saved_model/load.py#L240
  // https://github.com/tensorflow/tensorflow/blob/f19c6efb4a8ba60e2492eedc98ef5375abb39dc7/tensorflow/python/training/tracking/tracking.py#L233
  // For now, we assert that all create_resource functions must have no
  // captures. This aligns with the current behavior in python.
  // https://github.com/tensorflow/tensorflow/blob/50eac986bf7a0ad12594e080f083181f277e0b49/tensorflow/python/saved_model/load.py#L152-L155
  // TODO(bmzhao): We should do a topological sort instead.

  // 1a. Make sure all CreateResource functions have no captures.
  TF_RETURN_IF_ERROR(AssertAllCreateResourceFunctionsHaveNoCaptures(*this));

  // 1b. Initialize all CreateResource functions, storing them in `revived`
  TF_RETURN_IF_ERROR(
      InitializeCreateResourceFunctions(ctx, obj_graph, *this, revived));

  // 1c. Invoke all "CreateResource" functions and store their ResourceHandles
  // https://github.com/tensorflow/tensorflow/blob/3b6b41b68a95dc70c26dc816b29d359bfb88c116/tensorflow/python/training/tracking/tracking.py#L241-L247
  // in *this->resources.
  // TODO(bmzhao): Maybe store them separately, not in *this?
  TF_RETURN_IF_ERROR(CreateAllResourceHandles(ctx, obj_graph, this, revived));

  // 2. Initialize all the rest of the functions
  TF_RETURN_IF_ERROR(InitializeAllFunctions(ctx, obj_graph, *this, revived));

  // 3a. Move over all non-function, non-resource objects
  revived->variables = std::move(variables);
  revived->assets = std::move(assets);
  revived->constants = std::move(constants);
  revived->signatures_map = std::move(signatures_map);

  // 3b. Move over resources.
  TF_RETURN_IF_ERROR(BuildResources(ctx, obj_graph, this, revived));

  return absl::Status();
}

}  // namespace tensorflow
