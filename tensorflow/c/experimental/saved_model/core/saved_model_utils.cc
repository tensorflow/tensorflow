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

#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "tensorflow/c/experimental/saved_model/core/function_metadata.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"

namespace tensorflow {
namespace internal {
namespace {

using StructuredValueDictEntry =
    protobuf::MapPair<std::string, StructuredValue>;

// Maps from a Nodedef's name to its corresponding AttrValues, for a given
// Graphdef
using NodeAttrMap =
    gtl::FlatMap<absl::string_view, const AttrValueMap*, StringPieceHasher>;

// Maps from a FunctionDef's name to FunctionDef, for a given FunctionDefLibrary
using FunctionDefMap =
    gtl::FlatMap<absl::string_view, const tensorflow::FunctionDef*,
                 StringPieceHasher>;

// Looks up a SavedConstant's associated tensorproto from the NodeAttrMap and
// returns a tensorflow::Constant.
absl::Status ConstantFromSavedConstant(
    ImmediateExecutionContext* ctx,
    const tensorflow::SavedConstant& saved_constant,
    const NodeAttrMap& node_attr_map, std::unique_ptr<Constant>* output) {
  const std::string& const_op_name = saved_constant.operation();
  const auto& node_name_and_attrs = node_attr_map.find(const_op_name);
  if (node_name_and_attrs == node_attr_map.end()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unable to find Const operation with name'", const_op_name,
                     "' in SavedModel graphdef"));
  }
  const AttrValueMap* attrs = node_name_and_attrs->second;
  const auto& attr_name_and_value = attrs->find("value");
  if (attr_name_and_value == attrs->end()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unable to find Const operation '", const_op_name,
                     "'s value attribute"));
  }
  const TensorProto& tensor_proto = attr_name_and_value->second.tensor();
  return internal::TensorProtoToConstant(ctx, tensor_proto, output);
}

// Perform some basic sanity checks on SavedConcreteFunction's input and
// output signatures with respect to the corresponding FunctionDef's input
// and output args.
absl::Status ValidateSavedFunctionCompatibleWithFunctionDef(
    const SavedConcreteFunction& saved_concrete_function,
    const FunctionDef* function_def) {
  // tf.functions go through many transformations before becoming FunctionDefs
  // 1. flatten user-provided inputs:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/eager/function.py#L2671-L2675
  // 2. convert user-provided inputs to tensors:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/eager/function.py#L2687-L2688
  // 3. filter any non-tensor, non-variable inputs:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/eager/function.py#L1840-L1841
  // 4. concatenate any captured inputs:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/eager/function.py#L1912

  // Since our API is limited to tf.functions annotated with input signatures,
  // conditions 2 and 3 are trivially satisfied.
  // We need to ensure that:
  // flatten(input_signature).size() + captures.size() = fdef.signature().size()
  // A concrete function's serialized "canonicalized_input_signature" comes
  // from encoding its "structured_input_signature" field:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/saved_model/function_serialization.py#L70-L71
  // The "structured_input_signature" is guaranteed to be a tuple of the python
  // args, kwargs that correspond to the tf.function:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/eager/function.py#L1974-L1979

  const std::string& name = function_def->signature().name();

  const StructuredValue& input_signature =
      saved_concrete_function.canonicalized_input_signature();
  std::vector<const TensorSpecProto*> input_specs;
  TF_RETURN_IF_ERROR(FlattenSignature(input_signature, &input_specs));
  if (input_specs.size() + saved_concrete_function.bound_inputs_size() !=
      function_def->signature().input_arg_size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "FunctionDef ", name, " has ",
        function_def->signature().input_arg_size(),
        " inputs, but the SavedConcreteFunction has ", input_specs.size(),
        " flattened user inputs and ",
        saved_concrete_function.bound_inputs_size(), " captured inputs."));
  }

  const StructuredValue& output_signature =
      saved_concrete_function.output_signature();
  std::vector<const TensorSpecProto*> output_specs;
  TF_RETURN_IF_ERROR(FlattenSignature(output_signature, &output_specs));
  if (output_specs.size() != function_def->signature().output_arg_size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("FunctionDef ", name, " has ",
                     function_def->signature().output_arg_size(),
                     " outputs, but the SavedConcreteFunction has ",
                     output_specs.size(), " flattened outputs."));
  }

  return absl::Status();
}

// Looks up `function_name` in both `function_def_map` and `obj_graph`'s
// concrete_functions map. Both maps are keyed by function name, which comes
// from the untrusted SavedModel, so a malformed or crafted model can
// reference a function_name that is missing from either map --
// gtl::FlatMap::at() and protobuf::Map::at() both crash on a missing key,
// so this uses find() and returns an error instead.
absl::Status LookupFunctionAndSavedConcreteFunction(
    const FunctionDefMap& function_def_map, const SavedObjectGraph& obj_graph,
    const std::string& function_name, const FunctionDef** function_def,
    const SavedConcreteFunction** saved_concrete_func) {
  auto function_def_iter = function_def_map.find(function_name);
  if (function_def_iter == function_def_map.end()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unable to find FunctionDef with name '", function_name,
                     "' in SavedModel graphdef"));
  }
  *function_def = function_def_iter->second;

  const auto& concrete_functions = obj_graph.concrete_functions();
  auto concrete_func_iter = concrete_functions.find(function_name);
  if (concrete_func_iter == concrete_functions.end()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unable to find SavedConcreteFunction with name '",
                     function_name, "' in SavedObjectGraph"));
  }
  *saved_concrete_func = &concrete_func_iter->second;
  return absl::Status();
}

}  // namespace

absl::Status GetSignaturesMap(const SavedObjectGraph& saved_objects,
                              gtl::FlatMap<std::string, int>* signatures_map) {
  if (saved_objects.nodes().empty()) {
    return absl::FailedPreconditionError("Saved Object Graph was empty.");
  }
  const SavedObject& root = saved_objects.nodes(0);
  const SavedObject* signatures = nullptr;
  for (const auto& child : root.children()) {
    if (child.local_name() == "signatures") {
      if (child.node_id() < 0 ||
          child.node_id() >= saved_objects.nodes().size()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Signature object had child node id ", child.node_id(),
                         " which is out of range of the set of nodes"));
      }
      signatures = &saved_objects.nodes(child.node_id());
    }
  }

  // Some basic sanity checks that this object is actually our "signatures" map
  if (signatures == nullptr) {
    // This is where the "signatures" attribute is always set:
    // https://github.com/tensorflow/tensorflow/blob/a2c542a0d83227568f9214a2af9a38ae3625976f/tensorflow/python/saved_model/save.py#L1106-L1109
    return absl::FailedPreconditionError(
        "SavedObjectGraph's root object must have a child 'signatures' object");
  }
  if (signatures->kind_case() != SavedObject::kUserObject) {
    return absl::FailedPreconditionError(
        "Signatures must be a SavedObject of type UserObject.");
  }
  if (signatures->user_object().identifier() != "signature_map") {
    // This is where the string comes from:
    // https://github.com/tensorflow/tensorflow/blob/c59af2913aaec235d883f50428efef1086f4c0e6/tensorflow/python/saved_model/signature_serialization.py#L220
    return absl::FailedPreconditionError(
        "Signatures SavedObject must have identifier 'signature_map'.");
  }

  for (const auto& child : signatures->children()) {
    (*signatures_map)[child.local_name()] = child.node_id();
  }
  return absl::Status();
}

absl::Status ValidateSingleConcreteFunction(
    const SavedFunction& saved_function) {
  // We only allow loading functions that have an annotated input signature,
  // which means there is 1:1 correspondence between tf.function
  // <=> SavedFunction <=> SavedConcreteFunction <=> FunctionDef. This is
  // the same restriction that MLIR has:
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L2677-L2707
  if (saved_function.concrete_functions_size() != 1) {
    return absl::FailedPreconditionError(
        "Only tf.functions annotated with an input signature are supported "
        "by SavedModelAPI. This means that there should only be a single "
        "ConcreteFunction per tf.function");
  }
  return absl::Status();
}

absl::Status LoadSavedAsset(ImmediateExecutionContext* ctx,
                            const SavedAsset& asset,
                            const std::string& saved_model_dir,
                            absl::Span<const AssetFileDef> assets,
                            std::unique_ptr<Asset>* output) {
  int asset_index = asset.asset_file_def_index();
  if (asset_index >= assets.size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SavedAsset contained asset index ", asset_index,
        " but AssetFileDef only contains ", assets.size(), " # of assets"));
  }
  const std::string& asset_filename = assets[asset_index].filename();
  return Asset::Create(ctx, saved_model_dir, asset_filename, output);
}

absl::Status TensorProtoToConstant(ImmediateExecutionContext* ctx,
                                   const TensorProto& proto,
                                   std::unique_ptr<Constant>* output) {
  tensorflow::Tensor tensor;
  bool parse_result = tensor.FromProto(proto);
  if (!parse_result) {
    return absl::InternalError("Failed to parse tensor from tensorproto");
  }

  TensorInterface tensor_interface(std::move(tensor));
  return Constant::Create(ctx, &tensor_interface, output);
}

// This follows the python variable restoration logic:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/saved_model/load.py#L407
absl::Status LoadSavedVariable(ImmediateExecutionContext* ctx,
                               const SavedVariable& variable,
                               std::unique_ptr<Variable>* output) {
  const std::string& name = variable.name();
  tensorflow::TensorShape shape(variable.shape());
  tensorflow::DataType dtype = variable.dtype();
  std::vector<std::string> component_devices;

  for (const auto& component :
       variable.experimental_distributed_variable_components()) {
    component_devices.push_back(component.device());
  }

  TF_RETURN_IF_ERROR(Variable::CreateUninitialized(
      ctx, dtype, shape, name,
      variable.device().empty() ? nullptr : variable.device().c_str(),
      component_devices, output));
  return absl::Status();
}

absl::Status LoadTFConcreteFunction(
    const SavedConcreteFunction& saved_concrete_function,
    const FunctionDef* function_def,
    const std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>>&
        captured_objects,
    ImmediateExecutionContext* ctx, std::unique_ptr<TFConcreteFunction>* out) {
  TF_RETURN_IF_ERROR(ValidateSavedFunctionCompatibleWithFunctionDef(
      saved_concrete_function, function_def));

  // Copy over captures
  std::vector<ImmediateExecutionTensorHandle*> captures;
  captures.reserve(saved_concrete_function.bound_inputs_size());
  for (int bound_input : saved_concrete_function.bound_inputs()) {
    auto iter = captured_objects.find(bound_input);
    if (iter == captured_objects.end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Failed to find bound_input ", bound_input,
                       " for SavedConcreteFunction"));
    }
    captures.push_back(iter->second->handle());
  }

  return TFConcreteFunction::Create(function_def, std::move(captures), {}, ctx,
                                    out);
}

absl::Status FlattenSignature(
    const StructuredValue& signature,
    std::vector<const TensorSpecProto*>* flattened_specs) {
  // This follows the logic from
  // https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L2775
  switch (signature.kind_case()) {
    case StructuredValue::kDictValue: {
      // Dictionaries must be sorted in order of keys
      const DictValue& dict = signature.dict_value();
      std::vector<const StructuredValueDictEntry*> entries;
      entries.reserve(dict.fields_size());
      for (const auto& field : dict.fields()) {
        entries.push_back(&field);
      }

      std::sort(entries.begin(), entries.end(),
                [](const StructuredValueDictEntry* x,
                   const StructuredValueDictEntry* y) {
                  return x->first < y->first;
                });

      for (const auto& entry : entries) {
        TF_RETURN_IF_ERROR(FlattenSignature(entry->second, flattened_specs));
      }
      return absl::Status();
    }
    case StructuredValue::kTupleValue: {
      const TupleValue& tuple = signature.tuple_value();
      for (const StructuredValue& value : tuple.values()) {
        TF_RETURN_IF_ERROR(FlattenSignature(value, flattened_specs));
      }
      return absl::Status();
    }
    case StructuredValue::kListValue: {
      const ListValue& list = signature.list_value();
      for (const StructuredValue& value : list.values()) {
        TF_RETURN_IF_ERROR(FlattenSignature(value, flattened_specs));
      }
      return absl::Status();
    }
    case StructuredValue::kTensorSpecValue: {
      flattened_specs->push_back(&signature.tensor_spec_value());
      return absl::Status();
    }
    case StructuredValue::kNoneValue: {
      // Base case: do nothing.
      // This arises, for example, as the top-level object of an output
      // signature when there are no return values.
      return absl::Status();
    }
    default: {
      return absl::InternalError(absl::StrCat(
          "Unhandled structured value kind ", signature.kind_case()));
    }
  }
}

std::optional<int> FindNodeAtPath(absl::string_view path,
                                  const SavedObjectGraph& object_graph) {
  const auto& nodes = object_graph.nodes();
  if (nodes.empty()) {
    return std::nullopt;
  }

  // Starting from the root, iterate through the saved object graph, matching
  // object names as we go.
  int node_id = 0;
  const SavedObject* current_node = &nodes.Get(node_id);

  for (absl::string_view object_name : absl::StrSplit(path, '.')) {
    auto child_node_iter = std::find_if(
        current_node->children().begin(), current_node->children().end(),
        [object_name](
            const TrackableObjectGraph::TrackableObject::ObjectReference& obj) {
          return object_name == obj.local_name();
        });
    if (child_node_iter == current_node->children().end()) {
      return std::nullopt;
    }

    node_id = child_node_iter->node_id();
    if (node_id < 0 || node_id >= nodes.size()) {
      return std::nullopt;
    }
    current_node = &nodes.Get(node_id);
  }

  return node_id;
}

gtl::FlatMap<absl::string_view, const AttrValueMap*, StringPieceHasher>
NodeToAttrMap(const tensorflow::GraphDef& graphdef) {
  gtl::FlatMap<absl::string_view, const AttrValueMap*, StringPieceHasher>
      result;
  for (const tensorflow::NodeDef& node : graphdef.node()) {
    result[node.name()] = &node.attr();
  }
  return result;
}

gtl::FlatMap<absl::string_view, const tensorflow::FunctionDef*,
             StringPieceHasher>
FunctionNameToFunctionDefMap(const FunctionDefLibrary& library) {
  gtl::FlatMap<absl::string_view, const tensorflow::FunctionDef*,
               StringPieceHasher>
      result;
  for (const FunctionDef& function_def : library.function()) {
    result[function_def.signature().name()] = &function_def;
  }
  return result;
}

absl::Status PartiallyReviveSavedModelObjects(
    const MetaGraphDef& metagraph, ImmediateExecutionContext* context,
    const std::string& directory, PartiallyRevivedObjects* objects) {
  // This is needed to restore "Constant" nodes by looking up their
  // "Value" attribute.
  NodeAttrMap node_attr_map = NodeToAttrMap(metagraph.graph_def());

  // These are needed for creating "Assets", by looking up their filenames.
  std::vector<AssetFileDef> assets;
  TF_RETURN_IF_ERROR(GetAssetFileDefs(metagraph, &assets));

  // Signatures are needed for determining whether a function is a
  // SignatureDefFunction or not.
  gtl::FlatMap<std::string, int> signatures_map;
  TF_RETURN_IF_ERROR(
      GetSignaturesMap(metagraph.object_graph_def(), &signatures_map));

  gtl::FlatMap<int, std::string> reversed_signatures_map;
  reversed_signatures_map.reserve(signatures_map.size());
  for (const auto& signature_key_and_node : signatures_map) {
    reversed_signatures_map.emplace(signature_key_and_node.second,
                                    signature_key_and_node.first);
  }

  // FunctionDefs are needed to help construct
  // TFConcreteFunction/SignatureDefFunctions
  const FunctionDefMap function_def_map =
      internal::FunctionNameToFunctionDefMap(metagraph.graph_def().library());

  // Iterate through all the saved objects, restoring objects (if we can) as we
  // go. For objects that dependencies on other objects (resources/functions),
  // we partially initialize "builders" that correspond to their currently known
  // state, and gradually fill them out in subsequent passes.
  for (int i = 0; i < metagraph.object_graph_def().nodes_size(); ++i) {
    const SavedObject& node = metagraph.object_graph_def().nodes(i);
    if (node.kind_case() == SavedObject::kVariable) {
      std::unique_ptr<Variable> variable;
      TF_RETURN_IF_ERROR(
          LoadSavedVariable(context, node.variable(), &variable));
      objects->variables[i] = std::move(variable);
    } else if (node.kind_case() == SavedObject::kConstant) {
      std::unique_ptr<Constant> constant;
      TF_RETURN_IF_ERROR(ConstantFromSavedConstant(context, node.constant(),
                                                   node_attr_map, &constant));
      objects->constants[i] = std::move(constant);
    } else if (node.kind_case() == SavedObject::kAsset) {
      std::unique_ptr<Asset> asset;
      TF_RETURN_IF_ERROR(
          LoadSavedAsset(context, node.asset(), directory, assets, &asset));
      objects->assets[i] = std::move(asset);
    } else if (node.kind_case() == SavedObject::kResource) {
      RestoredResourceRevivalState resource_revival_state;
      // We'll set the resource's functions in a subsequent pass, once we get
      // all functions in a partially revived state.
      resource_revival_state.device = node.resource().device();
      objects->restored_resources[i] = std::move(resource_revival_state);
    } else if (node.kind_case() == SavedObject::kFunction) {
      // Get the SavedFunction node and skip if it has no concrete functions.
      const SavedFunction& saved_function = node.function();
      if (saved_function.concrete_functions_size() < 1) {
        continue;
      }

      // Retrieve related function information.
      const std::string& function_name = saved_function.concrete_functions(0);
      const FunctionDef* function_def = nullptr;
      const SavedConcreteFunction* saved_concrete_func = nullptr;
      TF_RETURN_IF_ERROR(LookupFunctionAndSavedConcreteFunction(
          function_def_map, metagraph.object_graph_def(), function_name,
          &function_def, &saved_concrete_func));
      const FunctionSpec& function_spec = saved_function.function_spec();

      // Construct either a SignatureDefFunctionBuilder or a
      // ConcreteFunctionBuilder, depending on whether this node was a child
      // of the "signatures" attribute from root object.
      auto reverse_signature_iter = reversed_signatures_map.find(i);
      if (reverse_signature_iter != reversed_signatures_map.end()) {
        TFSignatureDefFunctionRevivalState func_revival_state;
        func_revival_state.node_id = i;
        func_revival_state.fdef = function_def;
        func_revival_state.saved_concrete_func = saved_concrete_func;
        func_revival_state.signature_key = reverse_signature_iter->second;
        objects->signature_def_functions[i] = std::move(func_revival_state);
      } else {
        TFConcreteFunctionRevivalState func_revival_state;
        func_revival_state.node_id = i;
        func_revival_state.fdef = function_def;
        func_revival_state.saved_concrete_func = saved_concrete_func;
        func_revival_state.function_spec = &function_spec;
        objects->concrete_functions[i] = std::move(func_revival_state);
      }
    } else if (node.kind_case() == SavedObject::kBareConcreteFunction) {
      const SavedBareConcreteFunction& bare_cf = node.bare_concrete_function();

      // Retrieve related function information.
      const std::string& function_name = bare_cf.concrete_function_name();
      const FunctionDef* function_def = nullptr;
      const SavedConcreteFunction* saved_concrete_func = nullptr;
      TF_RETURN_IF_ERROR(LookupFunctionAndSavedConcreteFunction(
          function_def_map, metagraph.object_graph_def(), function_name,
          &function_def, &saved_concrete_func));

      // Check whether this is a SignatureDefFunction, or not.
      auto reverse_signature_iter = reversed_signatures_map.find(i);
      if (reverse_signature_iter != reversed_signatures_map.end()) {
        TFSignatureDefFunctionRevivalState func_revival_state;
        func_revival_state.node_id = i;
        func_revival_state.fdef = function_def;
        func_revival_state.saved_concrete_func = saved_concrete_func;
        func_revival_state.signature_key = reverse_signature_iter->second;
        objects->signature_def_functions[i] = std::move(func_revival_state);
      } else {
        TFConcreteFunctionRevivalState func_revival_state;
        func_revival_state.node_id = i;
        func_revival_state.fdef = function_def;
        func_revival_state.saved_concrete_func = saved_concrete_func;
        objects->concrete_functions[i] = std::move(func_revival_state);
      }
    }
  }

  // Now that we've partially restored all functions, we can have resources
  // point to them
  const SavedObjectGraph& obj_graph = metagraph.object_graph_def();
  for (auto& node_and_resource_revival_state : objects->restored_resources) {
    int node_id = node_and_resource_revival_state.first;
    const SavedObject& node = obj_graph.nodes(node_id);
    RestoredResourceRevivalState& resource =
        node_and_resource_revival_state.second;
    for (const TrackableObjectGraph::TrackableObject::ObjectReference& child :
         node.children()) {
      // Note(bmzhao): The expected functions saved by a resource object are:
      // "_create_resource", "_initialize", and "_destroy_resource".
      // https://github.com/tensorflow/tensorflow/blob/ad66f588c1666ade8051feb42811fa27b285271c/tensorflow/python/training/tracking/tracking.py#L277-L281
      if (child.local_name() != "_create_resource" &&
          child.local_name() != "_initialize" &&
          child.local_name() != "_destroy_resource") {
        continue;
      }
      int child_node_id = child.node_id();
      if (child_node_id < 0 || child_node_id >= obj_graph.nodes().size()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Resource object had child node id ", child_node_id,
                         " which is out of range of the set of nodes"));
      }
      if (obj_graph.nodes(child_node_id).kind_case() !=
          SavedObject::kFunction) {
        continue;
      }
      // Using operator[] here would silently insert a default-constructed
      // (and therefore null fdef/saved_concrete_func) entry if child_node_id
      // isn't actually a revived concrete function -- e.g. if it was
      // classified as a SignatureDefFunction instead, or skipped for having
      // no concrete functions. Look it up explicitly instead so a malformed
      // reference is rejected rather than silently producing a
      // TFConcreteFunctionRevivalState with null fields that get dereferenced
      // later.
      auto concrete_function_iter =
          objects->concrete_functions.find(child_node_id);
      if (concrete_function_iter == objects->concrete_functions.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Resource object's ", child.local_name(), " child node id ",
            child_node_id, " was not found in revived concrete functions"));
      }
      TFConcreteFunctionRevivalState* concrete_function =
          &concrete_function_iter->second;
      if (child.local_name() == "_create_resource") {
        resource.create_resource = concrete_function;
      } else if (child.local_name() == "_initialize") {
        resource.initialize = concrete_function;
      } else if (child.local_name() == "_destroy_resource") {
        resource.destroy_resource = concrete_function;
      }
    }
  }

  objects->signatures_map = std::move(signatures_map);

  return absl::Status();
}

}  // namespace internal
}  // namespace tensorflow
