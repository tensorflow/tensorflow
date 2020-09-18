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

#include "absl/strings/str_split.h"
#include "tensorflow/c/experimental/saved_model/core/function_metadata.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
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

// Perform some basic sanity checks on SavedConcreteFunction's input and
// output signatures with respect to the corresponding FunctionDef's input
// and output args.
Status ValidateSavedFunctionCompatibleWithFunctionDef(
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
    return errors::FailedPrecondition(
        "FunctionDef ", name, " has ",
        function_def->signature().input_arg_size(),
        " inputs, but the SavedConcreteFunction has ", input_specs.size(),
        " flattened user inputs and ",
        saved_concrete_function.bound_inputs_size(), " captured inputs.");
  }

  const StructuredValue& output_signature =
      saved_concrete_function.output_signature();
  std::vector<const TensorSpecProto*> output_specs;
  TF_RETURN_IF_ERROR(FlattenSignature(output_signature, &output_specs));
  if (output_specs.size() != function_def->signature().output_arg_size()) {
    return errors::FailedPrecondition(
        "FunctionDef ", name, " has ",
        function_def->signature().output_arg_size(),
        " outputs, but the SavedConcreteFunction has ", output_specs.size(),
        " flattened outputs.");
  }

  return Status();
}

}  // namespace

Status LoadSavedAsset(ImmediateExecutionContext* ctx, const SavedAsset& asset,
                      const std::string& saved_model_dir,
                      absl::Span<const AssetFileDef> assets,
                      std::unique_ptr<Asset>* output) {
  int asset_index = asset.asset_file_def_index();
  if (asset_index >= assets.size()) {
    return errors::FailedPrecondition(
        "SavedAsset contained asset index ", asset_index,
        " but AssetFileDef only contains ", assets.size(), " # of assets");
  }
  const std::string& asset_filename = assets[asset_index].filename();
  return Asset::Create(ctx, saved_model_dir, asset_filename, output);
}

Status TensorProtoToConstant(ImmediateExecutionContext* ctx,
                             const TensorProto& proto,
                             std::unique_ptr<Constant>* output) {
  tensorflow::Tensor tensor;
  bool parse_result = tensor.FromProto(proto);
  if (!parse_result) {
    return errors::Internal("Failed to parse tensor from tensorproto");
  }

  TensorInterface tensor_interface(std::move(tensor));
  return Constant::Create(ctx, &tensor_interface, output);
}

// This follows the python variable restoration logic:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/saved_model/load.py#L407
Status LoadSavedVariable(ImmediateExecutionContext* ctx,
                         const SavedVariable& variable,
                         std::unique_ptr<Variable>* output) {
  const std::string& name = variable.name();
  tensorflow::TensorShape shape(variable.shape());
  tensorflow::DataType dtype = variable.dtype();

  TF_RETURN_IF_ERROR(Variable::CreateUninitialized(
      ctx, dtype, shape, name,
      variable.device().empty() ? nullptr : variable.device().c_str(), output));
  return Status();
}

Status LoadTFConcreteFunction(
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
      return errors::FailedPrecondition("Failed to find bound_input ",
                                        bound_input,
                                        " for SavedConcreteFunction");
    }
    captures.push_back(iter->second->handle());
  }

  return TFConcreteFunction::Create(function_def, std::move(captures), {}, ctx,
                                    out);
}

Status FlattenSignature(const StructuredValue& signature,
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
      return Status();
    }
    case StructuredValue::kTupleValue: {
      const TupleValue& tuple = signature.tuple_value();
      for (const StructuredValue& value : tuple.values()) {
        TF_RETURN_IF_ERROR(FlattenSignature(value, flattened_specs));
      }
      return Status();
    }
    case StructuredValue::kListValue: {
      const ListValue& list = signature.list_value();
      for (const StructuredValue& value : list.values()) {
        TF_RETURN_IF_ERROR(FlattenSignature(value, flattened_specs));
      }
      return Status();
    }
    case StructuredValue::kTensorSpecValue: {
      flattened_specs->push_back(&signature.tensor_spec_value());
      return Status();
    }
    case StructuredValue::kNoneValue: {
      // Base case: do nothing.
      // This arises, for example, as the top-level object of an output
      // signature when there are no return values.
      return Status();
    }
    default: {
      return errors::Internal("Unhandled structured value kind ",
                              signature.kind_case());
    }
  }
}

const SavedObject* FindNodeAtPath(StringPiece path,
                                  const SavedObjectGraph& object_graph,
                                  int* node_id) {
  const auto& nodes = object_graph.nodes();
  if (nodes.empty()) {
    return nullptr;
  }

  // Starting from the root, iterate through the saved object graph, matching
  // object names as we go.
  const SavedObject* current_node = &nodes.Get(0);

  for (absl::string_view object_name : absl::StrSplit(path, '.')) {
    auto child_node_iter = std::find_if(
        current_node->children().begin(), current_node->children().end(),
        [object_name](
            const TrackableObjectGraph::TrackableObject::ObjectReference& obj) {
          return object_name == obj.local_name();
        });
    if (child_node_iter == current_node->children().end()) {
      return nullptr;
    }
    if (node_id) {
      *node_id = child_node_iter->node_id();
    }
    current_node = &nodes.Get(child_node_iter->node_id());
  }

  return current_node;
}

std::unordered_map<StringPiece, const AttrValueMap*, StringPieceHasher>
NodeToAttrMap(const tensorflow::GraphDef& graphdef) {
  std::unordered_map<StringPiece, const AttrValueMap*, StringPieceHasher>
      result;
  for (const tensorflow::NodeDef& node : graphdef.node()) {
    result[node.name()] = &node.attr();
  }
  return result;
}

std::unordered_map<StringPiece, const tensorflow::FunctionDef*,
                   StringPieceHasher>
FunctionNameToFunctionDefMap(const FunctionDefLibrary& library) {
  std::unordered_map<StringPiece, const tensorflow::FunctionDef*,
                     StringPieceHasher>
      result;
  for (const FunctionDef& function_def : library.function()) {
    result[function_def.signature().name()] = &function_def;
  }
  return result;
}

}  // namespace internal
}  // namespace tensorflow
