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

#include "tensorflow/core/grappler/optimizers/data/vectorization_utils.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {

using function_utils::FunctionDefTensorDesc;

namespace {

void AddMapDefunOutput(FunctionDef* map_defun_fn, NodeDef* map_defun_node,
                       const string& output_retval, const DataType t) {
  // Set to unknown shape
  TensorShapeProto tensor_shape_proto;
  PartialTensorShape().AsProto(&tensor_shape_proto);

  function_utils::AddFunctionOutputWithUniqueName(
      "vectorized_out", output_retval, map_defun_fn, t);

  *(*map_defun_node->mutable_attr())["output_shapes"]
       .mutable_list()
       ->add_shape() = tensor_shape_proto;
  (*map_defun_node->mutable_attr())["output_types"].mutable_list()->add_type(t);
}

void RemoveMapDefunOutput(FunctionDef* outer_scope, FunctionDef* map_defun_fn,
                          NodeDef* map_defun_node, int output_position) {
  DCHECK_LT(output_position, map_defun_fn->signature().output_arg_size())
      << "Trying to remove output that doesn't exist. Output number: "
      << output_position;

  int num_later_outputs =
      map_defun_fn->signature().output_arg_size() - output_position - 1;

  // Remove from map_defun_fn's ret dict and output args
  map_defun_fn->mutable_ret()->erase(
      map_defun_fn->signature().output_arg(output_position).name());
  map_defun_fn->mutable_signature()->mutable_output_arg()->DeleteSubrange(
      output_position, 1);

  // Renumber outputs that come after
  for (int i = 0; i < num_later_outputs; ++i) {
    function_utils::ReplaceReferences(
        strings::StrCat(map_defun_node->name(),
                        ":output:", output_position + i + 1),
        strings::StrCat(map_defun_node->name(),
                        ":output:", output_position + i),
        outer_scope);
  }
  map_defun_node->mutable_attr()
      ->at("output_shapes")
      .mutable_list()
      ->mutable_shape()
      ->DeleteSubrange(output_position, 1);
  map_defun_node->mutable_attr()
      ->at("output_types")
      .mutable_list()
      ->mutable_type()
      ->ExtractSubrange(output_position, 1, nullptr);
}

int FindOutputToConvert(const FunctionDef& function,
                        const std::set<string>& unconvertible,
                        FunctionDefTensorDesc* f) {
  for (int i = function.signature().output_arg_size() - 1; i >= 0; --i) {
    const string& ret_key = function.signature().output_arg(i).name();
    *f = FunctionDefTensorDesc(function.ret().at(ret_key));

    if (unconvertible.find(f->node_name) == unconvertible.end()) {
      return i;
    }
  }
  return -1;
}

// Helper class that vectorizes the body of a MapDefun node, adding new
// operations to the graph that collectively compute the same value as what
// running the MapDefun function on slices of the input would produce.
// Each instance of the class encapsulates all the data necessary to vectorize a
// MapDefun op in place.
class Vectorization {
 public:
  Vectorization(FunctionDef* outer_scope, FunctionDef* map_defun_fn,
                NodeDef* map_defun_node)
      : outer_scope_(outer_scope),
        map_defun_fn_(map_defun_fn),
        map_defun_node_(map_defun_node) {}

  // Repeatedly tries to convert outputs of map_defun_fn_ into new nodes in
  // the outer_scope_, until there are no convertible outputs remaining.
  // This method is idempotent.
  void Vectorize();

 private:
  // Vectorizes the map defun function's output at output_position
  Status ConvertOutput(int output_position, const FunctionDefTensorDesc& desc);
  // Given a descriptor of the original output tensor, gets a string
  // corresponding to the converted output tensor.
  Status ConvertOutputHelper(const FunctionDefTensorDesc& output_desc,
                             string* converted);
  Status AddConversionMappingFromInput(
      const FunctionDefTensorDesc& output_desc);

  // Adds mappings from node's outputs tensors to converted output tensors,
  // creating the necessary new node(s). Generally, the steps to convert an op
  // are:
  // 1) Promote the inputs of the op inputs to outputs of the map_defun_fn_,
  //    and modify map_defun_node_ attrs accordingly
  // 2) Create new node(s) in outer_scope_ that act on batched input tensors.
  //    These operations collectively compute the same value as what running
  //    the original operation on slices of the input tensors would produce.
  //    For example, a Cast op in MapDefun translates to a Cast op in
  //    outer_scope_, since the vectorized version of Cast is itself.
  // 3) Set inputs of new node(s) to the corresponding converted inputs (that
  //    are now outputs of map_defun_node_)
  // 4) For each output of the old node, add the mapping of output strings to
  //    the conversion map (eg "Cast:y:0" -> "Vectorize/Cast:y:0")
  Status AddConversionMappingFromOp(const NodeDef& node,
                                    const FunctionDefTensorDesc& output_desc);

  // Maps a tensor name to the name of the corresponding vectorized tensor. For
  // example, "Cast:y:0" -> "Vectorize/Cast:y:0"
  std::map<string, string> conversion_map_;
  // Unconvertible node names
  std::set<string> unconvertible_;

  FunctionDef* outer_scope_;
  FunctionDef* map_defun_fn_;
  NodeDef* map_defun_node_;
};

Status Vectorization::AddConversionMappingFromOp(
    const NodeDef& node, const FunctionDefTensorDesc& output_desc) {
  for (const string& input_name : node.input()) {
    if (IsControlInput(input_name)) {
      return errors::InvalidArgument(
          "Vectorizing outputs with control inputs is currently not "
          "supported.");
    }
  }

  // TODO(rachelim): Have some mechanism for registering converters and some
  // uniform, simpler way to represent them.

  DataTypeVector types;
  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node.op(), &op_def));
  TF_RETURN_IF_ERROR(InputTypesForNode(node, *op_def, &types));

  std::vector<string> promoted_inputs;
  promoted_inputs.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    promoted_inputs.push_back(strings::StrCat(
        map_defun_node_->name(),
        ":output:", map_defun_fn_->signature().output_arg_size() + i));
  }

  auto vectorizer = VectorizerRegistry::Global()->Get(node.op());
  if (vectorizer == nullptr) {
    return errors::Unimplemented("No vectorizer registered for op: ",
                                 node.op());
  }

  TF_RETURN_IF_ERROR(vectorizer->Vectorize(node, promoted_inputs, outer_scope_,
                                           &conversion_map_));

  // If we get here, the conversion was successful, so we promote the inputs
  // of the ops to MapDefun outputs.
  for (int i = 0; i < types.size(); ++i) {
    AddMapDefunOutput(map_defun_fn_, map_defun_node_, node.input(i), types[i]);
  }

  return Status::OK();
}

Status Vectorization::AddConversionMappingFromInput(
    const FunctionDefTensorDesc& output_desc) {
  int input_index = function_utils::FindFunctionInputWithName(
      output_desc.node_name, *map_defun_fn_);
  if (input_index == -1) {
    return errors::Internal("Cannot convert non-existent input.");
  }

  conversion_map_[output_desc.full_str] = map_defun_node_->input(input_index);
  return Status::OK();
}

Status Vectorization::ConvertOutputHelper(
    const FunctionDefTensorDesc& output_desc, string* converted) {
  // It's possible the output already has a mapping, if it comes from a node
  // that has already been converted.
  if (auto found = gtl::FindOrNull(conversion_map_, output_desc.full_str)) {
    *converted = *found;
    return Status::OK();
  }

  int index = function_utils::FindFunctionNodeWithName(output_desc.node_name,
                                                       *map_defun_fn_);
  if (index == -1) {  // The output comes from an input
    TF_RETURN_IF_ERROR(AddConversionMappingFromInput(output_desc));
  } else {
    TF_RETURN_IF_ERROR(AddConversionMappingFromOp(
        map_defun_fn_->node_def(index), output_desc));
  }
  *converted = conversion_map_.at(output_desc.full_str);
  return Status::OK();
}

Status Vectorization::ConvertOutput(int output_position,
                                    const FunctionDefTensorDesc& output_desc) {
  string converted_output_name;
  TF_RETURN_IF_ERROR(ConvertOutputHelper(output_desc, &converted_output_name));

  // Remove the old output and make everything that referenced it point
  // to the new string
  function_utils::ReplaceReferences(
      strings::StrCat(map_defun_node_->name(), ":output:", output_position),
      converted_output_name, outer_scope_);
  RemoveMapDefunOutput(outer_scope_, map_defun_fn_, map_defun_node_,
                       output_position);

  return Status::OK();
}

void Vectorization::Vectorize() {
  while (true) {
    FunctionDefTensorDesc desc;
    int output_position =
        FindOutputToConvert(*map_defun_fn_, unconvertible_, &desc);
    if (output_position == -1) break;

    if (!ConvertOutput(output_position, desc).ok()) {
      unconvertible_.insert(desc.node_name);
    }
  }

  // If we've converted all the outputs of the MapDefun function, we no longer
  // need the MapDefun node and can delete it.
  if (map_defun_fn_->signature().output_arg_size() == 0) {
    outer_scope_->mutable_node_def()->DeleteSubrange(
        function_utils::FindFunctionNodeWithName(map_defun_node_->name(),
                                                 *outer_scope_),
        1);
  }

  if (!unconvertible_.empty()) {
    VLOG(2) << "The following nodes could not be converted: ["
            << absl::StrJoin(unconvertible_, ", ") << "].";
  }
}
}  // namespace

void VectorizeMapDefun(FunctionDef* outer_scope, FunctionDef* map_defun_fn,
                       NodeDef* map_defun_node) {
  Vectorization(outer_scope, map_defun_fn, map_defun_node).Vectorize();
}

}  // end namespace vectorization_utils
}  // end namespace grappler
}  // end namespace tensorflow
