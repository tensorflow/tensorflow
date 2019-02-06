/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api_internal.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/lib/strings/strcat.h"

using tensorflow::errors::InvalidArgument;

namespace tensorflow {
namespace {

// Class that maintains a one-to-one original node name -> new node name
// mapping. We normalize the names used as input and output arguments to match
// regexp "[a-z][a-z0-9_]*" specified in definition of ArgDef.name.
// Once we rename them, we risk creating a name collision with the other
// node names, so if necessary we add a suffix to make
// names unique. If we have an input named "A" and a node in the function
// body named "a", they will be renamed to "a" and "a_0".
class NodeNameMapping {
 public:
  NodeNameMapping() = default;

  // Normalize the input name and make it unique. This is the same as the
  // function for output, expect that it adds a name mapping for the name.
  string GetInputName(const string& name);

  // Normalize the output name and make it unique.
  string GetOutputName(const string& name);

  // Make the node name unique.
  string Uniquify(const string& name);

  // Records name as a used name. If this name is already used,
  // returns an error status.
  Status UseOutputName(const string& name);

  // Look up how a node name was previously normalized/uniquified.
  // Returns empty if name was never seen.
  string Lookup(const string& name) const;

 private:
  string UniquifyHelper(const string& name) const;
  static string Normalize(string name);

  // The normalized/uniquified names already used as
  // input names (in signature), output names (in signature), and node names
  // (in node_def).
  // This is a superset of values in name_mapping_.
  std::unordered_set<string> used_names_;
  // Mapping from original node name from the graph to the normalized
  // and uniquified version of it.
  std::unordered_map<string, string> name_mapping_;
};

string NodeNameMapping::Normalize(string name) {
  // Convert letters to lowercase and non-alphanumeric characters to '_'.
  if (name.empty()) return "unknown";
  const int n = name.size();
  for (int i = 0; i < n; ++i) {
    char c = name[i];
    if (isalnum(c)) {
      if (isupper(c)) {
        name[i] = tolower(c);
      }
    } else {
      name[i] = '_';
    }
  }

  // Find the first letter and start with it.
  int i = 0;
  for (; i < n; ++i) {
    if (isalpha(name[i])) break;
  }

  // Return "unknown" if none of the name's chars were letters.
  return i == n ? "unknown" : name.substr(i);
}

string NodeNameMapping::UniquifyHelper(const string& name) const {
  // If the name hasn't been used yet, use it as-is.
  if (used_names_.find(name) == used_names_.end()) return name;
  // Add a suffix to name to make it unique.
  for (int i = 0;; ++i) {
    const string candidate = strings::StrCat(name, "_", i);
    if (used_names_.find(candidate) == used_names_.end()) return candidate;
  }
}

string NodeNameMapping::GetInputName(const string& name) {
  const string& input_name = GetOutputName(name);
  name_mapping_[name] = input_name;
  return input_name;
}

string NodeNameMapping::GetOutputName(const string& name) {
  const string& input_name = UniquifyHelper(Normalize(name));
  // Record that we used this name, but don't add it to name_mapping_
  // since this name is not for a node.
  used_names_.insert(input_name);
  return input_name;
}

string NodeNameMapping::Uniquify(const string& name) {
  const string uniqued = UniquifyHelper(name);
  name_mapping_[name] = uniqued;
  used_names_.insert(uniqued);
  return uniqued;
}

Status NodeNameMapping::UseOutputName(const string& name) {
  const auto& iter = used_names_.find(name);
  if (iter != used_names_.end()) {
    return InvalidArgument("Cannot have duplicate output names. Name '", name,
                           "' appears more than once in 'output_names' array.");
  }
  used_names_.insert(iter, name);
  return Status::OK();
}

string NodeNameMapping::Lookup(const string& name) const {
  const auto iter = name_mapping_.find(name);
  if (iter == name_mapping_.end()) return string();
  return iter->second;
}

Status ValidateNonRefOutput(const Node* node, int idx) {
  const DataType& dt = node->output_type(idx);
  return IsRefType(dt)
             ? InvalidArgument("Output ", idx, " of node '", node->name(),
                               "' has a reference type ", DataTypeString(dt))
             : Status::OK();
}

Status FillFunctionBody(
    const string& fn_name, const NodeNameMapping& node_names,
    const std::vector<const Node*>& body_nodes,
    const std::unordered_map<string, string>& tensor_renaming,
    FunctionDef* fdef) {
  std::unordered_set<string> func_attr_names;
  for (const auto& func_attr : fdef->signature().attr()) {
    func_attr_names.insert(func_attr.name());
  }

  std::vector<const Edge*> in_edges;
  std::vector<const Edge*> control_edges;
  for (const Node* node : body_nodes) {
    NodeDef* node_def = fdef->add_node_def();
    // First, copy the node_def as is. We will patch it next.
    *node_def = node->def();
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }
    node_def->set_name(node_names.Lookup(node->name()));

    // Input names must be set based on nested names in tensor_renaming.
    // Clear the flat input names we got from the original node_def
    // from the graph.
    node_def->clear_input();

    // Collect regular and control inputs. Regular inputs are indexed
    // by the index at which they come into the `node`. Control inputs
    // don't follow any order.
    in_edges.clear();
    in_edges.resize(node->num_inputs(), nullptr);
    control_edges.clear();
    for (const Edge* edge : node->in_edges()) {
      if (edge->src()->IsSource()) continue;
      if (edge->IsControlEdge()) {
        control_edges.push_back(edge);
      } else {
        in_edges[edge->dst_input()] = edge;
      }
    }

    // Add regular inputs.
    for (size_t i = 0; i < in_edges.size(); ++i) {
      const Edge* edge = in_edges[i];
      string original_input_name;
      if (edge == nullptr) {
        // A backedge might not appear as a regular Edge, but be only present
        // in the node_def. Such edges are referred to as requested_inputs().
        if (i >= node->requested_inputs().size()) {
          return InvalidArgument(
              "Graph to be converted to function appears to be malformed. ",
              "Node ", node->name(), " is missing input edge ", i);
        }
        original_input_name =
            ParseTensorName(node->requested_inputs()[i]).ToString();
      } else {
        original_input_name =
            strings::StrCat(edge->src()->name(), ":", edge->src_output());
      }

      const auto iter = tensor_renaming.find(original_input_name);
      if (iter == tensor_renaming.end()) {
        return InvalidArgument(
            "Input ", i, ", '", original_input_name, "', of node '",
            node->name(), "' in function '", fn_name,
            "' is not available. You might need to include it in inputs "
            "or include its source node in the body");
      }
      node_def->add_input(iter->second);
    }

    // Add control inputs.
    for (const Edge* edge : control_edges) {
      // Add this control input only if the src node is in the body or a part of
      // the inputs.
      const string normalized = node_names.Lookup(edge->src()->name());
      // If we did not find a name for the source of control edge, this
      // source must be outside of the body, and not an input. Raise an error.
      if (normalized.empty()) {
        return InvalidArgument(
            "The source of control edge ", edge->DebugString(),
            " is not in the body. Encountered while creating function '",
            fn_name, "'");
      }
      node_def->add_input(strings::StrCat("^", normalized));
    }

    // A function is stateful if any of its nodes are stateful.
    if (node->op_def().is_stateful()) {
      fdef->mutable_signature()->set_is_stateful(true);
    }

    // If this node has any attributes with placeholder value, add the
    // attribute to FunctionDef signature.
    for (const auto& iter : node->attrs()) {
      if (iter.second.placeholder().empty()) {
        continue;
      }

      // If we already added the attribute, skip it.
      string func_attr_name = iter.second.placeholder();
      if (func_attr_names.find(func_attr_name) != func_attr_names.end()) {
        continue;
      }

      // This node's attribute is a placeholder value, so it does not have type
      // information. We check node's OpDef for attribute type.
      string node_attr_name = iter.first;
      const OpDef_AttrDef* node_attr_def = nullptr;
      for (const auto& node_attr : node->op_def().attr()) {
        if (node_attr.name() == node_attr_name) {
          node_attr_def = &node_attr;
        }
      }
      if (!node_attr_def) {
        return errors::Internal("Cannot find attr ", node_attr_name,
                                " in OpDef ", node->op_def().DebugString());
      }
      OpDef_AttrDef* attr_def = fdef->mutable_signature()->add_attr();
      attr_def->set_name(func_attr_name);
      attr_def->set_type(node_attr_def->type());

      func_attr_names.insert(func_attr_name);
    }
  }
  return Status::OK();
}

// Graph to FunctionDef conversion. This code is closely modeled on the Python
// code in tensorflow/python/framework/function.py.
Status GraphToFunctionDef(const Graph& fn_body, const string& fn_name,
                          bool append_hash_to_fn_name,
                          const std::vector<const Node*>& body_nodes,
                          const std::vector<OutputTensor>& inputs,
                          const std::vector<OutputTensor>& outputs,
                          const std::vector<string>& output_names,
                          const char* description, FunctionDef* fdef) {
  if (!output_names.empty()) {
    DCHECK_EQ(output_names.size(), outputs.size());
  }

  if (description != nullptr) {
    fdef->mutable_signature()->set_description(description);
  }

  // Keep track of names we used and how we normalized them.
  NodeNameMapping node_names;

  // Mapping from original names of tensors (i.e. "<node_name>:<idx>") to the
  // name we used in the function:
  //  - For input tensors:
  //    {flat_tensor_name -> normalized_name_of_src_node}
  //    e.g. {In:3 -> in}
  //  - For tensors produced by nodes in function's body:
  //    {flat_tensor_name -> nested_tensor_name}
  //    e.g. {Add:3 -> add_0:z:1}
  std::unordered_map<string, string> tensor_renaming;

  // Fill outputs in function's signature.
  // We fill the outputs first to prevent output_names from colliding
  // with the input names we pick below. With this order, no names are used in
  // node_names yet, and output_names won't collide with anything (except
  // potentially with themselves).
  for (size_t i = 0; i < outputs.size(); ++i) {
    const Node* node = outputs[i].node;
    int idx = outputs[i].index;
    OpDef::ArgDef* argdef = fdef->mutable_signature()->add_output_arg();
    argdef->set_type(node->output_type(idx));
    if (!output_names.empty()) {
      TF_RETURN_IF_ERROR(node_names.UseOutputName(output_names[i]));
      argdef->set_name(output_names[i]);
    } else {
      argdef->set_name(node_names.GetOutputName(node->name()));
    }
  }

  // Fill inputs in function's signature.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Node* node = inputs[i].node;
    int idx = inputs[i].index;
    OpDef::ArgDef* argdef = fdef->mutable_signature()->add_input_arg();
    argdef->set_type(node->output_type(idx));
    const string& input_name = node_names.GetInputName(node->name());
    argdef->set_name(input_name);
    tensor_renaming[strings::StrCat(node->name(), ":", idx)] = input_name;
  }

  // Populate tensor_renaming and node_names.
  // Generate the new output names for every node in the function.
  // The NodeDefs in FunctionDefs use a different naming scheme for
  // their inputs than the NodeDefs in a graph (see the comment for
  // FunctionDef.node_def in function.proto). We do the
  // graph tensor name -> function tensor name conversion for every
  // possible input (i.e. every node's outputs) and store the result
  // in tensor_renaming.
  for (const Node* node : body_nodes) {
    // Make sure node_name does not collide with an input or output name.
    const string& node_name = node_names.Uniquify(node->name());
    // For each output_arg in the op_def, the output_ranges
    // map will have [start, end] range of indices that this arg produces
    // among all the output tensors of this op.
    NameRangeMap output_ranges;
    TF_RETURN_IF_ERROR(
        NameRangesForNode(*node, node->op_def(), nullptr, &output_ranges));
    for (const auto& output : output_ranges) {
      const StringPiece& output_name = output.first;
      int index_start = output.second.first;
      int index_end = output.second.second;
      for (int i = index_start; i < index_end; ++i) {
        const string& original_name = strings::StrCat(node->name(), ":", i);
        const string& new_name =
            strings::StrCat(node_name, ":", output_name, ":", i - index_start);
        // Record the mapping if this tensor is not already mapped.
        // Tensor can be already mapped if it is used as an input.
        if (tensor_renaming.find(original_name) == tensor_renaming.end()) {
          tensor_renaming[original_name] = new_name;
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(
      FillFunctionBody(fn_name, node_names, body_nodes, tensor_renaming, fdef));

  // Remap return values.
  for (int r = 0; r < fdef->signature().output_arg_size(); ++r) {
    const string& ret_name = fdef->signature().output_arg(r).name();
    // We convert this flat tensor name to the nested value
    // (e.g. `add:z:1`) that we stored in tensor_renaming.
    const string& return_value =
        strings::StrCat(outputs[r].node->name(), ":", outputs[r].index);
    const auto iter = tensor_renaming.find(return_value);
    if (iter == tensor_renaming.end()) {
      return InvalidArgument(
          "TF_Output ", return_value, " is neither in the function body ",
          "nor among function inputs. Encountered while creating function '",
          fn_name, "'");
    }
    (*fdef->mutable_ret())[ret_name] = iter->second;
  }

  if (append_hash_to_fn_name) {
    const uint64 hash = FunctionDefHash(*fdef);
    string encoded;
    TF_RETURN_IF_ERROR(Base64Encode(
        StringPiece(reinterpret_cast<const char*>(&hash), sizeof(hash)),
        &encoded));
    // Besides letters and digits our Base64 encoding uses '_' and '-'.
    // Dash is invalid in operation names and multiple underscores in random
    // places look strange. Since we never need to decode the hash back,
    // replace these chars with with 'a' and 'A'. Replacing with different
    // letters keeps more entropy.
    std::replace(encoded.begin(), encoded.end(), '-', 'a');
    std::replace(encoded.begin(), encoded.end(), '_', 'A');
    fdef->mutable_signature()->set_name(strings::StrCat(fn_name, "_", encoded));
  } else {
    fdef->mutable_signature()->set_name(fn_name);
  }

  return Status::OK();
}

// Converts `ninputs` and `inputs` into `inputs_tensors` and `input_nodes` and
// does various checks while doing so. `input_nodes` will contain the same
// information as input_tensors just in a different structure to make
// following processing easier. TODO(iga): Simplify this nested structure.
Status ProcessInputs(
    const TF_Graph* fn_body, const char* fn_name, int ninputs,
    const TF_Output* inputs, std::vector<OutputTensor>* input_tensors,
    std::unordered_map<const Node*, std::vector<int>>* input_nodes)
    EXCLUSIVE_LOCKS_REQUIRED(fn_body->mu) {
  input_tensors->reserve(ninputs);
  for (int i = 0; i < ninputs; ++i) {
    Node* node = &inputs[i].oper->node;
    int idx = inputs[i].index;

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        fn_body->graph.IsValidOutputTensor(node, idx),
        "Encountered while processing input ", i, " into function '", fn_name,
        "'");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(ValidateNonRefOutput(node, idx),
                                    "Encountered while processing input ", i,
                                    " into function '", fn_name, "'");

    input_tensors->emplace_back(node, idx);

    const auto& iter = input_nodes->find(node);
    if (iter == input_nodes->end()) {
      input_nodes->insert({node, {idx}});
    } else {
      auto& indices = iter->second;
      if (std::find(indices.begin(), indices.end(), idx) != indices.end()) {
        return InvalidArgument("TF_Output ", node->name(), ":", idx,
                               " appears more than once in the input list");
      }
      indices.push_back(idx);
    }
  }
  return Status::OK();
}

// Converts `noutputs` and `outputs` into `outputs_tensors` and does various
// checks while doing so.
Status ProcessOutputs(const TF_Graph* fn_body, const char* fn_name,
                      int noutputs, const TF_Output* outputs,
                      std::vector<OutputTensor>* output_tensors)
    EXCLUSIVE_LOCKS_REQUIRED(fn_body->mu) {
  output_tensors->reserve(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    Node* node = &outputs[i].oper->node;
    int idx = outputs[i].index;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        fn_body->graph.IsValidOutputTensor(node, idx),
        "Encountered while processing output ", i, " from function '", fn_name,
        "'");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(ValidateNonRefOutput(node, idx),
                                    "Encountered while creating function '",
                                    fn_name, "'");
    output_tensors->emplace_back(node, idx);
  }
  return Status::OK();
}

// Populates `body_nodes` with the nodes that will become function's body.
// Performs various checks.
Status ComputeBodyNodes(
    const TF_Graph* fn_body, const char* fn_name, int num_opers,
    const TF_Operation* const* opers,
    const std::unordered_map<const Node*, std::vector<int>>& input_nodes,
    std::vector<const Node*>* body_nodes)
    EXCLUSIVE_LOCKS_REQUIRED(fn_body->mu) {
  if (num_opers == -1) {
    for (const Node* node : fn_body->graph.op_nodes()) {
      const auto& iter = input_nodes.find(node);
      if (iter == input_nodes.end()) {
        // This node is not referenced in inputs. Add it to the body.
        body_nodes->push_back(node);
      } else {
        // This node is referenced in inputs. Currently, we place an
        // artificial restriction and require that when num_opers=-1, such
        // nodes must have a single output.
        if (node->num_outputs() != 1) {
          return InvalidArgument(
              "When `num_opers` is set to -1, nodes referenced in `inputs` "
              "must have a single output. Node ",
              node->name(), " has ", node->num_outputs(),
              " outputs. Encountered while creating function '", fn_name, "'");
        }
      }
    }
  } else {
    body_nodes->reserve(num_opers);
    for (int i = 0; i < num_opers; ++i) {
      const Node* node = &opers[i]->node;
      body_nodes->push_back(node);
    }
  }
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

using tensorflow::Node;
using tensorflow::string;

TF_Function* TF_GraphToFunction(const TF_Graph* fn_body, const char* fn_name,
                                unsigned char append_hash_to_fn_name,
                                int num_opers, const TF_Operation* const* opers,
                                int ninputs, const TF_Output* inputs,
                                int noutputs, const TF_Output* outputs,
                                const char* const* output_names,
                                const TF_FunctionOptions* opts,
                                const char* description, TF_Status* status) {
  tensorflow::mutex_lock l(*const_cast<tensorflow::mutex*>(&fn_body->mu));

  // Process inputs.
  std::vector<tensorflow::OutputTensor> input_tensors;
  std::unordered_map<const Node*, std::vector<int>> input_nodes;
  status->status = tensorflow::ProcessInputs(fn_body, fn_name, ninputs, inputs,
                                             &input_tensors, &input_nodes);
  if (!status->status.ok()) return nullptr;

  // Process outputs.
  std::vector<tensorflow::OutputTensor> output_tensors;
  status->status = tensorflow::ProcessOutputs(fn_body, fn_name, noutputs,
                                              outputs, &output_tensors);
  if (!status->status.ok()) return nullptr;

  // Process output names.
  std::vector<string> output_names_vec;
  if (output_names) {
    output_names_vec.reserve(noutputs);
    for (int i = 0; i < noutputs; ++i) {
      output_names_vec.push_back(string(output_names[i]));
    }
  }

  // Compute body nodes.
  std::vector<const Node*> body_nodes;
  status->status = tensorflow::ComputeBodyNodes(
      fn_body, fn_name, num_opers, opers, input_nodes, &body_nodes);
  if (!status->status.ok()) return nullptr;

  // Do the actual function creation.
  TF_Function* tf_function = new TF_Function();
  DCHECK(append_hash_to_fn_name <= 1);
  status->status = tensorflow::GraphToFunctionDef(
      fn_body->graph, fn_name, append_hash_to_fn_name != 0, body_nodes,
      input_tensors, output_tensors, output_names_vec, description,
      &tf_function->fdef);
  if (!status->status.ok()) {
    TF_DeleteFunction(tf_function);
    return nullptr;
  }
  return tf_function;
}

const char* TF_FunctionName(TF_Function* func) {
  return func->fdef.signature().name().c_str();
}

void TF_GraphCopyFunction(TF_Graph* g, const TF_Function* func,
                          const TF_Function* grad, TF_Status* status) {
  if (func == nullptr) {
    status->status = InvalidArgument(
        "'func' argument to TF_GraphCopyFunction cannot be null");
    return;
  }

  // TODO(iga): Add AddFunctionDef() and AddGradientDef() methods to graph
  // to avoid the extra copy here.
  tensorflow::FunctionDefLibrary fdef_lib;
  *fdef_lib.add_function() = func->fdef;
  if (grad) {
    *fdef_lib.add_function() = grad->fdef;
    tensorflow::GradientDef* gdef = fdef_lib.add_gradient();
    gdef->set_function_name(func->fdef.signature().name());
    gdef->set_gradient_func(grad->fdef.signature().name());
  }

  tensorflow::mutex_lock l(g->mu);
  status->status = g->graph.AddFunctionLibrary(fdef_lib);
}

int TF_GraphNumFunctions(TF_Graph* g) {
  tensorflow::mutex_lock l(g->mu);
  return g->graph.flib_def().num_functions();
}

int TF_GraphGetFunctions(TF_Graph* g, TF_Function** funcs, int max_func,
                         TF_Status* status) {
  tensorflow::FunctionDefLibrary lib;
  {
    tensorflow::mutex_lock l(g->mu);
    lib = g->graph.flib_def().ToProto();
  }
  const auto len = std::min(max_func, static_cast<int>(lib.function_size()));
  for (int i = 0; i < len; ++i) {
    TF_Function* func = new TF_Function();
    func->fdef = lib.function(i);
    funcs[i] = func;
  }
  status->status = tensorflow::Status::OK();
  return len;
}

void TF_FunctionToFunctionDef(TF_Function* func, TF_Buffer* output_func_def,
                              TF_Status* status) {
  status->status = MessageToBuffer(func->fdef, output_func_def);
}

TF_Function* TF_FunctionImportFunctionDef(const void* proto, size_t proto_len,
                                          TF_Status* status) {
  TF_Function* func = new TF_Function();
  if (!func->fdef.ParseFromArray(proto, proto_len)) {
    status->status = InvalidArgument(
        "Invalid FunctionDef given to TF_FunctionImportFunctionDef");
    TF_DeleteFunction(func);
    return nullptr;
  }
  status->status = tensorflow::Status::OK();
  return func;
}

void TF_FunctionSetAttrValueProto(TF_Function* func, const char* attr_name,
                                  const void* proto, size_t proto_len,
                                  TF_Status* status) {
  tensorflow::AttrValue attr_value;
  if (!attr_value.ParseFromArray(proto, proto_len)) {
    status->status = InvalidArgument(
        "Unparseable AttrValue proto passed to "
        "TF_FunctionSetAttrValueProto");
    return;
  }
  (*func->fdef.mutable_attr())[string(attr_name)] = attr_value;
  status->status = tensorflow::Status::OK();
}

void TF_FunctionGetAttrValueProto(TF_Function* func, const char* attr_name,
                                  TF_Buffer* output_attr_value,
                                  TF_Status* status) {
  const auto& it = func->fdef.attr().find(attr_name);
  if (it == func->fdef.attr().end()) {
    status->status =
        InvalidArgument("Function '", func->fdef.signature().name(),
                        "' has no attr named '", attr_name, "'.");
    return;
  }
  status->status = MessageToBuffer(it->second, output_attr_value);
}

void TF_DeleteFunction(TF_Function* func) { delete func; }
