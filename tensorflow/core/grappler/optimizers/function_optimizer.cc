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

#include "tensorflow/core/grappler/optimizers/function_optimizer.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/lower_case_op.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr const char* const kFuncAttr = FunctionLibraryDefinition::kFuncAttr;

// Do not specialize functions marked with '_nospecialize' attribute.
constexpr const char* const kNoSpecializeAttr = "_nospecialize";

// Mark functions that were created as a result of function specialization.
constexpr const char* const kGrapplerSpecializedFuncAttr =
    "_GrapplerSpecializedFunc";

// There are two ways of calling a Tensorflow function:
//
// 1. Direct function call: node.op() is the name of the function.
//
// 2. Indirect function call: the function name is passed through a node
//    attribute, and special Tensorflow kernels are responsible for calling the
//    function through the FunctionLibraryRuntime. Example: PartitionedCallOp.

// Check if func_node.op() matches the name in FunctionDef signature.
bool IsDirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  return func_node.op() == func.signature().name();
}

// Check if func_node has function attribute with a function name matching
// FunctionDef signature.
bool IsIndirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  if (!IsPartitionedCall(func_node) && !IsStatefulPartitionedCall(func_node)) {
    return false;
  }

  auto* func_attr = AttrSlice(func_node).Find(kFuncAttr);
  return func_attr != nullptr && func_attr->has_func() &&
         func_attr->func().name() == func.signature().name();
}

AttrSlice FunctionInstantiationAttributes(const FunctionDef& func,
                                          const NodeDef& func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    return AttrSlice(func_node);

  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* func_attr = AttrSlice(func_node).Find(kFuncAttr);
    return AttrSlice(&func_attr->func().attr());

  } else {
    LOG(WARNING) << "Can't resolve function instantiation attributes: "
                 << SummarizeNodeDef(func_node);
    return AttrSlice();
  }
}

// This is a fake device that should not be used for any op kernel execution,
// the only purpose of this device is to be passed as a part of DeviceSet to the
// Placer.
class FakeDevice : public Device {
 public:
  FakeDevice(Env* env, const string& device) : Device(env, attr(device)) {}
  explicit FakeDevice(const string& device) : FakeDevice(nullptr, device) {}
  absl::Status Sync() override { return absl::OkStatus(); }

 private:
  static DeviceAttributes attr(const string& device) {
    DeviceNameUtils::ParsedName parsed_name;
    bool parsed = DeviceNameUtils::ParseFullName(device, &parsed_name);
    DCHECK(parsed) << "Failed to parse full device name: " << device;

    DeviceAttributes attr;
    attr.set_name(device);
    attr.set_device_type(parsed_name.type);
    return attr;
  }
};

// -------------------------------------------------------------------------- //
// Function specialization.
//
// FunctionDef is somewhat similar to function template in C++, given all the
// type parameters (and attribute values) it generates a statically defined
// graph from the type parametrized "graph template" (function body).
//
// Function specialization instantiates a parametrized FunctionDef into a
// statically defined graph, and then converts it back to the fully defined
// FunctionDef (it doesn't have any unknown type parameters or attribute
// values, known as placeholders).
//
// Given the fully specified graph we can apply all the Grappler optimizers to
// it (see details in MetaOptimizer). Also we can push known constant inputs
// into the function body, and remove unused outputs/inputs.

bool MarkedNoSpecialize(const FunctionDef& fdef) {
  const auto attr = AttrSlice(&fdef.attr());
  bool nospecialize = false;
  return TryGetNodeAttr(attr, kNoSpecializeAttr, &nospecialize) && nospecialize;
}

// Specialized function instantiation type parameters, body parameters, and
// const inputs.
struct FunctionSpecializationSignature {
  // Currently we do not support functions with tensor lists as inputs or
  // outputs, so caller node input/output ports always match function
  // input/output arguments.
  using InputPort = int;
  using OutputPort = int;

  string func_name;
  bool is_in_fetch_set;
  absl::flat_hash_set<OutputPort> active_outputs;
  absl::flat_hash_map<string, DataType> type_parameters;
  absl::flat_hash_map<string, AttrValue> body_parameters;
  absl::flat_hash_map<InputPort, string> const_inputs;

  bool operator==(const FunctionSpecializationSignature& other) const {
    bool equals = func_name == other.func_name &&
                  is_in_fetch_set == other.is_in_fetch_set &&
                  active_outputs == other.active_outputs &&
                  type_parameters == other.type_parameters &&
                  const_inputs == other.const_inputs;

    if (!equals) return false;

    // Equality is not defined for AttrValue.
    if (body_parameters.size() != other.body_parameters.size()) return false;

    for (const auto& lhs : body_parameters) {
      auto it = other.body_parameters.find(lhs.first);
      if (it == other.body_parameters.end()) return false;
      if (!AreAttrValuesEqual(lhs.second, (*it).second,
                              /*allow_false_negatives=*/true)) {
        return false;
      }
    }

    return true;
  }

  template <typename H>
  friend H AbslHashValue(H h, const FunctionSpecializationSignature& s) {
    H base = H::combine(std::move(h), s.func_name, s.is_in_fetch_set);

    // First pre-compute hashes for all values in collections with
    // non-deterministic iteration order.
    std::vector<uint64> hashes;
    hashes.reserve(s.active_outputs.size()         //
                   + s.type_parameters.size() * 2  //
                   + s.body_parameters.size() * 2  //
                   + s.const_inputs.size() * 2);

    absl::c_transform(s.active_outputs, std::back_inserter(hashes),
                      hash<OutputPort>());

    using TypeParam = std::pair<const string, DataType>;
    absl::c_for_each(s.type_parameters, [&hashes](const TypeParam& type_param) {
      AttrValue attr_value;
      attr_value.set_type(type_param.second);
      hashes.push_back(Hash64(type_param.first));
      hashes.push_back(AttrValueHash(attr_value));
    });

    using BodyParam = std::pair<const string, AttrValue>;
    absl::c_for_each(s.body_parameters, [&hashes](const BodyParam& body_param) {
      hashes.push_back(Hash64(body_param.first));
      hashes.push_back(FastAttrValueHash(body_param.second));
    });

    using ConstInput = std::pair<const InputPort, string>;
    absl::c_for_each(s.const_inputs, [&hashes](const ConstInput& const_input) {
      hashes.push_back(hash<InputPort>()(const_input.first));
      hashes.push_back(Hash64(const_input.second));
    });

    // Combine all pre-computed hashes in a deterministic order.
    absl::c_sort(hashes);
    return H::combine_contiguous(std::move(base), hashes.data(), hashes.size());
  }
};

struct FunctionSpecialization {
  string specialized_func_name;
  // True if the function caller node is in GrapplerItem fetch set.
  bool is_in_fetch_set;
  // Names of the tensors that were pushed down into the function body.
  absl::flat_hash_set<string> const_inputs;
  // Control dependencies of pushed down const inputs have to be attached to
  // function caller node.
  absl::flat_hash_set<string> control_deps;
  // Output tensors (ports) that consumed by other nodes in the graph or in a
  // GrapplerItem fetch set.
  absl::flat_hash_set<int> active_outputs;
  // Mapping from original function output port to the output port of
  // specialized function. If function specialization changes the number of
  // function outputs it's required to update all node consumers.
  std::vector<std::pair<int, int>> output_mapping;
};

// Function optimizer context initialized once for each optimization pass, and
// it uses the latest available graph (for the first iteration it will be the
// GrapplerItem.graph, for next iterations it will be the output of previous
// function optimizer pass).
class FunctionOptimizerContext {
 public:
  explicit FunctionOptimizerContext(const GrapplerItem& item,
                                    RewriterConfig::Toggle opt_level,
                                    const GraphDef& graph)
      : item_(&item),
        opt_level_(opt_level),
        function_library_(OpRegistry::Global(), graph.library()),
        truly_const_nodes_(InferTrulyConstNodes(item, graph)),
        graph_view_(&graph) {}

  const GrapplerItem& item() const { return *item_; }

  const int graph_version() const { return item_->graph.versions().producer(); }

  RewriterConfig::Toggle opt_level() const { return opt_level_; }

  const FunctionLibraryDefinition& function_library() const {
    return function_library_;
  }
  FunctionLibraryDefinition& function_library() { return function_library_; }

  const absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>&
  tensor_mapping() const {
    return tensor_mapping_;
  }

  const GraphView& graph_view() const { return graph_view_; }

  bool IsFeedNode(const string& node_name) const {
    return absl::c_any_of(
        item_->feed, [&](const std::pair<std::string, Tensor>& feed) {
          return ParseTensorName(feed.first).node() == node_name;
        });
  }

  bool IsFetchNode(const string& node_name) const {
    return absl::c_any_of(item_->fetch, [&](const string& fetch) {
      return ParseTensorName(fetch).node() == node_name;
    });
  }

  bool IsTrulyConst(const string& name) const {
    return TrulyConstNode(name) != nullptr;
  }

  const NodeDef* TrulyConstNode(const string& name) const {
    return gtl::FindWithDefault(truly_const_nodes_, name, nullptr);
  }

  const FunctionSpecialization* FindFunctionSpecialization(
      const FunctionSpecializationSignature& sig) const {
    return gtl::FindOrNull(specialized_functions_, sig);
  }

  void AddSpecializedFunction(const FunctionSpecializationSignature& sig,
                              const FunctionSpecialization& specialized_func) {
    specialized_functions_.emplace(sig, specialized_func);
  }

  void AddTensorMapping(const SafeTensorId& from, const SafeTensorId& to) {
    DCHECK(from.index() != Graph::kControlSlot)
        << "Tensor mapping must be from regular tensor";
    DCHECK(to.index() != Graph::kControlSlot)
        << "Tensor mapping must be to regular tensor";

    auto inserted = tensor_mapping_.insert({from, to});
    DCHECK(inserted.second)
        << "Failed to insert duplicated tensor mapping: "
        << "from=" << from.ToString() << " to=" << to.ToString();
  }

  void AddTensorMapping(const string& func_node,
                        const FunctionSpecialization& specialized_func) {
    for (const auto& pair : specialized_func.output_mapping) {
      int from_idx = pair.first;
      int to_idx = pair.second;
      if (from_idx != to_idx) {
        SafeTensorId from_tensor(func_node, from_idx);
        SafeTensorId to_tensor(func_node, to_idx);
        AddTensorMapping(from_tensor, to_tensor);
      }
    }
  }

 private:
  static absl::flat_hash_map<string, const NodeDef*> InferTrulyConstNodes(
      const GrapplerItem& item, const GraphDef& graph) {
    absl::flat_hash_set<absl::string_view> feed_nodes;
    for (const auto& feed : item.feed) {
      feed_nodes.insert(feed.first);
    }

    absl::flat_hash_map<string, const NodeDef*> const_nodes;
    for (const NodeDef& node : graph.node()) {
      if (IsConstant(node) && !feed_nodes.contains(node.name())) {
        const_nodes[node.name()] = &node;
      }
    }

    return const_nodes;
  }

  const GrapplerItem* item_;  // must outlive this object
  RewriterConfig::Toggle opt_level_;

  // Function library constructed from current graph.
  FunctionLibraryDefinition function_library_;

  // Nodes that are Const and not in feed.
  absl::flat_hash_map<string, const NodeDef*> truly_const_nodes_;
  // Specialized functions.
  absl::flat_hash_map<FunctionSpecializationSignature,
                      const FunctionSpecialization>
      specialized_functions_;

  // After function specialization, the optimized graph might be in invalid
  // state, nodes can read from output index that is no longer valid after
  // unused outputs pruning.
  //
  // Tensor mapping that has to be applied to the graph after all functions
  // optimizations (invalidated tensor id -> optimized graph tensor id).
  absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>
      tensor_mapping_;

  // Use graph view to find active outputs of the function caller nodes.
  GraphView graph_view_;

  FunctionOptimizerContext(const FunctionOptimizerContext&) = delete;
  void operator=(const FunctionOptimizerContext&) = delete;
};

// Returns a pointer to the called function definition iff the given node is
// indeed a function call. Otherwise returns nullptr.
const FunctionDef* FindFunctionCall(const FunctionOptimizerContext& ctx,
                                    const NodeDef& node) {
  // Check if a node does indirect function call via PartitionedCallOp.
  if (IsPartitionedCall(node) || IsStatefulPartitionedCall(node)) {
    const AttrValue* func_attr = AttrSlice(node).Find("f");
    return (func_attr != nullptr && func_attr->has_func())
               ? ctx.function_library().Find(func_attr->func().name())
               : nullptr;
  }

  // Check if the function op itself is a function name.
  return ctx.function_library().Find(node.op());
}

absl::flat_hash_set<int> GetActiveOutputs(const NodeDef& node,
                                          const FunctionOptimizerContext& ctx,
                                          int size_hint = 0) {
  absl::flat_hash_set<int> active_outputs;
  active_outputs.reserve(static_cast<size_t>(size_hint));

  // 1. Output can be consumed by the other graph node.
  const auto node_fanout_edges =
      ctx.graph_view().GetFanoutEdges(node, /*include_controlled_edges=*/false);
  for (const GraphView::Edge& edge : node_fanout_edges) {
    active_outputs.insert(edge.src.port_id);
  }

  // 2. Or it can be in a fetch set.
  for (const string& fetch : ctx.item().fetch) {
    TensorId fetch_tensor = ParseTensorName(fetch);
    if (fetch_tensor.node() == node.name()) {
      active_outputs.insert(fetch_tensor.index());
    }
  }

  return active_outputs;
}

bool HasTrulyConstInputs(const NodeDef& node,
                         const FunctionOptimizerContext& ctx) {
  const auto is_truly_const = [&ctx](const string& input) {
    return ctx.IsTrulyConst(NodeName(input));
  };
  return absl::c_any_of(node.input(), is_truly_const);
}

bool HasUnusedOutputs(const NodeDef& func_node, const FunctionDef& func,
                      const FunctionOptimizerContext& ctx) {
  // Functions with tensor list outputs are not supported right now, so the
  // number of output args is the same as number of possible function caller
  // node outputs.
  int num_outputs = func.signature().output_arg_size();
  const absl::flat_hash_set<int> active_outputs =
      GetActiveOutputs(func_node, ctx, /*size_hind*/ num_outputs);
  int active_outputs_size = active_outputs.size();
  return active_outputs_size != num_outputs;
}

// Return pruned FunctionDefLibrary with functions that are reachable from
// the optimized graph.
FunctionDefLibrary PruneFunctionLibrary(const FunctionLibraryDefinition& flib,
                                        const GraphDef& optimized_graph) {
  FunctionLibraryDefinition pruned_flib =
      flib.ReachableDefinitions(optimized_graph);

  int pruned_functions = static_cast<int>(pruned_flib.num_functions()) -
                         static_cast<int>(flib.num_functions());

  VLOG(3) << "Pruned function library: " << pruned_flib.num_functions()
          << " functions (" << pruned_functions << ")";

  return pruned_flib.ToProto();
}

// Push all constant inputs of an instantiating node into the function body.
absl::Status PushDownConstInputs(const NodeDef& func_node,
                                 const FunctionOptimizerContext& ctx,
                                 GrapplerFunctionItem* item,
                                 absl::flat_hash_set<string>* const_inputs,
                                 absl::flat_hash_set<string>* control_deps) {
  // Record node control dependencies in the control_deps set.
  const auto record_control_deps = [&](const NodeDef* const_input) {
    for (int i = const_input->input_size() - 1; i >= 0; --i) {
      const string& input = const_input->input(i);
      if (IsControlInput(input))
        control_deps->insert(input);
      else
        break;
    }
  };

  for (int i = func_node.input_size() - 1; i >= 0; --i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) continue;

    const string node_name = NodeName(input);
    if (ctx.IsTrulyConst(node_name)) {
      VLOG(3) << "Push const into function body: input=" << input;
      const auto* const_input = CHECK_NOTNULL(ctx.TrulyConstNode(node_name));
      const_inputs->insert(input);
      record_control_deps(const_input);
      TF_RETURN_IF_ERROR(ReplaceInputWithConst(*const_input, i, item));
    }
  }

  return absl::OkStatus();
}

// Remove inputs that were pushed into the function body, and attach their
// control dependencies to the function caller node.
void RemovePushedDownConstInputs(const FunctionSpecialization& specialization,
                                 NodeDef* specialized_func_node) {
  // Nothing to do if it was no const inputs to the function node.
  if (specialization.const_inputs.empty()) return;

  // Keep only non-const inputs.
  std::vector<string> keep_inputs;
  const auto& inputs = specialized_func_node->input();
  absl::c_copy_if(inputs, std::back_inserter(keep_inputs),
                  [&](const string& input) {
                    return !specialization.const_inputs.contains(input);
                  });

  specialized_func_node->clear_input();
  for (const auto& keep : keep_inputs) specialized_func_node->add_input(keep);

  // Attach control dependencies of pushed down const input to the caller node.
  if (!specialization.control_deps.empty()) {
    absl::flat_hash_set<string> existing_control_deps;

    for (const string& input : keep_inputs) {
      existing_control_deps.insert(AsControlDependency(NodeName(input)));
    }

    for (const string& ctrl : specialization.control_deps) {
      if (!existing_control_deps.contains(ctrl)) {
        VLOG(3) << "Forward control dependency: input=" << ctrl;
        specialized_func_node->add_input(ctrl);
      }
    }
  }
}

// Remove Tin type parameters for pushed down const inputs.
void RemovePushedDownConstInputTypes(
    const FunctionSpecialization& specialization, const NodeDef& func_node,
    NodeDef* specialized_func_node) {
  // Nothing to do if it was no const inputs to the function node.
  if (specialization.const_inputs.empty()) return;

  // Make sure that original function caller has Tin attribute.
  const AttrValue* tin = AttrSlice(func_node).Find("Tin");
  if (tin == nullptr || !tin->has_list()) return;

  // Clear input types for the specialized node.
  auto* attr = specialized_func_node->mutable_attr();
  (*attr)["Tin"].mutable_list()->clear_type();

  // Keep types of non-const inputs.
  for (int i = 0; i < func_node.input_size(); ++i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) break;

    if (!specialization.const_inputs.contains(input)) {
      DataType dt = tin->list().type(i);
      (*attr)["Tin"].mutable_list()->add_type(dt);
    }
  }
}

// Remove Tout type parameters for pruned function outputs.
void RemoveUnusedOutputsTypes(const FunctionSpecialization& specialization,
                              const NodeDef& func_node,
                              NodeDef* specialized_func_node) {
  // Make sure that original function caller has Tout attribute.
  const AttrValue* tout = AttrSlice(func_node).Find("Tout");
  if (tout == nullptr || !tout->has_list()) return;

  // Nothing to do if all outputs are active.
  int specialization_active_outputs_size = specialization.active_outputs.size();
  if (specialization_active_outputs_size == tout->list().type_size()) return;

  // Clear input types for the specialized node.
  auto* attr = specialized_func_node->mutable_attr();
  (*attr)["Tout"].mutable_list()->clear_type();

  // Keep output types of active outputs only.
  for (int i = 0; i < tout->list().type_size(); ++i) {
    if (specialization.active_outputs.contains(i)) {
      DataType dt = tout->list().type(i);
      (*attr)["Tout"].mutable_list()->add_type(dt);
    }
  }
}

absl::Status UpdateSpecializedFunctionCallSite(
    const FunctionDef& func, const NodeDef& func_node,
    const string& specialized_func_name, NodeDef* specialized_func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    specialized_func_node->set_op(specialized_func_name);

  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* attr = specialized_func_node->mutable_attr();
    (*attr)[kFuncAttr].mutable_func()->set_name(specialized_func_name);

  } else {
    return absl::InvalidArgumentError("Unknown function call site");
  }

  return absl::OkStatus();
}

// Update a graph node created from the original function caller node, to the
// function specialization. Function specialization might change the number of
// inputs and outputs, so we have to make sure that graph node is updated
// accordingly.
absl::Status UpdateSpecializedFunctionNode(
    const FunctionDef& func, const NodeDef& func_node,
    const FunctionSpecialization& specialization,
    NodeDef* specialized_func_node) {
  // Function called indirectly via custom kernel (e.g. PartitionedCallOp).
  bool is_indirect_call = IsIndirectFunctionCall(func, func_node);

  // 1. Call the specialized function instead of original one.
  TF_RETURN_IF_ERROR(UpdateSpecializedFunctionCallSite(
      func, func_node, specialization.specialized_func_name,
      specialized_func_node));

  // 2. Remove inputs corresponding to the pushed down consts.
  RemovePushedDownConstInputs(specialization, specialized_func_node);

  // NOTE: PartitionedCallOp has `Tin` and `Tout` attributes for input/output
  // types, that must be in sync with updated function signature.

  // 3. Update input types for the indirect function calls.
  if (is_indirect_call) {
    RemovePushedDownConstInputTypes(specialization, func_node,
                                    specialized_func_node);
  }

  // 4. Update output types for the indirect function call. It's unsafe to
  // change the number of outputs for the fetch nodes, so we just skip them.
  if (is_indirect_call && !specialization.is_in_fetch_set) {
    RemoveUnusedOutputsTypes(specialization, func_node, specialized_func_node);
  }

  // 5. Remove custom gradient annotation.
  specialized_func_node->mutable_attr()->erase("_gradient_op_type");

  return absl::OkStatus();
}

absl::Status InitializeFunctionSpecializationSignature(
    const NodeDef& func_node, const FunctionDef& func,
    const AttrSlice& func_instantiation_attr,
    const FunctionOptimizerContext& ctx, FunctionSpecializationSignature* sig) {
  DCHECK(sig->const_inputs.empty());
  DCHECK(sig->active_outputs.empty());

  sig->func_name = func.signature().name();
  sig->is_in_fetch_set = ctx.IsFetchNode(func_node.name());
  sig->active_outputs = GetActiveOutputs(func_node, ctx);

  TF_RETURN_IF_ERROR(InstantiationTypeParameters(func, func_instantiation_attr,
                                                 &sig->type_parameters));
  TF_RETURN_IF_ERROR(InstantiationBodyParameters(func, func_instantiation_attr,
                                                 &sig->body_parameters));

  for (int i = 0; i < func_node.input_size(); ++i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) break;
    if (ctx.IsTrulyConst(input)) {
      sig->const_inputs.emplace(i, input);
    }
  }

  return absl::OkStatus();
}

// Create a name for the function specialization. The name of the function, name
// of the node instantiating it, and a Grappler item id should generate unique
// function name. Meta optimizer might create multiple Grappler items for the
// same graph when optimizing functions, but it's guaranteed that they all will
// have unique ids.
string SpecializedFunctionName(const FunctionOptimizerContext& ctx,
                               const FunctionDef& func,
                               const NodeDef& func_node) {
  return absl::Substitute(
      "$0_specialized_for_$1_at_$2", func.signature().name(),
      absl::StrReplaceAll(func_node.name(), {{"/", "_"}}), ctx.item().id);
}

absl::Status SpecializeFunction(const NodeDef& func_node,
                                const FunctionDef& func,
                                FunctionOptimizerContext* ctx,
                                GraphDef* optimized_graph) {
  VLOG(2) << "Specialize function call: " << SummarizeNodeDef(func_node);

  const AttrSlice func_instantiation_attr =
      FunctionInstantiationAttributes(func, func_node);

  FunctionSpecializationSignature signature;
  TF_RETURN_IF_ERROR(InitializeFunctionSpecializationSignature(
      func_node, func, func_instantiation_attr, *ctx, &signature));

  // Check if function was already specialized for identical context.
  const FunctionSpecialization* already_specialized =
      ctx->FindFunctionSpecialization(signature);

  if (already_specialized) {
    VLOG(2) << "Function was already specialized in identical context: "
               "specialized_name="
            << already_specialized->specialized_func_name;

    // Add a function call node for the specialized function.
    NodeDef* specialized_func_node = optimized_graph->add_node();
    *specialized_func_node = func_node;

    TF_RETURN_IF_ERROR(UpdateSpecializedFunctionNode(
        func, func_node, *already_specialized, specialized_func_node));

    ctx->AddTensorMapping(specialized_func_node->name(), *already_specialized);

    return absl::OkStatus();
  }

  // Add a new specialized function definition to the library.
  const auto& flib = ctx->function_library();

  // Make a GrapplerFunctionItem and convert it back to FunctionDef after
  // pushing all constant inputs into the function body.
  GrapplerFunctionItem item;
  TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
      func, func_instantiation_attr, flib, ctx->graph_version(), &item));

  // Push const inputs into the function body, and keep track of their control
  // dependencies.
  absl::flat_hash_set<string> const_inputs;
  absl::flat_hash_set<string> control_deps;
  TF_RETURN_IF_ERROR(PushDownConstInputs(func_node, *ctx, &item, &const_inputs,
                                         &control_deps));

  // Remove function outputs that do not have any consumers. We can't safely
  // update outputs for the fetch nodes, so we just skip them.
  std::vector<std::pair<int, int>> output_mapping;
  if (!signature.is_in_fetch_set) {
    int num_func_outputs = item.output_size();

    absl::flat_hash_set<int> remove;
    for (int i = 0; i < num_func_outputs; ++i) {
      if (!signature.active_outputs.count(i)) remove.insert(i);
    }

    TF_RETURN_IF_ERROR(RemoveFunctionOutputs(remove, &item, &output_mapping));
  }

  // TODO(ezhulenev): Push down known input shapes.
  FunctionDef specialized_func;
  TF_RETURN_IF_ERROR(MakeFunctionDef(item, flib, &specialized_func));

  // Find a name for specialized function.
  const string specialized_func_name =
      SpecializedFunctionName(*ctx, func, func_node);
  if (flib.Contains(specialized_func_name)) {
    // NOTE(ezhulenev): This should never happen. If it happens, it's a sign of
    // a serious internal error, that must be investigated.
    return absl::InternalError("Created duplicate function specialization");
  }

  specialized_func.mutable_signature()->set_name(specialized_func_name);
  auto* specialized_attr = specialized_func.mutable_attr();
  (*specialized_attr)[kGrapplerSpecializedFuncAttr].set_b(true);

  // Add specialized function to the library.
  TF_RETURN_IF_ERROR(ctx->function_library().AddFunctionDef(specialized_func));

  // Add a function call node for the specialized function.
  NodeDef* specialized_func_node = optimized_graph->add_node();
  *specialized_func_node = func_node;

  FunctionSpecialization func_specialization = {
      specialized_func_name, signature.is_in_fetch_set, const_inputs,
      control_deps,          signature.active_outputs,  output_mapping};

  TF_RETURN_IF_ERROR(UpdateSpecializedFunctionNode(
      func, func_node, func_specialization, specialized_func_node));

  ctx->AddSpecializedFunction(signature, func_specialization);
  ctx->AddTensorMapping(specialized_func_node->name(), func_specialization);

  return absl::OkStatus();
}

// -------------------------------------------------------------------------- //
// Inline function calls into a graph using function inlining implementation
// from common_runtime:
//
// 1) Convert GraphDef to Graph.
// 2) Inline function calls.
// 3) Convert Graph back to the GraphDef.

constexpr const char* const kLowerUsingSwitchMergeAttr =
    LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr;
constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsPass::kLowerAsMultiDeviceFunctionAttr;

using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
using OutputControlSource = InlineFunctionBodyOptions::OutputControlSource;

// Checks if boolean attribute is defined and its value is 'true'.
bool CheckBoolAttr(const Node* n, absl::string_view attr_name) {
  bool match;
  bool found = TryGetNodeAttr(n->attrs(), attr_name, &match);
  return found && match;
}

// Checks if string attribute is defined and it's not empty.
bool CheckStringAttr(const Node* n, absl::string_view attr_name) {
  const string& value = GetNodeAttrString(n->attrs(), attr_name);
  return !value.empty();
}

bool LowerUsingSwitchMergeIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerUsingSwitchMergeAttr);
}

bool LowerAsMultiDeviceFunctionIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerAsMultiDeviceFunctionAttr);
}

bool MarkedForXlaCompilation(const NodeDef& n) {
  auto is_enabled = [&](std::string attr_name) -> bool {
    auto it = n.attr().find(attr_name);
    return it != n.attr().end() && (!it->second.s().empty() || it->second.b());
  };
  return is_enabled("_xla_compile_id") || is_enabled("_tpu_replicate") ||
         is_enabled(kXlaMustCompileAttr);
}

const bool IsExemptFromSideEffectsExecutionValidation(const string& op) {
  static const auto* exemption = new absl::flat_hash_set<string>(
      {// LINT.IfChange
       // Op types that should not run in program order, e.g. because they need
       // to run asynchronously to avoid deadlock.
       "CollectiveGather", "CollectiveReduce", "CollectiveBcastSend",
       "CollectiveBcastRecv", "CollectiveBcastSendV2", "CollectiveBcastRecvV2",
       "NcclAllReduce", "Send", "Recv", "CollectiveAssignGroupsV2",
       "CollectiveInitializeCommunicator",

       // Legacy random ops.
       // See details in tensorflow/python/framework/auto_control_deps.py.
       "RandomUniform", "RandomUniformInt", "RandomStandardNormal",
       "ParameterizedTruncatedNormal", "TruncatedNormal", "RandomShuffle",
       "Multinomial", "RandomGamma", "RandomGammaGrad", "RandomPoisson",
       "RandomPoissonV2",

       // ReadVariableOp marked as stateful because it consumes DT_RESOURCE,
       // but it can't generate any observable side-effect.
       "ReadVariableOp",

       // CudnnRNN ops are stateful but they can't generate any observable
       // side-effect.
       "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNV2", "CudnnRNNV3",
       "CudnnRNNBackpropV2", "CudnnRNNBackpropV3",

       // TPUEmbedding EnqueueOps are stateful but this is only between ops with
       // the same device_ordinal on the same host.
       "EnqueueTPUEmbeddingSparseBatch", "EnqueueTPUEmbeddingIntegerBatch",
       "EnqueueTPUEmbeddingSparseTensorBatch",
       "EnqueueTPUEmbeddingRaggedTensorBatch",
       "EnqueueTPUEmbeddingArbitraryTensorBatch",
       "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch",

       // SaveV2 and RestoreV2 should be allowed to operate in parallel on
       // multiple hosts.
       "SaveV2", "RestoreV2",

       // InfeedEnqueue are stateful but should not be serialized for the
       // input pipeline
       "InfeedEnqueue", "InfeedEnqueueTuple"});
  // LINT.ThenChange(//tensorflow/python/framework/auto_control_deps.py)
  return exemption->contains(op);
}

// Validates that all side effects inside function body will be executed after
// function inlining. We do it by looking for a path from stateful ops, to one
// of the output control sources.
//
// When function executed via FunctionLibraryRuntime we do not have to check
// this, because `PruneFunctionBody` has special pruning rules for stateful ops.
absl::Status ValidateSideEffectsExecution(
    const FunctionBody& fbody, OutputControlSource output_control_source,
    bool has_outgoing_control_edges,
    bool validate_outgoing_control_edge = true) {
  // Find all nodes that can produce side effects in the function body graph. We
  // use 'is_stateful()' bit as an approximation of "has side effects" property.
  std::vector<const Node*> fbody_side_effects;
  absl::c_copy_if(
      fbody.graph->nodes(), std::back_inserter(fbody_side_effects),
      [](const Node* n) {
        return n->op_def().is_stateful() && !n->IsArg() && !n->IsRetval() &&
               !IsExemptFromSideEffectsExecutionValidation(n->type_string());
      });

  // When graph executed in TF-2.0 context with automatic control dependencies
  // tracking, absence of outgoing control edge indicates that no one is
  // interested in observing side effects, so it is safe to inline the function
  // body, even if some side-effects will not be executed.
  if (!fbody_side_effects.empty() && !has_outgoing_control_edges) {
    const string error_message =
        "Can't guarantee execution of function side-effects after inlining. "
        "Function call node has no outgoing control edges.";
    if (validate_outgoing_control_edge) {
      return absl::InternalError(error_message);
    } else {
      VLOG(3) << error_message;
    }
  }

  // Find all nodes in the function body that will be used as control sources.
  absl::flat_hash_set<const Node*> control_sources;
  if (output_control_source == OutputControlSource::kDataOutputs) {
    control_sources = {fbody.ret_nodes.begin(), fbody.ret_nodes.end()};
  } else if (output_control_source == OutputControlSource::kControlOutputs) {
    control_sources = {fbody.control_ret_nodes.begin(),
                       fbody.control_ret_nodes.end()};
  }

  for (const Node* side_effect : fbody_side_effects) {
    VLOG(4) << "Check that node " << side_effect->name()
            << " will execute after inlining.";
    bool will_execute = false;

    const auto is_control_source = [&](const Node* n) -> void {
      const auto it = control_sources.find(n);
      if (it != control_sources.end()) {
        VLOG(4) << "Found a path to control source: " << side_effect->name()
                << " ---> " << (*it)->name();
        will_execute = true;
      }
    };

    DFSFrom(*fbody.graph, {side_effect}, /*enter=*/is_control_source,
            /*leave=*/{}, NodeComparatorName{});

    if (!will_execute) {
      return absl::InternalError(absl::StrCat(
          "Can't guarantee execution of a side-effectful node, that is not "
          "reachable from function control source. Function body node: ",
          SummarizeNode(*side_effect)));
    }
  }

  return absl::OkStatus();
}

// Validates that no dead tensor can reach function output.
absl::Status ValidateNoDeadOutputs(const FunctionLibraryDefinition& flib_def,
                                   const FunctionBody& fbody) {
  absl::flat_hash_set<const Node*> output_nodes = {fbody.ret_nodes.begin(),
                                                   fbody.ret_nodes.end()};

  // Find all nodes that can produce dead tensors.
  std::vector<const Node*> dead_tensor_sources;
  for (const Node* n : fbody.graph->nodes()) {
    if (n->IsSwitch()) {
      VLOG(4) << "Add dead tensors source. Switch node: " << n->name();
      dead_tensor_sources.push_back(n);
      continue;
    }

    // Native function call can also produce dead tensors if the function body
    // has mergeless switches.
    const FunctionDef* fdef = flib_def.Find(n->type_string());
    if (fdef != nullptr) {
      std::unique_ptr<FunctionBody> nested_fbody;

      NameAttrList func;
      TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(n->def(), &func));
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                                 &flib_def, &nested_fbody));

      if (!ValidateNoDeadOutputs(flib_def, *nested_fbody).ok()) {
        VLOG(4) << "Add dead tensors source. Function call: " << func.name()
                << " node=" << n->name();
        dead_tensor_sources.push_back(n);
      }
    }
  }

  for (const Node* dead_tensor_source : dead_tensor_sources) {
    bool has_dead_output = false;

    const auto is_output_node = [&](const Node* n) -> void {
      const auto it = output_nodes.find(n);
      if (it != output_nodes.end()) {
        VLOG(4) << "Found a path to output node from dead tensor source: "
                << dead_tensor_source->name() << " ---> " << (*it)->name();
        has_dead_output = true;
      }
    };

    // Stop DFS traversal at a Merge node or if already found a dead output.
    const auto stop_traversal = [&has_dead_output](const Edge& edge) -> bool {
      return !edge.src()->IsMerge() || has_dead_output;
    };

    DFSFrom(*fbody.graph, {dead_tensor_source}, /*enter=*/is_output_node,
            /*leave=*/{}, NodeComparatorName{},
            /*edge_filter=*/stop_traversal);

    if (has_dead_output) {
      return absl::InternalError(absl::StrCat(
          "Can't inline a function with dead outputs. Dead tensor source: ",
          SummarizeNode(*dead_tensor_source)));
    }
  }

  return absl::OkStatus();
}

// Makes an instance of FunctionBody for inlining from a Node.
absl::Status MakeFunctionBodyForInlining(
    const Node& node, const FunctionLibraryDefinition& flib_def,
    std::unique_ptr<FunctionBody>* fbody) {
  VLOG(3) << "Make function body for inlining: " << SummarizeNode(node);

  // Finds a FunctionDef in a library and verifies that it exists.
  const auto find_fdef = [&flib_def, &node](
                             const string& name,
                             const FunctionDef** fdef) -> absl::Status {
    if ((*fdef = flib_def.Find(name)) == nullptr) {
      return absl::InternalError(absl::StrCat(
          "Was not able to find a function definition (name=", name,
          ") for a function call: ", SummarizeNode(node)));
    }
    return absl::OkStatus();
  };

  // SymbolicGradient is a special "function call" op, which has been
  // deprecated for a while, but we still support for compatibility reasons.
  if (node.type_string() == FunctionLibraryDefinition::kGradientOp) {
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), kFuncAttr, &func));

    const string grad = flib_def.FindGradient(func.name());

    if (!grad.empty()) {
      // Function has a custom gradient registered in a library.
      const FunctionDef* grad_fdef;
      TF_RETURN_IF_ERROR(find_fdef(grad, &grad_fdef));

      VLOG(4) << "Instantiate a custom SymbolicGradient: gradient=" << grad
              << " (function=" << func.name() << ")";
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          *grad_fdef, AttrSlice(&func.attr()), &flib_def, fbody));

    } else if (flib_def.Find(func.name()) == nullptr) {
      // Function is not really a function, but a primitive op.
      gradient::Creator creator;
      TF_RETURN_IF_ERROR(gradient::GetOpGradientCreator(func.name(), &creator));
      if (creator == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("No gradient is defined for ", func.name()));
      }
      FunctionDef grad_fdef;
      TF_RETURN_IF_ERROR(creator(AttrSlice(&func.attr()), &grad_fdef));

      VLOG(4) << "Instantiate a SymbolicGradient for a primitive op: "
              << func.name();
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          grad_fdef, AttrSlice(&func.attr()), &flib_def, fbody));

    } else {
      // Build a gradient graph from the function body.
      const FunctionDef* fdef;
      TF_RETURN_IF_ERROR(find_fdef(func.name(), &fdef));

      VLOG(4) << "Instantiate a SymbolicGradient for a function: "
              << func.name();
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                                 &flib_def, fbody));
      *fbody = SymbolicGradient(**fbody);
    }

  } else {
    NameAttrList func;
    TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node.def(), &func));
    const FunctionDef* fdef;
    TF_RETURN_IF_ERROR(find_fdef(func.name(), &fdef));

    VLOG(4) << "Instantiate a function call: function=" << func.name();
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                               &flib_def, fbody));
  }

  return absl::OkStatus();
}

// Adds a control edges from each data input to the 'caller' to enforce strict
// inputs semantics (all inputs are ready and alive). This is required when:
//
//  1) The function takes resources as inputs, and it doesn't have incoming
//     control edges. In Tensorflow v2 context (eager mode) this should never
//     happen, because automatic control dependencies tracking will add a
//     control edge from the last op touching the resource. However such graphs
//     might be produced by legacy v1 code without automatic dependency
//     tracking. In this case strict function call semantics is required for
//     enforcing side effects execution order.
//
//  2) One of the inputs is consuming Enter[is_constant=true] node, in which
//     case it will be always alive, and potentially can lead to partial
//     function execution after the last loop execution.
//
// Both of these cases would be considered illegal by construction in Tensorflow
// V2, however we have to guarantee that graphs constructed with Tensorflow V1
// will produce correct results.
void AddStrictInputSemantics(Node* caller, Graph* g) {
  absl::flat_hash_set<const Node*> existing_control_sources;
  for (const Edge* edge : caller->in_edges()) {
    if (edge->IsControlEdge()) {
      existing_control_sources.insert(edge->src());
    }
  }

  const bool has_incoming_control_edges = !existing_control_sources.empty();

  const bool has_resource_input =
      absl::c_any_of(caller->input_types(),
                     [](const DataType dtype) { return dtype == DT_RESOURCE; });

  const bool has_constant_enter_input =
      absl::c_any_of(caller->in_edges(), [](const Edge* edge) {
        Node* src = edge->src();
        return src->IsEnter() && CheckBoolAttr(src, "is_constant");
      });

  const bool requires_strict_semantics =
      (!has_incoming_control_edges && has_resource_input) ||  // Case #1
      (has_constant_enter_input);                             // Case #2
  if (!requires_strict_semantics) return;

  std::set<const Node*> data_inputs;
  for (const Edge* edge : caller->in_edges()) {
    if (!edge->IsControlEdge() &&
        !existing_control_sources.contains(edge->src())) {
      data_inputs.insert(edge->src());
    }
  }

  VLOG(3) << "Add control edges from all data inputs to enforce strict "
             "semantics with regard to function inputs";

  // Do not add control edges from placeholders, because it will prevent
  // pruning, and they can't produce any side effects anyway.
  const auto is_placeholder = [](const Node* node) -> bool {
    return node->type_string() == "Placeholder";
  };

  for (const Node* node : data_inputs) {
    if (is_placeholder(node)) continue;
    g->AddControlEdge(g->FindNodeId(node->id()), caller,
                      /*allow_duplicates=*/true);
  }
}

// Adds a control edge from a frame node if the 'caller' is executing inside a
// While loop (see control_flow.h for the 'frame' node explanation).
void AddFrameForwardingControlEdge(const std::vector<ControlFlowInfo>& info,
                                   Node* caller, Graph* g) {
  // All nodes added to the graph by v2 control flow lowering and function
  // inlining are guaranteed to have control edges to nested function calls.
  int info_size = info.size();
  if (caller->id() >= info_size) return;

  // Check if a lowered node is executing inside a while loop.
  const Node* frame = info[caller->id()].frame;
  const bool is_in_while_loop = frame->id() != Graph::kSourceId;
  if (!is_in_while_loop) return;

  // Check if a node already has an incoming control edge. All incoming edges
  // must be from the same execution frame (executor.cc invariant), so if we
  // already have an incoming control edge, it's guaranteed that it will "carry"
  // the same frame as all regular inputs.
  const bool has_incoming_control_edges =
      absl::c_any_of(caller->in_edges(),
                     [](const Edge* edge) { return edge->IsControlEdge(); });
  if (has_incoming_control_edges) return;

  VLOG(3) << "Add a frame forwarding control edge: from=" << frame->name()
          << " to=" << caller->name();
  Node* enter = g->FindNodeId(frame->id());
  bool is_constant_enter = enter->attrs().Find("is_constant")->b();
  if (is_constant_enter) {
    // Enter[is_constant=true] is always alive. So we directly add a control
    // edge from that.
    g->AddControlEdge(enter, caller);
  } else {
    // Enter[is_constant=false] activates nodes only in 0th iteration so we
    // add an edge from the Merge node which is activated in every iteration.
    // A non-constant Enter node must have an edge to a Merge node.
    auto it = absl::c_find_if(enter->out_edges(), [](const Edge* e) {
      return !e->IsControlEdge() && e->dst()->IsMerge();
    });
    if (it != enter->out_edges().end()) {
      g->AddControlEdge((*it)->dst(), caller);
    } else {
      LOG(WARNING) << "Enter[is_constant=false] node: " << enter->name()
                   << " does not have an outgoing edge to a Merge.";
    }
  }
}

// Inlines all function calls that are safe for inlining into the main graph.
// Also lowers control flow V2 ops (functional If/While) into the V1 low level
// ops (Switch/Merge/...).
//
// Runs a placer after inlining, to keep all nodes in a graph placed.
absl::Status InlineFunctionCalls(const GrapplerItem& item,
                                 const RewriterConfig::Toggle opt_level,
                                 const bool lower_control_flow,
                                 GraphDef* output_graph) {
  bool is_aggressive = opt_level == RewriterConfig::AGGRESSIVE;
  VLOG(2) << "Inline function calls: grappler_item_id=" << item.id
          << " (aggressive_mode=" << is_aggressive << ")";

  FunctionLibraryDefinition flib_def =
      FunctionLibraryDefinition(OpRegistry::Global(), item.graph.library());
  std::unique_ptr<Graph> graph = std::make_unique<Graph>(flib_def);

  GraphConstructorOptions graph_constructor_options;
  graph_constructor_options.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(graph_constructor_options,
                                            item.graph, graph.get()));

  using NodeNames = absl::flat_hash_set<absl::string_view>;
  NodeNames fetch_nodes;
  fetch_nodes.reserve(item.fetch.size());
  for (const string& fetch : item.fetch) {
    fetch_nodes.insert(ParseTensorName(fetch).node());
  }
  NodeNames keep_nodes(item.keep_ops.begin(), item.keep_ops.end());
  if (item.save_op.size() > 0) {
    keep_nodes.insert(item.save_op);
  }
  if (item.restore_op.size() > 0) {
    keep_nodes.insert(item.restore_op);
  }

  std::vector<string> inlined_function_names;

  // Do not inline function call nodes that are part of a feed set.
  NodeNames feed_nodes;
  feed_nodes.reserve(item.feed.size());
  for (const std::pair<std::string, Tensor>& feed : item.feed) {
    feed_nodes.insert(ParseTensorName(feed.first).node());
  }

  // If a function call is inside a While loop, it must have an incoming control
  // edge, because it will be used to pass execution frame into the function
  // body. All nodes without inputs in the function body (e.g. Const and NoOp)
  // will be added an extra control edge from the 'input_control_node'.
  std::vector<ControlFlowInfo> control_flow_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph.get(), &control_flow_info));

  // Function inlining always adds new nodes to the end of the list, so we keep
  // iterating until we are out of nodes.
  for (int i = 2; i < graph->num_node_ids(); ++i) {
    Node* n = graph->FindNodeId(i);
    if (n == nullptr) continue;  // deleted node

    // Special case for lowering functional control flow ops. We do not rely on
    // LowerFunctionOpsPass because in Grappler we have to be more restrictive
    // about what type of function calls we are allowed to inline.
    if (lower_control_flow && LowerUsingSwitchMergeIsOn(n)) {
      VLOG(2) << "Lower functional control flow op: " << SummarizeNode(*n);
      AddStrictInputSemantics(n, graph.get());
      AddFrameForwardingControlEdge(control_flow_info, n, graph.get());

      if (n->IsIfNode()) {
        TF_RETURN_IF_ERROR(RewriteIfNode(n, graph.get(), false));
      } else if (n->IsCaseNode()) {
        TF_RETURN_IF_ERROR(RewriteCaseNode(n, graph.get(), false));
      } else if (n->IsWhileNode()) {
        TF_RETURN_IF_ERROR(RewriteWhileNode(n, graph.get(), &flib_def, false));
      }
      continue;
    }

    // Skip nodes that are not function calls.
    if (!IsFunctionCall(flib_def, *n)) continue;
    // Skip function calls that we plan to compile later.
    if (MarkedForXlaCompilation(n->def())) continue;
    // Skip nodes in a feed set.
    if (feed_nodes.contains(n->name())) continue;
    // Skip save and restore nodes.
    if (n->name() == item.restore_op || n->name() == item.save_op) continue;

    // Function body that we will inline into the main graph. It can be a
    // function instantiation, or a gradient function instantiated from
    // SymbolicGradient op.
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(MakeFunctionBodyForInlining(*n, flib_def, &fbody));

    InlineFunctionBodyOptions inline_options;
    // Ignore '_noinline' flag in aggressive mode.
    inline_options.ignore_noinline = is_aggressive;

    // Function calls created after inlining If/While ops are always inlined as
    // multi-device functions and are not required to pass additional Grappler
    // validations (side effects execution validation below).
    bool force_inline_as_multi_device = LowerAsMultiDeviceFunctionIsOn(n);

    // `PartitionedCall` is a TF-2.0 function call mechanism for multi-device
    // functions:
    // a) Function can be multi-device.
    // b) Automatic control dependencies tracking guarantees that all function
    //    side-effectful nodes will have a path to one of the control outputs.
    //    Control outputs and control edges between side-effectful (stateful)
    //    nodes are used to explicitly mark the nodes that must execute, and to
    //    define their execution order.
    if (n->IsPartitionedCall() || force_inline_as_multi_device) {
      inline_options.output_control_src = OutputControlSource::kControlOutputs;
      inline_options.inlined_function_body_placer =
          InlinedFunctionBodyPlacer::MultiDevice();
    } else {
      inline_options.output_control_src = OutputControlSource::kDataOutputs;
      inline_options.inlined_function_body_placer =
          InlinedFunctionBodyPlacer::SingleDevice();
    }

    if (fetch_nodes.contains(n->name())) {
      inline_options.keep_caller_node = KeepCallerNode::kFetchable;
    } else if (keep_nodes.contains(n->name())) {
      inline_options.keep_caller_node = KeepCallerNode::kTargetable;
    } else {
      inline_options.keep_caller_node = KeepCallerNode::kDoNotKeep;
    }

    // Basic validation rules defined in common_runtime shared by all functions.
    absl::Status can_inline_function_call =
        ValidateInlining(n, fbody.get(), inline_options);

    // Additional validation rules defined only in Grappler.
    // TODO(ezhulenev): Move it to common_runtime InlineFunctionBodyOptions?
    if (can_inline_function_call.ok()) {
      bool has_outgoing_control_edges = absl::c_any_of(
          n->out_edges(),
          [](const Edge* edge) { return edge->IsControlEdge(); });

      can_inline_function_call = ValidateSideEffectsExecution(
          *fbody, inline_options.output_control_src,
          has_outgoing_control_edges);

      if (!can_inline_function_call.ok() &&
          (is_aggressive || force_inline_as_multi_device)) {
        VLOG(2) << "Ignore error: " << can_inline_function_call.message();
        can_inline_function_call = absl::OkStatus();
      }
    }
    if (can_inline_function_call.ok()) {
      can_inline_function_call = ValidateNoDeadOutputs(flib_def, *fbody);
    }

    if (can_inline_function_call.ok()) {
      VLOG(2) << "Inline function call node: " << n->name();
      AddStrictInputSemantics(n, graph.get());
      AddFrameForwardingControlEdge(control_flow_info, n, graph.get());

      TF_RETURN_IF_ERROR(InlineFunctionBody(flib_def, graph.get(), n,
                                            fbody.get(), inline_options));
      inlined_function_names.push_back(
          fbody->record->fdef().signature().name());

    } else {
      VLOG(2) << "Failed to inline function call node: "
              << can_inline_function_call.message();
    }
  }

  VLOG(4) << "Inlined " << inlined_function_names.size()
          << " function calls: " << absl::StrJoin(inlined_function_names, ", ");

  // ------------------------------------------------------------------------ //
  // Grappler receives the graph after PRE_PLACEMENT, Placer, and POST_PLACEMENT
  // passes, so each node has a valid device assignment. After function inlining
  // and control flow V2 lowering we have to keep graph placed.

  if (inlined_function_names.empty()) {
    VLOG(3) << "Not placing graph after function inlining"
            << " (did not inline any of the function calls).";

  } else if (item.devices().empty()) {
    // If there are no devices available for placer, we do not place graph after
    // function inlining. This happens when Grappler is optimizing the function
    // library, or when a graph optimized "offline", without an active runtime
    // session, for example as a part of batch job for graph
    // analysis/optimization. GrapplerItem instantiated from a function library
    // doesn't have to be fully placed after all optimizations; it will be
    // placed by the function library runtime before execution.
    VLOG(3) << "Not placing graph after function inlining"
            << " (device set is empty)";

  } else {
    // If we are running in an active runtime session, Grappler will get the
    // graph after initial placing is done, and we should have devices for the
    // placer.
    VLOG(3) << "Run placer for the graph after function inlining. "
            << "Devices: [" << absl::StrJoin(item.devices(), ", ") << "]";

    DeviceSet device_set;                               // does not own devices
    std::vector<std::unique_ptr<Device>> fake_devices;  // owns fake devices

    for (const string& name : item.devices()) {
      auto device = std::make_unique<FakeDevice>(name);
      device_set.AddDevice(device.get());
      fake_devices.push_back(std::move(device));
    }

    Placer placer(graph.get(), item.id, &flib_def, &device_set);
    TF_RETURN_IF_ERROR(placer.Run());
  }

  graph->ToGraphDef(output_graph);
  return absl::OkStatus();
}

// Restores tensor mapping after function specialization: all inputs must be
// connected to valid nodes.
void RestoreTensorMapping(const FunctionOptimizerContext& ctx,
                          GraphDef* optimized_graph) {
  if (ctx.tensor_mapping().empty()) return;

  // During function specialization, we might prune unused function outputs. We
  // need to "close the holes" that might appear in the function outputs.
  //
  // Example: prune unused output "f:1"
  //
  //   f = my_func[T=float](...)          f = my_func_specialized[T=float](...)
  //   a = Identity(f:0)             ->   a = Identity(f:0)
  //   b = Identity(f:2)                  b = Identity(f:1)
  //
  // Tensor mapping (size=1): [f:2 -> f:1]
  for (NodeDef& node : *optimized_graph->mutable_node()) {
    for (int idx = 0; idx < node.input_size(); ++idx) {
      TensorId input_tensor = ParseTensorName(node.input(idx));
      if (input_tensor.index() == Graph::kControlSlot) break;

      auto mapping = ctx.tensor_mapping().find(input_tensor);
      if (mapping != ctx.tensor_mapping().end()) {
        node.set_input(idx, TensorIdToString(mapping->second));
      }
    }
  }
}

}  // namespace

absl::Status FunctionOptimizer::RunFunctionOptimizerPass(
    const GrapplerItem& item, GraphDef* optimized_graph) const {
  VLOG(3) << "Run function optimizer pass: grappler_item_id=" << item.id;

  // Inline all function calls into a graph using common_runtime/function
  // implementation (see `InlineFunctionBody` function documentation).
  GraphDef graph_after_inlining;
  TF_RETURN_IF_ERROR(InlineFunctionCalls(item, opt_level_, lower_control_flow_,
                                         &graph_after_inlining));

  // Specialize function calls that we could not inline.
  FunctionOptimizerContext ctx(item, opt_level_, graph_after_inlining);

  for (const NodeDef& node : graph_after_inlining.node()) {
    // Function specialization can modify optimized graph only by adding new
    // nodes, we can check node size to make sure that graph was not modified.
    const int num_nodes_before = optimized_graph->node_size();
    const auto is_graph_modified = [&]() {
      int num_nodes = optimized_graph->node_size();
      DCHECK_GE(num_nodes, num_nodes_before) << "Nodes should not be removed";
      return num_nodes > num_nodes_before;
    };

    // Copy node from the `graph_after_inlining` to the `optimized_graph`.
    const auto copy_node = [&]() { *optimized_graph->add_node() = node; };

    // Find if a node is a function call (direct or indirect).
    const FunctionDef* func = FindFunctionCall(ctx, node);
    if (func == nullptr) {
      copy_node();
      continue;
    }

    const string& func_name = func->signature().name();

    // Specialize it to its instantiation context if it has something worth
    // specializing.
    const bool specialization_worthy = IsParametrized(*func) ||
                                       HasTrulyConstInputs(node, ctx) ||
                                       HasUnusedOutputs(node, *func, ctx);

    // Do not specialize if function has custom gradient or marked nospecialize.
    const string grad_func = ctx.function_library().FindGradient(func_name);
    const bool no_specialize =
        !grad_func.empty() || ctx.IsFeedNode(node.name()) ||
        MarkedNoSpecialize(*func) || MarkedForXlaCompilation(node);

    if (specialization_worthy && !no_specialize) {
      // TODO(ezhulenev): Specialize function call if input has a known shape.
      // Specialize function body for its instantiation attributes and inputs.
      absl::Status status =
          SpecializeFunction(node, *func, &ctx, optimized_graph);
      if (!status.ok() && is_graph_modified()) {
        return status;
      } else if (!status.ok() && !is_graph_modified()) {
        VLOG(3) << "Skip specialization error: " << status.message();
        copy_node();
      }
      continue;
    } else {
      VLOG(2) << "Skip function specialization: " << func->signature().name();
      copy_node();
    }
  }

  RestoreTensorMapping(ctx, optimized_graph);

  // Preserve the graph version.
  *optimized_graph->mutable_versions() = item.graph.versions();
  // Prune unreachable function from the library.
  *optimized_graph->mutable_library() =
      PruneFunctionLibrary(ctx.function_library(), *optimized_graph);

  return absl::OkStatus();
}

absl::Status FunctionOptimizer::Optimize(Cluster*, const GrapplerItem& item,
                                         GraphDef* optimized_graph) {
  // Nothing to do here.
  if (item.graph.library().function_size() == 0) {
    return absl::AbortedError("Nothing to do.");
  }

  TF_RETURN_IF_ERROR(RunFunctionOptimizerPass(item, optimized_graph));

  return absl::OkStatus();
}

}  // end namespace grappler
}  // end namespace tensorflow
