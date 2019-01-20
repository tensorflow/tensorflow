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
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace grappler {
namespace {

// WARNING: Code in this file implicitly assumes that function input and output
// arguments are plain tensors (tensor lists are not supported). Function inputs
// and outputs are always expanded to a single placeholder or output tensor.
// With this assumption, the calling node's input/output ports always match
// function input/output arguments.
//
// This is guaranteed by the implementation of MakeGrapplerFunctionItem.

// Mark functions that were created as a result of function specialization.
constexpr char kGrapplerSpecializedFuncAttr[] = "_GrapplerSpecializedFunc";

// Name of the attribute that defines the function for indirect function calls.
constexpr char kFuncAttrName[] = "f";

constexpr char kNoInlineAttr[] = "_noinline";

bool AttrIsTrue(const FunctionDef& func, const string& attr) {
  return func.attr().count(attr) != 0 && func.attr().at(attr).b();
}

bool MarkedSpecialized(const FunctionDef& func) {
  return AttrIsTrue(func, kGrapplerSpecializedFuncAttr);
}

bool MarkedNoInline(const FunctionDef& func) {
  return AttrIsTrue(func, kNoInlineAttr);
}

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
  auto* func_attr = AttrSlice(func_node).Find(kFuncAttrName);
  return func_attr != nullptr && func_attr->has_func() &&
         func_attr->func().name() == func.signature().name();
}

AttrSlice FunctionInstantiationAttributes(const FunctionDef& func,
                                          const NodeDef& func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    return AttrSlice(func_node);

  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* func_attr = AttrSlice(func_node).Find(kFuncAttrName);
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
  Status Sync() override { return Status::OK(); }

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
      if (!FastAreAttrValuesEqual(lhs.second, (*it).second)) return false;
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

class FunctionOptimizerContext {
 public:
  explicit FunctionOptimizerContext(RewriterConfig::Toggle opt_level,
                                    const GrapplerItem& item)
      : grappler_item_id_(item.id),
        graph_version_(item.graph.versions().producer()),
        opt_level_(opt_level),
        optimization_options_(item.optimization_options()),
        function_library_(OpRegistry::Global(), item.graph.library()),
        available_device_names_(item.devices().begin(), item.devices().end()),
        graph_view_(&item.graph) {
    InitializeTrulyConstNodes(item);
    InitializeFetchNodes(item);
  }

  int graph_version() const { return graph_version_; }

  const RewriterConfig::Toggle opt_level() const { return opt_level_; }

  const GrapplerItem::OptimizationOptions& optimization_options() const {
    return optimization_options_;
  }

  const FunctionLibraryDefinition& function_library() const {
    return function_library_;
  }

  FunctionLibraryDefinition* mutable_function_library() {
    return &function_library_;
  }

  FunctionLibraryRuntime* mutable_function_library_runtime() {
    InitializeFunctionLibraryRuntime();
    return flr_;
  }

  const absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>&
  tensor_mapping() const {
    return tensor_mapping_;
  }

  const absl::flat_hash_map<string, std::vector<string>>& control_overrides()
      const {
    return control_overrides_;
  }

  const GraphView& graph_view() const { return graph_view_; }

  const string& grappler_item_id() const { return grappler_item_id_; }

  const absl::flat_hash_set<string>& fetch_tensors() const {
    return fetch_tensors_;
  }

  const DeviceSet* devices() const {
    // Create fake devices lazily only if we need a DeviceSet.
    if (available_devices_.empty() && !available_device_names_.empty()) {
      for (const string& name : available_device_names_) {
        auto device = absl::make_unique<FakeDevice>(name);
        available_device_set_.AddDevice(device.get());
        available_devices_.push_back(std::move(device));
      }
    }
    return &available_device_set_;
  }

  bool IsFetchNode(const string& node_name) const {
    return fetch_nodes_.find(node_name) != fetch_nodes_.end();
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
        auto inserted = tensor_mapping_.insert({from_tensor, to_tensor});
        DCHECK(inserted.second);
      }
    }
  }

  void AddControlOverrides(const NodeDef& func_node,
                           const std::vector<string>& control_overrides) {
    control_overrides_[func_node.name()].reserve(control_overrides.size());
    for (const string& control_override : control_overrides) {
      control_overrides_[func_node.name()].push_back(control_override);
    }
  }

 private:
  void InitializeTrulyConstNodes(const GrapplerItem& item) {
    absl::flat_hash_set<string> feed_nodes;
    for (const auto& feed : item.feed) {
      feed_nodes.insert(NodeName(feed.first));
    }

    for (const NodeDef& node : item.graph.node()) {
      if (IsConstant(node) && feed_nodes.count(node.name()) == 0) {
        truly_const_nodes_[node.name()] = &node;
      }
    }
  }

  void InitializeFetchNodes(const GrapplerItem& item) {
    for (const string& fetch : item.fetch) {
      fetch_tensors_.insert(fetch);
      fetch_nodes_.insert(NodeName(fetch));
    }
  }

  void InitializeFunctionLibraryRuntime() {
    if (!flr_) {
      Env* env = Env::Default();
      std::vector<std::unique_ptr<Device>> devices;
      devices.push_back(absl::make_unique<FakeDevice>(env, "/device:CPU:0"));
      device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));
      OptimizerOptions optimizer_opts;
      optimizer_opts.set_do_function_inlining(true);
      process_flr_.reset(new ProcessFunctionLibraryRuntime(
          device_mgr_.get(), env, graph_version_, &function_library_,
          optimizer_opts));
      flr_ = process_flr_->GetFLR(device_mgr_->ListDevices()[0]->name());
    }
  }

  const string grappler_item_id_;
  const int graph_version_;
  const RewriterConfig::Toggle opt_level_;
  const GrapplerItem::OptimizationOptions optimization_options_;
  FunctionLibraryDefinition function_library_;

  // These fields initialized lazily only if needed.
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> process_flr_;
  FunctionLibraryRuntime* flr_ = nullptr;

  // Fully defined names of the devices available to the GrapplerItem.
  const absl::flat_hash_set<string> available_device_names_;

  // List of available `FakedDevices` (lazily initialized, see devices()).
  mutable std::vector<std::unique_ptr<Device>> available_devices_;

  // DeviceSet of fake devices (`FakeDevice`) constructed from
  // available_devices_ (lazily initialized).
  mutable DeviceSet available_device_set_;

  // Nodes that are Const and not in feed.
  absl::flat_hash_map<string, const NodeDef*> truly_const_nodes_;
  // Specialized functions.
  absl::flat_hash_map<FunctionSpecializationSignature,
                      const FunctionSpecialization>
      specialized_functions_;

  // GrapplerItem.fetch is a vector of tensors.
  absl::flat_hash_set<string> fetch_tensors_;  // format: node_name:port
  absl::flat_hash_set<string> fetch_nodes_;    // format: node_name

  // After function inlining and specialization, the optimized graph might be in
  // invalid state, nodes can read from non-existing function call nodes that
  // were inlined, or they can read from output index that is no longer valid
  // after unused outputs pruning.
  //
  // Tensor mapping that has to be applied to the graph after all functions
  // optimizations (invalidated tensor id -> optimized graph tensor id).
  absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>
      tensor_mapping_;

  // When we inline a function into the optimized graph, we no longer have the
  // function call node to anchor control dependencies. Instead we must expand
  // each function call control output edge into multiple control dependencies
  // to all side-effectful ops inside the function body.
  //
  // Invalidated function call node name -> Inlined side-effectful nodes
  absl::flat_hash_map<string, std::vector<string>> control_overrides_;

  // Use graph view to find active outputs of the function caller nodes.
  GraphView graph_view_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionOptimizerContext);
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
  for (const string& fetch_tensor : ctx.fetch_tensors()) {
    int port = NodePositionIfSameNode(fetch_tensor, node.name());
    if (port >= 0) active_outputs.insert(port);
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

  return active_outputs.size() != num_outputs;
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
Status PushDownConstInputs(const NodeDef& func_node,
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

  return Status::OK();
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
  std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(keep_inputs),
               [&](const string& input) {
                 return specialization.const_inputs.find(input) ==
                        specialization.const_inputs.end();
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
      if (existing_control_deps.find(ctrl) == existing_control_deps.end()) {
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

    if (specialization.const_inputs.find(input) ==
        specialization.const_inputs.end()) {
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
  if (specialization.active_outputs.size() == tout->list().type_size()) return;

  // Clear input types for the specialized node.
  auto* attr = specialized_func_node->mutable_attr();
  (*attr)["Tout"].mutable_list()->clear_type();

  // Keep output types of active outputs only.
  for (int i = 0; i < tout->list().type_size(); ++i) {
    if (specialization.active_outputs.find(i) !=
        specialization.active_outputs.end()) {
      DataType dt = tout->list().type(i);
      (*attr)["Tout"].mutable_list()->add_type(dt);
    }
  }
}

Status UpdateSpecializedFunctionCallSite(const FunctionDef& func,
                                         const NodeDef& func_node,
                                         const string& specialized_func_name,
                                         NodeDef* specialized_func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    specialized_func_node->set_op(specialized_func_name);

  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* attr = specialized_func_node->mutable_attr();
    (*attr)[kFuncAttrName].mutable_func()->set_name(specialized_func_name);

  } else {
    return errors::InvalidArgument("Unknown function call site");
  }

  return Status::OK();
}

// Update a graph node created from the original function caller node, to the
// function specialization. Function specialization might change the number of
// inputs and outputs, so we have to make sure that graph node is updated
// accordingly.
Status UpdateSpecializedFunctionNode(
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

  return Status::OK();
}

Status InitializeFunctionSpecializationSignature(
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

  return Status::OK();
}

// Create a name for the function specialization. The name of the function, name
// of the node instantiating it, and a Grappler item id should generate unique
// function name. Meta optimizer might create multiple Grappler items for the
// same graph when optimizing functions, but it's guaranteed that they all will
// have unique ids.
string SpecializedFunctionName(const FunctionOptimizerContext& ctx,
                               const FunctionDef& func,
                               const NodeDef& func_node) {
  return absl::Substitute("$0_specialized_for_$1_at_$2",
                          func.signature().name(),
                          absl::StrReplaceAll(func_node.name(), {{"/", "_"}}),
                          ctx.grappler_item_id());
}

Status SpecializeFunction(const NodeDef& func_node, const FunctionDef& func,
                          const int graph_def_version,
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

    return Status::OK();
  }

  // Add a new specialized function definition to the library.
  const auto& flib = ctx->function_library();

  // Make a GrapplerFunctionItem and convert it back to FunctionDef after
  // pushing all constant inputs into the function body.
  GrapplerFunctionItem item;
  TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(func, func_instantiation_attr,
                                              flib, graph_def_version, &item));

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
    int num_func_outputs = 0;
    for (const auto& out_arg : item.outputs()) {
      num_func_outputs += out_arg.output_nodes.size();
    }

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
    return errors::Internal("Created duplicate function specialization");
  }

  specialized_func.mutable_signature()->set_name(specialized_func_name);
  auto* specialized_attr = specialized_func.mutable_attr();
  (*specialized_attr)[kGrapplerSpecializedFuncAttr].set_b(true);

  // Add specialized function to the library.
  TF_RETURN_IF_ERROR(
      ctx->mutable_function_library()->AddFunctionDef(specialized_func));

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

  return Status::OK();
}

// -------------------------------------------------------------------------- //
// Inline direct functions calls.
//
// When we inline direct function calls, we instantiate the function body from
// its FunctionDef and caller node attributes, and embed the instantiated graph
// into the "main graph". When we do that, we must preserve the function call
// semantics:
//
// 1) All input nodes must be executed before any of function body nodes will
//    start executing.
// 2) All function body nodes must be executed before any of the nodes, reading
//    outputs of the function will start executing.
// 3) All nodes with side effects inside a function must be executed, this is
//    different from the nodes with side effects in the main graph, that can be
//    pruned if they are not in transitive dependency set of any of the fetch
//    nodes.
// 4) All nodes of the function body must be execute on the device specified by
//    the function caller node.
//
// To guarantee that function call semantics are preserved after inlining, we
// insert an IdentityN node before the inlined function body, and hook all
// inputs into that, and we insert another IdentityN node to hook all function
// outputs to it.

// Returns `Status::OK()` iff `node` is a direct function call of `func`, and we
// know how to inline it into the main graph, otherwise returns and error
// indicating why the function call is not inlinable.
Status IsInlinableDirectFunctionCall(const FunctionOptimizerContext& ctx,
                                     const FunctionDef& func,
                                     const NodeDef& func_node) {
  // Indirect function calls (PartitionedCallOp) have automatic control
  // dependencies and inlined separately from direct function calls.
  if (!IsDirectFunctionCall(func, func_node)) {
    return errors::InvalidArgument("Unsupported function call type: ",
                                   SummarizeNodeDef(func_node));
  }

  // For direct function  calls we insert IdentityN nodes before/after inlined
  // function body to preserve function call semantics (all inputs evaluated
  // before function evaluation starts, and all function body nodes finished
  // before output consumed by other nodes).
  if (func.signature().input_arg_size() == 0) {
    return errors::FailedPrecondition(
        "Can't inline direct function call with empty inputs: ",
        SummarizeNodeDef(func_node));
  }

  // TODO(ezhulenev): Relax constraint on output args?
  if (func.signature().output_arg_size() == 0) {
    return errors::FailedPrecondition(
        "Can't inline direct function call with empty outputs: ",
        SummarizeNodeDef(func_node));
  }

  // Function must execute all the nodes in a function body that might have side
  // effects. After inlining these nodes into the main graph, we can no longer
  // guarantee that. For now we disable inlining functions with side effects.
  //
  // Attaching control dependency to the output IdentityN node is not safe,
  // because it might be split or pruned in a later optimization pass.
  //
  // Indirect function calls (via PartitionedCallOp) have automatic dependency
  // tracking, and allow us to safely inline functions with side effects.
  bool has_side_effects =
      absl::c_any_of(func.node_def(), [&ctx](const NodeDef& node) {
        return !IsFreeOfSideEffect(node, &ctx.function_library());
      });
  if (has_side_effects) {
    return errors::FailedPrecondition(
        "Can't inline function with side-effects in the function body: ",
        SummarizeNodeDef(func_node));
  }

  // We ignore `_noinline` marker in aggressive mode.
  bool aggressive = ctx.opt_level() == RewriterConfig::AGGRESSIVE;
  if (MarkedNoInline(func) && !aggressive) {
    return errors::FailedPrecondition(
        "Can't inline function marked with '_noinline': ",
        SummarizeNodeDef(func_node));
  }

  // Function specialization and inlining must be mutually exclusive.
  if (MarkedSpecialized(func)) {
    return errors::FailedPrecondition(
        "Can't inline function created in Grappler function specialization: ",
        SummarizeNodeDef(func_node));
  }

  return Status::OK();
}

// Create an IdentityN node to hook the function inputs to: this ensures that
// they're all evaluated before the evaluation of the function body starts.
NodeDef InlinedFunctionInputsNode(const NodeDef& func_node,
                                  const GrapplerFunctionItem& item) {
  NodeDef inputs;
  inputs.set_name(strings::StrCat(func_node.name(), "/", "inlined_inputs"));
  inputs.set_op("IdentityN");
  inputs.set_device(func_node.device());
  *inputs.mutable_input() = func_node.input();
  AttrValue::ListValue* type_list =
      (*inputs.mutable_attr())["T"].mutable_list();

  for (const InputArgExpansion& input_arg : item.inputs()) {
    for (int i = 0; i < input_arg.placeholders.size(); ++i) {
      type_list->add_type(input_arg.data_type);
    }
  }

  return inputs;
}

// Create an IdentityN node to hook the function outputs to: this ensures that
// the function body is fully evaluated before its fanout gets scheduled.
NodeDef InlinedFunctionOutputsNode(
    const NodeDef& func_node, const GrapplerFunctionItem& item,
    const absl::flat_hash_map<absl::string_view, absl::string_view>
        output_tensors) {
  NodeDef outputs;
  outputs.set_name(func_node.name());
  outputs.set_op("IdentityN");
  outputs.set_device(func_node.device());
  AttrValue::ListValue* type_list =
      (*outputs.mutable_attr())["T"].mutable_list();

  for (const OutputArgExpansion& output_arg : item.outputs()) {
    for (const string& output_node : output_arg.output_nodes) {
      const absl::string_view output_tensor = output_tensors.at(output_node);
      type_list->add_type(output_arg.data_type);
      outputs.add_input(strings::StrCat(func_node.name(), "/", output_tensor));
    }
  }

  return outputs;
}

Status InlineDirectFunctionCall(const NodeDef& func_node,
                                const FunctionDef& func,
                                const int graph_def_version,
                                const FunctionOptimizerContext& ctx,
                                GraphDef* optimized_graph) {
  VLOG(2) << "Inline direct function call: " << SummarizeNodeDef(func_node);
  TF_RETURN_IF_ERROR(IsInlinableDirectFunctionCall(ctx, func, func_node));

  const AttrSlice func_instantiation_attr =
      FunctionInstantiationAttributes(func, func_node);

  GrapplerFunctionItem item;
  Status item_status = MakeGrapplerFunctionItem(func, func_instantiation_attr,
                                                ctx.function_library(),
                                                graph_def_version, &item);

  if (!item_status.ok()) {
    return errors::InvalidArgument("Failed to inline function ", func_node.op(),
                                   " instantiated by ", func_node.name(),
                                   ". Error: ", item_status.error_message());
  }

  // Mapping from input placeholder name to function input position.
  absl::flat_hash_map<absl::string_view, int> input_placeholders_idx;
  for (const InputArgExpansion& input_arg : item.inputs()) {
    for (const string& placeholder : input_arg.placeholders) {
      const int idx = input_placeholders_idx.size();
      input_placeholders_idx[placeholder] = idx;
    }
  }

  // Bypass identity nodes added to the graph in place of function outputs.
  absl::flat_hash_set<absl::string_view> output_nodes;
  for (const OutputArgExpansion& output_arg : item.outputs()) {
    for (const string& output_node : output_arg.output_nodes) {
      output_nodes.insert(output_node);
    }
  }

  // For each function output value we added an identity node that reads the
  // tensor from one of the function body nodes. When we inline function into
  // the main graph we want to bypass these nodes, so we keep a mapping from
  // 'output node name' -> 'output tensor name'.
  absl::flat_hash_map<absl::string_view, absl::string_view> output_tensors;

  // Hook inlined function inputs to IdentityN node.
  NodeDef* func_inputs = optimized_graph->add_node();
  *func_inputs = InlinedFunctionInputsNode(func_node, item);

  for (NodeDef& func_body_node : *item.mutable_function_body().mutable_node()) {
    const string& node_name = func_body_node.name();

    // Skip output identity node, and update a mapping to the output tensor.
    if (IsIdentity(func_body_node) && output_nodes.count(node_name)) {
      output_tensors.emplace(node_name, func_body_node.input(0));
      continue;
    }

    // Turn placeholders added in place of input arguments into identity nodes.
    const auto input_placeholder_idx = input_placeholders_idx.find(node_name);
    if (input_placeholder_idx != input_placeholders_idx.end()) {
      CHECK_EQ(0, func_body_node.input_size());
      func_body_node.set_op("Identity");
      (*func_body_node.mutable_attr())["T"] = func_body_node.attr().at("dtype");
      func_body_node.mutable_attr()->erase("dtype");
      func_body_node.mutable_attr()->erase("shape");
      func_body_node.add_input(strings::StrCat(func_inputs->name(), ":",
                                               input_placeholder_idx->second));
    } else {
      // Update the input names if any.
      for (string& input : *func_body_node.mutable_input()) {
        input = AddPrefixToNodeName(input, /*prefix=*/func_node.name());
      }
      // If the node has no input, make hook it up to the func_inputs node to
      // ensure it runs in the same frame as the other nodes of the function
      // body.
      if (func_body_node.input_size() == 0) {
        *func_body_node.add_input() = AsControlDependency(func_inputs->name());
      }
    }

    // Add the function node name as a prefix 1) to node name to avoid
    // collisions; 2) to frame name to avoid multiple LoopCond nodes in one
    // frame after inlining.
    const string prefix = strings::StrCat(func_node.name(), "/");
    TF_RETURN_IF_ERROR(
        AddPrefixAndSuffixToNode(prefix, "" /* suffix */, &func_body_node));

    // Make sure the node is placed.
    func_body_node.set_device(func_node.device());

    // Move the function body node to the optimized graph.
    const auto move_node_to_optimized_graph = [&]() {
      // Annotate the node with the function attributes.
      for (const auto& attr : func.attr()) {
        func_body_node.mutable_attr()->insert(attr);
      }
      // Move the node to the main graph.
      optimized_graph->add_node()->Swap(&func_body_node);
    };

    // Check if a body node is itself a function call and can be inlined.
    const FunctionDef* func_body_node_func =
        FindFunctionCall(ctx, func_body_node);

    if (func_body_node_func != nullptr) {
      Status inlinable = IsInlinableDirectFunctionCall(
          ctx, *func_body_node_func, func_body_node);
      if (inlinable.ok()) {
        TF_RETURN_IF_ERROR(
            InlineDirectFunctionCall(func_body_node, *func_body_node_func,
                                     graph_def_version, ctx, optimized_graph));
      } else {
        VLOG(2) << "Can't inline nested direct function call: "
                << inlinable.error_message();
        move_node_to_optimized_graph();
      }

    } else {
      move_node_to_optimized_graph();
    }
  }

  DCHECK(output_tensors.size() == item.output_size())
      << "Each function output must be mapped to an output tensor";

  // Hook inlined function outputs to IdentityN node.
  NodeDef* func_outputs = optimized_graph->add_node();
  *func_outputs = InlinedFunctionOutputsNode(func_node, item, output_tensors);

  return Status::OK();
}

Status InlineSymbolicGradient(const NodeDef& node,
                              FunctionOptimizerContext* ctx,
                              GraphDef* optimized_graph) {
  VLOG(2) << "Inline symbolic gradient: " << SummarizeNodeDef(node);

  GraphDef graph_def;

  // Create a node to anchor the gradient inputs
  NodeDef* inlined_input = graph_def.add_node();
  inlined_input->set_name("FunctionInputs");
  inlined_input->set_op("IdentityN");
  AttrValue::ListValue* type_list =
      (*inlined_input->mutable_attr())["T"].mutable_list();
  for (const auto& type : node.attr().at("Tin").list().type()) {
    type_list->add_type(static_cast<DataType>(type));
  }

  // Add the gradient node
  NodeDef* inlined = graph_def.add_node();
  *inlined = node;
  inlined->clear_input();
  for (int i = 0; i < node.attr().at("Tin").list().type_size(); ++i) {
    inlined->add_input(strings::StrCat(inlined_input->name(), ":", i));
  }

  // Create a node to anchor the gradient outputs
  NodeDef* inlined_output = graph_def.add_node();
  inlined_output->set_name("FunctionOutputs");
  inlined_output->set_op("IdentityN");
  type_list = (*inlined_output->mutable_attr())["T"].mutable_list();
  for (const auto& type : node.attr().at("Tout").list().type()) {
    type_list->add_type(static_cast<DataType>(type));
  }
  for (int i = 0; i < node.attr().at("Tout").list().type_size(); ++i) {
    inlined_output->add_input(strings::StrCat(inlined->name(), ":", i));
  }

  // Convert the graphdef to a graph
  GraphConstructorOptions graph_ctor_opts;
  graph_ctor_opts.allow_internal_ops = true;
  graph_ctor_opts.expect_device_spec = false;
  Graph graph(ctx->function_library());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_ctor_opts, graph_def, &graph));

  // Recursively inline the functions until there is nothing more to inline. We
  // should at least expand one function.
  int counter = 0;
  while (counter < 50 && ExpandInlineFunctions(
                             ctx->mutable_function_library_runtime(), &graph)) {
    ++counter;
  }

  GraphDef inlined_graph_def;
  graph.ToGraphDef(&inlined_graph_def);

  // Add the default values of attributes to the nodes that have been inlined.
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&inlined_graph_def,
                                               *graph.op_registry(), 0, true));

  // Add the inlined nodes to the graph
  for (NodeDef& inlined_node : *inlined_graph_def.mutable_node()) {
    if (inlined_node.name() == "FunctionOutputs") {
      inlined_node.set_name(node.name());
      for (int i = 0; i < inlined_node.input_size(); ++i) {
        inlined_node.set_input(
            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
      }
    } else if (inlined_node.name() == "FunctionInputs") {
      inlined_node.set_name(
          AddPrefixToNodeName(inlined_node.name(), node.name()));
      inlined_node.clear_input();
      for (int i = 0; i < node.input_size(); ++i) {
        inlined_node.add_input(node.input(i));
      }
    } else {
      inlined_node.set_name(
          AddPrefixToNodeName(inlined_node.name(), node.name()));
      for (int i = 0; i < inlined_node.input_size(); ++i) {
        inlined_node.set_input(
            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
      }
      // If the node has no input, hook it up to the function input node to make
      // sure it runs in the same frame as the other nodes of the function body.
      if (inlined_node.input_size() == 0) {
        *inlined_node.add_input() = AsControlDependency(
            AddPrefixToNodeName("FunctionInputs", node.name()));
      }
    }
    inlined_node.set_device(node.device());
    optimized_graph->add_node()->Swap(&inlined_node);
  }

  return Status::OK();
}

// -------------------------------------------------------------------------- //
// Inline indirect functions calls (aka PartitionedCallOp).
//
// When we inline indirect function calls, we instantiate the function body from
// its FunctionDef and caller node attributes, and embed the instantiated graph
// into the "main graph".
//
// In contrast to direct function calls, `PartitionedCallOp` has automatic
// dependency tracking via input/output control edges, and we relax some of the
// constraints that we have for direct function call inlining.
//
// "When a `PartitionedCallOp` function has a resource (DT_RESOURCE data type)
// input argument it "captures" the mutable resource.  This is implemented by
// automatically adding a incoming control edge from the previous side-effectful
// op touching that resource, and an outgoing control edge to the next
// side-effectful op using the same resource. This serializes the mutations of
// the resource to make graph execution deterministic.
//
// Function call inlining must preserve side effect visibility:
//
// 1) All side effects to the captured resources, that happened before function
//    call must be visible to the function body nodes using that resources.
// 2) All side effects to the captured resources, that happened inside function
//    body, must be visible to every op/function using that resource after the
//    function call completed.

// To guarantee that these properties are preserved after inlining we do:
//
// 1) Forward all input control dependencies from the function call node to the
//    inlined function inputs (Identity nodes).
// 2) Each side-effectful op inside function body adds itself as a control
//    dependency to all the nodes in output control set of function call node.
//
// We do not add any other control dependencies to/from function body nodes,
// because they are pure functions of input tensors, and can be freely
// reordered.

struct MaybeDeadOutput {
  const NodeDef* dead_tensor_src;
  const NodeDef* output_node_dst;
};

// Finds all function outputs that might return a dead tensor. This can happen
// if there is no `Merge` node on the path from the `Switch` node, to the
// function output.
Status MaybeDeadOutputs(const FunctionOptimizerContext& ctx,
                        const GrapplerFunctionItem& item,
                        std::vector<MaybeDeadOutput>* maybe_dead) {
  DCHECK(maybe_dead->empty()) << "Input argument must be an empty vector";

  std::vector<const NodeDef*> dead_tensor_srcs;
  for (const NodeDef& node : item.graph.node()) {
    if (IsSwitch(node)) {
      dead_tensor_srcs.push_back(&node);
      continue;
    }

    // Regular (aka 'direct') function call can also produce dead tensors if
    // the function body has mergeless switches.
    const FunctionDef* func = ctx.function_library().Find(node.op());
    if (func != nullptr) {
      GrapplerFunctionItem func_item;
      TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
          *func, FunctionInstantiationAttributes(*func, node),
          ctx.function_library(), ctx.graph_version(), &func_item));

      std::vector<MaybeDeadOutput> func_dead_outputs;
      TF_RETURN_IF_ERROR(MaybeDeadOutputs(ctx, func_item, &func_dead_outputs));

      if (!func_dead_outputs.empty()) dead_tensor_srcs.push_back(&node);
    }
  }

  // If we do not have dead tensor sources in the function body, it's
  // guaranteed that all output tensors can't become dead.
  if (dead_tensor_srcs.empty()) return Status::OK();

  // Names of the function body nodes that return function output values.
  absl::flat_hash_set<absl::string_view> output_nodes;
  for (const auto& output_expansion : item.outputs()) {
    for (const auto& output_node : output_expansion.output_nodes) {
      output_nodes.insert(output_node);
    }
  }

  GraphTopologyView topology_view;
  TF_RETURN_IF_ERROR(topology_view.InitializeFromGraph(item.graph));

  for (const NodeDef* dead_tensor_src : dead_tensor_srcs) {
    DfsTraversal(topology_view, {dead_tensor_src},
                 TraversalDirection::kFollowOutputs,
                 // Stop traversal when reached first `Merge` node.
                 DfsPredicates::Advance(
                     [](const NodeDef* node) { return !IsMerge(*node); }),
                 // If we reached output node, add MaybeDeadOutput edge.
                 DfsCallbacks::PreOrder([&](const NodeDef* node) {
                   if (output_nodes.find(node->name()) != output_nodes.end()) {
                     maybe_dead->push_back({dead_tensor_src, node});
                   }
                 }));
  }

  return Status::OK();
}

// Returns `Status::OK()` iff `node` is an indirect function call of `func`, and
// we know how to inline it into the main graph, otherwise returns and error
// indicating why the function call is not inlinable.
Status IsInlinableIndirectFunctionCall(const FunctionOptimizerContext& ctx,
                                       const FunctionDef& func,
                                       const NodeDef& func_node) {
  // We inline direct function calls above, using different rules.
  if (!IsIndirectFunctionCall(func, func_node)) {
    return errors::InvalidArgument("Unsupported function call type: ",
                                   SummarizeNodeDef(func_node));
  }

  if (MarkedNoInline(func)) {
    return errors::FailedPrecondition(
        "Can't inline function marked with '_noinline': ",
        SummarizeNodeDef(func_node));
  }

  // Function specialization and inlining must be mutually exclusive.
  if (MarkedSpecialized(func)) {
    return errors::FailedPrecondition(
        "Can't inline function created in Grappler function specialization: ",
        SummarizeNodeDef(func_node));
  }

  // We can't inline functions that are in a fetch set, because it would
  // invalidate fetch tensors (function call node fully inlined and doesn't
  // exist in the optimized graph).
  if (ctx.IsFetchNode(func_node.name())) {
    return errors::FailedPrecondition(
        "Can't inline function in a Grappler item fetch set: ",
        SummarizeNodeDef(func_node));
  }

  // TODO(b/120991525, b/120986912): We need to lower `If` and `While` nodes to
  // `Switch` nodes after function inlining (one more PRE_PLACEMENT pass?), but
  // because of the reason described above we are not sure that it's safe, for
  // now just disable inlining functions with functional control flow.
  const auto is_functional_ctrl_flow_op = [](const NodeDef& node) {
    return IsIf(node) || IsWhile(node);
  };
  if (absl::c_any_of(func.node_def(), is_functional_ctrl_flow_op)) {
    return errors::FailedPrecondition(
        "Can't inline function with `If` or `While` nodes in the function "
        "body: ",
        SummarizeNodeDef(func_node));
  }

  return Status::OK();
}

Status InlineIndirectFunctionCall(const NodeDef& func_node,
                                  const FunctionDef& func,
                                  const int graph_def_version,
                                  FunctionOptimizerContext* ctx,
                                  GraphDef* optimized_graph) {
  VLOG(2) << "Inline indirect function call: " << SummarizeNodeDef(func_node);
  TF_RETURN_IF_ERROR(IsInlinableIndirectFunctionCall(*ctx, func, func_node));

  const AttrSlice func_instantiation_attr =
      FunctionInstantiationAttributes(func, func_node);

  GrapplerFunctionItem item;
  Status item_status = MakeGrapplerFunctionItem(func, func_instantiation_attr,
                                                ctx->function_library(),
                                                graph_def_version, &item);

  if (!item_status.ok()) {
    return errors::InvalidArgument("Failed to inline function ", func_node.op(),
                                   " instantiated by ", func_node.name(),
                                   ". Error: ", item_status.error_message());
  }

  // `PartitionedCallOp` invokes functions with `allow_dead_tensors = true` to
  // reset dead flag, and return default initialized tensors instead of a dead
  // tensors. There is no way to express this in a regular Tensorflow graph, so
  // we choose not to inline if a function can have dead tensors as an output
  // position. In practice `mergeless switches` should not exists in a function
  // body, because tf-eager will only use v2 control flow ops.
  std::vector<MaybeDeadOutput> maybe_dead_outputs;
  TF_RETURN_IF_ERROR(MaybeDeadOutputs(*ctx, item, &maybe_dead_outputs));
  if (!maybe_dead_outputs.empty()) {
    struct MaybeDeadOutputFormatter {
      void operator()(string* out, const MaybeDeadOutput& md) const {
        absl::StrAppend(out, SummarizeNodeDef(*md.dead_tensor_src));
      }
    };
    return errors::FailedPrecondition(
        "Can't inline function with dead outputs. Dead tensor sources (size = ",
        maybe_dead_outputs.size(), "): ",
        absl::StrJoin(maybe_dead_outputs, "\n", MaybeDeadOutputFormatter()));
  }

  GraphView::InputPort control_input_port =
      ctx->graph_view().GetInputPort(func_node.name(), Graph::kControlSlot);
  GraphView::OutputPort control_output_port =
      ctx->graph_view().GetOutputPort(func_node.name(), Graph::kControlSlot);

  // Nodes that have side effects to the captured resources.
  std::vector<string> happens_before;
  absl::c_transform(
      ctx->graph_view().GetFanin(control_input_port),
      std::back_inserter(happens_before),
      [](const GraphView::OutputPort port) { return port.node->name(); });

  VLOG(3) << "Happens before set (size = " << happens_before.size()
          << "): " << absl::StrJoin(happens_before, ", ");

  // Nodes that must observe side effects to the captured resources.
  std::vector<string> happens_after;
  absl::c_transform(
      ctx->graph_view().GetFanout(control_output_port),
      std::back_inserter(happens_after),
      [](const GraphView::InputPort port) { return port.node->name(); });

  VLOG(3) << "Happens after set (size = " << happens_after.size()
          << "): " << absl::StrJoin(happens_after, ", ");

  // Regular (positional) inputs to the function call.
  std::vector<SafeTensorId> inputs;
  for (const string& input : func_node.input()) {
    SafeTensorId tensor_id = ParseTensorName(input);
    if (tensor_id.index() == Graph::kControlSlot) break;
    inputs.push_back(tensor_id);
  }

  // If we have a node inside the function body without inputs (e.g. Const), we
  // must attach a control dependency to it, to make sure that if a function
  // call happens inside a loop, the node will be evaluated in correct frame.
  //
  // If the function call node has no inputs and no control dependencies, it
  // means that it can't be a function call inside a loop, and we can safely
  // insert that node without inputs into the main graph.
  //
  // TODO(ezhulenev): Use FrameMap (see grappler/utils/frame.h) to find out if
  // the function is called inside a loop.
  std::vector<string> empty_inputs_hook;
  if (!item.inputs().empty()) {
    const InputArgExpansion& arg0 = item.inputs()[0];
    DCHECK(!arg0.placeholders.empty());
    empty_inputs_hook.push_back(AsControlDependency(AddPrefixToNodeName(
        arg0.placeholders[0], /*prefix=*/func_node.name())));
  } else if (!happens_before.empty()) {
    empty_inputs_hook.push_back(AsControlDependency(happens_before[0]));
  }

  // Mapping from input placeholder name to function input position.
  absl::flat_hash_map<absl::string_view, int> input_placeholders_idx;
  for (const InputArgExpansion& input_arg : item.inputs()) {
    for (const string& placeholder : input_arg.placeholders) {
      const int idx = input_placeholders_idx.size();
      input_placeholders_idx[placeholder] = idx;
    }
  }

  const string prefix = strings::StrCat(func_node.name(), "/");

  // ------------------------------------------------------------------------ //
  // Before placing the function body nodes we pin input placeholders to the
  // same device as their corresponding input nodes.

  for (NodeDef& func_body_node : *item.graph.mutable_node()) {
    const auto input_placeholder_idx =
        input_placeholders_idx.find(func_body_node.name());

    if (input_placeholder_idx != input_placeholders_idx.end()) {
      const int input_idx = input_placeholder_idx->second;
      const GraphView::OutputPort output_port =
          ctx->graph_view().GetRegularFanin({&func_node, input_idx});

      VLOG(3) << "Pin inlined function input node '" << func_body_node.name()
              << "' to the '" << output_port.node->device() << "' device.";
      func_body_node.set_device(output_port.node->device());
    }
  }

  // ------------------------------------------------------------------------ //
  // After placing nodes corresponding to the function inputs, we need to assign
  // device placements to all other function body nodes.

  GraphDef placed_graph_def;

  const DeviceSet* devices = ctx->devices();

  if (devices->devices().empty()) {
    // If there are no devices available for placer, we just put all nodes to
    // the same device as a function caller node. This can happen if Grappler is
    // running "offline", without active runtime session, for example as a part
    // of a batch job for graph analysis/optimization.
    VLOG(3) << "Assign function call node device to all function body nodes. "
            << "Device: " << func_node.device();
    placed_graph_def = item.mutable_function_body();
    for (NodeDef& node : *placed_graph_def.mutable_node()) {
      node.set_device(func_node.device());
    }
  } else {
    // If we are running in an active runtime session, Grappler will get the
    // graph after initial placing is done, and we should have devices for the
    // placer.
    VLOG(3) << "Run placer for instantiated function body. Devices: ["
            << absl::StrJoin(
                   devices->devices(), ", ",
                   [](string* out, const Device* d) { out->append(d->name()); })
            << "]";

    // Construct a Graph object from the instantiated function body.
    GraphConstructorOptions opts;
    Graph graph(ctx->function_library());
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(opts, item.function_body(), &graph));

    // Use function caller node device as a default for placer.
    const Device* default_device =
        devices->FindDeviceByName(func_node.device());

    Placer placer(&graph, devices, nullptr, /* No session options */
                  default_device);
    TF_RETURN_IF_ERROR(placer.Run());

    // Convert Graph back to the GraphDef.
    graph.ToGraphDef(&placed_graph_def);
  }

  // ------------------------------------------------------------------------ //
  // After all nodes placed we need to prepare them for inlining into the
  // optimized graph: turn placeholders into identities, update nodes
  // connectivity, etc...

  const auto inlined_node_name = [&func_node](const string& name) -> string {
    return AddPrefixToNodeName(name, /*prefix=*/func_node.name());
  };

  for (NodeDef& func_body_node : *placed_graph_def.mutable_node()) {
    const string& node_name = func_body_node.name();

    // Turn placeholders added in place of input arguments into identity nodes.
    const auto input_placeholder_idx = input_placeholders_idx.find(node_name);
    if (input_placeholder_idx != input_placeholders_idx.end()) {
      DCHECK_EQ(0, func_body_node.input_size());
      func_body_node.set_op("Identity");
      (*func_body_node.mutable_attr())["T"] = func_body_node.attr().at("dtype");
      func_body_node.mutable_attr()->erase("dtype");
      func_body_node.mutable_attr()->erase("shape");
      const int input_idx = input_placeholder_idx->second;
      func_body_node.add_input(inputs[input_idx].ToString());

      // All side effects must happen before inputs can start executing.
      for (const string& hb_node : happens_before) {
        func_body_node.add_input(AsControlDependency(hb_node));
      }

    } else {
      // Update inputs of the regular function body nodes.
      for (string& input : *func_body_node.mutable_input()) {
        input = inlined_node_name(input);
      }
      if (func_body_node.input_size() == 0 && !empty_inputs_hook.empty()) {
        *func_body_node.add_input() = empty_inputs_hook[0];
      }
    }

    // Add the function node name as a prefix 1) to node name to avoid
    // collisions; 2) to frame name to avoid multiple LoopCond nodes in one
    // frame after inlining.
    TF_RETURN_IF_ERROR(
        AddPrefixAndSuffixToNode(prefix, /*suffix=*/"", &func_body_node));

    // After inlining into the optimized graph, NodeDef must have all attributes
    // defined, which is not required for a node in a FunctionDef.
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(
        ctx->function_library().LookUpOpDef(func_body_node.op(), &op_def));
    AddDefaultsToNodeDef(*op_def, &func_body_node);
  }

  // Construct a graph view for the preprocessed function body graph.
  GraphView placed_graph_view(&placed_graph_def);

  // Keep track of side-effectful ops inside function body. Each outgoing
  // control edge from the function call node, must be replaced with control
  // edges from inlined side-effectful ops.
  std::vector<string> side_effectful_nodes;

  // We have to make sure that all side-effectful and dataset-output nodes
  // inside a function body will be executed after function inlining.
  for (NodeDef& func_body_node : *placed_graph_def.mutable_node()) {
    const bool node_must_execute =
        !IsFreeOfSideEffect(func_body_node, &ctx->function_library()) ||
        IsDataset(func_body_node);

    if (node_must_execute) {
      int num_fanouts = placed_graph_view.NumFanouts(
          func_body_node, /*include_controlled_nodes=*/true);

      // If the node doesn't have any outgoing edges and we do not have any
      // nodes in the `happens_after` set, we can't inline a function and
      // guarantee that it will be executed. The only exception if we do
      // function library optimization, and the GrapplerItem was instantiated
      // for the function body, because functions do not prune these ops.

      if (num_fanouts == 0 && happens_after.empty() &&
          !ctx->optimization_options().is_function_instantiation) {
        return errors::Internal(
            "Can't inline a function with a side-effectful op with empty "
            "fanouts and empty output control edge set. Function body node: ",
            SummarizeNodeDef(func_body_node));
      }

      side_effectful_nodes.push_back(func_body_node.name());
    }
  }

  // Identity nodes added to the function body in place of function outputs.
  absl::flat_hash_set<string> output_nodes;
  for (const OutputArgExpansion& output_arg : item.outputs()) {
    for (const string& output_node : output_arg.output_nodes) {
      output_nodes.insert(inlined_node_name(output_node));
    }
  }

  // For each function output value we added an identity node that reads the
  // tensor from one of the function body nodes. When we inline function into
  // the main graph we want to bypass these nodes, so we keep a mapping from
  // 'output node name' -> 'output tensor name'.
  absl::flat_hash_map<string, string> output_tensors;

  // Move all the nodes to the optimized graph after successful preprocessing.
  for (NodeDef& func_body_node : *placed_graph_def.mutable_node()) {
    const string& node_name = func_body_node.name();

    // Skip output identity node, and add a mapping to the output tensor.
    if (IsIdentity(func_body_node) && output_nodes.count(node_name)) {
      output_tensors.emplace(node_name, func_body_node.input(0));
      continue;
    }

    optimized_graph->add_node()->Swap(&func_body_node);
  }

  DCHECK(output_tensors.size() == item.output_size())
      << "Each function output must be mapped to an output tensor";

  // TODO(ezhulenev): Inline nested indirect function calls.

  // Indirect function call is fully inlined into the optimized graph, and we do
  // not copy the original function call node, so we have to setup tensor
  // mapping from old output tensors, to the outputs of inlined nodes.
  int output_idx = 0;
  for (const OutputArgExpansion& output : item.outputs()) {
    for (const string& output_node : output.output_nodes) {
      const string inlined_output = inlined_node_name(output_node);
      const string& output_tensor = output_tensors.at(inlined_output);

      const SafeTensorId from_tensor(func_node.name(), output_idx++);
      const SafeTensorId to_tensor = ParseTensorName(output_tensor);

      ctx->AddTensorMapping(from_tensor, to_tensor);
    }
  }

  // After inlining we'll have to forward all control dependencies from function
  // call node to all side-effectful ops inside function body.
  ctx->AddControlOverrides(func_node, side_effectful_nodes);

  VLOG(3) << "Successfully inlined indirect function call: "
          << SummarizeNodeDef(func_node);
  return Status::OK();
}

}  // namespace

Status FunctionOptimizer::Optimize(Cluster*, const GrapplerItem& item,
                                   GraphDef* optimized_graph) {
  // Nothing to do here.
  if (item.graph.library().function_size() == 0) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  FunctionOptimizerContext ctx(opt_level_, item);

  bool inline_gradients = options_.enable_symbolic_gradient_inlining;
  bool inline_func = options_.enable_function_inlining;
  bool specialize_func = options_.enable_function_specialization;

  for (const NodeDef& node : item.graph.node()) {
    // Each node optimization can modify optimized graph only by adding new
    // nodes, we can check node size to make sure that graph was not modified.
    const int num_nodes_before = optimized_graph->node_size();
    const auto is_graph_modified = [&]() {
      int num_nodes = optimized_graph->node_size();
      CHECK_GE(num_nodes, num_nodes_before) << "Nodes should not be removed";
      return num_nodes > num_nodes_before;
    };

    // Add a copy of an input graph node to the optimized graph.
    const auto add_node_copy = [&]() { *optimized_graph->add_node() = node; };

// Skip errors if optimized graph was not modified before error happened.
#define TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(...)                     \
  do {                                                             \
    const Status _status = (__VA_ARGS__);                          \
    if (TF_PREDICT_FALSE(!_status.ok() && is_graph_modified()))    \
      return _status;                                              \
    if (TF_PREDICT_FALSE(!_status.ok() && !is_graph_modified())) { \
      VLOG(3) << "Skip error: " << _status.error_message();        \
      add_node_copy();                                             \
    }                                                              \
  } while (0)

    // ---------------------------------------------------------------------- //
    // 1. Inline symbolic gradients into the optimized graph.                 //
    // ---------------------------------------------------------------------- //

    if (IsSymbolicGradient(node) && inline_gradients) {
      // Inline symbolic gradients only if the corresponding function is not
      // marked as `_noinline`.
      const auto* f_attr = gtl::FindOrNull(node.attr(), "f");
      const string f_name = f_attr != nullptr ? f_attr->func().name() : "";
      const FunctionDef* func = ctx.function_library().Find(f_name);
      if (func && !MarkedNoInline(*func)) {
        TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(
            InlineSymbolicGradient(node, &ctx, optimized_graph));
        continue;
      }
    }

    // ---------------------------------------------------------------------- //
    // 2. Inline or specialize function calls.                                //
    // ---------------------------------------------------------------------- //

    // Find if a node is a function call (direct or indirect).
    const FunctionDef* func = FindFunctionCall(ctx, node);

    if (func != nullptr) {
      const string& func_name = func->signature().name();
      const int graph_def_version = item.graph.versions().producer();

      const bool is_direct_func = IsDirectFunctionCall(*func, node);
      const bool is_indirect_func = IsIndirectFunctionCall(*func, node);

      // 2a. Inline direct function call if it's inlinable.
      if (inline_func && is_direct_func) {
        Status inlinable = IsInlinableDirectFunctionCall(ctx, *func, node);
        if (inlinable.ok()) {
          TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(InlineDirectFunctionCall(
              node, *func, graph_def_version, ctx, optimized_graph));
          continue;
        } else {
          VLOG(2) << inlinable.error_message();
        }
      }

      // 2b. Inline indirect function call if it's inlinable.
      if (inline_func && is_indirect_func) {
        Status inlinable = IsInlinableIndirectFunctionCall(ctx, *func, node);
        if (inlinable.ok()) {
          TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(InlineIndirectFunctionCall(
              node, *func, graph_def_version, &ctx, optimized_graph));
          continue;
        } else {
          VLOG(2) << inlinable.error_message();
        }
      }

      // 2c. Specialize it to its instantiation context if can't be inlined,
      // and it has something worth specializing.
      bool specialization_worthy = IsParametrized(*func) ||
                                   HasTrulyConstInputs(node, ctx) ||
                                   HasUnusedOutputs(node, *func, ctx);

      // Do not specialize if function has custom gradient.
      const string grad_func = ctx.function_library().FindGradient(func_name);

      if (specialize_func && grad_func.empty() && specialization_worthy) {
        // TODO(ezhulenev): Specialize function call if input has a known shape.
        // Specialize function body for its instantiation attributes and inputs.
        TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(
            SpecializeFunction(node, *func, item.graph.versions().producer(),
                               &ctx, optimized_graph));
        continue;
      }
    }

    // ---------------------------------------------------------------------- //
    // If we reached this point, node was not handled by any of the stages
    // (inline, specialize), simply add a copy to the graph.
    add_node_copy();

#undef TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED
  }

  // After function specialization and inlining graph might be in invalid
  // state, and some nodes can read tensors that do not exists anymore in the
  // optimized graph: function call node was fully inlined into the graph, or
  // output index was invalidated by the output pruning.

  if (!ctx.tensor_mapping().empty()) {
    for (NodeDef& node : *optimized_graph->mutable_node()) {
      for (int idx = 0; idx < node.input_size(); ++idx) {
        TensorId input_tensor = ParseTensorName(node.input(idx));
        if (input_tensor.index() == Graph::kControlSlot) break;

        auto mapping = ctx.tensor_mapping().find(input_tensor);
        if (mapping != ctx.tensor_mapping().end()) {
          node.set_input(idx, mapping->second.ToString());
        }
      }
    }
  }

  // Function inlining instantiates function body directly into the optimized
  // graph, and we might end up with control dependencies to the nodes that no
  // longer exist in a graph. We need to apply control overrides to all
  // invalidated nodes, and rewire control dependencies to the inlined
  // side-effectful function body nodes.

  // TODO(ezhulenev): With nested function call inlining, single pass over
  // `control_overrides` might not bring the graph into a valid state,
  // continue until it converges and all invalidated control dependencies
  // removed.

  if (!ctx.control_overrides().empty()) {
    for (NodeDef& node : *optimized_graph->mutable_node()) {
      // Keep track of new control inputs to the node.
      absl::flat_hash_set<string> add_ctrl_inputs;

      // Remove all invalidated control inputs.
      for (int idx = 0; idx < node.input_size(); /* see below */) {
        // TODO(ezhulenev): Use non-allocating TensorId after migrating
        // `control_overrides()` to absl::flat_hash_set.
        SafeTensorId input_tensor = ParseTensorName(node.input(idx));

        auto overrides = ctx.control_overrides().find(input_tensor.node());
        if (overrides != ctx.control_overrides().end()) {
          // If this happens it's a bug in the function inlining.
          if (input_tensor.index() != Graph::kControlSlot) {
            return errors::Internal(
                "Illegal input edge from inlined function call node");
          }
          // Remove control dependency to the inlined function call node.
          node.mutable_input()->SwapElements(idx, node.input_size() - 1);
          node.mutable_input()->RemoveLast();

          // Keep track of all overrides.
          for (const string& override : overrides->second) {
            add_ctrl_inputs.insert(AsControlDependency(override));
          }
        } else {
          // Go to the next input only if the current one was not invalidated,
          // otherwise we need to check the swapped input as well.
          ++idx;
        }
      }

      // Add overrides to the node inputs.
      for (const string& ctrl_input : add_ctrl_inputs) {
        node.add_input(ctrl_input);
      }
    }
  }

  *optimized_graph->mutable_versions() = item.graph.versions();
  *optimized_graph->mutable_library() =
      options_.enable_trim_function_library
          ? PruneFunctionLibrary(ctx.function_library(), *optimized_graph)
          : ctx.function_library().ToProto();

  return Status::OK();
}

void FunctionOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimized_graph,
                                 double result) {
  // Nothing to do for FunctionOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
