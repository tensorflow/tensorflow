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

#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
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

class FakeCPUDevice : public Device {
 public:
  FakeCPUDevice(Env* env, const DeviceAttributes& attr) : Device(env, attr) {}
  Status Sync() override { return Status::OK(); }
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
  gtl::FlatSet<OutputPort> active_outputs;
  std::unordered_map<string, DataType> type_parameters;
  std::unordered_map<string, AttrValue> body_parameters;
  std::unordered_map<InputPort, string> const_inputs;

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

  // TODO(ezhulenev): Migrate to AbslHashValue.
  // TODO(ezhulenev): Optimize performance by computing hashes of unordered
  // values first, and then compute a hash of sorted hashes.
  struct Hash {
    uint64 operator()(FunctionSpecializationSignature const& s) const {
      uint64 h = Hash64(s.func_name);
      h = Hash64Combine(std::hash<bool>()(s.is_in_fetch_set), h);

      // Use std::set/std::map for deterministic iteration order.

      std::set<OutputPort> active_outputs(s.active_outputs.begin(),
                                          s.active_outputs.end());
      for (const auto& active_output : active_outputs) {
        h = Hash64Combine(std::hash<int>()(active_output), h);
      }

      std::map<string, DataType> types(s.type_parameters.begin(),
                                       s.type_parameters.end());
      for (const auto& pair : types) {
        AttrValue attr_value;
        attr_value.set_type(pair.second);
        h = Hash64Combine(Hash64(pair.first), h);
        h = Hash64Combine(AttrValueHash(attr_value), h);
      }

      std::map<string, AttrValue> body(s.body_parameters.begin(),
                                       s.body_parameters.end());
      for (const auto& pair : body) {
        h = Hash64Combine(Hash64(pair.first), h);
        h = Hash64Combine(FastAttrValueHash(pair.second), h);
      }

      std::map<InputPort, string> inputs(s.const_inputs.begin(),
                                         s.const_inputs.end());
      for (const auto& pair : inputs) {
        h = Hash64Combine(std::hash<int>()(pair.first), h);
        h = Hash64Combine(Hash64(pair.second), h);
      }

      return h;
    }
  };
};

struct FunctionSpecialization {
  string specialized_func_name;
  // True if the function caller node is in GrapplerItem fetch set.
  bool is_in_fetch_set;
  // Names of the tensors that were pushed down into the function body.
  gtl::FlatSet<string> const_inputs;
  // Control dependencies of pushed down const inputs have to be attached to
  // function caller node.
  gtl::FlatSet<string> control_deps;
  // Output tensors (ports) that consumed by other nodes in the graph or in a
  // GrapplerItem fetch set.
  gtl::FlatSet<int> active_outputs;
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
        function_library_(OpRegistry::Global(), item.graph.library()),
        graph_view_(&item.graph) {
    InitializeTrulyConstNodes(item);
    InitializeFetchNodes(item);
  }

  const RewriterConfig::Toggle opt_level() const { return opt_level_; }

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

  const gtl::FlatMap<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>&
  tensor_mapping() const {
    return tensor_mapping_;
  }

  const GraphView& graph_view() const { return graph_view_; }

  const string& grappler_item_id() const { return grappler_item_id_; }

  const gtl::FlatSet<string>& fetch_tensors() const { return fetch_tensors_; }

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

 private:
  void InitializeTrulyConstNodes(const GrapplerItem& item) {
    gtl::FlatSet<string> feed_nodes;
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
      DeviceAttributes attr;
      attr.set_name("/device:CPU:0");
      attr.set_device_type("CPU");
      std::vector<std::unique_ptr<Device>> devices;
      devices.push_back(absl::make_unique<FakeCPUDevice>(env, attr));
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
  FunctionLibraryDefinition function_library_;

  // These fields initialized lazily only if needed.
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> process_flr_;
  FunctionLibraryRuntime* flr_ = nullptr;

  // Nodes that are Const and not in feed.
  std::unordered_map<string, const NodeDef*> truly_const_nodes_;
  // Specialized functions.
  std::unordered_map<FunctionSpecializationSignature,
                     const FunctionSpecialization,
                     FunctionSpecializationSignature::Hash>
      specialized_functions_;

  // GrapplerItem.fetch is a vector of tensors.
  gtl::FlatSet<string> fetch_tensors_;  // format: node_name:port
  gtl::FlatSet<string> fetch_nodes_;    // format: node_name

  // After function inlining and specialization, the optimized graph might be in
  // invalid state, nodes can read from non-existing function call nodes that
  // were inlined, or they can read from output index that is no longer valid
  // after unused outputs pruning.
  //
  // Tensor mapping that has to be applied to the graph after all functions
  // optimizations (invalidated tensor id -> optimized graph tensor id).
  gtl::FlatMap<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>
      tensor_mapping_;

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

gtl::FlatSet<int> GetActiveOutputs(const NodeDef& node,
                                   const FunctionOptimizerContext& ctx,
                                   int size_hint = 0) {
  gtl::FlatSet<int> active_outputs;
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
  return std::any_of(node.input().begin(), node.input().end(), is_truly_const);
}

bool HasUnusedOutputs(const NodeDef& func_node, const FunctionDef& func,
                      const FunctionOptimizerContext& ctx) {
  // Functions with tensor list outputs are not supported right now, so the
  // number of output args is the same as number of possible function caller
  // node outputs.
  int num_outputs = func.signature().output_arg_size();
  const gtl::FlatSet<int> active_outputs =
      GetActiveOutputs(func_node, ctx, /*size_hind*/ num_outputs);

  return active_outputs.size() != num_outputs;
}

// Return pruned FunctionDefLibrary with functions that are reachable from
// the optimized graph.
FunctionDefLibrary PruneFunctionLibrary(const FunctionLibraryDefinition& flib,
                                        const GraphDef& optimized_graph) {
  FunctionLibraryDefinition pruned_flib =
      ReachableFunctionLibraryDefinition(flib, optimized_graph);

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
                           gtl::FlatSet<string>* const_inputs,
                           gtl::FlatSet<string>* control_deps) {
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
    gtl::FlatSet<string> existing_control_deps;

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
  gtl::FlatSet<string> const_inputs;
  gtl::FlatSet<string> control_deps;
  TF_RETURN_IF_ERROR(PushDownConstInputs(func_node, *ctx, &item, &const_inputs,
                                         &control_deps));

  // Remove function outputs that do not have any consumers. We can't safely
  // update outputs for the fetch nodes, so we just skip them.
  std::vector<std::pair<int, int>> output_mapping;
  if (!signature.is_in_fetch_set) {
    TF_RETURN_IF_ERROR(
        RemoveUnusedOutputs(signature.active_outputs, &item, &output_mapping));
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

// Returns true iff `node` is a direct function call of `func`, and we know how
// to inline it into the main graph.
bool IsInlinableDirectFunctionCall(const FunctionOptimizerContext& ctx,
                                   const FunctionDef& func,
                                   const NodeDef& node) {
  // Indirect function calls (PartitionedCallOp) have automatic control
  // dependencies and inlined separately from direct function calls.
  bool is_direct_function_call = IsDirectFunctionCall(func, node);

  // For direct function  calls we insert IdentityN nodes before/after inlined
  // function body to preserve function call semantics (all inputs evaluated
  // before function evaluation starts, and all function body nodes finished
  // before output consumed by other nodes).
  bool has_inputs = func.signature().input_arg_size() > 0;
  // TODO(ezhulenev): Relax constraint on output args?
  bool has_outputs = func.signature().output_arg_size() > 0;

  // Function must execute all the nodes in a function body that might have side
  // effects. After inlining these nodes into the main graph, we can no longer
  // guarantee that. For now we disable inlining functions with side effects.
  //
  // Attaching control dependency to the output IdentityN node is not safe,
  // because it might be split or pruned in a later optimization pass.
  //
  // Indirect function calls (via PartitionedCallOp) have automatic dependency
  // tracking, and allow us to safely inline functions with side effects.
  bool free_of_side_effects =
      std::all_of(func.node_def().begin(), func.node_def().end(),
                  [&ctx](const NodeDef& node) {
                    return IsFreeOfSideEffect(node, &ctx.function_library());
                  });

  bool marked_noinline = MarkedNoInline(func);
  bool marked_specialized = MarkedSpecialized(func);

  // We ignore `_noinline` marker in aggressive mode.
  bool aggressive = ctx.opt_level() == RewriterConfig::AGGRESSIVE;

  return is_direct_function_call && has_inputs && has_outputs &&
         free_of_side_effects && !marked_specialized &&
         (!marked_noinline || aggressive);
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
NodeDef InlinedFunctionOutputsNode(const NodeDef& func_node,
                                   const GrapplerFunctionItem& item) {
  NodeDef outputs;
  outputs.set_name(func_node.name());
  outputs.set_op("IdentityN");
  outputs.set_device(func_node.device());
  AttrValue::ListValue* type_list =
      (*outputs.mutable_attr())["T"].mutable_list();

  for (const OutputArgExpansion& output_arg : item.outputs()) {
    for (const string& output_tensor : output_arg.output_tensors) {
      type_list->add_type(output_arg.data_type);
      outputs.add_input(strings::StrCat(func_node.name(), "/", output_tensor));
    }
  }

  return outputs;
}

Status InlineDirectFunctionCall(const NodeDef& func_node,
                                const FunctionDef& func,
                                const FunctionOptimizerContext& ctx,
                                const int graph_def_version,
                                GraphDef* optimized_graph) {
  VLOG(2) << "Inline direct function call: " << SummarizeNodeDef(func_node);

  // Indirect function calls (via PartitionedCallOp) have automatic control
  // dependencies, and doesn't need IdentityN nodes before/after inlined
  // function body, and we inline them separately.
  if (!IsDirectFunctionCall(func, func_node)) {
    return errors::InvalidArgument("Can't inline indirect function call");
  }

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
  int idx = 0;
  std::unordered_map<string, int> input_placeholders_idx;
  for (const InputArgExpansion& input_arg : item.inputs()) {
    for (const string& placeholder : input_arg.placeholders) {
      input_placeholders_idx[placeholder] = idx++;
    }
  }

  // Hook inlined function inputs to IdentityN node.
  NodeDef* func_inputs = optimized_graph->add_node();
  *func_inputs = InlinedFunctionInputsNode(func_node, item);

  for (NodeDef& func_body_node : *item.mutable_function_body().mutable_node()) {
    if (item.IsInputPlaceholder(func_body_node.name())) {
      // Turn input placeholders into identity nodes.
      CHECK_EQ(0, func_body_node.input_size());
      func_body_node.set_op("Identity");
      (*func_body_node.mutable_attr())["T"] = func_body_node.attr().at("dtype");
      func_body_node.mutable_attr()->erase("dtype");
      func_body_node.mutable_attr()->erase("shape");
      int input_idx = input_placeholders_idx[func_body_node.name()];
      func_body_node.add_input(
          strings::StrCat(func_inputs->name(), ":", input_idx));
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

    // Check if a body node is itself a function call and can be inlined.
    const FunctionDef* func_body_node_func =
        FindFunctionCall(ctx, func_body_node);
    if (func_body_node_func != nullptr &&
        IsInlinableDirectFunctionCall(ctx, *func_body_node_func,
                                      func_body_node)) {
      // Recursively inline function calls.
      TF_RETURN_IF_ERROR(
          InlineDirectFunctionCall(func_body_node, *func_body_node_func, ctx,
                                   graph_def_version, optimized_graph));
    } else {
      // Annotate the node with the function attributes.
      for (const auto& attr : func.attr()) {
        func_body_node.mutable_attr()->insert(attr);
      }
      // Move the node to the main graph.
      optimized_graph->add_node()->Swap(&func_body_node);
    }
  }

  // Hook inlined function outputs to IdentityN node.
  NodeDef* func_outputs = optimized_graph->add_node();
  *func_outputs = InlinedFunctionOutputsNode(func_node, item);

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

}  // namespace

Status FunctionOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
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

      // 2a. Inline direct function call if it's inlinable.
      if (inline_func && IsInlinableDirectFunctionCall(ctx, *func, node)) {
        // Inline function body into the optimized graph}
        TF_SKIP_ERROR_IF_GRAPH_UNMODIFIED(InlineDirectFunctionCall(
            node, *func, ctx, item.graph.versions().producer(),
            optimized_graph));
        continue;
      }

      // 2b. Specialize it to it's instantiation context if can't be inlined,
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
