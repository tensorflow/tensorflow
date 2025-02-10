/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/inline_function_utils.h"

#include <deque>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

/*static*/ constexpr const char* const
    LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr;
/*static*/ constexpr const char* const
    LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;

namespace {
// A few string constant used throughout this module.
static constexpr const char* const kArgOp = FunctionLibraryDefinition::kArgOp;
static constexpr const char* const kDeviceArgOp =
    FunctionLibraryDefinition::kDeviceArgOp;
static constexpr const char* const kRetOp = FunctionLibraryDefinition::kRetOp;
static constexpr const char* const kDeviceRetOp =
    FunctionLibraryDefinition::kDeviceRetOp;
static constexpr const char* const kGradientOp =
    FunctionLibraryDefinition::kGradientOp;
static constexpr const char* const kNodeLabel = "Func";
static constexpr const char* const kFuncAttr =
    FunctionLibraryDefinition::kFuncAttr;

// Represents the index-th output of a node.
struct Endpoint {
  Node* node;
  int index;

  // Returns the string name represents this endpoint.
  string name() const {
    if (index == 0) {
      return node->name();
    } else {
      return strings::StrCat(node->name(), ":", index);
    }
  }

  DataType dtype() const { return node->output_type(index); }
};

struct EndpointHash {
  uint64 operator()(const Endpoint& x) const {
    return Hash64(reinterpret_cast<const char*>(&x.node), sizeof(Node*),
                  x.index);
  }
};

struct EndpointEq {
  bool operator()(const Endpoint& x, const Endpoint& y) const {
    return (x.node == y.node) && (x.index == y.index);
  }
};

// The following Add* routines are used to add a few graph nodes while
// functions are transformed.
static Node* AddNoOp(absl::string_view name, Graph* g) {
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("NoOp");
  absl::Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

static Node* AddIdentity(absl::string_view name, Graph* g, Endpoint input) {
  DCHECK_LT(0, input.dtype());
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("Identity");
  ndef.add_input(input.name());
  AddNodeAttr("T", BaseType(input.dtype()), &ndef);
  absl::Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}

std::vector<string> InputDevices(const Node& caller) {
  std::vector<string> input_devices(caller.in_edges().size());
  std::vector<string> input_tensors(caller.in_edges().size());

  for (const Edge* edge : caller.in_edges()) {
    if (edge->IsControlEdge()) continue;
    const string& input_device = edge->src()->has_assigned_device_name()
                                     ? edge->src()->assigned_device_name()
                                     : edge->src()->requested_device();
    input_devices[edge->dst_input()] = input_device;
    input_tensors[edge->dst_input()] =
        absl::StrCat(edge->src()->name(), ":", edge->src_output());
  }

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Function instantiation input devices:";
    for (int i = 0; i < input_devices.size(); ++i) {
      if (input_tensors[i].empty()) continue;  // skip control edges
      VLOG(4) << "    [index " << i << "]"
              << " device: " << input_devices[i]
              << " (input: " << input_tensors[i] << ")";
    }
  }

  return input_devices;
}

// Place input nodes on the same device as the corresponding caller input
// node. Do not specify any placement for all other nodes.
class DefaultFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit DefaultFunctionBodyPlacer(const Node& caller)
      : input_devices_(InputDevices(caller)) {}

  absl::optional<string> InputNodeDevice(int input_index) const override {
    return input_devices_[input_index];
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return absl::nullopt;
  }
  bool ColocateInputOutputIdentities() const override { return false; }
  absl::optional<string> ControlNodeDevice() const override {
    return absl::nullopt;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    return absl::nullopt;
  }

 private:
  const std::vector<string> input_devices_;
};

// Place all nodes on the same device as caller node.
class SingleDeviceFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit SingleDeviceFunctionBodyPlacer(const Node& caller)
      : caller_device_(caller.def().device()) {}

  absl::optional<string> InputNodeDevice(int input_index) const override {
    return caller_device_;
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return caller_device_;
  }
  bool ColocateInputOutputIdentities() const override { return false; }
  absl::optional<string> ControlNodeDevice() const override {
    return caller_device_;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    return caller_device_;
  }

 private:
  const string caller_device_;
};

// Place input nodes on the same device as the corresponding caller input
// node. Do not place output node. Place control nodes on the same device as
// caller node. For all function body nodes overrides job, replica and task
// parts of the device assignment to match function caller node.
class MultiDeviceFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit MultiDeviceFunctionBodyPlacer(const Node& caller)
      : caller_device_(caller.def().device()),
        input_devices_(InputDevices(caller)) {
    has_parsed_caller_device_ =
        DeviceNameUtils::ParseFullName(caller_device_, &caller_parsed_device_);
  }

  absl::optional<string> InputNodeDevice(int input_index) const override {
    return input_devices_[input_index];
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return absl::nullopt;
  }
  bool ColocateInputOutputIdentities() const override { return true; }
  absl::optional<string> ControlNodeDevice() const override {
    return caller_device_;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    // LINT.IfChange
    // TODO(ezhulenev): If function would have been instantiated as a
    // multi-device function and executed via FunctionLibraryRuntime, it could
    // be potentially placed on any available device. However there are multiple
    // tests relying on this assumption. Fix them, and remove this line.
    if (ndef.device().empty()) return caller_device_;

    if (!has_parsed_caller_device_) return ndef.device();

    DeviceNameUtils::ParsedName ndef_parsed_device;
    if (!DeviceNameUtils::ParseFullName(ndef.device(), &ndef_parsed_device))
      return ndef.device();

    DeviceNameUtils::MergeUnsetDevNames(&ndef_parsed_device,
                                        caller_parsed_device_);
    return DeviceNameUtils::ParsedNameToString(ndef_parsed_device);
    // LINT.ThenChange(../../compiler/mlir/tensorflow/ir/tf_ops.cc)
  }

 private:
  string caller_device_;
  bool has_parsed_caller_device_;
  DeviceNameUtils::ParsedName caller_parsed_device_;
  std::vector<string> input_devices_;
};

}  // namespace

std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::DefaultPlacer(const Graph& graph,
                                         const Node& caller) {
  VLOG(3) << "Create default placer for inlined function body.";
  return std::make_unique<DefaultFunctionBodyPlacer>(caller);
}

std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::SingleDevicePlacer(const Graph& graph,
                                              const Node& caller) {
  VLOG(3) << "Create single device placer for inlined function body.";
  return std::make_unique<SingleDeviceFunctionBodyPlacer>(caller);
}

std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::MultiDevicePlacer(const Graph& graph,
                                             const Node& caller) {
  VLOG(3) << "Create multi device placer for inlined function body.";
  return std::make_unique<MultiDeviceFunctionBodyPlacer>(caller);
}

namespace {

absl::Status ValidateNoInline(const FunctionBody* fbody) {
  const auto attr = AttrSlice(&fbody->record->fdef().attr());
  bool noinline = false;
  if (TryGetNodeAttr(attr, kNoInlineAttr, &noinline) && noinline) {
    return errors::InvalidArgument(
        "Can't inline function marked with '_noinline'");
  }
  return absl::OkStatus();
}

using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;

// Propagate the debug info of `nodes` in function `func` to the `target` node.
// If the debug info of any node is missing, its node name and function name
// is used.
void PropagateDebugInfoToNode(const string& func,
                              const std::vector<const Node*>& nodes,
                              NodeDef* target) {
  if (nodes.empty() || target->has_experimental_debug_info()) {
    return;
  }
  for (const Node* node : nodes) {
    const auto& node_def = node->def();
    if (node_def.has_experimental_debug_info()) {
      target->mutable_experimental_debug_info()->MergeFrom(
          node_def.experimental_debug_info());
    } else {
      target->mutable_experimental_debug_info()->add_original_node_names(
          node_def.name());
      target->mutable_experimental_debug_info()->add_original_func_names(func);
    }
  }
}
}  // namespace

string InlineFunctionBodyOptions::DebugString() const {
  const auto true_false = [](bool b) { return b ? "true" : "false"; };

  const auto keep_caller_node_str = [this]() -> string {
    switch (keep_caller_node) {
      case KeepCallerNode::kDoNotKeep:
        return "DoNotKeep";
      case KeepCallerNode::kFetchable:
        return "Fetchable";
      case KeepCallerNode::kTargetable:
        return "Targetable";
    }
  };

  return absl::StrCat(
      "disable_inlining=", true_false(disable_inlining),
      ", ignore_noinline=", true_false(ignore_noinline),
      ", inline_impl_selection_group_functions=",
      true_false(inline_impl_selection_group_functions),
      ", keep_caller_node=", keep_caller_node_str(), ", output_control_src=",
      output_control_src == OutputControlSrc::kDataOutputs ? "DataOutputs"
                                                           : "ControlOutputs",
      ", inlined_function_body_placer=", inlined_function_body_placer.name,
      ", uniquify_frame_names=", true_false(uniquify_frame_names));
}

absl::Status ValidateInlining(const Node* node, const FunctionBody* fbody,
                              const InlineFunctionBodyOptions& options) {
  // TODO(ezhulenev): Currently common_runtime function inlining can't guarantee
  // that all side-effectful ops will be executed after inlining. See Grappler
  // function_optimizer for details. Unify all function inlining mechanism.
  // Do not inline if `!fbody->control_ret_nodes.empty()`.

  const auto num_node_inputs = static_cast<size_t>(node->num_inputs());
  const auto num_node_outputs = static_cast<size_t>(node->num_outputs());

  if (num_node_inputs != fbody->arg_types.size() ||
      num_node_inputs != fbody->arg_nodes.size()) {
    return errors::InvalidArgument(
        "Node inputs do not match function arguments: inputs=", num_node_inputs,
        " arg_types=", fbody->arg_types.size(),
        " arg_nodes=", fbody->arg_nodes.size());
  }

  if (num_node_outputs != fbody->ret_types.size() ||
      num_node_outputs != fbody->ret_nodes.size()) {
    return errors::InvalidArgument(
        "Node outputs do not match function returns: outputs=",
        num_node_outputs, " ret_types=", fbody->ret_types.size(),
        " ret_nodes=", fbody->ret_nodes.size());
  }

  for (int i = 0; i < node->num_inputs(); ++i) {
    if (node->input_type(i) != fbody->arg_types[i]) {
      return errors::InvalidArgument(
          "Node input type doesn't match function argument type: ",
          node->input_type(i), " != ", fbody->arg_types[i], " @ index=", i);
    }
  }
  for (int i = 0; i < node->num_outputs(); ++i) {
    if (node->output_type(i) != fbody->ret_types[i]) {
      return errors::InvalidArgument(
          "Node output type doesn't match function return type: ",
          node->output_type(i), " != ", fbody->ret_types[i], " @ index=", i);
    }
  }

  if (options.disable_inlining) {
    return errors::InvalidArgument(
        "Function inlining explicitly disabled by 'options.disable_inlining'");
  }

  if (!options.inline_impl_selection_group_functions) {
    bool is_impl_selection_group_function =
        fbody->record->fdef().attr().find("api_implements") !=
        fbody->record->fdef().attr().end();
    if (is_impl_selection_group_function) {
      return errors::InvalidArgument(
          "Inlining of implementation selection group function ",
          fbody->record->fdef().signature().name(),
          " is disabled by options.inline_impl_selection_group_functions");
    }
  }

  if (!options.ignore_noinline) {
    TF_RETURN_IF_ERROR(ValidateNoInline(fbody));
  }

  return absl::OkStatus();
}

// Function inlining must preserve function execution semantics with regards to
// side-effects visibility. Tensorflow in Eager mode has an automatic control
// dependencies tracking mechanism, which enforces well-defined execution order
// of all side-effects. Any other frontend (e.g. Swift) must produce graphs
// following the same rules, to ensure that function inlining works correctly.
//
// IMPORTANT: Currently we do not have a true notion of "side-effectful" node,
// we assume that all stateful nodes might have side-effects, though it's not
// true in practice, e.g. `ReadVariableOp` doesn't have an observable
// side-effect.
//
// Automatic control dependency rules in Tensorflow 2.0 (python in eager mode):
//
// 1) When a function has a resource (DT_RESOURCE data type) input argument it
//   "captures" the mutable resource.  This is implemented by automatically
//    adding a incoming control edge from the previous side-effectful op
//    touching that resource, and an outgoing control edge to the next
//    side-effectful op using the same resource. This serializes the mutations
//    of the resource to make graph execution deterministic.
//
// 2) All stateful ops inside a function body are guaranteed to execute in
//    program order, this is achieved by adding control edges between stateful
//    ops at graph construction time. Stateful ops (or ops that must execute)
//    should be in the function control return set. Having a data edge to the
//    regular function output might be not enough, because after function
//    inlining it might happen that data output is unused.
//
// 3) Furthermore, all ops accepting the same resource as an input are
//    guaranteed to run in program order. This is also done by adding control
//    edges at graph construction time. The last op touching the resource
//    must be in a control return set, which will guarantee that all side
//    effects to the resource will happen before function completion.
//
// Function inlining must preserve side-effect visibility:
//
// 1) All side-effects to the captured resources, that happened before function
//    call must be visible to the function body nodes using that resources.
//
// 2) All side-effects to the captured resources, that happened inside function
//    body, must be visible to every op/function using that resource after the
//    function call completed.
//
// To guarantee that these properties are preserved after inlining we:
//
// 1) Create "input_control_node" NoOp. Function call node incoming control
//    edges will be forwarded *to* this node. Function inputs (Identity nodes)
//    will have a control edge *from* this node. If function body has nodes
//    without inputs, they will have a control edge *from* this node.
//
// 2) Create "output_control_node" NoOp. All nodes that have incoming control
//    edge *from* the function call node, will be forwarded to this node.
//
//    We have two options for choosing which nodes will have a control edge *to*
//    the "output control node":
//       a) control returns            (`control_ret` field in FunctionDef)
//       b) data returns               (`ret` field in FunctionDef)
//
//    We do a) for multi-device function calls in Tensorflow v2 and b)
//    for the rest for compatibility with Tensorflow v1.
//
//    Following the automatic control dependencies tracking rules, a node that
//    has an incoming control edge from the function call node is dependent on
//    the side-effects happening inside the function body. The output control
//    node will guarantee side-effects execution order.
//
//    If function call node doesn't have an outgoing control edge, it means that
//    no one is interested in observing side-effects that might have happened.
//
// Function inlining might leave the graph in partially-placed state. Function
// inlining caller must call Placer to guarantee that all nodes are placed.
//
// Function inlining with `options.override_device=true` will leave graph in
// fully placed state, by overriding all inlined nodes devices with the caller
// node device, but it will make functions always single-device. These functions
// after inlining will not be able to handle resources on multiple devices. This
// is currently acceptable for XLA use cases (XLA cluster is always executed on
// a single device).
//
// TODO(ezhulenev): Documentation above is ahead of implementation below.
absl::Status InlineFunctionBody(const FunctionLibraryDefinition& flib_def,
                                Graph* g, Node* caller,
                                const FunctionBody* fbody,
                                const InlineFunctionBodyOptions& options) {
  VLOG(3) << "Inline function call: " << SummarizeNode(*caller) << " ["
          << options.DebugString() << "]";
  VLOG(4) << "Inlining function: "
          << fbody->record->fdef().DebugString();  // NOLINT
  VLOG(4) << "Current graphdef: " << g->ToGraphDefDebug().DebugString();
  VLOG(4) << "Caller: " << caller->DebugString();

  absl::Status validation = ValidateInlining(caller, fbody, options);
  if (!validation.ok()) {
    return errors::Internal("Inlining mismatch: ", validation.message());
  }

  // Placer is responsible for assigning devices for all nodes that we will add
  // to the graph.
  const std::unique_ptr<InlinedFunctionBodyPlacer> placer =
      options.inlined_function_body_placer.get(*g, *caller);

  // We can't possibly introduce a duplicate control edge during function
  // inlining, so we skip this check in calls to the 'g->AddControlEdge(...)'.
  static constexpr bool kDoNotCheckDuplicates = true;

  // ------------------------------------------------------------------------ //
  // Helper functions to create `NoOp` and `Identity` nodes for auxiliary
  // control nodes and inlined function inputs and outputs.

  // Add a NoOp node for function control inputs/outputs.
  const auto no_op = [&](absl::string_view name) -> Node* {
    Node* node = AddNoOp(absl::StrCat(caller->name(), "/", name), g);
    const absl::optional<string> device = placer->ControlNodeDevice();
    if (device.has_value()) node->set_requested_device(*device);
    return node;
  };

  // Add an Identity node for function input.
  const auto input_identity = [&](absl::string_view name, Endpoint input,
                                  int index) -> Node* {
    Node* node = AddIdentity(absl::StrCat(caller->name(), "/", name), g, input);
    const absl::optional<string> device = placer->InputNodeDevice(index);
    if (device.has_value()) node->set_requested_device(*device);
    bool colocate_identity = placer->ColocateInputOutputIdentities();
    if (colocate_identity) {
      node->AddAttr(kColocationAttrName,
                    std::vector<string>{absl::StrCat(kColocationGroupPrefix,
                                                     input.node->name())});
    }
    return node;
  };

  // Add an Identity node for function output.
  const auto output_identity = [&](absl::string_view name, Endpoint input,
                                   int index) -> Node* {
    Node* node = AddIdentity(absl::StrCat(caller->name(), "/", name), g, input);
    const absl::optional<string> device = placer->OutputNodeDevice(index);
    if (device.has_value()) node->set_requested_device(*device);
    bool colocate_identity = placer->ColocateInputOutputIdentities();
    if (colocate_identity) {
      node->AddAttr(kColocationAttrName,
                    std::vector<string>{absl::StrCat(kColocationGroupPrefix,
                                                     input.node->name())});
    }
    return node;
  };

  // ------------------------------------------------------------------------ //
  // Helper function to get an input/output argument name by index. For
  // functions instantiated from SymbolicGradien corresponding FunctionDef is
  // empty, and argument name is unknown.

  auto arg_name = [&](auto& args, size_t i) -> absl::string_view {
    if (i < args.size()) {
      return args[i].name();
    } else {
      return "<unknown>";
    }
  };

  // ------------------------------------------------------------------------ //
  // Input edges. For data edges coming into "caller", we first compute the
  // <src>:<src_output> for the i-th input in "inputs".
  // If "caller" has any input control dependencies, we add a NoOp
  // node "input_control_node", which depends on "caller"'s control inputs.
  std::vector<Endpoint> inputs(caller->num_inputs());
  Node* input_control_node = nullptr;
  for (const Edge* e : caller->in_edges()) {
    if (e->IsControlEdge()) {
      if (input_control_node == nullptr) {
        input_control_node = no_op("input_control_node");
      }
      g->AddControlEdge(e->src(), input_control_node, kDoNotCheckDuplicates);
    } else {
      inputs[e->dst_input()] = {e->src(), e->src_output()};
    }
  }
  if (input_control_node != nullptr) {
    VLOG(3) << "Created input control node: " << input_control_node->name();
  }

  // We create one Identity node for each input.
  std::vector<Node*> input_nodes;
  std::map<absl::string_view, absl::string_view> input_node_name_map;
  for (std::size_t i = 0; i < fbody->arg_nodes.size(); ++i) {
    if (inputs[i].node == nullptr)
      return errors::Internal("Null node found for input ", i);

    Node* n = input_identity("input", inputs[i], i);
    input_node_name_map[arg_name(fbody->record->fdef().signature().input_arg(),
                                 i)] = n->name();
    input_nodes.push_back(n);
  }

  // ------------------------------------------------------------------------ //
  // Duplicate fbody->graph into 'g'.  First, we copy the nodes of
  // fbody->graph into 'g' except the source and sink nodes.  We copy
  // edges among nodes in 'fbody->graph'.
  //
  // If 'x' is a node in fbody->graph and its copy in 'g' is 'y', we
  // remember 'y' in node_map[x->id()].
  std::unordered_set<string> fn_nodes;
  for (Node* n : fbody->graph->op_nodes()) {
    fn_nodes.insert(n->name());
  }
  std::vector<Node*> node_map(fbody->graph->num_node_ids());
  for (Node* n : fbody->graph->op_nodes()) {
    NodeDef ndef = n->def();

    // Maybe override requested node device assignment.
    const absl::optional<string> device = placer->BodyNodeDevice(ndef);
    if (device.has_value()) ndef.set_device(*device);

    // Add inlined function name to inlined node debug information.
    PropagateDebugInfoToNode(fbody->record->fdef().signature().name(), {n},
                             &ndef);

    // Add the function node name as a prefix:
    //  1) to node name to avoid collisions
    //  2) to frame name to avoid multiple LoopCond nodes in one frame
    //  3) to colocation attribute
    const string prefix = strings::StrCat(caller->name(), "/");
    TF_RETURN_IF_ERROR(AddPrefixAndSuffixToNode(prefix, /*suffix=*/"", &ndef,
                                                options.uniquify_frame_names));

    // If the colocation attribute is an input arg, we need to change it to the
    // new input (Identity) node now.
    TF_RETURN_IF_ERROR(
        MaybeUpdateColocationConstraintsWithMap(input_node_name_map, &ndef));

    TF_RETURN_IF_ERROR(
        MaybeAddPrefixToColocationConstraints(fn_nodes, prefix, &ndef));

    absl::Status added_node;
    Node* clone = g->AddNode(std::move(ndef), &added_node);
    TF_CHECK_OK(added_node);
    node_map[n->id()] = clone;
    clone->SetStackTrace(n->GetStackTrace());

    // If there is an input control node, and one of:
    // a) the node has no data or control inputs, or
    // b) the node is a function call (including SymbolicGradient),
    //    then add a control edge from the input control node to the clone (only
    //    if it does not already have a control input).
    //
    // We must not execute any nodes if the original function call would not
    // have executed. This is especially critical when the function call is
    // inside a control-flow construct like tf.cond(). Case (a) ensures that
    // such nodes do not run.
    //
    // The purpose of case (b) is to ensure that instances of case (a) created
    // by further inlining steps also receive the control dependency.
    //
    // This edge is required to transfer execution frame down to all function
    // body nodes of inlined nested function calls.
    if (input_control_node) {
      const auto is_input_edge = [](const Edge* e) -> bool {
        return !e->src()->IsSource();
      };
      const auto is_control_edge = [](const Edge* e) -> bool {
        return !e->src()->IsSource() && e->IsControlEdge();
      };

      // Forward execution frame if:
      //
      // a) The node has no data or control inputs.
      // b) OR the node is a function call without control inputs (control edge
      //    will be used in nested function inlining to forward execution frame
      //    to constants inside the function body).
      //
      // c) Do not forward control frame to function argument nodes, they will
      //    be connected to the corresponding function input later.
      const bool forward_execution_frame =
          (absl::c_none_of(n->in_edges(), is_input_edge) ||       // (a)
           (n->IsFunctionCall() &&                                // (b)
            absl::c_none_of(n->in_edges(), is_control_edge))) &&  //
          !n->IsArg();                                            // (c)

      if (forward_execution_frame) {
        VLOG(4) << "Add control edge from input control node to: "
                << clone->name();
        g->AddControlEdge(input_control_node, clone, kDoNotCheckDuplicates);
      }
    }
  }
  for (const Edge* e : fbody->graph->edges()) {
    if (e->src()->IsSource() || e->src()->IsSink() || e->dst()->IsSource() ||
        e->dst()->IsSink()) {
      continue;
    }
    Node* src_copy = node_map[e->src()->id()];
    Node* dst_copy = node_map[e->dst()->id()];
    g->AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }

  // ------------------------------------------------------------------------ //
  // Connect input edges.
  //
  // Then, we connect inputs[i] to the i-th identity node added. The nodes that
  // previously connected to the j-th output of i-th arg node are reconnected
  // to the i-th identity node.
  //
  // The added identity nodes depend on "input_control_node".
  VLOG(4) << "Add input Identity nodes for each function argument:";

  for (std::size_t i = 0; i < fbody->arg_nodes.size(); ++i) {
    Node* arg = node_map[fbody->arg_nodes[i]->id()];
    Node* n = input_nodes[i];
    VLOG(4) << "    [index " << i << "] "
            << arg_name(fbody->record->fdef().signature().input_arg(), i)
            << " as " << n->name() << " (input: " << inputs[i].name()
            << ", requested_device: " << n->requested_device() << ")";

    if (input_control_node) {
      g->AddControlEdge(input_control_node, n, kDoNotCheckDuplicates);
    }
    for (const Edge* e : arg->out_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(n, e->dst(), kDoNotCheckDuplicates);
      } else {
        g->AddEdge(n, 0, e->dst(), e->dst_input());
      }
    }
    node_map[fbody->arg_nodes[i]->id()] = n;
    g->RemoveNode(arg);  // 'arg' is disconnected.
  }

  // ------------------------------------------------------------------------ //
  // Connect output edges.
  //
  // For i-th return node in fbody->graph, we add in "g" an identity node
  // (outputs[i-th]). We then reconnect every incoming edge into the i-th return
  // node to the added identity node.
  //
  // For every data edge coming out of "callee"s i-th output, we reconnect it to
  // the i-th identity added above.
  //
  // If "callee" is control-depended upon by any other nodes, we add a NoOp node
  // "output_control_node". "output_control_node" depends on all identity nodes
  // added above or on all control return nodes (controlled by
  // `options.output_control_src` value). And nodes previously depend on
  // "callee" is changed to depend on "output_control_node".
  //
  // If `keep_node_fetchable` is `true` we always add an output control node, to
  // guarantee that executing a fetchable node will execute all side-effects.
  VLOG(4) << "Add output Identity nodes for each function output argument:";

  std::vector<Node*> outputs(caller->num_outputs());
  for (std::size_t i = 0; i < fbody->ret_nodes.size(); ++i) {
    Node* ret = node_map[fbody->ret_nodes[i]->id()];
    Endpoint data;  // Data input for the ret node.
    for (const Edge* e : ret->in_edges()) {
      if (!e->IsControlEdge()) {
        data = {e->src(), e->src_output()};
        break;
      }
    }
    CHECK(data.node != nullptr);
    Node* n = output_identity("output", data, i);
    outputs[i] = n;
    VLOG(4) << "    [index " << i << "] "
            << arg_name(fbody->record->fdef().signature().output_arg(), i)
            << " as " << n->name() << " (ret: " << data.node->name() << ":"
            << data.index << ", requested_device: " << n->requested_device()
            << ")";
    for (const Edge* e : ret->in_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(e->src(), n, kDoNotCheckDuplicates);
      }
    }
    g->RemoveNode(ret);  // 'ret' is disconnected.
  }

  Node* output_control_node = nullptr;
  const bool has_control_outputs = absl::c_any_of(
      caller->out_edges(), [](const Edge* e) { return e->IsControlEdge(); });

  using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
  const bool keep_caller_node =
      options.keep_caller_node == KeepCallerNode::kFetchable ||
      options.keep_caller_node == KeepCallerNode::kTargetable;

  if (has_control_outputs || keep_caller_node) {
    output_control_node = no_op("output_control_node");
    VLOG(4) << "Add output control node: " << output_control_node->name();
    if (options.output_control_src == OutputControlSrc::kDataOutputs) {
      for (Node* n : outputs) {
        VLOG(4) << "    [data output] add control edge from: " << n->name();
        g->AddControlEdge(n, output_control_node, kDoNotCheckDuplicates);
      }
    } else {
      for (Node* fbody_node : fbody->control_ret_nodes) {
        Node* n = node_map[fbody_node->id()];
        VLOG(4) << "    [control output] add control edge from: " << n->name();
        g->AddControlEdge(n, output_control_node, kDoNotCheckDuplicates);
      }
    }
  }

  // We can't leave output control node without incoming control edges, because
  // in this case outgoing control edge will loose execution frame information.
  // We connect input_control_node and output_control_node with a control edge
  // to forward execution frame to the controlled nodes. Above we add a control
  // edge to all function calls inside function body, to guarantee that we will
  // always have input_control_node when we need it.
  if (output_control_node && output_control_node->in_edges().empty()) {
    if (input_control_node) {
      VLOG(4) << "Add a control edge between input and output control nodes: "
              << input_control_node->name() << " to "
              << output_control_node->name();
      g->AddControlEdge(input_control_node, output_control_node,
                        kDoNotCheckDuplicates);
    } else {
      VLOG(4) << "Function inlining potentially dropped execution frame "
                 "information from outgoing control edges.";
    }
  }

  for (const Edge* e : caller->out_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(output_control_node, e->dst(), kDoNotCheckDuplicates);
    } else {
      g->AddEdge(outputs[e->src_output()], 0, e->dst(), e->dst_input());
    }
  }

  // ------------------------------------------------------------------------ //
  // Add an IdentityN or NoOp node in-place of caller node to keep `caller`
  // fetchable or targetable.

  if (keep_caller_node) {
    std::vector<NodeBuilder::NodeOut> output_tensors;
    absl::c_transform(outputs, std::back_inserter(output_tensors),
                      [](Node* n) { return NodeBuilder::NodeOut(n, 0); });

    Node* caller_substitute_node;
    if (options.keep_caller_node == KeepCallerNode::kTargetable ||
        output_tensors.empty()) {
      // IdentityN node must have at least one data input. If function has no
      // data outputs, we can't keep it fetchable.
      TF_CHECK_OK(NodeBuilder(caller->name(), "NoOp")
                      .Device(caller->requested_device())
                      .ControlInput(output_control_node)
                      .Finalize(g, &caller_substitute_node));

    } else if (options.keep_caller_node == KeepCallerNode::kFetchable) {
      TF_CHECK_OK(NodeBuilder(caller->name(), "IdentityN")
                      .Device(caller->requested_device())
                      .Input(output_tensors)
                      .ControlInput(output_control_node)
                      .Finalize(g, &caller_substitute_node));
    }
  }

  // ------------------------------------------------------------------------ //
  // 'caller' is replaced with inlined function body nodes and maybe IdentityN
  // to keep it fetchable.
  VLOG(3) << "Successfully inlined function call node: " << caller->name();
  g->RemoveNode(caller);

  VLOG(4) << "Final graph: " << g->ToGraphDefDebug().DebugString();

  return absl::OkStatus();
}

bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph,
                           const ExpandInlineFunctionsOptions& options) {
  std::vector<std::pair<Node*, const FunctionBody*>> candidates;

  const FunctionLibraryDefinition* fld = lib->GetFunctionLibraryDefinition();

  for (Node* node : graph->nodes()) {
    // Skip nodes that are not function calls or SymbolicGradient calls.
    if (!IsFunctionCall(*lib->GetFunctionLibraryDefinition(), *node)) {
      continue;
    }
    // Skip function calls that marked noinline.
    bool noinline;
    if (fld->GetAttr(*node, kNoInlineAttr, &noinline).ok() && noinline) {
      VLOG(3) << "noinline: " << SummarizeNode(*node);
      continue;
    }
    FunctionLibraryRuntime::Handle handle;
    absl::Status s = InstantiateFunctionCall(node->def(), lib, &handle);
    if (!s.ok()) {
      LOG(ERROR) << "Failed to instantiate a function:  " << s.message();
      continue;
    }
    const FunctionBody* fbody = lib->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    candidates.emplace_back(node, fbody);
  }

  bool inlined_any = false;
  for (const auto& p : candidates) {
    absl::Status inlined = InlineFunctionBody(*fld, graph, p.first, p.second,
                                              p.first->IsPartitionedCall()
                                                  ? options.multi_device_options
                                                  : options.native_options);
    if (inlined.ok()) {
      inlined_any = true;
    } else {
      VLOG(1) << "Failed to inline function call: node=" << p.first->name()
              << " error=" << inlined.message();
    }
  }

  // TODO(ezhulenev): Release handles for inlined function calls.

  return inlined_any;
}

}  // end namespace tensorflow
