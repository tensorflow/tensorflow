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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_INLINE_FUNCTION_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_INLINE_FUNCTION_UTILS_H_

#include <functional>
#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

static constexpr const char* const kNoInlineAttr = "_noinline";

// Optionally override device assignment for nodes added to the graph for
// inlined functions:
// (1) Identity nodes added in place of function input arguments.
// (2) Identity nodes added in place of function return values.
// (3) Special NoOp nodes that enforce side-effects execution order.
// (4) All nodes inside function body specified in FunctionDef.
class InlinedFunctionBodyPlacer {
 public:
  virtual ~InlinedFunctionBodyPlacer() = default;

  virtual absl::optional<string> InputNodeDevice(int input_index) const = 0;
  virtual absl::optional<string> OutputNodeDevice(int output_index) const = 0;
  // Returns true if the added input/output identity nodes should be colocated
  // with the corresponding input/output from the function body.
  virtual bool ColocateInputOutputIdentities() const = 0;
  virtual absl::optional<string> ControlNodeDevice() const = 0;
  virtual absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const = 0;

  // LINT.IfChange
  // Place input nodes on the same device as the corresponding caller input
  // node. Do not specify any placement for all other nodes.
  static std::unique_ptr<InlinedFunctionBodyPlacer> DefaultPlacer(
      const Graph& graph, const Node& caller);

  // Place all nodes on the same device as caller node.
  static std::unique_ptr<InlinedFunctionBodyPlacer> SingleDevicePlacer(
      const Graph& graph, const Node& caller);

  // Place input nodes on the same device as the corresponding caller input
  // node. Do not place output node. Place control nodes on the same device as
  // caller node. For all function body nodes set job, replica and task
  // parts of the device assignment to match function caller node where those
  // are unspecified.
  static std::unique_ptr<InlinedFunctionBodyPlacer> MultiDevicePlacer(
      const Graph& graph, const Node& caller);
  // LINT.ThenChange(lower_function_call_inline_policy.h)

  using Factory = std::function<std::unique_ptr<InlinedFunctionBodyPlacer>(
      const Graph&, const Node&)>;

  struct Config {
    string name;
    Factory get;
  };

  static Config Default() { return {"default", DefaultPlacer}; }
  static Config SingleDevice() { return {"single_device", SingleDevicePlacer}; }
  static Config MultiDevice() { return {"multi_device", MultiDevicePlacer}; }
};

struct InlineFunctionBodyOptions {
  // All nodes that have incoming control edge *from* the function call node,
  // will be forwarded to the "output control node". There are two options for
  // choosing which nodes will have a control edge *to* the "output control
  // node":
  //   a) control returns            (`control_ret` field in FunctionDef)
  //   b) data returns               (`ret` field in FunctionDef)
  enum class OutputControlSource { kDataOutputs, kControlOutputs };

  // Keep a node in a graph with the same name as the function call node:
  //
  // a) DoNotKeep: Function call node is fully inlined, and there is no node in
  //    a graph with the same name.
  //
  // b) Fetchable: Add an IdentityN node to the graph in place of the inlined
  //    function call node. It will have a control edge from inlined
  //    'output_control_node' and data edges from function output nodes.
  //    The IdentityN node will be placed on the same device as the caller node.
  //
  //    This is mostly for compatibility with Tensorflow v1 and sessions.
  //    When we prepare a graph for execution in
  //    GraphExecutionState::MakeForBaseGraph we don't know what nodes will be
  //    fetched, so we can't safely remove any of them. When graph executed as a
  //    function it has 'Retval' nodes for all fetched tensors, and we can
  //    safely inline function calls.
  //
  // c) Targetable: Add a NoOp node to the graph in place of the inlined
  //    function call node. It will have a control edge from inline
  //    'output_control_node' and no data edges. NoOp node will be placed on the
  //    same device as the caller node. This will keep the inlined function call
  //    node a valid 'session.run' target, and also will keep it a valid control
  //    output node.
  enum class KeepCallerNode { kDoNotKeep, kFetchable, kTargetable };

  // If 'true' function inlining is completely disabled. This allows to control
  // function inlining for different types of function calls (see
  // 'ExpandInlineFunctionsOptions' below).
  bool disable_inlining = false;
  // Ignore '_noinline' function attribute.
  bool ignore_noinline = false;
  // If 'true' function inlining will inline functions in implementation
  // selection group. Normally those functions should not be inlined; they will
  // be handled by Grappler.
  bool inline_impl_selection_group_functions = false;
  // Controls if we want to keep a node with the name as the function call node
  // in a graph after function inlining.
  KeepCallerNode keep_caller_node = KeepCallerNode::kDoNotKeep;
  // For compatibility with Tensorflow v1 by default we will use data outputs.
  // Control returns were added to Tensorflow v2 with automatic control
  // dependencies tracking in Eager mode.
  OutputControlSource output_control_src = OutputControlSource::kDataOutputs;
  // Inlined function body placer decides what requested device assignments
  // should be added to the nodes added to the graph. See documentation above
  // for available strategies.
  InlinedFunctionBodyPlacer::Config inlined_function_body_placer =
      InlinedFunctionBodyPlacer::Default();
  // If true, frame names in the function body will be
  // made unique in the resulting graph (e.g. by prepending a unique prefix).
  // NOTE(mrry): Only set this option to false when there is a single function
  // call in the graph (e.g. when making a remote function call via
  // ClusterFunctionLibraryRuntime). This option is provided because the graph
  // partitioner generates frame names that must remain unmodified across all
  // partitions of a multi-device function.
  bool uniquify_frame_names = true;

  // A human-readable debug string for this options.
  string DebugString() const;
};

// Returns 'OkStatus()' iff the function '*fbody' can be inlined at 'node'
// based on the type signature of 'node' and 'fbody':
//
// (1) Caller node has the same number of inputs and outputs as the function.
// (2) Caller node inputs and outputs have the same data types as function
//     inputs and returns.
// (3) Validation rules defined in InlineFunctionBodyOptions.
//
// If function can't be safely inlined, returns error message with details why
// inlining is not possible or safe.
absl::Status ValidateInlining(const Node* node, const FunctionBody* fbody,
                              const InlineFunctionBodyOptions& options);

// Given a "caller" in graph "g", which is a function call of a function
// to "fbody". Replaces the "caller" with fbody->graph and connects
// edges properly. "override_device" specifies whether inlining should replace
// explicitly specified devices inside fbody with the callee's device.
//
// Returns 'OkStatus()' if function was successfully inlined into the graph.
// If function inlining is not possible returns an error with a reason, and
// leaves the graph in unmodified state.
absl::Status InlineFunctionBody(const FunctionLibraryDefinition& flib_def,
                                Graph* g, Node* caller,
                                const FunctionBody* fbody,
                                const InlineFunctionBodyOptions& options);

// There are three types of function calls that could be invoked during
// *Tensorflow graph execution*:
//
// 1) Native function call (node.type_string() is the function name). These
//    functions are always executed on a single-device, which is the device of
//    the function call node.
//
// 2) Multi-device function calls (PartitionedCall or StatefulPartitionedCall
//    ops) can execute on multiple devices and accept DT_RESOURCE inputs that
//    belong to different devices. This type of functions was added in
//    Tensorflow 2.0 Eager mode, and it has control outputs to represent
//    side-effects that must always execute (see `control_ret` in FunctionDef).
//
// 3) SymbolicGradient has been deprecated for a while, but we still keep it and
//    use `native` options for inlining for compatibility.
//
// We need to have distinct inlining rules for compatibility with Tensorflow v1.
//
// There are few other places in Tensorflow that could execute functions:
//
// 1) common_runtime/eager/kernel_and_device.{h,cc} - executes "top level"
//    functions directly via function library runtime, without going through
//    the graph.
// 2) tf.data pipelines - also execute functions directly via function library
//    runtime with custom executors.
struct ExpandInlineFunctionsOptions {
  ExpandInlineFunctionsOptions() : native_options(), multi_device_options() {
    using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;
    multi_device_options.output_control_src = OutputControlSrc::kControlOutputs;
  }

  InlineFunctionBodyOptions native_options;
  InlineFunctionBodyOptions multi_device_options;
};

// WARNING(ezhulenev): PLEASE DO NOT USE THIS FUNCTION. This is a temporary
// workaround that will be enabled only during the function inlining unification
// (b/126811947). Contact ezhulenev@ if you think you need it.
// TODO(ezhulenev): Delete this function.
bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph,
                           const ExpandInlineFunctionsOptions& options);

// For each node in "graph", if "lib" indicates that the node is a
// function call, inline the function body. Returns true if at least
// one node is inlined.
//
// This routine goes through "graph" nodes once and applies the
// inlining. The caller may decide to apply the inlining on "graph"
// multiple times by calling ExpandInlineFunctions a few times.
//
// Function calls that can't be safely inlined into the graph (ValidateInlining
// returns error), are ignored.
//
// TODO(ezhulenev): We do not FunctionLibraryRuntime for this. We need just the
// FunctionLibraryDefinition and FunctionDefToBodyHelper to implement this (see
// lower_function_call.cc).
inline bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph) {
  return ExpandInlineFunctions(lib, graph, ExpandInlineFunctionsOptions());
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_INLINE_FUNCTION_UTILS_H_
