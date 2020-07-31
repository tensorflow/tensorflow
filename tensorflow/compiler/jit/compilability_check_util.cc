/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/compilability_check_util.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

bool HasResourceInput(const Node& node) {
  return absl::c_count(node.input_types(), DT_RESOURCE) != 0;
}

void LogNotCompilable(const Node& node, absl::string_view reason = "") {
  VLOG(3) << "Found uncompilable node " << node.name() << " (op "
          << node.type_string() << ")" << (reason.empty() ? "" : ": ")
          << reason;
}

Status MakeCallNodeFromAttribute(const Node& node, const std::string& attr_name,
                                 NodeDef* node_def) {
  const NameAttrList* name_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), attr_name, &name_attr));
  node_def->set_op(name_attr->name());
  *(node_def->mutable_attr()) = name_attr->attr();
  return Status::OK();
}

}  // anonymous namespace

RecursiveCompilabilityChecker::UncompilableNodesMap
RecursiveCompilabilityChecker::FindUncompilableNodes(
    const Node& node, FunctionLibraryRuntime* lib_runtime,
    const std::vector<RecursiveCompilabilityChecker::StackFrame>*
        node_stack_trace) const {
  std::vector<StackFrameView> stack_trace;
  // If `node_stack_trace` is provided, that means `node` is inside
  // a function body, and therefore, arg nodes and retval nodes are
  // not considered uncompilable.
  if (node_stack_trace != nullptr) {
    for (const auto& frame : *node_stack_trace) {
      stack_trace.emplace_back(StackFrameView{frame.name, frame.function_name});
    }
  }
  stack_trace.emplace_back(StackFrameView{node.name(), ""});

  RecursiveCompilabilityChecker::UncompilableNodesMap uncompilable_nodes;
  IsCompilableNode(node, lib_runtime, &stack_trace,
                   /*encapsulating_function=*/nullptr, &uncompilable_nodes);
  return uncompilable_nodes;
}

RecursiveCompilabilityChecker::UncompilableNodesMap
RecursiveCompilabilityChecker::FindUncompilableNodes(
    const NodeDef& call_def, FunctionLibraryRuntime* lib_runtime,
    const std::vector<RecursiveCompilabilityChecker::StackFrame>*
        node_stack_trace) const {
  // If `node_stack_trace` is provided, that means `call_def` is inside
  // a function body, and therefore, arg nodes and retval nodes are
  // not considered uncompilable.
  std::vector<StackFrameView> stack_trace;
  if (node_stack_trace != nullptr) {
    for (const auto& frame : *node_stack_trace) {
      stack_trace.emplace_back(StackFrameView{frame.name, frame.function_name});
    }
  }
  stack_trace.emplace_back(StackFrameView{call_def.name(), ""});

  RecursiveCompilabilityChecker::UncompilableNodesMap uncompilable_nodes;
  IsCompilableCall(call_def, lib_runtime, &stack_trace,
                   /*encapsulating_function=*/nullptr, &uncompilable_nodes);
  return uncompilable_nodes;
}

bool RecursiveCompilabilityChecker::HasXLAKernel(
    const Node& node, string* uncompilable_reason) const {
  // There is a SymbolicGradient kernel on the XLA_JIT device, but the gradient
  // is really a kind of function call and will be handled by
  // IsCompilableCall().
  if (node.type_string() == "SymbolicGradient") {
    *uncompilable_reason =
        "SymbolicGradient should be handled by IsCompilableCall().";
    return false;
  }
  if (node.type_string() == "Const") {
    // Skip Const op with type DT_STRING, since XLA doesn't support it, but the
    // registered Const KernelDef says that it does, to support no-op Assert for
    // tfcompile.
    const AttrValue* attr = node.attrs().Find("dtype");
    if (attr != nullptr && attr->type() == DT_STRING) {
      *uncompilable_reason =
          "Const op with type DT_STRING is not supported by XLA.";
      return false;
    }
  }

  // XLA does not offer guaranteed aliasing between the input and output of the
  // XLA cluster so it can't implement the forward-tensor-ref semantic.  Leave
  // such nodes out of XLA clusters.
  if (HasForwardedRefInput(node)) {
    VLOG(2) << "Rejecting " << node.name() << ": Identity with unsafe cast.";
    *uncompilable_reason = "Identity with unsafe cast.";
    return false;
  }

  Status s = FindKernelDef(jit_device_type_, node.def(), nullptr, nullptr);
  if (!s.ok()) {
    *uncompilable_reason = s.error_message();
    return false;
  }
  return true;
}

// Tests whether 'if_node' is compilable. Every operator in the then_branch and
// else_branch functions must be compilable for 'if_node' to be compilable.
bool RecursiveCompilabilityChecker::IsCompilableIf(
    const Node& if_node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
  bool is_compilable = true;
  is_compilable &= ExtractNodeDefAndCheckCompilability(
      if_node, "then_branch", "if_then", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);
  if (!uncompilable_nodes && !is_compilable) return is_compilable;

  is_compilable &= ExtractNodeDefAndCheckCompilability(
      if_node, "else_branch", "if_else", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);

  return is_compilable;
}

// Tests whether 'while_node' is a completely compilable loop.
// Every operator in the condition and body functions must be compilable for a
// while loop to be compilable.
bool RecursiveCompilabilityChecker::IsCompilableWhile(
    const Node& while_node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
  bool is_compilable = true;
  is_compilable &= ExtractNodeDefAndCheckCompilability(
      while_node, "cond", "while_cond", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);

  if (!uncompilable_nodes && !is_compilable) return is_compilable;

  is_compilable &= ExtractNodeDefAndCheckCompilability(
      while_node, "body", "while_body", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);

  return is_compilable;
}

bool RecursiveCompilabilityChecker::ExtractNodeDefAndCheckCompilability(
    const Node& node, const std::string& attr_name,
    const std::string& call_name, NameAttrList* encapsulating_function,
    FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
  NodeDef call;
  call.set_name(call_name);
  if (!MakeCallNodeFromAttribute(node, attr_name, &call).ok()) {
    const auto uncompilable_reason = absl::StrCat(
        "missing '", attr_name, "' attribute from node", node.name());
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    VLOG(2) << "Rejecting node " << node.name() << ": " << uncompilable_reason
            << ".";
    return false;
  }
  if (!IsCompilableCall(call, lib_runtime, stack_trace, encapsulating_function,
                        uncompilable_nodes)) {
    VLOG(2) << "Rejecting node " << node.name()
            << ": can't compile : " << call.op();
    return false;
  }
  return true;
}

// Tests whether 'call_def' is a call to a completely compilable function.
// Every operator in the function must be compilable for a function to be
// compilable.
bool RecursiveCompilabilityChecker::IsCompilableCall(
    const NodeDef& call_def, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
  if (stack_trace->size() > kMaxRecursionDepth) {
    std::string uncompilable_reason = "function depth limit exceeded";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    VLOG(2) << "Rejecting " << call_def.op() << ": " << uncompilable_reason
            << ".";
    return false;
  }

  FunctionLibraryRuntime::Handle handle;
  Status s;
  NameAttrList function;
  s = NameAndAttrsFromFunctionCall(call_def, &function);
  if (s.ok()) {
    s = lib_runtime->Instantiate(function.name(), AttrSlice(&function.attr()),
                                 &handle);
  }
  if (!s.ok()) {
    std::string uncompilable_reason =
        absl::StrCat("could not instantiate call: '", function.name(), "'");
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    VLOG(2) << "Rejecting " << call_def.DebugString() << ": "
            << uncompilable_reason << " : " << s;
    return false;
  }

  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });
  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
  bool is_compilable = true;
  for (const Node* node : fbody->graph->op_nodes()) {
    stack_trace->emplace_back(StackFrameView{node->name(), function.name()});
    is_compilable &= IsCompilableNode(*node, lib_runtime, stack_trace,
                                      &function, uncompilable_nodes);
    stack_trace->pop_back();
    if (!uncompilable_nodes && !is_compilable) return is_compilable;
  }

  return is_compilable;
}

bool RecursiveCompilabilityChecker::OpIsInaccurate(const Node& node) const {
  // b/127344411: SelfAdjointEigV2 and Svd precision issues.
  return node.type_string() == "SelfAdjointEigV2" ||
         node.type_string() == "Svd";
}

bool RecursiveCompilabilityChecker::OpIsSlow(const Node& node) const {
  // b/128001705: SelfAdjointEigV2 and Svd performance issues.
  // b/135640736: MatrixInverse performance issues.
  // b/111271662: MatrixSolve performance issues.
  // https://github.com/tensorflow/tensorflow/pull/31012:
  //    ResizeNearestNeighbor, ResizeBilinear, and ResizeBilinearGrad sometimes
  //    create convolutions too large for CuDNN to handle.
  return node.type_string() == "SelfAdjointEigV2" ||
         node.type_string() == "Svd" || node.type_string() == "Qr" ||
         node.type_string() == "MatrixInverse" ||
         node.type_string() == "MatrixSolve" ||
         node.type_string() == "ResizeNearestNeighbor" ||
         node.type_string() == "ResizeBilinear" ||
         node.type_string() == "ResizeBilinearGrad";
}

bool RecursiveCompilabilityChecker::IsCompilableNode(
    const Node& node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
  auto stack_depth = stack_trace->size();
  if (node.IsSource() || node.IsSink()) {
    absl::string_view uncompilable_reason = "source or sink node";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  // _Arg nodes in a top-level function represent feeds and _Retval nodes in a
  // top-level function represent fetches.
  if (stack_depth == 1 &&
      (node.type_string() == "_Arg" || node.type_string() == "_Retval")) {
    absl::string_view uncompilable_reason = "top level _Arg or _Retval";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (node.attrs().Find("_scoped_allocator") ||
      node.attrs().Find("_forward_from")) {
    // TODO(b/128858118): XLA does not support _scoped_allocator and
    // _forward_from.
    absl::string_view uncompilable_reason =
        "_scoped_allocator or _forward_from attribute";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  string uncompilable_reason;
  if (IsFunctionCall(*lib_runtime->GetFunctionLibraryDefinition(), node)) {
    if (!IsCompilableCall(node.def(), lib_runtime, stack_trace,
                          encapsulating_function, uncompilable_nodes)) {
      LogNotCompilable(node, "unsupported function");
      return false;
    }
  } else if (!HasXLAKernel(node, &uncompilable_reason)) {
    MaybeMarkUncompilableNode(
        absl::StrCat("unsupported op: ", uncompilable_reason), *stack_trace,
        encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (node.IsWhileNode() &&
      !IsCompilableWhile(node, lib_runtime, stack_trace, encapsulating_function,
                         uncompilable_nodes)) {
    LogNotCompilable(node, "unsupported while");
    return false;
  }

  if (node.IsIfNode() &&
      !IsCompilableIf(node, lib_runtime, stack_trace, encapsulating_function,
                      uncompilable_nodes)) {
    LogNotCompilable(node, "unsupported if");
    return false;
  }

  if (!op_filter_.allow_stateful_rng_ops &&
      IsStatefulRandomOp(node.type_string())) {
    absl::string_view uncompilable_reason = "stateful random op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_control_trigger && node.IsControlTrigger()) {
    absl::string_view uncompilable_reason = "not allowed control trigger";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_eliding_assert_and_checknumerics_ops &&
      IsAssertOrCheckNumerics(node.type_string())) {
    absl::string_view uncompilable_reason = "Assert or CheckNumerics";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_ops_producing_or_consuming_variant &&
      OpProducesOrConsumesVariant(node)) {
    absl::string_view uncompilable_reason = "DT_VARIANT producer/consumer";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_stack_ops && IsStackOp(node)) {
    absl::string_view uncompilable_reason = "Stack op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_tensor_array_ops && IsTensorArrayOp(node)) {
    absl::string_view uncompilable_reason = "TensorArray op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_resource_ops_in_called_functions && stack_depth > 1 &&
      HasResourceInput(node)) {
    absl::string_view uncompilable_reason =
        "resource variable op in called function";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_inaccurate_ops && OpIsInaccurate(node)) {
    absl::string_view uncompilable_reason =
        "operation with numerical accuracy issues";
    BroadcastOptimizationRemark(XlaOptimizationRemark::INACCURATE_OPERATION,
                                node.DebugString())
        .IgnoreError();
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_slow_ops && OpIsSlow(node)) {
    absl::string_view uncompilable_reason = "slow operation";
    BroadcastOptimizationRemark(XlaOptimizationRemark::SLOW_OPERATION,
                                node.DebugString())
        .IgnoreError();
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  return true;
}

RecursiveCompilabilityChecker::OperationFilter CreateOperationFilter(
    const XlaOpRegistry::DeviceRegistration& registration) {
  RecursiveCompilabilityChecker::OperationFilter op_filter;
  op_filter.allow_resource_ops_in_called_functions =
      registration.cluster_resource_variable_ops_unsafely;
  op_filter.allow_stack_ops = registration.cluster_stack_ops;
  op_filter.allow_tensor_array_ops = registration.cluster_tensor_array_ops;
  op_filter.allow_stateful_rng_ops = registration.cluster_stateful_rng_ops;
  op_filter.allow_control_trigger = registration.cluster_control_trigger;
  op_filter.allow_eliding_assert_and_checknumerics_ops =
      registration.elide_assert_and_checknumerics;
  op_filter.allow_ops_producing_or_consuming_variant =
      registration.cluster_variant_ops;
  op_filter.allow_slow_ops = registration.cluster_slow_ops;
  op_filter.allow_inaccurate_ops = registration.cluster_inaccurate_ops;
  return op_filter;
}

/*static*/ void RecursiveCompilabilityChecker::MaybeMarkUncompilableNode(
    const absl::string_view reason,
    const std::vector<StackFrameView>& stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes) {
  if (!uncompilable_nodes) return;

  UncompilableNodeInfo node_info;
  node_info.uncompilable_reason = std::string(reason);
  absl::c_transform(stack_trace, std::back_inserter(node_info.stack_trace),
                    [](const StackFrameView& stack_element) {
                      return StackFrame{
                          std::string(stack_element.name),
                          std::string(stack_element.function_name)};
                    });

  node_info.name = std::string(stack_trace.back().name);
  auto function =
      encapsulating_function ? *encapsulating_function : NameAttrList();
  auto function_identifier = function.ShortDebugString();

  auto it = uncompilable_nodes->find(function_identifier);
  if (it == uncompilable_nodes->end()) {
    std::vector<RecursiveCompilabilityChecker::UncompilableNodeInfo>
        uncompilable_node_info{std::move(node_info)};
    uncompilable_nodes->emplace(
        std::move(function_identifier),
        std::make_pair(function, std::move(uncompilable_node_info)));
  } else {
    it->second.second.emplace_back(std::move(node_info));
  }
}

bool CanCreateXlaKernel(const NodeDef& node_def) {
  // If kXlaMustCompileAttr is set on the node_def, use its value.
  const auto& it = node_def.attr().find(kXlaMustCompileAttr);
  return it != node_def.attr().end() && it->second.b();
}

Status GetBodyAndConstantsAndResources(FunctionLibraryRuntime* flr,
                                       const NodeDef& node_def,
                                       const FunctionBody** fbody,
                                       std::vector<int>* constant_arg_indices,
                                       std::vector<int>* resource_arg_indices) {
  FunctionLibraryRuntime::Handle handle;
  // If node_def is not instantiable, e.g., the function does not exist,
  // simply bail out.
  NameAttrList function;
  TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node_def, &function));

  TF_RETURN_IF_ERROR(
      flr->Instantiate(function.name(), AttrSlice(&function.attr()), &handle));
  *fbody = flr->GetFunctionBody(handle);
  CHECK(*fbody);  // Can't be nullptr since we just instantiated it.
  const DataTypeVector& arg_types = (*fbody)->arg_types;
  std::vector<bool> const_args(arg_types.size());
  // If we can't analyze the const args. Bail out.
  TF_RETURN_IF_ERROR(
      BackwardsConstAnalysis(*((*fbody)->graph), &const_args,
                             /*compile_time_const_nodes=*/nullptr, flr));

  for (size_t i = 0; i < const_args.size(); ++i) {
    if (const_args[i]) {
      constant_arg_indices->push_back(i);
    }
  }

  // There can be hundreds of resource variables. Reserve the space for them.
  // We don't reserve for constants above as they are usually few.
  resource_arg_indices->reserve(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (arg_types[i] == DT_RESOURCE) {
      resource_arg_indices->push_back(i);
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
