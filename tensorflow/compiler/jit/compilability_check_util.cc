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
#include <unordered_map>
#include <unordered_set>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_constructor.h"
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

void MarkInSummary(
    const absl::string_view function_name, const absl::string_view reason,
    const std::vector<absl::string_view>& stack_trace,
    RecursiveCompilabilityChecker::CompilabilityCheckerSummary* summary) {
  RecursiveCompilabilityChecker::UncompilableNodeInfo node_info;
  node_info.uncompilable_reason = std::string(reason);
  absl::c_transform(
      stack_trace, std::back_inserter(node_info.stack_trace),
      [](const absl::string_view node) { return std::string(node); });

  node_info.name = std::string(stack_trace.back());
  (*summary)[std::string(function_name)].emplace_back(std::move(node_info));
}

void MaybeMarkNodeInSummary(
    const Node& node,
    const absl::optional<absl::string_view> containing_function_name,
    const absl::string_view reason,
    const std::vector<absl::string_view>& stack_trace,
    RecursiveCompilabilityChecker::CompilabilityCheckerSummary* summary) {
  if (!summary) return;

  const auto& node_function_name =
      containing_function_name ? *containing_function_name : "";
  MarkInSummary(node_function_name, reason, stack_trace, summary);
}

void MaybeMarkNodeDefToSummary(
    const NodeDef& node_def, const std::string& reason,
    const std::vector<absl::string_view>& stack_trace,
    RecursiveCompilabilityChecker::CompilabilityCheckerSummary* summary) {
  if (!summary) return;

  const auto& call_op = node_def.op();
  MarkInSummary(call_op, reason, stack_trace, summary);
}

}  // anonymous namespace

bool RecursiveCompilabilityChecker::HasXLAKernel(const Node& node) {
  // There is a SymbolicGradient kernel on the XLA_JIT device, but the gradient
  // is really a kind of function call and will be handled by
  // IsCompilableCall().
  if (node.type_string() == "SymbolicGradient") return false;
  if (node.type_string() == "Const") {
    // Skip Const op with type DT_STRING, since XLA doesn't support it, but the
    // registered Const KernelDef says that it does, to support no-op Assert for
    // tfcompile.
    const AttrValue* attr = node.attrs().Find("dtype");
    if (attr != nullptr && attr->type() == DT_STRING) {
      return false;
    }
  }

  // XLA does not offer guaranteed aliasing between the input and output of the
  // XLA cluster so it can't implement the forward-tensor-ref semantic.  Leave
  // such nodes out of XLA clusters.
  if (HasForwardedRefInput(node)) {
    VLOG(2) << "Rejecting " << node.name() << ": Identity with unsafe cast.";
    return false;
  }

  return FindKernelDef(jit_device_type_, node.def(), nullptr, nullptr).ok();
}

// Tests whether 'while_node' is a completely compilable loop.
// Every operator in the condition and body functions must be compilable for a
// while loop to be compilable.
bool RecursiveCompilabilityChecker::IsCompilableWhile(
    const Node& while_node, FunctionLibraryRuntime* lib_runtime,
    std::vector<absl::string_view>* stack_trace,
    CompilabilityCheckerSummary* summary) {
  const NameAttrList* name_attr;
  NodeDef call;
  Status status;
  status = GetNodeAttr(while_node.attrs(), "cond", &name_attr);
  if (!status.ok()) {
    const std::string uncompilable_reason =
        "missing 'cond' attribute on While node";
    MaybeMarkNodeInSummary(while_node,
                           /*containing_function_name=*/absl::nullopt,
                           uncompilable_reason, *stack_trace, summary);
    VLOG(2) << "Rejecting While " << while_node.name() << ": "
            << uncompilable_reason << ".";
    return false;
  }
  const string cond_func = name_attr->name();
  call.set_name("while_cond");
  call.set_op(cond_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, lib_runtime, stack_trace, summary)) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": can't compile loop condition: " << cond_func;
    return false;
  }
  status = GetNodeAttr(while_node.attrs(), "body", &name_attr);
  if (!status.ok()) {
    const std::string uncompilable_reason =
        "missing 'body' attribute on While node";
    MaybeMarkNodeInSummary(while_node,
                           /*containing_function_name=*/absl::nullopt,
                           uncompilable_reason, *stack_trace, summary);
    VLOG(2) << "Rejecting While " << while_node.name() << ": "
            << uncompilable_reason << ".";
    return false;
  }
  const string body_func = name_attr->name();
  call.set_name("while_body");
  call.set_op(body_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, lib_runtime, stack_trace, summary)) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": can't compile loop body: " << body_func;
    return false;
  }
  return true;
}

// Tests whether 'call_def' is a call to a completely compilable function.
// Every operator in the function must be compilable for a function to be
// compilable.
bool RecursiveCompilabilityChecker::IsCompilableCall(
    const NodeDef& call_def, FunctionLibraryRuntime* lib_runtime,
    std::vector<absl::string_view>* stack_trace,
    CompilabilityCheckerSummary* summary) {
  if (stack_trace->size() > kMaxRecursionDepth) {
    std::string uncompilable_reason = "function depth limit exceeded";
    MaybeMarkNodeDefToSummary(call_def, uncompilable_reason, *stack_trace,
                              summary);
    VLOG(2) << "Rejecting " << call_def.op() << ": " << uncompilable_reason
            << ".";
    return false;
  }

  FunctionLibraryRuntime::Handle handle;
  Status status = InstantiateFunctionCall(call_def, lib_runtime, &handle);
  if (!status.ok()) {
    std::string uncompilable_reason = "could not instantiate call";
    MaybeMarkNodeDefToSummary(call_def, uncompilable_reason, *stack_trace,
                              summary);
    VLOG(2) << "Rejecting " << call_def.DebugString() << ": "
            << uncompilable_reason << " : " << status;
    return false;
  }

  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });
  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
  bool is_compilable = true;
  for (const Node* node : fbody->graph->op_nodes()) {
    stack_trace->push_back(node->name());
    is_compilable &= IsCompilableNode(*node, lib_runtime, stack_trace, summary,
                                      call_def.op());
    if (is_compilable) stack_trace->pop_back();
    if (!summary && !is_compilable) return is_compilable;
  }

  return is_compilable;
}

bool RecursiveCompilabilityChecker::OpIsInaccurate(const Node& node) {
  // b/127344411: SelfAdjointEigV2 and Svd precision issues.
  return node.type_string() == "SelfAdjointEigV2" ||
         node.type_string() == "Svd";
}

bool RecursiveCompilabilityChecker::OpIsSlow(const Node& node) {
  // b/128001705: SelfAdjointEigV2 and Svd performance issues.
  return node.type_string() == "SelfAdjointEigV2" ||
         node.type_string() == "Svd" || node.type_string() == "Qr";
}

bool RecursiveCompilabilityChecker::IsCompilableNode(
    const Node& node, FunctionLibraryRuntime* lib_runtime,
    std::vector<absl::string_view>* stack_trace,
    CompilabilityCheckerSummary* summary,
    const absl::optional<absl::string_view> containing_function_name) {
  auto stack_depth = stack_trace->size();
  if (node.IsSource() || node.IsSink()) {
    absl::string_view uncompilable_reason = "source or sink node";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  // _Arg nodes in a top-level function represent feeds and _Retval nodes in a
  // top-level function represent fetches.
  if (stack_depth == 1 &&
      (node.type_string() == "_Arg" || node.type_string() == "_Retval")) {
    absl::string_view uncompilable_reason = "top level _Arg or _Retval";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (node.attrs().Find("_scoped_allocator") ||
      node.attrs().Find("_forward_from")) {
    // TODO(b/128858118): XLA does not support _scoped_allocator and
    // _forward_from.
    absl::string_view uncompilable_reason =
        "_scoped_allocator or _forward_from attribute";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (IsFunctionCall(*lib_runtime->GetFunctionLibraryDefinition(), node)) {
    if (!IsCompilableCall(node.def(), lib_runtime, stack_trace, summary)) {
      LogNotCompilable(node, "unsupported function");
      return false;
    }
  } else if (!HasXLAKernel(node)) {
    absl::string_view uncompilable_reason = "unsupported op";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (node.type_string() == "While" &&
      !IsCompilableWhile(node, lib_runtime, stack_trace, summary)) {
    LogNotCompilable(node, "unsupported while");
    return false;
  }

  if (!op_filter_.allow_stateful_rng_ops &&
      IsStatefulRandomOp(node.type_string())) {
    absl::string_view uncompilable_reason = "stateful random op";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_control_trigger && node.IsControlTrigger()) {
    absl::string_view uncompilable_reason = "not allowed control trigger";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_eliding_assert_and_checknumerics_ops &&
      IsAssertOrCheckNumerics(node.type_string())) {
    absl::string_view uncompilable_reason = "Assert or CheckNumerics";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_ops_producing_or_consuming_variant &&
      OpProducesOrConsumesVariant(node)) {
    absl::string_view uncompilable_reason = "DT_VARIANT producer/consumer";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_stack_ops && IsStackOp(node)) {
    absl::string_view uncompilable_reason = "Stack op";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_tensor_array_ops && IsTensorArrayOp(node)) {
    absl::string_view uncompilable_reason = "TensorArray op";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_resource_ops_in_called_functions && stack_depth > 1 &&
      HasResourceInput(node)) {
    absl::string_view uncompilable_reason =
        "resource variable op in called function";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_inaccurate_ops && OpIsInaccurate(node)) {
    absl::string_view uncompilable_reason =
        "operation with numerical accuracy issues";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_slow_ops && OpIsSlow(node)) {
    absl::string_view uncompilable_reason = "slow operation";
    MaybeMarkNodeInSummary(node, containing_function_name, uncompilable_reason,
                           *stack_trace, summary);
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


}  // namespace tensorflow
