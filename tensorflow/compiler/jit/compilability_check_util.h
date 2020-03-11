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

#ifndef TENSORFLOW_COMPILER_JIT_COMPILABILITY_CHECK_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_COMPILABILITY_CHECK_UTIL_H_

#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
// Checks whether a TF node can be compiled or not.  "Recursive" as in for call
// and functional while nodes it recursively checks whether the callee functions
// can be compiled.
class RecursiveCompilabilityChecker {
 public:
  // Contains node name and function name. If the node is not inside a function
  // body, function name is an empty string.
  struct StackFrame {
    std::string name;
    std::string function_name;
  };

  // Contains information about uncompilable node inside a function body.
  struct UncompilableNodeInfo {
    std::string name;
    // A list representing a stacktrace from the highest level node in
    // increasing call depth to immediate node that fails the
    // compilability checker.
    std::vector<StackFrame> stack_trace;
    std::string uncompilable_reason;
  };

  // Aggregates information about what kinds of ops are allowed.
  struct OperationFilter {  // TODO(lzr): Add AllowEverything() helper.
    // Whether resource variable ops are allowed are allowed in callees.  We do
    // not allow resource variable ops in called functions (either as direct TF
    // calls or as higher order control flow ops) because we do not yet model
    // their memory effects in jit/resource_operation_safety_analysis.
    bool allow_resource_ops_in_called_functions = false;

    // Whether Stack operations are allowed.  We avoid auto-clustering Stack
    // operations in general because we do not support snapshotting them.
    //
    // TODO(b/112837194): This restriction can be lifted with some work.
    bool allow_stack_ops = false;

    // Whether TensorArray operations are allowed.  We avoid auto-clustering
    // TensorArray operations in general because we do not support snapshotting
    // them.
    //
    // TODO(b/112837194): This restriction can be lifted with some work.
    bool allow_tensor_array_ops = false;

    // Whether stateful RNG ops are allowed.  XLA's RNG does not have the same
    // seeding behavior as TensorFlow's RNG (b/34749654).  So we avoid
    // auto-clustering stateful RNG ops.
    bool allow_stateful_rng_ops = false;

    // TODO(b/118970344): Whether ControlTrigger ops are allowed.  It is unsound
    // to cluster ControlTrigger because of how we use deadness analysis.
    bool allow_control_trigger = false;

    // Whether it is okay to "cluster" Assert and CheckNumerics by simply
    // removing them (they're not removed during clustering, but their
    // XlaOpKernel is a no-op kernel).  We avoid auto-clustering these ops so
    // that the user is not surprised when XLA is implicitly enabled. If the
    // user explicitly specifies to use XLA, it is fine to resort to a dummy
    // implementation. Currently Assert and CheckNumerics ops have dummy XLA
    // implementations.
    bool allow_eliding_assert_and_checknumerics_ops = false;

    // Whether ops that produce or consume DT_VARIANT values are allowed.  We
    // don't auto-cluster these ops because we don't yet support live-in or
    // live-out DT_VARIANT values.
    bool allow_ops_producing_or_consuming_variant = false;

    // Whether ops known to be slow on XLA-GPU should be considered compilable.
    bool allow_slow_ops = false;

    // Whether ops known to have numerical accuracy issues should be considered
    // compilable..
    bool allow_inaccurate_ops = false;
  };

  RecursiveCompilabilityChecker(const OperationFilter* op_filter,
                                const DeviceType* jit_device_type)
      : op_filter_(*op_filter), jit_device_type_(*jit_device_type) {}

  using UncompilableNodesMap =
      std::map<std::string,
               std::pair<NameAttrList, std::vector<UncompilableNodeInfo>>>;

  // Returns a map where the key is the function identifier(short debug
  // string) of the function encapsulating the uncompilable nodes, and the
  // value is a pair of NameAttrList of the function and a vector of
  // uncompilable node info. When uncompilable node is not inside any
  // function call nodes, then key is a ShortDebugString() of an empty
  // NameAttrList.
  //
  // Also, when `node` is inside a function body, users can set
  // `node_stack_trace` to provide an additional context for `node`'s
  // placement within the outer most graph.
  UncompilableNodesMap FindUncompilableNodes(
      const Node& node, FunctionLibraryRuntime* lib_runtime,
      const std::vector<StackFrame>* node_stack_trace = nullptr) const;

  // Returns a map where the key is the function identifier(short debug
  // string) of the function encapsulating the uncompilable nodes, and the
  // value is a pair of NameAttrList of the function and a vector of
  // uncompilable node info. When uncompilable node is not inside any
  // function call nodes, then key is a ShortDebugString() of an empty
  // NameAttrList.
  //
  // Also, when `node` is inside a function body, users can set
  // `node_stack_trace` to provide an additional context for `node`'s
  // placement within the outer most graph.
  UncompilableNodesMap FindUncompilableNodes(
      const NodeDef& call_def, FunctionLibraryRuntime* lib_runtime,
      const std::vector<StackFrame>* node_stack_trace = nullptr) const;

  // Returns true if `node` can be compiled by XLA.
  bool IsCompilableNode(const Node& node,
                        FunctionLibraryRuntime* lib_runtime) const {
    std::vector<StackFrameView> stack_trace;
    stack_trace.emplace_back(StackFrameView{node.name(), ""});
    return IsCompilableNode(node, lib_runtime, &stack_trace);
  }

  // Returns true if `call_def` can be compiled by XLA.  It is assumed that
  // `call_def` is a call operation.
  bool IsCompilableCall(const NodeDef& call_def,
                        FunctionLibraryRuntime* lib_runtime) {
    std::vector<StackFrameView> stack_trace;
    stack_trace.emplace_back(StackFrameView{call_def.name(), ""});
    return IsCompilableCall(call_def, lib_runtime, &stack_trace);
  }

  // Returns true if XLA supports this Op, but we don't want to cluster it (ie:
  // due to performance or correctness concerns).
  bool OpIsInaccurate(const Node& node) const;
  bool OpIsSlow(const Node& node) const;

 private:
  struct StackFrameView {
    absl::string_view name;
    absl::string_view function_name;
  };

  bool IsCompilableNode(
      const Node& node, FunctionLibraryRuntime* lib_runtime,
      std::vector<StackFrameView>* stack_trace,
      NameAttrList* encapsulating_function = nullptr,
      UncompilableNodesMap* uncompilable_nodes = nullptr) const;
  bool IsCompilableCall(
      const NodeDef& call_def, FunctionLibraryRuntime* lib_runtime,
      std::vector<StackFrameView>* stack_trace,
      NameAttrList* encapsulating_function = nullptr,
      UncompilableNodesMap* uncompilable_nodes = nullptr) const;
  bool IsCompilableIf(const Node& if_node, FunctionLibraryRuntime* lib_runtime,
                      std::vector<StackFrameView>* stack_trace,
                      NameAttrList* encapsulating_function,
                      UncompilableNodesMap* uncompilable_nodes) const;
  bool IsCompilableWhile(const Node& while_node,
                         FunctionLibraryRuntime* lib_runtime,
                         std::vector<StackFrameView>* stack_trace,
                         NameAttrList* encapsulating_function,
                         UncompilableNodesMap* uncompilable_nodes) const;

  // Returns compilability of node def retrieved from `node`'s attribute with
  // name `attr_name`.
  bool ExtractNodeDefAndCheckCompilability(
      const Node& node, const std::string& attr_name,
      const std::string& call_name, NameAttrList* encapsulating_function,
      FunctionLibraryRuntime* lib_runtime,
      std::vector<StackFrameView>* stack_trace,
      UncompilableNodesMap* uncompilable_nodes) const;

  bool IsStackOp(const Node& node) const {
    const XlaResourceOpInfo* op_info =
        GetResourceOpInfoForOp(node.type_string());
    return op_info && op_info->resource_kind() == XlaResourceKind::kStack;
  }

  bool IsTensorArrayOp(const Node& node) const {
    const XlaResourceOpInfo* op_info =
        GetResourceOpInfoForOp(node.type_string());
    return op_info && op_info->resource_kind() == XlaResourceKind::kTensorArray;
  }

  bool IsAssertOrCheckNumerics(absl::string_view op_name) const {
    return op_name == "Assert" || op_name == "CheckNumerics";
  }

  bool IsStatefulRandomOp(absl::string_view op_name) const {
    return op_name == "RandomUniform" || op_name == "RandomShuffle" ||
           op_name == "RandomUniformInt" || op_name == "RandomStandardNormal" ||
           op_name == "TruncatedNormal" || op_name == "Multinomial";
  }

  bool OpProducesOrConsumesVariant(const Node& node) const {
    auto is_variant = [](DataType dtype) { return dtype == DT_VARIANT; };
    return absl::c_any_of(node.input_types(), is_variant) ||
           absl::c_any_of(node.output_types(), is_variant);
  }

  bool HasXLAKernel(const Node& node,
                    string* uncompilable_reason = nullptr) const;

  static void MaybeMarkUncompilableNode(
      const absl::string_view reason,
      const std::vector<StackFrameView>& stack_trace,
      NameAttrList* encapsulating_function,
      UncompilableNodesMap* uncompilable_nodes_map);

  // Make sure we don't recurse infinitely on recursive functions.
  const int kMaxRecursionDepth = 10;

  const OperationFilter& op_filter_;
  const DeviceType& jit_device_type_;
};

RecursiveCompilabilityChecker::OperationFilter CreateOperationFilter(
    const XlaOpRegistry::DeviceRegistration& registration);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_COMPILABILITY_CHECK_UTIL_H_
