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
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/union_find.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
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
    std::shared_ptr<AbstractStackTrace> stack_trace;
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

    // Require the function to be always compilable, regardless whether some
    // control flow branches might be dead for a given input.
    bool require_always_compilable = false;

    // Whether string constants are compilable.
    bool allow_string_consts = true;

    // Whether to allow the compilation of CollectiveReduceV2Op.
    bool allow_collective_reduce_v2 = true;

    // Whether to allow the compilation of UniqueOp. Compilation of the UniqueOp
    // generates output with bounded dynamic shape that may cause failures with
    // auto clustering.
    // TODO(b/209813421): Enable tf.unique during
    // autoclustering once all failures are rfixed.
    bool allow_unique_op = true;

    // Whether ops that are marked as outside compiled are always considered
    // compilable.
    // TODO(b/191502757):  Make this behavior true by default and remove this
    // option once inference converter supports outside compilation.
    bool allow_outside_compiled = false;
  };

  RecursiveCompilabilityChecker(OperationFilter op_filter,
                                DeviceType jit_device_type)
      : op_filter_(std::move(op_filter)),
        jit_device_type_(std::move(jit_device_type)) {}

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

  // Returns true if `node` can be compiled by XLA.
  bool IsCompilableNode(const Node& node,
                        FunctionLibraryRuntime* lib_runtime) const {
    std::vector<StackFrameView> stack_trace;
    stack_trace.emplace_back(StackFrameView{node.name(), ""});
    return IsCompilableNode(node, lib_runtime, &stack_trace);
  }

  // Returns true if XLA supports this Op, but we don't want to cluster it (ie:
  // due to performance or correctness concerns).
  bool OpIsInaccurate(const Node& node) const;
  bool OpIsSlow(const Node& node) const;

 private:
  struct StackFrameView {
    absl::string_view name;
    absl::string_view function_name;
    std::shared_ptr<AbstractStackTrace> stack_trace;
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

  // Tests whether 'case_node' is compilable. Every operator in all branches
  // must be compilable.
  bool IsCompilableCase(const Node& case_node,
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
  const size_t kMaxRecursionDepth = 50;

  const OperationFilter op_filter_;
  const DeviceType jit_device_type_;
};

RecursiveCompilabilityChecker::OperationFilter CreateOperationFilter(
    const XlaOpRegistry::DeviceRegistration& registration);

// Given a FunctionLibraryRuntime and a `function`, returns this function's body
// in `fbody` as well as the indices of its constant and resource arguments.
// `fbody` is owned by `flr`.
// `constant_arg_indices` and `resource_arg_indices` should be empty vector.
// They are sorted in ascending order on this function's return.
Status GetBodyAndConstantsAndResources(FunctionLibraryRuntime* flr,
                                       const NameAttrList& function,
                                       const FunctionBody** fbody,
                                       std::vector<int>* constant_arg_indices,
                                       std::vector<int>* resource_arg_indices);

// Given a NodeDef `node_def` returns true iff `node_def` has kXlaCompileAttr
// set.
bool CanCreateXlaKernel(const NodeDef& node_def);

// Returns memory types for the input.
// `constant_arg_indices` and `resource_arg_indices` are sorted arrays of
// indices corresponding to constant and resource arguments respectively.
//
// One might wonder, about the case where a compile-time constant argument
// (which must be in host memory) is also used as an input into an op,
// e.g. `Add`, that expects its inputs in device memory. Here is how it
// works now.
// First, what do we mean by "op expects an input in XYZ memory"?
// There are two types of "ops" here: the tf2xla kernel and the HLO
// computation it builds. The tf2xla kernel needs to retrieve the actual
// numeric value of the compile-time constant tensors, so it really expects
// them to be on in host memory. However, for other inputs, it refers to them
// using xla::ComputationDataHandle, which is just a symbolic handle that
// xla::ComputationBuilder assigns. How does this handle gets assigned for
// constant arguments? Even constant arguments get an _Arg node in the graph
// instantiated for Function compilation. The tf2xla kernel for constant _Arg
// nodes takes the constant value, converts it to XlaLiteral, and feeds it
// to xla::ComputationBuilder.ConstantLiteral, which returns the handle. This
// constant XlaLiteral is included in the HLO graph, and subsequently, in
// the actual executable, which is copied to the device before being
// executed. Thus, when this executable runs, the constant is available in
// device memory.
tensorflow::MemoryTypeVector GetInputMemoryTypes(
    const tensorflow::FunctionBody* fbody,
    absl::Span<int const> constant_arg_indices,
    absl::Span<int const> resource_arg_indices);

// Returns output memory types.
//
// XlaLaunch kernel keeps all outputs (including constants, which it copies),
// in device memory except for resources.
tensorflow::MemoryTypeVector GetOutputMemoryTypes(
    const tensorflow::FunctionBody* fbody);

// Check whether graph can trigger XLA compilation.
bool CanTriggerXlaCompilation(const GraphDef& graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_COMPILABILITY_CHECK_UTIL_H_
