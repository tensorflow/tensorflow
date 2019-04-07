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

#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"

#include <atomic>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_info_cache.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
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
using xla::StatusOr;

bool HasResourceOutput(const Node& node) {
  return absl::c_count(node.output_types(), DT_RESOURCE) != 0;
}

bool HasResourceInput(const Node& node) {
  return absl::c_count(node.input_types(), DT_RESOURCE) != 0;
}

// The clusters we create here are eventually lowered into an
// _XlaCompile/_XlaRun pair with a TF executor "fallback" that uses the
// PartitionedCall op to execute the cluster in the regular graph executor if
// need be.  PartitionedCall, however, reruns the entire TF graph optimization
// pipeline over the cluster which includes this mark for compilation pass.  To
// avoid endlessly recursing we tag nodes that we've already visited with this
// attribute so that we can bail out if we see them a second time.
//
// TODO(sanjoy): This method is not robust since it is possible that the
// optimizations run by PartitionedCall can mutate the cluster arbitrarily,
// dropping the kXlaAlreadyClustered attributes from all nodes in the process.
// The correct fix is to use the ConfigProto to pass in some sort of flag into
// the PartitionedCall kernel that tells it to not rerun auto-clustering on the
// cluster.
const char* kXlaAlreadyClustered = "_XlaAlreadyClustered";

// Checks whether a TF node can be compiled or not.  "Recursive" as in for call
// and functional while nodes it recursively checks whether the callee functions
// can be compiled.
class RecursiveCompilabilityChecker {
 public:
  // Aggregates information about what kinds of ops are allowed.
  struct OperationFilter {
    // Whether resource variable ops are allowed are allowed in callees.  We do
    // not allow resource variable ops in called functions (either as direct TF
    // calls or as higher order control flow ops) because we do not yet model
    // their memory effects in jit/resource_variable_safety_analysis.
    bool allow_resource_ops_in_called_functions;

    // Whether Stack operations are allowed.  We avoid auto-clustering Stack
    // operations in general because we do not support snapshotting them.
    //
    // TODO(b/112837194): This restriction can be lifted with some work.
    bool allow_stack_ops;

    // Whether TensorArray operations are allowed.  We avoid auto-clustering
    // TensorArray operations in general because we do not support snapshotting
    // them.
    //
    // TODO(b/112837194): This restriction can be lifted with some work.
    bool allow_tensor_array_ops;

    // Whether stateful RNG ops are allowed.  XLA's RNG does not have the same
    // seeding behavior as TensorFlow's RNG (b/34749654).  So we avoid
    // auto-clustering stateful RNG ops.
    bool allow_stateful_rng_ops;

    // TODO(b/118970344): Whether ControlTrigger ops are allowed.  It is unsound
    // to cluster ControlTrigger because of how we use deadness analysis.
    bool allow_control_trigger;

    // Whether it is okay to "cluster" Assert and CheckNumerics by simply
    // removing them (they're not removed during clustering, but their
    // XlaOpKernel is a no-op kernel).  We avoid auto-clustering these ops so
    // that the user is not surprised when XLA is implicitly enabled. If the
    // user explicitly specifies to use XLA, it is fine to resort to a dummy
    // implementation. Currently Assert and CheckNumerics ops have dummy XLA
    // implementations.
    bool allow_eliding_assert_and_checknumerics_ops;

    // Whether ops that produce or consume DT_VARIANT values are allowed.  We
    // don't auto-cluster these ops because we don't yet support live-in or
    // live-out DT_VARIANT values.
    bool allow_ops_producing_or_consuming_variant;
  };

  RecursiveCompilabilityChecker(const OperationFilter* op_filter,
                                const DeviceType* jit_device_type)
      : op_filter_(*op_filter), jit_device_type_(*jit_device_type) {}

  // Returns true if `node` can be compiled by XLA.
  bool IsCompilableNode(const Node& node, FunctionLibraryRuntime* lib_runtime) {
    return IsCompilableNode(node, /*depth=*/0, lib_runtime);
  }

  // Returns true if `call_def` can be compiled by XLA.  It is assumed that
  // `call_def` is a call operation.
  bool IsCompilableCall(const NodeDef& call_def,
                        FunctionLibraryRuntime* lib_runtime) {
    return IsCompilableCall(call_def, /*depth=*/0, lib_runtime);
  }

 private:
  bool IsCompilableNode(const Node& node, int depth,
                        FunctionLibraryRuntime* lib_runtime);
  bool IsCompilableCall(const NodeDef& call_def, int depth,
                        FunctionLibraryRuntime* lib_runtime);
  bool IsCompilableWhile(const Node& while_node, int depth,
                         FunctionLibraryRuntime* lib_runtime);

  bool IsStackOp(const Node& node) {
    const XlaResourceOpInfo* op_info =
        GetResourceOpInfoForOp(node.type_string());
    return op_info && op_info->resource_kind() == XlaResourceKind::kStack;
  }

  bool IsTensorArrayOp(const Node& node) {
    const XlaResourceOpInfo* op_info =
        GetResourceOpInfoForOp(node.type_string());
    return op_info && op_info->resource_kind() == XlaResourceKind::kTensorArray;
  }

  bool IsAssertOrCheckNumerics(absl::string_view op_name) {
    return op_name == "Assert" || op_name == "CheckNumerics";
  }

  bool IsStatefulRandomOp(absl::string_view op_name) {
    return op_name == "RandomUniform" || op_name == "RandomShuffle" ||
           op_name == "RandomUniformInt" || op_name == "RandomStandardNormal" ||
           op_name == "TruncatedNormal" || op_name == "Multinomial";
  }

  bool OpProducesOrConsumesVariant(const Node& node) {
    auto is_variant = [](DataType dtype) { return dtype == DT_VARIANT; };
    return absl::c_any_of(node.input_types(), is_variant) ||
           absl::c_any_of(node.output_types(), is_variant);
  }

  bool HasXLAKernel(const Node& node);

  // Make sure we don't recurse infinitely on recursive functions.
  const int kMaxRecursionDepth = 10;

  const OperationFilter& op_filter_;
  const DeviceType& jit_device_type_;
};

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
    const Node& while_node, int depth, FunctionLibraryRuntime* lib_runtime) {
  const NameAttrList* name_attr;
  NodeDef call;
  Status status;
  status = GetNodeAttr(while_node.attrs(), "cond", &name_attr);
  if (!status.ok()) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": missing 'cond' attribute on While node.";
    return false;
  }
  const string cond_func = name_attr->name();
  call.set_name("while_cond");
  call.set_op(cond_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, depth + 1, lib_runtime)) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": can't compile loop condition: " << cond_func;
    return false;
  }
  status = GetNodeAttr(while_node.attrs(), "body", &name_attr);
  if (!status.ok()) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": missing 'body' attribute on While node.";
    return false;
  }
  const string body_func = name_attr->name();
  call.set_name("while_body");
  call.set_op(body_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, depth + 1, lib_runtime)) {
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
    const NodeDef& call_def, int depth, FunctionLibraryRuntime* lib_runtime) {
  if (depth > kMaxRecursionDepth) {
    VLOG(2) << "Rejecting " << call_def.op()
            << ": function depth limit exceeded.";
    return false;
  }

  FunctionLibraryRuntime::Handle handle;
  Status status = InstantiateFunctionCall(call_def, lib_runtime, &handle);
  if (!status.ok()) {
    VLOG(2) << "Rejecting " << call_def.DebugString()
            << ": could not instantiate: " << status;
    return false;
  }

  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });

  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
  CHECK(fbody);
  for (Node* node : fbody->graph->op_nodes()) {
    if (!IsCompilableNode(*node, depth + 1, lib_runtime)) {
      return false;
    }
  }

  return true;
}

bool LogNotCompilableAndReturn(const Node& node,
                               absl::string_view reason = "") {
  VLOG(3) << "Not clustering " << node.name() << " (op " << node.type_string()
          << ")" << (reason.empty() ? "" : ": ") << reason;
  return false;
}

bool RecursiveCompilabilityChecker::IsCompilableNode(
    const Node& node, int depth, FunctionLibraryRuntime* lib_runtime) {
  // _Arg nodes in a top-level function represent feeds and _Retval nodes in a
  // top-level function represent fetches.
  if (depth == 0 &&
      (node.type_string() == "_Arg" || node.type_string() == "_Retval")) {
    return LogNotCompilableAndReturn(node, "depth is 0");
  }

  if (node.attrs().Find("_scoped_allocator") ||
      node.attrs().Find("_forward_from")) {
    // TODO(b/128858118): XLA does not support _scoped_allocator and
    // _forward_from.
    return LogNotCompilableAndReturn(
        node, "_scoped_allocator or _forward_from attribute");
  }

  if (IsFunctionCall(*lib_runtime->GetFunctionLibraryDefinition(), node)) {
    if (!IsCompilableCall(node.def(), depth + 1, lib_runtime)) {
      return LogNotCompilableAndReturn(node, "unsupported function");
    }
  } else if (!HasXLAKernel(node)) {
    return LogNotCompilableAndReturn(node, "unsupported op");
  }

  if (node.type_string() == "While" &&
      !IsCompilableWhile(node, depth + 1, lib_runtime)) {
    return LogNotCompilableAndReturn(node, "unsupported while");
  }

  if (!op_filter_.allow_stateful_rng_ops &&
      IsStatefulRandomOp(node.type_string())) {
    return LogNotCompilableAndReturn(node, "stateful random op");
  }

  if (!op_filter_.allow_control_trigger && node.IsControlTrigger()) {
    return LogNotCompilableAndReturn(node);
  }

  if (!op_filter_.allow_eliding_assert_and_checknumerics_ops &&
      IsAssertOrCheckNumerics(node.type_string())) {
    return LogNotCompilableAndReturn(node, "Assert or CheckNumerics");
  }

  if (!op_filter_.allow_ops_producing_or_consuming_variant &&
      OpProducesOrConsumesVariant(node)) {
    return LogNotCompilableAndReturn(node, "DT_VARIANT producer/consumer");
  }

  if (!op_filter_.allow_stack_ops && IsStackOp(node)) {
    return LogNotCompilableAndReturn(node, "Stack op");
  }

  if (!op_filter_.allow_tensor_array_ops && IsTensorArrayOp(node)) {
    return LogNotCompilableAndReturn(node, "TensorArray op");
  }

  if (!op_filter_.allow_resource_ops_in_called_functions && depth > 0 &&
      HasResourceInput(node)) {
    return LogNotCompilableAndReturn(node,
                                     "resource variable op in called function");
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
  return op_filter;
}

class MarkForCompilationPassImpl {
 public:
  struct DebugOptions {
    // If true, do not respect the results of deadness analysis.
    bool ignore_deadness_checks;

    // If true, do not respect the _XlaCompile=false attribute.
    bool ignore_xla_compile_attr;

    int max_cluster_size;
    int min_cluster_size;

    // Compiler fuel for the auto-clustering algorithm.
    //
    // We decrement this value by one on every time we choose a compilation
    // candidate and we stop clustering when it hits zero.  This means the
    // initial value for this variable (via --tf_xla_clustering_fuel=N)
    // effectively acts as a "cap" for how much we cluster and we can bisect
    // over this initial value to discover clustering decisions that cause a
    // miscompile or a performance regression.
    std::atomic<int64>* fuel;

    bool dump_graphs;
  };

  MarkForCompilationPassImpl(DebugOptions debug_options, Graph* graph,
                             FunctionLibraryDefinition* flib_def, Env* env,
                             OptimizerOptions::GlobalJitLevel global_jit_level)
      : debug_options_(debug_options),
        graph_(graph),
        flib_def_(flib_def),
        env_(env),
        global_jit_level_(global_jit_level) {}

  Status Run();

 private:
  struct Cluster {
    // Identifies the node that represents this cluster in the cycle detection
    // graph.
    int representative = -1;

    // The set of devices the nodes in this cluster are placed on.
    absl::flat_hash_set<string> devices;

    // If there are resource operation in the cluster then this is the device
    // that resource operations are placed on.  All resource operations in a
    // cluster must be placed on the same device.
    string resource_op_device;

    // If set then it is a predicate that is true iff the cluster is alive
    // (clusters are alive or dead as a single unit).  If unset we've decided to
    // (unsafely) ignore deadness analysis because the user asked us to.  If
    // this is unset on a single Cluster instance then it is unset on all
    // Cluster instances.
    absl::optional<DeadnessAnalysis::DeadnessPredicate> deadness_predicate;

    // True if any node in the cluster has an _XlaCompile attribute set to true.
    bool has_xla_compile_attr;
  };

  // Nodes that XLA can compile are put in `candidates`.  Nodes put in
  // `isolated_nodes` must either be unclustered or be put in trivial
  // single-node clusters.
  StatusOr<std::pair<OrderedNodeSet, absl::flat_hash_set<Node*>>>
  FindCompilationCandidates();

  bool CompilationDisallowedByXlaCompileAttr(Node* node,
                                             const DeviceType& jit_device_type);

  Status BuildInitialClusterSet(const OrderedNodeSet& compilation_candidates,
                                const DeadnessAnalysis* deadness_analysis,
                                std::vector<UnionFind<Cluster>>* clusters,
                                std::deque<UnionFind<Cluster>*>* worklist);

  StatusOr<bool> ShouldCompileClusterImpl(const Cluster& cluster);

  StatusOr<bool> ShouldCompileCluster(const Cluster& cluster);

  bool HasMismatchingXlaScope(Node* node_from, Node* node_to);

  StatusOr<bool> ClusteringWillIntroduceInterDeviceDependency(
      int to_node_id, const OrderedNodeSet& compilation_candidates,
      absl::Span<UnionFind<Cluster>> clusters, const GraphCycles& cycles);

  // Returns true if the devices in `cluster_a` and `cluster_b` are compatible
  // and therefore not a hindrance for combining the two clusters into a larger
  // cluster.
  StatusOr<bool> AreDevicesCompatible(const Cluster& cluster_a,
                                      const Cluster& cluster_b);

  void DumpPostClusteringGraphs();
  void VLogClusteringSummary();

  DebugOptions debug_options_;
  Graph* graph_;
  FunctionLibraryDefinition* flib_def_;
  Env* env_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  absl::flat_hash_map<int, bool> should_compile_cluster_cache_;
  DeviceInfoCache device_info_cache_;
};

StatusOr<bool>
MarkForCompilationPassImpl::ClusteringWillIntroduceInterDeviceDependency(
    int to_node_id, const OrderedNodeSet& compilation_candidates,
    absl::Span<UnionFind<Cluster>> clusters, const GraphCycles& cycles) {
  const Cluster& cluster_to = clusters[to_node_id].Get();

  // If any of the consumer's producers are on a different device, do not
  // cluster these nodes. This prevents other work on this device from being
  // delayed by work on other devices. We consider predecessors of the entire
  // cluster rather than just the inputs to the node to prevent the cluster
  // still being combined in cases where the 'to' cluster has multiple
  // dependencies on the 'from' cluster and another dependency leads to a
  // merging of the clusters.
  //
  // TODO(b/117085735): We probably want to handle the reciprocal of this case
  // where a cluster is producing data for multiple devices.
  for (const auto& in_id : cycles.Predecessors(to_node_id)) {
    if (in_id >= graph_->num_node_ids()) {
      continue;
    }

    Node* in = graph_->FindNodeId(in_id);
    const Cluster& cluster_in = clusters[in_id].Get();
    if (compilation_candidates.find(in) != compilation_candidates.cend()) {
      TF_ASSIGN_OR_RETURN(bool devices_compatible,
                          AreDevicesCompatible(cluster_to, cluster_in));
      if (!devices_compatible) {
        return true;
      }
    }
  }

  return false;
}

bool MarkForCompilationPassImpl::HasMismatchingXlaScope(Node* node_from,
                                                        Node* node_to) {
  // Look for an _XlaScope on both nodes.  If both nodes have a scope and the
  // scopes do not match, do not cluster along this edge. This restriction is
  // overridden if the global_jit_level_ is ON. If even one of the nodes lacks
  // an _XlaScope attribute, then it is treated as a "bridge" and a cluster may
  // be created along it.  We may want to restrict this behavior to require all
  // nodes marked with _XlaCompile=true to also have a _XlaScope property set
  // (and raise an error otherwise); but for now we don't do this.
  if (global_jit_level_ != OptimizerOptions::OFF) {
    return false;
  }

  string from_scope, to_scope;
  return GetNodeAttr(node_from->attrs(), kXlaScopeAttr, &from_scope).ok() &&
         GetNodeAttr(node_to->attrs(), kXlaScopeAttr, &to_scope).ok() &&
         from_scope != to_scope;
}

Status MarkForCompilationPassImpl::BuildInitialClusterSet(
    const OrderedNodeSet& compilation_candidates,
    const DeadnessAnalysis* deadness_analysis,
    std::vector<UnionFind<Cluster>>* clusters,
    std::deque<UnionFind<Cluster>*>* worklist) {
  clusters->resize(graph_->num_node_ids());
  for (Node* node : compilation_candidates) {
    Cluster* cluster = &(*clusters)[node->id()].Get();
    cluster->representative = node->id();

    if (deadness_analysis) {
      TF_ASSIGN_OR_RETURN(
          cluster->deadness_predicate,
          deadness_analysis->GetPredicateFor(node, Graph::kControlSlot));
    }

    const string& device = !node->assigned_device_name().empty()
                               ? node->assigned_device_name()
                               : node->requested_device();
    if (HasResourceInput(*node) || HasResourceOutput(*node)) {
      cluster->resource_op_device = device;
    }

    cluster->has_xla_compile_attr = false;

    bool xla_compile_attr;
    if (GetNodeAttr(node->attrs(), kXlaCompileAttr, &xla_compile_attr).ok()) {
      cluster->has_xla_compile_attr |= xla_compile_attr;
    }

    if (flib_def_->GetAttr(*node, kXlaCompileAttr, &xla_compile_attr).ok()) {
      cluster->has_xla_compile_attr |= xla_compile_attr;
    }

    cluster->devices.insert(device);
    worklist->push_back(&(*clusters)[node->id()]);
  }

  return Status::OK();
}

StatusOr<std::pair<OrderedNodeSet, absl::flat_hash_set<Node*>>>
MarkForCompilationPassImpl::FindCompilationCandidates() {
  OrderedNodeSet candidates;
  absl::flat_hash_set<Node*> isolated_nodes;

  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, env_, TF_GRAPH_DEF_VERSION,
                                        flib_def_, opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  std::vector<bool> compile_time_const_nodes(graph_->num_node_ids(), false);
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
      *graph_, /*compile_time_const_arg_indices=*/nullptr,
      &compile_time_const_nodes, lib_runtime));

  // Iterate over nodes in sorted order so that compiler fuel is deterministic.
  // We can't simply pass op_nodes().begin() and op_nodes().end to the
  // std::vector constructor because they're not proper iterators, with
  // iterator_traits defined and so on.
  std::vector<Node*> sorted_nodes;
  for (Node* node : graph_->op_nodes()) {
    sorted_nodes.push_back(node);
  }
  std::sort(sorted_nodes.begin(), sorted_nodes.end(), NodeComparatorID());

  if (*debug_options_.fuel >= std::numeric_limits<int64>::max() / 2) {
    // The assumption is that if fuel started out as INT64_MAX, it will forever
    // stay greater than INT64_MAX / 2.
    VLOG(2) << "Starting fuel: infinity";
  } else {
    VLOG(2) << "Starting fuel: " << *debug_options_.fuel;
  }

  VLOG(2) << "sorted_nodes.size() = " << sorted_nodes.size();

  for (Node* node : sorted_nodes) {
    if (*debug_options_.fuel <= 0) {
      VLOG(1)
          << "Hit fuel limit; not marking any remaining ops as clusterable.";
      break;
    }

    TF_ASSIGN_OR_RETURN(
        const DeviceType& device_type,
        device_info_cache_.GetDeviceTypeFor(node->assigned_device_name()));
    VLOG(4) << "Device type for " << node->name() << ": "
            << device_type.type_string();

    if (CompilationDisallowedByXlaCompileAttr(node, device_type)) {
      VLOG(2) << "Not clustering " << node->name()
              << ": disallowed by _XlaCompile attribute";
      continue;
    }

    const XlaOpRegistry::DeviceRegistration* registration;
    if (!XlaOpRegistry::GetCompilationDevice(device_type.type(),
                                             &registration)) {
      VLOG(2) << "Rejecting " << node->name()
              << ": could not find JIT device for " << device_type.type();
      continue;
    }

    DeviceType jit_device_type(registration->compilation_device_name);

    RecursiveCompilabilityChecker::OperationFilter op_filter =
        CreateOperationFilter(*registration);

    if (!RecursiveCompilabilityChecker{&op_filter, &jit_device_type}
             .IsCompilableNode(*node, lib_runtime)) {
      continue;
    }

    if (compile_time_const_nodes[node->id()]) {
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(
          graph_->op_registry()->LookUpOpDef(node->type_string(), &op_def));
      if (op_def->is_stateful()) {
        // It is easiest to demonstrate the problem we're trying to solve with
        // an example.  Say we have this graph:
        //
        //   shape = RandomUniformInt();
        //   reshape = Reshape(input, shape)
        //
        // Both RandomUniformInt and Reshape are compilable by XLA so, absent
        // any other reason, we will try to put both shape and reshape in the
        // same cluster.  However, since XLA only supports statically shaped
        // values, it will expect to be able to constant fold `shape` to get a
        // static shape for `reshape`.  This is a problem because side-effecting
        // ops like RandomUniformInt() cannot be constant folded.  We fix this
        // by putting `shape` and `reshape` in different clusters, which results
        // in us recompiling `reshape`'s cluster for every new value of `shape`,
        // making `reshape` statically sized within each compilation.  We
        // simplify the solution even further by disallowing operations like
        // `shape` from being part of *any* non-trivial cluster.  They're either
        // not compiled by XLA altogether or, if assigned to an XLA_* device
        // with "must compile" semantics, compiled into a trivial single-op
        // cluster.  This approach leaves some room for improvement, and we can
        // consider implementing a more aggressive data-flow-analysis based
        // solution in the future if needed.
        //
        // One ugly problem we have to contend with: certain sets of ops *have*
        // to be in the same cluster because values flowing between them have
        // types that can't be live-in or live-out of a cluster.  These ops are:
        //
        //  - TensorArray ops operating on the same TensorArray instance.
        //  - Stack ops operating on the same Stack instance.
        //
        // To work around this we avoid isolating these specific ops.  Because
        // of this concession it is unsound to auto-cluster them because then
        // we'd create clusters we could not compile (because we can't constant
        // fold, say, a TensorArrayRead or a StackPopV2).  But we don't
        // auto-cluster these operations today so we're good for now.
        const XlaResourceOpInfo* op_info =
            GetResourceOpInfoForOp(node->type_string());
        bool is_tensor_array_or_stack_op =
            op_info && op_info->resource_kind() != XlaResourceKind::kVariable;
        if (!is_tensor_array_or_stack_op) {
          VLOG(2) << "Isolating " << node->name()
                  << ": must-be-constant stateful op";
          isolated_nodes.insert(node);
        }
      }
    }

    candidates.insert(node);
    --(*debug_options_.fuel);
  }

  VLOG(2) << "candidates->size() = " << candidates.size();
  return {{candidates, isolated_nodes}};
}

bool MarkForCompilationPassImpl::CompilationDisallowedByXlaCompileAttr(
    Node* node, const DeviceType& device_type) {
  if (debug_options_.ignore_xla_compile_attr) {
    return false;
  }

  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    VLOG(2) << "Rejecting " << node->name() << ": could not find JIT device.";
    return false;
  }

  // If there is a _XlaCompile annotation, use its value.
  bool compile = false;
  Status status = GetNodeAttr(node->attrs(), kXlaCompileAttr, &compile);
  if (status.ok()) {
    if (!compile) {
      VLOG(2) << "Rejecting " << node->name() << ": kXlaCompileAttr("
              << kXlaCompileAttr << ") is false.";
    }
    return !compile;
  }

  status = flib_def_->GetAttr(*node, kXlaCompileAttr, &compile);
  if (status.ok()) {
    if (!compile) {
      VLOG(2) << "Rejecting " << node->name() << ": kXlaCompileAttr("
              << kXlaCompileAttr << ") on callee is false.";
    }
    return !compile;
  }

  return false;
}

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
bool IsShapeConsumerOp(const Node& node) {
  return node.type_string() == "Shape" || node.type_string() == "Rank" ||
         node.type_string() == "Size";
}

Status IgnoreResourceOpForSafetyAnalysis(DeviceInfoCache* device_info_cache,
                                         const Node& n, bool* ignore) {
  // If a resource operation is assigned to XLA_CPU or XLA_GPU explicitly then
  // ignore it during resource operation safety analysis.  We need this hack
  // because of two reasons:
  //
  //  1. Operations assigned to XLA_CPU and XLA_GPU have to always be compiled.
  //  2. We don't support live-out values of type DT_RESOURCE and live-in values
  //     of type DT_RESOURCE that are not resource variables.
  //
  // Together these imply we cannot let resource variable safety analysis
  // constrain e.g. a TensorArrayV3->TensorArrayAssignV3 edge to be in different
  // clusters: both of them will have to be clustered because of (1) and we
  // won't be able to keep the edge between the two as neither the input to the
  // second XLA cluster nor the output from the first XLA cluster are supported
  // because of (2).
  //
  // TODO(b/113100872): This can be fixed if the TensorFlow representation for
  // TensorArray and Stack on the XLA_{C|G}PU devices were the same in XLA; then
  // (2) would no longer hold.

  if (n.assigned_device_name().empty()) {
    *ignore = false;
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(
      const XlaOpRegistry::DeviceRegistration* registration,
      device_info_cache->GetCompilationDevice(n.assigned_device_name()));

  if (!registration) {
    *ignore = true;
  } else {
    *ignore = registration->cluster_resource_variable_ops_unsafely;
  }
  return Status::OK();
}

Status MarkForCompilationPassImpl::Run() {
  static std::atomic<int64> cluster_sequence_num;

  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();

  // Start the timer after XlaOpRegistry::RegisterCompilationKernels which does
  // some one-time work.
  XLA_SCOPED_LOGGING_TIMER_LEVEL("MarkForCompilationPassImpl::Run", 1);

  OrderedNodeSet compilation_candidates;
  absl::flat_hash_set<Node*> isolated_nodes;
  TF_ASSIGN_OR_RETURN(std::tie(compilation_candidates, isolated_nodes),
                      FindCompilationCandidates());

  if (compilation_candidates.empty()) {
    VLOG(2) << "No compilable candidates";
    return Status::OK();
  }

  GraphCycles cycles;
  TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok,
                      CreateCycleDetectionGraph(graph_, &cycles));
  if (!cycle_detection_graph_ok) {
    return Status::OK();
  }

  auto ignore_resource_ops = [&](const Node& n, bool* ignore) {
    return IgnoreResourceOpForSafetyAnalysis(&device_info_cache_, n, ignore);
  };

  TF_RETURN_IF_ERROR(AdjustCycleDetectionGraphForResourceOps(
      graph_, flib_def_, ignore_resource_ops, &cycles));

  std::unique_ptr<DeadnessAnalysis> deadness_analysis;
  if (!debug_options_.ignore_deadness_checks) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL("DeadnessAnalysis", 1);
    TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph_, &deadness_analysis));
  }

  // Each compilation candidate belongs to a cluster. The cluster's
  // representative names the node in the 'cycles' graph that represents the
  // cluster.
  std::vector<UnionFind<Cluster>> clusters;
  std::deque<UnionFind<Cluster>*> worklist;
  TF_RETURN_IF_ERROR(BuildInitialClusterSet(
      compilation_candidates, deadness_analysis.get(), &clusters, &worklist));

  int64 iteration_count = 0;

  // Repeatedly contract edges between clusters that are on the same device,
  // provided the contraction would not create a cycle.
  //
  // TODO(hpucha): Handle the case where kXlaClusterAttr is already set (for
  // example, from the Grappler fusion pass).
  while (!worklist.empty()) {
    Cluster* cluster_from = &worklist.front()->Get();
    int from = cluster_from->representative;
    worklist.pop_front();

    Node* node_from = graph_->FindNodeId(from);
    if (node_from->IsControlFlow()) {
      // Control flow nodes aren't compilation candidates and should never
      // appear.
      return errors::Internal(
          "Found control flow node in clustering worklist: ",
          node_from->type_string());
    }

    if (isolated_nodes.count(node_from)) {
      continue;
    }

    for (int to : cycles.Successors(from)) {
      iteration_count++;
      if (to >= graph_->num_node_ids()) {
        // Node is a fictitious node that is present only in the cycle detection
        // graph. No clustering is possible.
        continue;
      }

      const Cluster& cluster_to = clusters[to].Get();
      Node* node_to = graph_->FindNodeId(to);
      if (compilation_candidates.find(node_to) ==
          compilation_candidates.cend()) {
        continue;
      }

      DCHECK(cluster_from->deadness_predicate.has_value() ==
             cluster_to.deadness_predicate.has_value());
      if (cluster_from->deadness_predicate != cluster_to.deadness_predicate) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(bool devices_compatible,
                          AreDevicesCompatible(*cluster_from, cluster_to));
      if (!devices_compatible) {
        continue;
      }

      if (isolated_nodes.count(node_to)) {
        continue;
      }

      if (HasMismatchingXlaScope(node_from, node_to)) {
        continue;
      }

      // Ops that consume shapes cannot be the root of a cluster. This is an
      // optimization.
      if (clusters[from].Size() == 1 && IsShapeConsumerOp(*node_from)) {
        continue;
      }

      // Don't exceed the maximum cluster size.
      if (clusters[from].Size() + clusters[to].Size() >
          debug_options_.max_cluster_size) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(
          bool will_introduce_cross_device_dependency,
          ClusteringWillIntroduceInterDeviceDependency(
              to, compilation_candidates, absl::MakeSpan(clusters), cycles));

      if (will_introduce_cross_device_dependency) {
        continue;
      }

      // If contracting the edge would create a cycle, bail out.  However, just
      // because we can't merge the clusters now does not mean we won't be able
      // to merge them in the future.  e.g., if we have edges 1->2, 2->3 and
      // 1->3, we cannot contract edge 1->3. But if we first contract 1->2 then
      // we can later contract 1->3.
      if (!cycles.ContractEdge(from, to)) {
        continue;
      }

      // Merge the clusters. ContractEdge uses 'from' as the number of the
      // merged node, so make sure 'from' is the chosen representative.
      cluster_from->devices.insert(cluster_to.devices.begin(),
                                   cluster_to.devices.end());
      if (!cluster_to.resource_op_device.empty()) {
        cluster_from->resource_op_device = cluster_to.resource_op_device;
      }
      cluster_from->has_xla_compile_attr |= cluster_to.has_xla_compile_attr;
      clusters[from].Merge(&clusters[to]);

      worklist.push_back(&clusters[from]);
      break;
    }
  }

  VLOG(1) << iteration_count << " iterations in inner loop for graph with "
          << compilation_candidates.size()
          << " compilation candidates.  Iterations per compilation candidate: "
          << ((1.0 * iteration_count) / compilation_candidates.size());

  // Count the number of non-trivial elements in each cluster.
  std::vector<int> effective_cluster_sizes(graph_->num_node_ids());

  // has_functional_control_flow remembers if a cluster contains a functional
  // control flow node.
  std::vector<bool> has_functional_control_flow(graph_->num_node_ids());

  for (const Node* n : compilation_candidates) {
    int cluster = clusters[n->id()].Get().representative;
    // We want clusters to be big enough that the benefit from XLA's
    // optimizations offsets XLA related overhead (for instance we add some
    // Switch/Merge nodes into the graph to implement lazy compilation).  To
    // this end, we don't count Identity and Constant nodes because they do not
    // enable interesting optimizations by themselves.
    if (!n->IsIdentity() && !n->IsConstant()) {
      effective_cluster_sizes[cluster]++;
    }
    if (n->type_string() == "While" || n->type_string() == "If") {
      has_functional_control_flow[cluster] = true;
    }
  }

  // Names for each cluster.
  std::unordered_map<int, string> cluster_names;

  if (debug_options_.dump_graphs) {
    DumpGraphToFile("before_mark_for_compilation", *graph_, flib_def_);
  }

  // Mark clusters for compilation that:
  // * are placed on a device that requires compilation (an XlaDevice),
  // * are explicitly marked for compilation (_XlaCompile=true), or
  // * have more than debug_options_.xla_min_cluster_size elements (applicable
  //   only if compilation is enabled, otherwise there will be no such
  //   candidates).
  for (Node* n : compilation_candidates) {
    const Cluster& cluster = clusters[n->id()].Get();
    TF_ASSIGN_OR_RETURN(bool should_compile_cluster,
                        ShouldCompileCluster(cluster));
    if (!should_compile_cluster) {
      continue;
    }

    int cluster_repr = cluster.representative;

    // Compile if the user marked this node _XlaCompile=true
    bool compile_attr = false;
    bool marked_for_compilation = false;
    if (GetNodeAttr(n->attrs(), kXlaCompileAttr, &compile_attr).ok()) {
      marked_for_compilation = compile_attr;
    } else if (flib_def_->GetAttr(*n, kXlaCompileAttr, &compile_attr).ok()) {
      marked_for_compilation = compile_attr;
    }

    // We assume that functional If and While nodes have at least
    // min_cluster_size non-trivial nodes in them.  It would be more principled
    // to (recursively) verify this fact, but that's probably not worth the
    // trouble.

    if (effective_cluster_sizes[cluster_repr] >=
            debug_options_.min_cluster_size ||
        has_functional_control_flow[cluster_repr] || marked_for_compilation) {
      string& name = cluster_names[cluster_repr];

      if (name.empty()) {
        name = absl::StrCat("cluster_", cluster_sequence_num++);
      }
      n->AddAttr(kXlaClusterAttr, name);
      n->AddAttr(kXlaAlreadyClustered, true);
      VLOG(3) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }

  if (debug_options_.dump_graphs) {
    DumpPostClusteringGraphs();
  }

  VLogClusteringSummary();

  return Status::OK();
}

void MarkForCompilationPassImpl::DumpPostClusteringGraphs() {
  DumpGraphToFile("mark_for_compilation", *graph_, flib_def_);

  // We also dump out an annoated version of the TF graph where the nodes
  // names are prefixed with the cluster names.  This can help visualizing the
  // clustering decisions on TensorBoard.
  Graph new_graph(graph_->op_registry());
  CopyGraph(*graph_, &new_graph);

  for (Node* n : new_graph.nodes()) {
    if (absl::optional<absl::string_view> cluster_name =
            GetXlaClusterForNode(*n)) {
      n->set_name(absl::StrCat(*cluster_name, "/", n->name()));
    } else if (n->type_string() == "VarHandleOp") {
      n->set_name(absl::StrCat("varhandle/", n->name()));
    } else {
      // There is room for improvement here.  In particular, it may help to
      // split these unclustered nodes into classes where every node in a
      // specific class has edges to and from the same set of clusters.
      n->set_name(absl::StrCat("unclustered/", n->name()));
    }
  }

  DumpGraphToFile("mark_for_compilation_annotated", new_graph, flib_def_);
}

string RatioToString(int numerator, int denominator) {
  return absl::StrFormat("%d / %d (%.2f%%)", numerator, denominator,
                         (100.0 * numerator) / denominator);
}

void MarkForCompilationPassImpl::VLogClusteringSummary() {
  if (!VLOG_IS_ON(2)) {
    return;
  }

  std::map<absl::string_view, int> cluster_name_to_size;
  std::map<absl::string_view, std::map<absl::string_view, int>>
      cluster_name_to_op_histogram;
  std::map<absl::string_view, int> unclustered_op_histogram;
  int clustered_node_count = 0;

  for (Node* n : graph_->nodes()) {
    absl::optional<absl::string_view> cluster_name = GetXlaClusterForNode(*n);
    if (cluster_name) {
      clustered_node_count++;
      cluster_name_to_size[*cluster_name]++;
      cluster_name_to_op_histogram[*cluster_name][n->type_string()]++;
    } else {
      unclustered_op_histogram[n->type_string()]++;
    }
  }

  int unclustered_node_count = graph_->num_nodes() - clustered_node_count;

  VLOG(2) << "*** Clustering info for graph of size " << graph_->num_nodes();
  VLOG(2) << " Built " << cluster_name_to_size.size() << " clusters, size "
          << RatioToString(clustered_node_count, graph_->num_nodes());

  for (const auto& cluster_name_size_pair : cluster_name_to_size) {
    absl::string_view cluster_name = cluster_name_size_pair.first;
    int size = cluster_name_size_pair.second;
    VLOG(2) << "  " << cluster_name << " "
            << RatioToString(size, graph_->num_nodes());
    for (const auto& op_count_pair :
         cluster_name_to_op_histogram[cluster_name]) {
      VLOG(3) << "   " << op_count_pair.first << ": " << op_count_pair.second
              << " instances";
    }
  }

  if (!unclustered_op_histogram.empty()) {
    VLOG(2) << " Unclustered nodes: "
            << RatioToString(unclustered_node_count, graph_->num_nodes());
    for (const auto& pair : unclustered_op_histogram) {
      VLOG(3) << "  " << pair.first << ": " << pair.second << " instances";
    }
  }

  struct EdgeInfo {
    absl::string_view node_name;
    absl::optional<absl::string_view> cluster_name;

    absl::string_view GetClusterName() const {
      return cluster_name ? *cluster_name : "[none]";
    }

    std::pair<absl::string_view, absl::optional<absl::string_view>> AsPair()
        const {
      return {node_name, cluster_name};
    }

    bool operator<(const EdgeInfo& other) const {
      return AsPair() < other.AsPair();
    }
  };

  using EdgeInfoMap = std::map<absl::string_view, std::map<EdgeInfo, int64>>;

  EdgeInfoMap incoming_edge_infos;
  EdgeInfoMap outgoing_edge_infos;

  std::set<absl::string_view> cluster_names_to_print;

  for (const Edge* e : graph_->edges()) {
    const Node* from = e->src();
    absl::optional<absl::string_view> from_cluster_name =
        GetXlaClusterForNode(*from);

    const Node* to = e->dst();
    absl::optional<absl::string_view> to_cluster_name =
        GetXlaClusterForNode(*to);

    if (to_cluster_name == from_cluster_name) {
      continue;
    }

    if (to_cluster_name) {
      incoming_edge_infos[*to_cluster_name]
                         [EdgeInfo{from->name(), from_cluster_name}]++;
      cluster_names_to_print.insert(*to_cluster_name);
    }

    if (from_cluster_name) {
      outgoing_edge_infos[*from_cluster_name][{to->name(), to_cluster_name}]++;
      cluster_names_to_print.insert(*from_cluster_name);
    }
  }

  VLOG(2) << "*** Inter-Cluster edges:";
  if (cluster_names_to_print.empty()) {
    VLOG(2) << "   [none]";
  }

  auto print_edge_info_set_for_cluster = [&](absl::string_view cluster_name,
                                             const EdgeInfoMap& edge_info_map,
                                             absl::string_view desc) {
    auto it = edge_info_map.find(cluster_name);
    if (it != edge_info_map.end()) {
      VLOG(2) << "  " << it->second.size() << " " << desc << " edges";
      for (const auto& edge_info_count_pair : it->second) {
        VLOG(2) << "   " << edge_info_count_pair.first.GetClusterName() << " "
                << edge_info_count_pair.first.node_name << " # "
                << edge_info_count_pair.second;
      }
    } else {
      VLOG(2) << "  No " << desc << " edges.";
    }
  };

  for (absl::string_view cluster_name : cluster_names_to_print) {
    VLOG(2) << " ** Cluster " << cluster_name;
    print_edge_info_set_for_cluster(cluster_name, incoming_edge_infos,
                                    "incoming");
    print_edge_info_set_for_cluster(cluster_name, outgoing_edge_infos,
                                    "outgoing");
  }
}

StatusOr<bool> MarkForCompilationPassImpl::AreDevicesCompatible(
    const Cluster& cluster_a, const Cluster& cluster_b) {
  std::vector<string> devices;
  absl::c_remove_copy(cluster_a.devices, std::back_inserter(devices), "");
  absl::c_remove_copy(cluster_b.devices, std::back_inserter(devices), "");
  absl::c_sort(devices);

  if (devices.empty()) {
    return false;
  }

  // First check if we will even be able to pick a device for the larger
  // combined cluster.
  bool can_pick_device;
  TF_RETURN_IF_ERROR(CanPickDeviceForXla(
      devices, /*allow_mixing_unknown_and_cpu=*/false, &can_pick_device));
  if (!can_pick_device) {
    return false;
  }

  string chosen_device;
  TF_RETURN_IF_ERROR(PickDeviceForXla(
      devices, /*allow_mixing_unknown_and_cpu=*/false, &chosen_device));

  // If we are able to pick a device `chosen_device` for the larger cluster, the
  // resource operations in `cluster_a` and `cluster_b` must be placed on the
  // same device as `chosen_device`.  This is because the _XlaCompile and
  // _XlaRun kernels are going to run on and therefore try to access the
  // resource variables from `chosen_device`, which will be an error if the
  // resource variables are placed on some other device.
  auto resource_op_device_ok = [&](const string& resource_op_device) {
    return resource_op_device.empty() || resource_op_device == chosen_device;
  };

  if (!resource_op_device_ok(cluster_a.resource_op_device) ||
      !resource_op_device_ok(cluster_b.resource_op_device)) {
    return false;
  }

  // We will check this again later, but here we prune out clusters that would
  // never have been sent to XLA to save compile time.  Without this change we
  // will e.graph_-> create a CPU cluster only to later notice that the user did
  // not enable the CPU JIT via --tf_xla_cpu_global_jit.  With this change we
  // avoid creating the cluster to begin with.
  //
  // TODO(b/126629785): It is possible that this is just papering over O(n^2)
  // behavior in our clustering algorithm.
  TF_ASSIGN_OR_RETURN(const XlaOpRegistry::DeviceRegistration* registration,
                      device_info_cache_.GetCompilationDevice(chosen_device));
  TF_RET_CHECK(registration)
      << "chosen device = " << chosen_device << "; devices (" << devices.size()
      << ") = " << absl::StrJoin(devices, ", ");

  return cluster_a.has_xla_compile_attr || cluster_b.has_xla_compile_attr ||
         registration->autoclustering_policy ==
             XlaOpRegistry::AutoclusteringPolicy::kAlways ||
         (registration->autoclustering_policy ==
              XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
          global_jit_level_ != OptimizerOptions::OFF);
}

// Returns `true` iff we should compile `cluster`.
StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileClusterImpl(
    const Cluster& cluster) {
  std::vector<string> devices;
  absl::c_remove_copy(cluster.devices, std::back_inserter(devices), "");
  absl::c_sort(devices);

  string chosen_device;
  TF_RETURN_IF_ERROR(PickDeviceForXla(
      devices, /*allow_mixing_unknown_and_cpu=*/false, &chosen_device));

  TF_ASSIGN_OR_RETURN(const DeviceType& device_type,
                      device_info_cache_.GetDeviceTypeFor(chosen_device));
  TF_ASSIGN_OR_RETURN(const XlaOpRegistry::DeviceRegistration* registration,
                      device_info_cache_.GetCompilationDevice(chosen_device));
  TF_RET_CHECK(registration)
      << "chosen device = " << chosen_device
      << "; device type = " << device_type.type() << "; devices ("
      << devices.size() << ") = " << absl::StrJoin(devices, ", ");

  bool should_compile =
      cluster.has_xla_compile_attr ||
      registration->autoclustering_policy ==
          XlaOpRegistry::AutoclusteringPolicy::kAlways ||
      (registration->autoclustering_policy ==
           XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
       global_jit_level_ != OptimizerOptions::OFF);

  if (!should_compile &&
      registration->autoclustering_policy ==
          XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested &&
      device_type.type_string() == DEVICE_CPU) {
    static std::once_flag once;
    std::call_once(once, [] {
      LOG(WARNING)
          << "(One-time warning): Not using XLA:CPU for cluster because envvar "
             "TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want "
             "XLA:CPU, either set that envvar, or use experimental_jit_scope "
             "to enable XLA:CPU.  To confirm that XLA is active, pass "
             "--vmodule=xla_compilation_cache=1 (as a proper command-line "
             "flag, not via TF_XLA_FLAGS) or set the envvar "
             "XLA_FLAGS=--xla_hlo_profile.";
      MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
      if (flags->tf_xla_cpu_global_jit) {
        LOG(WARNING)
            << "(Although the tf_xla_cpu_global_jit flag is currently enabled, "
               "perhaps it wasn't enabled at process startup?)";
      }
    });
  }

  VLOG(3) << (should_compile ? "Compiling" : "Not compiling")
          << " cluster with device " << chosen_device;

  return should_compile;
}

StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileCluster(
    const Cluster& cluster) {
  auto it = should_compile_cluster_cache_.find(cluster.representative);
  if (it != should_compile_cluster_cache_.end()) {
    return it->second;
  }

  TF_ASSIGN_OR_RETURN(bool should_compile, ShouldCompileClusterImpl(cluster));
  should_compile_cluster_cache_.insert(
      {cluster.representative, should_compile});
  return should_compile;
}

Status MarkForCompilation(
    const GraphOptimizationPassOptions& options,
    const MarkForCompilationPassImpl::DebugOptions& debug_options) {
  Graph* graph = options.graph->get();
  FunctionLibraryDefinition* flib_def = options.flib_def;

  // Deadness analysis expects a graph with source and sink edges properly
  // connected but sometimes the incoming graph does not follow this invariant.
  // So fix up the source and sink edges before calling into deadness analysis.
  FixupSourceAndSinkEdges(graph);

  // See explanation on `kXlaAlreadyClustered`.
  for (Node* n : graph->nodes()) {
    if (n->attrs().Find(kXlaAlreadyClustered)) {
      return Status::OK();
    }
  }

  return MarkForCompilationPassImpl{debug_options, graph, flib_def,
                                    options.session_options != nullptr
                                        ? options.session_options->env
                                        : Env::Default(),
                                    GetGlobalJitLevelForGraph(options)}
      .Run();
}

std::atomic<int64>* GetPointerToFuel(int64 initial_value) {
  static std::atomic<int64>* fuel = [&]() {
    std::atomic<int64>* fuel = new std::atomic<int64>;
    *fuel = initial_value;
    return fuel;
  }();

  return fuel;
}
}  // anonymous namespace

bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef) {
  Device* device = flr->device();
  const XlaOpRegistry::DeviceRegistration* registration;
  CHECK(XlaOpRegistry::GetCompilationDevice(device->device_type(),
                                            &registration));
  DeviceType jit_device_type(registration->compilation_device_name);

  // We can always *compile* resource operations, stateful RNGs and dummy ops,
  // even if we are sometimes unable to auto-cluster them.
  RecursiveCompilabilityChecker::OperationFilter op_filter;
  op_filter.allow_resource_ops_in_called_functions = true;
  op_filter.allow_stack_ops = true;
  op_filter.allow_tensor_array_ops = true;
  op_filter.allow_stateful_rng_ops = true;
  op_filter.allow_control_trigger = true;
  op_filter.allow_eliding_assert_and_checknumerics_ops = true;
  op_filter.allow_ops_producing_or_consuming_variant = true;

  return RecursiveCompilabilityChecker{&op_filter, &jit_device_type}
      .IsCompilableCall(ndef, flr);
}

Status MarkForCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();

  MarkForCompilationPassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks =
      flags->tf_xla_disable_deadness_safety_checks_for_debugging;
  debug_options.ignore_xla_compile_attr = false;
  debug_options.max_cluster_size = flags->tf_xla_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_xla_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_xla_clustering_fuel);
  debug_options.dump_graphs = flags->tf_xla_clustering_debug;

  return MarkForCompilation(options, debug_options);
}

Status MarkForCompilationPass::RunForTest(
    const GraphOptimizationPassOptions& options,
    bool disable_deadness_analysis) {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();

  MarkForCompilationPassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks = disable_deadness_analysis;
  debug_options.ignore_xla_compile_attr = true;
  debug_options.max_cluster_size = flags->tf_xla_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_xla_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_xla_clustering_fuel);
  debug_options.dump_graphs = flags->tf_xla_clustering_debug;

  return MarkForCompilation(options, debug_options);
}
}  // namespace tensorflow
