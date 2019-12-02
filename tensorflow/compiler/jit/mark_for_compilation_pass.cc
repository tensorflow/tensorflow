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
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/deadness_analysis.h"
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
#include "tensorflow/core/framework/tensor.pb.h"
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
using DeadnessPredicate = DeadnessAnalysis::DeadnessPredicate;
using jit::DeviceId;
using jit::DeviceSet;
using xla::StatusOr;

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

class MarkForCompilationPassImpl {
 public:
  struct DebugOptions {
    // If true, do not respect the results of deadness analysis.
    bool ignore_deadness_checks;

    // If true, do not do safety checks to preserve TensorFlow's resource
    // variable concurrency semantics.
    bool ignore_resource_variable_checks;

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
  // Represents a "cluster" or a connected subgraph of a TensorFlow graph.
  class Cluster {
   public:
    // Constructs a trivial cluster representing a single TF node.
    Cluster(int tf_graph_node_id, int effective_cluster_size,
            bool has_functional_control_flow, DeviceSet devices,
            absl::optional<DeviceId> resource_op_device,
            absl::optional<int> resource_var_operation_node_id,
            absl::optional<DeadnessPredicate> deadness_predicate,
            bool is_xla_compile_attr_true, absl::optional<string> xla_scope)
        : cycles_graph_node_id_(tf_graph_node_id),
          effective_cluster_size_(effective_cluster_size),
          has_functional_control_flow_(has_functional_control_flow),
          devices_(std::move(devices)),
          resource_op_device_(resource_op_device),
          deadness_predicate_(deadness_predicate),
          is_xla_compile_attr_true_(is_xla_compile_attr_true),
          xla_scope_(std::move(xla_scope)) {
      if (resource_var_operation_node_id.has_value()) {
        resource_var_operation_node_ids_.push_back(
            *resource_var_operation_node_id);
      }
    }

    // Merges `other` into this cluster, and clears `other`.  This method is
    // closely tied with the implementation of `MarkForCompilationPassImpl`.
    void Merge(Cluster* other);

    // If this is a trivial cluster containing only one node then return the ID
    // of that node.  May not be called otherwise.
    int GetIdOfOnlyNode() const {
      DCHECK_EQ(cluster_size(), 1);
      return cycles_graph_node_id();
    }

    // The number of TF nodes in this cluster.
    int cluster_size() const { return cluster_size_; }

    // The ID of the cluster as represented in `cycles_graph_`.
    int cycles_graph_node_id() const { return cycles_graph_node_id_; }

    // The size of the cluster excluding constant and identity nodes.
    int effective_cluster_size() const { return effective_cluster_size_; }

    // True if the cluster has functional control flow like `If` and `While`.
    bool has_functional_control_flow() const {
      return has_functional_control_flow_;
    }

    // The set of devices nodes in the cluster are placed on.
    const DeviceSet& devices() const { return devices_; }

    // If the cluster has a resource operation then the device the resource
    // operation is placed on.  A cluster may have resource ops placed only on a
    // single device.
    const absl::optional<DeviceId>& resource_op_device() const {
      return resource_op_device_;
    }

    // If not nullopt the a predicate that is true iff the cluster is alive.
    // Otherwise the user has (unsafely) disabled deadness analysis.  If this is
    // unset on a single Cluster instance then it is unset on all Cluster
    // instances.
    const absl::optional<DeadnessPredicate>& deadness_predicate() const {
      return deadness_predicate_;
    }

    // If true then the cluster has a XlaCompile=true attribute on one of its
    // nodes.
    bool is_xla_compile_attr_true() const { return is_xla_compile_attr_true_; }

    // If not nullopt then the all nodes in the cluster either do not have the
    // XlaScope attribute set or have it set to the value returned.
    const absl::optional<string>& xla_scope() const { return xla_scope_; }

    // Returns the TF graph node IDs for the resource variable operations in
    // this cluster.
    absl::Span<const int> resource_var_operation_node_ids() const {
      return resource_var_operation_node_ids_;
    }

    string DebugString(const Graph& graph) const {
      Node* node = graph.FindNodeId(cycles_graph_node_id());
      if (!node) {
        // This should never happen but we try to be resilient because this is a
        // debugging aid.
        return absl::StrCat("NULL NODE IN #", cycles_graph_node_id());
      }

      if (cluster_size() == 1) {
        return absl::StrCat("<", node->name(), " #", cycles_graph_node_id(),
                            ">");
      }

      return absl::StrCat("<", node->name(), " + ", cluster_size() - 1,
                          " others #", cycles_graph_node_id(), ">");
    }

   private:
    int cluster_size_ = 1;
    int cycles_graph_node_id_;
    int effective_cluster_size_;
    bool has_functional_control_flow_;
    DeviceSet devices_;
    absl::optional<DeviceId> resource_op_device_;
    absl::optional<DeadnessPredicate> deadness_predicate_;
    bool is_xla_compile_attr_true_;
    absl::optional<string> xla_scope_;
    std::vector<int> resource_var_operation_node_ids_;

    TF_DISALLOW_COPY_AND_ASSIGN(Cluster);
  };

  // If `cluster` has only a single node then returns that, otherwise returns
  // nullptr.
  Node* GetOnlyNodeIn(const Cluster& cluster);

  // Returns true if `cluster` is a trivial cluster containing a "sink like"
  // node -- a NoOp node that only the Sink node control depends on.
  bool IsSinkLike(const Cluster& cluster);

  // Returns true if `cluster` looks like an "i++" operation on an integer
  // scalar resource variable.
  bool IsScalarIntegerResourceOperation(const Cluster& cluster);

  // ---------------------------------------------------------------------------
  // The pass proceeds in four steps, out of which `RunEdgeContractionLoop` and
  // `CreateClusters` do most of the heavy lifting.

  // Initializes some internal data structures.
  //
  // If this returns false then Initialize exited early (either because there is
  // nothing to do or we saw a graph that we can't handle) and not all the
  // fields in this MarkForCompilationPassImpl instance are set up.
  StatusOr<bool> Initialize();

  // Runs through the entire cluster graph in post-order and calls `fn(from,
  // to)` on each edge.  `fn(from, to)` is expected to return true if it was
  // able to contract `from`->`to`.
  //
  // Returns true if `fn` returned true for any edge.
  template <typename FnTy>
  StatusOr<bool> ForEachEdgeInPostOrder(FnTy fn);

  // Contracts as many edges as possible to create XLA clusters.  After this
  // finishes the clustering decisions made are implicitly stored in
  // `clusters_`.
  Status RunEdgeContractionLoop();

  // Manifests the clustering decisions into the TF graph by tagging nodes with
  // an `_XlaCluster` attribute.  Also some basic filter logic, like
  // tf_xla_min_cluster_size, are applied here.
  Status CreateClusters();

  Status DumpDebugInfo();

  bool IsCompilationCandidate(Node* n) const {
    return compilation_candidates_.find(n) != compilation_candidates_.end();
  }

  // Tries to contract the edge from cluster `from` to cluster `to`.  Returns
  // true if successful.
  StatusOr<bool> TryToContractEdge(Cluster* from, Cluster* to);

  // Nodes that XLA can compile are put in `compilation_candidates_`.
  Status FindCompilationCandidates();

  bool CompilationDisallowedByXlaCompileAttr(Node* node);

  // Populates `clusters_`.
  Status BuildInitialClusterSet();

  StatusOr<bool> ShouldCompileClusterImpl(const Cluster& cluster);

  StatusOr<bool> ShouldCompileCluster(const Cluster& cluster);

  StatusOr<bool> ClusteringWillIntroduceInterDeviceDependency(
      const Cluster& from, const Cluster& to);

  // Returns true if the devices in `cluster_a` and `cluster_b` are compatible
  // and therefore not a hindrance for combining the two clusters into a larger
  // cluster.
  StatusOr<bool> AreDevicesCompatible(const Cluster& cluster_a,
                                      const Cluster& cluster_b);

  void DumpPostClusteringGraphs();
  void VLogClusteringSummary();

  Cluster* MakeNewCluster(int cycles_graph_node_id, int effective_cluster_size,
                          bool has_functional_control_flow,
                          const DeviceSet& device_set,
                          absl::optional<DeviceId> resource_op_device,
                          absl::optional<int> resource_var_operation_node_id,
                          absl::optional<DeadnessPredicate> deadness_predicate,
                          bool is_xla_compile_attr_true,
                          absl::optional<string> xla_scope) {
    cluster_storage_.push_back(absl::make_unique<Cluster>(
        cycles_graph_node_id, effective_cluster_size,
        has_functional_control_flow, device_set, resource_op_device,
        resource_var_operation_node_id, deadness_predicate,
        is_xla_compile_attr_true, xla_scope));
    return cluster_storage_.back().get();
  }

  absl::optional<string> GetXlaScope(Node* n);

  // Returns the cluster for node `n`.  If two nodes, N1 and N2, are placed in
  // the same cluster by the clustering algorithm then this function will return
  // the same Cluster instance for N1 and N2.
  //
  // Returns nullptr if `n` is not a compilation candidate.
  Cluster* GetClusterForNode(Node* n) {
    return cluster_for_node_[n->id()].Get();
  }

  // Returns the cluster for a node in `cycles_graph_`.  This uses the same
  // underlying map because of how we set things up, but we can do an additional
  // CHECK in this accessor.
  //
  // Returns nullptr if `node_id` is not a compilation candidate.
  Cluster* GetClusterForCyclesGraphNode(int node_id) {
    // We have to check `graph_->FindNodeId(node) == nullptr` because we add all
    // nodes in [0, graph_->num_node_ids()) to the cycle detection graph but the
    // TF graph may be missing some node ids.
    if (node_id >= graph_->num_node_ids() ||
        graph_->FindNodeId(node_id) == nullptr) {
      return nullptr;
    }
    Cluster* cluster = cluster_for_node_[node_id].Get();
    if (cluster) {
      DCHECK_EQ(cluster->cycles_graph_node_id(), node_id);
    }
    return cluster;
  }

  bool LogNotContractableAndReturnFalse(Cluster* from, Cluster* to,
                                        absl::string_view reason);

  // Finds a path in `cycles_graph_` from `from` to `to` that is not a direct
  // edge from `from` to `to`.
  //
  // Tries to find a path that contains at least one unclusterable node.
  std::vector<int> FindAlternatePathForDebugging(int from, int to);

  // Returns a string representing `cycles_graph_node_id`.  If the node is
  // unclusterable (either it is a phatom "frame" node or is not a compilation
  // candidate) then set `*found_unclustered` to true.
  string DebugStringForCyclesGraphNode(int node_id, bool* found_unclustered);

  // We could not contract the edge from `from` to `to`.  Return a string
  // describing an alternate path from `from` to `to` (besides the direct edge
  // from `from` to `to`) which would have created a cycle had we contracted the
  // edge.
  //
  // Tries (if possible) to find a path that contains at least one unclusterable
  // node as it is surprising to the user if we print "A->B could not be
  // contracted because of the path [P,Q,R]" where P, Q and R are all clusters
  // since in that case a natural question is why we could not form a {A, P, Q,
  // R, B} cluster.
  string DescribePotentialCycle(int from, int to);

  // Merge the clusters `cluster_from` and `cluster_to`.  After this step the
  // larger combined cluster is represented by `cluster_from`'s ID in
  // `cycles_graph_`.
  bool MergeClusters(Cluster* cluster_from, Cluster* cluster_to) {
    int from = cluster_from->cycles_graph_node_id();
    int to = cluster_to->cycles_graph_node_id();

    if (!cycles_graph_.ContractEdge(from, to)) {
      VLOG(3) << "Could not contract " << cluster_from->DebugString(*graph_)
              << " -> " << cluster_to->DebugString(*graph_)
              << " because contracting the edge would create a cycle via "
              << DescribePotentialCycle(from, to) << ".";
      return false;
    }

    // Merge the clusters.
    cluster_from->Merge(cluster_to);

    // Merge the UnionFind<Cluster*>.
    cluster_for_node_[from].Merge(&cluster_for_node_[to]);

    return true;
  }

  string EdgeContractionFailureMsg(Cluster* from, Cluster* to,
                                   absl::string_view reason) {
    return absl::StrCat("Could not contract ", from->DebugString(*graph_),
                        " -> ", to->DebugString(*graph_), " because ", reason,
                        ".");
  }

  DebugOptions debug_options_;
  Graph* graph_;
  FunctionLibraryDefinition* flib_def_;
  Env* env_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  absl::flat_hash_map<const Cluster*, bool> should_compile_cluster_cache_;
  jit::DeviceInfoCache device_info_cache_;

  bool initialized_ = false;
  bool edges_contracted_ = false;
  bool clusters_created_ = false;

  std::vector<std::unique_ptr<Cluster>> cluster_storage_;
  std::vector<UnionFind<Cluster*>> cluster_for_node_;
  GraphCycles cycles_graph_;
  OrderedNodeSet compilation_candidates_;
  std::unique_ptr<DeadnessAnalysis> deadness_analysis_;
  int64 iteration_count_ = 0;
  absl::flat_hash_set<std::pair<int, int>> unsafe_resource_deps_;
};

std::vector<int> MarkForCompilationPassImpl::FindAlternatePathForDebugging(
    int from, int to) {
  std::vector<int> rpo = cycles_graph_.AllNodesInPostOrder();
  absl::c_reverse(rpo);

  // best_pred_for_node[n] contains a predecessor of `n` that has an
  // unclusterable node in some path from `from` to itself.
  // best_pred_for_node[n] is unpopulated for nodes that are not reachable from
  // `from`.  We build this table up inductively by traversing the cycles graph
  // in RPO.
  absl::flat_hash_map<int, int> best_pred_for_node;
  best_pred_for_node[from] = -1;

  int rpo_index = 0, current_rpo_node;
  do {
    current_rpo_node = rpo[rpo_index++];
    absl::optional<int> some_pred, preferred_pred;
    for (int pred : cycles_graph_.Predecessors(current_rpo_node)) {
      if (!best_pred_for_node.contains(pred)) {
        continue;
      }

      // Ignore the from->to edge since we're trying to find an alternate path.
      if (current_rpo_node == to && pred == from) {
        continue;
      }

      some_pred = pred;
      if (GetClusterForCyclesGraphNode(pred) == nullptr) {
        preferred_pred = pred;
      }
    }

    if (some_pred || preferred_pred) {
      best_pred_for_node[current_rpo_node] =
          preferred_pred.has_value() ? *preferred_pred : *some_pred;
    }
  } while (current_rpo_node != to);

  auto get_best_pred = [&](int n) {
    auto it = best_pred_for_node.find(n);
    CHECK(it != best_pred_for_node.end());
    return it->second;
  };

  std::vector<int> path;
  int current_path_node = get_best_pred(to);
  while (current_path_node != from) {
    path.push_back(current_path_node);
    current_path_node = get_best_pred(current_path_node);
  }

  absl::c_reverse(path);
  return path;
}

string MarkForCompilationPassImpl::DebugStringForCyclesGraphNode(
    int cycles_graph_node_id, bool* found_unclustered) {
  Cluster* cluster = GetClusterForCyclesGraphNode(cycles_graph_node_id);
  if (cluster) {
    return cluster->DebugString(*graph_);
  }

  *found_unclustered = true;
  if (cycles_graph_node_id >= graph_->num_node_ids()) {
    return absl::StrCat("<oob #", cycles_graph_node_id, ">");
  }

  Node* node = graph_->FindNodeId(cycles_graph_node_id);
  if (!node) {
    return absl::StrCat("<bad #", cycles_graph_node_id, ">");
  }

  return node->name();
}

string MarkForCompilationPassImpl::DescribePotentialCycle(int from, int to) {
  std::vector<string> path_str;
  bool found_unclustered = false;
  absl::c_transform(FindAlternatePathForDebugging(from, to),
                    std::back_inserter(path_str), [&](int node_id) {
                      return DebugStringForCyclesGraphNode(node_id,
                                                           &found_unclustered);
                    });
  return absl::StrCat(!found_unclustered ? "(all clusters) " : "", "[",
                      absl::StrJoin(path_str, ","), "]");
}

void MarkForCompilationPassImpl::Cluster::Merge(Cluster* other) {
  // We keep our own cycles_graph_node_id_ to mirror what GraphCycles does.

  // Clearing out data structures in `other` is just a memory saving
  // optimization and not needed for correctness.

  cluster_size_ += other->cluster_size_;
  effective_cluster_size_ += other->effective_cluster_size_;
  has_functional_control_flow_ |= other->has_functional_control_flow_;

  devices_.UnionWith(other->devices_);

  DCHECK(!(resource_op_device_.has_value() &&
           other->resource_op_device_.has_value()) ||
         *resource_op_device_ == *other->resource_op_device_)
      << "AreDevicesCompatible should have returned false otherwise!";

  if (!resource_op_device_.has_value()) {
    resource_op_device_ = other->resource_op_device_;
  }

  is_xla_compile_attr_true_ |= other->is_xla_compile_attr_true_;

  if (!xla_scope_.has_value()) {
    xla_scope_ = std::move(other->xla_scope_);
  }

  resource_var_operation_node_ids_.reserve(
      resource_var_operation_node_ids_.size() +
      other->resource_var_operation_node_ids_.size());
  absl::c_copy(other->resource_var_operation_node_ids_,
               std::back_inserter(resource_var_operation_node_ids_));
  other->resource_var_operation_node_ids_.clear();
}

Status IgnoreResourceOpForSafetyAnalysis(
    jit::DeviceInfoCache* device_info_cache, const Node& n, bool* ignore) {
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

StatusOr<bool> MarkForCompilationPassImpl::Initialize() {
  TF_RET_CHECK(!initialized_ && !edges_contracted_ && !clusters_created_);
  initialized_ = true;

  TF_RETURN_IF_ERROR(FindCompilationCandidates());

  if (compilation_candidates_.empty()) {
    VLOG(2) << "No compilable candidates";
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok,
                      CreateCycleDetectionGraph(graph_, &cycles_graph_));
  if (!cycle_detection_graph_ok) {
    // TODO(sanjoy): This should be logged via the XLA activity listener.
    VLOG(2) << "Could not form cycle detection graph";
    return false;
  }

  if (!debug_options_.ignore_deadness_checks) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL("DeadnessAnalysis", 1);
    TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph_, &deadness_analysis_));
  }

  // Each compilation candidate belongs to a cluster. The cluster's
  // representative names the node in the 'cycles' graph that represents the
  // cluster.
  TF_RETURN_IF_ERROR(BuildInitialClusterSet());
  return true;
}

template <typename FnTy>
StatusOr<bool> MarkForCompilationPassImpl::ForEachEdgeInPostOrder(FnTy fn) {
  bool changed = false;
  for (int32 node : cycles_graph_.AllNodesInPostOrder()) {
    Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
    if (!cluster_from) {
      continue;
    }

    // Make a copy of the set of successors because we may modify the graph in
    // TryToContractEdge.
    std::vector<int32> successors_copy =
        cycles_graph_.SuccessorsCopy(cluster_from->cycles_graph_node_id());

    for (int to : successors_copy) {
      iteration_count_++;

      Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
      if (!cluster_to) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(bool contracted_edge, fn(cluster_from, cluster_to));
      changed |= contracted_edge;
    }
  }

  return changed;
}

Node* MarkForCompilationPassImpl::GetOnlyNodeIn(const Cluster& cluster) {
  return cluster.cluster_size() == 1
             ? graph_->FindNodeId(cluster.GetIdOfOnlyNode())
             : nullptr;
}

bool MarkForCompilationPassImpl::IsSinkLike(const Cluster& cluster) {
  if (Node* n = GetOnlyNodeIn(cluster)) {
    return n->type_string() == "NoOp" && n->out_edges().size() == 1 &&
           (*n->out_edges().begin())->dst()->IsSink();
  }

  return false;
}

bool MarkForCompilationPassImpl::IsScalarIntegerResourceOperation(
    const Cluster& cluster) {
  Node* n = GetOnlyNodeIn(cluster);
  if (!n) {
    return false;
  }

  if (n->type_string() != "AssignAddVariableOp" &&
      n->type_string() != "AssignSubVariableOp") {
    return false;
  }

  DataType dtype;
  if (!TryGetNodeAttr(n->def(), "dtype", &dtype) || !DataTypeIsInteger(dtype)) {
    return false;
  }

  Node* const_input = nullptr;
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge() && e->src()->IsConstant()) {
      const_input = e->src();
      break;
    }
  }

  if (!const_input) {
    return false;
  }

  const TensorProto* proto = nullptr;
  if (!TryGetNodeAttr(const_input->def(), "value", &proto)) {
    return false;
  }

  return TensorShapeUtils::IsScalar(proto->tensor_shape());
}

Status MarkForCompilationPassImpl::RunEdgeContractionLoop() {
  TF_RET_CHECK(initialized_ && !edges_contracted_ && !clusters_created_);
  edges_contracted_ = true;

  // TODO(hpucha): Handle the case where kXlaClusterAttr is already set (for
  // example, from the Grappler fusion pass).

  // In general there are multiple maximal clusterings, but they are not all
  // equally performant.  Some clustering decision are likely to improve
  // performance much more than others, and we cannot order contractions on this
  // cost function, nor can we look at global information while deciding on
  // individual edges to contract.  Instead, we will make decisions on these
  // important edges then make decisions on all other edges, causing the highest
  // chance of all most important edges to be contracted.
  //
  // An example of where this might occur is with a digraph:
  // {A -> B, B -> C, A -> X, X -> C} where B is a Size operation and X is
  // not-compilable. In this case, the valid clusterings are {A,B} or {B,C}. B
  // should be clustered with A because it will prevent a potentially large
  // tensor from A being computed and copied.
  //
  // To choose better maximal clusterings we make multiple iterations over the
  // graph in post-order, where each such iteration is called a "phase".

  // Phase 0: contract metadata operations with their producer.

  VLOG(4) << "Running phase 0";
  TF_RETURN_IF_ERROR(
      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) -> StatusOr<bool> {
        // Shape consuming operations are desirable to cluster with their
        // operands because they return a small set of scalar values after
        // consuming a large amount of data.  For example, given a graph X -> Y
        // -> Size -> Z, where the possible clustering is [{X, Y, Size}, {Z}] or
        // [{X, Y}, {Size, Z}], the better clustering is Size with Y because the
        // output of size will be a small tensor while Y is a potentially large
        // tensor that must be computed and possible transposed/copied before
        // the second cluster executes.
        Node* n = GetOnlyNodeIn(*to);
        bool is_shape_consumer_op = n && IsShapeConsumerOp(*n);
        if (!is_shape_consumer_op) {
          return false;
        }

        return TryToContractEdge(from, to);
      }).status());

  // Phase 1: apply a heuristic to ensure that we don't mess up clustering due
  // to "group_deps".  After this phase most edges should have been contracted.

  VLOG(4) << "Running phase 1";
  TF_RETURN_IF_ERROR(
      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) -> StatusOr<bool> {
        // We split out this phase to get good clustering in the presence of a
        // specific pattern seen in some graphs:
        //
        // digraph {
        //   ApplyWeightUpdates_0 -> "iteration++"
        //   ApplyWeightUpdates_1 -> "iteration++"
        //   ApplyWeightUpdates_2 -> "iteration++"
        //   ApplyWeightUpdates_0 -> Computation_A
        //   ApplyWeightUpdates_1 -> Computation_B
        //   ApplyWeightUpdates_2 -> Computation_C
        //   Computation_A -> NoOp
        //   Computation_B -> NoOp
        //   Computation_C -> NoOp
        //   "iteration++" -> NoOp
        // }
        //
        // In the graph above we can't cluster iteration++ with any of the
        // gradient update operations since that will break the TF resource
        // variable memory model.  Given that constraint the ideal clustering
        // would be to put all the gradient updates and all of the Computation_*
        // nodes in one cluster, and leave iteration++ and NoOp unclustered.
        //
        // A naive post-order traversal would not create this good clustering,
        // however.  Instead it will first create a cluster that puts
        // Computation_* nodes, the NoOp and iteration++ node in a single
        // cluster, after which it will fail to put any of the
        // ApplyWeightUpdates_* nodes into this cluster. To avoid this fate we
        // instead run a pass that avoids contracting edges _into_ NoOps like
        // the above, and avoid clustering edges _from_ "iteration++" like the
        // above.  Then we run a second pass that contracts the edges we could
        // not contract the first time around.

        if (IsSinkLike(*to)) {
          return false;
        }

        if (IsScalarIntegerResourceOperation(*from)) {
          return false;
        }

        return TryToContractEdge(from, to);
      }).status());

  // Phase 2: contract any remaining edges.  After this phase we should have a
  // maximal clustering:
  //
  // A. We visit a cluster only after maximally clustering all its children.
  // B. By the time we're done with a node all of its children that could have
  //    been absorbed into the node have been absorbed.
  // C. We have an invariant that making a cluster larger does not make edges
  //    leaving it more contractable. That is, if we have
  //    digraph { X->Y; Y->Z; } then collapsing X->Y does not make it possible
  //    to contract Y->Z if Y->Z was not contractible originally.
  VLOG(4) << "Running phase 2";
  TF_RETURN_IF_ERROR(ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) {
                       return TryToContractEdge(from, to);
                     }).status());

  // Check that the conclusion made above (that iterating over the graph once in
  // post order gives a maximal clustering) holds.  Once the linear time
  // post-order scheme has been battle tested we can move this to happen only in
  // debug builds.
  VLOG(2) << "Checking idempotence";
  TF_ASSIGN_OR_RETURN(bool changed,
                      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) {
                        return TryToContractEdge(from, to);
                      }));
  TF_RET_CHECK(!changed);

  return Status::OK();
}

std::atomic<int64> cluster_sequence_num;

int64 GetNextClusterSequenceNumber() { return cluster_sequence_num++; }

Status MarkForCompilationPassImpl::CreateClusters() {
  TF_RET_CHECK(initialized_ && edges_contracted_ && !clusters_created_);
  clusters_created_ = true;

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
  for (Node* n : compilation_candidates_) {
    Cluster* cluster = GetClusterForNode(n);
    TF_ASSIGN_OR_RETURN(bool should_compile_cluster,
                        ShouldCompileCluster(*cluster));
    if (!should_compile_cluster) {
      continue;
    }

    // We assume that functional If and While nodes have at least
    // min_cluster_size non-trivial nodes in them.  It would be more principled
    // to (recursively) verify this fact, but that's probably not worth the
    // trouble.

    if (cluster->effective_cluster_size() >= debug_options_.min_cluster_size ||
        cluster->has_functional_control_flow() ||
        cluster->is_xla_compile_attr_true()) {
      string& name = cluster_names[cluster->cycles_graph_node_id()];

      if (name.empty()) {
        name = absl::StrCat("cluster_", GetNextClusterSequenceNumber());
      }

      n->AddAttr(kXlaClusterAttr, name);
      n->AddAttr(kXlaAlreadyClustered, true);
      VLOG(3) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }

  return Status::OK();
}

Status MarkForCompilationPassImpl::DumpDebugInfo() {
  TF_RET_CHECK(initialized_ && edges_contracted_ && clusters_created_);

  if (debug_options_.dump_graphs) {
    DumpPostClusteringGraphs();
  }

  VLogClusteringSummary();

  return Status::OK();
}

StatusOr<bool>
MarkForCompilationPassImpl::ClusteringWillIntroduceInterDeviceDependency(
    const Cluster& cluster_from, const Cluster& cluster_to) {
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
  for (const auto& in_id :
       cycles_graph_.Predecessors(cluster_to.cycles_graph_node_id())) {
    const Cluster* cluster_in = GetClusterForCyclesGraphNode(in_id);
    if (cluster_in) {
      TF_ASSIGN_OR_RETURN(bool devices_compatible,
                          AreDevicesCompatible(cluster_to, *cluster_in));
      if (!devices_compatible) {
        return true;
      }
      TF_ASSIGN_OR_RETURN(devices_compatible,
                          AreDevicesCompatible(cluster_from, *cluster_in));
      if (!devices_compatible) {
        return true;
      }
    }
  }

  return false;
}

absl::optional<string> MarkForCompilationPassImpl::GetXlaScope(Node* node) {
  // Look for either _XlaScope or _XlaInternalScope on both nodes to guide
  // clustering.  If both nodes have a scope and the scopes do not match, do
  // not cluster along this edge.  If even one of the nodes lacks a scope
  // attribute, then it is treated as a "bridge" and a cluster may be created
  // along it.
  //
  // The difference between _XlaScope and _XlaInternalScope is that _XlaScope is
  // provided by users through jit_scope APIs, while _XlaInternalScope is
  // automatically generated by the ClusterScopingPass when auto_jit is on.  As
  // such, we respect _XlaScope only when auto_jit is off, while respecting
  // _XlaInternalScope only when auto_jit is on.
  //
  // We may want to restrict the _XlaScope behavior to require all nodes marked
  // with _XlaCompile=true to also have a _XlaScope property set (and raise an
  // error otherwise); but for now we don't do this.

  if (global_jit_level_ != OptimizerOptions::OFF) {
    // If global_jit_level_ is ON, respect only _XlaInternalScope.
    const string& scope =
        GetNodeAttrString(node->attrs(), kXlaInternalScopeAttr);
    if (!scope.empty()) {
      return scope;
    }
  } else {
    // If global_jit_level_ is OFF, respect only _XlaScope.
    const string& scope = GetNodeAttrString(node->attrs(), kXlaScopeAttr);
    if (!scope.empty()) {
      return scope;
    }
  }

  return absl::nullopt;
}

Status MarkForCompilationPassImpl::BuildInitialClusterSet() {
  auto ignore_resource_ops = [&](const Node& n, bool* ignore) {
    return IgnoreResourceOpForSafetyAnalysis(&device_info_cache_, n, ignore);
  };

  std::vector<std::pair<int, int>> unsafe_resource_deps_vect;
  TF_RETURN_IF_ERROR(ComputeIncompatibleResourceOperationPairs(
      *graph_, flib_def_, ignore_resource_ops, &unsafe_resource_deps_vect));
  absl::c_copy(
      unsafe_resource_deps_vect,
      std::inserter(unsafe_resource_deps_, unsafe_resource_deps_.begin()));

  cluster_for_node_.resize(graph_->num_node_ids());
  for (Node* node : graph_->nodes()) {
    if (!IsCompilationCandidate(node)) {
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }

    // We want clusters to be big enough that the benefit from XLA's
    // optimizations offsets XLA related overhead (for instance we add some
    // Switch/Merge nodes into the graph to implement lazy compilation).  To
    // this end, we don't count Identity and Constant nodes because they do not
    // enable interesting optimizations by themselves.
    int effective_cluster_size =
        (node->IsIdentity() || node->IsConstant()) ? 0 : 1;

    bool has_functional_control_flow = node->IsWhileNode() || node->IsIfNode();

    absl::optional<DeadnessPredicate> deadness_predicate;
    if (deadness_analysis_) {
      TF_ASSIGN_OR_RETURN(
          deadness_predicate,
          deadness_analysis_->GetPredicateFor(node, Graph::kControlSlot));
    }

    const string& device_name_str = !node->assigned_device_name().empty()
                                        ? node->assigned_device_name()
                                        : node->requested_device();
    TF_ASSIGN_OR_RETURN(DeviceId device,
                        device_info_cache_.GetIdFor(device_name_str));

    bool is_resource_op = HasResourceInputOrOutput(*node);
    absl::optional<DeviceId> resource_op_device;
    if (is_resource_op) {
      resource_op_device = device;
    }

    absl::optional<int> resource_var_operation_node_id;
    if (is_resource_op || MayCallFunction(*node, flib_def_)) {
      resource_var_operation_node_id = node->id();
    }

    bool is_xla_compile_attr_true = false;

    bool xla_compile_attr;
    if (TryGetNodeAttr(node->attrs(), kXlaCompileAttr, &xla_compile_attr)) {
      is_xla_compile_attr_true |= xla_compile_attr;
    }

    if (flib_def_->GetAttr(*node, kXlaCompileAttr, &xla_compile_attr).ok()) {
      is_xla_compile_attr_true |= xla_compile_attr;
    }

    DeviceSet devices;
    devices.Insert(device);

    Cluster* new_cluster = MakeNewCluster(
        /*cycles_graph_node_id=*/node->id(),
        /*effective_cluster_size=*/effective_cluster_size,
        /*has_functional_control_flow=*/has_functional_control_flow, devices,
        resource_op_device, resource_var_operation_node_id, deadness_predicate,
        /*is_xla_compile_attr_true=*/is_xla_compile_attr_true,
        GetXlaScope(node));

    cluster_for_node_[node->id()].Get() = new_cluster;
  }

  return Status::OK();
}

StatusOr<bool> IsIdentityDrivingConstsInLoop(Node* node) {
  if (!node->IsIdentity()) {
    return false;
  }

  // Check if the Identity is driven by a Switch on its true path.
  auto it = absl::c_find_if(node->in_edges(), [](const Edge* e) {
    return e->src()->IsSwitch() && e->src_output() == 1;
  });
  if (it == node->in_edges().end()) {
    return false;
  }
  const Node* switch_node = (*it)->src();

  // Check if the Switch is driven by LoopCond.
  const Node* maybe_loop_cond;
  TF_RETURN_IF_ERROR(switch_node->input_node(1, &maybe_loop_cond));
  if (!maybe_loop_cond->IsLoopCond()) {
    return false;
  }

  // Check if the Identity is driving any const nodes through a control edge.
  bool driving_any_consts =
      absl::c_any_of(node->out_edges(), [](const Edge* e) {
        return e->dst()->IsConstant() && e->IsControlEdge();
      });
  if (!driving_any_consts) {
    return false;
  }

  return true;
}

const absl::flat_hash_map<string, std::vector<string>> whitelist_table = {
    // Unary
    {"PW",
     {"ComplexAbs", "Angle", "Conj", "Abs", "Acos", "Acosh", "Asin", "Atan",
      "Atanh", "Ceil", "Cos", "Cosh", "Sin", "Exp", "Expm1", "Floor",
      "IsFinite", "IsInf", "IsNan", "Inv", "Reciprocal", "Log", "Log1p",
      "Invert", "LogicalNot", "Neg", "Rint", "Round", "Rsqrt", "Sigmoid",
      "Sign", "Sinh", "Softplus", "Softsign", "Sqrt", "Square", "Tan", "Tanh",
      "Real", "Imag", "Erf", "Erfc", "Lgamma", "Digamma",
      // Binary
      "Add", "AddV2", "Sub", "Mul", "Div", "Atan2", "Complex", "DivNoNan",
      "MulNoNan", "FloorDiv", "Xlogy", "Xdivy", "FloorMod", "BitwiseAnd",
      "BitwiseOr", "BitwiseXor", "LeftShift", "RightShift", "LogicalAnd",
      "LogicalOr", "Mod", "Maximum", "Minimum", "RealDiv", "ReciprocalGrad",
      "RsqrtGrad", "SqrtGrad", "TruncateDiv", "TruncateMod", "Equal",
      "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual", "SigmoidGrad",
      "SoftplusGrad", "SoftsignGrad", "TanhGrad", "Pow", "SquaredDifference",
      "ApproximateEqual",
      // Others
      "AddN", "Bitcast", "Cast", "ClipByValue", "Const", "Empty", "Identity",
      "IdentityN", "Relu", "Relu6", "ReluGrad", "Relu6Grad", "LeakyReluGrad",
      "Elu", "EluGrad", "Selu", "SeluGrad", "Select", "SelectV2", "Transpose",
      "ConjugateTranspose", "_UnaryOpsComposition",
      // The following 4 operations are converted to identity
      "PlaceholderWithDefault", "PreventGradient", "StopGradient", "Snapshot"}},
    // clang-format off
    {"RED",
     {"All", "Any", "Min", "Max", "Mean", "Prod", "Sum"}},
    // clang-format on
    {"PWRED",
     {"ArgMax", "ArgMin", "DiagPart", "Softmax",
      "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
    {"REDUCEWINDOW",
     {"ArgMax", "ArgMin", "DiagPart", "Softmax",
      "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
    {"REDUCEWINDOWPW", {"BiasAddGrad", "LRN", "LRNGrad"}},
    {"BN",
     {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3",
      "_FusedBatchNormEx", "FusedBatchNormGrad", "FusedBatchNormGradV2",
      "FusedBatchNormGradV3"}},
    {"SORT", {"TopKV2"}}, // XLA version much faster then TF version.
    {"SMALL",
     // clang-format off
     {"BroadcastTo", "ExpandDims", "Fill", "NoOp",
      "Range", "Rank", "Reshape", "Shape", "ShapeN", "Size", "Squeeze",
      "Transpose", "ZerosLike", "OnesLike", "BiasAdd" /*PW + Broadcast*/,
      "BroadcastArgs", "BroadcastGradientArgs", "OneHot", "Concat", "ConcatV2",
      "ConcatOffset", "Const", "MirrorPad", "Pack", "Pad", "PadV2", "Reverse",
      "ReverseV2", "ReverseSequence", "Slice", "Split", "SplitV",
      "StridedSlice", "StridedSliceGrad", "ResourceStridedSliceAssign",
      "Tile", "Transpose", "InvertPermutation", "Unpack"}}};
// clang-format on

absl::flat_hash_set<string> GetWhitelist() {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  absl::flat_hash_set<string> whitelist;

  for (auto s : absl::StrSplit(flags->tf_xla_ops_to_cluster, ",")) {
    if (s == "FUSIBLE") {
      for (auto pair : whitelist_table) {
        whitelist.insert(pair.second.begin(), pair.second.end());
      }
    } else if (whitelist_table.contains(s)) {
      auto v = whitelist_table.at(s);
      whitelist.insert(v.begin(), v.end());
    } else if (!s.empty()) {
      // Should be a user provided TF operation.
      whitelist.insert(string(s));
    }
  }

  if (VLOG_IS_ON(2) && !whitelist.empty()) {
    std::vector<string> vwhitelist(whitelist.begin(), whitelist.end());
    absl::c_sort(vwhitelist);
    VLOG(2) << "XLA clustering will only consider the following TF operations: "
            << absl::StrJoin(vwhitelist, " ");
  }
  return whitelist;
}

Status MarkForCompilationPassImpl::FindCompilationCandidates() {
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, env_, /*config=*/nullptr,
                                        TF_GRAPH_DEF_VERSION, flib_def_, opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  std::vector<bool> compile_time_const_nodes(graph_->num_node_ids(), false);
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
      *graph_, /*compile_time_const_arg_indices=*/nullptr,
      &compile_time_const_nodes, lib_runtime));
  // Iterate over nodes in sorted order so that compiler fuel is deterministic.
  // We can't simply pass op_nodes().begin() and op_nodes().end() to the
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

  auto whitelist = GetWhitelist();

  std::vector<string> vall_ops = XlaOpRegistry::GetAllRegisteredOps();
  absl::flat_hash_set<string> all_ops(vall_ops.begin(), vall_ops.end());
  // Check that user's provided TF operation really exists.
  for (auto s: whitelist) {
    if (!all_ops.contains(string(s))) {
      return errors::InvalidArgument(
          "The operation '", s,
          "' passed to --tf_xla_ops_to_cluster is not supported by XLA.");
    }
  }

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

    if (CompilationDisallowedByXlaCompileAttr(node)) {
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

    if (whitelist.size() > 0 && !whitelist.contains(node->def().op())) {
      VLOG(1) << "Rejecting " << node->name()
              << " as it is not listed in --tf_xla_ops_to_cluster.";
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
          continue;
        }
      }
    }

    // This is a heuristic to avoid creating dependency between while loop
    // condition and body computations.  Dependency between them can be created
    // if a special Identity node in the following pattern is clustered in.
    // That is, an Identity node in the loop cond computation is used to drive
    // const nodes consumed by the loop body.  If this Identity node goes into
    // the same cluster with nodes from the loop body, extra dependency is
    // created between the loop cond and body computations and it hinders the
    // progression of the loop cond computation at runtime with significant
    // overhead.  Specifically, we look for the below pattern and do not cluster
    // in this Identity to avoid the described issue.  Since Identity has low
    // execution cost in native TF, the fact that this heuristic gives up these
    // special Identity nodes as candidates should not harm any performance.  If
    // other considerations emerge in the future, we can revisit the heuristic
    // and only disallow these Identities to go into the cluster with nodes from
    // the loop body but still consider them candidates.
    //
    // LoopCond ->
    // Merge    -> Switch -> Identity -> i++ -> ... -> NextIteration
    //                               ..> Const -> LoopBody
    //                            (control edge)
    TF_ASSIGN_OR_RETURN(bool is_identity_driving_consts_in_loop,
                        IsIdentityDrivingConstsInLoop(node));
    if (is_identity_driving_consts_in_loop) {
      VLOG(2) << "Rejecting " << node->name()
              << ": including it can create dependencies between while loop "
                 "condition and body computations with runtime overhead.";
      continue;
    }

    compilation_candidates_.insert(node);
    --(*debug_options_.fuel);
  }

  VLOG(2) << "compilation_candidates_.size() = "
          << compilation_candidates_.size();
  return Status::OK();
}

bool MarkForCompilationPassImpl::CompilationDisallowedByXlaCompileAttr(
    Node* node) {
  if (debug_options_.ignore_xla_compile_attr) {
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

bool MarkForCompilationPassImpl::LogNotContractableAndReturnFalse(
    Cluster* from, Cluster* to, absl::string_view reason) {
  VLOG(3) << EdgeContractionFailureMsg(from, to, reason);
  return false;
}

StatusOr<bool> MarkForCompilationPassImpl::TryToContractEdge(Cluster* from,
                                                             Cluster* to) {
  DCHECK(from->deadness_predicate().has_value() ==
         to->deadness_predicate().has_value());
  if (from->deadness_predicate() != to->deadness_predicate()) {
    VLOG(3) << EdgeContractionFailureMsg(
        from, to,
        absl::StrCat(
            "the two nodes have mismatching deadness: ",
            deadness_analysis_->DebugString(*from->deadness_predicate()),
            " and ",
            deadness_analysis_->DebugString(*to->deadness_predicate())));
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool devices_compatible,
                      AreDevicesCompatible(*from, *to));
  if (!devices_compatible) {
    return LogNotContractableAndReturnFalse(
        from, to, "the two nodes have incompatible devices");
  }

  if (from->xla_scope().has_value() && to->xla_scope().has_value() &&
      *from->xla_scope() != *to->xla_scope()) {
    return LogNotContractableAndReturnFalse(
        from, to, "the two nodes have mismatching XLA scopes");
  }

  // Don't exceed the maximum cluster size.
  if (from->cluster_size() + to->cluster_size() >
      debug_options_.max_cluster_size) {
    return LogNotContractableAndReturnFalse(
        from, to, "the new cluster will be larger than the max cluster size");
  }

  TF_ASSIGN_OR_RETURN(bool will_introduce_cross_device_dependency,
                      ClusteringWillIntroduceInterDeviceDependency(*from, *to));

  if (will_introduce_cross_device_dependency) {
    return LogNotContractableAndReturnFalse(
        from, to, "the new cluster will introduce a cross device dependency");
  }

  // Check if contracting this edge will break the resource variable concurrency
  // semantics.  In theory this is quadratic in the number of nodes, but seems
  // to not be a problem in practice so far.
  if (!debug_options_.ignore_resource_variable_checks) {
    for (int resource_var_from : from->resource_var_operation_node_ids()) {
      for (int resource_var_to : to->resource_var_operation_node_ids()) {
        // If unsafe_resource_deps_ contains {A, B} then
        //
        //  a. A and B are resource operations.
        //  b. A and B cannot be placed in the same cluster.
        //  c. There is no path from B to A in the cycles graph (but there may
        //     be a path from A to B).
        //
        // So check the legality of the edge contraction by checking if any of
        // the n^2 pairs of resource variable operations are forbidden.
        if (unsafe_resource_deps_.contains(
                {resource_var_from, resource_var_to})) {
          return LogNotContractableAndReturnFalse(
              from, to,
              "the new cluster would break resource variable semantics");
        }
      }
    }
  }

  return MergeClusters(from, to);
}

Status MarkForCompilationPassImpl::Run() {
  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();

  // Start the timer after XlaOpRegistry::RegisterCompilationKernels which does
  // some one-time work.
  XLA_SCOPED_LOGGING_TIMER_LEVEL("MarkForCompilationPassImpl::Run", 1);

  TF_ASSIGN_OR_RETURN(bool initialized, Initialize());
  if (!initialized) {
    // Initialization exited early which means this instance of
    // MarkForCompilationPassImpl is not set up to run the subsequent phases.
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(RunEdgeContractionLoop());
  TF_RETURN_IF_ERROR(CreateClusters());
  TF_RETURN_IF_ERROR(DumpDebugInfo());

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

  XlaAutoClusteringSummary auto_clustering_info =
      GetXlaAutoClusteringSummary(*graph_);

  VLOG(2) << "*** Clustering info for graph of size " << graph_->num_nodes();
  VLOG(2) << " Built " << auto_clustering_info.clusters_size()
          << " clusters, size "
          << RatioToString(auto_clustering_info.clustered_node_count(),
                           graph_->num_nodes());

  for (XlaAutoClusteringSummary::Cluster cluster :
       auto_clustering_info.clusters()) {
    absl::string_view cluster_name = cluster.name();
    int size = cluster.size();
    VLOG(2) << "  " << cluster_name << " "
            << RatioToString(size, graph_->num_nodes());
    for (const XlaAutoClusteringSummary::OpAndCount& op_count :
         cluster.op_histogram()) {
      VLOG(3) << "   " << op_count.op() << ": " << op_count.count()
              << " instances";
    }
  }

  if (!auto_clustering_info.unclustered_op_histogram().empty()) {
    VLOG(2) << " Unclustered nodes: "
            << RatioToString(auto_clustering_info.unclustered_node_count(),
                             graph_->num_nodes());
    for (const XlaAutoClusteringSummary::OpAndCount& op_count :
         auto_clustering_info.unclustered_op_histogram()) {
      VLOG(3) << "  " << op_count.op() << ": " << op_count.count()
              << " instances";
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

  VLOG(4) << "*** Inter-Cluster edges:";
  if (cluster_names_to_print.empty()) {
    VLOG(4) << "   [none]";
  }

  auto print_edge_info_set_for_cluster = [&](absl::string_view cluster_name,
                                             const EdgeInfoMap& edge_info_map,
                                             absl::string_view desc) {
    auto it = edge_info_map.find(cluster_name);
    if (it != edge_info_map.end()) {
      VLOG(4) << "  " << it->second.size() << " " << desc << " edges";
      for (const auto& edge_info_count_pair : it->second) {
        VLOG(4) << "   " << edge_info_count_pair.first.GetClusterName() << " "
                << edge_info_count_pair.first.node_name << " # "
                << edge_info_count_pair.second;
      }
    } else {
      VLOG(4) << "  No " << desc << " edges.";
    }
  };

  for (absl::string_view cluster_name : cluster_names_to_print) {
    VLOG(4) << " ** Cluster " << cluster_name;
    print_edge_info_set_for_cluster(cluster_name, incoming_edge_infos,
                                    "incoming");
    print_edge_info_set_for_cluster(cluster_name, outgoing_edge_infos,
                                    "outgoing");
  }
}

StatusOr<bool> MarkForCompilationPassImpl::AreDevicesCompatible(
    const Cluster& cluster_a, const Cluster& cluster_b) {
  DeviceSet devices = cluster_a.devices();
  devices.UnionWith(cluster_b.devices());

  TF_ASSIGN_OR_RETURN(
      absl::optional<jit::DeviceId> maybe_chosen_device,
      MaybePickDeviceForXla(device_info_cache_, devices,
                            /*allow_mixing_unknown_and_cpu=*/false));
  if (!maybe_chosen_device.has_value()) {
    return false;
  }

  jit::DeviceId chosen_device = *maybe_chosen_device;

  // If we are able to pick a device `chosen_device` for the larger cluster, the
  // resource operations in `cluster_a` and `cluster_b` must be placed on the
  // same device as `chosen_device`.  This is because the _XlaCompile and
  // _XlaRun kernels are going to run on and therefore try to access the
  // resource variables from `chosen_device`, which will be an error if the
  // resource variables are placed on some other device.
  auto resource_op_device_ok =
      [&](absl::optional<DeviceId> resource_op_device) {
        return !resource_op_device.has_value() ||
               *resource_op_device == chosen_device;
      };

  return resource_op_device_ok(cluster_a.resource_op_device()) &&
         resource_op_device_ok(cluster_b.resource_op_device());
}

// Returns `true` iff we should compile `cluster`.
StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileClusterImpl(
    const Cluster& cluster) {
  TF_ASSIGN_OR_RETURN(DeviceId chosen_device,
                      PickDeviceForXla(device_info_cache_, cluster.devices(),
                                       /*allow_mixing_unknown_and_cpu=*/false));

  const DeviceType& device_type =
      device_info_cache_.GetDeviceTypeFor(chosen_device);
  const XlaOpRegistry::DeviceRegistration* registration =
      device_info_cache_.GetCompilationDevice(chosen_device);
  TF_RET_CHECK(registration)
      << "chosen device = " << device_info_cache_.GetNameFor(chosen_device)
      << "; device type = " << device_type.type() << "; devices ("
      << device_info_cache_.DebugString(cluster.devices());

  bool should_compile =
      cluster.is_xla_compile_attr_true() ||
      registration->autoclustering_policy ==
          XlaOpRegistry::AutoclusteringPolicy::kAlways ||
      (registration->autoclustering_policy ==
           XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
       global_jit_level_ != OptimizerOptions::OFF);

  if (!should_compile && global_jit_level_ != OptimizerOptions::OFF &&
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
          << " cluster with device "
          << device_info_cache_.GetNameFor(chosen_device);

  return should_compile;
}

StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileCluster(
    const Cluster& cluster) {
  auto it = should_compile_cluster_cache_.find(&cluster);
  if (it != should_compile_cluster_cache_.end()) {
    return it->second;
  }

  TF_ASSIGN_OR_RETURN(bool should_compile, ShouldCompileClusterImpl(cluster));
  should_compile_cluster_cache_.insert({&cluster, should_compile});
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

bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef,
                  RecursiveCompilabilityChecker::UncompilableNodesMap*
                      uncompilable_node_info) {
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
  op_filter.allow_slow_ops = true;
  op_filter.allow_inaccurate_ops = true;

  RecursiveCompilabilityChecker checker{&op_filter, &jit_device_type};
  if (!uncompilable_node_info) {
    // We do not need uncompilable node info. Just return the result.
    return checker.IsCompilableCall(ndef, flr);
  }

  RecursiveCompilabilityChecker::UncompilableNodesMap uncompilable_node_result =
      checker.FindUncompilableNodes(ndef, flr);
  uncompilable_node_info->swap(uncompilable_node_result);
  return uncompilable_node_info->empty();
}

Status MarkForCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();

  MarkForCompilationPassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks =
      flags->tf_xla_disable_deadness_safety_checks_for_debugging;
  debug_options.ignore_resource_variable_checks =
      flags->tf_xla_disable_resource_variable_safety_checks_for_debugging;
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
  debug_options.ignore_resource_variable_checks =
      flags->tf_xla_disable_resource_variable_safety_checks_for_debugging;
  debug_options.ignore_xla_compile_attr = true;
  debug_options.max_cluster_size = flags->tf_xla_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_xla_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_xla_clustering_fuel);
  debug_options.dump_graphs = flags->tf_xla_clustering_debug;

  return MarkForCompilation(options, debug_options);
}

namespace testing {
void ResetClusterSequenceNumber() { cluster_sequence_num = 0; }
}  // namespace testing
}  // namespace tensorflow
