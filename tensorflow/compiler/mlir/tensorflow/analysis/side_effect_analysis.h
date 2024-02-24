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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_SIDE_EFFECT_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_SIDE_EFFECT_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"

namespace mlir {
namespace TF {
using ResourceId = int64_t;
inline constexpr ResourceId kUnknownResourceId =
    ResourceAliasAnalysis::Info::kUnknownResourceId;
static_assert(kUnknownResourceId < 0, "kUnknownResourceId must be < 0");

// Maps group IDs to branch IDs.
using ParallelIdsMap = std::map<std::string, std::string>;
using OpToParallelIdsMap = absl::flat_hash_map<Operation*, ParallelIdsMap>;

namespace detail {

class OpSideEffectCollector;

using StackResourceToOps = std::vector<
    absl::flat_hash_map<ResourceId, absl::flat_hash_set<Operation*>>>;

// Side effect analysis info for a single function.
//
// This class provides an interface for querying control predecessors and
// successors for ops of the given function. This information is computed from
// side effects, using resource alias analysis where possible.
// Remarks:
// - Control dependencies model execution order constraints for side-effecting
//   ops. For example, two ops writing to the same resource cannot switch their
//   order and cannot be executed in parallel.
// - A control dependency (A,B) means that op A has to be executed before op B.
//   A is a control predecessor of B, and B is a control successor of A.
// - The control dependencies provided by side effect analysis are guaranteed to
//   be sufficient for correct execution but they are not guaranteed to be
//   minimal (that means, some control dependencies might not be required for
//   correct execution).
class SideEffectAnalysisInfo {
 public:
  SideEffectAnalysisInfo() = default;

  // Constructs analysis info by analyzing the given function.
  SideEffectAnalysisInfo(func::FuncOp func_op,
                         const OpSideEffectCollector& op_side_effect_collector,
                         const TF::ResourceAliasAnalysis::Info& alias_analysis,
                         const OpToParallelIdsMap& op_to_parallel_ids)
      : op_side_effect_collector_(op_side_effect_collector),
        alias_analysis_(alias_analysis),
        op_to_parallel_ids_(op_to_parallel_ids) {
    AnalyzeFunction(func_op);
  }

  // Constructs analysis info by analyzing the given region.
  SideEffectAnalysisInfo(Region* region,
                         const OpSideEffectCollector& op_side_effect_collector,
                         const TF::ResourceAliasAnalysis::Info& alias_analysis,
                         const OpToParallelIdsMap& op_to_parallel_ids)
      : op_side_effect_collector_(op_side_effect_collector),
        alias_analysis_(alias_analysis),
        op_to_parallel_ids_(op_to_parallel_ids) {
    AnalyzeRegion(region);
  }

  SideEffectAnalysisInfo(SideEffectAnalysisInfo&&) = default;

  // Returns a vector of ops that are direct control predecessors of `op`,
  // sorted in program order. If `filter` is provided, only predecessors that
  // pass the filter (returning true) will be included.
  const llvm::SmallVector<Operation*, 4>& DirectControlPredecessors(
      Operation* op) const;
  llvm::SmallVector<Operation*, 4> DirectControlPredecessors(
      Operation* op, llvm::function_ref<bool(Operation*)> filter) const;

  // pass the filter (returning true) will be included.
  const llvm::SmallVector<Operation*, 4>& DirectControlSuccessors(
      Operation* op) const;
  llvm::SmallVector<Operation*, 4> DirectControlSuccessors(
      Operation* op, llvm::function_ref<bool(Operation*)> filter) const;

  // Returns a vector of ops that are control sinks (i.e. side-effecting ops
  // with no control successors).
  llvm::ArrayRef<Operation*> ControlSinks() const {
    return sorted_control_sinks_;
  }

  // Returns a vector with IDs of all resources that might be accessed by `op`.
  // This includes both op-based and value-based resources. The bool indicates
  // whether a resource is accessed read-only.
  const llvm::SmallVector<std::pair<ResourceId, bool>>& GetResourceIds(
      Operation* op) const;

  // Returns true iff given resource is allocated by op with
  // `UniqueResourceAllocation` trait. This can be utilized for while-loop
  // parallelization.
  bool IsUniqueResourceAllocationId(ResourceId resource_id) const {
    return alias_analysis_.IsUniqueResourceAllocationId(resource_id);
  }

  const TF::ResourceAliasAnalysis::Info& GetAliasAnalysis() const {
    return alias_analysis_;
  }

 private:
  // Runs the analysis and populates `sorted_control_predecessors_` and
  // `sorted_control_successors_` for `func_op`. Clears `control_predecessors_`.
  void AnalyzeFunction(func::FuncOp func_op);

  // Runs the analysis and populates `control_predecessors_` for `region`.
  void AnalyzeRegion(Region* region);

  // Runs the analysis and populates `control_predecessors_` for `op`.
  void AnalyzeOp(Operation* op);

  // Updates `control_predecessors_` for given `resource_id` and `op`.
  void AddPredecessorsForAccess(ResourceId resource_id, Operation* op,
                                bool read_only);

  // Updates resource access for given `resource_id` and `op` in
  // `per_resource_access_info_` and `op_to_resource_ids_`.
  void UpdateAccess(ResourceId resource_id, Operation* op, bool read_only);

  // Returns true iff the last unknown resource access is already indirectly
  // tracked by a previous `resource` access. `read_only` specifies the type of
  // access considered.
  bool IsUnknownAccessIndirectlyTrackedByResource(ResourceId resource,
                                                  bool read_only);

  // Returns a set of resource IDs that have potential dependencies to
  // `resource_id` (i.e., there are potential dependencies between the
  // resources corresponding to the IDs).
  llvm::SmallSet<ResourceId, 8> GetDependentIds(ResourceId resource_id,
                                                bool is_fetch_op) const;

  // Returns the parallel ids of the op.
  ParallelIdsMap GetParallelIdsMap(Operation* op);

  // Converts from read/write state that relates ops with the same parallel id
  // to a set of last accesses for use with other parallel ids. Reads/writes
  // between parallel ids are conservatively approximated as writes.
  absl::flat_hash_set<Operation*> GetLastWrites(ResourceId resource_id);

  // Sets the read/write state for ops within the same parallel id.
  void SetLastWrites(ResourceId resource_id,
                     absl::flat_hash_set<Operation*> last_writes);

  // Enters a sequence of ops that have the same parallel id. This converts
  // stack state to per_resource_access_info_.
  void Enter();

  // Exits a sequence of ops that have the same parallel id. This converts
  // per_resource_access_info_ to stack state.
  void Exit();

  // Steps down one parallel nesting level (i.e. increase parallel id size
  // by 1).
  void Down();

  // Steps laterally between parallel nesting levels.
  void Lateral();

  // Steps up one parallel nesting level.
  void Up();

  // Transitions nesting levels from `from` to `to`.
  void Transition(ParallelIdsMap from, ParallelIdsMap to);

  // Transitions nesting levels from the previous parallel id to `to`.
  void TransitionToParallelIdsMap(ParallelIdsMap to);

  // Transitions nesting levels from the previous parallel id to `to`.
  void TransitionToOp(Operation* to);

  // Initializes stack state for a function.
  void InitFunction();

  // Uninitializes stack state for a function.
  void UninitFunction();

  // Maps from an op to its control predecessors.
  llvm::SmallDenseMap<Operation*, llvm::SmallPtrSet<Operation*, 4>, 8>
      control_predecessors_;
  // Maps from an op to its control predecessors sorted in program order.
  llvm::SmallDenseMap<Operation*, llvm::SmallVector<Operation*, 4>, 8>
      sorted_control_predecessors_;
  // Maps from an op to its control successors sorted in program order.
  llvm::SmallDenseMap<Operation*, llvm::SmallVector<Operation*, 4>, 8>
      sorted_control_successors_;
  // Side-effecting ops with no control successors in this function.
  llvm::SmallVector<Operation*, 4> sorted_control_sinks_;

  // Maps from an op to its resource IDs along with a bool indicating if the
  // resource is accessed `read-only`.
  llvm::SmallDenseMap<Operation*,
                      llvm::SmallVector<std::pair<ResourceId, bool>>>
      op_to_resource_ids_;
  llvm::SmallVector<std::pair<ResourceId, bool>> empty_resource_ids_;

  // For predecessor / successor queries on ops we don't track.
  llvm::SmallVector<Operation*, 4> empty_operation_set_;

  // Internal per-resource data structure for building the dependencies.
  struct PerResourceAccessInfo {
    // Last writes to resource before the current op is being analyzed. In
    // general there can be multiple most recent accesses when ops have
    // different parallel ids.
    absl::flat_hash_set<Operation*> last_writes;
    // Read ops since `last_write` before the current op is being analyzed.
    llvm::SmallVector<Operation*, 8> reads_since_last_write;
    // Whether a previous access of this resource already tracks the last
    // unknown read(s).
    bool are_last_unknown_reads_tracked = false;
    // Whether a previous write access of this resource already tracks the last
    // unknown write.
    bool is_last_unknown_write_tracked_by_write = false;
    // Whether a previous read or write access of this resource already tracks
    // the last unknown write.
    bool is_last_unknown_write_tracked = false;
  };

  // Resource access info per resource ID.
  llvm::SmallDenseMap<ResourceId, PerResourceAccessInfo, 8>
      per_resource_access_info_;

  // Hold the last set of reads and writes that
  // will be depended on by ops with greater nesting depths.
  // For example, the last read/write with parallel_ids `{group0:branch0}`
  // lives at stack depth 1 and is depended on by ops with parallel_ids
  // of the form `{group0:branch0, ...}`.
  //
  // We track a set of reads/writes rather than a single read/write because
  // multiple parallel ops may be live at any particular point.
  StackResourceToOps stack_down_;

  // Hold the last set of reads and writes that will be depended on by
  // ops with lesser nesting depths. For example, the last read/writes
  // with parallel_ids `{group0:branch0}` and `{group0:branch1}` live at
  // stack depth 1 and are depended on by ops with parallel_ids `{}`.
  StackResourceToOps stack_up_;

  // Parallel ids of the previously traversed op in the same function.
  // The transition from the previous parallel_ids to the current parallel_ids
  // determines which stack actions occur.
  ParallelIdsMap previous_parallel_ids_;

  const OpSideEffectCollector& op_side_effect_collector_;
  const TF::ResourceAliasAnalysis::Info& alias_analysis_;

  // Map op to parallel_ids. If an op is not a key then it has empty parallel
  // ids, which corresponds to nesting depth 0.
  const OpToParallelIdsMap& op_to_parallel_ids_;
};

}  // namespace detail

// An analysis that runs on a function and infers the control predecessors and
// successors for each op, based on side effects on known and unknown resources.
// Side-effecting ops on unknown resources are conservatively treated as
// interfering with all known resource op accesses. It distinguishes accesses
// based on whether they are read-only, and read-only ops do not interfere with
// each other.
//
// If there are nested regions, each region is handled separately, and control
// dependencies are only tracked for ops under the same parent op.
class SideEffectAnalysis : public detail::PerFunctionAggregateAnalysis<
                               detail::SideEffectAnalysisInfo> {
 public:
  // Constructs analysis by analyzing the given module operation. Because no
  // parallel_ids are given, the program has sequential memory semantics.
  explicit SideEffectAnalysis(ModuleOp module_op);

  // Constructs analysis by analyzing the given module operation where
  // `op_to_parallel_ids` supplies the group to branch map. This is the map
  // that is encoded by op attribute `_parallel_execution_ids`. This map is
  // used to code which ops should be executed in parallel and which
  // ops should be executed in sequence after ops have been flattened.
  // For example, children of
  // `tf_device.parallel_execute` will be executed in parallel and
  // each replica child of a `tf_device.replicate` will be executed in parallel.
  // Otherwise, by default, an op's children will be executed in sequence.
  //
  // Two ops with the same groups and different branches are considered
  // parallel so are not made dependent. For example if `OpA` has parallel_ids
  //   `{group0:branch0, group1:branch0}`
  // and `OpB` has parallel_ids
  //   `{group0:branch1, graph1:branch0}`
  // then `OpA` and `OpB` are executed in parallel because `group0` is common
  // with a different branch.
  //
  // Two ops with the same branches between common groups are executed in
  // sequence so are made dependent. For example, if `OpA` has parallel_ids
  //   `{group0:branch0, group1:branch0}`
  // and `OpB` has parallel_ids
  //   `{group0:branch0, group2:branch0}`
  // then `OpA` and `OpB` are executed in sequence because the common groups
  // have the same branch.
  //
  // If an op is not in `op_to_parallel_ids` then it is considered to have the
  // empty map from groups to branches.
  SideEffectAnalysis(ModuleOp module_op, OpToParallelIdsMap op_to_parallel_ids);

 private:
  ResourceAliasAnalysis alias_analysis_;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_SIDE_EFFECT_ANALYSIS_H_
