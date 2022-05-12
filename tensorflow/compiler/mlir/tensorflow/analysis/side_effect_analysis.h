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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"

namespace mlir {
namespace TF {
using ResourceId = int64_t;

namespace detail {

class OpSideEffectCollector;

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
                         const TF::ResourceAliasAnalysis::Info& alias_analysis)
      : op_side_effect_collector_(op_side_effect_collector),
        alias_analysis_(alias_analysis) {
    AnalyzeFunction(func_op);
  }

  // Constructs analysis info by analyzing the given region.
  SideEffectAnalysisInfo(Region* region,
                         const OpSideEffectCollector& op_side_effect_collector,
                         const TF::ResourceAliasAnalysis::Info& alias_analysis)
      : op_side_effect_collector_(op_side_effect_collector),
        alias_analysis_(alias_analysis) {
    AnalyzeRegion(region);
  }

  SideEffectAnalysisInfo(SideEffectAnalysisInfo&&) = default;

  // Returns a vector of ops that are direct control predecessors of `op`,
  // sorted in program order. If `filter` is provided, only predecessors that
  // pass the filter (returning true) will be included.
  llvm::SmallVector<Operation*, 4> DirectControlPredecessors(
      Operation* op,
      llvm::function_ref<bool(Operation*)> filter = nullptr) const;

  // Returns a vector of ops that are direct control successors of `op`,
  // sorted in program order. If `filter` is provided, only successors that
  // pass the filter (returning true) will be included.
  llvm::SmallVector<Operation*, 4> DirectControlSuccessors(
      Operation* op,
      llvm::function_ref<bool(Operation*)> filter = nullptr) const;

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

  // Returns a set of resource IDs that are conflicting with `resource_id`, i.e.
  // there are potentially dependencies between the corresponding resources.
  llvm::SmallSet<ResourceId, 8> GetConflictingIds(ResourceId resource_id) const;

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

  // Internal per-resource data structure for building the dependencies.
  struct PerResourceAccessInfo {
    // Last op that writes to resource before the current op is being analyzed.
    Operation* last_write = nullptr;
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

  const OpSideEffectCollector& op_side_effect_collector_;
  const TF::ResourceAliasAnalysis::Info& alias_analysis_;
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
  // Constructs analysis by analyzing the given module operation.
  explicit SideEffectAnalysis(ModuleOp module);

 private:
  ResourceAliasAnalysis alias_analysis_;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_SIDE_EFFECT_ANALYSIS_H_
