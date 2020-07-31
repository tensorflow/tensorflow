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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {

namespace detail {

// This template defines an aggregate analysis base class, which analyzes a
// module but the analysis info is stored per function.
template <typename InfoT>
class PerFunctionAggregateAnalysis {
 public:
  using Info = InfoT;

  // Returns the analysis info for the given function.
  const Info& GetAnalysisForFunc(FuncOp func) const {
    auto it = info_map_.find(func);
    assert(it != info_map_.end());
    return it->second;
  }

 protected:
  llvm::SmallDenseMap<FuncOp, InfoT, 8> info_map_;
};

class BacktrackAnalysis;

// Resource alias analysis information for a single function.
class ResourceAliasAnalysisInfo {
 public:
  // Constructs analysis info by analyzing the given function.
  ResourceAliasAnalysisInfo(FuncOp func,
                            const BacktrackAnalysis& backtrack_analysis);

  ResourceAliasAnalysisInfo(ResourceAliasAnalysisInfo&&) = default;

  // Returns if the analysis fails to resolve a resource-type value.
  bool IsUnknownResource(const Value resource) const;

  // Returns the set unique IDs which `resource` could alias. Requires that
  // IsUnknownResource(resource) == false.
  const llvm::SmallSet<int64_t, 8>& GetResourceUniqueIds(Value resource) const;

  // Returns the set of values that are potentially aliases of `value`. Requires
  // that IsUnknownResource(resource) == false.
  llvm::SmallSetVector<Value, 8> GetResourceAliases(Value resource) const;

 private:
  // Maps resource value to unique ID and vice-versa.
  void AddValueUniqueIDMapping(Value value, int64_t id) {
    resource_value_to_ids_[value].insert(id);
    id_to_resource_values_[id].insert(value);
  }

  // Returns the set unique Values which map to `id`.
  const llvm::SmallSetVector<Value, 8>& GetUniqueIdResources(int64_t id) const;

  // Maps each resource-type value to a set of unique IDs that it could alias.
  llvm::SmallDenseMap<Value, llvm::SmallSet<int64_t, 8>, 8>
      resource_value_to_ids_;

  // Maps each unique ID to a set of resource-type values that could alias to
  // it. This is inverse of `resource_value_to_ids_` map.
  llvm::SmallDenseMap<int64_t, llvm::SmallSetVector<Value, 8>, 8>
      id_to_resource_values_;
};

}  // namespace detail

// An analysis that runs on a module and maps each resource-type value to a
// set of unique IDs representing the possible resources it could alias.
//
// Note that this is not an inter-procedural or inter-regional analysis, i.e.,
// each function and region are handled separately and cross-function or cross-
// region aliasing cannot be checked by this analysis.
class ResourceAliasAnalysis : public detail::PerFunctionAggregateAnalysis<
                                  detail::ResourceAliasAnalysisInfo> {
 public:
  // Constructs analysis by analyzing the given module operation.
  explicit ResourceAliasAnalysis(Operation* op);
};

namespace detail {
// Side effect analysis info for a single function.
class SideEffectAnalysisInfo {
 public:
  SideEffectAnalysisInfo() = default;

  // Constructs analysis info by analyzing the given function.
  SideEffectAnalysisInfo(
      FuncOp func_op, const TF::ResourceAliasAnalysis::Info& alias_analysis) {
    AnalyzeFunction(func_op, alias_analysis);
  }

  // Constructs analysis info by analyzing the given region.
  SideEffectAnalysisInfo(
      Region* region, const TF::ResourceAliasAnalysis::Info& alias_analysis) {
    AnalyzeRegion(region, alias_analysis);
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

 private:
  // Runs the analysis on `func_op` and populates sorted_control_predecessors_
  // and sorted_control_successors_.
  void AnalyzeFunction(FuncOp func_op,
                       const TF::ResourceAliasAnalysis::Info& alias_analysis);

  // Runs the analysis on `region` and populates control_predecessors_.
  void AnalyzeRegion(Region* region,
                     const TF::ResourceAliasAnalysis::Info& alias_analysis);

  // Updates control_predecessors_ for `op` that is being visited, on the given
  // `resource_id`.
  void AddPredecessorsForAccess(int64_t resource_id, Operation* op,
                                bool read_only);

  // Adds op's access to per_resource_access_info_.
  void TrackAccess(int64_t resource_id, Operation* op, bool read_only);

  // Maps from an op to its control predecessors.
  llvm::SmallDenseMap<Operation*, llvm::SmallPtrSet<Operation*, 4>, 8>
      control_predecessors_;
  // Maps from an op to its control predecessors sorted in program order.
  llvm::SmallDenseMap<Operation*, llvm::SmallVector<Operation*, 4>, 8>
      sorted_control_predecessors_;
  // Maps from an op to its control successors sorted in program order.
  llvm::SmallDenseMap<Operation*, llvm::SmallVector<Operation*, 4>, 8>
      sorted_control_successors_;

  // Internal per-resource data structure when we build the dependencies.
  struct PerResourceAccessInfo {
    // Last op that writes the resource before the current op being analyzed.
    Operation* last_write = nullptr;
    // Read ops since last_write before the current op being analyzed.
    llvm::SmallVector<Operation*, 8> reads_since_last_write;
    // Whether previous accesses of this resource already tracked last unknown
    // read for the current access being analyzed.
    bool tracked_last_unknown_read = false;
    // Whether previous accesses of this resource already tracked last unknown
    // write for a the current read being analyzed.
    bool tracked_last_unknown_write_for_read = false;
    // Whether previous accesses of this resource already tracked last unknown
    // write for a the current write being analyzed.
    bool tracked_last_unknown_write_for_write = false;
  };

  llvm::SmallDenseMap<int64_t, PerResourceAccessInfo, 8>
      per_resource_access_info_;
};

}  // namespace detail

// An analysis that runs on a function and infers the control predecessors and
// successors for each op, based on side-effects on known and unknown resources.
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
  explicit SideEffectAnalysis(Operation* op);
};

// Base CRTP class to help write passes that are consumes a per-function
// aggregate analysis and operate on all non-extern functions (similar to a
// FunctionPass, but with no concurrency between functions). The derived classes
// need to provide a runOnFunction() method that accepts the function and the
// analysis information for that function.
template <typename DerivedT, typename AnalysisT>
class PerFunctionAggregateAnalysisConsumerPass
    : public PassWrapper<
          PerFunctionAggregateAnalysisConsumerPass<DerivedT, AnalysisT>,
          OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp op = this->getOperation();
    DerivedT& derived = *static_cast<DerivedT*>(this);
    auto& analysis = this->template getAnalysis<AnalysisT>();

    for (auto func : op.getOps<FuncOp>())
      if (!func.isExternal())
        derived.runOnFunction(func, analysis.GetAnalysisForFunc(func));
  }
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_SIDE_EFFECT_ANALYSIS_H_
