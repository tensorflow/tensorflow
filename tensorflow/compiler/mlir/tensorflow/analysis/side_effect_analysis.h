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
  explicit SideEffectAnalysis(ModuleOp module);
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_SIDE_EFFECT_ANALYSIS_H_
