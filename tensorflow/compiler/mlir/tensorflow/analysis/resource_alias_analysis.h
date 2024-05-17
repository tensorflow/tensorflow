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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/per_function_aggregate_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace detail {
class BacktrackAnalysis;
class BacktrackAnalysisInfo;

// Resource alias analysis information for a single function.
class ResourceAliasAnalysisInfo {
 public:
  // Constructs analysis info by analyzing the given function.
  ResourceAliasAnalysisInfo(func::FuncOp func,
                            const BacktrackAnalysis& backtrack_analysis,
                            SymbolTableCollection& symbol_table_collection);

  ResourceAliasAnalysisInfo(ResourceAliasAnalysisInfo&&) = default;

  // Returns if the analysis fails to resolve a resource-type value.
  bool IsUnknownResource(Value resource) const;

  // Returns the set of unique IDs which `resource` could alias. Requires that
  // IsUnknownResource(resource) == false.
  const llvm::SmallSet<int64_t, 8>& GetResourceUniqueIds(Value resource) const;

  // Returns the set of values that are potentially aliases of `value`. Requires
  // `IsUnknownResource(resource) == false`.
  llvm::SmallSetVector<Value, 8> GetResourceAliases(Value resource) const;

  llvm::SmallSetVector<Value, 8> GetValuesForResourceId(int64_t id) const {
    auto it = id_to_resource_values_.find(id);
    if (it == id_to_resource_values_.end()) {
      return {};  // return empty set
    }
    return it->getSecond();
  }

  // Returns true iff given resource is allocated by op with
  // `UniqueResourceAllocation` trait. This can be utilized for while-loop
  // parallelization.
  bool IsUniqueResourceAllocationId(int64_t resource_id) const {
    return unique_resource_allocation_ids_.contains(resource_id);
  }

 private:
  // Maps resource value to unique ID and vice-versa. Returns true if the
  // mapping has changed.
  bool AddValueUniqueIDMapping(Value value, int64_t id) {
    resource_value_to_ids_[value].insert(id);
    return id_to_resource_values_[id].insert(value);
  }

  // Returns the set unique Values which map to `id`.
  const llvm::SmallSetVector<Value, 8>& GetUniqueIdResources(int64_t id) const;

  // Propagates the resource IDs from an input operand to a result. Returns
  // true of the mapping has changed.
  bool PropagateInputToOutput(const Value& operand, const OpResult& result);

  // Analyzes while loops to compute resource IDs for the loop results.
  // `body_info` is the backtrack analysis info for the loop body.
  void AnalyzeWhileLoop(Operation* while_op,
                        const BacktrackAnalysisInfo& body_info);

  // Analyzes tf.Case/tf.If ops to compute resource IDs.
  template <class CaseOrIfOp>
  void AnalyzeFunctionalCaseOrIfOp(CaseOrIfOp case_or_if_op,
                                   llvm::ArrayRef<func::FuncOp> functions,
                                   const BacktrackAnalysis& backtrack_analysis);

  // Analyzes tf.CaseRegion/tf.IfRegion ops to compute resource IDs.
  void AnalyzeRegionCaseOrIfOp(Operation* case_or_if_op,
                               const BacktrackAnalysis& backtrack_analysis);

  // Maps each resource-type value to a set of unique IDs that it could alias.
  llvm::SmallDenseMap<Value, llvm::SmallSet<int64_t, 8>, 8>
      resource_value_to_ids_;

  // Maps each unique ID to a set of resource-type values that could alias to
  // it. This is inverse of `resource_value_to_ids_` map.
  llvm::SmallDenseMap<int64_t, llvm::SmallSetVector<Value, 8>, 8>
      id_to_resource_values_;

  // Maps MLIR type IDs for resource types to internal resource type IDs.
  llvm::SmallDenseMap<TypeID, int64_t> type_id_to_internal_type_id_;

  // Contains IDs of all resources that are allocated by ops with
  // `UniqueResourceAllocation` trait.
  llvm::SmallDenseSet<int64_t, 32> unique_resource_allocation_ids_;

 public:
  // Resource IDs have the following semantics:
  // a) -1 represents an unknown resource (both instance and type unknown)
  // b) IDs in range [0,kMaxResourceTypeId] represent resource type IDs; we use
  //    such IDs when we know the resource type but not the instance
  // c) IDs > kMaxResourceTypeId represent resource instance IDs (i.e., we know
  //    the specific resource instance)
  //
  // Note: In general, there can be different ops allocating a resource of the
  // same type, for one we might assign a resource type ID and for the other
  // a resource instance ID. That means, they will be treated as non-aliasing.
  // This is correct for all current cases. A problematic case could be if we
  // had two ops A and B, A has the `ResourceHandleAllocatorInterface` and B has
  // not, and both ops might return a handle to the same resource (depending on
  // attributes). In this case, the return value of A would get a different ID
  // than the return value of B although both could point to the same resource.
  // It seems highly unlikely to encounter such a case but, to be safe, this
  // should be revisited for new resource-allocators that might potentially
  // break our currently guaranteed correctness.
  // For context, we are very conservative here compared to
  // `auto_control_deps.py` where it is assumed that allocated resource values
  // NEVER alias. We should align our assumptions in the future.
  static constexpr int64_t kUnknownResourceId = -1;
  static constexpr int64_t kInvalidResourceId = -2;
  static constexpr int64_t kMaxResourceTypeId = 9999;
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
  explicit ResourceAliasAnalysis(ModuleOp module);
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_
