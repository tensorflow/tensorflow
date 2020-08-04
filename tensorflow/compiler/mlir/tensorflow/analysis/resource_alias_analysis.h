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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/per_function_aggregate_analysis.h"

namespace mlir {
namespace TF {
namespace detail {
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

 public:
  static constexpr int64_t kUnknownResourceId = -1;
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

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_
