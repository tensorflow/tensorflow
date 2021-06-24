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

#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"

#include <cstdint>
#include <initializer_list>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {

constexpr auto kUnknownResourceId =
    ResourceAliasAnalysis::Info::kUnknownResourceId;

//===----------------------------------------------------------------------===//
// SideEffectAnalysisInfo helper functions.
//===----------------------------------------------------------------------===//

// Returns a set that contains only kUnknownResourceId.
llvm::SmallDenseSet<int64_t, 8> UnknownResourceSet() {
  llvm::SmallDenseSet<int64_t, 8> unknown_set;
  unknown_set.insert(kUnknownResourceId);
  return unknown_set;
}

// Returns all resources that could be accessed by op, or UnknownResourceSet()
// if we cannot find all of them.
llvm::SmallDenseSet<int64_t, 8> FindAccessedResources(
    Operation* op, const ResourceAliasAnalysis::Info& alias_analysis) {
  VLOG(1) << "Find accessed resources for: " << debugString(*op);
  llvm::SmallDenseSet<int64_t, 8> resources;

  for (auto operand : filter_resources(op->getOperands())) {
    if (alias_analysis.IsUnknownResource(operand)) {
      VLOG(1) << "\tunknown resource in operand";
      return UnknownResourceSet();
    }
    const auto& ids = alias_analysis.GetResourceUniqueIds(operand);
    resources.insert(ids.begin(), ids.end());
  }
  for (auto result : filter_resources(op->getResults())) {
    if (alias_analysis.IsUnknownResource(result)) {
      VLOG(1) << "\tunknown resource in result";
      return UnknownResourceSet();
    }
    const auto& ids = alias_analysis.GetResourceUniqueIds(result);
    resources.insert(ids.begin(), ids.end());
  }
  return resources;
}

// Helper struct defining what memory effects are present for a resource.
struct SideEffects {
  bool alloc = false;
  bool free = false;
  bool read = false;
  bool write = false;

  bool IsAllocOnly() const { return alloc && !free && !read && !write; }
  bool IsReadOnly() const { return !alloc && !free && read && !write; }
};

using SideEffectsByValue = llvm::SmallDenseMap<Value, SideEffects>;

bool MustExecute(const MemoryEffects::EffectInstance& effect) {
  VLOG(1) << "MustExecute check with: "
          << std::string(effect.getResource()->getName());
  if (llvm::isa<ResourceEffects::TPUEmbedding>(effect.getResource())) {
    assert(!effect.getValue() && !effect.getParameters() &&
           isa<MemoryEffects::Write>(effect.getEffect()));
    return true;
  }
  return false;
}

// Collects memory side effects for an operation by value (operands and
// results).
void GetSideEffectsByValue(Operation* op,
                           SideEffectsByValue& side_effects_by_value,
                           bool& must_execute) {
  VLOG(1) << "Querying for " << mlir::debugString(*op);
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) return;

  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  interface.getEffects(effects);

  for (auto& effect : effects) {
    if (MustExecute(effect)) {
      VLOG(1) << "\tmust execute";
      must_execute = true;
      continue;
    }

    // TODO(lyandy): Support effects with no value defined.
    if (!effect.getValue()) {
      VLOG(1) << "\teffect with no value, skipping";
      side_effects_by_value.clear();
      must_execute = false;
      return;
    }
    auto it = side_effects_by_value.try_emplace(effect.getValue());
    auto& side_effect = it.first->getSecond();
    auto* resource_effect = effect.getEffect();
    if (isa<MemoryEffects::Allocate>(resource_effect)) {
      VLOG(1) << "\tallocate effect";
      side_effect.alloc = true;
    } else if (isa<MemoryEffects::Free>(resource_effect)) {
      VLOG(1) << "\tfree effect";
      side_effect.free = true;
    } else if (isa<MemoryEffects::Read>(resource_effect)) {
      VLOG(1) << "\tread effect";
      side_effect.read = true;
    } else if (isa<MemoryEffects::Write>(resource_effect)) {
      VLOG(1) << "\twrite effect";
      side_effect.write = true;
    } else {
      VLOG(1) << "\tunknown effect, skipping";
      side_effects_by_value.clear();
      must_execute = false;
      return;
    }
  }
}

// Checks if a value is a result of `op`.
bool IsOperationResult(Operation* op, Value value) {
  return value.getDefiningOp() == op;
}

// Checks if an operation's resource operands are read only. Operation results
// are ignored.
bool IsResourceOpReadOnly(Operation* op,
                          const SideEffectsByValue& side_effects_by_value) {
  if (side_effects_by_value.empty()) return false;

  for (const auto& value_side_effect : side_effects_by_value) {
    Value value = value_side_effect.getFirst();
    if (IsOperationResult(op, value)) continue;
    const SideEffects& side_effects = value_side_effect.getSecond();
    if (!side_effects.IsReadOnly()) return false;
  }

  return true;
}

// Checks if an operation's resource results are alloc only and no side effects
// are present for its operands.
bool IsResourceOpAllocOnly(Operation* op,
                           const SideEffectsByValue& side_effects_by_value) {
  if (side_effects_by_value.empty()) return false;

  for (const auto& value_side_effect : side_effects_by_value) {
    // Operand with side effect.
    Value value = value_side_effect.getFirst();
    if (!IsOperationResult(op, value)) return false;
    const SideEffects& side_effects = value_side_effect.getSecond();
    if (!side_effects.IsAllocOnly()) return false;
  }

  return true;
}

// Returns if `op` is a resource declaration.
bool OpIsDeclaration(Operation* op,
                     const ResourceAliasAnalysis::Info& alias_analysis) {
  return llvm::isa<TF::IdentityNOp, TF::IdentityOp>(op) &&
         !FindAccessedResources(op, alias_analysis).empty();
}

// A vector of resource variable id's with their associated resource value.
using ResourceIdsByValue =
    llvm::SmallVector<std::pair<Value, const llvm::SmallSet<int64_t, 8>*>, 4>;

// Collects resource id's by resource value. If operation resource side effects
// are unknown or a resource is unknown, an empty optional is returned.
llvm::Optional<ResourceIdsByValue> GetResourceIdsByValue(
    Operation* op, const ResourceAliasAnalysis::Info& alias_analysis,
    const SideEffectsByValue& side_effects_by_value) {
  ResourceIdsByValue resource_ids_by_value;
  if (side_effects_by_value.empty()) return llvm::None;

  // Returns true iff all side-effect-related values are known to
  // `alias_analysis`.
  auto collect_ids = [&](ValueRange values) {
    for (auto value : values) {
      // Value is not related to any side-effect, skip.
      if (side_effects_by_value.count(value) == 0) continue;
      // Value is not a resource variable, thus not known to `alias_analysis`.
      if (!getElementTypeOrSelf(value.getType()).isa<TF::ResourceType>())
        return false;
      // Value is a resource variable not known to `alias_analysis`.
      if (alias_analysis.IsUnknownResource(value)) return false;
      // Value is a resource variable known to `alias_analysis`.
      const auto& ids = alias_analysis.GetResourceUniqueIds(value);
      resource_ids_by_value.push_back({value, &ids});
    }
    return true;
  };

  if (collect_ids(op->getOperands()) && collect_ids(op->getResults()))
    // No unknown side-effect-related values.
    return resource_ids_by_value;
  else
    return llvm::None;
}

// Returns true if `op` is known to not have any side effect.
bool OpIsKnownToHaveNoSideEffect(Operation* op) {
  // For op's in the Tensorflow dialect, query the dialect.
  if (isa_and_nonnull<TF::TensorFlowDialect>(op->getDialect()))
    return !TensorFlowDialect::CanHaveSideEffects(op);

  // Otherwise, conservatively assume that there can be side effects.
  return false;
}

}  // namespace

namespace detail {
//===----------------------------------------------------------------------===//
// SideEffectAnalysisInfo
//===----------------------------------------------------------------------===//

void SideEffectAnalysisInfo::TrackAccess(int64_t resource_id, Operation* op,
                                         bool read_only) {
  VLOG(1) << "TrackAccess for " << debugString(*op);
  if (resource_id == kUnknownResourceId) {
    VLOG(1) << "\tunknown resource id";
    if (read_only) {
      // New unknown read is not tracked by any known resource access.
      for (auto& entry : per_resource_access_info_) {
        entry.getSecond().tracked_last_unknown_read = false;
      }
    } else {
      // Unknown write can clear all other tracked information, since it acts
      // like a barrier.
      VLOG(1) << "\tclearing per resource access info";
      per_resource_access_info_.clear();
    }
  }
  VLOG(1) << "\tinfo for " << resource_id;
  auto& info = per_resource_access_info_[resource_id];
  if (read_only) {
    info.reads_since_last_write.push_back(op);
    // Resource read must have carried control dependencies of unknown write. It
    // can only avoid adding control edges (from uknown accesses) for a later
    // write, but not for a later read, because this read can be reordered with
    // a later read.
    info.tracked_last_unknown_write_for_write = true;
  } else {
    // Resource write must have carried control dependencies of unknown access.
    info.tracked_last_unknown_write_for_read = true;
    info.tracked_last_unknown_write_for_write = true;
    info.tracked_last_unknown_read = true;
    info.last_write = op;
    info.reads_since_last_write.clear();
  }
}

void SideEffectAnalysisInfo::AddPredecessorsForAccess(int64_t resource_id,
                                                      Operation* op,
                                                      bool read_only) {
  VLOG(1) << "Adding predecessors for resource " << resource_id << " and op "
          << debugString(*op);
  auto it = per_resource_access_info_.find(resource_id);
  if (it == per_resource_access_info_.end()) return;
  const auto& access_info = it->getSecond();
  auto& control_predecessors = control_predecessors_[op];
  bool read_tracked = false;
  if (!read_only) {
    control_predecessors.insert(access_info.reads_since_last_write.begin(),
                                access_info.reads_since_last_write.end());
    read_tracked = !access_info.reads_since_last_write.empty();
  }
  if (access_info.last_write && !read_tracked) {
    control_predecessors.insert(access_info.last_write);
  }
}

void SideEffectAnalysisInfo::AnalyzeFunction(
    FuncOp func_op, const TF::ResourceAliasAnalysis::Info& alias_analysis) {
  // AnalyzeRegion() recursively analyzes the function body, and only populates
  // control_predecessors_.
  AnalyzeRegion(&func_op.getBody(), alias_analysis);
  // Populate sorted_control_predecessors_ and sorted_control_successors_ based
  // on control_predecessors.
  for (auto& entry : control_predecessors_) {
    auto op = entry.getFirst();
    auto& sorted_predecessors = sorted_control_predecessors_[op];
    for (auto predecessor : entry.getSecond()) {
      sorted_predecessors.push_back(predecessor);
      sorted_control_successors_[predecessor].push_back(op);
    }
  }
  control_predecessors_.clear();
  for (auto& entry : sorted_control_predecessors_) {
    llvm::sort(entry.getSecond(), [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });
  }
  for (auto& entry : sorted_control_successors_) {
    llvm::sort(entry.getSecond(), [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });
  }
}

void SideEffectAnalysisInfo::AnalyzeRegion(
    Region* region, const TF::ResourceAliasAnalysis::Info& alias_analysis) {
  // This function populates control_predecessors_ by walking through the
  // region, and tracking resource accesses in per_resource_access_info_.

  // Returns whether an access to `resource` can skip control edges from
  // previous accesses to unknown resources, due to that earlier accesses to
  // `resource` already indirectly tracked previous accesses to unknown
  // resources. `read_only` specifies the type of access of the current op being
  // considered.
  auto unknown_access_indirectly_tracked_by_resource = [&](int64_t resource,
                                                           bool read_only) {
    VLOG(1) << "\tunknown access indirectly tracked by resource " << resource;
    auto it = per_resource_access_info_.find(resource);
    if (it == per_resource_access_info_.end()) {
      VLOG(1) << "\t\tnot found";
      return false;
    }
    auto unknown_it = per_resource_access_info_.find(kUnknownResourceId);
    const bool no_unknown_read =
        unknown_it == per_resource_access_info_.end() ||
        unknown_it->getSecond().reads_since_last_write.empty();
    bool ret = read_only ? it->second.tracked_last_unknown_write_for_read
                         : it->second.tracked_last_unknown_write_for_write &&
                               (it->second.tracked_last_unknown_read ||
                                no_unknown_read);
    VLOG(1) << "\t\tunknown access inderictly tracked by resource: " << ret;
    return ret;
  };

  // We explicitly iterates through the regions and blocks, in order to handle
  // different nested regions separately.
  for (auto& block : *region) {
    llvm::SmallPtrSet<Operation*, 8> non_resource_control_predecessors;
    for (auto& op : block) {
      for (Region& child : op.getRegions()) {
        SideEffectAnalysisInfo child_analysis(&child, alias_analysis);
        // Moves the control_predecessors_ fields in child region to current
        // region
        for (auto& entry : child_analysis.control_predecessors_)
          control_predecessors_[entry.first] = std::move(entry.second);
      }

      // We do not need explicit control edges for declaration ops.
      if (OpIsDeclaration(&op, alias_analysis)) continue;

      SideEffectsByValue side_effects_by_value;
      bool must_execute = false;
      GetSideEffectsByValue(&op, side_effects_by_value, must_execute);

      if (side_effects_by_value.empty() && OpIsKnownToHaveNoSideEffect(&op))
        continue;

      // TODO(jpienaar): This only currently uses unknown when not per value
      // resource is used.
      if (side_effects_by_value.empty() && must_execute) {
        VLOG(1) << "No resources & must execute: " << debugString(op);
        // Add unknown resource ops as predecessors of the op that must execute,
        // to guarantee ordering between unknown resource ops.
        AddPredecessorsForAccess(kUnknownResourceId, &op, /*read_only=*/false);
        non_resource_control_predecessors.insert(&op);
        continue;
      }

      if (IsResourceOpAllocOnly(&op, side_effects_by_value)) {
        VLOG(1) << "Resource alloc only: " << debugString(op);
        continue;
      }

      auto resource_ids_by_value =
          GetResourceIdsByValue(&op, alias_analysis, side_effects_by_value);
      const bool read_only = IsResourceOpReadOnly(&op, side_effects_by_value);
      bool indirectly_tracked_unknown_access = false;
      // First add edges from known resources.
      if (!resource_ids_by_value.hasValue()) {
        VLOG(1) << "Resource not by value: " << debugString(op);
        for (auto& entry : per_resource_access_info_) {
          if (entry.getFirst() == kUnknownResourceId) {
            VLOG(1) << "\tskipping over unknown resource id";
            continue;
          }
          AddPredecessorsForAccess(entry.getFirst(), &op, read_only);
          indirectly_tracked_unknown_access |=
              unknown_access_indirectly_tracked_by_resource(entry.getFirst(),
                                                            read_only);
        }
      } else {
        // Collect all resource id's and whether their side effect is read only.
        llvm::SmallDenseMap<int64_t, bool> read_only_by_resource_id;
        for (const auto& resource_ids : *resource_ids_by_value) {
          const bool is_result = resource_ids.first.getDefiningOp() == &op;
          auto value_side_effect =
              side_effects_by_value.find(resource_ids.first);
          bool resource_read_only = false;
          if (value_side_effect != side_effects_by_value.end()) {
            if (is_result && value_side_effect->getSecond().IsAllocOnly())
              continue;
            resource_read_only = value_side_effect->getSecond().IsReadOnly();
          }

          for (const auto& id : *resource_ids.second) {
            auto it =
                read_only_by_resource_id.try_emplace(id, resource_read_only);
            if (!it.second && !resource_read_only)
              it.first->getSecond() = resource_read_only;
          }
        }

        for (const auto& resource : read_only_by_resource_id) {
          const auto& resource_id = resource.getFirst();
          const auto& resource_read_only = resource.getSecond();
          AddPredecessorsForAccess(resource_id, &op, resource_read_only);
          indirectly_tracked_unknown_access |=
              unknown_access_indirectly_tracked_by_resource(resource_id,
                                                            resource_read_only);
          // Update access info for known resources.
          TrackAccess(resource_id, &op, resource_read_only);
        }
      }

      // If not indirectly tracked, add edges from the resource.
      if (!indirectly_tracked_unknown_access) {
        VLOG(1) << "Not indirectly tracked with unknown access: "
                << debugString(op);
        if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
          llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
          interface.getEffects(effects);
        }
        AddPredecessorsForAccess(kUnknownResourceId, &op, read_only);
      }
      if (!resource_ids_by_value.hasValue()) {
        VLOG(1) << "Indirectly tracked with no value: " << debugString(op);

        // Update access info for unknown resource.
        TrackAccess(kUnknownResourceId, &op, read_only);
        // Add ops that must execute to unknown resource op predecessors.
        auto& control_predecessors = control_predecessors_[&op];
        control_predecessors.insert(non_resource_control_predecessors.begin(),
                                    non_resource_control_predecessors.end());
        // Ops that must execute currently tracked are cleared as transitively
        // unknown resource ops will allow for such ops to be transitively
        // reachable.
        non_resource_control_predecessors.clear();
      }
    }
  }
}

llvm::SmallVector<Operation*, 4>
SideEffectAnalysisInfo::DirectControlPredecessors(
    Operation* op, llvm::function_ref<bool(Operation*)> filter) const {
  llvm::SmallVector<Operation*, 4> result;
  auto it = sorted_control_predecessors_.find(op);
  if (it == sorted_control_predecessors_.end()) return result;
  result.reserve(it->getSecond().size());
  for (auto predecessor : it->getSecond()) {
    if (!filter || filter(predecessor)) result.push_back(predecessor);
  }
  return result;
}

llvm::SmallVector<Operation*, 4>
SideEffectAnalysisInfo::DirectControlSuccessors(
    Operation* op, llvm::function_ref<bool(Operation*)> filter) const {
  llvm::SmallVector<Operation*, 4> result;
  auto it = sorted_control_successors_.find(op);
  if (it == sorted_control_successors_.end()) return result;
  result.reserve(it->getSecond().size());
  for (auto successor : it->getSecond()) {
    if (!filter || filter(successor)) result.push_back(successor);
  }
  return result;
}
}  // namespace detail

SideEffectAnalysis::SideEffectAnalysis(ModuleOp module) {
  // Analyze entire module for alias analysis info.
  ResourceAliasAnalysis alias_analysis(module);

  // Analyze all functions.
  for (auto func : module.getOps<FuncOp>())
    this->info_map_.try_emplace(func, func,
                                alias_analysis.GetAnalysisForFunc(func));
}

}  // namespace TF
}  // namespace mlir
