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

#include <bitset>
#include <string>

#include "absl/container/node_hash_map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {

constexpr ResourceId kUnknownResourceId =
    ResourceAliasAnalysis::Info::kUnknownResourceId;
static_assert(kUnknownResourceId < 0, "kUnknownResourceId must be < 0");

// A collection of Resource IDs. Note that `kUnknownResourceId` is smaller than
// all other resource IDs which are nonnegative (see check above) so it will
// always be the first element of a `ResourceIdSet` (we make use of this).
using ResourceIdSet = llvm::SmallSet<ResourceId, 8>;

// Note that we cannot simply define a `static const llvm::SmallSet` here
// because of missing `initializer_list` support for `llvm::SmallSet`.
const ResourceIdSet& UnknownResourceSet() {
  // clang-format off
  static auto* id_set = new ResourceIdSet();
  id_set->insert(kUnknownResourceId);
  return *id_set;
}

// Helper function to avoid frequent checks for unknown IDs.
const ResourceIdSet& GetResourceUniqueIdsOrUnknown(
    Value value,
    const ResourceAliasAnalysis::Info& alias_analysis) {
  if (!getElementTypeOrSelf(value.getType()).isa<TF::ResourceType>() ||
      alias_analysis.IsUnknownResource(value)) return UnknownResourceSet();
  return alias_analysis.GetResourceUniqueIds(value);
}

// Helper class for a collection of side effects for one resource.
class SideEffects {
  enum Type {
    kAlloc = 0,
    kFree = 1,
    kRead = 2,
    kWrite = 3
  };

 public:
  bool IsAlloc() const { return effects_.test(kAlloc); }
  bool IsFree() const { return effects_.test(kFree); }
  bool IsRead() const { return effects_.test(kRead); }
  bool IsWrite() const { return effects_.test(kWrite); }
  bool IsAllocOnly() const { return IsAlloc() && effects_.count() == 1; }
  bool IsReadOnly() const { return IsRead() && effects_.count() == 1; }
  ResourceId GetResourceId() const { return resource_id_; }

  void SetAlloc() { effects_.set(kAlloc); }
  void SetFree() { effects_.set(kFree); }
  void SetRead() { effects_.set(kRead); }
  void SetWrite() { effects_.set(kWrite); }
  void SetUnknownEffect() { effects_.set(); }
  void SetResourceId(ResourceId resource_id) { resource_id_ = resource_id; }
  void AddEffects(const SideEffects& other_effects) {
    effects_ |= other_effects.effects_;
  }

 private:
  std::bitset<4> effects_ = 0;
  ResourceId resource_id_ = kUnknownResourceId;
};

// We use `std::map` here because we rely on the order of elements.
using SideEffectsByResourceId = std::map<ResourceId, SideEffects>;

// We use `std::unordered_map` here for pointer stability reasons.
// Note: If memory usage ever becomes a bottleneck here (not expected) we could
// use a Trie-like data structure to avoid storing side effects in both parent
// op and all its child ops (recursively), at the expense of lookup time.
using OpSideEffectMap = std::unordered_map<Operation*, SideEffectsByResourceId>;

// Update `side_effects_by_resource_id` with `side_effects`.
void UpdateSideEffectsByResourceId(
    const SideEffects& side_effects,
    SideEffectsByResourceId& side_effects_by_resource_id) {
  ResourceId id = side_effects.GetResourceId();
  auto iter = side_effects_by_resource_id.find(id);
  if (iter == side_effects_by_resource_id.end()) {
    side_effects_by_resource_id[id] = side_effects;
  } else {
    iter->second.AddEffects(side_effects);
  }
}

bool MayHaveSideEffect(Operation* op) {
  if (isa_and_nonnull<TF::TensorFlowDialect>(op->getDialect()))
    return TensorFlowDialect::CanHaveSideEffects(op);

  if (mlir::MemoryEffectOpInterface::hasNoEffect(op)) return false;
  // Conservatively assume that there can be side effects.
  return true;
}

bool ShouldUseResourceAliasAnalysis(
    const MemoryEffects::EffectInstance& effect) {
  Value value = effect.getValue();
  if (value && getElementTypeOrSelf(value.getType()).isa<ResourceType>()) {
    // For value-based effects on resource values we can use resource alias
    // analysis.
    return true;
  }
  // For all other effects don't rely on resource alias analysis. Note that
  // non-resource values are not processed in resource alias analysis.
  return false;
}

//===----------------------------------------------------------------------===//
// SideEffectAnalysisInfo helper functions.
//===----------------------------------------------------------------------===//

SideEffects GetSideEffectsFromEffectInstance(
    const MemoryEffects::EffectInstance& effect_instance, Operation* op) {
  mlir::SideEffects::Effect* effect = effect_instance.getEffect();
  SideEffects side_effects;
  if (isa<MemoryEffects::Allocate>(effect)) {
    side_effects.SetAlloc();
  } else if (isa<MemoryEffects::Free>(effect)) {
    side_effects.SetFree();
  } else if (isa<MemoryEffects::Read>(effect)) {
    side_effects.SetRead();
  } else if (isa<MemoryEffects::Write>(effect)) {
    side_effects.SetWrite();
  } else {
    LOG(WARNING) << "Unsupported effect for op "
                 << op->getName().getStringRef().str();
    side_effects.SetUnknownEffect();
  }
  return side_effects;
}

}  // namespace

namespace detail {

// Class for propagating op-based side effects bottom-up and collecting them
// per op, by resource ID.
class OpSideEffectCollector {
 public:
  // Recursively collects op-based side effects for all ops in module and
  // populates `op_side_effect_map_`.
  explicit OpSideEffectCollector(ModuleOp module) {
    symbol_table_collection_.getSymbolTable(module);
    for (auto func : module.getOps<func::FuncOp>()) {
      CollectOpSideEffects(func);
    }
  }

  // Returns op-based side effects by resource ID for `op`.
  const SideEffectsByResourceId& GetSideEffectsForOp(Operation* op) const {
    auto iter = op_side_effect_map_.find(op);
    if (iter != op_side_effect_map_.end()) return iter->second;
    return empty_side_effects_map_;
  }

  // Returns true iff resource with given ID is only self-dependent, i.e., there
  // are no dependencies to other resources (including unknown resources).
  bool IsOnlySelfDependent(ResourceId resource_id) const {
    return self_dependent_only_ids_.contains(resource_id);
  }

 private:
  // Adds op-based side effects from all ops in `region` to `op` side effects.
  // Collects side effects for ops that weren't visited before.
  void AddRegionSideEffectsForOp(Region& region, Operation* op) {
    for (Block& block : region) {
      for (Operation& curr_op : block) {
        if (op_side_effect_map_.count(&curr_op) == 0) {
          CollectOpSideEffects(&curr_op);
        }
        for (const auto& entry : op_side_effect_map_[&curr_op]) {
          UpdateSideEffectsByResourceId(entry.second, op_side_effect_map_[op]);
        }
      }
    }
  }

  // Collects op-based side effects for `op` in `op_side_effect_map_[op]`.
  void CollectOpSideEffects(Operation* op) {
    if (!MayHaveSideEffect(op)) return;
    // Skip following ops to avoid that every island, graph and function is
    // classified as unknown side-effecting.
    if (isa<tf_executor::YieldOp, tf_executor::FetchOp,
            mlir::func::ReturnOp>(op))
      return;

    // Propagate side effects from regions or functions attached to `op` for
    // some special cases.
    if (auto func = llvm::dyn_cast<func::FuncOp>(op)) {
      AddRegionSideEffectsForOp(func.getBody(), op);
    } else if (auto call = llvm::dyn_cast<CallOpInterface>(op)) {
      func::FuncOp func_op = dyn_cast<func::FuncOp>(
          call.resolveCallable(&symbol_table_collection_));
      if (func_op) {
        AddRegionSideEffectsForOp(func_op.getBody(), op);
      }
    } else if (auto if_op = llvm::dyn_cast<IfOp>(op)) {
      AddRegionSideEffectsForOp(if_op.then_function().getBody(), op);
      AddRegionSideEffectsForOp(if_op.else_function().getBody(), op);
    } else if (auto while_op = dyn_cast<WhileOp>(op)) {
      AddRegionSideEffectsForOp(while_op.body_function().getBody(), op);
    } else if (auto while_region_op = dyn_cast<WhileRegionOp>(op)) {
      AddRegionSideEffectsForOp(while_region_op.body(), op);
    } else if (auto case_op = dyn_cast<CaseOp>(op)) {
      llvm::SmallVector<func::FuncOp, 4> branch_funcs;
      case_op.get_branch_functions(branch_funcs);
      for (auto branch_func : branch_funcs) {
        AddRegionSideEffectsForOp(branch_func.getBody(), op);
      }
    } else if (isa<tf_device::LaunchOp, tf_device::ClusterOp,
                   tf_executor::IslandOp, tf_executor::GraphOp, IfRegionOp,
                   CaseRegionOp>(op)) {
      for (Region& region : op->getRegions()) {
        AddRegionSideEffectsForOp(region, op);
      }
    } else {
      // Now handle all other ops.
      auto& side_effects_by_resource_id = op_side_effect_map_[op];
      llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
      auto interface = dyn_cast<MemoryEffectOpInterface>(op);
      if (interface) interface.getEffects(effects);
      if (effects.empty()) {
        // The op is potentially side-effecting and doesn't have any effect
        // assigned, treat it as unknown side effect.
        SideEffects side_effects;
        side_effects.SetResourceId(kUnknownResourceId);
        side_effects.SetUnknownEffect();
        UpdateSideEffectsByResourceId(side_effects,
                                      side_effects_by_resource_id);
        // An unknown side effect dominates other side effects so we don't have
        // to add them and can return here.
        return;
      }
      // Add op-based side effects from regions (if any).
      for (Region& region : op->getRegions()) {
        AddRegionSideEffectsForOp(region, op);
      }
      // Add op-based side effects for the op itself.
      for (const auto& effect : effects) {
        // We handle value-based side effects for which we can use resource
        // alias analysis at a different place, skip here.
        if (ShouldUseResourceAliasAnalysis(effect)) continue;
        if (llvm::isa<ResourceEffects::MustExecute>(effect.getResource()))
          // We have this fake resource to avoid that certain ops are considered
          // dead or get pruned, ignore it for side effect analysis.
          continue;

        // Add side effects for op resource ID.
        std::string instance_str = "";
        SideEffects side_effects(GetSideEffectsFromEffectInstance(effect, op));
        if (auto resource_instance_op =
            dyn_cast<GetResourceInstanceInterface>(op)) {
          instance_str = resource_instance_op.GetResourceInstanceStr();
        }
        TypeID type_id = effect.getResource()->getResourceID();
        ResourceId resource_id = GetOpResourceId(type_id, instance_str);
        side_effects.SetResourceId(resource_id);
        UpdateSideEffectsByResourceId(side_effects,
                                      side_effects_by_resource_id);
        if (ResourceEffects::IsOnlySelfDependent(type_id)) {
          self_dependent_only_ids_.insert(resource_id);
        }
      }
    }
  }

  // Get internal op resource ID from MLIR type ID and instance ID.
  ResourceId GetOpResourceId(TypeID type_id, std::string instance_str) {
    auto emplace_result = type_instance_str_to_op_resource_id_.try_emplace(
        std::make_pair(type_id.getAsOpaquePointer(), instance_str),
        next_op_resource_id_);
    // Increment type ID if we have encountered a new resource type.
    if (emplace_result.second) ++next_op_resource_id_;
    return emplace_result.first->second;
  }

  // We use [0, kMaxResourceId] for resource IDs returned by resource alias
  // analysis and [kMaxResourceId + 1, ...] for resource IDs which we generate
  // for op-based side effects.
  const ResourceId kMaxResourceId =
      std::numeric_limits<ResourceId>::max() / 2;
  // Next available ID for op-based resources (resources not handled by resource
  // alias analysis).
  ResourceId next_op_resource_id_ = kMaxResourceId + 1;
  // Maps (type ID, instance ID) pairs to internal IDs for op-based resources.
  // Also see comment above. Instead of using TypeID directly we use its opaque
  // pointer.
  absl::node_hash_map<std::pair<const void*, std::string>, ResourceId>
    type_instance_str_to_op_resource_id_;
  // Used for faster callable resolution.
  SymbolTableCollection symbol_table_collection_;
  // Collect all op-based side effects here.
  OpSideEffectMap op_side_effect_map_;
  const SideEffectsByResourceId empty_side_effects_map_;

  // Set of all resource IDs which only have dependencies to themselves, not to
  // any other resource ID (including unknown resource ID).
  llvm::SmallDenseSet<ResourceId, 8> self_dependent_only_ids_;
};

// Collects all op-based and value-based side effects for `op` per resource ID.
SideEffectsByResourceId CollectSideEffectsByResourceId(
    Operation* op,
    const OpSideEffectCollector& op_side_effect_collector,
    const TF::ResourceAliasAnalysis::Info& alias_analysis) {
  SideEffectsByResourceId side_effects_by_resource_id;
  if (!MayHaveSideEffect(op)) return side_effects_by_resource_id;

  if (isa<tf_device::LaunchOp, tf_device::ClusterOp, tf_executor::IslandOp,
          tf_executor::GraphOp, IfRegionOp, CaseRegionOp, WhileRegionOp>(op)) {
    // For ops that are side-effecting only if their attached regions are,
    // collect effects for all ops in the regions instead of collecting effects
    // for the op itself. This is important to avoid conservatism and to find
    // resource variable accesses in regions which are not exposed to the op
    // interface.
    for (Region& region : op->getRegions()) {
      for (Operation& region_op : region.front().without_terminator()) {
        SideEffectsByResourceId region_op_effects =
            CollectSideEffectsByResourceId(
                &region_op,
                op_side_effect_collector,
                alias_analysis);
        for (const auto& [resource_id, side_effect] : region_op_effects) {
          UpdateSideEffectsByResourceId(side_effect,
                                        side_effects_by_resource_id);
        }
      }
    }
    return side_effects_by_resource_id;
  }

  // Copy op-based side effects.
  side_effects_by_resource_id =
      op_side_effect_collector.GetSideEffectsForOp(op);
  bool found_any_effect = !side_effects_by_resource_id.empty();

  // Collect value-based side effects from op interface.
  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (interface) interface.getEffects(effects);

  llvm::SmallDenseSet<Value, 8> processed_values;
  for (const auto& effect : effects) {
    Value value = effect.getValue();
    found_any_effect = true;

    // We only collect value-based side effects here for which we can use
    // resource alias analysis. Other side effects are treated as op-based
    // side effects.
    if (!ShouldUseResourceAliasAnalysis(effect)) continue;
    if (value) processed_values.insert(value);

    TypeID type_id = effect.getResource()->getResourceID();
    if (ResourceEffects::IsOnlySelfDependent(type_id)) {
      // For value-based side effects we currently treat resource types that are
      // only self-dependent conservatively, i.e., we do add dependencies
      // to/from unknown resource types. Currently, we don't have such cases and
      // there is no indication that we will need to support them in the future.
      LOG(WARNING) << "Self-dependent-only resource types are treated "
                      "conservatively for value-based side effects.";
    }

    // Add side effects for every potentially accessed resource ID.
    SideEffects side_effects(GetSideEffectsFromEffectInstance(effect, op));
    const auto& ids = GetResourceUniqueIdsOrUnknown(value, alias_analysis);
    for (ResourceId id : ids) {
      side_effects.SetResourceId(id);
      UpdateSideEffectsByResourceId(side_effects, side_effects_by_resource_id);
    }
  }

  auto add_remaining_effects = [&](auto resource_values) {
    for (Value resource_value : resource_values) {
      // If we already processed this value before, skip it.
      if (processed_values.count(resource_value) > 0) continue;
      found_any_effect = true;

      // Conservatively set unknown effect.
      SideEffects unknown_effect;
      unknown_effect.SetUnknownEffect();

      // Add side effects for every potentially accessed resource ID.
      const auto& ids =
          GetResourceUniqueIdsOrUnknown(resource_value, alias_analysis);
      for (ResourceId id : ids) {
        unknown_effect.SetResourceId(id);
        UpdateSideEffectsByResourceId(unknown_effect,
                                      side_effects_by_resource_id);
      }
    }
  };
  // Add value-based side effects for resource values which are not covered by
  // any side effect so far, for example, resource values being passed to
  // `tf.While` or `tf.If` ops which are not part of the op definition but
  // appear in a variadic input list.
  add_remaining_effects(filter_resources(op->getOperands()));
  add_remaining_effects(filter_resources(op->getResults()));

  if (!found_any_effect) {
    // We haven't collected any side effect but the op is potentially
    // side-effecting (otherwise we would have returned), therefore we have an
    // unknown side effect for an unknown resource.
    SideEffects unknown_effect;
    unknown_effect.SetUnknownEffect();
    unknown_effect.SetResourceId(kUnknownResourceId);
    UpdateSideEffectsByResourceId(unknown_effect,
                                  side_effects_by_resource_id);
  }
  return side_effects_by_resource_id;
}

//===----------------------------------------------------------------------===//
// SideEffectAnalysisInfo
//===----------------------------------------------------------------------===//

void SideEffectAnalysisInfo::AddPredecessorsForAccess(ResourceId resource_id,
                                                      Operation* op,
                                                      bool read_only) {
  VLOG(2) << "    Adding predecessors for resource " << resource_id;
  auto it = per_resource_access_info_.find(resource_id);
  if (it == per_resource_access_info_.end()) return;
  const auto& access_info = it->getSecond();

  auto& control_predecessors = control_predecessors_[op];
  bool is_last_write_indirectly_tracked = false;
  if (!read_only) {
    // Add reads after last write as predecessors.
    control_predecessors.insert(access_info.reads_since_last_write.begin(),
                                access_info.reads_since_last_write.end());
    // Last write is indirectly tracked by any read predecessor we added.
    is_last_write_indirectly_tracked =
        !access_info.reads_since_last_write.empty();
  }
  if (access_info.last_write && !is_last_write_indirectly_tracked) {
    // Add last write as predecessor.
    control_predecessors.insert(access_info.last_write);
  }
}

void SideEffectAnalysisInfo::UpdateAccess(ResourceId resource_id,
                                          Operation* op,
                                          bool read_only) {
  VLOG(2) << "    Updating access for resource " << resource_id;
  op_to_resource_ids_[op].push_back({resource_id, read_only});
  if (resource_id == kUnknownResourceId) {
    if (read_only) {
      // New unknown read is not tracked by any known resource access.
      for (auto& entry : per_resource_access_info_) {
        entry.getSecond().are_last_unknown_reads_tracked = false;
      }
    } else {
      // Unknown write can clear all other tracked information, since it acts
      // like a barrier.
      per_resource_access_info_.clear();
    }
  }
  auto& access_info = per_resource_access_info_[resource_id];
  if (read_only) {
    access_info.reads_since_last_write.push_back(op);
    // Last unknown write is indirectly tracked by this read (we have added the
    // write as a predecessor for `op` before).
    access_info.is_last_unknown_write_tracked = true;
  } else {
    access_info.last_write = op;
    access_info.reads_since_last_write.clear();
    // Last unknown read(s) and write are indirectly tracked by this write (we
    // have added the read(s) and write as predecessors for `op` before).
    access_info.are_last_unknown_reads_tracked = true;
    access_info.is_last_unknown_write_tracked = true;
    access_info.is_last_unknown_write_tracked_by_write = true;
  }
}

void SideEffectAnalysisInfo::AnalyzeFunction(func::FuncOp func_op) {
  // AnalyzeRegion() recursively analyzes the function body, and only populates
  // control_predecessors_.
  AnalyzeRegion(&func_op.getBody());
  // Populate sorted_control_predecessors_ and sorted_control_successors_ based
  // on control_predecessors.
  for (auto& entry : control_predecessors_) {
    auto op = entry.getFirst();
    auto& predecessors = entry.getSecond();
    auto& sorted_predecessors = sorted_control_predecessors_[op];
    for (Operation* predecessor : predecessors) {
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

  // Populate the control sinks (i.e. side-effecting ops with no control
  // successors) in the top level block.
  for (const auto& entry : sorted_control_predecessors_) {
    auto* op = entry.getFirst();
    if (op->getBlock() == &func_op.front() &&
        sorted_control_successors_.count(op) == 0) {
      sorted_control_sinks_.push_back(op);
    }
  }
  llvm::sort(sorted_control_sinks_, [](Operation* a, Operation* b) {
    return a->isBeforeInBlock(b);
  });
}

void SideEffectAnalysisInfo::AnalyzeRegion(Region* region) {
  // We explicitly iterate through the regions and blocks in order to handle
  // different nested regions separately.
  for (Block& block : *region) {
    for (Operation& op : block) {
      for (Region& child_region : op.getRegions()) {
        SideEffectAnalysisInfo child_analysis(
            &child_region, op_side_effect_collector_, alias_analysis_);
        // Move data from `child_analysis` to current region.
        for (auto& entry : child_analysis.control_predecessors_)
          control_predecessors_[entry.first] = std::move(entry.second);
        for (auto& entry : child_analysis.op_to_resource_ids_)
          op_to_resource_ids_[entry.first] = std::move(entry.second);
      }
      AnalyzeOp(&op);
    }
  }
}

ResourceIdSet
SideEffectAnalysisInfo::GetConflictingIds(ResourceId resource_id,
                                          bool is_fetch_op)  const {
  ResourceIdSet conflicting_ids;
  if (resource_id == kUnknownResourceId) {
    // Unknown resource has potential conflict with all other resources, except
    // those that are only self-dependent. For `Fetch` op make every resource
    // conflicting in any case to ensure that all side-effecting ops in
    // `Graph` feed into `Fetch` (its terminator).
    for (auto& entry : per_resource_access_info_) {
      ResourceId other_id = entry.getFirst();
      if (!op_side_effect_collector_.IsOnlySelfDependent(other_id) ||
          is_fetch_op)
        conflicting_ids.insert(other_id);
    }
  } else {
    conflicting_ids.insert(resource_id);
    // Resource has potential conflict with unknown resource, if not only
    // self-dependent.
    if (!op_side_effect_collector_.IsOnlySelfDependent(resource_id))
      conflicting_ids.insert(kUnknownResourceId);
  }
  return conflicting_ids;
}

void SideEffectAnalysisInfo::AnalyzeOp(Operation* op) {
  VLOG(2) << "Processing op " << mlir::debugString(*op);
  SideEffectsByResourceId side_effects_by_resource_id =
        CollectSideEffectsByResourceId(
            op,
            op_side_effect_collector_,
            alias_analysis_);

  // If the side-effecting op is a control source (i.e. it has no control
  // predecessors), then `control_predecessors_` won't be updated below.
  // However, we still want to track this op as it may have side effects visible
  // to ops outside the function.
  if (!side_effects_by_resource_id.empty()) control_predecessors_[op];

  // Traverse all resource IDs and their associated side effects.
  bool had_unknown_resource_read = false;
  for (auto pair : side_effects_by_resource_id) {
    ResourceId resource_id = pair.first;
    const SideEffects& side_effects = pair.second;
    const bool read_only = side_effects.IsReadOnly();
    VLOG(2) << "  Processing resource ID: " << resource_id
            << ", read-only effect: " << read_only;
    // An op that only allocates a resource is expected to return a handle that
    // is used by all other accesses of the same resource. That means, other ops
    // that access the same resource already have a data dependency on the
    // allocating op so it doesn't need any control predecessors or successors.
    if (side_effects.IsAllocOnly()) continue;
    // Effect is dominated by previous unknown resource read effect.
    if (read_only && had_unknown_resource_read) continue;

    ResourceIdSet conflicting_ids = GetConflictingIds(
        resource_id, isa<tf_executor::FetchOp>(op));

    // Add predecessors for conflicting IDs.
    bool is_unknown_access_indirectly_tracked = false;
    for (ResourceId id : conflicting_ids) {
      // Handle unknown resource later, access might already be indirectly
      // tracked by another resource access.
      if (id == kUnknownResourceId) continue;

      AddPredecessorsForAccess(id, op, read_only);
      is_unknown_access_indirectly_tracked |=
          IsUnknownAccessIndirectlyTrackedByResource(id, read_only);
    }
    // Add predecessors for unknown resource if necessary.
    if (conflicting_ids.contains(kUnknownResourceId) &&
        !is_unknown_access_indirectly_tracked)
      AddPredecessorsForAccess(kUnknownResourceId, op, read_only);
    // Update resource access.
    UpdateAccess(resource_id, op, read_only);

    // If this effect dominates all other possible effects, return here. Note
    // that if there is any effect for an unknown resource, then we encounter it
    // in the first iteration since `kUnknownResourceId` is smaller than all
    // other resource IDs.
    if (resource_id == kUnknownResourceId && !read_only) return;
    if (resource_id == kUnknownResourceId && read_only) {
      had_unknown_resource_read = true;
    }
  }
}

bool SideEffectAnalysisInfo::IsUnknownAccessIndirectlyTrackedByResource(
    ResourceId resource_id, bool read_only) {
  auto it = per_resource_access_info_.find(resource_id);
  if (it == per_resource_access_info_.end()) return false;
  auto access_info = it->getSecond();

  auto unknown_it = per_resource_access_info_.find(kUnknownResourceId);
  if (unknown_it == per_resource_access_info_.end()) return true;
  auto unknown_access_info = unknown_it->getSecond();

  bool no_unknown_read = unknown_access_info.reads_since_last_write.empty();
  bool no_unknown_write = (unknown_access_info.last_write == nullptr);

  // For the read-only case we only need that the last unknown write is already
  // tracked by the last `resource` write since we don't have dependencies to
  // any other read accesses.
  // Otherwise, we need that the last unknown read(s) and write are already
  // tracked by any read or write accesses of `resource`.
  bool is_tracked = read_only ?
      no_unknown_write || access_info.is_last_unknown_write_tracked_by_write :
      (no_unknown_write || access_info.is_last_unknown_write_tracked) &&
      (no_unknown_read || access_info.are_last_unknown_reads_tracked);
  if (is_tracked) {
    VLOG(2) << "      Unknown access indirectly tracked by resource "
            << resource_id;
  }
  return is_tracked;
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

const llvm::SmallVector<std::pair<ResourceId, bool>>&
SideEffectAnalysisInfo::GetResourceIds(Operation* op) const {
  auto it = op_to_resource_ids_.find(op);
  if (it == op_to_resource_ids_.end()) return empty_resource_ids_;
  return it->getSecond();
}

}  // namespace detail

SideEffectAnalysis::SideEffectAnalysis(ModuleOp module)
  // Analyze entire module for alias analysis info.
    : alias_analysis_(module) {
  // Collect op-based side effects for entire module.
  detail::OpSideEffectCollector op_side_effect_collector(module);

  // Analyze side effects for all functions in module.
  for (auto func : module.getOps<func::FuncOp>())
    this->info_map_.try_emplace(func, func,
                                op_side_effect_collector,
                                alias_analysis_.GetAnalysisForFunc(func));
}

}  // namespace TF
}  // namespace mlir
