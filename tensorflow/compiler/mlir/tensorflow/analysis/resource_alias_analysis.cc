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

#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"

#include <cstdint>
#include <initializer_list>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {
namespace detail {

//===----------------------------------------------------------------------===//
// BacktrackAnalysisInfo
//===----------------------------------------------------------------------===//
// Class to hold backtrack analysis for a results of a region. Backtrack
// analysis will trace back the definition of return values of regions through
// pass-through operations, so that the return value of the region will have the
// same value as the backtracked value.
class BacktrackAnalysisInfo {
 public:
  // Initializes the backtrack analysis for the given region.
  explicit BacktrackAnalysisInfo(Region& region,
                                 detail::BacktrackAnalysis& backtrack_analysis);

  BacktrackAnalysisInfo(BacktrackAnalysisInfo&&) = default;

  // Returns the value to which the given result number of the region can be
  // backtracked to.
  Value GetValue(int result_index) const {
    return backtracked_values_[result_index];
  }

  // Returns the argument index of the region to which the given result number
  // can backtracked to. Such results will be called "function passthrough". If
  // the result cannot be backtracked to a region argument, returns llvm::None.
  llvm::Optional<int> GetArg(int result_index) const {
    if (auto arg = GetValue(result_index).dyn_cast<BlockArgument>())
      if (arg.getParentBlock() == &region_->front()) return arg.getArgNumber();
    return llvm::None;
  }

 private:
  friend class detail::BacktrackAnalysis;

  // Region for which this object holds the analysis info.
  Region* region_;

  // Backtracked values indexed by the result number.
  llvm::SmallVector<Value, 4> backtracked_values_;
};

//===----------------------------------------------------------------------===//
// BacktrackAnalysis
//===----------------------------------------------------------------------===//
// Holds backtrack analysis for all functions and regions within a module.
class BacktrackAnalysis {
 public:
  using InfoT = BacktrackAnalysisInfo;

  // Constructs the analysis by analyzing the given module.
  BacktrackAnalysis(ModuleOp module,
                    SymbolTableCollection& symbol_table_collection);

  // Returns backtracking analysis for the given region.
  const InfoT& GetAnalysisForRegion(Region& region) const {
    auto it = info_map_.find(&region);
    assert(it != info_map_.end());
    return it->second;
  }

  // Returns backtracking analysis for the given function.
  const InfoT& GetAnalysisForFunc(func::FuncOp func) const {
    return GetAnalysisForRegion(func.getBody());
  }

  // Backtracks the given value.
  Value BacktrackValue(Value value);

 private:
  // Returns the analysis for the given region (analyzing the region if it has
  // not yet been analyzed).
  const InfoT& GetOrCreateAnalysis(Region& region) {
    auto it = info_map_.find(&region);
    if (it == info_map_.end()) {
      // Note: Keep object construction and insertion separate. If we use
      // emplace() to construct and insert in a single shot, when analyzing
      // this region, calls to BacktrackValue() may end up inserting additional
      // entries in the map, causing the underlying storage to be moved. This
      // would also include this pertially constructed object that we have just
      // inserted into the map and are constructing it. To avoid this issue,
      // construct the analysis object separately and then insert it into the
      // map.
      InfoT info(region, *this);
      info_map_.insert({&region, std::move(info)});
    }

    return GetAnalysisForRegion(region);
  }

  // Returns the backtrack analysis for the given region if it exists.
  // If the region has not yet been analyzed, returns llvm::None.
  Optional<const InfoT*> GetAnalysisIfExists(Region& region) const {
    auto it = info_map_.find(&region);
    if (it == info_map_.end()) return llvm::None;
    return &it->second;
  }

  Optional<const InfoT*> GetAnalysisIfExists(func::FuncOp func) const {
    return GetAnalysisIfExists(func.getBody());
  }

 private:
  llvm::SmallDenseMap<Region*, InfoT> info_map_;
  SymbolTableCollection& symbol_table_collection_;
};

// Analyzes all regions attached to all operations in the module.
BacktrackAnalysis::BacktrackAnalysis(
    ModuleOp module, SymbolTableCollection& symbol_table_collection)
    : symbol_table_collection_(symbol_table_collection) {
  const CallGraph call_graph(module);

  // Visit functions bottom up when doing the analysis. Note that SCC iterator
  // has the property that if there is an edge from SCC1->SCC2, SCC1 is visited
  // after SCC2, i.e., the graph is traversed bottom up just the way we want.
  auto scc_begin = llvm::scc_begin(&call_graph);
  auto scc_end = llvm::scc_end(&call_graph);
  for (auto& scc : make_range(scc_begin, scc_end)) {
    // Each SCC node is a collection of callgraph nodes that form a cycle. We
    // will visit these nodes in an arbitrary order. If a node being visited
    // calls a function that has not yet been analyzed, we will not be able to
    // backtrack through that function call (our analysis will be correct but
    // pessimistic).
    for (CallGraphNode* node : scc) {
      if (node->isExternal()) continue;
      Region* region = node->getCallableRegion();
      GetOrCreateAnalysis(*region);
    }
  }

  // This above call graph analysis will cover all regions attached to functions
  // but we also need to analyze regions attached to other ops.
  module->walk([this](Operation* op) {
    if (op->hasTrait<OpTrait::NoTerminator>()) return;
    for (Region& region : op->getRegions()) GetOrCreateAnalysis(region);
  });
}

// Backtracks the definition of `value` looking through passthrough ops.
// Returns a non-null value and can return `value` if backtracking is not
// possible.
Value BacktrackAnalysis::BacktrackValue(Value value) {
  while (Operation* op = value.getDefiningOp()) {
    int res_index = value.cast<OpResult>().getResultNumber();
    if (auto graph = dyn_cast<tf_executor::GraphOp>(op)) {
      value = graph.GetFetch().getOperand(res_index);
    } else if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
      // Control output is generated by the IslandOp, not the yield in
      // in the Island body.
      if (value == island.control()) break;
      value = island.GetYield().getOperand(res_index);
    } else if (isa<IdentityNOp, IdentityOp>(op)) {
      value = op->getOperand(res_index);
    } else if (auto call = dyn_cast<CallOpInterface>(op)) {
      func::FuncOp func = dyn_cast<func::FuncOp>(
          call.resolveCallable(&symbol_table_collection_));
      if (!func) break;
      // Check if the function being called has been analyzed. if not,
      // we cannot backtrack the value further.
      Optional<const InfoT*> callee_info = GetAnalysisIfExists(func);
      if (!callee_info) break;
      Optional<int> passthrough_arg = callee_info.getValue()->GetArg(res_index);
      if (!passthrough_arg) break;
      value = call.getArgOperands()[passthrough_arg.getValue()];
    } else if (isa<tf_device::LaunchOp, tf_device::ClusterOp>(op)) {
      value = op->getRegion(0).front().getTerminator()->getOperand(res_index);
    } else {
      break;
    }
  }
  return value;
}

// Analyze the region.
BacktrackAnalysisInfo::BacktrackAnalysisInfo(
    Region& region, detail::BacktrackAnalysis& backtrack_analysis)
    : region_(&region) {
  if (region.empty()) return;

  assert(llvm::hasSingleElement(region.getBlocks()));

  auto results = region.front().getTerminator()->getOperands();
  if (results.empty()) return;

  backtracked_values_.reserve(results.size());
  for (auto result : results)
    backtracked_values_.push_back(backtrack_analysis.BacktrackValue(result));
}

//===----------------------------------------------------------------------===//
// ResourceAliasAnalysisInfo
//===----------------------------------------------------------------------===//

namespace {

constexpr char kResourceArgUniqueIdAttr[] = "tf._resource_arg_unique_id";

bool IsResourceAllocatingOp(Operation* op) {
  auto mem_interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!mem_interface) return false;

  for (Value value : filter_resources(op->getResults())) {
    llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
    mem_interface.getEffectsOnValue(value, effects);
    for (auto& effect_instance : effects) {
      if (isa<MemoryEffects::Allocate>(effect_instance.getEffect())) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

constexpr int64_t ResourceAliasAnalysisInfo::kUnknownResourceId;

void IncrementResourceTypeId(int64_t& resource_type_id) {
  if (resource_type_id == ResourceAliasAnalysisInfo::kMaxResourceTypeId) {
    // We don't expect this to happen, currently there are 10 resource types in
    // TF dialect. Still, it should be visible if this ever happens.
    LOG(WARNING) << "reached limit for supported number of resource types ("
                 << ResourceAliasAnalysisInfo::kMaxResourceTypeId
                 << "); this could lead to overly conservative execution order";
    // Note: By not incrementing `resource_type_id` we still maintain
    // correctness, we might only handle different resource types as the same
    // type (for ID `kMaxResourceTypeId`) which is overly conservative.
  } else {
    ++resource_type_id;
  }
}

// Constructs the analysis info by analyzing the given function.
ResourceAliasAnalysisInfo::ResourceAliasAnalysisInfo(
    func::FuncOp func_op, const BacktrackAnalysis& backtrack_analysis,
    SymbolTableCollection& symbol_table_collection) {
  // This function populates resource_value_to_ids_ and id_to_resource_values_.

  // See `ResourceAliasAnalysisInfo` class for ID semantics.
  int64_t next_unique_type_id = 0;
  int64_t next_unique_instance_id = kMaxResourceTypeId + 1;

  // Helper to assign new unique id for all resources in the given list of
  // values.
  auto assign_unique_id_to_all = [&](ValueRange values) {
    for (Value value : filter_resources(values)) {
      AddValueUniqueIDMapping(value, next_unique_instance_id++);
    }
  };

  // Helper to assign new unknown id for all resources in the given list of
  // values.
  auto assign_unknown_id_to_all = [&](ValueRange values) {
    for (Value value : filter_resources(values)) {
      AddValueUniqueIDMapping(value, kUnknownResourceId);
    }
  };

  // If `tf.resource_arg_unique_id` argument attributes are present for
  // resource-type arguments, use those to decide which arguments correspond to
  // the same resource (and thus need the same ID). Otherwise, they must not
  // alias.
  const bool has_arg_unique_id_attrs =
      llvm::any_of(func_op.getArguments(), [&](const BlockArgument& arg) {
        return func_op.getArgAttr(arg.getArgNumber(), kResourceArgUniqueIdAttr);
      });
  if (has_arg_unique_id_attrs) {
    // Resource arguments have IDs attached (via `kResourceArgUniqueIdAttr`)
    // that represent different resources. Map those IDs to the internal
    // instance IDs used by this pass.
    llvm::SmallDenseMap<int64_t, int64_t> attr_id_to_internal_id;
    for (auto arg : filter_resources(func_op.getArguments())) {
      auto id_attr = func_op.getArgAttrOfType<IntegerAttr>(
          arg.getArgNumber(), kResourceArgUniqueIdAttr);
      assert(id_attr &&
             "tf.resource_arg_unique_id attribute should exist on either "
             "none or all arguments.");
      auto emplace_res = attr_id_to_internal_id.try_emplace(
          id_attr.getInt(), next_unique_instance_id);
      AddValueUniqueIDMapping(arg, emplace_res.first->getSecond());
      // Only increment ID if it has been used.
      if (emplace_res.second) ++next_unique_instance_id;
    }
  } else {
    // No `kResourceArgUniqueIdAttr` attribute is present, so all resource
    // arguments must correspond to different resources and we can assign unique
    // IDs.
    assign_unique_id_to_all(func_op.getArguments());
  }

  // Since this analysis is neither inter-procedural nor inter-regional,
  // each region attached to Op's within a function is analyzed independently.
  // Seed this analysis for each such region by mapping all resource arguments
  // for such regions to a new unique-id. This is required because walk() walks
  // the attached regions first before visiting the op, so there is no
  // opportunity during the walk to seed region arguments. Also note that walk
  // eventually also visits the Op on which the walk() is called, so make sure
  // we do not overwrite the function argument mapping here.
  func_op.walk([&](Operation* op) {
    if (op == func_op) return;
    for (Region& region : op->getRegions()) {
      assign_unique_id_to_all(region.getArguments());
    }
  });

  llvm::SmallDenseMap<ResourceHandle, int64_t> resource_handle_id_map;
  func_op.walk([&](Operation* op) {
    if (auto resource_alloc = dyn_cast<ResourceHandleAllocatorInterface>(op)) {
      llvm::SmallVector<ResourceHandleValueAndId, 4> resources =
          resource_alloc.GetResourceHandleValueAndIdList(
              resource_handle_id_map, next_unique_instance_id);
      for (auto& resource_handle : resources) {
        AddValueUniqueIDMapping(resource_handle.value, resource_handle.id);
        // Keep track of IDs of resources that are allocated by ops with
        // `UniqueResourceAllocation` trait, this can be utilized for while-loop
        // parallelization (every iteration creates a new unique resource).
        if (op->hasTrait<OpTrait::TF::UniqueResourceAllocation>()) {
          unique_resource_allocation_ids_.insert(resource_handle.id);
        }
      }
    } else if (llvm::isa<TPUReplicatedInputOp>(op)) {
      // TPUReplicateInput only has a single result but we get all results
      // to use filter_resources and for consistency.
      for (auto result : filter_resources(op->getResults())) {
        for (auto operand : op->getOperands()) {
          PropagateInputToOutput(operand, result);
        }
      }
    } else if (llvm::isa<IdentityNOp, IdentityOp>(op)) {
      for (auto result : filter_resources(op->getResults()))
        PropagateInputToOutput(op->getOperand(result.getResultNumber()),
                               result);
    } else if (auto while_op = dyn_cast<WhileOp>(op)) {
      AnalyzeWhileLoop(while_op, backtrack_analysis.GetAnalysisForFunc(
                                     while_op.body_function()));
    } else if (auto while_region = dyn_cast<WhileRegionOp>(op)) {
      AnalyzeWhileLoop(while_region, backtrack_analysis.GetAnalysisForRegion(
                                         while_region.body()));
    } else if (auto case_op = dyn_cast<CaseOp>(op)) {
      llvm::SmallVector<func::FuncOp, 4> functions;
      case_op.get_branch_functions(functions);
      AnalyzeFunctionalCaseOrIfOp(case_op, functions, backtrack_analysis);
    } else if (auto if_op = dyn_cast<IfOp>(op)) {
      AnalyzeFunctionalCaseOrIfOp(
          if_op, {if_op.then_function(), if_op.else_function()},
          backtrack_analysis);
    } else if (llvm::isa<CaseRegionOp, IfRegionOp>(op)) {
      AnalyzeRegionCaseOrIfOp(op, backtrack_analysis);
    } else if (auto call = dyn_cast<CallOpInterface>(op)) {
      func::FuncOp func = dyn_cast_or_null<func::FuncOp>(
          call.resolveCallable(&symbol_table_collection));
      if (!func) {
        assign_unknown_id_to_all(op->getResults());
        return WalkResult::advance();
      }
      const auto& func_info = backtrack_analysis.GetAnalysisForFunc(func);
      for (auto result : filter_resources(op->getResults())) {
        auto passthrough_arg = func_info.GetArg(result.getResultNumber());
        if (passthrough_arg) {
          PropagateInputToOutput(
              call.getArgOperands()[passthrough_arg.getValue()], result);
        } else {
          AddValueUniqueIDMapping(result, kUnknownResourceId);
        }
      }
    } else if (isa<tf_device::LaunchOp, tf_device::ClusterOp,
                   tf_executor::IslandOp, tf_executor::GraphOp>(op) &&
               op->getNumRegions() == 1) {
      Region& region = op->getRegion(0);
      const auto& body_info = backtrack_analysis.GetAnalysisForRegion(region);
      for (auto result : filter_resources(op->getResults())) {
        Value body_result = body_info.GetValue(result.getResultNumber());
        PropagateInputToOutput(body_result, result);
      }
    } else {
      auto mem_interface = dyn_cast<MemoryEffectOpInterface>(op);
      for (Value value : filter_resources(op->getResults())) {
        // Set unknown ID first, reset later if applicable.
        int64_t resource_id = kUnknownResourceId;

        if (mem_interface) {
          auto alloc_effect =
              mem_interface.getEffectOnValue<MemoryEffects::Allocate>(value);
          if (alloc_effect) {
            TypeID mlir_type_id =
                alloc_effect.getValue().getResource()->getResourceID();
            // Update or lookup internal type ID.
            auto emplace_result = type_id_to_internal_type_id_.try_emplace(
                mlir_type_id, next_unique_type_id);
            // Change unknown ID to type-based ID.
            resource_id = emplace_result.first->getSecond();
            // Only increment ID if we have encountered a new resource type.
            if (emplace_result.second)
              IncrementResourceTypeId(next_unique_type_id);
          }
        }
        AddValueUniqueIDMapping(value, resource_id);
      }
    }
    return WalkResult::advance();
  });
}

// Propagates the resource IDs from an input operand to a result. Returns true
// if the mapping changed.
bool ResourceAliasAnalysisInfo::PropagateInputToOutput(const Value& operand,
                                                       const OpResult& result) {
  auto operand_it = resource_value_to_ids_.find(operand);
  assert(operand_it != resource_value_to_ids_.end() &&
         "A resource-type output does not have the corresponding "
         "resource-type input.");
  bool change = false;
  for (int64_t id : operand_it->second)
    change = AddValueUniqueIDMapping(result, id) || change;
  return change;
}

// Analyzes while loops to compute resourceIDs for the loop results.
//
// (1) The base case for the analysis is that if the loop body does not execute
//     at all, the resource IDs for each result is the same as the resource IDs
//     of the corresponding input.
// (2) If the loop does execute one or more times, then we need to account for
//     data flow through the body of the while loop. If result #r is the same
//     as arg #a of the loop body (pass through argument), then we can reason
//     further, else if the result is not a passthrough, we mark it as unknown.
// (3) For passthrough results, if result #r is the same as arg #a of the loop
//     body, after one iteration, result #r = arg #a, so we need to also
//     propagate arg #a to result #r. After another iteration, arg #a of the
//     loop body will be result #a of the previous iteration. So then we need
//     propagate from result #a to result #r. Generalizing, the resource ID
//     propagation (for results which are passthrough) looks like:
//
//     for r in (0, num_results) : result[r] = arg[r];
//     repeat till no change {
//       a = passthrough arg for result #r;
//       result[r] += result[a];
//     }
//
void ResourceAliasAnalysisInfo::AnalyzeWhileLoop(
    Operation* while_op, const BacktrackAnalysisInfo& body_info) {
  // Seed the resource IDs for the results using either the resource ID of the
  // passthrough arg, or unknown. We need to perform further analysis if we
  // find a passthrough arg which is not the same as corresponding the result #.
  llvm::SmallVector<Optional<int>, 4> passthrough_args(
      while_op->getNumResults());
  bool need_analysis = false;
  for (auto result : filter_resources(while_op->getResults())) {
    int result_index = result.getResultNumber();
    passthrough_args[result_index] = body_info.GetArg(result_index);
    if (passthrough_args[result_index]) {
      int passthru_index = passthrough_args[result_index].getValue();
      PropagateInputToOutput(while_op->getOperand(passthru_index), result);
      need_analysis |=
          !IsUnknownResource(result) && passthru_index != result_index;
    } else {
      AddValueUniqueIDMapping(result, kUnknownResourceId);
    }
  }

  if (!need_analysis) return;

  // We found a result that is not unknown and whose passthrough operand index
  // is not the same as the result index, which means there is "crosstalk"
  // between 2 or more operands. In that case, we do an iterative propagation
  // of resource IDs till the results converge.
  bool change = true;
  while (change) {
    change = false;
    for (auto result : filter_resources(while_op->getResults())) {
      if (IsUnknownResource(result)) continue;
      // If this result has a valid passthrough arg, propagate resource IDs
      // from the result of the passthrough arg
      int result_index = result.getResultNumber();
      int passthru_index = passthrough_args[result_index].getValue();
      change =
          PropagateInputToOutput(while_op->getResult(passthru_index), result) ||
          change;
    }
  }
}

template <class CaseOrIfOp>
void ResourceAliasAnalysisInfo::AnalyzeFunctionalCaseOrIfOp(
    CaseOrIfOp case_or_if_op, llvm::ArrayRef<func::FuncOp> functions,
    const BacktrackAnalysis& backtrack_analysis) {
  llvm::SmallVector<const BacktrackAnalysisInfo*, 2> infos;
  infos.reserve(functions.size());
  for (func::FuncOp func : functions)
    infos.push_back(&backtrack_analysis.GetAnalysisForFunc(func));

  // If a result is a passthrough of all branches' inputs, merge the resource
  // IDs of corresponding operands for all the inputs.
  for (auto result : filter_resources(case_or_if_op.getResults())) {
    llvm::SmallVector<llvm::Optional<int>, 2> passthrough_args;
    passthrough_args.reserve(functions.size());
    for (const auto* info : infos)
      passthrough_args.emplace_back(info->GetArg(result.getResultNumber()));

    const bool all_passthrough_args_known = llvm::all_of(
        passthrough_args, [](const llvm::Optional<int>& passthrough_arg) {
          return passthrough_arg.has_value();
        });
    if (all_passthrough_args_known) {
      for (const auto& passthrough_arg : passthrough_args) {
        Value operand = case_or_if_op.input()[passthrough_arg.getValue()];
        PropagateInputToOutput(operand, result);
      }
    } else {
      AddValueUniqueIDMapping(result, kUnknownResourceId);
    }
  }
}

void ResourceAliasAnalysisInfo::AnalyzeRegionCaseOrIfOp(
    Operation* case_or_if_op, const BacktrackAnalysis& backtrack_analysis) {
  llvm::SmallVector<const BacktrackAnalysisInfo*, 2> infos;
  infos.reserve(case_or_if_op->getNumRegions());
  for (Region& region : case_or_if_op->getRegions())
    infos.push_back(&backtrack_analysis.GetAnalysisForRegion(region));

  // For region Case/If, the walk would have visited all branch regions before
  // visiting the Case/If op. Backtracking of each region results will either
  // give a value computed within these regions, or a region capture. If it is a
  // region capture computed before this Case/If, it will have been visited
  // earlier and a mapping would exist for that value. If it is computed within
  // the region, then again a mapping would exist.
  for (auto result : filter_resources(case_or_if_op->getResults())) {
    for (const auto* info : infos) {
      Value region_result = info->GetValue(result.getResultNumber());
      PropagateInputToOutput(region_result, result);
    }
  }
}

bool ResourceAliasAnalysisInfo::IsUnknownResource(Value resource) const {
  auto it = resource_value_to_ids_.find(resource);
  assert(it != resource_value_to_ids_.end() && !it->getSecond().empty());
  // The set is sorted so we only need to check the first element since
  // kUnknownResourceId < 0.
  static_assert(kUnknownResourceId < 0,
                "kUnknownResourceId should be negative");
  return *it->getSecond().begin() == kUnknownResourceId;
}

const llvm::SmallSet<int64_t, 8>&
ResourceAliasAnalysisInfo::GetResourceUniqueIds(Value resource) const {
  assert(!IsUnknownResource(resource));
  auto it = resource_value_to_ids_.find(resource);
  assert(it != resource_value_to_ids_.end() && "Unseen resource was queried");
  return it->getSecond();
}

const llvm::SmallSetVector<Value, 8>&
ResourceAliasAnalysisInfo::GetUniqueIdResources(const int64_t id) const {
  auto it = id_to_resource_values_.find(id);
  assert(it != id_to_resource_values_.end() && "Unseen id was queried");
  return it->getSecond();
}

llvm::SmallSetVector<Value, 8> ResourceAliasAnalysisInfo::GetResourceAliases(
    Value resource) const {
  assert(!IsUnknownResource(resource) && "Unknown resource was queried");
  llvm::SmallSetVector<Value, 8> aliases;
  for (int64_t id : GetResourceUniqueIds(resource)) {
    const llvm::SmallSetVector<Value, 8>& resources_aliasing_id =
        GetUniqueIdResources(id);
    aliases.insert(resources_aliasing_id.begin(), resources_aliasing_id.end());
  }
  // If there are resources that were marked as unknown, they alias with all
  // other resources.
  auto it = id_to_resource_values_.find(kUnknownResourceId);
  if (it != id_to_resource_values_.end())
    aliases.insert(it->getSecond().begin(), it->getSecond().end());
  return aliases;
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// ResourceAliasAnalysis
//===----------------------------------------------------------------------===//

ResourceAliasAnalysis::ResourceAliasAnalysis(ModuleOp module) {
  // Create symbol table for module.
  SymbolTableCollection symbol_table_collection;
  symbol_table_collection.getSymbolTable(module);
  // Analyze all regions for backtracking info.
  detail::BacktrackAnalysis backtrack_analysis(module, symbol_table_collection);

  // Analyze each function.
  for (auto func : module.getOps<func::FuncOp>())
    this->info_map_.try_emplace(func, func, backtrack_analysis,
                                symbol_table_collection);
}

}  // namespace TF
}  // namespace mlir
