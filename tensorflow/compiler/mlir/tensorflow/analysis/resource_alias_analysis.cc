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

#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace mlir {
namespace TF {

namespace {
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
}  // namespace

namespace detail {

//===----------------------------------------------------------------------===//
// BacktrackAnalysis
//===----------------------------------------------------------------------===//
// Holds backtrack analysis for all functions and regions within a module.
class BacktrackAnalysis {
 public:
  using InfoT = BacktrackAnalysisInfo;

  // Constructs the analysis by analyzing the given module.
  explicit BacktrackAnalysis(ModuleOp module);

  // Returns backtracking analysis for the given region.
  const InfoT& GetAnalysisForRegion(Region& region) const {
    auto it = info_map_.find(&region);
    assert(it != info_map_.end());
    return it->second;
  }

  // Returns backtracking analysis for the given function.
  const InfoT& GetAnalysisForFunc(FuncOp func) const {
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

 private:
  llvm::SmallDenseMap<Region*, InfoT> info_map_;
};

// Analyzes all regions attached to all operations in the module.
BacktrackAnalysis::BacktrackAnalysis(ModuleOp module) {
  module.walk([this](Operation* op) {
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
    } else {
      break;
    }
  }
  return value;
}
}  // namespace detail

namespace {

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
// ResourceAliasAnalysisInfo helper functions.
//===----------------------------------------------------------------------===//

constexpr char kResourceArgUniqueIdAttr[] = "tf._resource_arg_unique_id";

// Returns if a VarHandleOp is anonymous, which means it always creates a new
// variable.
bool IsResourceHandleAnonymous(VarHandleOp handle) {
  return handle.shared_name() == tensorflow::ResourceHandle::ANONYMOUS_NAME;
}

// Returns a string unique identifier for a non-anonymous VarHandleOp.
std::string GetVarHandleStringId(VarHandleOp handle) {
  auto device = handle.getAttrOfType<StringAttr>("device");
  return absl::StrCat(handle.container().str(), "/", handle.shared_name().str(),
                      "/", device ? device.getValue().str() : std::string(""));
}

// Finds a unique ID for a VarHandleOp's output. If it is anonymous, always
// creates a new ID; otherwise, tries to reuse the existing ID for the
// referenced variable if it exists, or creates a new one if not.
int64_t GetOrCreateIdForVarHandle(VarHandleOp handle, int64_t* next_id,
                                  llvm::StringMap<int64_t>* name_id_map) {
  // Always create a new ID for anonymous handle.
  if (IsResourceHandleAnonymous(handle)) return (*next_id)++;

  auto name = GetVarHandleStringId(handle);
  auto emplace_res = name_id_map->try_emplace(name, *next_id);
  // New ID created, increment next_id.
  if (emplace_res.second) ++(*next_id);
  return emplace_res.first->second;
}

}  // namespace

namespace detail {
//===----------------------------------------------------------------------===//
// ResourceAliasAnalysisInfo
//===----------------------------------------------------------------------===//

// Constructs the analysis info by analyzing the given function.
ResourceAliasAnalysisInfo::ResourceAliasAnalysisInfo(
    FuncOp func_op, const detail::BacktrackAnalysis& backtrack_analysis) {
  // This function populates resource_value_to_ids_ and id_to_resource_values_.

  int64_t next_unique_id = 0;

  // Helper to assign new unique id for all resources in the given list of
  // values.
  auto assign_unique_id_to_all = [&](ValueRange values) {
    for (Value value : filter_resources(values)) {
      AddValueUniqueIDMapping(value, next_unique_id++);
    }
  };

  // Helper to assign new unknown id for all resources in the given list of
  // values.
  auto assign_unknown_id_to_all = [&](ValueRange values) {
    for (Value value : filter_resources(values)) {
      AddValueUniqueIDMapping(value, kUnknownResourceId);
    }
  };

  // If the "tf.resource_arg_unique_id" argument attributes are present for
  // resource-type arguments, respect them when choosing IDs; otherwise, they
  // must not alias.
  const bool has_arg_unique_id_attrs =
      llvm::any_of(func_op.getArguments(), [&](const BlockArgument& arg) {
        return func_op.getArgAttr(arg.getArgNumber(), kResourceArgUniqueIdAttr);
      });
  // Maps the kResourceArgUniqueIdAttr attribute value to the internal integer
  // ID used by this pass.
  if (has_arg_unique_id_attrs) {
    llvm::SmallDenseMap<int64_t, int64_t> attr_id_to_internal_id;
    for (auto arg : filter_resources(func_op.getArguments())) {
      auto id_attr = func_op.getArgAttrOfType<IntegerAttr>(
          arg.getArgNumber(), kResourceArgUniqueIdAttr);
      assert(id_attr &&
             "tf.resource_arg_unique_id attribute should exist on either "
             "none or all arguments.");
      auto emplace_res = attr_id_to_internal_id.try_emplace(id_attr.getInt(),
                                                            next_unique_id++);
      AddValueUniqueIDMapping(arg, emplace_res.first->getSecond());
    }
  } else {
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

  llvm::StringMap<int64_t> var_handle_name_id_map;
  auto forward_input_to_output = [&](const Value& operand,
                                     const OpResult& result) {
    auto operand_it = resource_value_to_ids_.find(operand);
    assert(operand_it != resource_value_to_ids_.end() &&
           "A resource-type output does not have the corresponding "
           "resource-type input.");
    for (int64_t id : operand_it->second) AddValueUniqueIDMapping(result, id);
  };

  func_op.walk([&](Operation* op) {
    if (auto var_handle = dyn_cast<VarHandleOp>(op)) {
      AddValueUniqueIDMapping(
          var_handle.resource(),
          GetOrCreateIdForVarHandle(var_handle, &next_unique_id,
                                    &var_handle_name_id_map));
    } else if (llvm::isa<IdentityNOp, IdentityOp>(op)) {
      for (auto result : filter_resources(op->getResults()))
        forward_input_to_output(op->getOperand(result.getResultNumber()),
                                result);
    } else if (auto while_op = dyn_cast<WhileOp>(op)) {
      const auto& body_info =
          backtrack_analysis.GetAnalysisForFunc(while_op.body_func());
      // If a result is a passthrough of the body input, use the corresponding
      // operand's resource IDs.
      for (auto result : filter_resources(while_op.getResults())) {
        auto passthrough_arg = body_info.GetArg(result.getResultNumber());
        if (passthrough_arg) {
          forward_input_to_output(
              while_op.getOperand(passthrough_arg.getValue()), result);
        } else {
          AddValueUniqueIDMapping(result, kUnknownResourceId);
        }
      }
    } else if (auto while_region = dyn_cast<WhileRegionOp>(op)) {
      const auto& body_info =
          backtrack_analysis.GetAnalysisForRegion(while_region.body());
      // If a result is a passthrough of the body input, use the corresponding
      // operand's resource IDs.
      for (auto result : filter_resources(while_region.getResults())) {
        auto passthrough_arg = body_info.GetArg(result.getResultNumber());
        if (passthrough_arg) {
          forward_input_to_output(
              while_region.getOperand(passthrough_arg.getValue()), result);
        } else {
          AddValueUniqueIDMapping(result, kUnknownResourceId);
        }
      }
    } else if (auto if_op = dyn_cast<IfOp>(op)) {
      const auto& then_info =
          backtrack_analysis.GetAnalysisForFunc(if_op.then_func());
      const auto& else_info =
          backtrack_analysis.GetAnalysisForFunc(if_op.else_func());
      // If a result is a passthrough of both branches' inputs, merge the
      // resource IDs of corresponding operands for the two inputs.
      for (auto result : filter_resources(if_op.getResults())) {
        auto passthrough_then_arg = then_info.GetArg(result.getResultNumber());
        auto passthrough_else_arg = else_info.GetArg(result.getResultNumber());
        if (passthrough_then_arg && passthrough_else_arg) {
          Value then_operand = if_op.input()[passthrough_then_arg.getValue()];
          Value else_operand = if_op.input()[passthrough_else_arg.getValue()];
          forward_input_to_output(then_operand, result);
          forward_input_to_output(else_operand, result);
        } else {
          AddValueUniqueIDMapping(result, kUnknownResourceId);
        }
      }
    } else if (auto if_region = dyn_cast<IfRegionOp>(op)) {
      const auto& then_info =
          backtrack_analysis.GetAnalysisForRegion(if_region.then_branch());
      const auto& else_info =
          backtrack_analysis.GetAnalysisForRegion(if_region.else_branch());
      for (auto result : filter_resources(if_region.getResults())) {
        Value then_result = then_info.GetValue(result.getResultNumber());
        Value else_result = else_info.GetValue(result.getResultNumber());
        // For IfRegion, the walk would have visited the else and then regions
        // before visiting the IfRegion op. Backtracking of the then and else
        // results will either give a value computed within these regions,
        // or a region capture. If its a region capture, computed before this
        // IfRegion, it will have been visited earlier and a mapping would
        // exist for that value. If its computed within the region, then again
        // a mapping would exist.
        forward_input_to_output(then_result, result);
        forward_input_to_output(else_result, result);
      }
    } else if (auto call = dyn_cast<CallOpInterface>(op)) {
      FuncOp func = dyn_cast<FuncOp>(call.resolveCallable());
      if (!func) {
        assign_unknown_id_to_all(op->getResults());
        return WalkResult::advance();
      }
      const auto& func_info = backtrack_analysis.GetAnalysisForFunc(func);
      for (auto result : filter_resources(op->getResults())) {
        auto passthrough_arg = func_info.GetArg(result.getResultNumber());
        if (passthrough_arg) {
          forward_input_to_output(
              call.getArgOperands()[passthrough_arg.getValue()], result);
        } else {
          AddValueUniqueIDMapping(result, kUnknownResourceId);
        }
      }
    } else {
      assign_unknown_id_to_all(op->getResults());
    }
    return WalkResult::advance();
  });
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

ResourceAliasAnalysis::ResourceAliasAnalysis(Operation* op) {
  auto module = dyn_cast<ModuleOp>(op);
  assert(module);

  // Analyze all regions for backtracking info.
  detail::BacktrackAnalysis backtrack_analysis(module);

  // Analyze each function.
  for (auto func : module.getOps<FuncOp>())
    this->info_map_.try_emplace(func, func, backtrack_analysis);
}

}  // namespace TF
}  // namespace mlir
