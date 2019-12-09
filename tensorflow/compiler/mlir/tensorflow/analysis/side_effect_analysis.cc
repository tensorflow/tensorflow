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

#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace mlir {
namespace TF {

namespace {

constexpr int64_t kUnknownResourceId = -1;

// Returns if a VarHandleOp is anonymous, which means it always creates a new
// variable.
bool IsResourceHandleAnonymous(TF::VarHandleOp handle) {
  return handle.shared_name() == tensorflow::ResourceHandle::ANONYMOUS_NAME;
}

// Returns a string unique identifier for a non-anonymous VarHandleOp.
std::string GetVarHandleStringId(TF::VarHandleOp handle) {
  auto device = handle.getAttrOfType<StringAttr>("device");
  return absl::StrCat(handle.container().str(), "/", handle.shared_name().str(),
                      "/", device ? device.getValue().str() : std::string(""));
}

// Finds a unique ID for a VarHandleOp's output. If it is anonymous, always
// creates a new ID; otherwise, tries to reuse the existing ID for the
// referenced variable if it exists, or creates a new one if not.
int64_t GetOrCreateIdForVarHandle(TF::VarHandleOp handle, int64_t* next_id,
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

ResourceAliasAnalysis::ResourceAliasAnalysis(Operation* op) {
  auto func_op = llvm::dyn_cast<FuncOp>(op);
  if (!func_op) return;
  AnalyzeFunction(func_op);
}

void ResourceAliasAnalysis::AnalyzeFunction(FuncOp func_op) {
  // This function populates resource_value_to_ids_.
  //
  // TODO(yuanzx): Pass variable aliasing information to functions so we can
  // properly resolve aliasing arguments.
  //
  // Before having that, we assume function arguments do not alias each other.
  int64_t next_unique_id = 0;
  for (auto arg : func_op.getArguments()) {
    if (!mlir::getElementTypeOrSelf(arg->getType()).isa<TF::ResourceType>())
      continue;
    resource_value_to_ids_[arg].insert(next_unique_id++);
  }
  llvm::StringMap<int64_t> var_handle_name_id_map;
  auto forward_input_to_output = [&](Value* operand, Value* result) {
    if (!mlir::getElementTypeOrSelf(result->getType()).isa<TF::ResourceType>())
      return;
    auto& result_ids = resource_value_to_ids_[result];
    auto operand_it = resource_value_to_ids_.find(operand);
    assert(operand_it != resource_value_to_ids_.end() &&
           "A resource-type output does not have the corresponding "
           "resource-type input.");
    result_ids.insert(operand_it->getSecond().begin(),
                      operand_it->getSecond().end());
  };
  // TODO(yuanzx): Consider control-flow ops.
  func_op.walk([&](Operation* op) {
    if (auto var_handle = llvm::dyn_cast<TF::VarHandleOp>(op)) {
      resource_value_to_ids_[var_handle.resource()].insert(
          GetOrCreateIdForVarHandle(var_handle, &next_unique_id,
                                    &var_handle_name_id_map));
    } else if (llvm::isa<TF::IdentityNOp>(op) ||
               llvm::isa<TF::IdentityOp>(op)) {
      for (auto operand_and_result :
           llvm::zip(op->getOperands(), op->getResults())) {
        forward_input_to_output(std::get<0>(operand_and_result),
                                std::get<1>(operand_and_result));
      }
    } else if (auto replicate = llvm::dyn_cast<tf_device::ReplicateOp>(op)) {
      // The nested block for RepliateOp is handled separately in side-effect
      // analysis. Inside that block, we can still treat its block arguments as
      // different resources.
      for (auto arg : replicate.GetBody().getArguments()) {
        if (mlir::getElementTypeOrSelf(arg->getType())
                .isa<TF::ResourceType>()) {
          resource_value_to_ids_[arg].insert(next_unique_id++);
        }
      }
    } else {
      for (auto result : op->getResults()) {
        if (!mlir::getElementTypeOrSelf(result->getType())
                 .isa<TF::ResourceType>())
          continue;
        resource_value_to_ids_[result].insert(kUnknownResourceId);
      }
    }
  });
}

bool ResourceAliasAnalysis::IsUnknownResource(const Value* resource) const {
  auto it = resource_value_to_ids_.find(resource);
  assert(it != resource_value_to_ids_.end() && !it->getSecond().empty());
  // The set is sorted so we only need to check the first element since
  // kUnknownResourceId < 0.
  static_assert(kUnknownResourceId < 0,
                "kUnknownResourceId should be negative");
  return *it->getSecond().begin() == kUnknownResourceId;
}

const llvm::SmallSet<int64_t, 8>& ResourceAliasAnalysis::GetResourceUniqueIds(
    const Value* resource) const {
  auto it = resource_value_to_ids_.find(resource);
  assert(it != resource_value_to_ids_.end() && "Unseen resource was queried");
  return it->getSecond();
}

namespace {

// Returns a set that contains only kUnknownResourceId.
llvm::SmallDenseSet<int64_t, 8> UnknownResourceSet() {
  llvm::SmallDenseSet<int64_t, 8> unknown_set;
  unknown_set.insert(kUnknownResourceId);
  return unknown_set;
}

// Returns all resources that could be accessed by op, or UnknownResourceSet()
// if we cannot find all of them.
llvm::SmallDenseSet<int64_t, 8> FindAccessedResources(
    Operation* op, const ResourceAliasAnalysis& alias_analysis) {
  llvm::SmallDenseSet<int64_t, 8> resources;

  for (auto operand : op->getOperands()) {
    if (!mlir::getElementTypeOrSelf(operand->getType()).isa<TF::ResourceType>())
      continue;
    if (alias_analysis.IsUnknownResource(operand)) return UnknownResourceSet();
    const auto& ids = alias_analysis.GetResourceUniqueIds(operand);
    resources.insert(ids.begin(), ids.end());
  }
  for (auto result : op->getResults()) {
    if (!mlir::getElementTypeOrSelf(result->getType()).isa<TF::ResourceType>())
      continue;
    if (alias_analysis.IsUnknownResource(result)) return UnknownResourceSet();
    const auto& ids = alias_analysis.GetResourceUniqueIds(result);
    resources.insert(ids.begin(), ids.end());
  }
  return resources;
}

// Returns an XlaResourceOpInfo (or nullptr if it does not exist) that specifies
// the resource access type of the op. It tells whether the op is read only,
// etc.
//
// TODO(yuanzx): Define this information in a different place. Currently we use
// tensorflow/compiler/tf2xla/resource_operation_table.h.
const tensorflow::XlaResourceOpInfo* GetResourceInfoForOp(Operation* op) {
  auto op_name = op->getName().getStringRef().str();
  if (op->getName().getDialect() !=
      TF::TensorFlowDialect::getDialectNamespace()) {
    return nullptr;
  }
  return tensorflow::GetResourceOpInfoForOp(
      op->getName().getStringRef().split('.').second.str());
}

// Returns whether `op` accesses resources and it is known to be read-only.
bool OpIsReadOnly(Operation* op) {
  auto resource_op_info = GetResourceInfoForOp(op);
  return resource_op_info &&
         resource_op_info->kind() == tensorflow::XlaResourceOpKind::kRead;
}

// Returns if `op` is a resource declaration.
bool OpIsDeclaration(Operation* op,
                     const ResourceAliasAnalysis& alias_analysis) {
  // TODO(yuanzx): Add other types of resources.
  return llvm::isa<TF::VarHandleOp>(op) ||
         ((llvm::isa<TF::IdentityNOp>(op) || llvm::isa<TF::IdentityOp>(op)) &&
          !FindAccessedResources(op, alias_analysis).empty());
}

}  // namespace

void SideEffectAnalysis::TrackAccess(int64_t resource_id, Operation* op,
                                     bool read_only) {
  if (resource_id == kUnknownResourceId) {
    if (read_only) {
      // New unknown read is not tracked by any known resource access.
      for (auto& entry : per_resource_access_info_) {
        entry.getSecond().tracked_last_unknown_read = false;
      }
    } else {
      // Unknown write can clear all other tracked information, since it acts
      // like a barrier.
      per_resource_access_info_.clear();
    }
  }
  auto& info = per_resource_access_info_[resource_id];
  if (read_only) {
    info.reads_since_last_write.push_back(op);
    // Resource read must have carried control dependencies of unknown write.
    info.tracked_last_unknown_write = true;
  } else {
    // Resource write must have carried control dependencies of unknown access.
    info.tracked_last_unknown_write = true;
    info.tracked_last_unknown_read = true;
    info.last_write = op;
    info.reads_since_last_write.clear();
  }
}

void SideEffectAnalysis::AddPredecessorsForAccess(int64_t resource_id,
                                                  Operation* op,
                                                  bool read_only) {
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

void SideEffectAnalysis::AnalyzeFunction(
    FuncOp func_op, const ResourceAliasAnalysis& alias_analysis) {
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

void SideEffectAnalysis::AnalyzeRegion(
    Region* region, const ResourceAliasAnalysis& alias_analysis) {
  // This function populates control_predecessors_ by walking through the
  // region, and tracking resource accesses in per_resource_access_info_.

  // Returns whether an access to `resource` can skip control edges from
  // prevoius accesses to unknown resources, due to that earlier accesses to
  // `resource` already indirectly tracked previous accesses to uknown
  // resources. `read_only` specifies the type of access of the current op being
  // considered.
  auto unknown_access_indirectly_tracked_by_resource = [&](int64_t resource,
                                                           bool read_only) {
    auto it = per_resource_access_info_.find(resource);
    if (it == per_resource_access_info_.end()) return false;
    auto unknown_it = per_resource_access_info_.find(kUnknownResourceId);
    const bool no_unknown_read =
        unknown_it == per_resource_access_info_.end() ||
        unknown_it->getSecond().reads_since_last_write.empty();
    return read_only
               ? it->second.tracked_last_unknown_write
               : it->second.tracked_last_unknown_write &&
                     (it->second.tracked_last_unknown_read || no_unknown_read);
  };

  // We explicitly iterates through the regions and blocks, in order to handle
  // different nested regions separately.
  for (auto& block : *region) {
    for (auto& op : block) {
      if (op.getNumRegions() > 0) {
        llvm::SmallVector<SideEffectAnalysis, 4> child_analyses;
        for (auto& child_region : op.getRegions()) {
          child_analyses.emplace_back();
          child_analyses.back().AnalyzeRegion(&child_region, alias_analysis);
        }
        ConsumeChildAnalyses(std::move(child_analyses));
      }

      // We do not need explicit control edges for declaration ops.
      if (OpIsDeclaration(&op, alias_analysis)) continue;

      auto resource_op_info = GetResourceInfoForOp(&op);
      if (!resource_op_info && op.hasNoSideEffect()) continue;

      llvm::SmallDenseSet<int64_t, 8> resources =
          resource_op_info ? FindAccessedResources(&op, alias_analysis)
                           : UnknownResourceSet();
      assert(!resources.empty());
      const bool is_unknown = resources.count(kUnknownResourceId) > 0;
      const bool read_only = OpIsReadOnly(&op);
      bool indirectly_tracked_unknown_access = false;
      // First add edges from known resources.
      if (is_unknown) {
        for (auto& entry : per_resource_access_info_) {
          if (entry.getFirst() == kUnknownResourceId) continue;
          AddPredecessorsForAccess(entry.getFirst(), &op, read_only);
          indirectly_tracked_unknown_access |=
              unknown_access_indirectly_tracked_by_resource(entry.getFirst(),
                                                            read_only);
        }
      } else {
        for (int64_t resource : resources) {
          AddPredecessorsForAccess(resource, &op, read_only);
          indirectly_tracked_unknown_access |=
              unknown_access_indirectly_tracked_by_resource(resource,
                                                            read_only);
          // Update access info for known resources.
          TrackAccess(resource, &op, read_only);
        }
      }
      // If not indirectly tracked, add edges from the unknown resource.
      if (!indirectly_tracked_unknown_access) {
        AddPredecessorsForAccess(kUnknownResourceId, &op, read_only);
      }
      if (is_unknown) {
        // Update access info for unknown resource.
        TrackAccess(kUnknownResourceId, &op, read_only);
      }
    }
  }
}

void SideEffectAnalysis::ConsumeChildAnalyses(
    llvm::SmallVector<SideEffectAnalysis, 4>&& children) {
  for (auto& child : children) {
    for (auto& entry : child.control_predecessors_) {
      control_predecessors_[entry.getFirst()] = std::move(entry.getSecond());
    }
  }
}

llvm::SmallVector<Operation*, 4> SideEffectAnalysis::DirectControlPredecessors(
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

llvm::SmallVector<Operation*, 4> SideEffectAnalysis::DirectControlSuccessors(
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

SideEffectAnalysis::SideEffectAnalysis(Operation* op) {
  auto func_op = llvm::dyn_cast<FuncOp>(op);
  if (!func_op) return;
  ResourceAliasAnalysis alias_analysis(op);
  AnalyzeFunction(func_op, alias_analysis);
}

}  // namespace TF
}  // namespace mlir
