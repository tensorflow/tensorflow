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

// This pass lifts resource variable operations outside of device computation.

#include <cstddef>
#include <cstdint>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace mlir {

namespace {

constexpr char kDeviceAttr[] = "device";

#define GEN_PASS_DEF_RESOURCEOPLIFTINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

// Lift resource operations out of device computation.
struct ResourceOpLiftingPass
    : public impl::ResourceOpLiftingPassBase<ResourceOpLiftingPass> {
  void runOnOperation() override;
};

bool IsResource(Value value) {
  return getElementTypeOrSelf(value.getType()).isa<TF::ResourceType>();
}

// Get the type of the data contained in a resource. Returns null if there is
// no single type in the resource.
Type GetResourceSubtype(Value value) {
  auto resource_type =
      getElementTypeOrSelf(value.getType()).dyn_cast<TF::ResourceType>();
  auto subtypes = resource_type.getSubtypes();
  if (subtypes.size() == 1) return subtypes[0];
  return nullptr;
}

// Replaces all `tf.VarIsInitializedOp` in a block with a constant true.
// TODO(b/171039585): Replace this with proper analysis of
// `tf.VarIsInitializedOp` in regards to resource writes and control flow.
void SetAllVarIsInitializedToTrue(Block* block) {
  auto builder = OpBuilder::atBlockBegin(block);
  TF::ConstOp const_true = nullptr;
  for (auto op :
       llvm::make_early_inc_range(block->getOps<TF::VarIsInitializedOp>())) {
    builder.setInsertionPoint(op);
    if (!const_true)
      const_true = builder.create<TF::ConstOp>(
          op.getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get(/*shape=*/{}, builder.getI1Type()), true));

    op.getIsInitialized().replaceAllUsesWith(const_true);
    op.erase();
  }
}

// Performs store-load forwarding. This effectively removes
// 1) Any resource loads after a store to that same resource is done
// 2) Any resource stores except the last one.
// TODO(ycao): Store-load forwarding implemented here is only correct when
// computation is purely sequential (no concurrency). Need to support concurrent
// computation as well.
void ForwardStoreToLoad(Block* block) {
  // resource_handle_to_last_store_op keeps track of the most recent (last)
  // store to each resource. Non-existent entry indicates that a resource has
  // not been stored to yet.
  llvm::SmallDenseMap<Value, TF::AssignVariableOp>
      resource_handle_to_last_store_op;

  // Only iterate through ops directly in the block as we can't handle ops
  // nested deeper in regions.
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    if (auto read_variable_op = dyn_cast<TF::ReadVariableOp>(&op)) {
      Value resource = read_variable_op.getResource();
      auto last_store = resource_handle_to_last_store_op[resource];
      if (!last_store) continue;

      // Use stored value in last_store to replace all uses of current resource
      // load's result, then erase this resource load. Add an intermediate
      // CastOp if the shape of types doesn't exactly match.
      Type read_type = read_variable_op.getValue().getType();
      if (read_type != last_store.getValue().getType()) {
        OpBuilder builder(last_store);
        builder.setInsertionPointAfter(last_store);
        auto cast = builder.create<TF::CastOp>(
            last_store.getLoc(), read_type, last_store.getValue(),
            /*Truncate=*/builder.getBoolAttr(false));
        read_variable_op.getValue().replaceAllUsesWith(cast);
      } else {
        read_variable_op.getValue().replaceAllUsesWith(last_store.getValue());
      }

      read_variable_op.erase();
      continue;
    }

    if (auto assign_variable_op = dyn_cast<TF::AssignVariableOp>(&op)) {
      Value resource = assign_variable_op.getResource();
      auto last_store = resource_handle_to_last_store_op[resource];
      // Previous store ops to same resource can be erased.
      if (last_store) last_store.erase();

      resource_handle_to_last_store_op[resource] = assign_variable_op;
    }
  }
}

//===----------------------------------------------------------------------===//
// RegionResourceHoister
//===----------------------------------------------------------------------===//

// Helper class to hoist resource ops out of regions attached to an op.
class RegionResourceHoister {
 public:
  explicit RegionResourceHoister(Operation* op) : op_(op) {}

  // Analyzes attached regions to record resources read and written.
  LogicalResult Analyze();

  // Returns all resources accessed by the regions attached the op.
  auto& GetResources() { return resources_; }

  // Returns if the given value is a resource that needs lifting.
  bool Contains(Value resource) const {
    return resources_.find(resource) != resources_.end();
  }

  // Drops the given resource from lifting.
  void DropResource(Value resource) {
    resources_.erase(resource);
    written_resources_.remove(resource);
  }

  // Replaces all resource loads in all regions attached to the op.
  void ReplaceResourceLoads(bool read_only) {
    llvm::for_each(op_->getRegions(), [&](Region& region) {
      ReplaceResourceLoads(region, read_only);
    });
  }

  static LogicalResult ReplaceOpWithNewOp(Operation* op);

 private:
  // Returns if any resources need lifting.
  bool NeedsLifting() const { return !resources_.empty(); }

  // Returns the number of results generated by the lifted op.
  int GetLiftedNumResults() const { return num_new_results_; }

  // Generates hoisted reads for resources that need them before the op.
  void GenerateHoistedReads();

  // Replaces all resource loads in the given region with hoisted loads. If
  // `read_only` is true, limit this replacement to read only resources.
  void ReplaceResourceLoads(Region& region, bool read_only);

  // Appends final values writte to resources to the region returns for the
  // given set of regions.
  void AppendResourceStoreValueToReturn(RegionRange regions);

  // Performs the final replacement of the op.
  void ReplaceOpWithNewOp();

  // Returns is this resource was written to in any of the regions.
  bool IsWritten(Value resource) const {
    return written_resources_.contains(resource);
  }

  static LogicalResult HoistResourcesOutOfIfCaseCluster(Operation* op);
  static LogicalResult HoistResourcesOutOfWhileRegion(TF::WhileRegionOp op);

  Operation* op_;

  // Per resource information about accesses to that resource.
  struct ResourceInfo {
    // Is this resource read in any of the regions?
    bool is_read;
    // Is this resource written in any of the regions?
    bool is_written;
    // Is this resource written in all of the regions?
    bool is_written_all;
    // The hoisted read used to replace region reads.
    Value hoisted_read;
    // the type of the data held by the resource.
    Type data_type;
    // For written resources, the result # of the lifted op which will hold the
    // value of the resource. This result will be used to generates writes to
    // the resource after the lifted op.
    int result_index;
    // Attributes on the read operation.
    DictionaryAttr read_attrs;
    // Attributes on the write operation.
    DictionaryAttr write_attrs;

    ResourceInfo()
        : is_read(false),
          is_written(false),
          is_written_all(false),
          hoisted_read(nullptr),
          data_type(nullptr),
          result_index(-1) {}

    bool IsResultIndexAssigned() { return result_index != -1; }

    // Refine the resource type using the given type `type`.
    void RefineType(Type type) {
      if (!data_type) {
        data_type = type;
      } else {
        data_type = TF::GetCastCompatibleType(data_type, type,
                                              /*may_ignore_ref_type_a=*/false);
        assert(data_type != nullptr && "Resource used with incompatible types");
      }
    }
  };
  llvm::MapVector<Value, ResourceInfo> resources_;
  llvm::SetVector<Value> written_resources_;
  // number of new results after lifting.
  int num_new_results_;
};

// Analyzes resources that are read or written within attached regions.
LogicalResult RegionResourceHoister::Analyze() {
  // Hoisting of child regions might have created opportunity for store-load
  // forwarding.
  for (Region& region : op_->getRegions()) {
    ForwardStoreToLoad(&region.front());
  }

  llvm::SetVector<Value> all_resources;
  bool is_func = false;
  // For functions, the resources to analyze are the function arguments.
  // Otherwise, its the region captures.
  if (func::FuncOp func = dyn_cast<func::FuncOp>(op_)) {
    is_func = true;
    Region& body = func.getBody();
    for (BlockArgument arg : body.getArguments()) {
      if (IsResource(arg)) all_resources.insert(arg);
    }
  } else {
    getUsedValuesDefinedAbove(op_->getRegions(), all_resources);
    all_resources.remove_if([](Value value) { return !IsResource(value); });
  }

  num_new_results_ = op_->getNumResults();

  for (auto resource : all_resources) {
    ResourceInfo info;
    info.data_type = GetResourceSubtype(resource);
    llvm::BitVector written_regions(op_->getNumRegions());
    bool unsupported_use = false;
    for (OpOperand& use : resource.getUses()) {
      Operation* user = use.getOwner();
      // If the user is not in one of the regions, we are not interested in it.
      // Since all the sub-regions within this region (i.e., regions attached to
      // op's in this region) have themselves gone through lifting, all resource
      // users are expected to be operations in this region and not embedded
      // within other sub-regions attached to op's in this region. So the check
      // for whether a user is in one of the regions attached to this op is
      // straightforward.
      if (user->getParentRegion()->getParentOp() != op_) continue;

      // For functions, if the resource is used as a return operand, use that
      // as its result index.
      if (is_func && isa<func::ReturnOp>(user)) {
        assert(!info.IsResultIndexAssigned() &&
               "Expect resource argument to returned no more than once");
        info.result_index = use.getOperandNumber();
        continue;
      }

      auto read = dyn_cast<TF::ReadVariableOp>(user);
      auto write = dyn_cast<TF::AssignVariableOp>(user);
      if (!read && !write) {
        unsupported_use = true;
        break;
      }

      if (read && !info.is_read) {
        info.is_read = true;
        info.RefineType(read.getValue().getType());
        info.read_attrs = user->getAttrDictionary();
      }

      if (write) {
        info.is_written = true;
        info.RefineType(write.getValue().getType());
        info.write_attrs = user->getAttrDictionary();
        written_regions.set(user->getParentRegion()->getRegionNumber());
      }
    }

    // If the resource is used in an op that we do not understand, skip
    // lifting for that resource.
    if (unsupported_use) continue;

    info.is_written_all = written_regions.count() == op_->getNumRegions();

    // If the resource is written in some but not all regions, we would need
    // a read for the value before these regions. Note that this is applicable
    // only to multi-region ops:
    // If/Case: If not all regions write to the resource, post hoisting the read
    //   value need to be routed through all paths that don't write.
    // While: since while condition cannot write, any resource written in the
    //   while body will need to be read as well in case the while body is never
    //   executed.
    // Both cases are handled by the condition below.
    if (info.is_written && !info.is_written_all) info.is_read = true;

    // Allocate a result index for written resources that don't have one.
    if (info.is_written) {
      written_resources_.insert(resource);
      if (!info.IsResultIndexAssigned()) info.result_index = num_new_results_++;
    }

    resources_.insert({resource, info});
  }
  return success();
}

// Generates hoisted reads for all resources that need them just before the op.
void RegionResourceHoister::GenerateHoistedReads() {
  OpBuilder builder(op_);
  DictionaryAttr empty_attrs = builder.getDictionaryAttr({});
  for (auto& resource_it : GetResources()) {
    Value resource = resource_it.first;
    auto& info = resource_it.second;

    if (info.is_read) {
      Operation* read = builder.create<TF::ReadVariableOp>(
          op_->getLoc(), info.data_type, resource);
      read->setAttrs(info.read_attrs ? info.read_attrs : empty_attrs);
      read->removeAttr(kDeviceAttr);
      info.hoisted_read = read->getResult(0);
    }
  }
}

// Replaces all resource reads with the hoisted read.
void RegionResourceHoister::ReplaceResourceLoads(Region& region,
                                                 bool read_only) {
  assert(llvm::hasSingleElement(region) && "Expected single block region");
  // Only iterate through ops directly in the body as we can't handle
  // ops nested deeper in regions.
  auto all_reads = region.front().getOps<TF::ReadVariableOp>();
  for (auto read_op : llvm::make_early_inc_range(all_reads)) {
    Value resource = read_op.getResource();
    if (!Contains(resource)) continue;

    ResourceInfo& info = resources_[resource];
    // If replacing loads for read only resources, skip if the resource
    // was written to.
    if (read_only && info.is_written) continue;

    read_op.replaceAllUsesWith(info.hoisted_read);
    read_op.erase();
  }
}

// For written resources, add its value at the end of each region to that
// regions return value. For a region, its value at the end may be a value
// written to that resource in that region, or its hoisted read value if the
// resource is not written in that region. The return value can be vended out
// either as an existing return value, or a newly allocated return value.
void RegionResourceHoister::AppendResourceStoreValueToReturn(
    RegionRange regions) {
  for (Region* region : regions) {
    assert(llvm::hasSingleElement(*region) && "Expected single block region");
    Block& front = region->front();
    auto old_return = front.getTerminator();
    assert(old_return->getNumOperands() == op_->getNumResults());
    auto new_return_operands = llvm::to_vector<4>(old_return->getOperands());
    new_return_operands.resize(num_new_results_);

    // initialize return values for written resources to be the hoisted reads.
    for (Value resource : written_resources_) {
      const ResourceInfo& info = resources_[resource];
      new_return_operands[info.result_index] = info.hoisted_read;
    }

    // Only iterate through ops directly in the body as op's embedded in child
    // regions should have been lifted out.
    auto assign_ops = front.getOps<TF::AssignVariableOp>();
    for (auto assign_variable_op : llvm::make_early_inc_range(assign_ops)) {
      Value resource = assign_variable_op.getResource();
      if (!IsWritten(resource)) continue;

      // TODO(ycao): Prevent same value from being returned multiple times.
      // TODO(ycao): Do not return resource store value if it is defined outside
      // of cluster. Both of these can be post-resource-op-lifting cleanup
      // passes.
      int result_index = resources_[resource].result_index;
      new_return_operands[result_index] = assign_variable_op.getValue();
      assign_variable_op.erase();
    }
    old_return->setOperands(new_return_operands);
  }
}

// Replace the old op with a new op (with potentially additional results), and
// add stores to written resources after the new op.
void RegionResourceHoister::ReplaceOpWithNewOp() {
  auto new_result_types = llvm::to_vector<4>(op_->getResultTypes());
  int result_region = isa<TF::WhileRegionOp>(op_) ? 1 : 0;
  Operation* terminator = op_->getRegion(result_region).front().getTerminator();
  auto extra_result_types =
      terminator->getOperands().drop_front(op_->getNumResults()).getTypes();
  new_result_types.insert(new_result_types.end(), extra_result_types.begin(),
                          extra_result_types.end());
  OpBuilder builder(op_);
  // Clone this old operation but with new result types.
  Operation* new_op = Operation::create(
      op_->getLoc(), op_->getName(), new_result_types, op_->getOperands(),
      op_->getAttrs(), op_->getSuccessors(), op_->getNumRegions());
  builder.insert(new_op);

  // Move regions to the new op.
  for (auto it : llvm::zip(op_->getRegions(), new_op->getRegions())) {
    Region& old_region = std::get<0>(it);
    Region& new_region = std::get<1>(it);
    new_region.takeBody(old_region);
  }

  // Insert stores to all written resources.
  for (Value resource : written_resources_) {
    ResourceInfo& info = resources_[resource];
    Value value_to_write = new_op->getResult(info.result_index);
    Operation* write = builder.create<TF::AssignVariableOp>(
        op_->getLoc(), resource, value_to_write);
    write->setAttrs(info.write_attrs);
    write->removeAttr(kDeviceAttr);
  }

  // As a part of lifting, we either reuse an existing slot for resource type
  // results or add a new slot. Resource type results should not have any uses
  // to begin with. So we can safely replace each old op result with the
  // corresponding new op result.
  int old_num_results = op_->getNumResults();
  op_->replaceAllUsesWith(new_op->getResults().take_front(old_num_results));
  op_->erase();
  op_ = nullptr;
}

// Lift resource load and stores out of regions attached to `op`, where op is
// an If/case/cluster op.
LogicalResult RegionResourceHoister::HoistResourcesOutOfIfCaseCluster(
    Operation* op) {
  RegionResourceHoister hoister(op);
  if (failed(hoister.Analyze())) return failure();

  // If there are no resource region captures, then nothing to do.
  if (!hoister.NeedsLifting()) return success();

  // Start the transformation. For each region, replace the resource read with
  // the value read before the op.
  hoister.GenerateHoistedReads();
  hoister.ReplaceResourceLoads(/*read_only=*/false);
  hoister.AppendResourceStoreValueToReturn(op->getRegions());
  hoister.ReplaceOpWithNewOp();
  return success();
}

// Lift resource loads and stores out of WhileRegion
LogicalResult RegionResourceHoister::HoistResourcesOutOfWhileRegion(
    TF::WhileRegionOp op) {
  // For WhileRegion, post canonicalization all resource used within the
  // body and condition regions are replaced with captured values, so we do not
  // need to take into account the body and condition region arguments.
  RegionResourceHoister hoister(op);

  if (failed(hoister.Analyze())) return failure();

  // If there are no resource region captures, then nothing to do.
  if (!hoister.NeedsLifting()) return success();

  // The resources captured for While loop fall into two categories:
  // (a) read-only. These reads can be replaced by a hoisted read created
  //        before the WhileOp (similar to if and case).
  // (b) written: since the value is written in the loop (which can only in
  //        loop body, all these will become loop variables. Since all resource
  //        variables are removed from the loop variabled during
  //        canonicalizationW, we need to create new operand/result slots. The
  //        input operands for these slots are the read values
  //        prior to the op, and all references to these are replaced by the
  //        corresponding slot argument. We need to generate writes following
  //        the while for these resources.
  //
  // Note that for WhileRegion ops, if a resource is written, it will be written
  // only in the body and not the condition, so the hoister analysis will infer
  // it as needing a read as well.

  // Generate hoisted reads before the while.
  hoister.GenerateHoistedReads();

  // Replace just the read-only resources with the hoisted reads.
  hoister.ReplaceResourceLoads(/*read_only=*/true);

  // For written resources, add additional operands to the while op.
  int num_old_results = op.getNumResults();
  int num_new_results = hoister.GetLiftedNumResults();
  int num_extra_results = num_new_results - num_old_results;

  SmallVector<Type, 4> new_result_types;
  SmallVector<Value, 4> new_while_operands;
  new_result_types.resize(num_extra_results);
  new_while_operands.resize(num_extra_results);

  for (auto& it : hoister.GetResources()) {
    if (!it.second.is_written) continue;
    int index = it.second.result_index - num_old_results;
    new_result_types[index] = it.second.data_type;
    new_while_operands[index] = it.second.hoisted_read;
  }
  op.getOperation()->insertOperands(op.getNumOperands(), new_while_operands);

  // Patch the cond and body regions to have additional arguments, and replace
  // the remaining resource reads (which will be resource reads for written
  // resources) with these arguments.
  Location loc = op.getLoc();
  for (Region* region : op.getRegions()) {
    region->addArguments(new_result_types,
                         SmallVector<Location>(new_result_types.size(), loc));
    // Point hoisted read for written resources to the region's arguments.
    for (auto& it : hoister.GetResources()) {
      if (!it.second.is_written) continue;
      it.second.hoisted_read = region->getArgument(it.second.result_index);
    }
    hoister.ReplaceResourceLoads(*region, /*read_only=*/false);
  }

  // Add additional return values to body return. These correspond to values
  // written to resources in the body region.
  hoister.AppendResourceStoreValueToReturn(op.getRegions().drop_front());

  // Finally, create a new while with additional return values.
  hoister.ReplaceOpWithNewOp();
  return success();
}

// Lift resources out of the regions attached to `op`
LogicalResult RegionResourceHoister::ReplaceOpWithNewOp(Operation* op) {
  if (auto while_op = dyn_cast<TF::WhileRegionOp>(op))
    return HoistResourcesOutOfWhileRegion(while_op);
  return HoistResourcesOutOfIfCaseCluster(op);
}

// Holds information about a function's use of a resource argument.
struct ResourceArgUseInfo {
  // Data type of the data contained in the resource.
  Type data_type;
  // Is the resource argument used in an assign op?
  bool updated;
  // Is the resource argument used in a read or assign op?
  bool used;
};

// Finds the ResourceArgUseInfo for each resource argument. Forwarding to the
// output (i.e., the argument is an operand of the return op) is not considered
// as a use. This doesn't support nesting of ops, so before calling this, nested
// ops/functions need to be already resource-lifted.
LogicalResult FindResourceArgUseInfo(
    func::FuncOp func_op,
    llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>* result) {
  auto return_op = func_op.front().getTerminator();
  for (auto arg : TF::filter_resources(func_op.getArguments())) {
    ResourceArgUseInfo info;
    info.used = false;
    info.updated = false;
    bool read_or_assigned = false;
    bool used_in_unsupported_op = false;
    for (auto user : arg.getUsers()) {
      if (user == return_op) continue;
      info.used = true;
      if (auto read = llvm::dyn_cast<TF::ReadVariableOp>(user)) {
        read_or_assigned = true;
        info.data_type = read.getType();
        continue;
      }

      if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(user)) {
        read_or_assigned = true;
        info.updated = true;
        info.data_type = assign.getValue().getType();
        continue;
      }

      used_in_unsupported_op = true;
      break;
    }

    // If the arg is used in an unsupported op, skip lifting it.
    if (used_in_unsupported_op) continue;
    (*result)[arg.getArgNumber()] = info;
  }
  return success();
}

// Merges two sets of resource arg use infos. An argument is considered used in
// the merged result as long as either set marks it as used. This is used to
// merge results from functions that have aliasing inputs, e.g., a while loop's
// body and condition. The sets of keys of the two maps must be the same.
llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> MergeArgResourceUseInfo(
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& infos0,
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& infos1) {
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> result;
  for (const auto& entry : infos0) {
    auto info1_it = infos1.find(entry.getFirst());
    // If the entry is missing in any input, we should not touch this entry.
    if (info1_it == infos1.end()) continue;
    auto& info = result[entry.getFirst()];
    info = entry.getSecond();
    if (info.updated) continue;
    if (info1_it->getSecond().used) {
      info.used = true;
      info.updated = info1_it->getSecond().updated;
      info.data_type = info1_it->getSecond().data_type;
    }
  }
  return result;
}

// Removes the unused resource arguments, and the return values that forward the
// removed arguments. If old_to_new_arg_indices is provided, it will store the
// new argument index that corresponds to each original index (-1 means it is
// removed). If remaining_resource_data_types is provided, it will store the
// data types of the remaining resource arguments, where the indices are after
// removing unused ones.
void RemoveUnusedResourceArgumentsAndForwardedRetvals(
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& infos,
    func::FuncOp func_op,
    llvm::SmallVector<int64_t, 4>* old_to_new_arg_indices = nullptr,
    llvm::SmallDenseMap<int64_t, Type>* remaining_resource_data_types =
        nullptr) {
  // Remove return values forwarded from unused arguments.
  auto return_op = func_op.front().getTerminator();
  auto old_return_vals = llvm::to_vector<8>(return_op->getOperands());
  int64_t skipped_retvals = 0;
  for (auto entry : llvm::enumerate(old_return_vals)) {
    auto return_val = entry.value();
    if (auto arg = return_val.dyn_cast<BlockArgument>()) {
      auto it = infos.find(arg.getArgNumber());
      if (it != infos.end() && !it->getSecond().used) {
        return_op->eraseOperand(entry.index() - skipped_retvals++);
      }
    }
  }
  llvm::BitVector indices_to_erase(func_op.getNumArguments());
  llvm::SmallVector<Type, 4> new_types;
  int64_t skipped_args = 0;
  for (auto arg : func_op.getArguments()) {
    auto it = infos.find(arg.getArgNumber());
    if (it != infos.end() && !it->getSecond().used) {
      indices_to_erase.set(arg.getArgNumber());
      skipped_args++;
      if (old_to_new_arg_indices != nullptr) {
        old_to_new_arg_indices->push_back(-1);
      }
    } else {
      new_types.push_back(arg.getType());
      if (old_to_new_arg_indices != nullptr) {
        old_to_new_arg_indices->push_back(arg.getArgNumber() - skipped_args);
      }
      if (it != infos.end() && remaining_resource_data_types != nullptr) {
        (*remaining_resource_data_types)[arg.getArgNumber() - skipped_args] =
            it->second.data_type;
      }
    }
  }
  func_op.eraseArguments(indices_to_erase);
  func_op.setType(
      FunctionType::get(func_op.getContext(), new_types,
                        llvm::to_vector<4>(return_op->getOperandTypes())));
}

// Lifts reads/writes of resource arguments from func_op and changes its
// signature. resource_data_types is the (index, data type) pair for each
// resource argument. handle_updated_arg_value is a caller-provided function
// that handles the updated value for an resource argument.
LogicalResult LiftArgRetResourcesForFunction(
    func::FuncOp func_op,
    const llvm::SmallDenseMap<int64_t, Type>& resource_data_types,
    llvm::function_ref<void(int64_t, Value)> handle_updated_arg_value) {
  RegionResourceHoister hoister(func_op);
  if (failed(hoister.Analyze())) return failure();

  // Each of these resources could be read or written in the function. If its
  // read, we need to replace the resource arg with a value arg to get the
  // read value. If its written, we need to replace the write with an additional
  // value to be written.

  // Now create read values that will be used to replace each resource that
  // is read in the function body. These read values are just the same argument
  // with type replaced.
  llvm::SmallVector<Value, 4> skipped_args;
  for (auto& it : hoister.GetResources()) {
    BlockArgument arg = it.first.dyn_cast<BlockArgument>();
    assert(arg && "Expect resources for FuncOp to be its arguments");
    auto type_iter = resource_data_types.find(arg.getArgNumber());
    if (type_iter == resource_data_types.end()) {
      // Skip lifting the resource if it's not present in the data type map.
      // This indicates that the resource is not to be lifted because it is used
      // in an unsupported op in some other function.
      skipped_args.push_back(arg);
    } else {
      arg.setType(type_iter->second);
      it.second.hoisted_read = arg;
    }
  }

  // Drop all the args that have to be skipped.
  for (Value arg : skipped_args) hoister.DropResource(arg);

  hoister.ReplaceResourceLoads(/*read_only=*/false);

  // For writes, invoke the callback and then erase the write.
  auto assign_ops = func_op.front().getOps<TF::AssignVariableOp>();
  for (auto assign_variable_op : llvm::make_early_inc_range(assign_ops)) {
    Value resource = assign_variable_op.getResource();
    if (!hoister.Contains(resource)) continue;

    auto arg = resource.dyn_cast<BlockArgument>();
    handle_updated_arg_value(arg.getArgNumber(), assign_variable_op.getValue());
    assign_variable_op.erase();
  }

  func_op.setType(FunctionType::get(
      func_op.getContext(), func_op.front().getArgumentTypes(),
      func_op.front().getTerminator()->getOperandTypes()));

  return success();
}

// Returns a vector filtered from range where the unused elements (specified by
// resource_arg_uses) are removed.
template <typename T, typename Range>
llvm::SmallVector<T, 4> FilterRange(
    Range range,
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& resource_arg_uses) {
  llvm::SmallVector<T, 4> filtered;
  for (auto entry : llvm::enumerate(range)) {
    auto it = resource_arg_uses.find(entry.index());
    if (it == resource_arg_uses.end() || it->getSecond().used)
      filtered.push_back(entry.value());
  }
  return filtered;
}

// Changes the types of the control flow op (e.g., while, if) and adds loads and
// stores around it. arg_data_type_and_updated_output_index maps an operand (to
// be changed) index to its data type and the updated value index in the output
// (-1 means not updated.)
void AddLoadsStoresOutsideControlFlowOp(
    Operation* caller,
    const llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>&
        arg_data_type_and_updated_output_index) {
  OpBuilder builder(caller);
  auto new_operands = llvm::to_vector<8>(caller->getOperands());
  llvm::SmallVector<int64_t, 8> changed_indices;
  // Find the operands to change, and create the loads.
  for (auto& entry : arg_data_type_and_updated_output_index) {
    int64_t index = entry.getFirst();
    Type new_type = entry.getSecond().first;
    int64_t updated_index = entry.getSecond().second;
    auto operand = caller->getOperand(index);
    builder.setInsertionPoint(caller);
    new_operands[index] = builder.create<TF::ReadVariableOp>(
        caller->getLoc(), ArrayRef<Type>{new_type}, ArrayRef<Value>{operand});
    caller->setOperand(index, new_operands[index]);
    if (updated_index < 0) continue;
    builder.setInsertionPointAfter(caller);
    builder.create<TF::AssignVariableOp>(
        caller->getLoc(), ArrayRef<Type>{},
        ArrayRef<Value>{operand, caller->getResult(updated_index)});
  }
}

// Lifts loads/stores from while loop's body and cond functions.
LogicalResult HandleWhileLoop(TF::WhileOp while_op, func::FuncOp body,
                              func::FuncOp cond) {
  auto return_op = body.front().getTerminator();
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> body_use_info;
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> cond_use_info;
  if (failed(FindResourceArgUseInfo(body, &body_use_info)) ||
      failed(FindResourceArgUseInfo(cond, &cond_use_info))) {
    return failure();
  }
  // A resource is considered used as long as it is used in either body or cond.
  auto resource_arg_uses =
      MergeArgResourceUseInfo(body_use_info, cond_use_info);
  if (resource_arg_uses.empty()) return success();

  // Remove unused resources in functions.
  llvm::SmallVector<int64_t, 4> old_to_new_indices;
  llvm::SmallDenseMap<int64_t, Type> remaining_resource_data_types;
  RemoveUnusedResourceArgumentsAndForwardedRetvals(
      resource_arg_uses, body, &old_to_new_indices,
      &remaining_resource_data_types);
  RemoveUnusedResourceArgumentsAndForwardedRetvals(resource_arg_uses, cond);
  (void)LiftArgRetResourcesForFunction(
      body, remaining_resource_data_types,
      [&](int64_t index, Value value) { return_op->setOperand(index, value); });
  (void)LiftArgRetResourcesForFunction(cond, remaining_resource_data_types,
                                       [&](int64_t index, Value value) {
                                         // We already checked that cond should
                                         // not have variable writes.
                                         assert(false && "Should not happen");
                                       });
  // Recreate the while op.
  OpBuilder builder(while_op);
  // Now use the filtered original operands, which will be replaced by
  // AddLoadsStoresOutsideControlFlowOp().
  auto new_while = builder.create<TF::WhileOp>(
      while_op.getLoc(), body.getFunctionType().getResults(),
      FilterRange<Value, OperandRange>(while_op.getOperands(),
                                       resource_arg_uses),
      while_op->getAttrs());
  // Prepare for AddLoadsStoresOutsideControlFlowOp().
  llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>
      arg_data_type_and_updated_output_index;
  for (const auto& entry : remaining_resource_data_types) {
    int64_t update_index = return_op->getOperand(entry.getFirst()) ==
                                   body.getArgument(entry.getFirst())
                               ? -1
                               : entry.getFirst();
    arg_data_type_and_updated_output_index[entry.getFirst()] = {
        entry.getSecond(), update_index};
  }
  AddLoadsStoresOutsideControlFlowOp(new_while,
                                     arg_data_type_and_updated_output_index);
  // Replace uses.
  for (int64_t i = 0, end = old_to_new_indices.size(); i < end; ++i) {
    if (old_to_new_indices[i] >= 0) {
      while_op.getResult(i).replaceAllUsesWith(
          new_while.getResult(old_to_new_indices[i]));
    }
  }
  while_op.erase();
  return success();
}

// Lifts loads/stores from an IfOp or CaseOp's branches.
template <class CaseOrIfOp>
LogicalResult HandleCaseOrIfOp(CaseOrIfOp op, ArrayRef<func::FuncOp> branches) {
  // For canonicalized If/Case, there should not be any resource outputs
  int64_t non_resource_results = op.getNumResults();

  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> resource_arg_uses;
  if (failed(FindResourceArgUseInfo(branches.front(), &resource_arg_uses)))
    return failure();

  for (auto func : branches.drop_front()) {
    llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> branch_use_info;
    if (failed(FindResourceArgUseInfo(func, &branch_use_info)))
      return failure();
    // A resource is considered used as long as it is used in either branch.
    resource_arg_uses =
        MergeArgResourceUseInfo(resource_arg_uses, branch_use_info);
  }

  if (resource_arg_uses.empty()) return success();
  // Remove unused resources in functions.
  llvm::SmallDenseMap<int64_t, Type> remaining_resource_data_types;
  RemoveUnusedResourceArgumentsAndForwardedRetvals(
      resource_arg_uses, branches.front(), /*old_to_new_arg_indices=*/nullptr,
      &remaining_resource_data_types);
  for (auto func : branches.drop_front())
    RemoveUnusedResourceArgumentsAndForwardedRetvals(resource_arg_uses, func);

  // Forward resource inputs updated in any branch to the outputs of both
  // branches. First prepare the mapping from arg to new update output.
  llvm::SmallDenseMap<int64_t, int64_t> resource_arg_to_new_output;
  {
    int64_t removed_args = 0;
    for (const auto& entry : resource_arg_uses) {
      if (!entry.getSecond().used) {
        removed_args++;
        continue;
      }
      if (!entry.getSecond().updated) continue;
      int64_t new_output_index =
          non_resource_results + resource_arg_to_new_output.size();
      resource_arg_to_new_output[entry.getFirst() - removed_args] =
          new_output_index;
    }
  }

  // Append resource updates to the return ops: now they are just forwarded
  // input resources, but will be replaced by the data value in
  // LiftArgRetResourcesForFunction().
  for (auto branch : branches) {
    auto new_retvals =
        llvm::to_vector<4>(branch.front().getTerminator()->getOperands());
    new_retvals.resize(new_retvals.size() + resource_arg_to_new_output.size());
    for (const auto& entry : resource_arg_to_new_output) {
      int64_t resource_arg_index = entry.getFirst();
      int64_t output_index = entry.getSecond();
      new_retvals[output_index] = branch.getArgument(resource_arg_index);
    }
    auto old_return = branch.front().getTerminator();
    OpBuilder builder(old_return);
    auto new_return =
        builder.create<func::ReturnOp>(old_return->getLoc(), new_retvals);
    old_return->erase();
    (void)LiftArgRetResourcesForFunction(
        branch, remaining_resource_data_types, [&](int64_t index, Value value) {
          new_return.setOperand(resource_arg_to_new_output[index], value);
        });
  }

  // Recreate the op without resource operands.
  OpBuilder builder(op);
  // Now use the filtered original operands, which will be replaced by
  // AddLoadsStoresOutsideControlFlowOp().
  auto new_operands =
      FilterRange<Value, OperandRange>(op.getInput(), resource_arg_uses);
  new_operands.insert(new_operands.begin(), op.getOperand(0));
  func::FuncOp first_func = branches.front();
  auto new_op = builder.create<CaseOrIfOp>(
      op.getLoc(), first_func.getFunctionType().getResults(), new_operands,
      op->getAttrs());
  // Prepare for AddLoadsStoresOutsideControlFlowOp()
  llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>
      arg_data_type_and_updated_output_index;
  for (const auto& entry : remaining_resource_data_types) {
    auto new_output_it = resource_arg_to_new_output.find(entry.getFirst());
    int64_t update_index = new_output_it == resource_arg_to_new_output.end()
                               ? -1
                               : new_output_it->getSecond();
    arg_data_type_and_updated_output_index[entry.getFirst() + 1] = {
        entry.getSecond(), update_index};
  }
  AddLoadsStoresOutsideControlFlowOp(new_op,
                                     arg_data_type_and_updated_output_index);
  // Replace uses.
  op.replaceAllUsesWith(new_op.getResults().take_front(op.getNumResults()));
  op.erase();
  return success();
}

// A resource-lifted function for (potentially multiple) PartitionedCallOps and
// information about the lifting changes.
struct PartitionedCallLiftingInfo {
  // Function with resources lifted. Can be nullptr if nothing needs to change.
  func::FuncOp lifted_callee;
  // Mapping from old resource outputs to their aliasing output inputs.
  llvm::SmallDenseMap<int64_t, int64_t> old_outputs_aliasing_old_inputs;
  // Mapping from old to new output indices in case any output is removed.
  llvm::SmallVector<int64_t, 4> old_to_new_output_indices;
  // ResourceArgUseInfo for each old resource argument.
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> use_info;
  // Input for AddLoadsStoresOutsideControlFlowOp(), see its comment.
  llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>
      arg_data_type_and_updated_output_index;
};

// Lifts loads/stores from a PartitionedCallOp's callee function. If anything
// needs to be changed, the original function will be preserved, and the lifting
// happens on a clone, which will be stored in `result`.
LogicalResult HandlePartitionedCallOpCallee(
    func::FuncOp callee, PartitionedCallLiftingInfo* result) {
  // Sanity check: return of resources should be aliases of inputs. Such outputs
  // will be removed later.
  int64_t non_resource_results = 0;
  for (auto entry :
       llvm::enumerate(callee.front().getTerminator()->getOperands())) {
    auto retval = entry.value();
    if (!getElementTypeOrSelf(retval.getType()).isa<TF::ResourceType>()) {
      result->old_to_new_output_indices.push_back(non_resource_results++);
      continue;
    }
    auto aliasing_arg = retval.dyn_cast<BlockArgument>();
    if (!aliasing_arg) {
      return callee.emitOpError("unsupported function call: ")
             << "resource return value does not alias an input.";
    }
    result->old_outputs_aliasing_old_inputs[entry.index()] =
        aliasing_arg.getArgNumber();
    result->old_to_new_output_indices.push_back(-1);
  }

  if (failed(FindResourceArgUseInfo(callee, &result->use_info))) {
    return failure();
  }
  if (result->use_info.empty()) {
    result->lifted_callee = nullptr;
    return success();
  }

  // Clone the callee before making changes.
  SmallString<64> name_base = callee.getName();
  auto module = callee->getParentOfType<ModuleOp>();
  name_base += "_resource_lifted";
  auto name = name_base;
  callee = callee.clone();
  callee.setPrivate();
  callee.setName(mlir::StringAttr::get(callee->getContext(), name));
  SymbolTable(module).insert(callee);
  result->lifted_callee = callee;

  // Remove unused resources in functions.
  llvm::SmallDenseMap<int64_t, Type> remaining_resource_data_types;
  RemoveUnusedResourceArgumentsAndForwardedRetvals(
      result->use_info, callee, /*old_to_new_arg_indices=*/nullptr,
      &remaining_resource_data_types);
  for (const auto& entry : remaining_resource_data_types) {
    result->arg_data_type_and_updated_output_index[entry.getFirst()] = {
        entry.getSecond(), -1};
  }
  llvm::SmallVector<int64_t, 4> retval_indices_to_preserve;
  for (auto& val : callee.front().getTerminator()->getOpOperands()) {
    // Store indices of results that are not resources.
    if (!getElementTypeOrSelf(val.get().getType()).isa<TF::ResourceType>())
      retval_indices_to_preserve.push_back(val.getOperandNumber());
  }
  int64_t num_retvals = retval_indices_to_preserve.size();
  llvm::SmallVector<Value, 4> new_retvals;
  // Lift resources.
  (void)LiftArgRetResourcesForFunction(
      callee, remaining_resource_data_types, [&](int64_t index, Value value) {
        result->arg_data_type_and_updated_output_index[index].second =
            num_retvals++;
        new_retvals.push_back(value);
      });

  auto old_return = callee.front().getTerminator();
  llvm::SmallVector<Value, 4> old_and_new_retvals;
  old_and_new_retvals.reserve(retval_indices_to_preserve.size() +
                              new_retvals.size());
  for (int64_t retval_index : retval_indices_to_preserve)
    old_and_new_retvals.push_back(old_return->getOperand(retval_index));

  old_and_new_retvals.append(new_retvals.begin(), new_retvals.end());
  // Replace old return with the new ones with update values.
  OpBuilder builder(old_return);
  auto new_return =
      builder.create<func::ReturnOp>(old_return->getLoc(), old_and_new_retvals);
  old_return->erase();
  callee.setType(FunctionType::get(
      callee.getContext(), callee.getFunctionType().getInputs(),
      llvm::to_vector<4>(new_return.getOperandTypes())));
  return success();
}

// Updates a PartitionedCallOp/StatefulPartitionedCallOp according to the
// resource-lifted new callee function in lifting_info.
template <typename CallOpType>
void UpdatePartitionedCallOpWithNewCallee(
    CallOpType call_op, PartitionedCallLiftingInfo& lifting_info) {
  if (!lifting_info.lifted_callee) return;
  // Replace output resource uses with the aliasing input, so that we can remove
  // this output.
  for (const auto& entry : lifting_info.old_outputs_aliasing_old_inputs) {
    call_op.getResult(entry.getFirst())
        .replaceAllUsesWith(call_op.getOperand(entry.getSecond()));
  }
  // Recreate the call op.
  OpBuilder builder(call_op);
  // Now use the filtered original operands, which will be replaced by
  // AddLoadsStoresOutsideControlFlowOp().
  auto new_operands = FilterRange<Value, OperandRange>(call_op.getArgs(),
                                                       lifting_info.use_info);
  auto new_call = builder.create<CallOpType>(
      call_op.getLoc(),
      lifting_info.lifted_callee.getFunctionType().getResults(), new_operands,
      call_op->getAttrs());
  new_call->setAttr("f",
                    SymbolRefAttr::get(builder.getContext(),
                                       lifting_info.lifted_callee.getName()));
  AddLoadsStoresOutsideControlFlowOp(
      new_call, lifting_info.arg_data_type_and_updated_output_index);
  // Replace uses.
  for (int64_t i = 0, end = lifting_info.old_to_new_output_indices.size();
       i < end; ++i) {
    if (lifting_info.old_to_new_output_indices[i] >= 0) {
      call_op.getResult(i).replaceAllUsesWith(
          new_call.getResult(lifting_info.old_to_new_output_indices[i]));
    }
  }
  call_op.erase();
}

LogicalResult HoistForControlFlow(
    Block*, ModuleOp, bool,
    llvm::SmallDenseMap<llvm::StringRef, PartitionedCallLiftingInfo>*);

// A templated routine for handling both PartitionedCallOp and
// StatefulPartitionedCallOp. If the callee is already lifted, it just updates
// the caller op itself; otherwise, it first recursively handles nested control
// flow, then performs lifting on the callee.
template <typename CallOpType>
LogicalResult HandlePartitionedCallOp(
    CallOpType call_op, func::FuncOp callee, ModuleOp module,
    bool vars_initialized,
    llvm::SmallDenseMap<llvm::StringRef, PartitionedCallLiftingInfo>*
        lifted_callees) {
  auto emplace_res = lifted_callees->try_emplace(callee.getName(),
                                                 PartitionedCallLiftingInfo());
  if (emplace_res.second) {
    // Unseen callee. Perform resource lifting on it.
    if (failed(HoistForControlFlow(&callee.front(), module, vars_initialized,
                                   lifted_callees)))
      return failure();

    if (failed(HandlePartitionedCallOpCallee(
            callee, &emplace_res.first->getSecond()))) {
      return failure();
    }
  }
  UpdatePartitionedCallOpWithNewCallee(call_op, emplace_res.first->getSecond());
  return success();
}

// Hoists resource loads/stores from control flow ops in `block` outside the
// body/cond/branch/callee functions.
LogicalResult HoistForControlFlow(
    Block* block, ModuleOp module, bool vars_initialized,
    llvm::SmallDenseMap<llvm::StringRef, PartitionedCallLiftingInfo>*
        lifted_partitioned_call_callees) {
  if (vars_initialized) SetAllVarIsInitializedToTrue(block);

  for (Operation& op : llvm::make_early_inc_range(*block)) {
    if (auto while_op = llvm::dyn_cast<TF::WhileOp>(&op)) {
      auto body = while_op.body_function();
      auto cond = while_op.cond_function();
      // Recursively handle the nested control flow.
      (void)HoistForControlFlow(&body.front(), module, vars_initialized,
                                lifted_partitioned_call_callees);
      (void)HoistForControlFlow(&cond.front(), module, vars_initialized,
                                lifted_partitioned_call_callees);
      if (failed(HandleWhileLoop(while_op, body, cond))) return failure();
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(&op)) {
      auto then_branch = if_op.then_function();
      auto else_branch = if_op.else_function();
      // Recursively handle the nested control flow.
      (void)HoistForControlFlow(&then_branch.front(), module, vars_initialized,
                                lifted_partitioned_call_callees);
      (void)HoistForControlFlow(&else_branch.front(), module, vars_initialized,
                                lifted_partitioned_call_callees);
      if (failed(HandleCaseOrIfOp(if_op, {then_branch, else_branch})))
        return failure();
    } else if (auto case_op = llvm::dyn_cast<TF::CaseOp>(&op)) {
      SmallVector<func::FuncOp, 4> branch_functions;
      case_op.get_branch_functions(branch_functions);
      for (func::FuncOp func : branch_functions) {
        // Recursively handle the nested control flow.
        (void)HoistForControlFlow(&func.front(), module, vars_initialized,
                                  lifted_partitioned_call_callees);
      }
      if (failed(HandleCaseOrIfOp(case_op, branch_functions))) return failure();
    } else if (auto call_op = llvm::dyn_cast<TF::PartitionedCallOp>(&op)) {
      auto callee = call_op.func();
      if (!callee) {
        return call_op.emitOpError(
            "resource lifting does not support call with nested references.");
      }
      if (failed(HandlePartitionedCallOp(call_op, callee, module,
                                         vars_initialized,
                                         lifted_partitioned_call_callees))) {
        // Nested control flow handling is done in HandlePartitionedCallOp().
        return failure();
      }
    } else if (auto call_op =
                   llvm::dyn_cast<TF::StatefulPartitionedCallOp>(&op)) {
      if (failed(HandlePartitionedCallOp(call_op, call_op.func(), module,
                                         vars_initialized,
                                         lifted_partitioned_call_callees))) {
        return failure();
      }
    } else if (isa<TF::IfRegionOp, TF::CaseRegionOp, TF::WhileRegionOp>(op)) {
      for (Region& region : op.getRegions())
        (void)HoistForControlFlow(&region.front(), module, vars_initialized,
                                  lifted_partitioned_call_callees);
      LogicalResult result = RegionResourceHoister::ReplaceOpWithNewOp(&op);
      if (failed(result)) return failure();
    }
  }

  // After we have hoisted operations in the block, we may have added new read
  // and writes of resources to this block. Clean them up by doing store-load
  // forwarding.
  ForwardStoreToLoad(block);
  return success();
}

// Lifts resource operation from tf_device.cluster ops nested in `op` outside.
// Returns failure if there are remaining resource-type values that can not be
// lifted.
void ResourceOpLiftingPass::runOnOperation() {
  llvm::SmallDenseMap<llvm::StringRef, PartitionedCallLiftingInfo>
      lifted_partitioned_call_callees;
  ModuleOp module = getOperation();

  if (failed(TF::CleanupAndCanonicalizeForResourceOpLifting(module)))
    return signalPassFailure();

  auto walk_result = module.walk([&](func::FuncOp func_op) {
    return func_op.walk([&](tf_device::ClusterOp cluster) {
      LogicalResult result = HoistForControlFlow(
          &cluster.GetBody(), module, /*vars_initialized=*/true,
          &lifted_partitioned_call_callees);
      if (failed(result)) return WalkResult::interrupt();
      result = RegionResourceHoister::ReplaceOpWithNewOp(cluster);
      if (failed(result)) return WalkResult::interrupt();
      return WalkResult::advance();
    });
  });

  if (walk_result.wasInterrupted()) return signalPassFailure();
}

#define GEN_PASS_DEF_RESOURCEOPLIFTINGFORMAINFUNCTIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

struct ResourceOpLiftingForMainFunctionPass
    : public impl::ResourceOpLiftingForMainFunctionPassBase<
          ResourceOpLiftingForMainFunctionPass> {
  void runOnOperation() override;
};

void ResourceOpLiftingForMainFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp main_func = module.lookupSymbol<func::FuncOp>("main");
  if (!main_func) {
    return;
  }

  if (failed(TF::ResourceLiftingForFunctionalControlFlow(main_func))) {
    return signalPassFailure();
  }
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceOpLiftingPass() {
  return std::make_unique<ResourceOpLiftingPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateResourceOpLiftingForMainFunctionPass() {
  return std::make_unique<ResourceOpLiftingForMainFunctionPass>();
}

}  // namespace TFDevice

namespace TF {
LogicalResult ResourceLiftingForFunctionalControlFlow(func::FuncOp function) {
  // This routine should only be called when control flow operations are still
  // represented with TF IfOp and WhileOp operations. In this case, there should
  // be only one basic blocks in the MLIR representation.
  if (!llvm::hasSingleElement(function)) {
    return function.emitError()
           << "expect the function to have 1 block while it has "
           << function.getBlocks().size();
  }

  if (failed(TF::CleanupAndCanonicalizeForResourceOpLifting(function)))
    return failure();

  llvm::SmallDenseMap<llvm::StringRef, PartitionedCallLiftingInfo>
      lifted_partitioned_call_callees;
  if (failed(HoistForControlFlow(
          &function.front(), cast<ModuleOp>(function->getParentOp()),
          /*vars_initialized=*/false, &lifted_partitioned_call_callees)))
    return failure();

  // Clean up and canonicalize to remove dead local variables as some local
  // variables might be dead after hoisting resource loads/stores from control
  // flow ops.
  return TF::CleanupAndCanonicalizeForResourceOpLifting(function);
}
}  // namespace TF

}  // namespace mlir
