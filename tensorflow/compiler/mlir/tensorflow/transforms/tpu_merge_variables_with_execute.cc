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

#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "tf-tpu-merge-variables-with-execute"

namespace mlir {
namespace TFTPU {

namespace {
constexpr char kAliasingAttr[] = "tf.aliasing_output";
constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

#define GEN_PASS_DEF_TPUMERGEVARIABLESWITHEXECUTEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class TPUMergeVariablesWithExecutePass
    : public impl::TPUMergeVariablesWithExecutePassBase<
          TPUMergeVariablesWithExecutePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    // We need this here because at the moment we deserialize the TPUCompileMlir
    // operation which contains annotation like `mhlo.sharding` attributes.
    registry.insert<mhlo::MhloDialect>();
  }
  void runOnOperation() override;
};

// Information for a pair of input/output of the TPUExecute op and the
// surrounding read/assign ops.
struct VariableAccessInfo {
  int execute_input_index = -1;
  int execute_output_index = -1;
  Operation* read = nullptr;
  Operation* assign = nullptr;
};

// Information about all resource accesses to be merged into a TPUExecute op.
struct VariableAccessesForTPUExecute {
  // Maps each detected resource to a VariableAccessInfo. Eventually, this will
  // contain all values for which we want to merge the accessing ops with a
  // TPUExecute op.
  llvm::SmallDenseMap<Value, VariableAccessInfo, 8> per_resource_info;
  // The corresponding new output index in TPUExecuteAndUpdateVariables for
  // each old output index in TPUExecute.
  llvm::SmallVector<int, 8> old_to_new_output_mapping;
  // The resources read by ReadVariableOps that are inputs to TPUExecute,
  // ordered by the input indices to TPUExecute.
  llvm::SmallVector<Value, 8> resources_read;
  // Operands for the new TPUExecuteAndUpdateVariables.
  llvm::SmallVector<Value, 8> new_operand_values;
};

// Returns true iff the read or assign op associated with `resource` can be
// safely merged.
//
// `resource_ids` contains IDs of all previously accessed resources
// `previous_unknown_resource_access` is true if we had any previous unknown
// resource access.
bool IsResourceSafeForMerge(
    Value resource,
    const mlir::TF::ResourceAliasAnalysis::Info& resource_analysis_info,
    const VariableAccessesForTPUExecute& infos,
    const llvm::SmallDenseSet<int64_t>& resource_ids,
    bool previous_unknown_resource_access) {
  // If we had any unknown resource access before, then we conservatively assume
  // that `resource` has been accessed before.
  // If `resource` is an unknown resource, then we conservatively assume that
  // the same resource has been accessed before.
  if (previous_unknown_resource_access ||
      resource_analysis_info.IsUnknownResource(resource))
    return false;
  const auto& ids = resource_analysis_info.GetResourceUniqueIds(resource);
  for (int64_t id : ids) {
    if (resource_ids.contains(id)) return false;
  }
  return true;
}

// Adds IDs of resources which `op` accesses to `resource_ids`.
// Returns true iff op accesses a resource unknown to `resource_analysis_info`
// in which case we have to conservatively assume that any resource might be
// accessed.
bool AddAccessedResourceIds(
    Operation* op,
    const mlir::TF::ResourceAliasAnalysis::Info& resource_analysis_info,
    llvm::SmallDenseSet<int64_t>& resource_ids) {
  for (Value operand : TF::filter_resources(op->getOperands())) {
    if (resource_analysis_info.IsUnknownResource(operand)) {
      VLOG(2) << "  unknown access";
      return true;
    }
    const auto& ids = resource_analysis_info.GetResourceUniqueIds(operand);
    VLOG(2) << "  accesses following resources: " << absl::StrJoin(ids, ", ");
    resource_ids.insert(ids.begin(), ids.end());
  }
  return false;
}

/* Resources may be merged with an execute op when they are on its device or a
 * `COMPOSITE`. Note that a `COMPOSITE` represents a set of devices, they
 * are typically associated with packed variables. Presently, we assume this
 * set spans all the devices. So, a variable on a `COMPOSITE` will have a local
 * instance on the execute op's device.
 */
bool IsResourceMergeable(Attribute& resource_attr, Attribute& device_attr) {
  return resource_attr &&
         ((resource_attr == device_attr) ||
          (resource_attr.cast<mlir::StringAttr>().getValue().find(
               "COMPOSITE") != llvm::StringRef::npos));
}

// Finds the variable access info for a TPUExecute op.
//  - `check_device` specifies  whether it checks the device assignment of the
//  variables to match the TPUExecute op. This is optional in some context,
//  e.g., guaranteed by replication.
//  - `check_same_region` specifies whether the reads/assigns need to be in the
//  same region as `execute`. This is needed if `execute` is inside ReplicateOp.
VariableAccessesForTPUExecute BuildVariableAccessInfo(
    tf_device::LaunchOp execute_launch,
    const mlir::TF::ResourceAliasAnalysis::Info& resource_analysis_info,
    bool check_device, bool check_same_region) {
  VariableAccessesForTPUExecute var_access_info;
  Attribute device_attr = execute_launch.getDeviceAttr();
  if (check_device && !device_attr) return var_access_info;
  auto func = execute_launch->getParentOfType<mlir::func::FuncOp>();

  // Track the first read op found, which is used later to check if there are
  // assign ops between it and the TPUExecute op. We will exclude reads before
  // interfering accesses in a conservative way (see below). We do not consider
  // resource accesses in other islands since their ordering is enforced by
  // inter-island dependencies.
  Operation* first_read = nullptr;
  auto execute = cast<TF::TPUExecuteOp>(execute_launch.GetBody().front());
  auto parallel_execute = llvm::dyn_cast<tf_device::ParallelExecuteOp>(
      execute_launch->getParentOp());
  Operation* execute_parent =
      parallel_execute ? parallel_execute.getOperation() : execute_launch;
  // Collect all operands of `execute` whose defining ops are variable reads
  // that might get merged, and add relevant information to `var_access_info`.
  for (auto operand : llvm::enumerate(execute->getOpOperands())) {
    var_access_info.new_operand_values.push_back(operand.value().get());
    auto read_op = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.value().get().getDefiningOp());
    if (!read_op) continue;
    if (check_same_region &&
        read_op->getParentRegion() != execute_parent->getParentRegion())
      continue;

    auto resource = read_op.getResource();
    if (check_device) {
      // TODO(lyandy): Wrap resource ops in tf_device.launch.
      if (auto* resource_op = resource.getDefiningOp()) {
        auto resource_attr = resource_op->getAttr(kDeviceAttr);
        // Check device matching for the node defining the resource.
        if (!IsResourceMergeable(resource_attr, device_attr)) continue;
      } else {
        auto resource_arg = resource.dyn_cast<BlockArgument>();
        assert(resource_arg);
        if (resource_arg.getOwner() != &func.front()) continue;
        // Check device matching for the argument defining the resource.
        auto resource_attr = func.getArgAttrOfType<mlir::StringAttr>(
            resource_arg.getArgNumber(), kFuncDeviceAttr);
        if (!IsResourceMergeable(resource_attr, device_attr)) continue;
      }
    }

    auto emplace_res = var_access_info.per_resource_info.try_emplace(
        resource, VariableAccessInfo());
    if (!emplace_res.second) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping execute that has multiple reads of a variable: "
                 << execute << "\n");
      var_access_info.per_resource_info.shrink_and_clear();
      return var_access_info;
    }

    VLOG(2) << "Adding read op to merge candidates: " << debugString(read_op);
    auto& info = emplace_res.first->getSecond();
    info.execute_input_index = operand.index();
    info.read = read_op;
    var_access_info.new_operand_values[operand.index()] = resource;
    var_access_info.resources_read.push_back(resource);
    if (!first_read || info.read->isBeforeInBlock(first_read)) {
      first_read = info.read;
    }
  }

  if (!first_read) return var_access_info;

  // Walk backwards from `execute_parent` to `first_read` and remove merge
  // candidates based on resource modifications.
  llvm::SmallDenseSet<int64_t> resource_ids;
  bool previous_unknown_resource_access = false;
  for (Operation& op : llvm::reverse(llvm::make_range(
           first_read->getIterator(), execute_parent->getIterator()))) {
    if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(&op)) {
      VLOG(2) << "Processing read op " << debugString(op);
      auto info_it =
          var_access_info.per_resource_info.find(read_op.getResource());
      bool is_merge_candidate =
          info_it != var_access_info.per_resource_info.end();

      if (is_merge_candidate &&
          !IsResourceSafeForMerge(read_op.getResource(), resource_analysis_info,
                                  var_access_info, resource_ids,
                                  previous_unknown_resource_access)) {
        VLOG(2) << "  removing op from merge candidates";
        int input_index = info_it->getSecond().execute_input_index;
        var_access_info.new_operand_values[input_index] =
            execute.getOperand(input_index);
        var_access_info.per_resource_info.erase(info_it);
      }
    }
    previous_unknown_resource_access |=
        AddAccessedResourceIds(&op, resource_analysis_info, resource_ids);
  }

  if (var_access_info.per_resource_info.empty()) {
    return var_access_info;
  }

  // Find outputs that are variable assigns.
  Operation* last_assign = nullptr;
  llvm::SmallPtrSet<Operation*, 8> all_assigns;
  llvm::SmallVector<bool, 8> output_merged(execute_launch.getNumResults(),
                                           false);

  auto execute_outputs =
      parallel_execute
          ? parallel_execute.GetRegionOutputs(
                execute_launch->getParentRegion()->getRegionNumber())
          : execute_launch.getResults();
  for (auto execute_output : llvm::enumerate(execute_outputs)) {
    // TODO(lyandy): Handle updates to resource writes by remapping to parent
    // launch result and checking if launch result is an AssignVariableOp.
    auto result = execute_output.value();
    if (!result.hasOneUse()) continue;

    auto assign_op = llvm::dyn_cast<TF::AssignVariableOp>(*result.user_begin());
    if (!assign_op) continue;
    auto resource = assign_op.getResource();
    auto it = var_access_info.per_resource_info.find(resource);
    if (it == var_access_info.per_resource_info.end()) continue;
    auto& info = it->getSecond();
    if (info.assign) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping execute that has multiple assigns of a variable: "
                 << execute << "\n");
      var_access_info.per_resource_info.shrink_and_clear();
      return var_access_info;
    }
    info.execute_output_index = execute_output.index();
    info.assign = assign_op;
    if (!last_assign || last_assign->isBeforeInBlock(assign_op)) {
      last_assign = assign_op;
    }
    VLOG(2) << "Adding assign op to merge candidates: "
            << debugString(assign_op);
    all_assigns.insert(assign_op);
    output_merged[execute_output.index()] = true;
  }

  if (last_assign != nullptr) {
    // Walk forward from `execute_parent` to `last_assign` and remove merge
    // candidates based on resource modifications.
    resource_ids.clear();
    previous_unknown_resource_access = false;
    for (Operation& op :
         llvm::make_range(std::next(execute_parent->getIterator()),
                          std::next(last_assign->getIterator()))) {
      if (auto assign_op = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
        VLOG(2) << "Processing assign op " << debugString(op);
        bool is_merge_candidate = true;
        if (all_assigns.count(assign_op) == 0) is_merge_candidate = false;
        auto info_it =
            var_access_info.per_resource_info.find(assign_op.getResource());
        if (info_it == var_access_info.per_resource_info.end())
          is_merge_candidate = false;

        if (is_merge_candidate &&
            !IsResourceSafeForMerge(assign_op.getResource(),
                                    resource_analysis_info, var_access_info,
                                    resource_ids,
                                    previous_unknown_resource_access)) {
          VLOG(2) << "  removing op from merge candidates";
          output_merged[info_it->second.execute_output_index] = false;
          info_it->second.execute_output_index = -1;
          info_it->second.assign = nullptr;
        }
      }
      previous_unknown_resource_access |=
          AddAccessedResourceIds(&op, resource_analysis_info, resource_ids);
    }
  }

  // Populate var_access_info.old_to_new_output_mapping.
  int new_output_index = 0;
  var_access_info.old_to_new_output_mapping.resize(
      execute_launch.getNumResults());
  for (int i = 0, end = execute_launch.getNumResults(); i < end; ++i) {
    if (output_merged[i]) {
      var_access_info.old_to_new_output_mapping[i] = -1;
    } else {
      var_access_info.old_to_new_output_mapping[i] = new_output_index;
      ++new_output_index;
    }
  }
  return var_access_info;
}

// Appends result types of tf_device.parallel_execute from `start` index region
// (inclusive) to `end` index region (exclusive) to `output_types` and returns
// the number of types added.
int AppendTypes(llvm::SmallVectorImpl<Type>* output_types,
                tf_device::ParallelExecuteOp parallel_execute, int start,
                int end) {
  const int size_before = output_types->size();
  for (int index = start; index < end; ++index) {
    Block& block = parallel_execute.GetRegionBlockWithIndex(index);
    auto terminator_operand_types = block.getTerminator()->getOperandTypes();
    output_types->append(terminator_operand_types.begin(),
                         terminator_operand_types.end());
  }
  return output_types->size() - size_before;
}

// Replaces TPUExecute with TPUExecuteAndUpdateVariables in a
// tf_device.parallel_execute op.
void ReplaceParallelExecute(
    tf_device::ParallelExecuteOp parallel_execute,
    tf_device::LaunchOp execute_launch,
    tf_device::LaunchOp merged_execute_launch,
    const VariableAccessesForTPUExecute& var_access_info, OpBuilder* builder) {
  Operation* parallel_execute_op = parallel_execute.getOperation();

  // Collect result types of tf_device.parallel_execute and update region
  // result types with the new merged execute result types.
  llvm::SmallVector<Type, 8> output_types;
  const int parallel_execute_num_results = parallel_execute_op->getNumResults();
  output_types.reserve(parallel_execute_num_results);
  Region* execute_region = merged_execute_launch->getParentRegion();
  const int region_index = execute_region->getRegionNumber();
  const int num_results_before_region =
      AppendTypes(&output_types, parallel_execute, 0, region_index);
  // Append updated results from merged execute.
  output_types.append(merged_execute_launch.getResultTypes().begin(),
                      merged_execute_launch.getResultTypes().end());
  const int num_regions = parallel_execute_op->getNumRegions();
  const int num_results_after_region = AppendTypes(
      &output_types, parallel_execute, region_index + 1, num_regions);

  builder->setInsertionPoint(parallel_execute);
  auto new_parallel_execute = builder->create<tf_device::ParallelExecuteOp>(
      parallel_execute.getLoc(), num_regions, output_types);

  // Replace the uses of the original parallel_execute before region containing
  // merged execute.
  Operation* new_parallel_execute_op = new_parallel_execute.getOperation();
  for (int i = 0; i < num_results_before_region; ++i)
    parallel_execute_op->getResult(i).replaceAllUsesWith(
        new_parallel_execute_op->getResult(i));

  // Replace the uses of the original parallel_execute after region containing
  // merged execute. The number of results changed in the region containing the
  // merged execute, but they should match, so results are replaced starting
  // from the ends of both parallel_execute.
  const int new_parallel_execute_num_results =
      new_parallel_execute_op->getNumResults();
  for (int i = 0; i < num_results_after_region; ++i)
    parallel_execute_op->getResult(parallel_execute_num_results - i - 1)
        .replaceAllUsesWith(new_parallel_execute_op->getResult(
            new_parallel_execute_num_results - i - 1));

  // Replace the uses of the original parallel_execute for the region containing
  // the merged execute.
  auto old_region_results = parallel_execute.GetRegionOutputs(region_index);
  for (int i = 0, end = var_access_info.old_to_new_output_mapping.size();
       i < end; ++i) {
    if (var_access_info.old_to_new_output_mapping[i] < 0) continue;
    old_region_results[i].replaceAllUsesWith(new_parallel_execute_op->getResult(
        var_access_info.old_to_new_output_mapping[i] +
        num_results_before_region));
  }

  // Replace original terminator with new terminator for returning merged
  // execute results.
  Operation* old_terminator = execute_region->front().getTerminator();
  builder->setInsertionPointToEnd(&execute_region->front());
  builder->create<tf_device::ReturnOp>(old_terminator->getLoc(),
                                       merged_execute_launch.getResults());
  old_terminator->erase();

  // Remove the original TPUExecute op.
  execute_launch.erase();

  // Move all regions from old parallel_execute to new parallel_execute.
  for (auto region : llvm::zip(new_parallel_execute_op->getRegions(),
                               parallel_execute_op->getRegions()))
    std::get<0>(region).takeBody(std::get<1>(region));

  // Remove the original parallel_execute.
  parallel_execute_op->dropAllUses();
  parallel_execute.erase();
}

// Replaces TPUExecute with TPUExecuteAndUpdateVariables.
void ReplaceExecute(tf_device::LaunchOp execute_launch,
                    tf_device::LaunchOp merged_execute_launch,
                    const VariableAccessesForTPUExecute& var_access_info) {
  // Replace the uses.
  for (int i = 0, end = var_access_info.old_to_new_output_mapping.size();
       i < end; ++i) {
    if (var_access_info.old_to_new_output_mapping[i] < 0) continue;
    execute_launch.getResult(i).replaceAllUsesWith(
        merged_execute_launch.getResult(
            var_access_info.old_to_new_output_mapping[i]));
  }

  // Remove the original TPUExecute op.
  execute_launch.getOperation()->dropAllUses();
  execute_launch.erase();
}

// Merges the variable accesses into one TPUExecute op.
LogicalResult MergeForOneTPUExecute(
    tf_device::LaunchOp execute_launch,
    const mlir::TF::ResourceAliasAnalysis::Info& resource_analysis_info,
    bool check_device, bool check_same_region, OpBuilder* builder) {
  auto var_access_info = BuildVariableAccessInfo(
      execute_launch, resource_analysis_info, check_device, check_same_region);
  if (var_access_info.per_resource_info.empty()) return success();

  // Start creating the new TPUExecuteAndUpdateVariables op.
  builder->setInsertionPoint(execute_launch);
  // Output types. Skip the original outputs for merged assigns.
  llvm::SmallVector<Type, 8> new_output_types;
  int old_output_index = 0;
  for (const auto& type : execute_launch.getResultTypes()) {
    if (var_access_info.old_to_new_output_mapping[old_output_index] >= 0) {
      new_output_types.push_back(type);
    }
    ++old_output_index;
  }
  // The attributes for merged variable reads and updates.
  llvm::SmallVector<int64_t, 8> device_var_reads_indices;
  llvm::SmallVector<int64_t, 8> device_var_updates_indices;
  for (auto resource : var_access_info.resources_read) {
    auto info_it = var_access_info.per_resource_info.find(resource);
    if (info_it == var_access_info.per_resource_info.end()) continue;
    device_var_reads_indices.push_back(info_it->second.execute_input_index);
    device_var_updates_indices.push_back(info_it->second.execute_output_index);
  }

  // Check that all resources are either read or written to.
  for (auto it : llvm::enumerate(var_access_info.new_operand_values)) {
    Type type = it.value().getType();
    if (type.isa<TensorType>() &&
        type.cast<TensorType>().getElementType().isa<TF::ResourceType>()) {
      if (!llvm::is_contained(device_var_reads_indices, it.index()) &&
          !llvm::is_contained(device_var_updates_indices, it.index())) {
        return execute_launch.GetBody().front().emitError("operand #")
               << it.index()
               << " is a resource that was neither read nor written to; this "
                  "resource potentially failed to be hoisted";
      }
    }
  }

  // Create the merged execute and update variables op.
  auto merged_execute = builder->create<TF::TPUExecuteAndUpdateVariablesOp>(
      execute_launch.getLoc(), new_output_types,
      var_access_info.new_operand_values,
      llvm::ArrayRef<NamedAttribute>{
          builder->getNamedAttr(
              "device_var_reads_indices",
              builder->getI64ArrayAttr(device_var_reads_indices)),
          builder->getNamedAttr(
              "device_var_updates_indices",
              builder->getI64ArrayAttr(device_var_updates_indices))});

  // Wrap in launch for device assignment.
  auto merged_execute_launch = builder->create<tf_device::LaunchOp>(
      merged_execute.getLoc(), execute_launch.getDeviceAttr(),
      merged_execute.getResultTypes());
  merged_execute_launch.getBody().push_back(new Block);

  builder->setInsertionPointToEnd(&merged_execute_launch.GetBody());
  builder->create<tf_device::ReturnOp>(merged_execute.getLoc(),
                                       merged_execute.getResults());

  merged_execute.getOperation()->moveBefore(
      merged_execute_launch.GetBody().getTerminator());

  if (auto parallel_execute = llvm::dyn_cast<tf_device::ParallelExecuteOp>(
          execute_launch->getParentOp()))
    ReplaceParallelExecute(parallel_execute, execute_launch,
                           merged_execute_launch, var_access_info, builder);
  else
    ReplaceExecute(execute_launch, merged_execute_launch, var_access_info);

  // Remove the assign ops.
  for (const auto& entry : var_access_info.per_resource_info) {
    const auto& info = entry.getSecond();
    if (info.assign) info.assign->erase();
  }

  // Remove the read ops if they have no more uses.
  for (const auto& entry : var_access_info.per_resource_info) {
    const auto& info = entry.getSecond();
    if (info.read->use_empty()) info.read->erase();
  }
  return success();
}

// Checks if an ops parent is a tf_device.parallel_execute and the region the
// op is in is perfectly wrapped.
bool ParentParallelExecuteWrapsSingleOp(Operation* op) {
  auto parallel_execute =
      llvm::dyn_cast<tf_device::ParallelExecuteOp>(op->getParentOp());
  if (!parallel_execute) return true;

  return parallel_execute.RegionWrapsSingleOp(
      op->getParentRegion()->getRegionNumber());
}

void TPUMergeVariablesWithExecutePass::runOnOperation() {
  ModuleOp module = getOperation();
  mlir::TF::ResourceAliasAnalysis resource_analysis(module);
  module.walk([&](func::FuncOp func) {
    const auto& resource_analysis_info =
        resource_analysis.GetAnalysisForFunc(func);
    // Find all the executes first, since we will mutate the nodes around each
    // execute.
    llvm::SmallVector<tf_device::LaunchOp, 8> execute_launches;
    func.walk([&](tf_device::LaunchOp op) {
      if (op.WrapsSingleOp() &&
          llvm::isa<TF::TPUExecuteOp>(op.GetBody().front()) &&
          ParentParallelExecuteWrapsSingleOp(op))
        execute_launches.push_back(op);
    });

    for (auto execute_launch : execute_launches) {
      OpBuilder builder(&getContext());
      const bool parent_is_replicate =
          llvm::isa<tf_device::ReplicateOp>(execute_launch->getParentOp()) ||
          (llvm::isa<tf_device::ParallelExecuteOp>(
               execute_launch->getParentOp()) &&
           llvm::isa<tf_device::ReplicateOp>(
               execute_launch->getParentOp()->getParentOp()));

      // If this is inside a tf_device::ReplicateOp, the variables are
      // guaranteed to be on the same device as the TPUExecute op. Skip device
      // checking in that case, but we need to check that we are only merging
      // reads/assigns that are also in this replicated region.
      if (failed(MergeForOneTPUExecute(
              execute_launch, resource_analysis_info,
              /*check_device=*/!parent_is_replicate,
              /*check_same_region=*/parent_is_replicate, &builder))) {
        signalPassFailure();
        return;
      }
    }
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUMergeVariablesWithExecutePass() {
  return std::make_unique<TPUMergeVariablesWithExecutePass>();
}

}  // namespace TFTPU
}  // namespace mlir
