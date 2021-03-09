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
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#define DEBUG_TYPE "tf-tpu-merge-variables-with-execute"

namespace mlir {
namespace TFTPU {

namespace {
constexpr char kAliasingAttr[] = "tf.aliasing_output";
constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// A pass that finds on-device resource variable reads/assigns surrounding a
// tf.TPUExecute op, and merges them into a tf.TPUExecuteAndUpdateVariables.
// This allows the TPU execution to perform in-place variable updates.
//
// For example,
//
//   %0 = "tf.ReadVariableOp"(%arg0)
//   %1 = "tf.ReadVariableOp"(%arg1)
//   %2 = "tf.TPUExecute"(%0, %1, %compile)
//   %3 = "tf.AssignVariableOp"(%arg0, %2)
//
// will be transformed into
//
//   %2 = "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %compile)
//     { device_var_reads_indices = [0, 1],
//       device_var_updates_indices = [0, -1] }
//
// The transformation happens only for on-device variables. The above
// transformation requires %arg0, %arg1 to have the same device assignment as
// the TPUExecute op.

struct TPUMergeVariablesWithExecutePass
    : public PassWrapper<TPUMergeVariablesWithExecutePass, FunctionPass> {
  void runOnFunction() override;
};

// Information for a pair of input/output of the TPUExecute op and the
// surrounding read/assign ops.
struct VariableAccessInfo {
  int execute_input_index = -1;
  int execute_output_index = -1;
  Operation* read = nullptr;
  Operation* assign = nullptr;
};

// Information about all resource accesses to be fused into a TPUExecute op.
struct VariableAccessesForTPUExecute {
  // Maps each resource detected to VariableAccessInfo.
  llvm::SmallDenseMap<Value, VariableAccessInfo, 8> per_resource_info;
  // The corresponding new output index in TPUExecuteAndUpdateVariables for
  // each old output index in TPUExecute.
  llvm::SmallVector<int, 8> old_to_new_output_mapping;
  // The resources read by ReadVariableOps that are inputs to TPUExecute.
  // Ordered by the input indices to TPUExecute
  llvm::SmallVector<Value, 8> resources_read;
  // Operands for the new TPUExecuteAndUpdateVariables.
  llvm::SmallVector<Value, 8> new_operand_values;
};

// Returns if an op accesses a resource.
//
// TODO(yuanzx): Decide how to make this fine-grained. Right now we do not know
// if the resources alias.
bool OpAccessesResource(Operation* op) {
  return llvm::any_of(op->getOperandTypes(), [](const Type& type) {
    return type.isa<TF::ResourceType>() ||
           (type.isa<TensorType>() &&
            type.cast<TensorType>().getElementType().isa<TF::ResourceType>());
  });
}

// Finds the variable access info for a TPUExecute op.
//  - `check_device` specifies  whether it checks the device assignment of the
//  variables to match the TPUExecute op. This is optional in some context,
//  e.g., guaranteed by replication.
//  - `check_same_region` specifies whether the reads/assigns need to be in the
//  same region as `execute`. This is needed if `execute` is inside ReplicateOp.
VariableAccessesForTPUExecute BuildVariableAccessInfo(
    tf_device::LaunchOp execute_launch, bool check_device,
    bool check_same_region) {
  VariableAccessesForTPUExecute infos;
  Attribute device_attr = execute_launch.deviceAttr();
  if (check_device && !device_attr) return infos;
  auto func = execute_launch->getParentOfType<mlir::FuncOp>();

  // Track the first read op found, which is used later to check if there are
  // assign ops between it and the TPUExecute op. We will exclude reads before
  // interferencing accesses in a conservative way (see below). We do not
  // consider resource accesses in other islands since they ordering is enforced
  // by inter-island dependencies.
  Operation* first_read = nullptr;
  auto execute = cast<TF::TPUExecuteOp>(execute_launch.GetBody().front());
  auto parallel_execute = llvm::dyn_cast<tf_device::ParallelExecuteOp>(
      execute_launch->getParentOp());
  Operation* execute_parent =
      parallel_execute ? parallel_execute.getOperation() : execute_launch;
  // Find inputs that are variable reads.
  for (auto operand : llvm::enumerate(execute->getOpOperands())) {
    infos.new_operand_values.push_back(operand.value().get());
    auto read_op = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.value().get().getDefiningOp());
    if (!read_op) continue;
    if (check_same_region &&
        read_op->getParentRegion() != execute_parent->getParentRegion())
      continue;

    auto resource = read_op.resource();

    if (check_device) {
      // TODO(lyandy): Wrap resource ops in tf_device.launch.
      if (auto* resource_op = resource.getDefiningOp()) {
        auto resource_attr = resource_op->getAttr(kDeviceAttr);
        // Check device matching for the node defining the resource.
        if (!resource_attr || resource_attr != device_attr) continue;
      } else {
        auto resource_arg = resource.dyn_cast<BlockArgument>();
        assert(resource_arg);
        if (resource_arg.getOwner() != &func.front()) continue;
        // Check device matching for the argument defining the resource.
        auto resource_attr = func.getArgAttrOfType<mlir::StringAttr>(
            resource_arg.getArgNumber(), kFuncDeviceAttr);
        if (!resource_attr || resource_attr != device_attr) continue;
      }
    }

    auto emplace_res =
        infos.per_resource_info.try_emplace(resource, VariableAccessInfo());
    if (!emplace_res.second) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping execute that has multiple reads of a variable: "
                 << execute << "\n");
      infos.per_resource_info.shrink_and_clear();
      return infos;
    }

    auto& info = emplace_res.first->getSecond();
    info.execute_input_index = operand.index();
    info.read = read_op;
    infos.new_operand_values[operand.index()] = resource;
    infos.resources_read.push_back(resource);
    if (!first_read || info.read->isBeforeInBlock(first_read)) {
      first_read = info.read;
    }
  }

  if (!first_read) return infos;

  // Conservatively find the last resource-accessing op between first_read and
  // execute, excluding ReadVariableOps since they are read-only. This should
  // work fine for the reads/assigns created by resource lifting, since they are
  // placed close to the TPUExecute.
  Operation* last_may_modify_resource_access_before_execute = nullptr;
  for (Operation& op :
       llvm::reverse(llvm::make_range(std::next(first_read->getIterator()),
                                      execute_parent->getIterator()))) {
    if (llvm::dyn_cast<TF::ReadVariableOp>(&op)) continue;
    if (!OpAccessesResource(&op)) continue;
    last_may_modify_resource_access_before_execute = &op;
    break;
  }

  if (last_may_modify_resource_access_before_execute) {
    // Remove the reads before last_unknown_resource_access_before_execute.
    for (auto& op : llvm::make_range(
             first_read->getIterator(),
             last_may_modify_resource_access_before_execute->getIterator())) {
      auto read = llvm::dyn_cast<TF::ReadVariableOp>(&op);
      if (!read) continue;
      auto info_it = infos.per_resource_info.find(read.resource());
      if (info_it == infos.per_resource_info.end()) continue;
      int input_index = info_it->getSecond().execute_input_index;
      infos.new_operand_values[input_index] = execute.getOperand(input_index);
      infos.per_resource_info.erase(info_it);
    }
    infos.resources_read.erase(
        llvm::remove_if(infos.resources_read,
                        [&](const Value resource) {
                          return infos.per_resource_info.count(resource) == 0;
                        }),
        infos.resources_read.end());
  }

  if (infos.per_resource_info.empty()) {
    return infos;
  }

  // Find outputs that are variable assigns.
  Operation* last_assign = nullptr;
  llvm::SmallPtrSet<Operation*, 8> all_assigns;
  llvm::SmallVector<bool, 8> output_fused(execute_launch.getNumResults(),
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
    auto resource = assign_op.resource();
    auto it = infos.per_resource_info.find(resource);
    if (it == infos.per_resource_info.end()) continue;
    auto& info = it->getSecond();
    if (info.assign) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping execute that has multiple assigns of a variable: "
                 << execute << "\n");
      infos.per_resource_info.shrink_and_clear();
      return infos;
    }
    info.execute_output_index = execute_output.index();
    info.assign = assign_op;
    if (!last_assign || last_assign->isBeforeInBlock(assign_op)) {
      last_assign = assign_op;
    }
    all_assigns.insert(assign_op);
    output_fused[execute_output.index()] = true;
  }

  // Check if there are other resource accesses after execute.
  Operation* first_unknown_resource_access_after_execute = nullptr;
  if (last_assign) {
    for (auto& op : llvm::make_range(std::next(execute_parent->getIterator()),
                                     last_assign->getIterator())) {
      if (all_assigns.count(&op) > 0) continue;
      if (!OpAccessesResource(&op)) continue;
      first_unknown_resource_access_after_execute = &op;
      break;
    }
  }
  if (first_unknown_resource_access_after_execute) {
    // Remove the assigns after first_unknown_resource_access_after_execute.
    for (auto& op : llvm::make_range(
             first_unknown_resource_access_after_execute->getIterator(),
             std::next(last_assign->getIterator()))) {
      if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
        if (all_assigns.count(assign) == 0) continue;
        auto info_it = infos.per_resource_info.find(assign.resource());
        if (info_it == infos.per_resource_info.end()) continue;
        output_fused[info_it->second.execute_output_index] = false;
        info_it->second.execute_output_index = -1;
        info_it->second.assign = nullptr;
      }
    }
  }

  // Populate infos.old_to_new_output_mapping.
  int new_output_index = 0;
  infos.old_to_new_output_mapping.resize(execute_launch.getNumResults());
  for (int i = 0, end = execute_launch.getNumResults(); i < end; ++i) {
    if (output_fused[i]) {
      infos.old_to_new_output_mapping[i] = -1;
    } else {
      infos.old_to_new_output_mapping[i] = new_output_index;
      ++new_output_index;
    }
  }
  return infos;
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
void ReplaceParallelExecute(tf_device::ParallelExecuteOp parallel_execute,
                            tf_device::LaunchOp execute_launch,
                            tf_device::LaunchOp merged_execute_launch,
                            const VariableAccessesForTPUExecute& infos,
                            OpBuilder* builder) {
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
  for (int i = 0, end = infos.old_to_new_output_mapping.size(); i < end; ++i) {
    if (infos.old_to_new_output_mapping[i] < 0) continue;
    old_region_results[i].replaceAllUsesWith(new_parallel_execute_op->getResult(
        infos.old_to_new_output_mapping[i] + num_results_before_region));
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
                    const VariableAccessesForTPUExecute& infos) {
  // Replace the uses.
  for (int i = 0, end = infos.old_to_new_output_mapping.size(); i < end; ++i) {
    if (infos.old_to_new_output_mapping[i] < 0) continue;
    execute_launch.getResult(i).replaceAllUsesWith(
        merged_execute_launch.getResult(infos.old_to_new_output_mapping[i]));
  }

  // Remove the original TPUExecute op.
  execute_launch.getOperation()->dropAllUses();
  execute_launch.erase();
}

// Returns TPUCompileMlir op that generates the program executed by the
// TPUExecute op.
TF::_TPUCompileMlirOp GetTPUCompileOp(tf_device::LaunchOp execute_launch) {
  auto execute =
      llvm::dyn_cast<TF::TPUExecuteOp>(execute_launch.GetBody().front());
  if (!execute) return {};
  auto compile_launch = llvm::dyn_cast_or_null<tf_device::LaunchOp>(
      execute.getOperand(execute.getNumOperands() - 1).getDefiningOp());
  if (!compile_launch) return {};
  return llvm::dyn_cast<TF::_TPUCompileMlirOp>(
      compile_launch.GetBody().front());
}

// Updates the serialized module associated with the TPUExecute op to reflect
// the aliasing information for better management of device memory.
LogicalResult UpdateSerializedModule(tf_device::LaunchOp execute_launch,
                                     VariableAccessesForTPUExecute& infos) {
  TF::_TPUCompileMlirOp compile = GetTPUCompileOp(execute_launch);

  // Skip adding alias information in case of model parallelism i.e.,
  // TPUCompileMlir op generates multiple programs.
  if (!compile || compile.program().size() > 1) return failure();

  // Parse the serialized module
  mlir::OwningModuleRef module_ref;
  tensorflow::Status status = tensorflow::DeserializeMlirModule(
      compile.mlir_module().str(), compile.getContext(), &module_ref);
  if (!status.ok()) {
    LLVM_DEBUG(llvm::dbgs() << "Error in parsing serialized module: "
                            << status.error_message() << "\n");

    return failure();
  }

  // Add aliasing information to main function arguments.
  FuncOp main_func = module_ref->lookupSymbol<FuncOp>("main");
  if (!main_func) return failure();

  OpBuilder builder(main_func.getContext());
  for (auto resource : infos.resources_read) {
    auto& info = infos.per_resource_info[resource];
    if (info.execute_input_index < 0 || info.execute_output_index < 0) continue;
    auto aliasing_attr = main_func.getArgAttrOfType<mlir::IntegerAttr>(
        info.execute_input_index, kAliasingAttr);

    // Set only if aliasing attribute does not exist.
    if (!aliasing_attr) {
      main_func.setArgAttr(
          info.execute_input_index, kAliasingAttr,
          builder.getI64IntegerAttr(info.execute_output_index));
      continue;
    }
    // If aliasing attribute already exists, it must match the new value.
    assert(aliasing_attr.getInt() == info.execute_output_index);
  }

  // Serialize the updated module back into the TPUCompileMlir op.
  auto module_string = tensorflow::SerializeMlirModule(module_ref.get());
  compile.mlir_moduleAttr(
      mlir::StringAttr::get(module_ref->getContext(), module_string));
  return success();
}

// Merges the variable accesses into one TPUExecute op.
LogicalResult MergeForOneTPUExecute(tf_device::LaunchOp execute_launch,
                                    bool check_device, bool check_same_region,
                                    OpBuilder* builder) {
  auto infos =
      BuildVariableAccessInfo(execute_launch, check_device, check_same_region);
  if (infos.per_resource_info.empty()) return success();

  // Update the serialized module with aliasing information for better memory
  // management on device.
  // TODO(b/172608422): Benchmark the cost of deserialization/serialization of
  // the attached module. We can avoid it by serializing it at the end of the
  // bridge pipeline.
  if (failed(UpdateSerializedModule(execute_launch, infos))) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Unable to update the serialized module with aliasing information "
           "which can lead to poor memory management on device.\n");
  }

  // Start creating the new TPUExecuteAndUpdateVariables op.
  builder->setInsertionPoint(execute_launch);
  // Output types. Skip the original outputs for fused assigns.
  llvm::SmallVector<Type, 8> new_output_types;
  int old_output_index = 0;
  for (const auto& type : execute_launch.getResultTypes()) {
    if (infos.old_to_new_output_mapping[old_output_index] >= 0) {
      new_output_types.push_back(type);
    }
    ++old_output_index;
  }
  // The attributes for fused variable reads and updates.
  llvm::SmallVector<int64_t, 8> device_var_reads_indices;
  llvm::SmallVector<int64_t, 8> device_var_updates_indices;
  for (auto resource : infos.resources_read) {
    const auto& info = infos.per_resource_info[resource];
    device_var_reads_indices.push_back(info.execute_input_index);
    device_var_updates_indices.push_back(info.execute_output_index);
  }

  // Check that all resources op are either read or written to.
  for (auto it : llvm::enumerate(infos.new_operand_values)) {
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
      execute_launch.getLoc(), new_output_types, infos.new_operand_values,
      llvm::ArrayRef<NamedAttribute>{
          builder->getNamedAttr(
              "device_var_reads_indices",
              builder->getI64ArrayAttr(device_var_reads_indices)),
          builder->getNamedAttr(
              "device_var_updates_indices",
              builder->getI64ArrayAttr(device_var_updates_indices))});

  // Wrap in launch for device assignment.
  auto merged_execute_launch = builder->create<tf_device::LaunchOp>(
      merged_execute.getLoc(), execute_launch.deviceAttr(),
      merged_execute.getResultTypes());
  merged_execute_launch.body().push_back(new Block);

  builder->setInsertionPointToEnd(&merged_execute_launch.GetBody());
  builder->create<tf_device::ReturnOp>(merged_execute.getLoc(),
                                       merged_execute.getResults());

  merged_execute.getOperation()->moveBefore(
      merged_execute_launch.GetBody().getTerminator());

  if (auto parallel_execute = llvm::dyn_cast<tf_device::ParallelExecuteOp>(
          execute_launch->getParentOp()))
    ReplaceParallelExecute(parallel_execute, execute_launch,
                           merged_execute_launch, infos, builder);
  else
    ReplaceExecute(execute_launch, merged_execute_launch, infos);

  // Remove the assign ops.
  for (const auto& entry : infos.per_resource_info) {
    const auto& info = entry.getSecond();
    if (info.assign) info.assign->erase();
  }

  // Remove the read ops if they have no more uses.
  for (const auto& entry : infos.per_resource_info) {
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

void TPUMergeVariablesWithExecutePass::runOnFunction() {
  // Find all the executes first, since we will mutate the nodes around each
  // execute.
  llvm::SmallVector<tf_device::LaunchOp, 8> execute_launches;
  getFunction().walk([&](tf_device::LaunchOp op) {
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

    // If this is inside a tf_device::ReplicateOp, the variables are guaranteed
    // to be on the same device as the TPUExecute op. Skip device checking in
    // that case, but we need to check that we are only merging reads/assigns
    // that are also in this replicated region.
    if (failed(MergeForOneTPUExecute(
            execute_launch, /*check_device=*/!parent_is_replicate,
            /*check_same_region=*/parent_is_replicate, &builder))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUMergeVariablesWithExecutePass() {
  return std::make_unique<TPUMergeVariablesWithExecutePass>();
}

static PassRegistration<TPUMergeVariablesWithExecutePass> pass(
    "tf-tpu-merge-variables-with-execute",
    "Merges device variable reads/updates into tpu execute nodes");

}  // namespace TFTPU
}  // namespace mlir
