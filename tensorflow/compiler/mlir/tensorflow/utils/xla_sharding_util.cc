/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tensorflow {

const char* const kXlaShardingAttrName = "_XlaSharding";
const char* const kInputShardingAttr = "input_sharding_configuration";
const char* const kOutputShardingAttr = "output_sharding_configuration";

llvm::Optional<mlir::StringRef> ParseShardingAttribute(
    mlir::Operation* operation) {
  const auto& sharding_attr =
      operation->getAttrOfType<mlir::StringAttr>(kXlaShardingAttrName);
  if (!sharding_attr) return llvm::Optional<mlir::StringRef>();
  return sharding_attr.getValue();
}

llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4>
ExtractInputsForLogicalDevices(int num_logical_cores,
                               mlir::tf_device::LaunchFuncOp launch_func) {
  // Initialize the input list for each logical devices.
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> input_list;
  input_list.reserve(num_logical_cores);
  for (int i = 0; i < num_logical_cores; ++i)
    input_list.emplace_back(llvm::SmallVector<mlir::Value, 4>());

  llvm::SmallVector<mlir::Value, 4> launch_func_inputs(
      launch_func.getOperands());
  auto sharding_attrs =
      launch_func.getOperation()->getAttrOfType<mlir::ArrayAttr>(
          kInputShardingAttr);
  // If sharding attribute does not exist, then all inputs are placed on 0th
  // logical core by default.
  if (!sharding_attrs) {
    input_list[0] = launch_func_inputs;
    return input_list;
  }

  // Enumerate sharding configuration for each inputs. If input has replicate
  // sharding then all logical devices take the value as input. If input has
  // maximal sharding then only the specified logical device take the value as
  // the input.
  for (const auto& sharding_attr_and_index : llvm::enumerate(sharding_attrs)) {
    const auto& sharding_attr = sharding_attr_and_index.value();
    const auto input_index = sharding_attr_and_index.index();
    const auto& input_value = launch_func_inputs[input_index];

    xla::OpSharding sharding;
    sharding.ParseFromString(
        sharding_attr.cast<mlir::StringAttr>().getValue().str());

    const auto input_sharing_type = sharding.type();
    if (input_sharing_type == xla::OpSharding::OTHER)
      launch_func.emitError(
          "tiled inputs are not yet supported for model parallelism");

    if (input_sharing_type == xla::OpSharding::REPLICATED) {
      for (auto inputs : input_list) inputs.emplace_back(input_value);
    } else {
      assert(input_sharing_type == xla::OpSharding::MAXIMAL);
      const int logical_device_id = sharding.tile_assignment_devices(0);
      input_list[logical_device_id].emplace_back(input_value);
    }
  }
  return input_list;
}

mlir::LogicalResult ParseAndValidateOutputSharding(
    mlir::tf_device::LaunchFuncOp launch_func,
    mlir::SmallVector<xla::OpSharding, 4>* output_sharding_list) {
  output_sharding_list->reserve(launch_func.getNumResults());

  const auto output_sharding_attrs =
      launch_func.getOperation()->getAttrOfType<mlir::ArrayAttr>(
          kOutputShardingAttr);
  if (!output_sharding_attrs)
    return launch_func.emitError(
        "output_sharding_configuration missing from launch func");

  if (output_sharding_attrs.size() != launch_func.getNumResults())
    return launch_func.emitError("incorrect number of output sharding");

  for (auto output_sharding_and_index :
       llvm::enumerate(output_sharding_attrs)) {
    const auto& output_sharding = output_sharding_and_index.value();
    const int sharding_index = output_sharding_and_index.index();
    if (!output_sharding.isa<mlir::StringAttr>())
      return launch_func.emitError(llvm::formatv(
          "non-string output sharding at index {0}", sharding_index));

    xla::OpSharding sharding;
    if (!sharding.ParseFromString(
            output_sharding.cast<mlir::StringAttr>().getValue().str()))
      return launch_func.emitError("incorrect sharding format for outputs");

    const auto output_sharing_type = sharding.type();
    if (output_sharing_type == xla::OpSharding::OTHER)
      return launch_func.emitError(
          "tiled outputs are not yet supported for model parallelism");

    output_sharding_list->emplace_back(std::move(sharding));
  }
  return mlir::success();
}

namespace {

bool IsAssignedToLogicalDevice(const int core_id,
                               const xla::OpSharding& sharding) {
  return sharding.type() == xla::OpSharding::MAXIMAL &&
         sharding.tile_assignment_devices(0) == core_id;
}

// Returns the index of the return value of region in
// `tf_device.parallel_execute` that represents launch func output at
// index |launch_func_output_index|. Regions of parallel_execute may
// have different return values depending on outside sharding
// configuration.
int MapLaunchOutputIndexWithRegionOutputIndex(
    llvm::ArrayRef<xla::OpSharding> output_sharding_config, const int core_id,
    const int launch_func_output_index) {
  int region_output_index = 0;
  for (int output_index = 0; output_index < launch_func_output_index;
       ++output_index) {
    const auto& sharding = output_sharding_config[output_index];
    if (sharding.type() == xla::OpSharding::REPLICATED ||
        IsAssignedToLogicalDevice(core_id, sharding))
      region_output_index++;
  }

  return region_output_index;
}

}  // namespace

mlir::SmallVector<mlir::Type, 4> GetOutputTypesForLogicalDeviceComputation(
    const int logical_device_id,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::LaunchFuncOp launch_func) {
  mlir::SmallVector<mlir::Type, 4> output_types;
  output_types.reserve(launch_func.getNumResults());

  for (auto result_and_index : llvm::enumerate(launch_func.getResults())) {
    const auto output_index = result_and_index.index();
    const auto& output_sharding = output_sharding_config[output_index];
    const auto output_sharding_type = output_sharding.type();
    const auto& launch_func_output = result_and_index.value();

    if (output_sharding_type == xla::OpSharding::REPLICATED ||
        IsAssignedToLogicalDevice(logical_device_id, output_sharding))
      output_types.emplace_back(launch_func_output.getType());
  }

  return output_types;
}

void RemapOutputsFromLogicalDevices(
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::LaunchFuncOp launch_func,
    mlir::tf_device::ParallelExecuteOp parallel_execute) {
  for (auto result_and_index : llvm::enumerate(launch_func.getResults())) {
    const auto output_index = result_and_index.index();
    const auto& launch_func_output = result_and_index.value();
    const auto& output_sharding = output_sharding_config[output_index];
    const auto output_sharing_type = output_sharding.type();

    int logical_device_id = 0;
    if (output_sharing_type == xla::OpSharding::MAXIMAL)
      logical_device_id = output_sharding.tile_assignment_devices(0);

    // For maximal sharding configuration, correctly remap outputs from
    // parallel_execute region to users of the launch func.
    const int region_output_index = MapLaunchOutputIndexWithRegionOutputIndex(
        output_sharding_config, logical_device_id, output_index);

    const auto output_from_logical_device = parallel_execute.GetRegionOutputs(
        logical_device_id)[region_output_index];

    launch_func_output.replaceAllUsesWith(output_from_logical_device);
  }
}

}  // namespace tensorflow
