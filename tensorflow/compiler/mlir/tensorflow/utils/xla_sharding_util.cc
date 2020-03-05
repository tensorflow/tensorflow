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

#include "llvm/ADT/SmallVector.h"
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

}  // namespace tensorflow
