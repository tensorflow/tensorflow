/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This pass inserts corert.transfer op to make sure any argument of any op is
// on the same device of the op itself.

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {

namespace {

using DeviceNameUtils = ::tensorflow::DeviceNameUtils;

constexpr const char *kDeviceAttr = "device";
constexpr const char *kTFRTDeviceAttr = "tfrt.device";
// TODO(b/175480458): Do not assign default device once every op in the TF
// dialect has the device attribute.
constexpr const char *kDefaultDevice =
    "/job:localhost/replica:0/task:0/device:CPU:0";

// This method canonicalizes the device name so that we can use string
// comparison to see if two devices are the same. It does the following
// transformations:
// 1) Set device ID to 0 if device ID is not already specified.
// 2) Change the device type to uppercase string.
static std::string CanonicalizeDeviceName(const std::string &device) {
  if (device.empty()) return kDefaultDevice;

  DeviceNameUtils::ParsedName parsed_name;
  if (!device.empty() && device.at(0) == '/') {
    DeviceNameUtils::ParseFullName(device, &parsed_name);
  } else {
    DeviceNameUtils::ParseFullName("/device:" + device, &parsed_name);
  }

  if (!parsed_name.has_id) {
    parsed_name.has_id = true;
    parsed_name.id = 0;
  }

  if (parsed_name.type == "cpu")
    parsed_name.type = "CPU";
  else if (parsed_name.type == "gpu")
    parsed_name.type = "GPU";
  else if (parsed_name.type == "tpu")
    parsed_name.type = "TPU";
  return DeviceNameUtils::ParsedNameToString(parsed_name);
}

// Return the device of the given operation.
static std::string GetDevice(Operation *op) {
  std::string device = "";
  if (StringAttr device_attr = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
    device = device_attr.getValue().str();
  } else if (auto execute_op = llvm::dyn_cast<tfrt::corert::ExecuteOp>(op)) {
    SmallVector<std::pair<StringRef, Attribute>, 4> attrs;
    execute_op.getOpAttrs(&attrs);
    for (std::pair<StringRef, Attribute> entry : attrs) {
      if (entry.first == kDeviceAttr && mlir::isa<StringAttr>(entry.second)) {
        device = mlir::cast<StringAttr>(entry.second).getValue().str();
        break;
      }
    }
  }

  return CanonicalizeDeviceName(device);
}

// Return the device of the given value.
static std::string GetDevice(mlir::Value value, func::FuncOp parent_func_op) {
  std::string device = "";
  if (BlockArgument block_arg = mlir::dyn_cast<BlockArgument>(value)) {
    if (StringAttr device_attr = parent_func_op.getArgAttrOfType<StringAttr>(
            block_arg.getArgNumber(), kTFRTDeviceAttr)) {
      device = device_attr.getValue().str();
    }
  } else {
    device = GetDevice(value.getDefiningOp());
  }

  return CanonicalizeDeviceName(device);
}

struct CrossDeviceTransferPass
    : public PassWrapper<CrossDeviceTransferPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrossDeviceTransferPass)

  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "tfrt-cross-device-transfer";
  }

  llvm::StringRef getDescription() const final {
    return "This pass inserts corert.transfer op to make sure any argument of "
           "any op is on the same device of the op itself.";
  }
};

void CrossDeviceTransferPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  llvm::DenseMap<mlir::Value, llvm::StringMap<mlir::Value>>
      transferred_value_by_value_and_device;

  func_op.getBody().walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) return WalkResult::advance();
    // Do not transfer the argument of corert.transfer op.
    if (llvm::isa<tfrt::corert::TransferOp>(op)) return WalkResult::advance();

    OpBuilder builder(op);
    std::string dst_device = GetDevice(op);
    mlir::Type tensor_type_type =
        builder.getType<::tfrt::compiler::TensorTypeType>();
    mlir::Type device_type = builder.getType<::tfrt::compiler::DeviceType>();

    for (mlir::Value arg : op->getOperands()) {
      // Do not transfer non-TensorHandle values.
      if (!mlir::isa<tfrt::corert::TensorHandleType>(arg.getType())) continue;

      // Do not transfer the result of corert.transfer op.
      if (OpResult op_result = mlir::dyn_cast<OpResult>(arg)) {
        Operation *defining_op = arg.getDefiningOp();
        if (llvm::isa<tfrt::corert::TransferOp>(defining_op)) continue;
      }

      std::string src_device = GetDevice(arg, func_op);

      if (DeviceNameUtils::LocalName(src_device) ==
          DeviceNameUtils::LocalName(dst_device))
        continue;

      // Re-use the value already transferred to the given device.
      llvm::StringMap<mlir::Value> &transferred_value_by_device =
          transferred_value_by_value_and_device[arg];
      auto iter = transferred_value_by_device.find(dst_device);
      if (iter != transferred_value_by_device.end()) {
        op->replaceUsesOfWith(arg, iter->second);
        continue;
      }

      mlir::Value chain_in = func_op.getArgument(0);
      auto get_device_op = builder.create<tfrt::compiler::GetDeviceOp>(
          op->getLoc(), device_type, chain_in, dst_device);
      auto get_tensor_type_op =
          builder.create<tfrt::corert::GetDstTensorTypeOp>(
              op->getLoc(), tensor_type_type, arg, get_device_op.getResult());
      auto transfer_op = builder.create<tfrt::corert::TransferOp>(
          op->getLoc(), arg.getType(), arg, get_device_op.getResult(),
          get_tensor_type_op.getResult());
      mlir::Value new_arg = transfer_op.getResult();
      transferred_value_by_device[dst_device] = new_arg;
      op->replaceUsesOfWith(arg, new_arg);
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateCrossDeviceTransferPass() {
  return std::make_unique<CrossDeviceTransferPass>();
}

static PassRegistration<CrossDeviceTransferPass> pass;

}  // namespace tensorflow
