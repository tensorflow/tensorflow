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

#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

  Status Sync() override { return errors::Unimplemented("FakeDevice::Sync()"); }

  static std::unique_ptr<Device> Make(const string& name) {
    DeviceNameUtils::ParsedName parsed_name;
    DeviceNameUtils::ParseFullName(name, &parsed_name);

    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(parsed_name.type);
    return std::make_unique<FakeDevice>(device_attributes);
  }
};

TEST(DeviceUtilTest, AddDeviceToOp) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  DeviceSet device_set;
  llvm::SmallVector<std::unique_ptr<Device>, 2> devices;
  devices.push_back(
      FakeDevice::Make("/job:worker/replica:0/task:0/device:CPU:0"));
  devices.push_back(
      FakeDevice::Make("/job:worker/replica:1/task:2/device:GPU:3"));
  for (auto& device : devices) device_set.AddDevice(device.get());

  AddDevicesToOp(*module_ref, &device_set);
  auto devices_attr = module_ref->getAttrOfType<mlir::ArrayAttr>("tf.devices");
  ASSERT_NE(devices_attr, nullptr);
  ASSERT_EQ(devices_attr.size(), 2);
  auto device_attr_0 = devices_attr.getValue()[0].dyn_cast<mlir::StringAttr>();
  ASSERT_NE(device_attr_0, nullptr);
  EXPECT_EQ(device_attr_0.getValue(),
            "/job:worker/replica:0/task:0/device:CPU:0");
  auto device_attr_1 = devices_attr.getValue()[1].dyn_cast<mlir::StringAttr>();
  ASSERT_NE(device_attr_1, nullptr);
  EXPECT_EQ(device_attr_1.getValue(),
            "/job:worker/replica:1/task:2/device:GPU:3");
}

TEST(DeviceUtilTest, AddDeviceToOpNullDeviceSet) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  AddDevicesToOp(*module_ref, /*device_set=*/nullptr);
  EXPECT_EQ(module_ref->getAttr("tf.devices"), nullptr);
}

TEST(DeviceUtilTest, GetDevicesFromOpNoDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  llvm::SmallVector<DeviceNameUtils::ParsedName, 8> devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesAttributeType) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  module_ref->setAttr("tf.devices", builder.getBoolAttr(false));

  llvm::SmallVector<DeviceNameUtils::ParsedName, 8> devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesAttributeArraySubtype) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  module_ref->setAttr("tf.devices", builder.getI32ArrayAttr({8}));

  llvm::SmallVector<DeviceNameUtils::ParsedName, 8> devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesInDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  module_ref->setAttr("tf.devices", builder.getStrArrayAttr({"bad_device"}));

  llvm::SmallVector<DeviceNameUtils::ParsedName, 8> devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpValidDeviceInDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  module_ref->setAttr(
      "tf.devices",
      builder.getStrArrayAttr({"/job:worker/replica:0/task:0/device:CPU:0"}));

  llvm::SmallVector<DeviceNameUtils::ParsedName, 8> devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));
  ASSERT_EQ(devices.size(), 1);
  EXPECT_EQ(DeviceNameUtils::ParsedNameToString(devices[0]),
            "/job:worker/replica:0/task:0/device:CPU:0");
}

}  // anonymous namespace
}  // namespace tensorflow
