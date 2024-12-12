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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/ir/types/dialect.h"
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

  absl::Status Sync() override {
    return errors::Unimplemented("FakeDevice::Sync()");
  }

  static std::unique_ptr<Device> Make(const string& name,
                                      const string& desc = "") {
    DeviceNameUtils::ParsedName parsed_name;
    DeviceNameUtils::ParseFullName(name, &parsed_name);

    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(parsed_name.type);
    device_attributes.set_physical_device_desc(desc);
    return std::make_unique<FakeDevice>(device_attributes);
  }
};

TEST(DeviceUtilTest, AddDeviceToOp) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_type::TFTypeDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  const std::string cpu0 = "/job:worker/replica:0/task:0/device:CPU:0";
  const std::string gpu0 = "/job:worker/replica:1/task:2/device:GPU:0";
  const std::string gpu1 = "/job:worker/replica:1/task:2/device:GPU:1";

  llvm::SmallVector<std::unique_ptr<Device>, 2> devices;
  devices.push_back(FakeDevice::Make(cpu0));
  devices.push_back(FakeDevice::Make(gpu0, "compute capability: 7.0"));
  devices.push_back(FakeDevice::Make(gpu1));

  DeviceSet device_set;
  for (auto& device : devices) device_set.AddDevice(device.get());
  AddDevicesToOp(*module_ref, &device_set);

  auto devices_attr =
      (*module_ref)->getAttrOfType<mlir::DictionaryAttr>("tf.devices");
  ASSERT_NE(devices_attr, nullptr);
  ASSERT_EQ(devices_attr.size(), 3);

  // CPU device added with an empty metadata.
  auto device_meta_0 = mlir::dyn_cast<mlir::UnitAttr>(devices_attr.get(cpu0));
  ASSERT_NE(device_meta_0, nullptr);

  // GPU device successfully parsed compute capability from description.
  auto device_meta_1 =
      mlir::dyn_cast<mlir::TF::GpuDeviceMetadata>(devices_attr.get(gpu0));
  ASSERT_NE(device_meta_1, nullptr);
  ASSERT_EQ(device_meta_1.getCcMajor(), 7);
  ASSERT_EQ(device_meta_1.getCcMinor(), 0);

  // If description is empty GPU devices added with an empty metadata.
  auto device_meta_2 = mlir::dyn_cast<mlir::UnitAttr>(devices_attr.get(gpu1));
  ASSERT_NE(device_meta_2, nullptr);
}

TEST(DeviceUtilTest, AddDeviceToOpNullDeviceSet) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  AddDevicesToOp(*module_ref, /*device_set=*/nullptr);
  EXPECT_EQ((*module_ref)->getAttr("tf.devices"), nullptr);
}

TEST(DeviceUtilTest, GetDevicesFromOpNoDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesAttributeType) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  (*module_ref)->setAttr("tf.devices", builder.getBoolAttr(false));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesAttributeArraySubtype) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  (*module_ref)->setAttr("tf.devices", builder.getI32ArrayAttr({8}));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpBadDevicesInDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);
  (*module_ref)
      ->setAttr("tf.devices",
                builder.getDictionaryAttr(builder.getNamedAttr(
                    "bad_device", builder.getDictionaryAttr({}))));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::failed(GetDevicesFromOp(*module_ref, &devices)));
}

TEST(DeviceUtilTest, GetDevicesFromOpValidDeviceInDevicesAttribute) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::Builder builder(*module_ref);

  auto device_dict = builder.getDictionaryAttr(
      {builder.getNamedAttr("/job:worker/replica:0/task:0/device:CPU:0",
                            builder.getDictionaryAttr({}))});
  (*module_ref)->setAttr("tf.devices", device_dict);

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));

  ASSERT_EQ(devices.NumDevices(), 1);
  ASSERT_EQ(devices.device_names().size(), 1);
  ASSERT_EQ(DeviceNameUtils::ParsedNameToString(devices.device_names()[0]),
            "/job:worker/replica:0/task:0/device:CPU:0");
}

TEST(DeviceUtilTest, GetGpuDeviceMetadata) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_type::TFTypeDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  mlir::Builder builder(*module_ref);

  const std::string gpu0 = "/job:worker/replica:0/task:0/device:GPU:0";
  const std::string gpu1 = "/job:worker/replica:0/task:0/device:GPU:1";

  llvm::SmallVector<mlir::NamedAttribute, 2> metadata;
  metadata.push_back(builder.getNamedAttr(
      gpu0, mlir::TF::GpuDeviceMetadata::get(module_ref->getContext(), 1, 2)));

  (*module_ref)->setAttr("tf.devices", builder.getDictionaryAttr(metadata));

  mlir::TF::RuntimeDevices devices;
  EXPECT_TRUE(mlir::succeeded(GetDevicesFromOp(*module_ref, &devices)));

  DeviceNameUtils::ParsedName parsed_name;
  DeviceNameUtils::ParseFullName(gpu0, &parsed_name);
  auto meta_0 = devices.GetGpuDeviceMetadata(parsed_name);
  ASSERT_TRUE(meta_0.has_value());
  ASSERT_EQ(meta_0->getCcMajor(), 1);
  ASSERT_EQ(meta_0->getCcMinor(), 2);

  DeviceNameUtils::ParseFullName(gpu1, &parsed_name);
  auto meta_1 = devices.GetGpuDeviceMetadata(parsed_name);
  ASSERT_FALSE(meta_1.has_value());
}

TEST(DeviceUtilTest, GetDeviceOrdinalFromDeviceString) {
  const std::string tpu0 = "/job:worker/replica:0/task:0/device:TPU:0";
  const std::string tpu1 = "/job:worker/replica:0/task:0/device:TPU:1";

  mlir::MLIRContext context;
  auto unknown_loc = mlir::UnknownLoc::get(&context);

  int64_t device_ordinal0 = -1;
  mlir::LogicalResult result0 =
      GetDeviceOrdinalFromDeviceString(unknown_loc, tpu0, &device_ordinal0);
  EXPECT_TRUE(mlir::succeeded(result0));
  EXPECT_EQ(device_ordinal0, 0);

  int64_t device_ordinal1 = -1;
  mlir::LogicalResult result1 =
      GetDeviceOrdinalFromDeviceString(unknown_loc, tpu1, &device_ordinal1);
  EXPECT_TRUE(mlir::succeeded(result1));
  EXPECT_EQ(device_ordinal1, 1);
}

TEST(DeviceUtilTest, GetDeviceOrdinalFromDeviceStringInvalid) {
  mlir::MLIRContext context;
  auto unknown_loc = mlir::UnknownLoc::get(&context);

  int64_t device_ordinal = -1;
  mlir::LogicalResult result = GetDeviceOrdinalFromDeviceString(
      unknown_loc, "bad_device", &device_ordinal);
  EXPECT_TRUE(mlir::failed(result));
}

TEST(DeviceUtilTest, GetDeviceOrdinalFromDeviceStringNoId) {
  const std::string tpu_no_id = "/job:worker/replica:0/task:0/device:TPU";

  mlir::MLIRContext context;
  auto unknown_loc = mlir::UnknownLoc::get(&context);

  int64_t device_ordinal = -1;
  mlir::LogicalResult result =
      GetDeviceOrdinalFromDeviceString(unknown_loc, tpu_no_id, &device_ordinal);
  EXPECT_TRUE(mlir::failed(result));
}

}  // anonymous namespace
}  // namespace tensorflow
