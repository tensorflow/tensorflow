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

#include "tensorflow/core/common_runtime/eager/context.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

typedef FunctionDefHelper FDH;

// Return a fake device.
static Device* CreateDevice(const string& type, int n) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/device:" + type + ":" +
                std::to_string(n));
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class EagerContextTest : public ::testing::Test {
 public:
  EagerContextTest() : device_manager_(nullptr), context_(nullptr) {}

  ~EagerContextTest() override {
    delete device_manager_;
    if (context_) {
      context_->Unref();
    }
  }

  EagerContext* context() { return context_; }

  void InitContext(const SessionOptions& opts,
                   ContextDevicePlacementPolicy policy) {
    ASSERT_EQ(context_, nullptr);
    InitDeviceManager();
    context_ =
        new EagerContext(opts, policy,
                         /* async */ false, device_manager_,
                         /* device_mgr_owned */ false, /* rendezvous */ nullptr,
                         /* cluster_flr */ nullptr);
  }

 protected:
  void InitDeviceManager() {
    ASSERT_EQ(device_manager_, nullptr);
    device_manager_ = new DynamicDeviceMgr();
    std::vector<std::unique_ptr<Device>> added_devices;
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 1));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));
  }

  DynamicDeviceMgr* device_manager_;
  EagerContext* context_;
};

TEST_F(EagerContextTest, CompositeDevice) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  std::vector<string> underlying_devices = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_0));
  EXPECT_EQ(composite_device_0->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:0");
  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0", &device));
  EXPECT_EQ(device, composite_device_0);
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
  underlying_devices.push_back("/job:worker/replica:0/task:0/device:CPU:2");
  CompositeDevice* composite_device_2 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_2));
  EXPECT_EQ(composite_device_2->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:1");
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:1", &device));
  EXPECT_EQ(device, composite_device_2);

  EXPECT_TRUE(errors::IsNotFound(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:2", &device)));
}

TEST_F(EagerContextTest, CompositeDeviceWithGivenName) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const std::vector<string> underlying_devices_0 = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  const string composite_device_name =
      "/job:worker1/replica:0/task:0/device:COMPOSITE:5";
  // Create a CompositeDevice with the given name.
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_0, composite_device_name, &composite_device_0));
  EXPECT_EQ(composite_device_0->name(), composite_device_name);

  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(
      context()->FindCompositeDeviceFromName(composite_device_name, &device));
  EXPECT_EQ(device, composite_device_0);

  std::vector<string> underlying_devices_1 = {
      "/job:worker/replica:0/task:0/device:CPU:1",
      "/job:worker/replica:0/task:0/device:CPU:2"};
  // Find a CompositeDevice with the given name.
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_1, composite_device_0->name(), &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
}

TEST_F(EagerContextTest, AddFunctionDef) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatSame) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two_copy));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatDifferent) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const Tensor kThree = test::AsScalar<int64>(3);
  // Same function name but body is different. This should error out.
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kThree}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  Status s = context()->AddFunctionDef(x_times_two_copy);
  EXPECT_FALSE(s.ok());
}

}  // namespace
}  // namespace tensorflow
