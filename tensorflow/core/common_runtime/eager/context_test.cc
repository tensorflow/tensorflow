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

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

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
    context_ = new EagerContext(
        opts, policy,
        /* default_mirroring_policy */ MIRRORING_NONE,
        /* async */ false,
        /* lazy_copy_function_remote_inputs */ false, device_manager_,
        /* device_mgr_owned */ false, /* rendezvous */ nullptr,
        /* custom_kernel_creator */ nullptr,
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

TEST_F(EagerContextTest, SelectDeviceExplicitHardPlacement) {
  SessionOptions options;
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(false);
  InitContext(options, DEVICE_PLACEMENT_EXPLICIT);

  Device* dev;
  DeviceNameUtils::ParsedName requested;
  const PrioritizedDeviceTypeVector supported{
      std::make_pair(DeviceType(DEVICE_GPU), 20),
      std::make_pair(DeviceType(DEVICE_CPU), 10),
  };

  // No supported devices should result in an error.
  requested.Clear();
  Status status = context()->SelectDevice(
      requested, PrioritizedDeviceTypeVector{}, DT_INVALID, &dev);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "No supported device found"))
      << "unexpected error message " << status.error_message();

  // An invalid requested device should also cause an error.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("GPU:99", &requested));
  status = context()->SelectDevice(requested, supported, DT_INVALID, &dev);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "Could not satisfy device specification"))
      << "unexpected error message " << status.error_message();

  // Should pick the "best" supported device if given no constraints.
  requested.Clear();
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_INVALID, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_GPU);

  // Should pick a CPU if asked to.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("CPU:1", &requested));
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_INVALID, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_CPU);

  // String tensors stay in GPU under hard device placement.
  requested.Clear();
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_STRING, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_GPU);
}

TEST_F(EagerContextTest, SelectDeviceExplicitSoftPlacement) {
  SessionOptions options;
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(true);
  InitContext(options, DEVICE_PLACEMENT_EXPLICIT);

  Device* dev;
  DeviceNameUtils::ParsedName requested;
  const PrioritizedDeviceTypeVector supported{
      std::make_pair(DeviceType(DEVICE_GPU), 20),
      std::make_pair(DeviceType(DEVICE_CPU), 10),
  };

  // No supported devices should result in an error.
  requested.Clear();
  Status status = context()->SelectDevice(
      requested, PrioritizedDeviceTypeVector{}, DT_INVALID, &dev);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "No supported device found"))
      << "unexpected error message " << status.error_message();

  // An invalid requested device should be replaced by the "best" one.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("GPU:99", &requested));
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_INVALID, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_GPU);

  // Should pick the "best" supported device if given no constraints.
  requested.Clear();
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_INVALID, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_GPU);

  // Should pick a CPU if asked to.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("CPU:1", &requested));
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_INVALID, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_CPU);

  // String tensors move to CPU under soft device placement.
  requested.Clear();
  TF_ASSERT_OK(context()->SelectDevice(requested, supported, DT_STRING, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_CPU);
}

TEST_F(EagerContextTest, CompositeDevice) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  std::vector<string> underlying_devices = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      &composite_device_0));
  EXPECT_EQ(composite_device_0->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:0");
  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0", &device));
  EXPECT_EQ(device, composite_device_0);
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
  underlying_devices.push_back("/job:worker/replica:0/task:0/device:CPU:2");
  CompositeDevice* composite_device_2 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      &composite_device_2));
  EXPECT_EQ(composite_device_2->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:1");
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:1", &device));
  EXPECT_EQ(device, composite_device_2);

  EXPECT_TRUE(errors::IsNotFound(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:2", &device)));
}

}  // namespace
}  // namespace tensorflow
