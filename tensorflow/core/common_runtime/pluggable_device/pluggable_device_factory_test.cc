/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"

#include <memory>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/device.h"  // IWYU pragma: keep
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

extern "C" {
// from test_pluggable_device.cc
void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status);
}

class PluggableDeviceFactoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    PluggableDeviceInit_Api api;
    api.init_plugin_fn = [](SE_PlatformRegistrationParams* const params,
                            TF_Status* const status) {
      SE_InitPlugin(params, status);
      params->platform->name = "MY_PLATFORM";
      params->platform->use_bfc_allocator = true;
    };
    TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));

    // Initialize executors for the test platform.
    auto platform_or =
        stream_executor::PlatformManager::PlatformWithName("MY_PLATFORM");
    TF_ASSERT_OK(platform_or);
    stream_executor::Platform* platform = platform_or.value();
    for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
      TF_ASSERT_OK(platform->ExecutorForDevice(i));
    }
  }

  // Returns a SessionOptions object with given virtual device settings. In
  // `memory_limit_mb` and `priorities` the outer vector indexes physical
  // devices, the inner vectors index the virtual devices to be created on each
  // physical devices.
  SessionOptions MakeSessionOptions(
      const std::vector<std::vector<float>>& memory_limit_mb,
      const std::vector<std::vector<int>>& priorities) {
    SessionOptions options;
    ConfigProto* config = &options.config;
    GPUOptions* gpu_options = config->mutable_pluggable_device_options();
    if (!memory_limit_mb.empty()) {
      for (int i = 0; i < memory_limit_mb.size(); ++i) {
        auto virtual_devices =
            gpu_options->mutable_experimental()->add_virtual_devices();
        for (float mb : memory_limit_mb[i]) {
          virtual_devices->add_memory_limit_mb(mb);
        }
        for (int priority : priorities[i]) {
          virtual_devices->add_priority(priority);
        }
      }
    }
    return options;
  }
};

TEST_F(PluggableDeviceFactoryTest, VirtualDevicesMemoryLimitTest) {
  DeviceIdManager::TestOnlyReset();
  SessionOptions opts = MakeSessionOptions(
      /*memory_limit_mb=*/{{123, 456}, {789}}, /*priorities=*/{{0, 1}, {2}});
  std::vector<std::unique_ptr<Device>> devices;
  PluggableDeviceFactory factory("MY_DEVICE", "MY_PLATFORM");
  TF_ASSERT_OK(
      factory.CreateDevices(opts, "/job:localhost/replica:0/task:0", &devices));
  EXPECT_EQ(devices.size(), 3);
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 123 << 20);
  EXPECT_EQ(devices[1]->attributes().memory_limit(), 456 << 20);
  EXPECT_EQ(devices[2]->attributes().memory_limit(), 789 << 20);
}

// Test TF device to platform device mapping.

TEST_F(PluggableDeviceFactoryTest, VirtualDevicesMappingDefaultTest) {
  DeviceIdManager::TestOnlyReset();
  SessionOptions opts = MakeSessionOptions(
      /*memory_limit_mb=*/{}, /*priorities=*/{});  // no virtual devices
  std::vector<std::unique_ptr<Device>> devices;
  PluggableDeviceFactory factory("MY_DEVICE", "MY_PLATFORM");
  TF_ASSERT_OK(
      factory.CreateDevices(opts, "/job:localhost/replica:0/task:0", &devices));

  for (int i = 0; i < devices.size(); ++i) {
    TfDeviceId tf_device_id(i);
    PlatformDeviceId platform_device_id;
    TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(
        DeviceType("MY_DEVICE"), tf_device_id, &platform_device_id));
    EXPECT_EQ(platform_device_id.value(), i);
  }
}

TEST_F(PluggableDeviceFactoryTest, VirtualDevicesMappingExplicitLimitTest) {
  DeviceIdManager::TestOnlyReset();
  SessionOptions opts = MakeSessionOptions(
      /*memory_limit_mb=*/{{100, 200}, {300}}, /*priorities=*/{{0, 1}, {2}});
  std::vector<std::unique_ptr<Device>> devices;
  PluggableDeviceFactory factory("MY_DEVICE", "MY_PLATFORM");
  TF_ASSERT_OK(
      factory.CreateDevices(opts, "/job:localhost/replica:0/task:0", &devices));

  EXPECT_EQ(devices.size(), 3);

  auto device_type = DeviceType("MY_DEVICE");
  PlatformDeviceId plat_id;
  TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(device_type, TfDeviceId(0),
                                                     &plat_id));
  EXPECT_EQ(plat_id.value(), 0);

  TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(device_type, TfDeviceId(1),
                                                     &plat_id));
  EXPECT_EQ(plat_id.value(), 0);

  TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(device_type, TfDeviceId(2),
                                                     &plat_id));
  EXPECT_EQ(plat_id.value(), 1);
}

TEST_F(PluggableDeviceFactoryTest, VirtualDevicesMappingEmptyLimitTest) {
  DeviceIdManager::TestOnlyReset();
  // Empty inner vector means 1 virtual device taking all available memory.
  SessionOptions opts = MakeSessionOptions(/*memory_limit_mb=*/{{100, 200}, {}},
                                           /*priorities=*/{{0, 1}, {}});
  std::vector<std::unique_ptr<Device>> devices;
  PluggableDeviceFactory factory("MY_DEVICE", "MY_PLATFORM");
  TF_ASSERT_OK(
      factory.CreateDevices(opts, "/job:localhost/replica:0/task:0", &devices));

  EXPECT_EQ(devices.size(), 3);

  auto device_type = DeviceType("MY_DEVICE");
  PlatformDeviceId plat_id;
  TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(device_type, TfDeviceId(0),
                                                     &plat_id));
  EXPECT_EQ(plat_id.value(), 0);

  TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(device_type, TfDeviceId(1),
                                                     &plat_id));
  EXPECT_EQ(plat_id.value(), 0);

  TF_ASSERT_OK(DeviceIdManager::TfToPlatformDeviceId(device_type, TfDeviceId(2),
                                                     &plat_id));
  EXPECT_EQ(plat_id.value(), 1);
}

}  // namespace
}  // namespace tensorflow
