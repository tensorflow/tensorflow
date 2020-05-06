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

#include "tensorflow/core/common_runtime/composite_device.h"

#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

TEST(CompositeDeviceTest, Basic) {
  std::vector<string> underlying_devices;
  {
    Status status;
    std::unique_ptr<CompositeDevice> composite_device =
        CompositeDevice::MakeDevice(underlying_devices, /*unique_device_id=*/0,
                                    &status);
    EXPECT_EQ(composite_device, nullptr);
    EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
    EXPECT_TRUE(absl::StrContains(status.error_message(),
                                  "underlying_devices should not be empty"))
        << status.ToString();
  }

  {
    Status status;
    underlying_devices.push_back(
        "/job:localhost/replica:0/task:0/device:CPU:0");
    underlying_devices.push_back(
        "/job:localhost/replica:0/task:0/device:CPU:1");
    std::unique_ptr<CompositeDevice> composite_device =
        CompositeDevice::MakeDevice(underlying_devices, /*unique_device_id=*/0,
                                    &status);
    TF_ASSERT_OK(status);
    EXPECT_EQ(composite_device->device_type(), kCompositeDeviceType);
    EXPECT_EQ(underlying_devices, *composite_device->underlying_devices());
  }

  {
    Status status;
    underlying_devices.push_back(
        "/job:localhost/replica:0/task:0/device:CPU:0");
    std::unique_ptr<CompositeDevice> composite_device =
        CompositeDevice::MakeDevice(underlying_devices, /*unique_device_id=*/1,
                                    &status);
    EXPECT_EQ(composite_device, nullptr);
    EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
    EXPECT_TRUE(
        absl::StrContains(status.error_message(), "Got a duplicated device"))
        << status.ToString();
    underlying_devices.pop_back();
  }

  {
    Status status;
    underlying_devices.push_back(
        "/job:localhost/replica:0/task:0/device:GPU:0");
    std::unique_ptr<CompositeDevice> composite_device =
        CompositeDevice::MakeDevice(underlying_devices, /*unique_device_id=*/1,
                                    &status);
    EXPECT_EQ(composite_device, nullptr);
    EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
    EXPECT_TRUE(absl::StrContains(status.error_message(),
                                  "Expect device type CPU; but got type GPU"))
        << status.ToString();
  }
}

}  // namespace tensorflow
