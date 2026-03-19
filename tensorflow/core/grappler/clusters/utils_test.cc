/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/clusters/utils.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(UtilsTest, GetLocalGPUInfo) {
  GpuIdManager::TestOnlyReset();
#if GOOGLE_CUDA
  LOG(INFO) << "CUDA is enabled.";
  DeviceProperties properties;

  // Invalid platform GPU ID.
  properties = GetLocalGPUInfo(PlatformDeviceId(100));
  EXPECT_EQ("UNKNOWN", properties.type());

  // Succeed when a valid platform GPU id was inserted.
  properties = GetLocalGPUInfo(PlatformDeviceId(0));
  EXPECT_EQ("GPU", properties.type());
  EXPECT_EQ("NVIDIA", properties.vendor());
#elif TENSORFLOW_USE_ROCM
  LOG(INFO) << "ROCm is enabled.";
  DeviceProperties properties;

  // Invalid platform GPU ID.
  properties = GetLocalGPUInfo(PlatformDeviceId(100));
  EXPECT_EQ("UNKNOWN", properties.type());

  // Succeed when a valid platform GPU id was inserted.
  properties = GetLocalGPUInfo(PlatformDeviceId(0));
  EXPECT_EQ("GPU", properties.type());
  EXPECT_EQ("Advanced Micro Devices, Inc", properties.vendor());
#else
  LOG(INFO) << "CUDA is not enabled.";
  DeviceProperties properties;

  properties = GetLocalGPUInfo(PlatformDeviceId(0));
  EXPECT_EQ("GPU", properties.type());

  properties = GetLocalGPUInfo(PlatformDeviceId(100));
  EXPECT_EQ("GPU", properties.type());
#endif
}

TEST(UtilsTest, GetDeviceInfo) {
  GpuIdManager::TestOnlyReset();
  DeviceNameUtils::ParsedName device;
  DeviceProperties properties;

  // Invalid type.
  properties = GetDeviceInfo(device);
  EXPECT_EQ("UNKNOWN", properties.type());

  // Cpu info.
  device.type = "CPU";
  properties = GetDeviceInfo(device);
  EXPECT_EQ("CPU", properties.type());

  // No TF GPU id provided.
  device.type = "GPU";
  device.has_id = false;
  properties = GetDeviceInfo(device);
  EXPECT_EQ("GPU", properties.type());
#if GOOGLE_CUDA
  EXPECT_EQ("NVIDIA", properties.vendor());
#elif TENSORFLOW_USE_ROCM
  EXPECT_EQ("Advanced Micro Devices, Inc", properties.vendor());
#endif

  // TF to platform GPU id mapping entry doesn't exist.
  device.has_id = true;
  device.id = 0;
  properties = GetDeviceInfo(device);
  EXPECT_EQ("UNKNOWN", properties.type());

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // Invalid platform GPU id.
  TF_ASSERT_OK(GpuIdManager::InsertTfPlatformDeviceIdPair(
      TfDeviceId(0), PlatformDeviceId(100)));
  properties = GetDeviceInfo(device);
  EXPECT_EQ("UNKNOWN", properties.type());

  // Valid platform GPU id.
  TF_ASSERT_OK(GpuIdManager::InsertTfPlatformDeviceIdPair(TfDeviceId(1),
                                                          PlatformDeviceId(0)));
  device.id = 1;
  properties = GetDeviceInfo(device);
  EXPECT_EQ("GPU", properties.type());
#if GOOGLE_CUDA
  EXPECT_EQ("NVIDIA", properties.vendor());
#elif TENSORFLOW_USE_ROCM
  EXPECT_EQ("Advanced Micro Devices, Inc", properties.vendor());
#endif
#endif
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
