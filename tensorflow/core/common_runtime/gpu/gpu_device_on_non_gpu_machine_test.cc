/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TEST(GPUDeviceOnNonGPUMachineTest, CreateGPUDevicesOnNonGPUMachine) {
  SessionOptions opts;
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, "/job:localhost/replica:0/task:0", &devices));
  EXPECT_TRUE(devices.empty());
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

int main(int argc, char** argv) {
#if GOOGLE_CUDA
  // Sets CUDA_VISIBLE_DEVICES to empty string to simulate non-gpu environment.
  setenv("CUDA_VISIBLE_DEVICES", "", 1);
#endif  // GOOGLE_CUDA
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
