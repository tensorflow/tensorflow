/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_DEVICE_INFO_FOR_TESTS_H_
#define XLA_SERVICE_GPU_GPU_DEVICE_INFO_FOR_TESTS_H_

#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class TestGpuDeviceInfo {
 public:
  static stream_executor::DeviceDescription RTXA6000DeviceInfo(
      stream_executor::GpuComputeCapability cc =
          stream_executor::CudaComputeCapability(8, 9));
  static stream_executor::DeviceDescription AMDMI210DeviceInfo();
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_DEVICE_INFO_FOR_TESTS_H_
