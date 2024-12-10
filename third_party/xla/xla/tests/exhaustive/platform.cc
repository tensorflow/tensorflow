/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tests/exhaustive/platform.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"

namespace xla {
namespace exhaustive_op_test {

Platform::Value GetPlatformValue(const stream_executor::Platform& platform) {
  if (platform.Name() == "Host") {
// We process these copts in a library instead of the final exhaustive_xla_test
// target because we assume the final target will use the same target CPU arch
// as this target.
#ifdef __x86_64__
    return Platform::CpuValue::X86_64;
#endif
#ifdef __aarch64__
    return Platform::CpuValue::AARCH64;
#endif
  } else if (platform.Name() == "CUDA") {
    auto device_descriptor_status = platform.DescriptionForDevice(0);
    CHECK_OK(device_descriptor_status);
    std::unique_ptr<stream_executor::DeviceDescription> device_descriptor =
        std::move(*device_descriptor_status);

    auto cuda_compute_compatibility =
        device_descriptor->cuda_compute_capability();
    // If not available, CudaComputeCompatibility will have major version 0.
    if (cuda_compute_compatibility.IsAtLeast(1, 0)) {
      return cuda_compute_compatibility;
    }
  } else if (platform.Name() == "ROCM") {
    auto device_descriptor_status = platform.DescriptionForDevice(0);
    CHECK_OK(device_descriptor_status);
    std::unique_ptr<stream_executor::DeviceDescription> device_descriptor =
        std::move(*device_descriptor_status);

    auto rocm_compute_compatibility =
        device_descriptor->rocm_compute_capability();
    // If not available, RocmComputeCompatibility will be an invalid platform
    // value.
    if (rocm_compute_compatibility.gfx_version() == "gfx000") {
      return rocm_compute_compatibility;
    }
  }
  LOG(FATAL) << "Unhandled stream_executor::Platform: " << platform.Name()
             << ". Please add support to " __FILE__ ".";
}

bool Platform::IsNvidiaP100() const {
  return std::holds_alternative<stream_executor::CudaComputeCapability>(
             value_) &&
         !std::get<stream_executor::CudaComputeCapability>(value_).IsAtLeast(
             stream_executor::CudaComputeCapability::Volta());
}

bool Platform::IsNvidiaV100() const {
  return std::holds_alternative<stream_executor::CudaComputeCapability>(
             value_) &&
         std::get<stream_executor::CudaComputeCapability>(value_) ==
             stream_executor::CudaComputeCapability::Volta();
}

bool Platform::IsNvidiaA100() const {
  return std::holds_alternative<stream_executor::CudaComputeCapability>(
             value_) &&
         std::get<stream_executor::CudaComputeCapability>(value_) ==
             stream_executor::CudaComputeCapability::Ampere();
}

bool Platform::IsNvidiaH100() const {
  return std::holds_alternative<stream_executor::CudaComputeCapability>(
             value_) &&
         std::get<stream_executor::CudaComputeCapability>(value_) ==
             stream_executor::CudaComputeCapability::Hopper();
}

Platform::Platform(const stream_executor::Platform& platform)
    : value_(GetPlatformValue(platform)) {}

}  // namespace exhaustive_op_test
}  // namespace xla
