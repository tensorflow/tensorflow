/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_DEVICE_SPEC_H_
#define XLA_CODEGEN_DEVICE_SPEC_H_

#include <variant>

#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

struct CpuDeviceSpec {};
using DeviceSpecType =
    std::variant<stream_executor::DeviceDescription, CpuDeviceSpec>;

// Used by codegen passes that can target different targets, namely CPU or GPU.
class DeviceSpec {
 public:
  DeviceSpec() = default;
  explicit DeviceSpec(
      const stream_executor::DeviceDescription& gpu_device_description)
      : type_(gpu_device_description) {}
  explicit DeviceSpec(const CpuDeviceSpec& cpu_device_spec)
      : type_(cpu_device_spec) {}

  const DeviceSpecType& type() const { return type_; }
  DeviceSpecType* mutable_type() { return &type_; }

  const stream_executor::DeviceDescription& gpu() const {
    CHECK(IsGpu());
    return std::get<stream_executor::DeviceDescription>(type_);
  }

  bool IsCpu() const { return std::holds_alternative<CpuDeviceSpec>(type_); }
  bool IsGpu() const {
    return std::holds_alternative<stream_executor::DeviceDescription>(type_);
  }
  bool IsAmdGpu() const {
    return IsGpu() &&
           std::holds_alternative<stream_executor::RocmComputeCapability>(
               gpu().gpu_compute_capability());
  }
  bool IsNvidiaGpu() const {
    return IsGpu() &&
           std::holds_alternative<stream_executor::CudaComputeCapability>(
               gpu().gpu_compute_capability());
  }

 private:
  DeviceSpecType type_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_DEVICE_SPEC_H_
