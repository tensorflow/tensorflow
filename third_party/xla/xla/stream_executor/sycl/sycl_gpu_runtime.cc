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

#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <cassert>
#include <iostream>
#include <unordered_map>

#include "absl/base/call_once.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/util/env_var.h"

namespace stream_executor::sycl {

namespace {

absl::Status IsValidDeviceOrdinal(int device_ordinal,
                                  const absl::string_view& function_name) {
  TF_ASSIGN_OR_RETURN(int device_count, SyclDevicePool::GetDeviceCount());
  if (device_ordinal >= 0 && device_ordinal < device_count) {
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        function_name, ": Invalid device ordinal: ", device_ordinal));
  }
}

}  // namespace

DevicePool SyclDevicePool::device_pool_;

absl::Status SyclDevicePool::InitDevicePool() {
  static absl::once_flag device_init_flag;
  static absl::Status init_status = absl::OkStatus();
  absl::call_once(device_init_flag, []() {
    DevicePool devices;
    std::vector<::sycl::platform> platform_list =
        ::sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      std::string platform_name =
          platform.get_info<::sycl::info::platform::name>();
      // Add all Level-Zero backend GPUs to the device pool so that it can be
      // used by the SYCL runtime.
      if (platform_name.find("Level-Zero") != std::string::npos) {
        LOG(INFO) << "Selected platform: " << platform_name;
        std::vector<::sycl::device> device_list = platform.get_devices();
        for (const auto& device : device_list) {
          if (device.is_gpu()) {
            devices.push_back(device);
          }
        }
      }
    }
    if (devices.empty()) {
      init_status = absl::InternalError(
          "SyclDevicePool::InitDevicePool: No SYCL devices found with "
          "Level-Zero "
          "backend. Check oneAPI installation and environment variables.");
      return;
    }
    device_pool_ = std::move(devices);
  });
  return init_status;
}

absl::StatusOr<::sycl::context> SyclDevicePool::GetDeviceContext() {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  static ::sycl::context device_context(device_pool_);
  return device_context;
}

absl::StatusOr<int> SyclDevicePool::GetDeviceCount() {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  // Cast to int since device_ordinal is usually an int.
  return static_cast<int>(device_pool_.size());
}

absl::StatusOr<int> SyclDevicePool::GetDeviceOrdinal(
    const ::sycl::device& device) {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  auto it = std::find(device_pool_.begin(), device_pool_.end(), device);
  if (it != device_pool_.end()) {
    return static_cast<int>(it - device_pool_.begin());
  } else {
    return absl::InternalError(
        "SyclDevicePool::GetDeviceOrdinal failed, got invalid device");
  }
}

absl::StatusOr<::sycl::device> SyclDevicePool::GetDevice(int device_ordinal) {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  TF_RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclDevicePool::GetDevice"));
  return device_pool_[device_ordinal];
}

}  // namespace stream_executor::sycl
