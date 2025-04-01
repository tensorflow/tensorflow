// Copyright 2024 The TensorFlow Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// this is a copy of ml_drift/cl/cl_device.cc
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace cl {

ClDevice::ClDevice(cl_device_id id, cl_platform_id platform_id)
    : id_(id), platform_id_(platform_id) {}

ClDevice::ClDevice(const ClDevice& device) = default;

ClDevice& ClDevice::operator=(const ClDevice& device) {
  if (this != &device) {
    id_ = device.id_;
    platform_id_ = device.platform_id_;
  }
  return *this;
}

ClDevice::ClDevice(ClDevice&& device)
    : id_(device.id_), platform_id_(device.platform_id_) {
  device.id_ = nullptr;
  device.platform_id_ = nullptr;
}

ClDevice& ClDevice::operator=(ClDevice&& device) {
  if (this != &device) {
    id_ = nullptr;
    platform_id_ = nullptr;
    std::swap(id_, device.id_);
    std::swap(platform_id_, device.platform_id_);
  }
  return *this;
}

absl::Status CreateDefaultGPUDevice(ClDevice* result) {
  cl_uint num_platforms;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }
  if (num_platforms == 0) {
    return absl::UnknownError("No supported OpenCL platform.");
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }

  cl_platform_id platform_id = platforms[0];
  cl_uint num_devices;
  status =
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }
  if (num_devices == 0) {
    return absl::UnknownError("No GPU on current platform.");
  }

  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }

  *result = ClDevice(devices[0], platform_id);
  LoadOpenCLFunctionExtensions(platform_id);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace litert
