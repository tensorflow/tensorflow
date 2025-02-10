// Copyright 2024 The ML Drift Authors.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_DEVICE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_DEVICE_H_

#include <string>

#include "absl/status/status.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace litert {
namespace cl {

// A wrapper around opencl device id
class ClDevice {
 public:
  ClDevice() = default;
  ClDevice(cl_device_id id, cl_platform_id platform_id);

  ClDevice(ClDevice&& device);
  ClDevice& operator=(ClDevice&& device);
  ClDevice(const ClDevice&);
  ClDevice& operator=(const ClDevice&);

  ~ClDevice() = default;

  cl_device_id id() const { return id_; }
  cl_platform_id platform() const { return platform_id_; }
  std::string GetPlatformVersion() const;

 private:
  cl_device_id id_ = nullptr;
  cl_platform_id platform_id_ = nullptr;
};

absl::Status CreateDefaultGPUDevice(ClDevice* result);

template <typename T>
T GetDeviceInfo(cl_device_id id, cl_device_info info) {
  T result;
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS) {
    return {};
  }
  return result;
}

template <typename T>
absl::Status GetDeviceInfo(cl_device_id id, cl_device_info info, T* result) {
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), result, nullptr);
  if (error != CL_SUCCESS) {
    return absl::InvalidArgumentError("cl error:" + std::to_string(error));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_DEVICE_H_
