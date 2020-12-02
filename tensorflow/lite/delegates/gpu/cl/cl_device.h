/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_DEVICE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_DEVICE_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

// A wrapper around opencl device id
class CLDevice {
 public:
  CLDevice() = default;
  CLDevice(cl_device_id id, cl_platform_id platform_id);

  CLDevice(CLDevice&& device);
  CLDevice& operator=(CLDevice&& device);
  CLDevice(const CLDevice&);
  CLDevice& operator=(const CLDevice&);

  ~CLDevice() {}

  cl_device_id id() const { return id_; }
  cl_platform_id platform() const { return platform_id_; }
  std::string GetPlatformVersion() const;

  // To track bug on some Adreno. b/131099086
  void DisableOneLayerTextureArray();

  const GpuInfo& GetInfo() const { return info_; }
  // We update device info during context creation, so as supported texture
  // formats can be requested from context only.
  mutable GpuInfo info_;

 private:
  cl_device_id id_ = nullptr;
  cl_platform_id platform_id_ = nullptr;
};

absl::Status CreateDefaultGPUDevice(CLDevice* result);

template <typename T>
T GetDeviceInfo(cl_device_id id, cl_device_info info) {
  T result;
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS) {
    return -1;
  }
  return result;
}

template <typename T>
absl::Status GetDeviceInfo(cl_device_id id, cl_device_info info, T* result) {
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), result, nullptr);
  if (error != CL_SUCCESS) {
    return absl::InvalidArgumentError(CLErrorCodeToString(error));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_DEVICE_H_
