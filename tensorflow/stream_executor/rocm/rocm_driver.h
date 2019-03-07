/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// ROCM userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_

#define TENSORFLOW_ROCM_USE_DYNAMIC_LINKING
#include "tensorflow/stream_executor/gpu/gpu_driver.h"


namespace stream_executor {
namespace gpu {

// GpuContext wraps the device_ordinal.
// Only reason we need this wrapper class is to make the GpuDriver* API
class GpuContext {
 public:
  GpuContext(const int v) : device_ordinal_(v) {}

  int device_ordinal() const { return device_ordinal_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  const int device_ordinal_;
};

}  // namespace gpu

namespace rocm {

using MemorySpace = gpu::MemorySpace;

using ROCMDriver = gpu::GpuDriver;

using ScopedActivateContext = gpu::ScopedActivateContext;

using ROCMContext = gpu::GpuContext;

}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
