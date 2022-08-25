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

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

GPUcudaMallocAllocator::GPUcudaMallocAllocator(
    PlatformDeviceId platform_device_id) {
  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .ValueOrDie();
}

void* GPUcudaMallocAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
  // allocate with cudaMalloc
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  CUdeviceptr rv = 0;
  CUresult res = cuMemAlloc(&rv, num_bytes);
  if (res != CUDA_SUCCESS) {
    const char* error_name;
    const char* error_string;
    cuGetErrorName(res, &error_name);
    cuGetErrorString(res, &error_string);
    LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes
               << "\n Error name: " << error_name
               << "\n Error string: " << error_string;
    return nullptr;
  }
  VLOG(10) << "AllocateRaw " << Name() << "  " << num_bytes << " "
           << reinterpret_cast<void*>(rv);
  return reinterpret_cast<void*>(rv);
#else
  return nullptr;
#endif  // GOOGLE_CUDA
}
void GPUcudaMallocAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  // free with cudaFree
  CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
  if (res == CUDA_ERROR_DEINITIALIZED) {
    // It happens with multi-GPU that TF free the GPU allocation after
    // the driver is unloaded. It is safe to ignore this error here.
    // cuGetErrorName and cuGetErrorString doesn't return any useful
    // information here.
    // TODO: Find how to fix the shutdown steps in TF.
    VLOG(1) << "Ignoring CUDA_ERROR_DEINITIALIZED Error";
  } else if (res != CUDA_SUCCESS) {
    const char* error_name;
    const char* error_string;
    cuGetErrorName(res, &error_name);
    cuGetErrorString(res, &error_string);
    LOG(ERROR) << "cuMemFree failed to free " << ptr
               << "\n Error name: " << error_name
               << "\n Error string: " << error_string;
  }
  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // GOOGLE_CUDA
}

bool GPUcudaMallocAllocator::TracksAllocationSizes() const { return false; }

}  // namespace tensorflow
