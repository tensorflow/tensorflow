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

// GPU (ROCm / CUDA) specific type handle resolution

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_

#if TENSORFLOW_USE_ROCM

#define __HIP_DISABLE_CPP_FUNCTIONS__

#include "rocm/include/hip/hip_complex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hiprand/hiprand.h"

#else  // CUDA

#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cuda.h"

// cannot include curand.h here
//   because it triggers the #error in cuda/cuda_gpu_executor.cc
//     (because curand.h includes cuda_runtime.h)
// so explicitly adding the lone typedef we need from that file
typedef struct curandGenerator_st* curandGenerator_t;

#endif

namespace stream_executor {
namespace gpu {

#if TENSORFLOW_USE_ROCM

using GpuContextHandle = hipCtx_t;
using GpuStreamHandle = hipStream_t;
using GpuEventHandle = hipEvent_t;
using GpuFunctionHandle = hipFunction_t;
using GpuFunctionAttribute = hipDeviceAttribute_t;  // not a typo!
using GpuDeviceHandle = hipDevice_t;
using GpuDevicePtr = hipDeviceptr_t;
using GpuDeviceAttribute = hipDeviceAttribute_t;
using GpuDeviceProperty = hipDeviceProp_t;
using GpuModuleHandle = hipModule_t;
using GpuStatus = hipError_t;
using GpuFuncCachePreference = hipFuncCache_t;
using GpuSharedMemConfig = hipSharedMemConfig;
using GpuComplexType = hipComplex;
using GpuDoubleComplexType = hipDoubleComplex;
using GpuRngHandle = hiprandGenerator_t;

#else  // CUDA

using GpuContextHandle = CUcontext;
using GpuStreamHandle = CUstream;
using GpuEventHandle = CUevent;
using GpuFunctionHandle = CUfunction;
using GpuFunctionAttribute = CUfunction_attribute;
using GpuDeviceHandle = CUdevice;
using GpuDevicePtr = CUdeviceptr;
using GpuDeviceAttribute = CUdevice_attribute;
using GpuDeviceProperty = CUdevprop;
using GpuModuleHandle = CUmodule;
using GpuStatus = CUresult;
using GpuFuncCachePreference = CUfunc_cache;
using GpuSharedMemConfig = CUsharedconfig;
using GpuComplexType = cuComplex;
using GpuDoubleComplexType = cuDoubleComplex;
using GpuRngHandle = curandGenerator_t;

#endif

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
