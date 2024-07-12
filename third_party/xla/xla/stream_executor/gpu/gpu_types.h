/* Copyright 2019 The OpenXLA Authors.

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

// GPU (SYCL / ROCm / CUDA) specific type handle resolution

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_

#if TENSORFLOW_USE_SYCL

#include "sycl/sycl.hpp"

#elif TENSORFLOW_USE_ROCM

#define __HIP_DISABLE_CPP_FUNCTIONS__

#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hiprand/hiprand.h"

#else  // CUDA

#include "third_party/gpus/cuda/include/cuda.h"

#endif

namespace stream_executor {
namespace gpu {

// An empty struct to be used as a handle for all unsupported features in
// current CUDA/HIP/SYCL version.
struct UnsupportedGpuFeature {};

#if TENSORFLOW_USE_SYCL

using GpuStreamHandle = ::sycl::queue*;
using GpuEventHandle = ::sycl::event*;
using GpuFunctionHandle = ::sycl::kernel*;
using GpuFunctionAttribute = UnsupportedGpuFeature;
using GpuDeviceHandle = ::sycl::device*;
using GpuDevicePtr = void*;
using GpuDeviceAttribute = UnsupportedGpuFeature;
using GpuDeviceProperty = UnsupportedGpuFeature;
using GpuModuleHandle = ze_module_handle_t;
using GpuFuncCachePreference = UnsupportedGpuFeature;
using GpuSharedMemConfig = UnsupportedGpuFeature;
using GpuRngHandle = UnsupportedGpuFeature;
using GpuGraphHandle = UnsupportedGpuFeature;
using GpuGraphExecHandle = UnsupportedGpuFeature;
using GpuGraphNodeHandle = UnsupportedGpuFeature;
using GpuGraphConditionalHandle = UnsupportedGpuFeature;

#elif TENSORFLOW_USE_ROCM

using GpuStreamHandle = hipStream_t;
using GpuEventHandle = hipEvent_t;
using GpuFunctionHandle = hipFunction_t;
using GpuFunctionAttribute = hipFunction_attribute;
using GpuDeviceHandle = hipDevice_t;
using GpuDevicePtr = hipDeviceptr_t;
using GpuDeviceAttribute = hipDeviceAttribute_t;
using GpuDeviceProperty = hipDeviceProp_t;
using GpuModuleHandle = hipModule_t;
using GpuFuncCachePreference = hipFuncCache_t;
using GpuSharedMemConfig = hipSharedMemConfig;
using GpuRngHandle = hiprandGenerator_t;
using GpuGraphHandle = hipGraph_t;
using GpuGraphExecHandle = hipGraphExec_t;
using GpuGraphNodeHandle = hipGraphNode_t;
using GpuGraphConditionalHandle = UnsupportedGpuFeature;
#else  // CUDA

using GpuStreamHandle = CUstream;
using GpuEventHandle = CUevent;
using GpuFunctionHandle = CUfunction;
using GpuFunctionAttribute = CUfunction_attribute;
using GpuDeviceHandle = CUdevice;
using GpuDevicePtr = CUdeviceptr;
using GpuDeviceAttribute = CUdevice_attribute;
using GpuDeviceProperty = CUdevprop;
using GpuModuleHandle = CUmodule;
using GpuFuncCachePreference = CUfunc_cache;
using GpuSharedMemConfig = CUsharedconfig;
using GpuGraphHandle = CUgraph;
using GpuGraphExecHandle = CUgraphExec;
using GpuGraphNodeHandle = CUgraphNode;

#if CUDA_VERSION >= 12030
using GpuGraphConditionalHandle = CUgraphConditionalHandle;
#else
using GpuGraphConditionalHandle = UnsupportedGpuFeature;
#endif  // #if CUDA_VERSION >= 12030

#endif

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
