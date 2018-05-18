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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const string& name)
    : GPUBFCAllocator(cuda_gpu_id, total_memory, GPUOptions(), name) {}

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name)
    : BFCAllocator(
          new GPUMemAllocator(
              GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie()),
          total_memory, gpu_options.allow_growth(), name) {}

}  // namespace tensorflow
