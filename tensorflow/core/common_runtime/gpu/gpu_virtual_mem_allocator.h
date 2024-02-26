/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VIRTUAL_MEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VIRTUAL_MEM_ALLOCATOR_H_

#include "xla/stream_executor/integrations/gpu_virtual_mem_allocator.h"  // IWYU pragma: keep

#if GOOGLE_CUDA
namespace tensorflow {
using stream_executor::GpuVirtualMemAllocator;
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VIRTUAL_MEM_ALLOCATOR_H_
