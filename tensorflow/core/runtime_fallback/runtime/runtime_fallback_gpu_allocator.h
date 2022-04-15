/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_GPU_ALLOCATOR_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_GPU_ALLOCATOR_H_

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tfrt/gpu/device/gpu_config.h"  // from @tf_runtime

namespace tensorflow {

tfrt::gpu::GpuAllocatorFactory CreateRuntimeFallbackGpuAllocatorFactory(
    tensorflow::Allocator* tf_gpu_allocator);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_RUNTIME_FALLBACK_GPU_ALLOCATOR_H_
