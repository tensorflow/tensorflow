/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/topk_kernel.cu.h"
#include "xla/types.h"

namespace xla::gpu {

template void* GetTopKKernelForK<bfloat16, 1>(int n);
template void* GetTopKKernelForK<bfloat16, 2>(int n);
template void* GetTopKKernelForK<bfloat16, 4>(int n);
template void* GetTopKKernelForK<bfloat16, 8>(int n);
template void* GetTopKKernelForK<bfloat16, 16>(int n);

template int32_t GetTopKWaveFrontSize<bfloat16>();

}  // namespace xla::gpu
