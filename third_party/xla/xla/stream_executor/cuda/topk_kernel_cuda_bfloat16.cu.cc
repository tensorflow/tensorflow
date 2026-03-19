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

#include <cstdint>

#include "xla/stream_executor/cuda/topk_kernel_cuda_common.cu.h"
#include "xla/types.h"

namespace stream_executor::cuda {

using xla::bfloat16;

REGISTER_TOPK_KERNEL(1, bfloat16, uint16_t);
REGISTER_TOPK_KERNEL(2, bfloat16, uint16_t);
REGISTER_TOPK_KERNEL(4, bfloat16, uint16_t);
REGISTER_TOPK_KERNEL(8, bfloat16, uint16_t);
REGISTER_TOPK_KERNEL(16, bfloat16, uint16_t);

REGISTER_TOPK_KERNEL(1, bfloat16, uint32_t);
REGISTER_TOPK_KERNEL(2, bfloat16, uint32_t);
REGISTER_TOPK_KERNEL(4, bfloat16, uint32_t);
REGISTER_TOPK_KERNEL(8, bfloat16, uint32_t);
REGISTER_TOPK_KERNEL(16, bfloat16, uint32_t);

}  // namespace stream_executor::cuda
