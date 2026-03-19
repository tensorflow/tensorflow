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

#include "xla/stream_executor/rocm/topk_kernel_rocm_common.cu.h"

namespace stream_executor::rocm {

REGISTER_TOPK_KERNEL(1, float, uint16_t);
REGISTER_TOPK_KERNEL(2, float, uint16_t);
REGISTER_TOPK_KERNEL(4, float, uint16_t);
REGISTER_TOPK_KERNEL(8, float, uint16_t);
REGISTER_TOPK_KERNEL(16, float, uint16_t);

REGISTER_TOPK_KERNEL(1, float, uint32_t);
REGISTER_TOPK_KERNEL(2, float, uint32_t);
REGISTER_TOPK_KERNEL(4, float, uint32_t);
REGISTER_TOPK_KERNEL(8, float, uint32_t);
REGISTER_TOPK_KERNEL(16, float, uint32_t);

}  // namespace stream_executor::rocm
