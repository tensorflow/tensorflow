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

#include "xla/service/gpu/sleep_kernel.h"

namespace xla::gpu {
namespace {

// Use busy waiting instead of __nanosleep() to make the code more portable
// (__nanosleep requires __CUDA_ARCH__ >= 700)
__global__ void sleep(int64_t num_clocks) {
  int64_t start = clock64();
  while (clock64() - start < num_clocks) continue;
}

}  // namespace

void* GetSleepKernel() { return reinterpret_cast<void*>(&sleep); }

}  // namespace xla::gpu
