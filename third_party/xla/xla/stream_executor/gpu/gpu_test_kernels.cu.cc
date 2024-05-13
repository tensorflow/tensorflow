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

#include "xla/stream_executor/gpu/gpu_test_kernels.h"

#include <cstdint>

namespace stream_executor::gpu::internal {

__global__ void AddI32(int32_t* a, int32_t* b, int32_t* c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

__global__ void MulI32(int32_t* a, int32_t* b, int32_t* c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] * b[index];
}

__global__ void IncAndCmp(int32_t* counter, bool* pred, int32_t value) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  pred[index] = counter[index] < value;
  counter[index] += 1;
}

__global__ void AddI32Ptrs3(Ptrs3<int32_t> ptrs) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  ptrs.c[index] = ptrs.a[index] + ptrs.b[index];
}

void* GetAddI32Kernel() { return reinterpret_cast<void*>(&AddI32); }

void* GetMulI32Kernel() { return reinterpret_cast<void*>(&MulI32); }

void* GetIncAndCmpKernel() { return reinterpret_cast<void*>(&IncAndCmp); }

void* GetAddI32Ptrs3Kernel() { return reinterpret_cast<void*>(&AddI32Ptrs3); }

}  // namespace stream_executor::gpu::internal
