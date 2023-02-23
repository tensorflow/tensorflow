/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <hip/hip_runtime.h>

#include <limits>
namespace stream_executor {
namespace gpu {

__global__ void rocm_Broadcast_fp32Kernel(float* dst, int dst_stride, int batches,
                                         float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride*2;
  float* dst4 = dst + dst_stride*3;
  bool b2 = (blockIdx.y * 4 + 1 < batches);
  bool b3 = (blockIdx.y * 4 + 2 < batches);
  bool b4 = (blockIdx.y * 4 + 3 < batches);
  for (int i=threadIdx.x + blockIdx.x * 256; i<size; i+=blockDim.x * gridDim.x){
  	dst[i] = src[i];
  	if (b2){
  		dst2[i] = src[i];
    }
  	if (b3){
  		dst3[i] = src[i];
    }
  	if (b4){
  		dst4[i] = src[i];
    }
  }
}

void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                        int src_batches, float* src, int size) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_fp32Kernel, dim3(x_blocks, (batches+3)/4, src_batches), 
                     min(256, (int)size), 0, (hipStream_t)stream, dst, dst_stride, 
                     batches, src, size);
}

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//
__global__ void __xla_MakeBatchPointers(char* base, int stride, int n,
                                        void** ptrs_out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  ptrs_out[idx] = base + idx * stride;
}

void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                            void** ptrs_out) {
  const int threads_per_block = 256;
  hipLaunchKernelGGL(
      __xla_MakeBatchPointers,
      dim3((n + threads_per_block - 1) / threads_per_block, 1, 1),
      dim3(threads_per_block, 1, 1), 0, (hipStream_t)stream, base, stride, n,
      ptrs_out);
}

};  // namespace gpu
};  // namespace stream_executor
