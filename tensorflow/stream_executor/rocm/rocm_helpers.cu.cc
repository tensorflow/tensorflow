#include <hip/hip_runtime.h>
#include <limits>
namespace stream_executor {
namespace gpu {

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//

__global__ void __xla_MakeBatchPointers(char* base, int stride, int n, void** ptrs_out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  ptrs_out[idx] = base + idx * stride;
}

void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n, void** ptrs_out) {
  const int threads_per_block = 256;
  hipLaunchKernelGGL(__xla_MakeBatchPointers, dim3((n + threads_per_block - 1)/threads_per_block, 1, 1),
                     dim3(threads_per_block, 1, 1), 0, (hipStream_t)stream, base, stride, n, ptrs_out);
}

};  // namespace gpu
};  // namespace stream_executor
