/* Copyright 2021 The OpenXLA Authors.

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

// Creates some GPU activity to test functionalities of gpuperfcounter/gputrace.
#include "xla/backends/profiler/gpu/cuda_test.h"

#include <vector>

#if GOOGLE_CUDA
#include <stdio.h>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#endif

#include "tsl/platform/test.h"

namespace xla {
namespace profiler {
namespace test {

#if GOOGLE_CUDA
namespace {

// Simple printf kernel.
__global__ void simple_print() { printf("hello, world!\n"); }

// Empty kernel.
__global__ void empty() {}

// Simple kernel accesses memory.
__global__ void access(int *addr) { *addr = *addr * 2; }

unsigned *g_device_copy;

unsigned *gpu0_buf, *gpu1_buf;

}  // namespace
#endif  // GOOGLE_CUDA

void PrintfKernel(int iters) {
#if GOOGLE_CUDA
  for (int i = 0; i < iters; ++i) {
    simple_print<<<1, 1>>>();
  }
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void EmptyKernel(int iters) {
#if GOOGLE_CUDA
  for (int i = 0; i < iters; ++i) {
    empty<<<1, 1>>>();
  }
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void AccessKernel(int *addr) {
#if GOOGLE_CUDA
  access<<<1, 1>>>(addr);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void Synchronize() {
#if GOOGLE_CUDA
  cudaDeviceSynchronize();
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void UnifiedMemoryHtoDAndDtoH() {
#if GOOGLE_CUDA
  int *addr = nullptr;
  cudaMallocManaged(reinterpret_cast<void **>(&addr), sizeof(int));
  // The page is now in host memory.
  *addr = 1;
  // The kernel wants to access the page. HtoD transfer happens.
  AccessKernel(addr);
  Synchronize();
  // The page is now in device memory. CPU wants to access the page. DtoH
  // transfer happens.
  EXPECT_EQ(*addr, 2);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyH2D() {
#if GOOGLE_CUDA
  unsigned host_val = 0x12345678;
  cudaMalloc(reinterpret_cast<void **>(&g_device_copy), sizeof(unsigned));
  cudaMemcpy(g_device_copy, &host_val, sizeof(unsigned),
             cudaMemcpyHostToDevice);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyH2D_Async() {
#if GOOGLE_CUDA
  unsigned host_val = 0x12345678;
  cudaMalloc(reinterpret_cast<void **>(&g_device_copy), sizeof(unsigned));
  cudaMemcpyAsync(g_device_copy, &host_val, sizeof(unsigned),
                  cudaMemcpyHostToDevice);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyD2H() {
#if GOOGLE_CUDA
  unsigned host_val = 0;
  cudaMalloc(reinterpret_cast<void **>(&g_device_copy), sizeof(unsigned));
  cudaMemcpy(&host_val, g_device_copy, sizeof(unsigned),
             cudaMemcpyDeviceToHost);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

namespace {

// Helper function to set up memory buffers on two devices.
void P2PMemcpyHelper() {
#if GOOGLE_CUDA
  cudaSetDevice(0);
  cudaMalloc(reinterpret_cast<void **>(&gpu0_buf), sizeof(unsigned));
  cudaDeviceEnablePeerAccess(/*peerDevice=*/1, /*flags=*/0);
  cudaSetDevice(1);
  cudaMalloc(reinterpret_cast<void **>(&gpu1_buf), sizeof(unsigned));
  cudaDeviceEnablePeerAccess(/*peerDevice=*/0, /*flags=*/0);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

}  // namespace

bool MemCopyP2PAvailable() {
#if GOOGLE_CUDA
  int can_access_01 = 0;
  cudaDeviceCanAccessPeer(&can_access_01, /*device=*/0, /*peerDevice=*/1);
  int can_access_10 = 0;
  cudaDeviceCanAccessPeer(&can_access_01, /*device=*/1, /*peerDevice=*/0);
  return can_access_01 && can_access_10;
#else
  return false;
#endif
}

void MemCopyP2PImplicit() {
#if GOOGLE_CUDA
  P2PMemcpyHelper();
  cudaMemcpy(gpu1_buf, gpu0_buf, sizeof(unsigned), cudaMemcpyDefault);
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

void MemCopyP2PExplicit() {
#if GOOGLE_CUDA
  P2PMemcpyHelper();
  cudaMemcpyPeer(gpu1_buf, 1 /* device */, gpu0_buf, 0 /* device */,
                 sizeof(unsigned));
#else
  GTEST_FAIL() << "Build with --config=cuda";
#endif
}

#if GOOGLE_CUDA

// The test about cuda graph is based on Nvidia's CUPTI sample code
// under extras/CUPTI/samples/cuda_graphs_trace/ dir of CUDA distribution.
__global__ void VecAdd(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void VecSub(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}

void CudaGraphCreateAndExecute() {
#if CUDA_VERSION >= 11070  // CUDA 11.7
  constexpr size_t kNumElements = 2048;
  constexpr size_t kNumBytes = kNumElements * sizeof(int);
  constexpr int kThreadsPerBlock = 256;
  int blocks_per_grid = 0;

  cudaStream_t stream = nullptr;
  cudaKernelNodeParams kernel_params;
  cudaMemcpy3DParms memcpy_params = {nullptr};
  cudaGraph_t graph;
  cudaGraph_t cloned_graph;
  cudaGraphExec_t graph_exec;
  cudaGraphNode_t nodes[5];

  // Allocates input/output vectors in host memory.
  std::vector<int> vec_a(kNumElements);
  std::vector<int> vec_b(kNumElements);
  std::vector<int> vec_c(kNumElements);

  // Allocates vectors in device memory.
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, kNumBytes);
  cudaMalloc((void **)&d_b, kNumBytes);
  cudaMalloc((void **)&d_c, kNumBytes);

  cudaGraphCreate(&graph, 0);

  // Init memcpy params.
  memcpy_params.kind = cudaMemcpyHostToDevice;
  memcpy_params.srcPtr.ptr = vec_a.data();
  memcpy_params.dstPtr.ptr = d_a;
  memcpy_params.extent.width = kNumBytes;
  memcpy_params.extent.height = 1;
  memcpy_params.extent.depth = 1;
  cudaGraphAddMemcpyNode(&nodes[0], graph, nullptr, 0, &memcpy_params);

  memcpy_params.srcPtr.ptr = vec_b.data();
  memcpy_params.dstPtr.ptr = d_b;
  cudaGraphAddMemcpyNode(&nodes[1], graph, nullptr, 0, &memcpy_params);

  // Init kernel params.
  int num = kNumElements;
  void *kernelArgs[] = {(void *)&d_a, (void *)&d_b, (void *)&d_c, (void *)&num};
  blocks_per_grid = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_params.func = (void *)VecAdd;
  kernel_params.gridDim = dim3(blocks_per_grid, 1, 1);
  kernel_params.blockDim = dim3(kThreadsPerBlock, 1, 1);
  kernel_params.sharedMemBytes = 0;
  kernel_params.kernelParams = (void **)kernelArgs;
  kernel_params.extra = nullptr;

  cudaGraphAddKernelNode(&nodes[2], graph, &nodes[0], 2, &kernel_params);

  kernel_params.func = (void *)VecSub;
  cudaGraphAddKernelNode(&nodes[3], graph, &nodes[2], 1, &kernel_params);

  memcpy_params.kind = cudaMemcpyDeviceToHost;
  memcpy_params.srcPtr.ptr = d_c;
  memcpy_params.dstPtr.ptr = vec_c.data();
  memcpy_params.extent.width = kNumBytes;
  memcpy_params.extent.height = 1;
  memcpy_params.extent.depth = 1;
  cudaGraphAddMemcpyNode(&nodes[4], graph, &nodes[3], 1, &memcpy_params);

  cudaGraphClone(&cloned_graph, graph);

  cudaGraphInstantiate(&graph_exec, cloned_graph, nullptr, nullptr, 0);

  cudaGraphLaunch(graph_exec, stream);

  cudaStreamSynchronize(stream);

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
#endif  // CUDA_VERSION >= 11070
}

#else

void CudaGraphCreateAndExecute() { GTEST_FAIL() << "Build with --config=cuda"; }

#endif

bool IsCudaNewEnoughForGraphTraceTest() {
#if CUDA_VERSION >= 11070  // CUDA 11.7
  return true;
#else
  return false;
#endif  // CUDA_VERSION >= 11070
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
