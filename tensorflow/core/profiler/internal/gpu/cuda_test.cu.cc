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

// Creates some GPU activity to test functionalities of gpuperfcounter/gputrace.
#include "tensorflow/core/profiler/internal/gpu/cuda_test.h"

#if GOOGLE_CUDA
#include <stdio.h>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#endif

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
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

}  // namespace test
}  // namespace profiler
}  // namespace tensorflow
