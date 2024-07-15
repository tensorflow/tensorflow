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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_TEST_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_TEST_H_

namespace xla {
namespace profiler {
namespace test {
// Calls a function on the device to print a string as many times as indicated
// by iters.
void PrintfKernel(int iters = 1);

// Calls an empty kernel (named "empty") on the device as many times as
// indicated by iters.
void EmptyKernel(int iters = 1);

// Waits for device activity to complete.
void Synchronize();

// Copies a few bytes of memory from host to device.
void MemCopyH2D();

// Copies a few bytes of memory from device to host, asynchronously.
void MemCopyH2D_Async();

// Copies a few bytes of memory from device to host.
void MemCopyD2H();

// Returns true if it s possible to copy bytes from device 0 to device 1.
bool MemCopyP2PAvailable();

// Copies a few bytes of memory from device 0 to device 1.
void MemCopyP2PImplicit();

// Copies a few bytes of memory from device 0 to device 1.
void MemCopyP2PExplicit();

// Creates a simple cuda graph, clones it, instantiates it and executes it.
void CudaGraphCreateAndExecute();

// Return true if the cuda version is new enough so cuda graph activity trace
// is supported by its CUPTI.
bool IsCudaNewEnoughForGraphTraceTest();

}  // namespace test
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_TEST_H_
