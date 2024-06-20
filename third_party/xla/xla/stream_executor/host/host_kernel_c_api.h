/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_C_API_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_C_API_H_

#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// StreamExecutor Host Kernel API
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif

// StreamExecutor host kernel API is an integration point between a codegen
// backend and a runtime. XLA:CPU backend compiles fusion regions to native
// functions (via LLVM backend) that are compatible with a kernel API (and ABI),
// and the runtime is simply invoking them with user buffers and orchestrates
// multi-threaded execution.

// WARNING: This API does not provide any backward compatibility guarantees as
// today XLA:CPU backend is statically linked and we do not plan to load
// kernels from dynamic libraries. It's defined as C API because we have to
// match it in the codegen backend (built on top of LLVM) and C structs have
// trivial layout that can be expressed as llvm stuct (*).
//
// (*) https://llvm.org/docs/LangRef.html#structure-types

// Similar to a Gpu backend an XLA:CPU compiler generates a tiled function from
// an HLO fusion where each tile is responsible for computing a part of the
// output. It's up to compiler to chose the tiling strategy, from StreamExecutor
// perspective it's simply an iteration space where each task is independent and
// can be executed concurrently.
typedef struct SE_HOST_KernelDim3 {
  uint64_t x;
  uint64_t y;
  uint64_t z;
} SE_HOST_KernelDim3;

// Kernel grid size roughly corresponds to a CUDA block size.
typedef struct SE_HOST_KernelDim3 SE_HOST_KernelThreadDim;

// Kernel grid coordinate roughly corresponds to a CUDA block, with an
// assumption that all kernel invocations can run concurrently.
typedef struct SE_HOST_KernelDim3 SE_HOST_KernelThread;

// A CPU kernel argument that corresponds to se::DeviceMemoryBase.
typedef struct SE_HOST_KernelArg {
  void* data;
  size_t size;
} SE_HOST_KernelArg;

// A CPU kernel call frame.
typedef struct SE_HOST_KernelCallFrame {
  SE_HOST_KernelThreadDim* thread_dims;
  SE_HOST_KernelThread* thread;

  size_t num_args;
  const SE_HOST_KernelArg* args;
} SE_HOST_KernelCallFrame;

// Error reporting for host kernels. NULL means success.
typedef struct SE_HOST_KernelError SE_HOST_KernelError;

// Host kernel API.
typedef SE_HOST_KernelError* SE_HOST_Kernel(
    const SE_HOST_KernelCallFrame* call_frame);

#ifdef __cplusplus
}
#endif

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_KERNEL_C_API_H_
