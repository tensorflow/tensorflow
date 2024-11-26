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

#ifndef XLA_BACKENDS_CPU_RUNTIME_KERNEL_C_API_H_
#define XLA_BACKENDS_CPU_RUNTIME_KERNEL_C_API_H_

#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// CPU Kernel API
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif

// CPU kernel API is an integration point between a codegen
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
// output. It's up to compiler to chose the tiling strategy, from CPU runtime
// perspective it's simply an iteration space where each task is independent and
// can be executed concurrently.
typedef struct XLA_CPU_KernelDim3 {
  uint64_t x;
  uint64_t y;
  uint64_t z;
} XLA_CPU_KernelDim3;

// Kernel grid size roughly corresponds to a CUDA block size.
typedef struct XLA_CPU_KernelDim3 XLA_CPU_KernelThreadDim;

// Kernel grid coordinate roughly corresponds to a CUDA block, with an
// assumption that all kernel invocations can run concurrently.
typedef struct XLA_CPU_KernelDim3 XLA_CPU_KernelThread;

// A CPU kernel argument that corresponds to se::DeviceMemoryBase.
typedef struct XLA_CPU_KernelArg {
  void* data;
  size_t size;
} XLA_CPU_KernelArg;

// A CPU kernel call frame.
typedef struct XLA_CPU_KernelCallFrame {
  const XLA_CPU_KernelThreadDim* thread_dims;
  const XLA_CPU_KernelThread* thread;

  size_t num_args;
  const XLA_CPU_KernelArg* args;
} XLA_CPU_KernelCallFrame;

// Error reporting for host kernels. NULL means success.
typedef struct XLA_CPU_KernelError XLA_CPU_KernelError;

// Host kernel API.
typedef XLA_CPU_KernelError* XLA_CPU_Kernel(
    const XLA_CPU_KernelCallFrame* call_frame);

#ifdef __cplusplus
}
#endif

#endif  // XLA_BACKENDS_CPU_RUNTIME_KERNEL_C_API_H_
