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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PRECOMPILED_KERNELS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PRECOMPILED_KERNELS_H_

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/stream.h"

// Hardcoded GPU kernels for various simple tasks.

namespace xla {
namespace gpu {

// In GPU memory, does
//
//   char* base_ptr = ...;
//   void* ptrs_out = ...;
//   for (i = 0; i < n; i++) {
//     ptrs_out[i] = base_ptr + i * stride;
//   }
//
// This is useful for functions like cublasTrsmBatched that operate on an array
// of pointers in GPU memory.  In XLA these aren't usually arbitrary pointers
// but rather are all contiguous values.
//
// Instead of using a kernel, a simpler way of doing this would be to create
// this buffer on the host and then copy it to device.  But using a kernel
// instead of an H2D copy avoids a few performance pitfalls.
//
//  - Only one H2D copy can run on a given GPU at a time.  If there's already
//    a copy ongoing as part of other work on the GPU, our copy here will
//    block.  In contrast, multiple kernels can run simultaneously.
//
//  - H2D copies from CUDA unpinned memory can acquire a global lock in the
//    driver and slow down *all* work on the GPU.  So to do this right, we'd
//    need to allocate the host memory as pinned, one alloc per stream.  Then
//    we'd need to manage this memory without leaks.  This becomes complex!
Status MakeBatchPointers(se::Stream* stream, const se::GpuAsmOpts& asm_opts,
                         se::DeviceMemoryBase base_ptr, int stride_bytes, int n,
                         se::DeviceMemoryBase ptrs_out);

}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PRECOMPILED_KERNELS_H_
