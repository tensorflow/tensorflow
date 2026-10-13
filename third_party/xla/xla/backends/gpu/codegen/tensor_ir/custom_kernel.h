/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CUSTOM_KERNEL_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CUSTOM_KERNEL_H_

#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/kernel_reuse_cache.h"

namespace xla::gpu::tensor_ir {

// Extracts what is needed to launch a kernel compiled by the TensorIR CudaTile
// backend with a `CustomKernelThunk`: the cubin, the entry point and the launch
// grid. Going through the generic thunk rather than a bespoke one is what lets
// XLA serialize the kernel and record it into a command buffer, and returning a
// `KernelReuseCache::Entry` lets identical fusions share one compiled kernel.
//
// `kernel` must have come from a compiler created with
// `CompilerBackend::CudaTile`, which only ever returns a
// `CudaTileRuntimeKernel`.
absl::StatusOr<KernelReuseCache::Entry> MakeKernelCacheEntry(
    const ::tensor_ir::rt::IRuntimeKernel& kernel);

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CUSTOM_KERNEL_H_
