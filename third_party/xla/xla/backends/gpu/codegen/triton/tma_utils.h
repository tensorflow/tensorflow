/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TMA_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TMA_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace xla::gpu {

// Returns a TmaDescriptor for a 2D tensor to be emitted in Triton.
absl::StatusOr<stream_executor::gpu::TmaDescriptor> Create2DTmaDescriptor(
    llvm::ArrayRef<int64_t> global_shape, llvm::ArrayRef<int64_t> tile_shape,
    llvm::ArrayRef<int64_t> tile_strides, llvm::ArrayRef<int64_t> layout,
    int element_byte_size, mlir::triton::xla::SwizzleMode swizzle_mode);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TMA_UTILS_H_
