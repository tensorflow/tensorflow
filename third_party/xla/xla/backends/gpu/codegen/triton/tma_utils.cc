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

#include "xla/backends/gpu/codegen/triton/tma_utils.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

using ::llvm::SmallVector;
using mlir::triton::xla::SwizzleMode;
using ::stream_executor::gpu::TmaDescriptor;

absl::StatusOr<TmaDescriptor::TmaSwizzle> GetTmaSwizzleMode(
    mlir::triton::xla::SwizzleMode swizzle_mode) {
  switch (swizzle_mode) {
    case SwizzleMode::kNone:
      return TmaDescriptor::TmaSwizzle::kNone;
    case SwizzleMode::k32b:
      return TmaDescriptor::TmaSwizzle::k32B;
    case SwizzleMode::k64b:
      return TmaDescriptor::TmaSwizzle::k64B;
    case SwizzleMode::k128b:
      return TmaDescriptor::TmaSwizzle::k128B;
    case SwizzleMode::kUnset:
      return absl::InvalidArgumentError("swizzle mode must be set");
  }
}

absl::StatusOr<TmaDescriptor> Create2DTmaDescriptor(
    llvm::ArrayRef<int64_t> global_shape, llvm::ArrayRef<int64_t> block_shape,
    int element_byte_size, SwizzleMode swizzle_mode) {
  auto tma_swizzle_mode = GetTmaSwizzleMode(swizzle_mode);
  if (!tma_swizzle_mode.ok()) {
    return tma_swizzle_mode.status();
  }
  return Create2DTmaDescriptor(global_shape, block_shape, element_byte_size,
                               tma_swizzle_mode.value());
}

// Returns a TmaDescriptor for a 2D tensor to be emitted in Triton.
//
// This function follows the defaults and logic found in fill2DTMADescriptor in
// @triton/third_party/nvidia/backend/cuda_utils.cc
absl::StatusOr<TmaDescriptor> Create2DTmaDescriptor(
    llvm::ArrayRef<int64_t> global_shape, llvm::ArrayRef<int64_t> block_shape,
    int element_byte_size,
    stream_executor::gpu::TmaDescriptor::TmaSwizzle swizzle_mode) {
  if (global_shape.size() != 2) {
    return absl::InvalidArgumentError("expected 2D global shape");
  }
  if (block_shape.size() != 2) {
    return absl::InvalidArgumentError("expected 2D block shape");
  }
  // TODO(b/413351367): Figure out if we need (and how) to handle non-normalized
  // layouts.
  SmallVector<uint64_t, 2> global_dims = {
      static_cast<uint64_t>(global_shape[1]),
      static_cast<uint64_t>(global_shape[0])};
  auto global_strides = {global_dims[0] * element_byte_size};
  SmallVector<uint32_t, 2> box_dims = {static_cast<uint32_t>(block_shape[1]),
                                       static_cast<uint32_t>(block_shape[0])};
  SmallVector<uint32_t, 2> element_strides = {1, 1};

  uint32_t contig_dim_size_in_byte = element_byte_size * box_dims[0];
  if (contig_dim_size_in_byte > 128) {
    box_dims[0] = 128 / element_byte_size;
  }

  TF_ASSIGN_OR_RETURN(
      auto tma_desc, TmaDescriptor::Create(
                         global_dims, global_strides, box_dims, element_strides,
                         element_byte_size, TmaDescriptor::TmaInterleave::kNone,
                         swizzle_mode, TmaDescriptor::TmaL2Promotion::k128B));
  return tma_desc;
}

}  // namespace xla::gpu
