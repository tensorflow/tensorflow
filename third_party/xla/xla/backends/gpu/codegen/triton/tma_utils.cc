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
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

using ::llvm::SmallVector;
using ::stream_executor::gpu::TmaDescriptor;

// Returns a TmaDescriptor for a 2D tensor to be emitted in Triton.
//
// This function follows the defaults and logic found in fill2DTMADescriptor in
// @triton/third_party/nvidia/backend/cuda_utils.cc
absl::StatusOr<TmaDescriptor> Create2DTmaDescriptor(
    llvm::ArrayRef<int64_t> global_shape, llvm::ArrayRef<int64_t> block_shape,
    llvm::ArrayRef<int64_t> layout, int element_byte_size) {
  if (global_shape.size() != 2) {
    return absl::InvalidArgumentError("expected 2D global shape");
  }
  if (block_shape.size() != 2) {
    return absl::InvalidArgumentError("expected 2D block shape");
  }

  SmallVector<uint64_t, 2> global_dims = {
      static_cast<uint64_t>(global_shape[layout[0]]),
      static_cast<uint64_t>(global_shape[layout[1]])};
  auto global_strides = {global_dims[0] * element_byte_size};
  SmallVector<uint32_t, 2> box_dims = {
      static_cast<uint32_t>(block_shape[layout[0]]),
      static_cast<uint32_t>(block_shape[layout[1]])};
  SmallVector<uint32_t, 2> element_strides = {1, 1};
  TmaDescriptor::TmaSwizzle swizzle;
  uint32_t contig_dim_size_in_byte = element_byte_size * box_dims[0];
  if (contig_dim_size_in_byte >= 128) {
    swizzle = TmaDescriptor::TmaSwizzle::k128B;
  } else if (contig_dim_size_in_byte >= 64) {
    swizzle = TmaDescriptor::TmaSwizzle::k64B;
  } else if (contig_dim_size_in_byte >= 32) {
    swizzle = TmaDescriptor::TmaSwizzle::k32B;
  } else {
    return absl::FailedPreconditionError("contiguous dimension size too small");
  }
  if (contig_dim_size_in_byte > 128) {
    box_dims[0] = 128 / element_byte_size;
  }
  TF_ASSIGN_OR_RETURN(
      auto tma_desc, TmaDescriptor::Create(
                         global_dims, global_strides, box_dims, element_strides,
                         element_byte_size, TmaDescriptor::TmaInterleave::kNone,
                         swizzle, TmaDescriptor::TmaL2Promotion::k128B));
  return tma_desc;
}

}  // namespace xla::gpu
