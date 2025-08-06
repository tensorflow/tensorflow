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

TmaDescriptor::TmaSwizzle GetTmaSwizzleMode(SwizzleMode swizzle_mode) {
  switch (swizzle_mode) {
    case SwizzleMode::kNone:
      return TmaDescriptor::TmaSwizzle::kNone;
    case SwizzleMode::k32b:
      return TmaDescriptor::TmaSwizzle::k32B;
    case SwizzleMode::k64b:
      return TmaDescriptor::TmaSwizzle::k64B;
    case SwizzleMode::k128b:
      return TmaDescriptor::TmaSwizzle::k128B;
  }
}

// Returns a TmaDescriptor for a tensor to be emitted in Triton.
//
// This function follows the defaults and logic found in fillTMADescriptor in
// @triton/third_party/nvidia/backend/cuda_utils.cc
absl::StatusOr<TmaDescriptor> CreateTmaDescriptor(
    llvm::ArrayRef<int64_t> global_shape, llvm::ArrayRef<int64_t> tile_shape,
    llvm::ArrayRef<int64_t> tile_strides, llvm::ArrayRef<int64_t> layout,
    int element_byte_size, TmaDescriptor::TmaSwizzle swizzle_mode) {
  if (global_shape.size() != tile_shape.size()) {
    return absl::InvalidArgumentError(
        "global_shape and tile_shape must have the same size");
  }
  if (global_shape.empty() || global_shape.size() > 5) {
    return absl::InvalidArgumentError(
        "expected global/tile shapes to be between 1D and 5D");
  }

  SmallVector<uint64_t, 5> global_dims;
  for (auto layout_dim : layout) {
    global_dims.push_back(global_shape[layout_dim]);
  }

  SmallVector<uint64_t, 4> global_strides;
  if (global_dims.size() >= 2) {
    global_strides.push_back(global_dims[0] * element_byte_size);
    for (int i = 1; i < global_dims.size() - 1; ++i) {
      global_strides.push_back(global_dims[i] * global_strides[i - 1]);
    }
  }

  // Tile strides are reflected in the element strides. Note that the most minor
  // dimension should have a stride of 1.
  CHECK(tile_strides[layout[0]] == 1)
      << "tile stride must be 1 for the most minor dimension";
  SmallVector<uint32_t, 5> element_strides;
  for (auto layout_dim : layout) {
    element_strides.push_back(tile_strides[layout_dim]);
  }

  // When the tile strides are > 1, the box dimensions no longer reflect the
  // number of elements in the tile. To load the correct number of
  // elements, we need to multiply the tile strides by the number of elements in
  // the tile.
  SmallVector<uint32_t, 5> box_dims;
  for (auto layout_dim : layout) {
    box_dims.push_back(static_cast<uint32_t>(tile_shape[layout_dim]) *
                       tile_strides[layout_dim]);
  }

  // We need to respect maximum limit restrictions on box_dims imposed by TMA.
  // Documented in
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // Triton handles this by clamping the block size in a similar fashion:
  // triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp;l=102-119;rcl=759529316
  // Replicating their code-comment here:
  // "We clamp the block size and the codegen will emit multiple copy
  // operations."
  // This only works because Triton expects this way of handling and makes the
  // corresponding codegen adjustments under the hood.
  uint32_t contig_dim_size_in_byte = element_byte_size * box_dims[0];
  if (swizzle_mode == TmaDescriptor::TmaSwizzle::k32B &&
      contig_dim_size_in_byte > 32) {
    box_dims[0] = 32 / element_byte_size;
  } else if (swizzle_mode == TmaDescriptor::TmaSwizzle::k64B &&
             contig_dim_size_in_byte > 64) {
    box_dims[0] = 64 / element_byte_size;
  } else if (swizzle_mode == TmaDescriptor::TmaSwizzle::k128B &&
             contig_dim_size_in_byte > 128) {
    box_dims[0] = 128 / element_byte_size;
  }

  TF_ASSIGN_OR_RETURN(
      auto tma_desc, TmaDescriptor::Create(
                         global_dims, global_strides, box_dims, element_strides,
                         element_byte_size, TmaDescriptor::TmaInterleave::kNone,
                         swizzle_mode, TmaDescriptor::TmaL2Promotion::k128B));
  return tma_desc;
}

absl::StatusOr<TmaDescriptor> CreateTmaDescriptor(
    llvm::ArrayRef<int64_t> global_shape, llvm::ArrayRef<int64_t> tile_shape,
    llvm::ArrayRef<int64_t> tile_strides, llvm::ArrayRef<int64_t> layout,
    int element_byte_size, SwizzleMode swizzle_mode) {
  return CreateTmaDescriptor(global_shape, tile_shape, tile_strides, layout,
                             element_byte_size,
                             GetTmaSwizzleMode(swizzle_mode));
}

}  // namespace xla::gpu
