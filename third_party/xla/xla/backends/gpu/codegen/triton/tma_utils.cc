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
#include <optional>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu {

namespace mt = ::mlir::triton;

using ::llvm::SmallVector;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::Value;
using ::stream_executor::gpu::TmaDescriptor;
using ::stream_executor::gpu::TmaMetadata;

// Returns a TmaDescriptor for a 2D tensor to be emitted in Triton.
//
// This function follows the defaults and logic found in fill2DTMADescriptor in
// @triton/third_party/nvidia/backend/cuda_utils.cc
absl::StatusOr<TmaDescriptor> Create2DTmaDescriptor(
    Shape global_shape, llvm::ArrayRef<int64_t> block_shape,
    Type element_type) {
  if (global_shape.dimensions().size() != 2) {
    return absl::InvalidArgumentError("expected 2D global shape");
  }
  if (block_shape.size() != 2) {
    return absl::InvalidArgumentError("expected 2D block shape");
  }
  int byte_width = element_type.getIntOrFloatBitWidth() / 8;
  SmallVector<uint64_t, 2> global_dims = {
      static_cast<uint64_t>(global_shape.dimensions(1)),
      static_cast<uint64_t>(global_shape.dimensions(0))};
  auto global_strides = {global_dims[0] * byte_width};
  SmallVector<uint32_t, 2> box_dims = {static_cast<uint32_t>(block_shape[1]),
                                       static_cast<uint32_t>(block_shape[0])};
  SmallVector<uint32_t, 2> element_strides = {1, 1};
  TmaDescriptor::TmaSwizzle swizzle;
  uint32_t contig_dim_size_in_byte = byte_width * box_dims[0];
  if (contig_dim_size_in_byte >= 128) {
    swizzle = TmaDescriptor::TmaSwizzle::k128B;
  } else if (contig_dim_size_in_byte >= 64) {
    swizzle = TmaDescriptor::TmaSwizzle::k64B;
  } else if (contig_dim_size_in_byte >= 32) {
    swizzle = TmaDescriptor::TmaSwizzle::k32B;
  } else {
    return absl::FailedPreconditionError(
        "continguous dimension size too small");
  }
  if (contig_dim_size_in_byte > 128) {
    box_dims[0] = 128 / byte_width;
  }
  TF_ASSIGN_OR_RETURN(
      auto tma_desc, TmaDescriptor::Create(
                         global_dims, global_strides, box_dims, element_strides,
                         byte_width, TmaDescriptor::TmaInterleave::kNone,
                         swizzle, TmaDescriptor::TmaL2Promotion::k128B));
  return tma_desc;
}

Value EmitTmaDescriptor(EmitterLocOpBuilder& b, Value arg,
                        RankedTensorType tensor_type) {
  // Create a barrier to retrieve the descriptor from device memory before use.
  // This will no longer be necessary once the descriptor is passed by value
  // to the kernel.
  b.create<mt::ExperimentalTensormapFenceproxyAcquireOp>(arg);
  auto desc_type = mt::TensorDescType::get(b.getContext(), tensor_type);
  return b.create<mt::ReinterpretTensorDescOp>(desc_type, arg);
}

void RewriteFunctionForTma(EmitterLocOpBuilder& b, mlir::triton::FuncOp fn,
                           std::optional<TmaMetadata> tma_metadata) {
  if (!tma_metadata.has_value()) {
    return;
  }
  for (auto& [parameter_number, _] : tma_metadata->arg_index_to_tma_info) {
    fn.setArgAttr(parameter_number, "tt.nv_tma_desc", b.getI32IntegerAttr(1));
  }
}

// Returns true if TMA is enabled for the given fusion & device.
bool TmaIsEnabled(const HloModuleConfig& config,
                  const stream_executor::DeviceDescription& device_info) {
  return config.debug_options().xla_gpu_experimental_enable_triton_tma() &&
         device_info.cuda_compute_capability().IsAtLeastHopper();
}

// Returns true if TMA is possible on the given shape.
bool CanUseTmaOnInput(const Shape& global_shape,
                      llvm::ArrayRef<int64_t> block_shape) {
  // Limitations of TMA:
  // - The minor dimension of the global input must be divisible by 16.
  // - The block size must be less than 256 in every dimension.
  // See source:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  if (primitive_util::ByteWidth(global_shape.element_type()) *
          global_shape.dimensions(1) % 16 !=
      0) {
    return false;
  }
  return llvm::none_of(block_shape, [](int64_t dim) { return dim > 256; });
}

}  // namespace xla::gpu
