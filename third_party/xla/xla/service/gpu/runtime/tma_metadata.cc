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
#include "xla/service/gpu/runtime/tma_metadata.h"

#include <stdint.h>

#include <algorithm>
#include <initializer_list>
#include <set>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Types.h"

namespace xla {
namespace gpu {

static constexpr std::initializer_list<int> kValidElementSizes = {1, 2, 4};

absl::StatusOr<TmaInfo> TmaInfo::Create(llvm::ArrayRef<int64_t> tensor_shape,
                                        absl::Span<const int64_t> global_dims,
                                        mlir::Type element_type) {
  if (tensor_shape.size() != 2) {
    return absl::InvalidArgumentError("TMA only supports 2D tensors for now");
  }
  if (global_dims.size() != 2) {
    return absl::InvalidArgumentError("TMA only supports 2D tensors for now");
  }
  if (tensor_shape[0] > 256 || tensor_shape[1] > 256) {
    return absl::InvalidArgumentError(
        "tensor dims are too large for TMA. Must be <= 256.");
  }
  auto element_size = element_type.getIntOrFloatBitWidth() / 8;
  if (std::find(std::begin(kValidElementSizes), std::end(kValidElementSizes),
                element_size) == std::end(kValidElementSizes)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported element size: %d", element_size));
  }
  if (element_size * tensor_shape[0] < 32) {
    return absl::InvalidArgumentError("block size too small");
  }
  return TmaInfo(tensor_shape, global_dims, element_size);
}

// Defaults and logic taken from fill2DTMADescriptor in:
// @triton/third_party/nvidia/backend/cuda_utils.cc
// API and restrictions based on:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
TmaInfo::TmaInfo(llvm::ArrayRef<int64_t> tensor_shape,
                 absl::Span<const int64_t> dims, int element_size) {
  tensor_dims_[0] = tensor_shape[1];
  tensor_dims_[1] = tensor_shape[0];
  global_dims_[0] = dims[1];
  global_dims_[1] = dims[0];

  uint32_t contig_dim_size_in_byte = element_size * tensor_dims_[0];
  if (contig_dim_size_in_byte >= 128) {
    swizzle_ = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if (contig_dim_size_in_byte >= 64) {
    swizzle_ = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (contig_dim_size_in_byte >= 32) {
    swizzle_ = CU_TENSOR_MAP_SWIZZLE_32B;
  } else {
    CHECK(false && "block size too small.");
  }

  if (contig_dim_size_in_byte > 128) {
    tensor_dims_[0] = 128 / element_size;
  }

  global_strides_[0] = global_dims_[0] * element_size;
  global_strides_[1] = global_strides_[0] * global_dims_[1];

  switch (element_size) {
    case 1:
      data_type_ = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    case 2:
      data_type_ = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 4:
      data_type_ = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    default:
      CHECK(false && "unsupported element size. Must be 1, 2, or 4.");
  }
}

CUresult TmaInfo::CreateTensorMap(CUtensorMap* tensor_map,
                                  void* global_address) {
  return cuTensorMapEncodeTiled(tensor_map, data_type_, rank_, global_address,
                                global_dims_, global_strides_, tensor_dims_,
                                element_strides_, interleave_, swizzle_,
                                l2_promotion_, float_oob_fill_);
}

std::string TmaInfo::ToString() {
  std::ostringstream oss;
  oss << "TmaInfo{tensor_dims: {" << tensor_dims_[0] << ", " << tensor_dims_[1]
      << "}, global_dims: {" << global_dims_[0] << ", " << global_dims_[1]
      << "}, global_strides: {" << global_strides_[0] << ", "
      << global_strides_[1] << "}, rank: " << rank_
      << ", data_type: " << data_type_ << ", swizzle: " << swizzle_
      << ", interleave: " << interleave_ << ", l2_promotion: " << l2_promotion_
      << ", float_oob_fill: " << float_oob_fill_ << ", element_strides: {"
      << element_strides_[0] << ", " << element_strides_[1] << "}}";
  return oss.str();
}

}  // namespace gpu
}  // namespace xla
