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

#include "xla/stream_executor/cuda/tma_util.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace stream_executor::gpu {

absl::StatusOr<CUtensorMapDataType> GetTensorMapDataType(int element_size) {
  switch (element_size) {
    case 1:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case 2:
      return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    case 4:
      return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case 8:
      return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("unsupported element size: %d", element_size));
  }
}

CUtensorMapSwizzle GetTensorMapSwizzle(TmaDescriptor::TmaSwizzle swizzle) {
  switch (swizzle) {
    case TmaDescriptor::TmaSwizzle::kNone:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case TmaDescriptor::TmaSwizzle::k32B:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case TmaDescriptor::TmaSwizzle::k64B:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case TmaDescriptor::TmaSwizzle::k128B:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
}

CUtensorMapL2promotion GetTensorMapL2Promotion(
    TmaDescriptor::TmaL2Promotion l2_promotion) {
  switch (l2_promotion) {
    case TmaDescriptor::TmaL2Promotion::kNone:
      return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case TmaDescriptor::TmaL2Promotion::k64B:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case TmaDescriptor::TmaL2Promotion::k128B:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case TmaDescriptor::TmaL2Promotion::k256B:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }
}

CUtensorMapFloatOOBfill GetTensorMapFloatOOBFill(
    TmaDescriptor::TmaFloatOobFill oob_fill) {
  switch (oob_fill) {
    case TmaDescriptor::TmaFloatOobFill::kNone:
      return CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case TmaDescriptor::TmaFloatOobFill::kNanRequestZeroFma:
      return CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  }
}

CUtensorMapInterleave GetTensorMapInterleave(
    TmaDescriptor::TmaInterleave interleave) {
  switch (interleave) {
    case TmaDescriptor::TmaInterleave::kNone:
      return CU_TENSOR_MAP_INTERLEAVE_NONE;
    case TmaDescriptor::TmaInterleave::k16B:
      return CU_TENSOR_MAP_INTERLEAVE_16B;
    case TmaDescriptor::TmaInterleave::k32B:
      return CU_TENSOR_MAP_INTERLEAVE_32B;
  }
}

}  // namespace stream_executor::gpu
