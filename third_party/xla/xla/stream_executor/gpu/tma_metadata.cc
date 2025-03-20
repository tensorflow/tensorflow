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
#include "xla/stream_executor/gpu/tma_metadata.h"

#include <stdint.h>

#include <cmath>
#include <initializer_list>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor {
namespace gpu {

// Constants & TMA limitations taken from:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

// Supported element byte widths for TMA.
static constexpr std::initializer_list<int> kValidElementByteWidths = {1, 2, 4,
                                                                       8};

// `boxDim`s are limited to 256 by Nvidia's TMA API.
const int kMaxBoxDim = 256;

// Minimum and maximum rank of a tensor supported by TMA.
const int kMinRank = 1;
const int kMaxRank = 5;

// Maximum global dimension.
const uint64_t kMaxGlobalDim = pow(2, 32) - 1;

// Maximum global stride.
const uint64_t kMaxGlobalStide = pow(2, 40) - 1;

// Maximum element stride.
const uint32_t kMaxElementStride = 8;

absl::Status ValidateRank(llvm::ArrayRef<uint64_t> global_dims,
                          llvm::ArrayRef<uint64_t> global_strides,
                          llvm::ArrayRef<uint32_t> box_dims,
                          llvm::ArrayRef<uint32_t> element_strides,
                          TmaDescriptor::TmaInterleave interleave) {
  int rank = global_dims.size();
  if (rank < kMinRank || rank > kMaxRank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported rank for TMA: %d. Must be 1-5", rank));
  }
  if (element_strides.size() != rank || box_dims.size() != rank) {
    return absl::FailedPreconditionError(
        "global_dims, box_dims and element_strides must have the same rank");
  }
  if (global_strides.size() != rank - 1) {
    return absl::FailedPreconditionError(
        "global_strides must have a rank of: rank(global_dims) - 1.");
  }
  if (interleave != TmaDescriptor::TmaInterleave::kNone && rank < 3) {
    return absl::FailedPreconditionError(
        "If TmaInterleave is not kNone, then tensor rank must additionally be "
        ">= 3.");
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobalDims(llvm::ArrayRef<uint64_t> global_dims) {
  if (llvm::any_of(global_dims, [](uint64_t dim) {
        return dim == 0 || dim > kMaxGlobalDim;
      })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("global_dims (%s) must be non-zero and <= 2^32.",
                        absl::StrJoin(global_dims, ",")));
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobalStrides(llvm::ArrayRef<uint64_t> global_dims,
                                   llvm::ArrayRef<uint64_t> global_strides,
                                   TmaDescriptor::TmaInterleave interleave) {
  for (auto [i, stride] : llvm::enumerate(global_strides)) {
    if (stride % 16 != 0 || stride > kMaxGlobalStide) {
      return absl::InvalidArgumentError(
          absl::StrFormat("global_strides (%s) must be a multiple of 16 and "
                          "<= 2^40.",
                          absl::StrJoin(global_strides, ",")));
    }
    if (interleave == TmaDescriptor::TmaInterleave::k32B && stride % 32 != 0) {
      return absl::FailedPreconditionError(
          absl::StrFormat("global_strides (%s) must be a multiple of 32 when "
                          "interleave is 32B.",
                          absl::StrJoin(global_strides, ",")));
    }
    if (i > 0 && stride % global_strides[i - 1] != 0) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "global_stride (%d) must be a multiple of the previous stride (%d).",
          stride, global_strides[i - 1]));
    }
    if (stride < global_dims[i]) {
      return absl::FailedPreconditionError(
          absl::StrFormat("global_stride (%d) must be >= global_dim (%d).",
                          stride, global_dims[i]));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateBoxDims(llvm::ArrayRef<uint32_t> box_dims,
                             int element_byte_width,
                             TmaDescriptor::TmaInterleave interleave) {
  if (llvm::any_of(box_dims,
                   [](uint32_t dim) { return dim == 0 || dim > kMaxBoxDim; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("box_dims [%s] must be non-zero and <= 256.",
                        absl::StrJoin(box_dims, ",")));
  }
  if (interleave == TmaDescriptor::TmaInterleave::kNone &&
      box_dims[0] * element_byte_width % 16 != 0) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "when interleave is kNone, box_dims[0] (%d) * element_byte_width (%d) "
        "must be a multiple of 16 bytes.",
        box_dims[0], element_byte_width));
  }
  return absl::OkStatus();
}

absl::Status ValidateInterleaveAndSwizzleCombos(
    TmaDescriptor::TmaInterleave interleave, TmaDescriptor::TmaSwizzle swizzle,
    llvm::ArrayRef<uint32_t> box_dims, int element_byte_width) {
  if (interleave == TmaDescriptor::TmaInterleave::kNone &&
      swizzle != TmaDescriptor::TmaSwizzle::kNone) {
    uint32_t bounding_box_inner_dim = box_dims[0] * element_byte_width;
    if (swizzle == TmaDescriptor::TmaSwizzle::k32B &&
        bounding_box_inner_dim > 32) {
      return absl::FailedPreconditionError(
          "when interleave is kNone and swizzle is k32B, box_dims[0] * "
          "element_byte_width must be <= 32.");
    } else if (swizzle == TmaDescriptor::TmaSwizzle::k64B &&
               bounding_box_inner_dim > 64) {
      return absl::FailedPreconditionError(
          "when interleave is kNone and swizzle is k64B, box_dims[0] * "
          "element_byte_width must be <= 64.");
    } else if (swizzle == TmaDescriptor::TmaSwizzle::k128B &&
               bounding_box_inner_dim > 128) {
      return absl::FailedPreconditionError(
          "when interleave is kNone and swizzle is k128B, box_dims[0] * "
          "element_byte_width must be <= 128.");
    }
  }
  if (interleave == TmaDescriptor::TmaInterleave::k32B &&
      swizzle != TmaDescriptor::TmaSwizzle::k32B) {
    return absl::FailedPreconditionError(
        "when interleave is k32B, swizzle must be k32B.");
  }
  return absl::OkStatus();
}

absl::Status ValidateElementStrides(llvm::ArrayRef<uint32_t> element_strides) {
  if (llvm::any_of(element_strides, [](uint32_t stride) {
        return stride == 0 || stride > kMaxElementStride;
      })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("element_strides (%s) must be non-zero and <= 8.",
                        absl::StrJoin(element_strides, ",")));
  }
  return absl::OkStatus();
}

absl::StatusOr<TmaDescriptor> TmaDescriptor::Create(
    llvm::ArrayRef<uint64_t> global_dims,
    llvm::ArrayRef<uint64_t> global_strides, llvm::ArrayRef<uint32_t> box_dims,
    llvm::ArrayRef<uint32_t> element_strides, int element_byte_width,
    TmaInterleave interleave, TmaSwizzle swizzle, TmaL2Promotion l2_promotion,
    TmaFloatOobFill float_oob_fill) {
  // Validate each of the parameters as documented here:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

  // Validate element byte width.
  if (!absl::c_linear_search(kValidElementByteWidths, element_byte_width)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported element size: %d", element_byte_width));
  }

  TF_RETURN_IF_ERROR(ValidateRank(global_dims, global_strides, box_dims,
                                  element_strides, interleave));
  TF_RETURN_IF_ERROR(ValidateGlobalDims(global_dims));
  TF_RETURN_IF_ERROR(
      ValidateGlobalStrides(global_dims, global_strides, interleave));
  TF_RETURN_IF_ERROR(ValidateBoxDims(box_dims, element_byte_width, interleave));
  TF_RETURN_IF_ERROR(ValidateElementStrides(element_strides));
  TF_RETURN_IF_ERROR(ValidateInterleaveAndSwizzleCombos(
      interleave, swizzle, box_dims, element_byte_width));

  return TmaDescriptor(global_dims, global_strides, box_dims, element_strides,
                       element_byte_width, interleave, swizzle, l2_promotion,
                       float_oob_fill);
}

TmaDescriptor::TmaDescriptor(llvm::ArrayRef<uint64_t> global_dims,
                             llvm::ArrayRef<uint64_t> global_strides,
                             llvm::ArrayRef<uint32_t> box_dims,
                             llvm::ArrayRef<uint32_t> element_strides,
                             int element_size, TmaInterleave interleave,
                             TmaSwizzle swizzle, TmaL2Promotion l2_promotion,
                             TmaFloatOobFill float_oob_fill)
    : element_size_(element_size),
      num_dimensions_(global_dims.size()),
      global_dims_(global_dims.begin(), global_dims.end()),
      global_strides_(global_strides.begin(), global_strides.end()),
      box_dims_(box_dims.begin(), box_dims.end()),
      element_strides_(element_strides.begin(), element_strides.end()),
      interleave_(interleave),
      swizzle_(swizzle),
      l2_promotion_(l2_promotion),
      float_oob_fill_(float_oob_fill) {}

std::string TmaDescriptor::ToString() const {
  return absl::StrFormat(
      "TmaDescriptor{element_size: %d, rank: %d, global_dims: {%s}, "
      "global_strides: {%s}, box_dims: {%s}, element_strides: {%s}, "
      "interleave: %d, swizzle: %d, l2_promotion: %d, "
      "float_oob_fill: %d}",
      element_size_, num_dimensions_, absl::StrJoin(global_dims_, ","),
      absl::StrJoin(global_strides_, ","), absl::StrJoin(box_dims_, ","),
      absl::StrJoin(element_strides_, ","), interleave_, swizzle_,
      l2_promotion_, float_oob_fill_);
}

}  // namespace gpu
}  // namespace stream_executor
