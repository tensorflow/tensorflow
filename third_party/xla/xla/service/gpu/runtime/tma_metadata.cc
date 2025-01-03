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

#include <initializer_list>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {
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

absl::StatusOr<TmaDescriptor> TmaDescriptor::Create(
    llvm::ArrayRef<int64_t> global_dims, llvm::ArrayRef<int64_t> global_strides,
    llvm::ArrayRef<int32_t> box_dims, llvm::ArrayRef<int32_t> element_strides,
    int element_byte_width, tma_interleave interleave, tma_swizzle swizzle,
    tma_l2_promotion l2_promotion, tma_float_oob_fill float_oob_fill) {
  int rank = global_dims.size();
  if (rank < kMinRank || rank > kMaxRank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported rank for TMA: %d", rank));
  }
  if (global_strides.size() != rank || box_dims.size() != rank ||
      element_strides.size() != rank) {
    return absl::InvalidArgumentError(
        "global_dims, global_strides, box_dims and "
        "element_strides must have the same rank");
  }
  for (auto box_dim : box_dims) {
    if (box_dim > kMaxBoxDim) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "box dim %d too large for TMA. Must be <= 256.", box_dim));
    }
  }
  if (!absl::c_linear_search(kValidElementByteWidths, element_byte_width)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported element size: %d", element_byte_width));
  }
  return TmaDescriptor(global_dims, global_strides, box_dims, element_strides,
                       element_byte_width, interleave, swizzle, l2_promotion,
                       float_oob_fill);
}

TmaDescriptor::TmaDescriptor(llvm::ArrayRef<int64_t> global_dims,
                             llvm::ArrayRef<int64_t> global_strides,
                             llvm::ArrayRef<int32_t> box_dims,
                             llvm::ArrayRef<int32_t> element_strides,
                             int element_size, tma_interleave interleave,
                             tma_swizzle swizzle, tma_l2_promotion l2_promotion,
                             tma_float_oob_fill float_oob_fill)
    : element_size_(element_size),
      rank_(global_dims.size()),
      global_dims_(global_dims.begin(), global_dims.end()),
      global_strides_(global_strides.begin(), global_strides.end()),
      box_dims_(box_dims.begin(), box_dims.end()),
      element_strides_(element_strides.begin(), element_strides.end()),
      interleave_(interleave),
      swizzle_(swizzle),
      l2_promotion_(l2_promotion),
      float_oob_fill_(float_oob_fill) {}

std::string TmaDescriptor::ToString() const {
  std::string result = "TmaDescriptor{";
  llvm::raw_string_ostream os(result);
  os << "element_size: " << element_size_ << ", rank: " << rank_
     << ", global_dims: {";
  llvm::interleaveComma(global_dims_, os);
  os << "}, global_strides: {";
  llvm::interleaveComma(global_strides_, os);
  os << "}, box_dims: {";
  llvm::interleaveComma(box_dims_, os);
  os << "}, element_strides: {";
  llvm::interleaveComma(element_strides_, os);
  os << "}, interleave: " << interleave_ << ", swizzle: " << swizzle_
     << ", l2_promotion: " << l2_promotion_
     << ", float_oob_fill: " << float_oob_fill_ << "}";
  return result;
}

}  // namespace gpu
}  // namespace xla
