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
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

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
const uint64_t kMaxGlobalDim = pow(2, 32);

// Maximum global stride.
const uint64_t kMaxGlobalStide = pow(2, 40) - 1;

// Maximum element stride.
const uint32_t kMaxElementStride = 8;

absl::Status ValidateRank(absl::Span<const uint64_t> global_dims,
                          absl::Span<const uint64_t> global_strides,
                          absl::Span<const uint32_t> box_dims,
                          absl::Span<const uint32_t> element_strides,
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

absl::Status ValidateGlobalDims(absl::Span<const uint64_t> global_dims) {
  if (absl::c_any_of(global_dims, [](uint64_t dim) {
        return dim == 0 || dim > kMaxGlobalDim;
      })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("global_dims (%s) must be non-zero and <= 2^32.",
                        absl::StrJoin(global_dims, ",")));
  }
  return absl::OkStatus();
}

absl::Status ValidateGlobalStrides(absl::Span<const uint64_t> global_dims,
                                   absl::Span<const uint64_t> global_strides,
                                   TmaDescriptor::TmaInterleave interleave) {
  for (int i = 0; i < global_strides.size(); ++i) {
    uint64_t stride = global_strides[i];
    if (stride % 16 != 0 || stride > kMaxGlobalStide) {
      return absl::InvalidArgumentError(
          absl::StrFormat("global_strides (%s) must be a multiple of 16 and "
                          "< 2^40.",
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

absl::Status ValidateBoxDims(absl::Span<const uint32_t> box_dims,
                             int element_byte_width,
                             TmaDescriptor::TmaInterleave interleave) {
  if (absl::c_any_of(box_dims, [](uint32_t dim) {
        return dim == 0 || dim > kMaxBoxDim;
      })) {
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
    absl::Span<const uint32_t> box_dims, int element_byte_width) {
  if (interleave == TmaDescriptor::TmaInterleave::kNone &&
      swizzle != TmaDescriptor::TmaSwizzle::kNone) {
    uint32_t bounding_box_inner_dim = box_dims[0] * element_byte_width;
    if (swizzle == TmaDescriptor::TmaSwizzle::k32B &&
        bounding_box_inner_dim > 32) {
      return absl::FailedPreconditionError(
          "when interleave is kNone and swizzle is k32B, box_dims[0] * "
          "element_byte_width must be <= 32.");
    }
    if (swizzle == TmaDescriptor::TmaSwizzle::k64B &&
        bounding_box_inner_dim > 64) {
      return absl::FailedPreconditionError(
          "when interleave is kNone and swizzle is k64B, box_dims[0] * "
          "element_byte_width must be <= 64.");
    }
    if (swizzle == TmaDescriptor::TmaSwizzle::k128B &&
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

absl::Status ValidateElementStrides(
    absl::Span<const uint32_t> element_strides) {
  if (element_strides[0] != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("element_strides[0] must be 1 for TMA. Got %d instead.",
                        element_strides[0]));
  }
  if (absl::c_any_of(element_strides, [](uint32_t stride) {
        return stride == 0 || stride > kMaxElementStride;
      })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("element_strides (%s) must be non-zero and <= 8.",
                        absl::StrJoin(element_strides, ",")));
  }
  return absl::OkStatus();
}

absl::StatusOr<TmaDescriptor> TmaDescriptor::Create(
    absl::Span<const uint64_t> global_dims,
    absl::Span<const uint64_t> global_strides,
    absl::Span<const uint32_t> box_dims,
    absl::Span<const uint32_t> element_strides, int element_byte_width,
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

TmaDescriptor::TmaDescriptor(absl::Span<const uint64_t> global_dims,
                             absl::Span<const uint64_t> global_strides,
                             absl::Span<const uint32_t> box_dims,
                             absl::Span<const uint32_t> element_strides,
                             int element_size, TmaInterleave interleave,
                             TmaSwizzle swizzle, TmaL2Promotion l2_promotion,
                             TmaFloatOobFill float_oob_fill)
    : element_size_(element_size),
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
      "TmaDescriptor{element_size: %d, global_dims: {%s}, "
      "global_strides: {%s}, box_dims: {%s}, element_strides: {%s}, "
      "interleave: %d, swizzle: %d, l2_promotion: %d, "
      "float_oob_fill: %d}",
      element_size_, absl::StrJoin(global_dims_, ","),
      absl::StrJoin(global_strides_, ","), absl::StrJoin(box_dims_, ","),
      absl::StrJoin(element_strides_, ","), interleave_, swizzle_,
      l2_promotion_, float_oob_fill_);
}

TmaDescriptorProto TmaDescriptor::ToProto() const {
  TmaDescriptorProto proto;
  proto.set_element_size(element_size_);
  proto.mutable_global_dims()->Add(global_dims_.begin(), global_dims_.end());
  proto.mutable_global_strides()->Add(global_strides_.begin(),
                                      global_strides_.end());
  proto.mutable_box_dims()->Add(box_dims_.begin(), box_dims_.end());
  proto.mutable_element_strides()->Add(element_strides_.begin(),
                                       element_strides_.end());

  switch (interleave_) {
    case TmaInterleave::kNone:
      proto.set_interleave(TmaDescriptorProto::INTERLEAVE_NONE);
      break;
    case TmaInterleave::k16B:
      proto.set_interleave(TmaDescriptorProto::INTERLEAVE_BYTES16);
      break;
    case TmaInterleave::k32B:
      proto.set_interleave(TmaDescriptorProto::INTERLEAVE_BYTES32);
      break;
  }

  switch (swizzle_) {
    case TmaSwizzle::kNone:
      proto.set_swizzle(TmaDescriptorProto::SWIZZLE_NONE);
      break;
    case TmaSwizzle::k32B:
      proto.set_swizzle(TmaDescriptorProto::SWIZZLE_BYTES32);
      break;
    case TmaSwizzle::k64B:
      proto.set_swizzle(TmaDescriptorProto::SWIZZLE_BYTES64);
      break;
    case TmaSwizzle::k128B:
      proto.set_swizzle(TmaDescriptorProto::SWIZZLE_BYTES128);
      break;
  }

  switch (l2_promotion_) {
    case TmaL2Promotion::kNone:
      proto.set_l2_promotion(TmaDescriptorProto::L2_PROMOTION_NONE);
      break;
    case TmaL2Promotion::k64B:
      proto.set_l2_promotion(TmaDescriptorProto::L2_PROMOTION_BYTES64);
      break;
    case TmaL2Promotion::k128B:
      proto.set_l2_promotion(TmaDescriptorProto::L2_PROMOTION_BYTES128);
      break;
    case TmaL2Promotion::k256B:
      proto.set_l2_promotion(TmaDescriptorProto::L2_PROMOTION_BYTES256);
      break;
  }

  switch (float_oob_fill_) {
    case TmaFloatOobFill::kNone:
      proto.set_float_oob_fill(TmaDescriptorProto::FLOAT_OOB_FILL_NONE);
      break;
    case TmaFloatOobFill::kNanRequestZeroFma:
      proto.set_float_oob_fill(
          TmaDescriptorProto::FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
      break;
  }

  return proto;
}

absl::StatusOr<TmaDescriptor> TmaDescriptor::FromProto(
    const TmaDescriptorProto& proto) {
  TmaInterleave interleave;
  switch (proto.interleave()) {
    case TmaDescriptorProto::INTERLEAVE_NONE:
      interleave = TmaInterleave::kNone;
      break;
    case TmaDescriptorProto::INTERLEAVE_BYTES16:
      interleave = TmaInterleave::k16B;
      break;
    case TmaDescriptorProto::INTERLEAVE_BYTES32:
      interleave = TmaInterleave::k32B;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("unsupported interleave: %d", proto.interleave()));
  }

  TmaSwizzle swizzle;
  switch (proto.swizzle()) {
    case TmaDescriptorProto::SWIZZLE_NONE:
      swizzle = TmaSwizzle::kNone;
      break;
    case TmaDescriptorProto::SWIZZLE_BYTES32:
      swizzle = TmaSwizzle::k32B;
      break;
    case TmaDescriptorProto::SWIZZLE_BYTES64:
      swizzle = TmaSwizzle::k64B;
      break;
    case TmaDescriptorProto::SWIZZLE_BYTES128:
      swizzle = TmaSwizzle::k128B;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("unsupported swizzle: %d", proto.swizzle()));
  }

  TmaL2Promotion l2_promotion;
  switch (proto.l2_promotion()) {
    case TmaDescriptorProto::L2_PROMOTION_NONE:
      l2_promotion = TmaL2Promotion::kNone;
      break;
    case TmaDescriptorProto::L2_PROMOTION_BYTES64:
      l2_promotion = TmaL2Promotion::k64B;
      break;
    case TmaDescriptorProto::L2_PROMOTION_BYTES128:
      l2_promotion = TmaL2Promotion::k128B;
      break;
    case TmaDescriptorProto::L2_PROMOTION_BYTES256:
      l2_promotion = TmaL2Promotion::k256B;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "unsupported l2_promotion: %d", proto.l2_promotion()));
  }

  TmaFloatOobFill float_oob_fill;
  switch (proto.float_oob_fill()) {
    case TmaDescriptorProto::FLOAT_OOB_FILL_NONE:
      float_oob_fill = TmaFloatOobFill::kNone;
      break;
    case TmaDescriptorProto::FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA:
      float_oob_fill = TmaFloatOobFill::kNanRequestZeroFma;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "unsupported float_oob_fill: %d", proto.float_oob_fill()));
  }

  // We create these temporary vectors to convert from signed types to unsigned
  // types. We only check for negative values here since `TmaDescriptor::Create`
  // will do a more thorough validation.
  if (absl::c_any_of(proto.global_dims(),
                     [](int64_t dim) { return dim < 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("global_dims (%s) must be non-negative.",
                        absl::StrJoin(proto.global_dims(), ",")));
  }

  constexpr int kMaximumSupportedRank = 5;
  absl::InlinedVector<uint64_t, kMaximumSupportedRank> global_dims(
      proto.global_dims().begin(), proto.global_dims().end());

  if (absl::c_any_of(proto.global_strides(),
                     [](int64_t stride) { return stride < 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("global_strides (%s) must be non-negative.",
                        absl::StrJoin(proto.global_strides(), ",")));
  }
  absl::InlinedVector<uint64_t, kMaximumSupportedRank> global_strides(
      proto.global_strides().begin(), proto.global_strides().end());

  if (absl::c_any_of(proto.box_dims(), [](int32_t dim) { return dim < 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("box_dims (%s) must be non-negative.",
                        absl::StrJoin(proto.box_dims(), ",")));
  }
  absl::InlinedVector<uint32_t, kMaximumSupportedRank> box_dims(
      proto.box_dims().begin(), proto.box_dims().end());

  if (absl::c_any_of(proto.element_strides(),
                     [](int32_t stride) { return stride < 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("element_strides (%s) must be non-negative.",
                        absl::StrJoin(proto.element_strides(), ",")));
  }
  absl::InlinedVector<uint32_t, kMaximumSupportedRank> element_strides(
      proto.element_strides().begin(), proto.element_strides().end());

  return TmaDescriptor::Create(
      global_dims, global_strides, box_dims, element_strides,
      proto.element_size(), interleave, swizzle, l2_promotion, float_oob_fill);
}

TmaMetadataProto TmaMetadata::ToProto() const {
  TmaMetadataProto proto;
  for (const auto& [arg_index, tma_info] : arg_index_to_tma_info) {
    proto.mutable_arg_index_to_tma_info()->insert(
        {arg_index, tma_info.ToProto()});
  }
  return proto;
}

absl::StatusOr<TmaMetadata> TmaMetadata::FromProto(
    const TmaMetadataProto& proto) {
  TmaMetadata metadata;
  for (const auto& [arg_index, tma_info] : proto.arg_index_to_tma_info()) {
    TF_ASSIGN_OR_RETURN(TmaDescriptor descriptor,
                        TmaDescriptor::FromProto(tma_info));
    metadata.arg_index_to_tma_info.insert({arg_index, std::move(descriptor)});
  }
  return metadata;
}

// TODO(b/463912789): Re-enable TMA for Blackwell once the bug is fixed.
bool IsTmaAvailableForDevice(
    const stream_executor::DeviceDescription& device_info) {
  if (auto* cuda_cc =
          device_info.gpu_compute_capability().cuda_compute_capability()) {
    return cuda_cc->IsAtLeastHopper();
  }
  return false;
}

// Limitations of TMA:
// - The global shape must be > 0 and <= 2^32.
// - The minor dimension of the tile (in bytes) must be divisible by 16.
// - The minor dimension must be contiguous. i.e. its tile stride must be 1.
// - Global strides (in bytes) must be divisible by 16 and < 2^40.
// - The tile shape must be less than 256 in every dimension.
// - The element byte size must be 1, 2, 4, or 8.
// - Tile strides must be non-zero and <= 8.
// See source:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
absl::Status IsTmaCompatible(absl::Span<const int64_t> global_shape,
                             absl::Span<const int64_t> tile_shape,
                             absl::Span<const int64_t> tile_strides,
                             absl::Span<const int64_t> minor_to_major_layout,
                             int element_byte_size) {
  llvm::SmallVector<uint64_t, 5> normalized_global_shape;
  for (auto layout_dim : minor_to_major_layout) {
    normalized_global_shape.push_back(global_shape[layout_dim]);
  }

  llvm::SmallVector<uint64_t, 4> global_strides;
  if (normalized_global_shape.size() >= 2) {
    global_strides.push_back(normalized_global_shape[0] * element_byte_size);
    for (int64_t i = 1; i < normalized_global_shape.size() - 1; ++i) {
      global_strides.push_back(global_strides[i - 1] *
                               normalized_global_shape[i]);
    }
  }

  llvm::SmallVector<uint32_t, 5> element_strides;
  for (auto layout_dim : minor_to_major_layout) {
    element_strides.push_back(tile_strides[layout_dim]);
  }

  // When the tile strides are > 1, the box dimensions no longer reflect the
  // number of elements in the tile. To load the correct number of
  // elements, we need to multiply the tile strides by the number of elements in
  // the tile.
  llvm::SmallVector<uint32_t, 5> box_dims;
  for (auto layout_dim : minor_to_major_layout) {
    box_dims.push_back(static_cast<uint32_t>(tile_shape[layout_dim]) *
                       tile_strides[layout_dim]);
  }

  const TmaDescriptor::TmaInterleave default_interleave =
      TmaDescriptor::TmaInterleave::kNone;
  const TmaDescriptor::TmaSwizzle default_swizzle =
      TmaDescriptor::TmaSwizzle::kNone;
  const TmaDescriptor::TmaL2Promotion default_l2_promotion =
      TmaDescriptor::TmaL2Promotion::kNone;

  // Attempt to construct a TmaDescriptor with the default values. If this
  // fails, then TMA is not compatible.
  TF_ASSIGN_OR_RETURN(
      auto tma_desc, TmaDescriptor::Create(
                         normalized_global_shape, global_strides, box_dims,
                         element_strides, element_byte_size, default_interleave,
                         default_swizzle, default_l2_promotion));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace stream_executor
