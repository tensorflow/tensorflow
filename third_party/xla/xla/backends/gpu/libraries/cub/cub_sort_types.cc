/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/cub/cub_sort_types.h"

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

bool IsCubSortKeysOnlyKeySupported(PrimitiveType key_type) {
  switch (key_type) {
    case BF16:
    case F16:
    case F32:
    case F64:
    case S8:
    case S16:
    case S32:
    case S64:
    case U8:
    case U16:
    case U32:
    case U64:
      return true;
    default:
      return false;
  }
}

bool IsCubSortPairKeySupported(PrimitiveType key_type) {
  switch (key_type) {
    case U8:
    case U16:
    case S32:
    case U32:
    case U64:
    case F32:
      return true;
    default:
      return false;
  }
}

bool IsCubSortPairValueBitWidthSupported(int value_bit_width) {
  return value_bit_width == 16 || value_bit_width == 32 ||
         value_bit_width == 64;
}

}  // namespace

bool IsCubSortSupported(PrimitiveType key_type,
                        std::optional<PrimitiveType> value_type) {
  if (!value_type.has_value()) {
    return IsCubSortKeysOnlyKeySupported(key_type);
  }
  if (!IsCubSortPairKeySupported(key_type) ||
      !primitive_util::IsArrayType(*value_type)) {
    return false;
  }
  return IsCubSortPairValueBitWidthSupported(
      primitive_util::BitWidth(*value_type));
}

absl::Status ValidateCubSortTypes(PrimitiveType key_type,
                                  std::optional<PrimitiveType> value_type) {
  if (!value_type.has_value()) {
    if (!IsCubSortKeysOnlyKeySupported(key_type)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUDA CUB keys-only sort: ",
                       PrimitiveType_Name(key_type)));
    }
    return absl::OkStatus();
  }

  if (!IsCubSortPairKeySupported(key_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported key type for CUDA CUB key-value sort: ",
                     PrimitiveType_Name(key_type)));
  }

  if (!primitive_util::IsArrayType(*value_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported non-array value type for CUDA CUB sort: ",
                     PrimitiveType_Name(*value_type)));
  }
  const int val_bits = primitive_util::BitWidth(*value_type);
  if (!IsCubSortPairValueBitWidthSupported(val_bits)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported value bit width (", val_bits,
                     ") for CUDA CUB sort: ", PrimitiveType_Name(*value_type)));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
