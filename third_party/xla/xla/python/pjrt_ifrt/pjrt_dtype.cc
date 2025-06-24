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

#include "xla/python/pjrt_ifrt/pjrt_dtype.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype) {
  switch (dtype.kind()) {
#define CASE(DT, PT)                                                      \
  case DT:                                                                \
    static_assert(PT ==                                                   \
                  static_cast<xla::PrimitiveType>(static_cast<int>(DT))); \
    return PT
    CASE(DType::kInvalid, xla::PrimitiveType::PRIMITIVE_TYPE_INVALID);
    CASE(DType::kPred, xla::PrimitiveType::PRED);
    CASE(DType::kS2, xla::PrimitiveType::S2);
    CASE(DType::kS4, xla::PrimitiveType::S4);
    CASE(DType::kS8, xla::PrimitiveType::S8);
    CASE(DType::kS16, xla::PrimitiveType::S16);
    CASE(DType::kS32, xla::PrimitiveType::S32);
    CASE(DType::kS64, xla::PrimitiveType::S64);
    CASE(DType::kU2, xla::PrimitiveType::U2);
    CASE(DType::kU4, xla::PrimitiveType::U4);
    CASE(DType::kU8, xla::PrimitiveType::U8);
    CASE(DType::kU16, xla::PrimitiveType::U16);
    CASE(DType::kU32, xla::PrimitiveType::U32);
    CASE(DType::kU64, xla::PrimitiveType::U64);
    CASE(DType::kF4E2M1FN, xla::PrimitiveType::F4E2M1FN);
    CASE(DType::kF8E3M4, xla::PrimitiveType::F8E3M4);
    CASE(DType::kF8E4M3, xla::PrimitiveType::F8E4M3);
    CASE(DType::kF8E4M3FN, xla::PrimitiveType::F8E4M3FN);
    CASE(DType::kF8E4M3B11FNUZ, xla::PrimitiveType::F8E4M3B11FNUZ);
    CASE(DType::kF8E4M3FNUZ, xla::PrimitiveType::F8E4M3FNUZ);
    CASE(DType::kF8E5M2, xla::PrimitiveType::F8E5M2);
    CASE(DType::kF8E5M2FNUZ, xla::PrimitiveType::F8E5M2FNUZ);
    CASE(DType::kF8E8M0FNU, xla::PrimitiveType::F8E8M0FNU);
    CASE(DType::kF16, xla::PrimitiveType::F16);
    CASE(DType::kF32, xla::PrimitiveType::F32);
    CASE(DType::kBF16, xla::PrimitiveType::BF16);
    CASE(DType::kF64, xla::PrimitiveType::F64);
    CASE(DType::kC64, xla::PrimitiveType::C64);
    CASE(DType::kC128, xla::PrimitiveType::C128);
    CASE(DType::kToken, xla::PrimitiveType::TOKEN);
    CASE(DType::kOpaque, xla::PrimitiveType::OPAQUE_TYPE);
#undef CASE
    case DType::kString:
      return absl::InvalidArgumentError(
          absl::StrCat("Not supported as XLA PrimitiveType: ",
                       static_cast<int>(dtype.kind())));
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid DType: ", static_cast<int>(dtype.kind())));
}

absl::StatusOr<DType> ToDType(xla::PrimitiveType primitive_type) {
  switch (primitive_type) {
    case xla::PrimitiveType::PRIMITIVE_TYPE_INVALID:
    case xla::PrimitiveType::PRED:
    case xla::PrimitiveType::S2:
    case xla::PrimitiveType::S4:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U2:
    case xla::PrimitiveType::U4:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::U64:
    case xla::PrimitiveType::F4E2M1FN:
    case xla::PrimitiveType::F8E3M4:
    case xla::PrimitiveType::F8E4M3:
    case xla::PrimitiveType::F8E4M3FN:
    case xla::PrimitiveType::F8E4M3B11FNUZ:
    case xla::PrimitiveType::F8E4M3FNUZ:
    case xla::PrimitiveType::F8E5M2:
    case xla::PrimitiveType::F8E5M2FNUZ:
    case xla::PrimitiveType::F8E8M0FNU:
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F64:
    case xla::PrimitiveType::C64:
    case xla::PrimitiveType::C128:
    case xla::PrimitiveType::TOKEN:
    case xla::PrimitiveType::OPAQUE_TYPE:
      return DType(static_cast<DType::Kind>(static_cast<int>(primitive_type)));
    default:
      return absl::InvalidArgumentError(
          absl::Substitute("Invalid XLA PrimitiveType: $0 ($1)",
                           static_cast<int>(primitive_type),
                           xla::PrimitiveType_Name(primitive_type)));
  }
}

}  // namespace ifrt
}  // namespace xla
