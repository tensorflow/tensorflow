/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/dtype.h"

#include <optional>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/dtype.pb.h"

namespace xla {
namespace ifrt {

std::optional<int> DType::byte_size() const {
  switch (kind_) {
    case kS2:
    case kU2:
    case kS4:
    case kU4:
    case kF4E2M1FN:
      // Smaller than a byte.
      return std::nullopt;
    case kPred:
    case kS8:
    case kU8:
    case kF8E3M4:
    case kF8E4M3:
    case kF8E8M0FNU:
    // The following types are https://arxiv.org/abs/2209.05433
    case kF8E4M3FN:
    case kF8E4M3B11FNUZ:
    case kF8E4M3FNUZ:
    case kF8E5M2:
    case kF8E5M2FNUZ:
      return 1;
    case kS16:
    case kU16:
    case kF16:
    case kBF16:
      return 2;
    case kS32:
    case kU32:
    case kF32:
      return 4;
    case kS64:
    case kU64:
    case kF64:
    case kC64:
      return 8;
    case kC128:
      return 16;
    case kToken:
    case kOpaque:
    case kInvalid:
    case kString:
      return std::nullopt;
  }
}

std::optional<int> DType::bit_size() const {
  switch (kind_) {
    case kS2:
    case kU2:
      return 2;
    case kS4:
    case kU4:
    case kF4E2M1FN:
      return 4;
    case kPred:
    case kS8:
    case kU8:
    case kF8E3M4:
    case kF8E4M3:
    case kF8E8M0FNU:
    // The following types are https://arxiv.org/abs/2209.05433
    case kF8E4M3FN:
    case kF8E4M3B11FNUZ:
    case kF8E4M3FNUZ:
    case kF8E5M2:
    case kF8E5M2FNUZ:
      return 8;
    case kS16:
    case kU16:
    case kF16:
    case kBF16:
      return 16;
    case kS32:
    case kU32:
    case kF32:
      return 32;
    case kS64:
    case kU64:
    case kF64:
    case kC64:
      return 64;
    case kC128:
      return 128;
    case kToken:
    case kOpaque:
    case kInvalid:
    case kString:
      return std::nullopt;
  }
}

absl::StatusOr<DType> DType::FromProto(const DTypeProto& dtype_proto) {
  switch (dtype_proto.kind()) {
    case DTypeProto::KIND_PRED:
      return DType(DType::Kind::kPred);
    case DTypeProto::KIND_TOKEN:
      return DType(DType::Kind::kToken);
    case DTypeProto::KIND_OPAQUE:
      return DType(DType::Kind::kOpaque);
#define CASE(X)              \
  case DTypeProto::KIND_##X: \
    return DType(DType::Kind::k##X);
      CASE(S4);
      CASE(S8);
      CASE(S16);
      CASE(S32);
      CASE(S64);
      CASE(U4);
      CASE(U8);
      CASE(U16);
      CASE(U32);
      CASE(U64);
      CASE(F16);
      CASE(F32);
      CASE(F64);
      CASE(BF16);
      CASE(C64);
      CASE(C128);
      CASE(F4E2M1FN);
      CASE(F8E3M4);
      CASE(F8E4M3);
      CASE(F8E8M0FNU);
      CASE(F8E4M3FN);
      CASE(F8E4M3B11FNUZ);
      CASE(F8E4M3FNUZ);
      CASE(F8E5M2);
      CASE(F8E5M2FNUZ);
#undef CASE
    case DTypeProto::KIND_STRING:
      return DType(DType::Kind::kString);
    default:
      return DType(DType::Kind::kInvalid);
  }
}

DTypeProto DType::ToProto() const {
  DTypeProto dtype_proto;
  switch (kind()) {
    case DType::Kind::kPred:
      dtype_proto.set_kind(DTypeProto::KIND_PRED);
      break;
    case DType::Kind::kToken:
      dtype_proto.set_kind(DTypeProto::KIND_TOKEN);
      break;
    case DType::Kind::kOpaque:
      dtype_proto.set_kind(DTypeProto::KIND_OPAQUE);
      break;
#define CASE(X)                                 \
  case DType::Kind::k##X:                       \
    dtype_proto.set_kind(DTypeProto::KIND_##X); \
    break;
      CASE(S4);
      CASE(S8);
      CASE(S16);
      CASE(S32);
      CASE(S64);
      CASE(U4);
      CASE(U8);
      CASE(U16);
      CASE(U32);
      CASE(U64);
      CASE(F16);
      CASE(F32);
      CASE(F64);
      CASE(BF16);
      CASE(C64);
      CASE(C128);
      CASE(F4E2M1FN);
      CASE(F8E3M4);
      CASE(F8E4M3);
      CASE(F8E8M0FNU);
      CASE(F8E4M3FN);
      CASE(F8E4M3B11FNUZ);
      CASE(F8E4M3FNUZ);
      CASE(F8E5M2);
      CASE(F8E5M2FNUZ);
#undef CASE
    case DType::Kind::kString:
      dtype_proto.set_kind(DTypeProto::KIND_STRING);
      break;
    default:
      dtype_proto.set_kind(DTypeProto::KIND_UNSPECIFIED);
      break;
  }
  return dtype_proto;
}

std::string DType::DebugString() const {
  switch (kind_) {
    case kInvalid:
      return "INVALID";
    case kPred:
      return "PRED";
    case kS2:
      return "S2";
    case kS4:
      return "S4";
    case kS8:
      return "S8";
    case kS16:
      return "S16";
    case kS32:
      return "S32";
    case kS64:
      return "S64";
    case kU2:
      return "U2";
    case kU4:
      return "U4";
    case kU8:
      return "U8";
    case kU16:
      return "U16";
    case kU32:
      return "U32";
    case kU64:
      return "U64";
    case kF16:
      return "F16";
    case kF32:
      return "F32";
    case kF64:
      return "F64";
    case kBF16:
      return "BF16";
    case kC64:
      return "C64";
    case kC128:
      return "C128";
    case kToken:
      return "TOKEN";
    case kOpaque:
      return "OPAQUE";
    case kF4E2M1FN:
      return "F4E2M1FN";
    case kF8E3M4:
      return "F8E3M4";
    case kF8E4M3:
      return "F8E4M3";
    case kF8E4M3FN:
      return "F8E4M3FN";
    case kF8E4M3B11FNUZ:
      return "F8E4M3B11FNUZ";
    case kF8E4M3FNUZ:
      return "F8E4M3FNUZ";
    case kF8E5M2:
      return "F8E5M2";
    case kF8E5M2FNUZ:
      return "F8E5M2FNUZ";
    case kF8E8M0FNU:
      return "F8E8M0FNU";
    case kString:
      return "STRING";
    default:
      return absl::StrCat("UNKNOWN(", static_cast<int>(kind_), ")");
  }
}

std::ostream& operator<<(std::ostream& os, const DType& dtype) {
  return os << dtype.DebugString();
}

}  // namespace ifrt
}  // namespace xla
