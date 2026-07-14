/* Copyright 2025 The JAX Authors

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

#include "xla/python/dlpack_types.h"

#include "absl/status/statusor.h"
#include "include/dlpack/dlpack.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<DLDataType> PrimitiveTypeToDLDataType(PrimitiveType type) {
  switch (type) {
    case S8:
      return DLDataType{kDLInt, 8, 1};
    case S16:
      return DLDataType{kDLInt, 16, 1};
    case S32:
      return DLDataType{kDLInt, 32, 1};
    case S64:
      return DLDataType{kDLInt, 64, 1};
    case U8:
      return DLDataType{kDLUInt, 8, 1};
    case U16:
      return DLDataType{kDLUInt, 16, 1};
    case U32:
      return DLDataType{kDLUInt, 32, 1};
    case U64:
      return DLDataType{kDLUInt, 64, 1};
    case F4E2M1FN:
      return DLDataType{kDLFloat4_e2m1fn, 4, 1};
    case F8E3M4:
      return DLDataType{kDLFloat8_e3m4, 8, 1};
    case F8E4M3:
      return DLDataType{kDLFloat8_e4m3, 8, 1};
    case F8E4M3B11FNUZ:
      return DLDataType{kDLFloat8_e4m3b11fnuz, 8, 1};
    case F8E4M3FN:
      return DLDataType{kDLFloat8_e4m3fn, 8, 1};
    case F8E4M3FNUZ:
      return DLDataType{kDLFloat8_e4m3fnuz, 8, 1};
    case F8E5M2:
      return DLDataType{kDLFloat8_e5m2, 8, 1};
    case F8E5M2FNUZ:
      return DLDataType{kDLFloat8_e5m2fnuz, 8, 1};
    case F8E8M0FNU:
      return DLDataType{kDLFloat8_e8m0fnu, 8, 1};
    case BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case F16:
      return DLDataType{kDLFloat, 16, 1};
    case F32:
      return DLDataType{kDLFloat, 32, 1};
    case F64:
      return DLDataType{kDLFloat, 64, 1};
    case PRED:
      return DLDataType{kDLBool, 8, 1};
    case C64:
      return DLDataType{kDLComplex, 64, 1};
    case C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      return Unimplemented("XLA type %s has no DLPack equivalent",
                           PrimitiveType_Name(type));
  }
}

absl::StatusOr<PrimitiveType> DLDataTypeToPrimitiveType(DLDataType type) {
  if (type.lanes != 1) {
    return Unimplemented("DLPack types with lanes != 1 not implemented, got %d",
                         type.lanes);
  }
  switch (type.code) {
    case kDLBool:
      switch (type.bits) {
        case 8:
          return PRED;
        default:
          return Unimplemented(
              "Only 8-bit DLPack booleans are supported, got %d bits",
              type.bits);
      }
    case kDLInt:
      switch (type.bits) {
        case 8:
          return S8;
        case 16:
          return S16;
        case 32:
          return S32;
        case 64:
          return S64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack integer width: %d bits",
              type.bits);
      }
    case kDLUInt:
      switch (type.bits) {
        case 8:
          return U8;
        case 16:
          return U16;
        case 32:
          return U32;
        case 64:
          return U64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack unsigned integer width: %d bits",
              type.bits);
      }
    case kDLFloat4_e2m1fn:
      if (type.bits == 4) {
        return F4E2M1FN;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float4_e2m1fn width: %d bits",
          type.bits);
    case kDLFloat8_e3m4:
      if (type.bits == 8) {
        return F8E3M4;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e3m4 width: %d bits",
          type.bits);
    case kDLFloat8_e4m3:
      if (type.bits == 8) {
        return F8E4M3;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e4m3 width: %d bits",
          type.bits);
    case kDLFloat8_e4m3b11fnuz:
      if (type.bits == 8) {
        return F8E4M3B11FNUZ;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e4m3b11fnuz width: %d bits",
          type.bits);
    case kDLFloat8_e4m3fn:
      if (type.bits == 8) {
        return F8E4M3FN;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e4m3fn width: %d bits",
          type.bits);
    case kDLFloat8_e4m3fnuz:
      if (type.bits == 8) {
        return F8E4M3FNUZ;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e4m3fnuz width: %d bits",
          type.bits);
    case kDLFloat8_e5m2:
      if (type.bits == 8) {
        return F8E5M2;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e5m2 width: %d bits",
          type.bits);
    case kDLFloat8_e5m2fnuz:
      if (type.bits == 8) {
        return F8E5M2FNUZ;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e5m2fnuz width: %d bits",
          type.bits);
    case kDLFloat8_e8m0fnu:
      if (type.bits == 8) {
        return F8E8M0FNU;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack float8_e8m0fnu width: %d bits",
          type.bits);
    case kDLBfloat:
      if (type.bits == 16) {
        return BF16;
      }
      return Unimplemented(
          "Invalid or unsupported DLPack bfloat width: %d bits", type.bits);
    case kDLFloat:
      switch (type.bits) {
        case 16:
          return F16;
        case 32:
          return F32;
        case 64:
          return F64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack float width: %d bits", type.bits);
      }
    case kDLComplex:
      switch (type.bits) {
        case 64:
          return C64;
        case 128:
          return C128;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack complex width: %d bits",
              type.bits);
      }
    default:
      return Unimplemented("Unknown or invalid DLPack type code %d", type.code);
  }
}

}  // namespace xla
