/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/dtype.h"

#include <optional>
#include <ostream>
#include <string>

#include "absl/strings/str_cat.h"

namespace xla {
namespace ifrt {

std::optional<int> DType::byte_size() const {
  switch (kind_) {
    case kS8:
    case kU8:
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
    default:
      return std::nullopt;
  }
}

std::optional<int> DType::bit_size() const {
  switch (kind_) {
    case kPred:
      return 1;
    case kS8:
    case kU8:
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
    default:
      return std::nullopt;
  }
}

std::string DType::DebugString() const {
  switch (kind_) {
    case kInvalid:
      return "INVALID";
    case kPred:
      return "PRED";
    case kS8:
      return "S8";
    case kS16:
      return "S16";
    case kS32:
      return "S32";
    case kS64:
      return "S64";
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
