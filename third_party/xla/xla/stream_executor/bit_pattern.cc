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

#include "xla/stream_executor/bit_pattern.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "absl/strings/str_cat.h"

namespace stream_executor {
namespace {

// Broadcasts a pattern value of 1/2/4 bytes to a 4 byte value.
struct BitPatternToValue {
  unsigned operator()(uint8_t pattern) const {
    return (pattern << 24) | (pattern << 16) | (pattern << 8) | pattern;
  }
  unsigned operator()(uint16_t pattern) const {
    return (pattern << 16) | pattern;
  }
  unsigned operator()(uint32_t pattern) const { return pattern; }
};

struct BitPatternToString {
  std::string operator()(uint8_t pattern) const {
    return absl::StrCat("u8:", pattern);
  }
  std::string operator()(uint16_t pattern) const {
    return absl::StrCat("u16:", pattern);
  }
  std::string operator()(uint32_t pattern) const {
    return absl::StrCat("u32:", pattern);
  }
};
}  // namespace

uint32_t BitPattern::GetPatternBroadcastedToUint32() const {
  return std::visit(
      BitPatternToValue(),
      static_cast<const std::variant<uint8_t, uint16_t, uint32_t>&>(*this));
}
std::string BitPattern::ToString() const {
  return std::visit(
      BitPatternToString(),
      static_cast<const std::variant<uint8_t, uint16_t, uint32_t>&>(*this));
}

size_t BitPattern::GetElementSize() const {
  // uint8_t, idx=0, return 1
  // uint16_t, idx=1, return 2
  // uint32_t, idx=2, return 4
  return 1 << index();
}
}  // namespace stream_executor
