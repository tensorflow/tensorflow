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

#ifndef XLA_STREAM_EXECUTOR_BIT_PATTERN_H_
#define XLA_STREAM_EXECUTOR_BIT_PATTERN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace stream_executor {

// BitPattern represents a 4 byte bit pattern. It can be constructed from either
// a 8 bit, 16 bit or 32 bit pattern and it gets broadcasted to a uint32_t.
class BitPattern : public std::variant<uint8_t, uint16_t, uint32_t> {
 public:
  using std::variant<uint8_t, uint16_t, uint32_t>::variant;

  // Returns the size of the pattern in bytes.
  size_t GetElementSize() const;

  // Returns the pattern broadcasted to a uint32_t.
  uint32_t GetPatternBroadcastedToUint32() const;

  // Returns a string representation of the pattern - mainly meant for debugging
  // and logging.
  std::string ToString() const;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_BIT_PATTERN_H_
