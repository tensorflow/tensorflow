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

#ifndef XLA_UTIL_DYNAMIC_BITSET_H_
#define XLA_UTIL_DYNAMIC_BITSET_H_

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"

namespace xla {

class DynamicBitset {
 public:
  DynamicBitset() = default;
  explicit DynamicBitset(int64_t num_bits);

  void Add(int64_t index) {
    const BitPosition pos = GetBitPosition(index);
    if (pos.word_index >= words_.size()) {
      words_.resize(pos.word_index + 1, 0);
    }
    words_[pos.word_index] |= pos.mask;
  }

  void Clear(int64_t index) {
    const BitPosition pos = GetBitPosition(index);
    if (pos.word_index < words_.size()) {
      words_[pos.word_index] &= ~pos.mask;
    }
  }

  bool Contains(int64_t index) const {
    const BitPosition pos = GetBitPosition(index);
    if (pos.word_index >= words_.size()) {
      return false;
    }
    return (words_[pos.word_index] & pos.mask) != 0;
  }

  bool Empty() const {
    for (const uint64_t word : words_) {
      if (word != 0) {
        return false;
      }
    }
    return true;
  }

  void Merge(const DynamicBitset& other);

  bool operator==(const DynamicBitset& other) const;
  bool operator!=(const DynamicBitset& other) const {
    return !(*this == other);
  }

 private:
  struct BitPosition {
    uint64_t word_index;
    uint64_t mask;
  };

  static BitPosition GetBitPosition(int64_t index) {
    DCHECK_GE(index, 0);
    const uint64_t u_index = static_cast<uint64_t>(index);
    return {u_index / 64, uint64_t{1} << (u_index % 64)};
  }

  absl::InlinedVector<uint64_t, 4> words_;
};

}  // namespace xla

#endif  // XLA_UTIL_DYNAMIC_BITSET_H_
