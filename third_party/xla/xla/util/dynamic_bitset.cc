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

#include "xla/util/dynamic_bitset.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"

namespace xla {

DynamicBitset::DynamicBitset(int64_t num_bits) {
  DCHECK_GE(num_bits, 0);
  words_.resize((num_bits + 63) / 64, 0);
}

void DynamicBitset::Merge(const DynamicBitset& other) {
  if (other.words_.size() > words_.size()) {
    words_.resize(other.words_.size(), 0);
  }
  for (size_t i = 0; i < other.words_.size(); ++i) {
    words_[i] |= other.words_[i];
  }
}

bool DynamicBitset::operator==(const DynamicBitset& other) const {
  const size_t min_size = std::min(words_.size(), other.words_.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (words_[i] != other.words_[i]) {
      return false;
    }
  }
  if (words_.size() > min_size) {
    return std::all_of(words_.begin() + min_size, words_.end(),
                       [](const uint64_t w) { return w == 0; });
  }
  if (other.words_.size() > min_size) {
    return std::all_of(other.words_.begin() + min_size, other.words_.end(),
                       [](const uint64_t w) { return w == 0; });
  }
  return true;
}

}  // namespace xla
