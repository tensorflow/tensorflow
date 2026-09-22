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

#ifndef XLA_INLINED_BIT_SET_H_
#define XLA_INLINED_BIT_SET_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"

namespace xla {

// A fixed-size bit set (or bitmap) that stores up to `InlinedWords * 64` bits
// inline without heap allocation, spilling to the heap only when `num_bits`
// exceeds this capacity. Useful for tracking sets of dense integer indices
// over a typically small range `[0, num_bits)`.
template <size_t InlinedWords = 1>
class InlinedBitSet {
 public:
  using Word = uint64_t;
  static constexpr int kWordSize = sizeof(Word) * 8;

  InlinedBitSet() = default;
  explicit InlinedBitSet(int num_bits) : words_(NumWords(num_bits), 0) {}

  void Set(int index) {
    DCHECK_GE(index, 0);
    const size_t word_idx = IndexToWord(index);
    words_[word_idx] |= (Word{1} << (index % kWordSize));
  }

  // Sets all bits in [0, num_bits) to 1
  void SetAll(int num_bits) {
    DCHECK_GE(num_bits, 0);
    DCHECK_LE(static_cast<size_t>(num_bits), words_.size() * kWordSize);
    const size_t full_words = num_bits / kWordSize;
    std::fill_n(words_.begin(), full_words, ~Word{0});
    const size_t remainder = num_bits % kWordSize;
    if (remainder != 0) {
      words_[full_words] |= (Word{1} << remainder) - 1;
    }
  }

  void Clear(int index) {
    const size_t word_idx = IndexToWord(index);
    words_[word_idx] &= ~(Word{1} << (index % kWordSize));
  }

  bool Test(int index) const {
    const size_t word_idx = IndexToWord(index);
    return (words_[word_idx] & (Word{1} << (index % kWordSize))) != 0;
  }

  InlinedBitSet& operator|=(const InlinedBitSet& other) {
    DCHECK_EQ(words_.size(), other.words_.size());
    for (size_t i = 0; i < words_.size(); ++i) {
      words_[i] |= other.words_[i];
    }
    return *this;
  }

 private:
  static size_t NumWords(int num_bits) {
    DCHECK_GE(num_bits, 0);
    return (num_bits + kWordSize - 1) / kWordSize;
  }

  size_t IndexToWord(int index) const {
    DCHECK_GE(index, 0);
    const size_t word_idx = index / kWordSize;
    DCHECK_LT(word_idx, words_.size());
    return word_idx;
  }

  absl::InlinedVector<Word, InlinedWords> words_;
};

}  // namespace xla

#endif  // XLA_INLINED_BIT_SET_H_
