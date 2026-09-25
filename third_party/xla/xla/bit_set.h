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

#ifndef XLA_BIT_SET_H_
#define XLA_BIT_SET_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

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

// rows bit sets of num_bits bits each, in one heap allocation. Rows are
// addressed by index; the operations across rows are ORs of one row into
// another. Useful when an analysis keeps one bit set per element of a large
// dense index space, for example per instruction, and would otherwise
// allocate each set separately.
class DenseBitSets {
 public:
  using Word = uint64_t;
  static constexpr int kWordSize = sizeof(Word) * 8;

  DenseBitSets(int64_t rows, int64_t num_bits)
      : rows_(rows),
        num_bits_(num_bits),
        words_per_row_((num_bits + kWordSize - 1) / kWordSize),
        words_(rows * words_per_row_, 0) {
    DCHECK_GE(rows, 0);
    DCHECK_GE(num_bits, 0);
  }

  int64_t rows() const { return rows_; }
  int64_t num_bits() const { return num_bits_; }

  void Set(int64_t row, int64_t bit) { Row(row)[WordIndex(bit)] |= Mask(bit); }

  bool Test(int64_t row, int64_t bit) const {
    return (Row(row)[WordIndex(bit)] & Mask(bit)) != 0;
  }

  // row |= other.
  void Or(int64_t row, int64_t other) {
    Word* dst = Row(row);
    const Word* src = Row(other);
    for (int64_t w = 0; w < words_per_row_; ++w) {
      dst[w] |= src[w];
    }
  }

  // row |= other, with ignored_bit of other left out.
  void OrIgnoringBit(int64_t row, int64_t other, int64_t ignored_bit) {
    Word* dst = Row(row);
    const Word* src = Row(other);
    const int64_t ignored_word = WordIndex(ignored_bit);
    for (int64_t w = 0; w < words_per_row_; ++w) {
      const Word keep = w == ignored_word ? ~Mask(ignored_bit) : ~Word{0};
      dst[w] |= src[w] & keep;
    }
  }

 private:
  static constexpr Word Mask(int64_t bit) {
    return Word{1} << (bit % kWordSize);
  }

  int64_t WordIndex(int64_t bit) const {
    DCHECK_GE(bit, 0);
    DCHECK_LT(bit, num_bits_);
    return bit / kWordSize;
  }

  Word* Row(int64_t row) {
    DCHECK_GE(row, 0);
    DCHECK_LT(row, rows_);
    return words_.data() + row * words_per_row_;
  }

  const Word* Row(int64_t row) const {
    DCHECK_GE(row, 0);
    DCHECK_LT(row, rows_);
    return words_.data() + row * words_per_row_;
  }

  int64_t rows_;
  int64_t num_bits_;
  int64_t words_per_row_;
  std::vector<Word> words_;
};

}  // namespace xla

#endif  // XLA_BIT_SET_H_
