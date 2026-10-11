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

// Holds num_rows bit sets of num_bits bits each in a single heap allocation.
// Useful when an analysis keeps one bit set (of the same size) per element of
// a large dense index space, for example per instruction: one allocation
// instead of one per element, and the rows sit next to each other in memory,
// so an OR of one row into another streams over two contiguous word ranges.
// Rows are addressed by index.
//
// Supported ops:
// - Set: sets one bit of a row.
// - Test: reads one bit of a row.
// - Or: ORs one row into another.
// - OrIgnoringBit: ORs one row into another, leaving one bit out.
class DenseBitSets {
 public:
  using Word = uint64_t;
  static constexpr int kWordSize = sizeof(Word) * 8;

  DenseBitSets(int num_rows, int num_bits)
      : num_rows_(CheckNonNegative(num_rows)),
        num_bits_(CheckNonNegative(num_bits)),
        words_per_row_(num_bits_ / kWordSize + (num_bits_ % kWordSize != 0)),
        words_(static_cast<size_t>(num_rows_) * words_per_row_, 0) {}

  int num_rows() const { return num_rows_; }
  int num_bits() const { return num_bits_; }

  // Sets bit 'bit' of row 'row'.
  void Set(int row, int bit) { Row(row)[WordIndex(bit)] |= Mask(bit); }

  // Returns whether bit 'bit' of row 'row' is set.
  bool Test(int row, int bit) const {
    return (Row(row)[WordIndex(bit)] & Mask(bit)) != 0;
  }

  // row |= other.
  void Or(int row, int other) {
    Word* dst = Row(row);
    const Word* src = Row(other);
    for (int w = 0; w < words_per_row_; ++w) {
      dst[w] |= src[w];
    }
  }

  // row |= other, with ignored_bit of other left out: a plain Or, which the
  // compiler vectorizes, then the destination's own value of the bit is put
  // back.
  void OrIgnoringBit(int row, int other, int ignored_bit) {
    Word& word = Row(row)[WordIndex(ignored_bit)];
    const Word own_bit = word & Mask(ignored_bit);
    Or(row, other);
    word = (word & ~Mask(ignored_bit)) | own_bit;
  }

 private:
  static int CheckNonNegative(int count) {
    DCHECK_GE(count, 0);
    return count;
  }

  static constexpr Word Mask(int bit) { return Word{1} << (bit % kWordSize); }

  int WordIndex(int bit) const {
    DCHECK_GE(bit, 0);
    DCHECK_LT(bit, num_bits_);
    return bit / kWordSize;
  }

  // The word offset of a row; computed in size_t because the whole block can
  // hold more than 2^31 words.
  size_t RowOffset(int row) const {
    DCHECK_GE(row, 0);
    DCHECK_LT(row, num_rows_);
    return static_cast<size_t>(row) * words_per_row_;
  }

  Word* Row(int row) { return words_.data() + RowOffset(row); }
  const Word* Row(int row) const { return words_.data() + RowOffset(row); }

  int num_rows_;
  int num_bits_;
  int words_per_row_;
  std::vector<Word> words_;
};

}  // namespace xla

#endif  // XLA_BIT_SET_H_
