/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_CORE_BITMAP_H_
#define TENSORFLOW_CORE_LIB_CORE_BITMAP_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace core {

class Bitmap {
 public:
  // Create a bitmap that holds 0 bits.
  Bitmap();

  // Create a bitmap that holds n bits, all initially zero.
  explicit Bitmap(size_t n);

  ~Bitmap();

  Bitmap(const Bitmap&) = delete;
  Bitmap& operator=(const Bitmap&) = delete;

  // Return the number of bits that the bitmap can hold.
  size_t bits() const;

  // Replace contents of *this with a bitmap of n bits, all set to zero.
  void Reset(size_t n);

  // Return the contents of the ith bit.
  // REQUIRES: i < bits()
  bool get(size_t i) const;

  // Set the contents of the ith bit to true.
  // REQUIRES: i < bits()
  void set(size_t i);

  // Set the contents of the ith bit to false.
  // REQUIRES: i < bits()
  void clear(size_t i);

  // Return the smallest i such that i >= start and !get(i).
  // Returns bits() if no such i exists.
  size_t FirstUnset(size_t start) const;

  // Returns the bitmap as an ascii string of '0' and '1' characters, bits()
  // characters in length.
  string ToString() const;

 private:
  typedef uint32 Word;
  static constexpr size_t kBits = 32;

  // Return the number of words needed to store n bits.
  static size_t NumWords(size_t n) { return (n + kBits - 1) / kBits; }

  // Return the mask to use for the ith bit in a word.
  static Word Mask(size_t i) { return 1ull << i; }

  size_t nbits_;  // Length of bitmap in bits.
  Word* word_;
};

// Implementation details follow.  Clients should ignore.

inline Bitmap::Bitmap() : nbits_(0), word_(nullptr) {}

inline Bitmap::Bitmap(size_t n) : Bitmap() { Reset(n); }

inline Bitmap::~Bitmap() { delete[] word_; }

inline size_t Bitmap::bits() const { return nbits_; }

inline bool Bitmap::get(size_t i) const {
  DCHECK_LT(i, nbits_);
  return word_[i / kBits] & Mask(i % kBits);
}

inline void Bitmap::set(size_t i) {
  DCHECK_LT(i, nbits_);
  word_[i / kBits] |= Mask(i % kBits);
}

inline void Bitmap::clear(size_t i) {
  DCHECK_LT(i, nbits_);
  word_[i / kBits] &= ~Mask(i % kBits);
}

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_BITMAP_H_
