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

#include "xla/tsl/lib/core/bitmap.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/numeric/bits.h"

namespace tsl {
namespace core {

void Bitmap::Reset(size_t n) {
  const size_t num_words = NumWords(n);
  if (num_words != NumWords(nbits_)) {
    // Reallocate.
    Word* w = new Word[num_words];
    delete[] word_;
    word_ = w;
  }
  memset(word_, 0, sizeof(word_[0]) * num_words);
  nbits_ = n;
}

// Return 1+index of the first set bit in w; return 0 if w == 0.
static size_t FindFirstSet(uint32_t w) {
  return w == 0 ? 0 : absl::countr_zero(w) + 1;
}

size_t Bitmap::FirstUnset(size_t start) const {
  if (start >= nbits_) {
    return nbits_;
  }

  // Mask to or-into first word to account for bits to skip in that word.
  size_t mask = (1ull << (start % kBits)) - 1;
  const size_t nwords = NumWords(nbits_);
  for (size_t i = start / kBits; i < nwords; i++) {
    Word word = word_[i] | mask;
    mask = 0;  // Only ignore bits in the first word we process.
    size_t r = FindFirstSet(~word);

    if (r) {
      size_t result = i * kBits + (r - 1);
      if (result > nbits_) result = nbits_;
      return result;
    }
  }

  return nbits_;
}

std::string Bitmap::ToString() const {
  std::string result;
  result.resize(bits());
  for (size_t i = 0; i < nbits_; i++) {
    result[i] = get(i) ? '1' : '0';
  }
  return result;
}

}  // namespace core
}  // namespace tsl
