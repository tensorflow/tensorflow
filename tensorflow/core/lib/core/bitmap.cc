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

#include "tensorflow/core/lib/core/bitmap.h"

#include <string.h>

namespace tensorflow {
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
static size_t FindFirstSet(uint32 w) {
  // TODO(jeff,sanjay): If this becomes a performance issue, we could
  // use the __builtin_ffs(w) routine on GCC, or the ffs(w) routine on
  // some other platforms.

  // clang-format off
  static uint8 kLowestBitSet[256] = {
    /*  0*/ 0,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 16*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 32*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 48*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 64*/ 7,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 80*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 96*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*112*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*128*/ 8,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*144*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*160*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*176*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*192*/ 7,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*208*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*224*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*240*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
  };
  // clang-format on
  if (w & 0xff) {
    return kLowestBitSet[w & 0xff];
  } else if ((w >> 8) & 0xff) {
    return kLowestBitSet[(w >> 8) & 0xff] + 8;
  } else if ((w >> 16) & 0xff) {
    return kLowestBitSet[(w >> 16) & 0xff] + 16;
  } else if ((w >> 24) & 0xff) {
    return kLowestBitSet[(w >> 24) & 0xff] + 24;
  } else {
    return 0;
  }
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

string Bitmap::ToString() const {
  string result;
  result.resize(bits());
  for (size_t i = 0; i < nbits_; i++) {
    result[i] = get(i) ? '1' : '0';
  }
  return result;
}

}  // namespace core
}  // namespace tensorflow
