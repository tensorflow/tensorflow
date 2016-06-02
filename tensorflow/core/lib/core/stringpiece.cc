/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/core/stringpiece.h"

#include <algorithm>
#include <iostream>
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

size_t StringPiece::Hasher::operator()(StringPiece s) const {
  return Hash64(s.data(), s.size());
}

std::ostream& operator<<(std::ostream& o, StringPiece piece) {
  o.write(piece.data(), piece.size());
  return o;
}

bool StringPiece::contains(StringPiece s) const {
  return std::search(begin(), end(), s.begin(), s.end()) != end();
}

size_t StringPiece::find(char c, size_t pos) const {
  if (pos >= size_) {
    return npos;
  }
  const char* result =
      reinterpret_cast<const char*>(memchr(data_ + pos, c, size_ - pos));
  return result != NULL ? result - data_ : npos;
}

// Search range is [0..pos] inclusive.  If pos == npos, search everything.
size_t StringPiece::rfind(char c, size_t pos) const {
  if (size_ == 0) return npos;
  for (const char* p = data_ + std::min(pos, size_ - 1); p >= data_; p--) {
    if (*p == c) {
      return p - data_;
    }
  }
  return npos;
}

StringPiece StringPiece::substr(size_t pos, size_t n) const {
  if (pos > size_) pos = size_;
  if (n > size_ - pos) n = size_ - pos;
  return StringPiece(data_ + pos, n);
}

const StringPiece::size_type StringPiece::npos = size_type(-1);

}  // namespace tensorflow
