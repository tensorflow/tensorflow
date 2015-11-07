#include "tensorflow/core/lib/core/stringpiece.h"

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
  return memmem(data_, size_, s.data_, s.size_) != nullptr;
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

bool StringPiece::Consume(StringPiece x) {
  if (starts_with(x)) {
    remove_prefix(x.size_);
    return true;
  }
  return false;
}

StringPiece StringPiece::substr(size_t pos, size_t n) const {
  if (pos > size_) pos = size_;
  if (n > size_ - pos) n = size_ - pos;
  return StringPiece(data_ + pos, n);
}

const StringPiece::size_type StringPiece::npos = size_type(-1);

}  // namespace tensorflow
