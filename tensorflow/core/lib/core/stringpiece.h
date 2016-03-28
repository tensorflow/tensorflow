/* Copyright 2015 Google Inc. All Rights Reserved.

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

// StringPiece is a simple structure containing a pointer into some external
// storage and a size.  The user of a StringPiece must ensure that the slice
// is not used after the corresponding external storage has been
// deallocated.
//
// Multiple threads can invoke const methods on a StringPiece without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same StringPiece must use
// external synchronization.

#ifndef TENSORFLOW_LIB_CORE_STRINGPIECE_H_
#define TENSORFLOW_LIB_CORE_STRINGPIECE_H_

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <iosfwd>
#include <string>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class StringPiece {
 public:
  typedef size_t size_type;

  // Create an empty slice.
  StringPiece() : data_(""), size_(0) {}

  // Create a slice that refers to d[0,n-1].
  StringPiece(const char* d, size_t n) : data_(d), size_(n) {}

  // Create a slice that refers to the contents of "s"
  StringPiece(const string& s) : data_(s.data()), size_(s.size()) {}

  // Create a slice that refers to s[0,strlen(s)-1]
  StringPiece(const char* s) : data_(s), size_(strlen(s)) {}

  void set(const void* data, size_t len) {
    data_ = reinterpret_cast<const char*>(data);
    size_ = len;
  }

  // Return a pointer to the beginning of the referenced data
  const char* data() const { return data_; }

  // Return the length (in bytes) of the referenced data
  size_t size() const { return size_; }

  // Return true iff the length of the referenced data is zero
  bool empty() const { return size_ == 0; }

  typedef const char* const_iterator;
  typedef const char* iterator;
  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }

  static const size_t npos;

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()
  char operator[](size_t n) const {
    assert(n < size());
    return data_[n];
  }

  // Change this slice to refer to an empty array
  void clear() {
    data_ = "";
    size_ = 0;
  }

  // Drop the first "n" bytes from this slice.
  void remove_prefix(size_t n) {
    assert(n <= size());
    data_ += n;
    size_ -= n;
  }

  void remove_suffix(size_t n) {
    assert(size_ >= n);
    size_ -= n;
  }

  size_t find(char c, size_t pos = 0) const;
  size_t rfind(char c, size_t pos = npos) const;
  bool contains(StringPiece s) const;

  // Checks whether StringPiece starts with x and if so advances the beginning
  // of it to past the match.  It's basically a shortcut for starts_with
  // followed by remove_prefix.
  bool Consume(StringPiece x) {
    if (starts_with(x)) {
      remove_prefix(x.size_);
      return true;
    }
    return false;
  }

  StringPiece substr(size_t pos, size_t n = npos) const;

  struct Hasher {
    size_t operator()(StringPiece arg) const;
  };

  // Return a string that contains the copy of the referenced data.
  std::string ToString() const { return std::string(data_, size_); }

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  int compare(StringPiece b) const;

  // Return true iff "x" is a prefix of "*this"
  bool starts_with(StringPiece x) const {
    return ((size_ >= x.size_) && (memcmp(data_, x.data_, x.size_) == 0));
  }
  // Return true iff "x" is a suffix of "*this"
  bool ends_with(StringPiece x) const {
    return ((size_ >= x.size_) &&
            (memcmp(data_ + (size_ - x.size_), x.data_, x.size_) == 0));
  }

 private:
  const char* data_;
  size_t size_;

  // Intentionally copyable
};

inline bool operator==(StringPiece x, StringPiece y) {
  return ((x.size() == y.size()) &&
          (memcmp(x.data(), y.data(), x.size()) == 0));
}

inline bool operator!=(StringPiece x, StringPiece y) { return !(x == y); }

inline bool operator<(StringPiece x, StringPiece y) { return x.compare(y) < 0; }
inline bool operator>(StringPiece x, StringPiece y) { return x.compare(y) > 0; }
inline bool operator<=(StringPiece x, StringPiece y) {
  return x.compare(y) <= 0;
}
inline bool operator>=(StringPiece x, StringPiece y) {
  return x.compare(y) >= 0;
}

inline int StringPiece::compare(StringPiece b) const {
  const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
  int r = memcmp(data_, b.data_, min_len);
  if (r == 0) {
    if (size_ < b.size_)
      r = -1;
    else if (size_ > b.size_)
      r = +1;
  }
  return r;
}

// allow StringPiece to be logged
extern std::ostream& operator<<(std::ostream& o, tensorflow::StringPiece piece);

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_STRINGPIECE_H_
