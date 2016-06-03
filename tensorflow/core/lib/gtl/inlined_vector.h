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

// An InlinedVector<T,N,A> is like a std::vector<T,A>, except that storage
// for sequences of length <= N are provided inline without requiring
// any heap allocation.  Typically N is very small (e.g., 4) so that
// sequences that are expected to be short do not require allocations.
//
// Only some of the std::vector<> operations are currently implemented.
// Other operations may be added as needed to facilitate migrating
// code that uses std::vector<> to InlinedVector<>.
//
// NOTE: If you want an inlined version to replace use of a
// std::vector<bool>, consider using util::bitmap::InlinedBitVector<NBITS>
// in util/bitmap/inlined_bitvector.h
//
// TODO(billydonahue): change size_t to size_type where appropriate.

#ifndef TENSORFLOW_LIB_GTL_INLINED_VECTOR_H_
#define TENSORFLOW_LIB_GTL_INLINED_VECTOR_H_

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include <initializer_list>  // NOLINT(build/include_order)

namespace tensorflow {
namespace gtl {

template <typename T, int N>
class InlinedVector {
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ssize_t difference_type;
  typedef pointer iterator;
  typedef const_pointer const_iterator;

  // Create an empty vector
  InlinedVector();

  // Create a vector with n copies of value_type().
  explicit InlinedVector(size_t n);

  // Create a vector with n copies of elem
  InlinedVector(size_t n, const value_type& elem);

  // Create and initialize with the elements [range_start .. range_end).
  // The unused enable_if argument restricts this constructor so that it is
  // elided when value_type is an integral type.  This prevents ambiguous
  // interpretation between a call to this constructor with two integral
  // arguments and a call to the preceding (n, elem) constructor.
  template <typename InputIterator>
  InlinedVector(
      InputIterator range_start, InputIterator range_end,
      typename std::enable_if<!std::is_integral<InputIterator>::value>::type* =
          NULL) {
    InitRep();
    AppendRange(range_start, range_end);
  }

  InlinedVector(std::initializer_list<value_type> init) {
    InitRep();
    AppendRange(init.begin(), init.end());
  }

  InlinedVector(const InlinedVector& v);

  ~InlinedVector() { clear(); }

  InlinedVector& operator=(const InlinedVector& v) {
    // Optimized to avoid reallocation.
    // Prefer reassignment to copy construction for elements.
    const size_t s = size();
    const size_t vs = v.size();
    if (s < vs) {  // grow
      reserve(vs);
      if (s) std::copy(v.begin(), v.begin() + s, begin());
      std::copy(v.begin() + s, v.end(), std::back_inserter(*this));
    } else {  // maybe shrink
      erase(begin() + vs, end());
      std::copy(v.begin(), v.end(), begin());
    }
    return *this;
  }

  size_t size() const { return size_internal(); }

  bool empty() const { return (size() == 0); }

  // Return number of elements that can be stored in vector
  // without requiring a reallocation of underlying memory
  size_t capacity() const {
    if (is_inline()) {
      return kFit;
    } else {
      return static_cast<size_t>(1) << u_.data[kSize - 2];
    }
  }

  // Return a pointer to the underlying array.
  // Only result[0,size()-1] are defined.
  pointer data() {
    if (is_inline()) {
      return reinterpret_cast<T*>(u_.data);
    } else {
      return outofline_pointer();
    }
  }
  const_pointer data() const {
    return const_cast<InlinedVector<T, N>*>(this)->data();
  }

  // Remove all elements
  void clear() {
    DiscardStorage();
    u_.data[kSize - 1] = 0;
  }

  // Return the ith element
  // REQUIRES: 0 <= i < size()
  const value_type& at(size_t i) const {
    DCHECK_LT(i, size());
    return data()[i];
  }
  const value_type& operator[](size_t i) const {
    DCHECK_LT(i, size());
    return data()[i];
  }

  // Return a non-const reference to the ith element
  // REQUIRES: 0 <= i < size()
  value_type& at(size_t i) {
    DCHECK_LT(i, size());
    return data()[i];
  }
  value_type& operator[](size_t i) {
    DCHECK_LT(i, size());
    return data()[i];
  }

  value_type& back() {
    DCHECK(!empty());
    return at(size() - 1);
  }

  const value_type& back() const {
    DCHECK(!empty());
    return at(size() - 1);
  }

  value_type& front() {
    DCHECK(!empty());
    return at(0);
  }

  const value_type& front() const {
    DCHECK(!empty());
    return at(0);
  }

  // Append t to the vector.
  // Increases size() by one.
  // Amortized complexity: O(1)
  // Worst-case complexity: O(size())
  inline void push_back(const value_type& t) {
    size_t s = size();
    DCHECK_LE(s, capacity());
    if (s < capacity()) {
      new (data() + s) T(t);
      set_size_internal(s + 1);
    } else {
      PushBackSlow(t);
    }
  }

  inline void pop_back() {
    DCHECK(!empty());
    const size_t s = size();
    Destroy(data() + s - 1, 1);
    set_size_internal(s - 1);
  }

  // Resizes the vector to contain "n" elements.
  // If "n" is smaller than the initial size, extra elements are destroyed.
  // If "n" is larger than the initial size, enough copies of "elem"
  // are appended to increase the size to "n". If "elem" is omitted,
  // new elements are value-initialized.
  void resize(size_t n) { Resize<ValueInit>(n, nullptr); }
  void resize(size_t n, const value_type& elem) { Resize<Fill>(n, &elem); }

  iterator begin() { return data(); }
  const_iterator begin() const { return data(); }

  iterator end() { return data() + size(); }
  const_iterator end() const { return data() + size(); }

  iterator insert(iterator pos, const value_type& v);

  iterator erase(iterator pos) {
    DCHECK_LT(pos, end());
    DCHECK_GE(pos, begin());
    std::copy(pos + 1, end(), pos);
    pop_back();
    return pos;
  }

  iterator erase(iterator first, iterator last);

  // Enlarges the underlying representation so it can hold at least
  // "n" elements without reallocation.
  // Does not change size() or the actual contents of the vector.
  void reserve(size_t n) {
    if (n > capacity()) {
      // Make room for new elements
      Grow<Move>(n);
    }
  }

  // Swap the contents of *this with other.
  // REQUIRES: value_type is swappable and copyable.
  void swap(InlinedVector& other);

 private:
  // Representation can either be inlined or out-of-line.
  // In either case, at least sizeof(void*) + 8 bytes are available.
  //
  // Inlined:
  //   Last byte holds the length.
  //   First (length*sizeof(T)) bytes stores the elements.
  // Outlined:
  //   Last byte holds kSentinel.
  //   Second-last byte holds lg(capacity)
  //   Preceding 6 bytes hold size.
  //   First sizeof(T*) bytes hold pointer.

  // Compute rep size.
  static const size_t kSizeUnaligned = N * sizeof(T) + 1;  // Room for tag
  static const size_t kSize = ((kSizeUnaligned + 15) / 16) * 16;  // Align

  // See how many fit T we can fit inside kSize, but no more than 254
  // since 255 is used as sentinel tag for out-of-line allocation.
  static const unsigned int kSentinel = 255;
  static const size_t kFit1 = (kSize - 1) / sizeof(T);
  static const size_t kFit = (kFit1 >= kSentinel) ? (kSentinel - 1) : kFit1;

  union {
    unsigned char data[kSize];
    // Force data to be aligned enough for a pointer.
    T* unused_aligner;
  } u_;

  inline void InitRep() { u_.data[kSize - 1] = 0; }
  inline bool is_inline() const { return u_.data[kSize - 1] != kSentinel; }

  inline T* outofline_pointer() const {
    T* ptr;
    memcpy(&ptr, &u_.data[0], sizeof(ptr));
    return ptr;
  }

  inline void set_outofline_pointer(T* p) {
    memcpy(&u_.data[0], &p, sizeof(p));
  }

  inline uint64_t outofline_word() const {
    uint64_t word;
    memcpy(&word, &u_.data[kSize - 8], sizeof(word));
    return word;
  }

  inline void set_outofline_word(uint64_t w) {
    memcpy(&u_.data[kSize - 8], &w, sizeof(w));
  }

  inline size_t size_internal() const {
    uint8_t s = static_cast<uint8_t>(u_.data[kSize - 1]);
    if (s != kSentinel) {
      return static_cast<size_t>(s);
    } else {
      const uint64_t word = outofline_word();
      if (port::kLittleEndian) {
        // The sentinel and capacity bits are most-significant bits in word.
        return static_cast<size_t>(word & 0xffffffffffffull);
      } else {
        // The sentinel and capacity bits are least-significant bits in word.
        return static_cast<size_t>(word >> 16);
      }
    }
  }

  void set_size_internal(size_t n) {
    if (is_inline()) {
      DCHECK_LT(n, kSentinel);
      u_.data[kSize - 1] = static_cast<unsigned char>(n);
    } else {
      uint64_t word;
      if (port::kLittleEndian) {
        // The sentinel and capacity bits are most-significant bits in word.
        word = (static_cast<uint64_t>(n) |
                (static_cast<uint64_t>(u_.data[kSize - 2]) << 48) |
                (static_cast<uint64_t>(kSentinel) << 56));
      } else {
        // The sentinel and capacity bits are least-significant bits in word.
        word = ((static_cast<uint64_t>(n) << 16) |
                (static_cast<uint64_t>(u_.data[kSize - 2]) << 8) |
                (static_cast<uint64_t>(kSentinel)));
      }
      set_outofline_word(word);
      DCHECK_EQ(u_.data[kSize - 1], kSentinel) << n;
    }
  }

  void DiscardStorage() {
    T* base = data();
    size_t n = size();
    Destroy(base, n);
    if (!is_inline()) {
      free(base);
    }
  }

  void PushBackSlow(const T& t) {
    const size_t s = size();
    DCHECK_EQ(s, capacity());
    Grow<Move, Fill>(s + 1, &t);
    set_size_internal(s + 1);
  }

  // Does nothing.
  static void Nop(const T* src, size_t n, T* dst) {}

  // Initializes dst[0,n-1] with empty constructor.
  static void ValueInit(const T*, size_t n, T* dst) {
    for (size_t i = 0; i < n; i++) {
      new (dst + i) T();
    }
  }

  // Initializes dst[0,n-1] with copies of *src.
  static void Fill(const T* src, size_t n, T* dst) {
    for (size_t i = 0; i < n; i++) {
      new (dst + i) T(*src);
    }
  }

  // Moves srcs[0,n-1] contents to dst[0,n-1].
  static void Move(const T* src, size_t n, T* dst) {
    for (size_t i = 0; i < n; i++) {
      new (dst + i) T(std::move(*(src + i)));
    }
  }

  void Destroy(T* src, int n) {
    if (!std::is_trivially_destructible<T>::value) {
      for (int i = 0; i < n; i++) {
        (src + i)->~T();
      }
    }
  }

  // Grow so that capacity >= n.  Uses Mover to move existing elements
  // to new buffer.  If elem is not-null, stores it at slot numbered
  // size() before destroying the old buffer by calling Initializer.
  // We pass the Initializer and Mover as template arguments so that
  // this code compiles even if T does not support copying.
  template <void(Mover)(const T*, size_t, T*),
            void(Initializer)(const T*, size_t, T*) = Nop>
  void Grow(size_t n, const T* elem = nullptr) {
    size_t s = size();
    DCHECK_LE(s, capacity());

    // Compute new capacity by repeatedly doubling current capacity
    size_t target = 1;
    size_t target_lg = 0;
    while (target < kFit || target < n) {
      // TODO(psrc): Check and avoid overflow?
      target_lg++;
      target <<= 1;
    }

    T* src = data();
    T* dst = static_cast<T*>(malloc(target * sizeof(T)));

    // Need to copy elem before discarding src since it might alias src.
    if (elem) {
      Initializer(elem, 1, dst + s);
    }

    Mover(src, s, dst);
    DiscardStorage();

    u_.data[kSize - 1] = kSentinel;
    u_.data[kSize - 2] = target_lg;
    set_size_internal(s);
    DCHECK_EQ(capacity(), target);
    set_outofline_pointer(dst);
  }

  // Resize to size n.  Any new elements are initialized by passing
  // elem and the destination to Initializer.  We pass the Initializer
  // as a template argument so that this code compiles even if T does
  // not support copying.
  template <void(Initializer)(const T*, size_t, T*)>
  void Resize(size_t n, const T* elem) {
    size_t s = size();
    if (n <= s) {
      Destroy(data() + n, s - n);
      set_size_internal(n);
      return;
    }
    reserve(n);
    DCHECK_GE(capacity(), n);
    set_size_internal(n);
    Initializer(elem, n - s, data() + s);
  }

  template <typename Iter>
  void AppendRange(Iter first, Iter last, std::input_iterator_tag);

  // Faster path for forward iterators.
  template <typename Iter>
  void AppendRange(Iter first, Iter last, std::forward_iterator_tag);

  template <typename Iter>
  void AppendRange(Iter first, Iter last);
};

// Provide linkage for constants.
template <typename T, int N>
const size_t InlinedVector<T, N>::kSizeUnaligned;
template <typename T, int N>
const size_t InlinedVector<T, N>::kSize;
template <typename T, int N>
const unsigned int InlinedVector<T, N>::kSentinel;
template <typename T, int N>
const size_t InlinedVector<T, N>::kFit1;
template <typename T, int N>
const size_t InlinedVector<T, N>::kFit;

template <typename T, int N>
inline void swap(InlinedVector<T, N>& a, InlinedVector<T, N>& b) {
  a.swap(b);
}

template <typename T, int N>
inline bool operator==(const InlinedVector<T, N>& a,
                       const InlinedVector<T, N>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <typename T, int N>
inline bool operator!=(const InlinedVector<T, N>& a,
                       const InlinedVector<T, N>& b) {
  return !(a == b);
}

template <typename T, int N>
inline bool operator<(const InlinedVector<T, N>& a,
                      const InlinedVector<T, N>& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <typename T, int N>
inline bool operator>(const InlinedVector<T, N>& a,
                      const InlinedVector<T, N>& b) {
  return b < a;
}

template <typename T, int N>
inline bool operator<=(const InlinedVector<T, N>& a,
                       const InlinedVector<T, N>& b) {
  return !(b < a);
}

template <typename T, int N>
inline bool operator>=(const InlinedVector<T, N>& a,
                       const InlinedVector<T, N>& b) {
  return !(a < b);
}

// ========================================
// Implementation

template <typename T, int N>
inline InlinedVector<T, N>::InlinedVector() {
  InitRep();
}

template <typename T, int N>
inline InlinedVector<T, N>::InlinedVector(size_t n) {
  InitRep();
  if (n > capacity()) {
    Grow<Nop>(n);  // Must use Nop in case T is not copyable
  }
  set_size_internal(n);
  ValueInit(nullptr, n, data());
}

template <typename T, int N>
inline InlinedVector<T, N>::InlinedVector(size_t n, const value_type& elem) {
  InitRep();
  if (n > capacity()) {
    Grow<Nop>(n);  // Can use Nop since we know we have nothing to copy
  }
  set_size_internal(n);
  Fill(&elem, n, data());
}

template <typename T, int N>
inline InlinedVector<T, N>::InlinedVector(const InlinedVector& v) {
  InitRep();
  *this = v;
}

template <typename T, int N>
typename InlinedVector<T, N>::iterator InlinedVector<T, N>::insert(
    iterator pos, const value_type& v) {
  DCHECK_GE(pos, begin());
  DCHECK_LE(pos, end());
  if (pos == end()) {
    push_back(v);
    return end() - 1;
  }
  size_t s = size();
  size_t idx = std::distance(begin(), pos);
  if (s == capacity()) {
    Grow<Move>(s + 1);
  }
  CHECK_LT(s, capacity());
  pos = begin() + idx;  // Reset 'pos' into a post-enlarge iterator.
  Fill(data() + s - 1, 1, data() + s);  // data[s] = data[s-1]
  std::copy_backward(pos, data() + s - 1, data() + s);
  *pos = v;

  set_size_internal(s + 1);
  return pos;
}

template <typename T, int N>
typename InlinedVector<T, N>::iterator InlinedVector<T, N>::erase(
    iterator first, iterator last) {
  DCHECK_LE(begin(), first);
  DCHECK_LE(first, last);
  DCHECK_LE(last, end());

  size_t s = size();
  ptrdiff_t erase_gap = std::distance(first, last);
  std::copy(last, data() + s, first);
  Destroy(data() + s - erase_gap, erase_gap);
  set_size_internal(s - erase_gap);
  return first;
}

template <typename T, int N>
void InlinedVector<T, N>::swap(InlinedVector& other) {
  using std::swap;  // Augment ADL with std::swap.
  if (&other == this) {
    return;
  }

  InlinedVector* a = this;
  InlinedVector* b = &other;

  const bool a_inline = a->is_inline();
  const bool b_inline = b->is_inline();

  if (!a_inline && !b_inline) {
    // Just swap the top-level representations.
    T* aptr = a->outofline_pointer();
    T* bptr = b->outofline_pointer();
    a->set_outofline_pointer(bptr);
    b->set_outofline_pointer(aptr);

    uint64_t aword = a->outofline_word();
    uint64_t bword = b->outofline_word();
    a->set_outofline_word(bword);
    b->set_outofline_word(aword);
    return;
  }

  // Make a the larger of the two to reduce number of cases.
  size_t a_size = a->size();
  size_t b_size = b->size();
  if (a->size() < b->size()) {
    swap(a, b);
    swap(a_size, b_size);
  }
  DCHECK_GE(a_size, b_size);

  if (b->capacity() < a_size) {
    b->Grow<Move>(a_size);
  }

  // One is inline and one is not.
  // 'a' is larger. Swap the elements up to the smaller array size.
  std::swap_ranges(a->data(), a->data() + b_size, b->data());
  std::uninitialized_copy(a->data() + b_size, a->data() + a_size,
                          b->data() + b_size);
  Destroy(a->data() + b_size, a_size - b_size);
  a->set_size_internal(b_size);
  b->set_size_internal(a_size);
  DCHECK_EQ(b->size(), a_size);
  DCHECK_EQ(a->size(), b_size);
}

template <typename T, int N>
template <typename Iter>
inline void InlinedVector<T, N>::AppendRange(Iter first, Iter last,
                                             std::input_iterator_tag) {
  std::copy(first, last, std::back_inserter(*this));
}

template <typename T, int N>
template <typename Iter>
inline void InlinedVector<T, N>::AppendRange(Iter first, Iter last,
                                             std::forward_iterator_tag) {
  typedef typename std::iterator_traits<Iter>::difference_type Length;
  Length length = std::distance(first, last);
  size_t s = size();
  reserve(s + length);
  std::uninitialized_copy_n(first, length, data() + s);
  set_size_internal(s + length);
}

template <typename T, int N>
template <typename Iter>
inline void InlinedVector<T, N>::AppendRange(Iter first, Iter last) {
  typedef typename std::iterator_traits<Iter>::iterator_category IterTag;
  AppendRange(first, last, IterTag());
}

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_INLINED_VECTOR_H_
