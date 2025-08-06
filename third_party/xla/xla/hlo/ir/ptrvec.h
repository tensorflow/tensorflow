/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_HLO_IR_PTRVEC_H_
#define XLA_HLO_IR_PTRVEC_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

// PtrVec<T*> is like a std::vector<T*> or absl::InlinedVector<T*>, but
// optimized to use less memory for empty and single element vectors.
//
// T must be a pointer type (e.g., char*, const int*, double*, etc.).
template <typename T>
class PtrVec {
 public:
  static_assert(std::is_pointer<T>::value);

  // Default constructible.
  PtrVec();
  ~PtrVec();

  // Copyable.
  PtrVec(const PtrVec& x);
  PtrVec& operator=(const PtrVec& x);

  // Movable.
  PtrVec(PtrVec&& x);
  PtrVec& operator=(PtrVec&& x);

  // Construct from list of pointers.
  PtrVec(std::initializer_list<T> list);

  // Construct from [first,last)
  template <typename InputIter>
  PtrVec(InputIter first, InputIter last);

  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_reference = T const&;

  using iterator = T*;
  using const_iterator = T const*;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;

  // Subset of vector-like operations.
  size_t size() const;
  bool empty() const;
  size_t capacity() const;
  T* data();
  T const* data() const;
  T& operator[](size_t i);
  T operator[](size_t i) const;
  T at(size_t i) const;
  T front() const;
  T back() const;
  void resize(size_t new_len);
  void resize(size_t new_len, T value);
  void reserve(size_t new_capacity);
  void clear();
  void pop_back();
  void push_back(T x);
  void erase(const_iterator iter);

  template <typename InputIter>
  void assign(InputIter first, InputIter last);

  // For compatibility with existing code, allow conversion to vector.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::vector<T>() const;

 private:
  // rep_ is either a T, or its bottom two bits are interpreted as a tag:
  //    kEmptyTag       empty
  //    kBigTag         remaining bits are a Big*
  //
  // kEmptyTag and kBigTag have bottom bit 1. If we attempt to store a single
  // pointer whose bottom bit is 1, we immediately switch to the big
  // representation to avoid ambiguity.
  // Empty vectors are represented uniquely in the small representation.
  static constexpr uintptr_t kEmptyTag = 0x1;
  static constexpr uintptr_t kBigTag = 0x3;
  static constexpr uintptr_t kTagMask = 0x3;

  struct Big {
    size_t size;
    size_t capacity;
    T data[];  // Beginning of variable sized portion
  };

  inline static bool can_inline(T ptr) {
    // T must have enough alignment to allow us to steal its bottom bit.
    return alignof(decltype(*ptr)) >= 2;
  }

  inline bool is_big() const { return (rep_ & kTagMask) == kBigTag; }

  inline Big* big() const {
    DCHECK(is_big());
    return reinterpret_cast<Big*>(rep_ & ~kTagMask);
  }

  // big_size returns the number of bytes to allocate for a Big representation
  // that can store up to the specified number of elements.
  inline static size_t big_size(size_t n) {
    // Verify that we won't overflow.
    static constexpr size_t kMaxFit =
        (std::numeric_limits<size_t>::max() - sizeof(Big)) / sizeof(T);
    DCHECK_LE(n, kMaxFit);
    const size_t result = sizeof(Big) + n * sizeof(T);
    DCHECK_GE(result, sizeof(Big));
    return result;
  }

  // MakeEmptyBig switches to an empty Big representation with at least the
  // specified capacity. Caller is responsible for freeing any old Big
  // representation.
  inline Big* MakeEmptyBig(size_t capacity) {
    Big* big = static_cast<Big*>(malloc(big_size(capacity)));
    big->size = 0;
    big->capacity = capacity;
    rep_ = reinterpret_cast<uintptr_t>(big) | kBigTag;
    return big;
  }

  inline static void FreeBig(Big* big) { free(big); }

  // Create new big representation with the specified capacity and filled
  // with the current elements.
  void MakeBigFromCurrentState(size_t needed_capacity);

  uintptr_t rep_;
};

// Implementation details:

template <class T>
inline PtrVec<T>::PtrVec() : rep_(kEmptyTag) {}

template <class T>
inline PtrVec<T>::~PtrVec() {
  if (is_big()) FreeBig(big());
}

template <class T>
inline PtrVec<T>::PtrVec(const PtrVec& x) : rep_(kEmptyTag) {
  *this = x;
}

template <class T>
inline PtrVec<T>& PtrVec<T>::operator=(const PtrVec& x) {
  if (this == &x) {
    return *this;
  }

  const size_t n = x.size();
  Big* b;
  if (!is_big()) {
    // Stick with small representation if we can.
    if (n < 2) {
      if (n == 0) {
        rep_ = kEmptyTag;
        return *this;
      }
      T single = x.front();
      if (can_inline(single)) {
        rep_ = reinterpret_cast<uintptr_t>(single);
        DCHECK(!empty());
        DCHECK(!is_big());
        return *this;
      }
    }

    // Switch to big representation.
    b = MakeEmptyBig(x.size());
  } else {
    if (n == 0) {
      // Make empty() faster by always using a unique representation for empty
      // vectors (tag is empty).
      clear();
      return *this;
    }
    b = big();
    if (b->capacity < n) {
      FreeBig(b);
      b = MakeEmptyBig(n);
    }
  }

  memcpy(b->data, x.data(), n * sizeof(T));
  b->size = n;
  return *this;
}

template <class T>
inline PtrVec<T>::PtrVec(PtrVec&& x) : rep_(x.rep_) {
  x.rep_ = kEmptyTag;
}

template <class T>
inline PtrVec<T>& PtrVec<T>::operator=(PtrVec&& x) {
  if (this != &x) {
    if (is_big()) {
      FreeBig(big());
    }
    rep_ = x.rep_;
    x.rep_ = kEmptyTag;
  }
  return *this;
}

template <class T>
template <class InputIter>
inline PtrVec<T>::PtrVec(InputIter first, InputIter last) : rep_(kEmptyTag) {
  assign(first, last);
}

template <class T>
inline PtrVec<T>::PtrVec(std::initializer_list<T> list)
    : PtrVec(list.begin(), list.end()) {}

template <class T>
inline size_t PtrVec<T>::size() const {
  return is_big() ? big()->size : (rep_ != kEmptyTag ? 1 : 0);
}

template <class T>
inline bool PtrVec<T>::empty() const {
  return rep_ == kEmptyTag;
}

template <class T>
inline size_t PtrVec<T>::capacity() const {
  T an_element = nullptr;
  return is_big() ? big()->capacity : (can_inline(an_element) ? 1 : 0);
}

template <class T>
inline T* PtrVec<T>::data() {
  return is_big() ? big()->data : reinterpret_cast<T*>(&rep_);
}

template <class T>
inline T const* PtrVec<T>::data() const {
  return is_big() ? big()->data : reinterpret_cast<T const*>(&rep_);
}

template <class T>
inline T& PtrVec<T>::operator[](size_t i) {
  DCHECK_LT(i, size());
  return *(data() + i);
}

template <class T>
inline T PtrVec<T>::operator[](size_t i) const {
  DCHECK_LT(i, size());
  return *(data() + i);
}

template <class T>
inline T PtrVec<T>::at(size_t i) const {
  DCHECK_LT(i, size());
  return *(data() + i);
}

template <class T>
inline T PtrVec<T>::front() const {
  return (*this)[0];
}

template <class T>
inline T PtrVec<T>::back() const {
  return (*this)[size() - 1];
}

template <class T>
inline typename PtrVec<T>::iterator PtrVec<T>::begin() {
  return data();
}

template <class T>
inline typename PtrVec<T>::iterator PtrVec<T>::end() {
  return data() + size();
}

template <class T>
inline typename PtrVec<T>::const_iterator PtrVec<T>::begin() const {
  return data();
}

template <class T>
inline typename PtrVec<T>::const_iterator PtrVec<T>::end() const {
  return data() + size();
}

template <class T>
inline typename PtrVec<T>::const_iterator PtrVec<T>::cbegin() const {
  return data();
}

template <class T>
inline typename PtrVec<T>::const_iterator PtrVec<T>::cend() const {
  return data() + size();
}

template <class T>
inline typename PtrVec<T>::const_reverse_iterator PtrVec<T>::rbegin() const {
  return const_reverse_iterator(data() + size());
}

template <class T>
inline typename PtrVec<T>::const_reverse_iterator PtrVec<T>::rend() const {
  return const_reverse_iterator(data());
}

template <class T>
inline void PtrVec<T>::clear() {
  if (is_big()) {
    FreeBig(big());
  }
  rep_ = kEmptyTag;
}

template <class T>
inline void PtrVec<T>::pop_back() {
  DCHECK(!empty());
  if (is_big()) {
    big()->size--;
    if (big()->size == 0) {
      // Revert to unique representation of empty vectors.
      clear();
    }
  } else {
    rep_ = kEmptyTag;  // From length 1 to length 0
  }
}

template <class T>
inline void PtrVec<T>::push_back(T x) {
  if (!is_big()) {
    if (rep_ == kEmptyTag) {
      if (can_inline(x)) {
        // Switch from empty to singleton representation.
        rep_ = reinterpret_cast<uintptr_t>(x);
        DCHECK(!empty());
        DCHECK(!is_big());
      } else {
        // Avoid ambiguity by jumping from empty to big representation.
        Big* b = MakeEmptyBig(1);
        b->size = 1;
        b->data[0] = x;
      }
    } else {
      // Switch from singleton to Big representation.
      T singleton = front();
      Big* b = MakeEmptyBig(2);
      b->size = 2;
      b->data[0] = singleton;
      b->data[1] = x;
    }
  } else {
    // See if x fits in current Big.
    Big* b = big();
    const size_t n = b->size;
    DCHECK_LE(n, b->capacity);
    if (n == b->capacity) {
      Big* old = b;
      b = MakeEmptyBig(std::max<size_t>(2, 2 * old->capacity));
      memcpy(b->data, old->data, n * sizeof(T));
      FreeBig(old);
    }
    b->data[n] = x;
    b->size = n + 1;
  }
}

template <class T>
inline void PtrVec<T>::resize(size_t new_len) {
  resize(new_len, nullptr);
}

template <class T>
inline void PtrVec<T>::resize(size_t new_len, T value) {
  const size_t old_size = size();
  if (new_len <= old_size) {
    // Shrink or stay at same size.
    if (new_len == 0) {
      clear();  // Switches to unique empty representation
    } else if (is_big()) {
      big()->size = new_len;
    } else {
      // size() must be <= 1 since !is_big()
      // and we know that new_len != 0 && new_len <= size.
      // So both new_new and size must be 1.
      DCHECK_EQ(new_len, 1);
      DCHECK_EQ(size(), 1);
      // Nothing to do.
    }
  } else if (!is_big() && new_len == 1) {
    // From empty to singleton
    DCHECK(empty());
    push_back(value);
  } else {
    // Need to grow to big representation.
    MakeBigFromCurrentState(new_len);
    DCHECK(is_big());
    Big* b = big();
    b->size = new_len;
    for (size_t i = old_size; i < new_len; i++) {
      b->data[i] = value;
    }
  }
}

template <class T>
inline void PtrVec<T>::reserve(size_t new_capacity) {
  if (new_capacity <= capacity()) {
    // Already have enough space.
  } else {
    MakeBigFromCurrentState(new_capacity);
  }
}

template <class T>
inline void PtrVec<T>::MakeBigFromCurrentState(size_t needed_capacity) {
  if (rep_ == kEmptyTag) {
    Big* new_b = MakeEmptyBig(needed_capacity);
    new_b->size = 0;
  } else if (!is_big()) {
    T f = front();
    Big* new_b = MakeEmptyBig(needed_capacity);
    new_b->data[0] = f;  // Move singleton (or nullptr if empty)
    new_b->size = 1;
  } else {
    Big* existing = big();
    Big* new_b = MakeEmptyBig(needed_capacity);
    new_b->size = existing->size;
    memcpy(new_b->data, existing->data, existing->size * sizeof(T));
    FreeBig(existing);
  }
}

template <class T>
inline void PtrVec<T>::erase(const_iterator iter) {
  DCHECK_GE(iter, begin());
  DCHECK_LT(iter, end());
  if (!is_big()) {
    // Must be going from single element to zero.
    rep_ = kEmptyTag;
  } else {
    Big* b = big();
    const size_t index = iter - b->data;
    memmove(b->data + index, b->data + index + 1,
            (b->size - index - 1) * sizeof(T));
    b->size--;
    if (b->size == 0) {
      // Revert to unique representation for empty vectors.
      clear();
    }
  }
}

template <class T>
template <typename InputIter>
void PtrVec<T>::assign(InputIter first, InputIter last) {
  // Could add fast path for pointer based iterators
  // if (std::is_same_v<T*, InputIter> || std::is_same_v<const T*, InputIter>)
  if (!empty()) {
    // Instead of clear() to discard old contents, we overwrite them so that
    // things work even if the input iterators point into *this.
    const size_t init_size = size();
    T* dst = data();
    size_t i = 0;
    for (; i < init_size && first != last; ++i) {
      dst[i] = *first;
      ++first;
    }
    // Discard any remaining data in *this in case input is shorter.
    if (i < init_size) {
      resize(i);
      return;
    }
  }

  // Copy rest of input.
  for (; first != last; ++first) {
    push_back(*first);
  }
}

template <class T>
inline PtrVec<T>::operator std::vector<T>() const {
  if (empty()) return {};
  return std::vector<T>(begin(), end());
}

template <typename T>
bool operator==(const PtrVec<T>& a, const PtrVec<T>& b) {
  auto a_data = a.data();
  auto b_data = b.data();
  return std::equal(a_data, a_data + a.size(), b_data, b_data + b.size());
}

template <typename T>
bool operator!=(const PtrVec<T>& a, const PtrVec<T>& b) {
  return !(a == b);
}

}  // namespace xla

#endif  // XLA_HLO_IR_PTRVEC_H_
