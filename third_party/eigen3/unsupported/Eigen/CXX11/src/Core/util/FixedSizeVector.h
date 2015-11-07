// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FIXEDSIZEVECTOR_H
#define EIGEN_FIXEDSIZEVECTOR_H

namespace Eigen {

/** \class FixedSizeVector
  * \ingroup Core
  *
  * \brief The FixedSizeVector class.
  *
  * The %FixedSizeVector provides a subset of std::vector functionality.
  *
  * The goal is to provide basic std::vector operations when using
  * std::vector is not an option (e.g. on GPU or when compiling using
  * FMA/AVX, as this can cause either compilation failures or illegal
  * instruction failures).
  *
  */
template <typename T>
class FixedSizeVector {
 public:
  // Construct a new FixedSizeVector, reserve n elements.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  explicit FixedSizeVector(size_t n)
      : reserve_(n), size_(0),
        data_(static_cast<T*>(internal::aligned_malloc(n * sizeof(T)))) {
    for (size_t i = 0; i < n; ++i) { new (&data_[i]) T; }
  }

  // Construct a new FixedSizeVector, reserve and resize to n.
  // Copy the init value to all elements.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  explicit FixedSizeVector(size_t n, const T& init)
      : reserve_(n), size_(n),
        data_(static_cast<T*>(internal::aligned_malloc(n * sizeof(T)))) {
    for (size_t i = 0; i < n; ++i) { new (&data_[i]) T(init); }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  ~FixedSizeVector() {
    for (size_t i = 0; i < size_; ++i) {
      data_[i].~T();
    }
    internal::aligned_free(data_);
  }

  // Append new elements (up to reserved size).
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void push_back(const T& t) {
    eigen_assert(size_ < reserve_);
    data_[size_++] = t;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T& operator[] (size_t i) const {
    eigen_assert(i < size_);
    return data_[i];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T& operator[] (size_t i) {
    eigen_assert(i < size_);
    return data_[i];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T& back() {
    eigen_assert(size_ > 0);
    return data_[size_ - 1];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T& back() const {
    eigen_assert(size_ > 0);
    return data_[size_ - 1];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void pop_back() {
    // NOTE: This does not destroy the value at the end the way
    // std::vector's version of pop_back() does.  That happens when
    // the Vector is destroyed.
    eigen_assert(size_ > 0);
    size_--;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  size_t size() const { return size_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  bool empty() const { return size_ == 0; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T* data() { return data_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T* data() const { return data_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T* begin() { return data_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T* end() { return data_ + size_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T* begin() const { return data_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const T* end() const { return data_ + size_; }

 private:
  size_t reserve_;
  size_t size_;
  T* data_;
};

}  // namespace Eigen

#endif  // EIGEN_FIXEDSIZEVECTOR_H
