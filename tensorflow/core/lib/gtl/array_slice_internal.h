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

// NOT FOR INCLUSION BY CLIENT CODE. This file is only to be included by
// array_slice.h.

// Helper functions and templates for ArraySlice.

#ifndef TENSORFLOW_LIB_GTL_ARRAY_SLICE_INTERNAL_H_
#define TENSORFLOW_LIB_GTL_ARRAY_SLICE_INTERNAL_H_

#include <stddef.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace gtl {
namespace array_slice_internal {

// Template logic for generic constructors.

// Wrappers whose Get() delegates to the appropriate method of a container, and
// is defined when this method exists. Delegates to the const method if C is a
// const type.
struct Data {
  template <typename C>
  static decltype(std::declval<C>().data()) Get(C* v) {
    return v->data();
  }
};

struct MutableData {
  template <typename C>
  static decltype(std::declval<C>().mutable_data()) Get(C* v) {
    return v->mutable_data();
  }
};

struct Size {
  template <typename C>
  static decltype(std::declval<C>().size()) Get(C* v) {
    return v->size();
  }
};

struct MutableStringData {
  // Defined only for string.
  static char* Get(string* v) { return v->empty() ? nullptr : &*v->begin(); }
};

// Checks whether M::Get(C*) is defined and has a return type R such that
// Checker::valid<R>()==true.
template <typename M, typename Checker, typename C>
struct HasGetHelper : public M {
 private:
  struct None {};
  // M::Get is selected when it is viable. Get(...) is selected otherwise.
  using M::Get;
  static None Get(...);

 public:
  static constexpr bool HasGet() {
    using Result = decltype(Get(std::declval<C*>()));
    return !std::is_same<Result, None>() && Checker::template valid<Result>();
  }
};

// Defines HasGet() for a particular method, container, and checker. If
// HasGet()==true, provides Get() that delegates to the method.
template <typename M, typename Checker, typename C,
          bool /*has_get*/ = HasGetHelper<M, Checker, C>::HasGet()>
struct Wrapper {
  static constexpr bool HasGet() { return false; }
};

template <typename M, typename Checker, typename C>
struct Wrapper<M, Checker, C, true> {
  static constexpr bool HasGet() { return true; }
  static decltype(M::Get(std::declval<C*>())) Get(C* v) { return M::Get(v); }
};

// Type checker for a method returning an integral value.
struct SizeChecker {
  template <typename R>
  static constexpr bool valid() {
    return std::is_integral<R>::value;
  }
};

// Type checker for a method returning either a pointer to T or a less const
// version of that.
template <typename T>
struct DataChecker {
  // We want to enable conversion from std::vector<T*> to ArraySlice<const T*>
  // but
  // disable conversion from std::vector<Derived> to ArraySlice<Base>. Here we
  // use
  // the fact that U** is convertible to Q* const* if and only if Q is the same
  // type or a more cv-qualified version of U.
  template <typename R>
  static constexpr bool valid() {
    return std::is_convertible<R*, T* const*>::value;
  }
};

// Aliases to A if A::HasGet()==true, or to B otherwise.
template <typename A, typename B>
using FirstWithGet = typename std::conditional<A::HasGet(), A, B>::type;

// Wraps C::data() const, returning a pointer to const data.
template <typename T, typename C>
using ContainerData = Wrapper<Data, DataChecker<const T>, const C>;

// Wraps a method returning a pointer to mutable data. Prefers data() over
// mutable_data(), and handles strings when T==char. If data() returns a pointer
// to mutable data, it is most likely overloaded, but may also be a single
// method 'T* C::data() const' in a non-STL-compliant container.
template <typename T, typename C>
using ContainerMutableData =
    FirstWithGet<Wrapper<Data, DataChecker<T>, C>,
                 FirstWithGet<Wrapper<MutableData, DataChecker<T>, C>,
                              Wrapper<MutableStringData, DataChecker<T>, C>>>;

// Wraps C::size() const.
template <typename C>
using ContainerSize = Wrapper<Size, SizeChecker, const C>;

// Implementation class for ArraySlice and MutableArraySlice. In the case of
// ArraySlice, T will be a const type; for MutableArraySlice, T will be a
// mutable type.
template <typename T>
class ArraySliceImplBase {
 public:
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  static const size_type npos = static_cast<size_type>(-1);

  ArraySliceImplBase(pointer array, size_type length)
      : ptr_(array), length_(length) {}

  // Substring of another ArraySlice.
  // pos must be non-negative and <= x.length().
  // len must be non-negative and will be pinned to at most x.length() - pos.
  ArraySliceImplBase(const ArraySliceImplBase& x, size_type pos, size_type len)
      : ptr_(x.ptr_ + pos), length_(std::min(x.length_ - pos, len)) {}

  // Some of the const methods below return pointers and references to mutable
  // data. This is only the case in this internal class; ArraySlice and
  // MutableArraySlice provide deep-constness.

  pointer data() const { return ptr_; }
  size_type size() const { return length_; }

  void clear() {
    ptr_ = nullptr;
    length_ = 0;
  }

  reference operator[](size_type i) const { return ptr_[i]; }
  reference at(size_type i) const {
    DCHECK_LT(i, length_);
    return ptr_[i];
  }
  reference front() const {
    DCHECK_GT(length_, 0);
    return ptr_[0];
  }
  reference back() const {
    DCHECK_GT(length_, 0);
    return ptr_[length_ - 1];
  }

  void remove_prefix(size_type n) {
    DCHECK_GE(length_, n);
    ptr_ += n;
    length_ -= n;
  }
  void remove_suffix(size_type n) {
    DCHECK_GE(length_, n);
    length_ -= n;
  }

  iterator begin() const { return ptr_; }
  iterator end() const { return ptr_ + length_; }
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  bool operator==(const ArraySliceImplBase& other) const {
    if (size() != other.size()) return false;
    if (data() == other.data()) return true;
    return std::equal(data(), data() + size(), other.data());
  }
  bool operator!=(const ArraySliceImplBase& other) const {
    return !(*this == other);
  }

 private:
  pointer ptr_;
  size_type length_;
};

template <typename T>
class ArraySliceImpl : public ArraySliceImplBase<const T> {
 public:
  using ArraySliceImplBase<const T>::ArraySliceImplBase;

  // Defined iff the data and size accessors for the container C have been
  // defined.
  template <typename C>
  using EnableIfConvertibleFrom =
      typename std::enable_if<ContainerData<T, C>::HasGet() &&
                              ContainerSize<C>::HasGet()>::type;

  // Constructs from a container when EnableIfConvertibleFrom is
  // defined. std::addressof handles types with overloaded operator&.
  template <typename C>
  explicit ArraySliceImpl(const C& v)
      : ArraySliceImplBase<const T>(ContainerData<T, C>::Get(std::addressof(v)),
                                    ContainerSize<C>::Get(std::addressof(v))) {}
};

template <typename T>
class MutableArraySliceImpl : public ArraySliceImplBase<T> {
 public:
  using ArraySliceImplBase<T>::ArraySliceImplBase;

  template <typename C>
  using EnableIfConvertibleFrom =
      typename std::enable_if<ContainerMutableData<T, C>::HasGet() &&
                              ContainerSize<C>::HasGet()>::type;

  template <typename C>
  explicit MutableArraySliceImpl(C* v)
      : ArraySliceImplBase<T>(ContainerMutableData<T, C>::Get(v),
                              ContainerSize<C>::Get(v)) {}
};

}  // namespace array_slice_internal
}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_ARRAY_SLICE_INTERNAL_H_
