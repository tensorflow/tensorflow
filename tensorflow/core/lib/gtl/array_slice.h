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

// An ArraySlice<T> represents an immutable array of elements of type
// T.  It has a length "length", and a base pointer "ptr", and the
// array it represents contains the elements "ptr[0] .. ptr[len-1]".
// The backing store for the array is *not* owned by the ArraySlice
// object, and clients must arrange for the backing store to remain
// live while the ArraySlice object is in use.
//
// An ArraySlice<T> is somewhat analogous to a StringPiece, but for
// array elements of type T.
//
// Implicit conversion operations are provided from types such as
// std::vector<T> and util::gtl::InlinedVector<T, N>.  Note that ArraySlice
// objects constructed from types in this way may be invalidated by
// any operations that mutate the underlying vector.
//
// One common use for ArraySlice is when passing arguments to a
// routine where you want to be able to accept a variety of array
// types (e.g. a vector, a util::gtl::InlinedVector, a C-style array,
// etc.).  The usual approach here is to have the client explicitly
// pass in a pointer and a length, as in:
//
//   void MyRoutine(const int* elems, int N) {
//     for (int i = 0; i < N; i++) { .. do something with elems[i] .. }
//   }
//
// Unfortunately, this leads to ugly and error-prone code at the call site:
//
//   std::vector<int> my_vector;
//   MyRoutine(vector_as_array(&my_vector), my_vector.size());
//
//   util::gtl::InlinedVector<int, 4> my_inline_vector;
//   MyRoutine(my_inline_vector.array(), my_inline_vector.size());
//
//   int my_array[10];
//   MyRoutine(my_array, 10);
//
// Instead, you can use an ArraySlice as the argument to the routine:
//
//   void MyRoutine(ArraySlice<int> a) {
//     for (int i = 0; i < a.size(); i++) { .. do something with a[i] .. }
//   }
//
// This makes the call sites cleaner, for the most part:
//
//   std::vector<int> my_vector;
//   MyRoutine(my_vector);
//
//   util::gtl::InlinedVector<int, 4> my_inline_vector;
//   MyRoutine(my_inline_vector);
//
//   int my_array[10];
//   MyRoutine(my_array);
//
//   int* my_array = new int[10];
//   MyRoutine(gtl::ArraySlice<int>(my_array, 10));
//
// MutableArraySlice<T> represents a mutable array of elements, and, like
// ArraySlice, does not own the backing store. The implicit constructors it
// provides allow functions not to worry about whether their mutable arguments
// refer to vectors, arrays, proto2::RepeatedFields, etc.:
//
//   void MyMutatingRoutine(MutableArraySlice<int> a) {
//     for (int i = 0; i < a.size(); i++) { .. mutate a[i] .. }
//   }
//
//   std::vector<int> my_vector;
//   MyMutatingRoutine(&my_vector);
//
//   int my_array[10];
//   MyMutatingRoutine(my_array);
//
//   int* my_array = new int[10];
//   MyMutatingRoutine(gtl::MutableArraySlice<int>(my_array, 10));
//
//   MyProto my_proto;
//   for (int i = 0; i < 10; ++i) { my_proto.add_value(i); }
//   MyMutatingRoutine(my_proto.mutable_value());

#ifndef TENSORFLOW_LIB_GTL_ARRAY_SLICE_H_
#define TENSORFLOW_LIB_GTL_ARRAY_SLICE_H_

#include <initializer_list>
#include <type_traits>
#include <vector>

#include "tensorflow/core/lib/gtl/array_slice_internal.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace gtl {

template <typename T>
class ArraySlice {
 private:
  typedef array_slice_internal::ArraySliceImpl<T> Impl;

 public:
  typedef T value_type;
  typedef typename Impl::pointer pointer;
  typedef typename Impl::const_pointer const_pointer;
  typedef typename Impl::reference reference;
  typedef typename Impl::const_reference const_reference;
  typedef typename Impl::iterator iterator;
  typedef typename Impl::const_iterator const_iterator;
  typedef typename Impl::reverse_iterator reverse_iterator;
  typedef typename Impl::const_reverse_iterator const_reverse_iterator;
  typedef typename Impl::size_type size_type;
  typedef typename Impl::difference_type difference_type;

  static const size_type npos = Impl::npos;

  ArraySlice() : impl_(nullptr, 0) {}
  ArraySlice(const_pointer array, size_type length) : impl_(array, length) {}

  // Implicit conversion constructors
  ArraySlice(const std::vector<value_type>& v)  // NOLINT(runtime/explicit)
      : impl_(v.data(), v.size()) {}

  template <size_t N>
  ArraySlice(const value_type (&a)[N])  // NOLINT(runtime/explicit)
      : impl_(a, N) {}

  template <int N>
  ArraySlice(const InlinedVector<value_type, N>& v)  // NOLINT(runtime/explicit)
      : impl_(v.data(), v.size()) {}

  // The constructor for any class supplying 'data() const' that returns either
  // const T* or a less const-qualified version of it, and 'some_integral_type
  // size() const'. proto2::RepeatedField<T>, string and (since C++11)
  // std::vector<T,A> and std::array<T, N> are examples of this. See
  // array_slice_internal.h for details.
  template <typename V,
            typename = typename Impl::template EnableIfConvertibleFrom<V>>
  ArraySlice(const V& v)  // NOLINT(runtime/explicit)
      : impl_(v) {}

  // Implicitly constructs an ArraySlice from an initializer list. This makes it
  // possible to pass a brace-enclosed initializer list to a function expecting
  // an ArraySlice:
  //   void Process(ArraySlice<int> x);
  //   Process({1, 2, 3});
  // The data referenced by the initializer_list must outlive this
  // ArraySlice. For example, "ArraySlice<int> s={1,2};" and "return
  // ArraySlice<int>({3,4});" are errors, as the resulting ArraySlice may
  // reference data that is no longer valid.
  ArraySlice(std::initializer_list<value_type> v)  // NOLINT(runtime/explicit)
      : impl_(v.begin(), v.size()) {}

  // Substring of another ArraySlice.
  // pos must be non-negative and <= x.length().
  // len must be non-negative and will be pinned to at most x.length() - pos.
  // If len==npos, the substring continues till the end of x.
  ArraySlice(const ArraySlice& x, size_type pos, size_type len)
      : impl_(x.impl_, pos, len) {}

  const_pointer data() const { return impl_.data(); }
  size_type size() const { return impl_.size(); }
  size_type length() const { return size(); }
  bool empty() const { return size() == 0; }

  void clear() { impl_.clear(); }

  const_reference operator[](size_type i) const { return impl_[i]; }
  const_reference at(size_type i) const { return impl_.at(i); }
  const_reference front() const { return impl_.front(); }
  const_reference back() const { return impl_.back(); }

  const_iterator begin() const { return impl_.begin(); }
  const_iterator end() const { return impl_.end(); }
  const_reverse_iterator rbegin() const { return impl_.rbegin(); }
  const_reverse_iterator rend() const { return impl_.rend(); }

  void remove_prefix(size_type n) { impl_.remove_prefix(n); }
  void remove_suffix(size_type n) { impl_.remove_suffix(n); }
  void pop_back() { remove_suffix(1); }
  void pop_front() { remove_prefix(1); }

  // These relational operators have the same semantics as the
  // std::vector<T> relational operators: they do deep (element-wise)
  // comparisons.  Array slices are equal iff their size is the same
  // and all their elements are equal.
  bool operator==(ArraySlice<T> other) const { return impl_ == other.impl_; }
  bool operator!=(ArraySlice<T> other) const { return impl_ != other.impl_; }

 private:
  Impl impl_;
};

// Mutable version of ArraySlice, which allows the clients to mutate the
// underlying data. It is implicitly convertible to ArraySlice since it provides
// the data() and size() methods with correct signatures. When a
// MutableArraySlice is created from a pointer to a container (as opposed to raw
// memory pointer), the pointer must not be null.
//
// A note on const-ness: "mutable" here refers to the mutability of the
// underlying data, not of the slice itself. It is perfectly reasonable to have
// a variable of type "const MutableArraySlice<T>"; this means that the bounds
// of the view on the array cannot be changed, but the underlying data in the
// array still may be modified. This is akin to a "T* const" pointer, as opposed
// to a "const T*" pointer (corresponding to a non-const ArraySlice<T>).
template <typename T>
class MutableArraySlice {
 private:
  typedef array_slice_internal::MutableArraySliceImpl<T> Impl;

 public:
  typedef T value_type;
  typedef typename Impl::pointer pointer;
  typedef typename Impl::const_pointer const_pointer;
  typedef typename Impl::reference reference;
  typedef typename Impl::const_reference const_reference;
  typedef typename Impl::iterator iterator;
  typedef typename Impl::const_iterator const_iterator;
  typedef typename Impl::reverse_iterator reverse_iterator;
  typedef typename Impl::const_reverse_iterator const_reverse_iterator;
  typedef typename Impl::size_type size_type;
  typedef typename Impl::difference_type difference_type;

  static const size_type npos = Impl::npos;

  MutableArraySlice() : impl_(nullptr, 0) {}
  MutableArraySlice(pointer array, size_type length) : impl_(array, length) {}

  // Implicit conversion constructors
  MutableArraySlice(std::vector<value_type>* v)  // NOLINT(runtime/explicit)
      : impl_(v->data(), v->size()) {}

  template <size_t N>
  MutableArraySlice(value_type (&a)[N])  // NOLINT(runtime/explicit)
      : impl_(a, N) {}

  template <int N>
  MutableArraySlice(
      InlinedVector<value_type, N>* v)  // NOLINT(runtime/explicit)
      : impl_(v->data(), v->size()) {}

  // The constructor for any class supplying 'T* data()' or 'T* mutable_data()'
  // (the former is called if both exist), and 'some_integral_type size()
  // const'. proto2::RepeatedField is an example of this. Also supports string
  // arguments, when T==char. The appropriate ctor is selected using SFINAE. See
  // array_slice_internal.h for details.
  template <typename V,
            typename = typename Impl::template EnableIfConvertibleFrom<V>>
  MutableArraySlice(V* v)  // NOLINT(runtime/explicit)
      : impl_(v) {}

  // Substring of another MutableArraySlice.
  // pos must be non-negative and <= x.length().
  // len must be non-negative and will be pinned to at most x.length() - pos.
  // If len==npos, the substring continues till the end of x.
  MutableArraySlice(const MutableArraySlice& x, size_type pos, size_type len)
      : impl_(x.impl_, pos, len) {}

  // Accessors.
  pointer data() const { return impl_.data(); }
  size_type size() const { return impl_.size(); }
  size_type length() const { return size(); }
  bool empty() const { return size() == 0; }

  void clear() { impl_.clear(); }

  reference operator[](size_type i) const { return impl_[i]; }
  reference at(size_type i) const { return impl_.at(i); }
  reference front() const { return impl_.front(); }
  reference back() const { return impl_.back(); }

  iterator begin() const { return impl_.begin(); }
  iterator end() const { return impl_.end(); }
  reverse_iterator rbegin() const { return impl_.rbegin(); }
  reverse_iterator rend() const { return impl_.rend(); }

  void remove_prefix(size_type n) { impl_.remove_prefix(n); }
  void remove_suffix(size_type n) { impl_.remove_suffix(n); }
  void pop_back() { remove_suffix(1); }
  void pop_front() { remove_prefix(1); }

  bool operator==(ArraySlice<T> other) const {
    return ArraySlice<T>(*this) == other;
  }
  bool operator!=(ArraySlice<T> other) const {
    return ArraySlice<T>(*this) != other;
  }

  // DEPRECATED(jacobsa): Please use data() instead.
  pointer mutable_data() const { return impl_.data(); }

 private:
  Impl impl_;
};

template <typename T>
const typename ArraySlice<T>::size_type ArraySlice<T>::npos;
template <typename T>
const typename MutableArraySlice<T>::size_type MutableArraySlice<T>::npos;

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_ARRAY_SLICE_H_
