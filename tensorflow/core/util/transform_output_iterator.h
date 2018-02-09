/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_UTIL_TRANSFORM_OUTPUT_ITERATOR_H_
#define TENSORFLOW_UTIL_TRANSFORM_OUTPUT_ITERATOR_H_

#include <iostream>
#include <iterator>

namespace tensorflow {

template <typename StoreType, typename InputType, typename ConversionOp,
          typename OffsetT = ptrdiff_t>
class TransformOutputIterator {
 protected:
  // Proxy object
  struct Reference {
    StoreType* ptr;
    ConversionOp conversion_op;

    /// Constructor
    __host__ __device__ __forceinline__ Reference(StoreType* ptr,
                                                  ConversionOp conversion_op)
        : ptr(ptr), conversion_op(conversion_op) {}

    /// Assignment
    __host__ __device__ __forceinline__ InputType operator=(InputType val) {
      *ptr = conversion_op(val);
      return val;
    }
  };

 public:
  // Required iterator traits
  typedef TransformOutputIterator self_type;  ///< My own type
  typedef OffsetT difference_type;            ///< Type to express the result of
                                    ///< subtracting one iterator from another
  typedef void
      value_type;        ///< The type of the element the iterator can point to
  typedef void pointer;  ///< The type of a pointer to an element the iterator
                         ///< can point to
  typedef Reference reference;  ///< The type of a reference to an element the
                                ///< iterator can point to

  typedef std::random_access_iterator_tag
      iterator_category;  ///< The iterator category

  /*private:*/

  StoreType* ptr;
  ConversionOp conversion_op;

 public:
  /// Constructor
  template <typename QualifiedStoreType>
  __host__ __device__ __forceinline__ TransformOutputIterator(
      QualifiedStoreType* ptr,
      ConversionOp conversionOp)  ///< Native pointer to wrap
      : ptr(ptr), conversion_op(conversionOp) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    ptr++;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    ptr++;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const {
    return Reference(ptr, conversion_op);
  }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(ptr + n, conversion_op);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type& operator+=(Distance n) {
    ptr += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(ptr - n, conversion_op);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type& operator-=(Distance n) {
    ptr -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type
  operator-(self_type other) const {
    return ptr - other.ptr;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    return Reference(ptr + n, conversion_op);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_type& rhs) {
    return (ptr == rhs.ptr);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs) {
    return (ptr != rhs.ptr);
  }

  /// ostream operator
  friend std::ostream& operator<<(std::ostream& os, const self_type& itr) {
    return os;
  }
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_TRANSFORM_OUTPUT_ITERATOR_H_
