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

// This file provides utility functions for use with STL

#ifndef TENSORFLOW_LIB_GTL_STL_UTIL_H_
#define TENSORFLOW_LIB_GTL_STL_UTIL_H_

#include <stddef.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorflow {
namespace gtl {

// Returns a char* pointing to the beginning of a string's internal buffer.
// The result is a valid "null-terminated byte string", even if *str is empty.
// Up to C++14 it is not valid to *write* to the null terminator; as of C++17,
// it is valid to write zero to the null terminator (but not any other value).
inline char* string_as_array(string* str) { return &*str->begin(); }

// The following vector_as_array functions return raw pointers to the underlying
// data buffer. The return value is unspecified (but valid) if the input range
// is empty.
template <typename T, typename Allocator>
inline T* vector_as_array(std::vector<T, Allocator>* v) {
  return v->data();
}

template <typename T, typename Allocator>
inline const T* vector_as_array(const std::vector<T, Allocator>* v) {
  return v->data();
}

// Like str->resize(new_size), except any new characters added to "*str" as a
// result of resizing may be left uninitialized, rather than being filled with
// '0' bytes. Typically used when code is then going to overwrite the backing
// store of the string with known data. Uses a Google extension to ::string.
inline void STLStringResizeUninitialized(string* s, size_t new_size) {
#if __google_stl_resize_uninitialized_string
  s->resize_uninitialized(new_size);
#else
  s->resize(new_size);
#endif
}

// Calls delete (non-array version) on the SECOND item (pointer) in each pair in
// the range [begin, end).
//
// Note: If you're calling this on an entire container, you probably want to
// call STLDeleteValues(&container) instead, or use ValueDeleter.
template <typename ForwardIterator>
void STLDeleteContainerPairSecondPointers(ForwardIterator begin,
                                          ForwardIterator end) {
  while (begin != end) {
    ForwardIterator temp = begin;
    ++begin;
    delete temp->second;
  }
}

// Deletes all the elements in an STL container and clears the container. This
// function is suitable for use with a vector, set, hash_set, or any other STL
// container which defines sensible begin(), end(), and clear() methods.
//
// If container is NULL, this function is a no-op.
template <typename T>
void STLDeleteElements(T* container) {
  if (!container) return;
  auto it = container->begin();
  while (it != container->end()) {
    auto temp = it;
    ++it;
    delete *temp;
  }
  container->clear();
}

// Given an STL container consisting of (key, value) pairs, STLDeleteValues
// deletes all the "value" components and clears the container. Does nothing in
// the case it's given a NULL pointer.
template <typename T>
void STLDeleteValues(T* container) {
  if (!container) return;
  auto it = container->begin();
  while (it != container->end()) {
    auto temp = it;
    ++it;
    delete temp->second;
  }
  container->clear();
}

// Sorts and removes duplicates from a sequence container.
template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_STL_UTIL_H_
