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

// Returns a mutable char* pointing to a string's internal buffer, which may not
// be null-terminated. Returns NULL for an empty string. If not non-null,
// writing through this pointer will modify the string.
//
// string_as_array(&str)[i] is valid for 0 <= i < str.size() until the
// next call to a string method that invalidates iterators.
//
// In C++11 you may simply use &str[0] to get a mutable char*.
//
// Prior to C++11, there was no standard-blessed way of getting a mutable
// reference to a string's internal buffer. The requirement that string be
// contiguous is officially part of the C++11 standard [string.require]/5.
// According to Matt Austern, this should already work on all current C++98
// implementations.
inline char* string_as_array(string* str) {
  return str->empty() ? NULL : &*str->begin();
}

// Returns the T* array for the given vector, or NULL if the vector was empty.
//
// Note: If you know the array will never be empty, you can use &*v.begin()
// directly, but that is may dump core if v is empty. This function is the most
// efficient code that will work, taking into account how our STL is actually
// implemented. THIS IS NON-PORTABLE CODE, so use this function instead of
// repeating the nonportable code everywhere. If our STL implementation changes,
// we will need to change this as well.
template <typename T, typename Allocator>
inline T* vector_as_array(std::vector<T, Allocator>* v) {
#if defined NDEBUG && !defined _GLIBCXX_DEBUG
  return &*v->begin();
#else
  return v->empty() ? NULL : &*v->begin();
#endif
}
// vector_as_array overload for const std::vector<>.
template <typename T, typename Allocator>
inline const T* vector_as_array(const std::vector<T, Allocator>* v) {
#if defined NDEBUG && !defined _GLIBCXX_DEBUG
  return &*v->begin();
#else
  return v->empty() ? NULL : &*v->begin();
#endif
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
