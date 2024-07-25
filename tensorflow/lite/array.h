/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_ARRAY_H_
#define TENSORFLOW_LITE_ARRAY_H_

#include <stdlib.h>

#ifdef __cplusplus
#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>

extern "C" {
#endif  // __cplusplus

// C interface

// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct TfLiteIntArray {
  int size;

#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  int data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  // gcc 6.1+ have a bug where flexible members aren't properly handled
  // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

// Fixed size list of floats. Used for per-channel quantization.
typedef struct TfLiteFloatArray {
  int size;
#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  float data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  // gcc 6.1+ have a bug where flexible members aren't properly handled
  // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
  float data[0];
#else
  float data[];
#endif
} TfLiteFloatArray;

// Given the size (number of elements) in a TfLiteIntArray, calculate its size
// in bytes.
size_t TfLiteIntArrayGetSizeInBytes(int size);

// Given the size (number of elements) in a TfLiteFloatArray, calculate its size
// in bytes.
int TfLiteFloatArrayGetSizeInBytes(int size);

// Check if two intarrays are equal. Returns 1 if they are equal, 0 otherwise.
int TfLiteIntArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b);

// Check if an intarray equals an array. Returns 1 if equals, 0 otherwise.
int TfLiteIntArrayEqualsArray(const TfLiteIntArray* a, int b_size,
                              const int b_data[]);

#ifndef TF_LITE_STATIC_MEMORY
// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using TfLiteIntArrayFree().
TfLiteIntArray* TfLiteIntArrayCreate(int size);

// Create a copy of an array passed as `src`.
// You are expected to free memory with TfLiteIntArrayFree
TfLiteIntArray* TfLiteIntArrayCopy(const TfLiteIntArray* src);

// Free memory of array `a`.
void TfLiteIntArrayFree(TfLiteIntArray* a);

// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using TfLiteFloatArrayFree().
TfLiteFloatArray* TfLiteFloatArrayCreate(int size);

// Create a copy of an array passed as `src`.
// You are expected to free memory with TfLiteFloatArrayFree.
TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src);

// Free memory of array `a`.
void TfLiteFloatArrayFree(TfLiteFloatArray* a);
#endif  // TF_LITE_STATIC_MEMORY

#ifdef __cplusplus
}
#endif  // __cplusplus

// C++ interface

#ifdef __cplusplus

namespace tflite {

/// TfLite array helpers

namespace array_internal {

// Function object used as a deleter for unique_ptr holding TFLite*Array
// objects.
struct TfLiteArrayDeleter {
  void operator()(TfLiteIntArray* a);
  void operator()(TfLiteFloatArray* a);
};

// Maps T to the corresponding TfLiteArray type.
template <class T>
struct TfLiteArrayInfo;

template <>
struct TfLiteArrayInfo<int> {
  using Type = TfLiteIntArray;
};

template <>
struct TfLiteArrayInfo<float> {
  using Type = TfLiteFloatArray;
};

}  // namespace array_internal

// `unique_ptr` wrapper for TfLite arrays.
template <class T>
using TfLiteArrayUniquePtr =
    std::unique_ptr<typename array_internal::TfLiteArrayInfo<T>::Type,
                    array_internal::TfLiteArrayDeleter>;

// `unique_ptr` wrapper for `TfLiteIntArray`s.
using IntArrayUniquePtr = TfLiteArrayUniquePtr<int>;

// `unique_ptr` wrapper for `TfLiteFloatArray`s.
using FloatArrayUniquePtr = TfLiteArrayUniquePtr<float>;

// Allocates a TfLiteArray of given size using malloc.
//
// This builds an int array by default as this is the overwhelming part of the
// use cases.
template <class T = int>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(int size);

// Allocates a TfLiteIntArray of given size using malloc.
template <>
inline IntArrayUniquePtr BuildTfLiteArray<int>(const int size) {
  return IntArrayUniquePtr(TfLiteIntArrayCreate(size));
}

// Allocates a TfLiteFloatArray of given size using malloc.
template <>
inline FloatArrayUniquePtr BuildTfLiteArray<float>(const int size) {
  return FloatArrayUniquePtr(TfLiteFloatArrayCreate(size));
}

// Allocates a TFLiteArray of given size and initializes it with the given
// values.
//
// `values` is expected to holds `size` elements.
//
// If T is explicitely specified and the type of values is not the same as T,
// then a static_cast is performed.
template <class T = void, class U,
          class Type = std::conditional_t<std::is_same<T, void>::value, U, T>>
TfLiteArrayUniquePtr<Type> BuildTfLiteArray(const int size,
                                            const U* const values) {
  TfLiteArrayUniquePtr<Type> array = BuildTfLiteArray<Type>(size);
  // If size is 0, the array pointer may be null.
  if (array && values) {
    if (std::is_same<Type, U>::value) {
      memcpy(array->data, values, size * sizeof(Type));
    } else {
      for (int i = 0; i < size; ++i) {
        array->data[i] = static_cast<Type>(values[i]);
      }
    }
  }
  return array;
}

// Allocates a TFLiteArray and initializes it with the given array.
//
// `values` is expected to holds `size` elements.
template <class T, size_t N>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(const T (&values)[N]) {
  return BuildTfLiteArray<T>(static_cast<int>(N), values);
}

// Allocates a TFLiteArray and initializes it with the given values.
//
// This uses SFINAE to only be picked up by for types that implement `data()`
// and `size()` member functions. We cannot reuse detection facilities provided
// by Abseil in this code.
//
// To conform with the other overloads, we allow specifying the type of the
// array as well as deducing it from the container.
template <
    class T = void, class Container,
    class ElementType =
        std::decay_t<decltype(*std::declval<Container>().data())>,
    class SizeType = std::decay_t<decltype(std::declval<Container>().size())>,
    class Type =
        std::conditional_t<std::is_same<T, void>::value, ElementType, T>>
TfLiteArrayUniquePtr<Type> BuildTfLiteArray(const Container& values) {
  return BuildTfLiteArray<Type>(static_cast<int>(values.size()), values.data());
}

// Allocates a TFLiteArray and initializes it with the given values.
template <class T>
TfLiteArrayUniquePtr<T> BuildTfLiteArray(
    const std::initializer_list<T>& values) {
  return BuildTfLiteArray(static_cast<int>(values.size()), values.begin());
}

// Allocates a TFLiteArray and initializes it with the given array.
IntArrayUniquePtr BuildTfLiteArray(const TfLiteIntArray& other);

// Allocates a TFLiteArray and initializes it with the given array.
FloatArrayUniquePtr BuildTfLiteArray(const TfLiteFloatArray& other);

// Compare two TfLite array contents.
bool TfLiteArrayEqual(const TfLiteIntArray& a, const TfLiteIntArray& b);

// Compare two TfLite array contents.
bool TfLiteArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b);

// Compare two TfLite array contents.
bool TfLiteArrayEqual(const TfLiteIntArray* a, const int* b, int b_size);

// Compare two TfLite array contents.
bool TfLiteArrayEqual(const TfLiteFloatArray& a, const TfLiteFloatArray& b,
                      float tolerance);

// Compare two TfLite array contents.
bool TfLiteArrayEqual(const TfLiteFloatArray* a, const TfLiteFloatArray* b,
                      float tolerance);

// Compare two TfLite array contents.
bool TfLiteArrayEqual(const TfLiteFloatArray* a, const float* b, int b_size,
                      float tolerance);

}  // namespace tflite

#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_ARRAY_H_
