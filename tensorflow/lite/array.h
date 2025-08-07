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

#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

/// TfLite*Array helpers

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

#ifndef TF_LITE_STATIC_MEMORY
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
#endif  // TF_LITE_STATIC_MEMORY

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
inline IntArrayUniquePtr BuildTfLiteArray(const TfLiteIntArray& other) {
  return BuildTfLiteArray(other.size, other.data);
}

// Allocates a TFLiteArray and initializes it with the given array.
inline FloatArrayUniquePtr BuildTfLiteArray(const TfLiteFloatArray& other) {
  return BuildTfLiteArray(other.size, other.data);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ARRAY_H_
