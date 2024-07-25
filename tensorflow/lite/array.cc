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

#include "tensorflow/lite/array.h"

#include <cstdlib>
#include <cstring>

namespace tflite {
namespace array_internal {

void TfLiteArrayDeleter::operator()(TfLiteIntArray* a) {
  if (a) {
    TfLiteIntArrayFree(a);
  }
}
void TfLiteArrayDeleter::operator()(TfLiteFloatArray* a) {
  if (a) {
    TfLiteFloatArrayFree(a);
  }
}

namespace {

template <class T>
size_t TfLiteArrayGetSizeInBytes(const int size) {
  using ArrayType = typename TfLiteArrayUniquePtr<T>::element_type;
  constexpr size_t data_size = sizeof(std::declval<ArrayType>().data[0]);
  size_t computed_size = sizeof(T) + data_size * size;
#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  computed_size -= data_size;
#endif
  return computed_size;
}

#ifndef TF_LITE_STATIC_MEMORY

template <class T>
TfLiteArrayUniquePtr<T> TfLiteArrayCreate(const int size) {
  using ArrayType = typename TfLiteArrayUniquePtr<T>::element_type;
  const size_t alloc_size = TfLiteArrayGetSizeInBytes<T>(size);
  if (alloc_size <= 0) {
    return nullptr;
  }
  TfLiteArrayUniquePtr<T> ret(reinterpret_cast<ArrayType*>(malloc(alloc_size)));
  if (!ret) {
    return nullptr;
  }
  ret->size = size;
  return ret;
}

template <class T>
void TfLiteArrayFree(T* a) {
  free(a);
}

#endif  // TF_LITE_STATIC_MEMORY

}  // namespace
}  // namespace array_internal

IntArrayUniquePtr BuildTfLiteArray(const TfLiteIntArray& other) {
  return BuildTfLiteArray(other.size, other.data);
}

FloatArrayUniquePtr BuildTfLiteArray(const TfLiteFloatArray& other) {
  return BuildTfLiteArray(other.size, other.data);
}

bool TfLiteArrayEqual(const TfLiteIntArray& a, const TfLiteIntArray& b) {
  return TfLiteArrayEqual(&a, &b);
}

bool TfLiteArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b) {
  if (a == b) {
    return true;
  }
  if (a == nullptr || b == nullptr) {
    return false;
  }
  return TfLiteArrayEqual(a, b->data, b->size);
}

bool TfLiteArrayEqual(const TfLiteIntArray* a, const int* b_data,
                      const int b_size) {
  if (a == nullptr) {
    return b_size == 0;
  }
  if (a->size != b_size) {
    return false;
  }
  return !memcmp(a->data, b_data, a->size * sizeof(a->data[0]));
}

bool TfLiteArrayEqual(const TfLiteFloatArray& a, const TfLiteFloatArray& b,
                      const float tolerance) {
  return TfLiteArrayEqual(&a, &b, tolerance);
}

bool TfLiteArrayEqual(const TfLiteFloatArray* a, const TfLiteFloatArray* b,
                      const float tolerance) {
  if (a == b) {
    return true;
  }
  if (a == nullptr || b == nullptr) {
    return false;
  }
  return TfLiteArrayEqual(a, b->data, b->size, tolerance);
}

bool TfLiteArrayEqual(const TfLiteFloatArray* a, const float* b_data,
                      int b_size, const float tolerance) {
  if (a == nullptr) {
    return b_size == 0;
  }
  if (a->size != b_size) {
    return false;
  }
  for (int i = 0; i < b_size; ++i) {
    if (a->data[i] < b_data[i] - tolerance ||
        a->data[i] > b_data[i] + tolerance) {
      return false;
    }
  }
  return true;
}

}  // namespace tflite

extern "C" {

size_t TfLiteIntArrayGetSizeInBytes(int size) {
  return tflite::array_internal::TfLiteArrayGetSizeInBytes<int>(size);
}

int TfLiteFloatArrayGetSizeInBytes(int size) {
  return tflite::array_internal::TfLiteArrayGetSizeInBytes<float>(size);
}

int TfLiteIntArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b) {
  return tflite::TfLiteArrayEqual(a, b);
}

int TfLiteIntArrayEqualsArray(const TfLiteIntArray* a, int b_size,
                              const int b_data[]) {
  return tflite::TfLiteArrayEqual(a, b_data, b_size);
}

#ifndef TF_LITE_STATIC_MEMORY

TfLiteIntArray* TfLiteIntArrayCreate(int size) {
  return tflite::array_internal::TfLiteArrayCreate<int>(size).release();
}

TfLiteFloatArray* TfLiteFloatArrayCreate(int size) {
  return tflite::array_internal::TfLiteArrayCreate<float>(size).release();
}

TfLiteIntArray* TfLiteIntArrayCopy(const TfLiteIntArray* src) {
  return src ? tflite::BuildTfLiteArray(*src).release() : nullptr;
}

TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
  return src ? tflite::BuildTfLiteArray(*src).release() : nullptr;
}

void TfLiteIntArrayFree(TfLiteIntArray* a) {
  tflite::array_internal::TfLiteArrayFree(a);
}

void TfLiteFloatArrayFree(TfLiteFloatArray* a) {
  tflite::array_internal::TfLiteArrayFree(a);
}

#endif  // TF_LITE_STATIC_MEMORY
}
