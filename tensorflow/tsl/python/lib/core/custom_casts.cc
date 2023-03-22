/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
// Must be included first
// clang-format off
#include "tensorflow/tsl/python/lib/core/numpy.h" // NOLINT
// clang-format on

#include <locale>  // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/python/lib/core/bfloat16.h"
#include "tensorflow/tsl/python/lib/core/custom_casts.h"
#include "tensorflow/tsl/python/lib/core/float8.h"

namespace tsl {

namespace {

template <typename T>
int GetNumpyType();

template <>
int GetNumpyType<float8_e4m3b11>() {
  return Float8_E4M3B11NumpyType();
}

template <>
int GetNumpyType<bfloat16>() {
  return Bfloat16NumpyType();
}

template <>
int GetNumpyType<float8_e4m3fn>() {
  return Float8e4m3fnNumpyType();
}

template <>
int GetNumpyType<float8_e5m2>() {
  return Float8e5m2NumpyType();
}

// Performs a NumPy array cast from type 'From' to 'To' via float.
template <typename From, typename To>
void FloatPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                 void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(static_cast<float>(from[i]));
  }
}

template <typename Type1, typename Type2>
bool RegisterTwoWayCustomCast() {
  int nptype1 = GetNumpyType<Type1>();
  int nptype2 = GetNumpyType<Type2>();
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, FloatPyCast<Type1, Type2>) <
      0) {
    return false;
  }
  PyArray_Descr* descr2 = PyArray_DescrFromType(nptype2);
  if (PyArray_RegisterCastFunc(descr2, nptype1, FloatPyCast<Type2, Type1>) <
      0) {
    return false;
  }
  return true;
}

}  // namespace

bool RegisterCustomCasts() {
  bool success = true;
  // Continue trying to register casts, just in case some types are not
  // registered (i.e. float8_e4m3b11)
  success &= RegisterTwoWayCustomCast<float8_e4m3b11, float8_e4m3fn>();
  success &= RegisterTwoWayCustomCast<float8_e4m3b11, float8_e5m2>();
  success &= RegisterTwoWayCustomCast<bfloat16, float8_e4m3fn>();
  success &= RegisterTwoWayCustomCast<bfloat16, float8_e5m2>();
  return success;
}

}  // namespace tsl
