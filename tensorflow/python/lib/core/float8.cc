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
#include "tensorflow/tsl/python/lib/core/numpy.h" //NOLINT
// clang-format on

#include "tensorflow/python/lib/core/float8.h"

#include <array>   // NOLINT
#include <cmath>   // NOLINT
#include <limits>  // NOLINT
#include <locale>  // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "absl/strings/str_cat.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/custom_float.h"

namespace tensorflow {
namespace custom_float_internal {

namespace ufuncs {

template <typename T>
struct CopySignFloat8 {
  T operator()(T a, T b) {
    constexpr uint8_t kSignMask = static_cast<uint8_t>(1)
                                  << (sizeof(T) * 8 - 1);
    return Eigen::numext::bit_cast<T>(static_cast<uint8_t>(
        (Eigen::numext::bit_cast<uint8_t>(a) & ~kSignMask) |
        (Eigen::numext::bit_cast<uint8_t>(b) & kSignMask)));
  }
};

template <>
struct CopySign<float8_e4m3fn> : CopySignFloat8<float8_e4m3fn> {};

template <>
struct CopySign<float8_e5m2> : CopySignFloat8<float8_e5m2> {};

template <typename T>
struct NextAfterFloat8 {
  T operator()(T from, T to) {
    if (Eigen::numext::isnan(from) || Eigen::numext::isnan(to)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    uint8_t from_as_int = Eigen::numext::bit_cast<uint8_t>(from);
    uint8_t to_as_int = Eigen::numext::bit_cast<uint8_t>(to);
    if (from_as_int == to_as_int) {
      return to;
    }

    constexpr uint8_t kSignMask = static_cast<uint8_t>(1)
                                  << (sizeof(T) * 8 - 1);
    uint8_t from_abs = from_as_int & ~kSignMask;
    uint8_t to_abs = to_as_int & ~kSignMask;
    if (from_abs == 0) {
      if (to_abs == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        return Eigen::numext::bit_cast<T>(
            static_cast<uint8_t>((to_as_int & kSignMask) | 1));
      }
    }
    uint8_t from_sign = from_as_int & kSignMask;
    uint8_t to_sign = to_as_int & kSignMask;
    uint8_t magnitude_adjustment = (from_abs > to_abs || from_sign != to_sign)
                                       ? static_cast<uint8_t>(-1)
                                       : static_cast<uint8_t>(1);
    uint8_t out_int = from_as_int + magnitude_adjustment;
    return Eigen::numext::bit_cast<T>(out_int);
  }
};

template <>
struct NextAfter<float8_e4m3fn> : NextAfterFloat8<float8_e4m3fn> {};

template <>
struct NextAfter<float8_e5m2> : NextAfterFloat8<float8_e5m2> {};

// Since float8_e4m3fn doesn't have `inf`, we need to modify to use `max`.
template <>
struct Spacing<float8_e4m3fn> {
  float8_e4m3fn operator()(float8_e4m3fn x) {
    CopySign<float8_e4m3fn> copysign;
    if (Eigen::numext::abs(x) == std::numeric_limits<float8_e4m3fn>::max()) {
      return copysign(std::numeric_limits<float8_e4m3fn>::quiet_NaN(), x);
    }
    float8_e4m3fn away = copysign(std::numeric_limits<float8_e4m3fn>::max(), x);
    return NextAfter<float8_e4m3fn>()(x, away) - x;
  }
};

}  // namespace ufuncs

template <>
struct TypeDescriptor<float8_e4m3fn>
    : custom_float_internal::CustomFloatTypeDescriptor<float8_e4m3fn> {
  typedef float8_e4m3fn T;
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  // We must register float8_e4m3fn with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.  Using 'V' to mirror bfloat16 vs float16.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '4';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2>
    : custom_float_internal::CustomFloatTypeDescriptor<float8_e5m2> {
  typedef float8_e5m2 T;
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  // Treating e5m2 as the natural "float" type since it is IEEE-754 compliant.
  static constexpr char kNpyDescrKind = 'f';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '5';
  static constexpr char kNpyDescrByteorder = '=';
};

}  // namespace custom_float_internal

namespace {

// Initializes the module.
bool Initialize() {
  tsl::ImportNumpy();
  import_umath1(false);

  custom_float_internal::Safe_PyObjectPtr numpy_str =
      custom_float_internal::make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  custom_float_internal::Safe_PyObjectPtr numpy =
      custom_float_internal::make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  if (!custom_float_internal::RegisterNumpyDtype<float8_e4m3fn>(numpy.get())) {
    return false;
  }
  if (!custom_float_internal::RegisterNumpyDtype<float8_e5m2>(numpy.get())) {
    return false;
  }
  return true;
}

}  // namespace

bool RegisterNumpyFloat8e4m3fn() {
  if (custom_float_internal::TypeDescriptor<float8_e4m3fn>::Dtype() !=
      NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load float8_e4m3fn module.");
    }
    PyErr_Print();
    return false;
  }
  return true;
}

PyObject* Float8e4m3fnDtype() {
  return reinterpret_cast<PyObject*>(
      custom_float_internal::TypeDescriptor<float8_e4m3fn>::type_ptr);
}

int Float8e4m3fnNumpyType() {
  return custom_float_internal::TypeDescriptor<float8_e4m3fn>::Dtype();
}

bool RegisterNumpyFloat8e5m2() {
  if (custom_float_internal::TypeDescriptor<float8_e5m2>::Dtype() !=
      NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load float8_e5m2 module.");
    }
    PyErr_Print();
    return false;
  }
  return true;
}

PyObject* Float8e5m2Dtype() {
  return reinterpret_cast<PyObject*>(
      custom_float_internal::TypeDescriptor<float8_e5m2>::type_ptr);
}

int Float8e5m2NumpyType() {
  return custom_float_internal::TypeDescriptor<float8_e5m2>::Dtype();
}

}  // namespace tensorflow
