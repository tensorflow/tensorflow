/* Copyright 2024 The OpenXLA Authors.

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

// Nanobind wrappers for NumPy types.
//
// Unlike pybind11, nanobind does not provide direct wrappers for NumPy types.
// This file provides nanobind equivalents of pybind11::dtype and
// pybind11::array.

#ifndef XLA_PYTHON_NB_NUMPY_H_
#define XLA_PYTHON_NB_NUMPY_H_

#include <Python.h>

#include <string_view>

#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "tsl/python/lib/core/numpy.h"  // NOLINT

namespace xla {

// Caution: to use this type you must call tsl::ImportNumpy() in your module
// initialization function. Otherwise PyArray_DescrCheck will be nullptr.
class nb_dtype : public nanobind::object {
 public:
  NB_OBJECT_DEFAULT(nb_dtype, object, "dtype", PyArray_DescrCheck);  // NOLINT

  explicit nb_dtype(const nanobind::str& format)
      : nb_dtype(from_args(format)) {}
  explicit nb_dtype(std::string_view format)
      : nb_dtype(from_args(nanobind::str(format.data(), format.size()))) {}

  static nb_dtype from_args(const nanobind::object& args);

  int char_() const {
    auto* descr = reinterpret_cast<PyArray_Descr*>(ptr());
    return descr->type;
  }

  int itemsize() const {
    auto* descr = reinterpret_cast<PyArray_Descr*>(ptr());
    return descr->elsize;
  }

  /// Single-character code for dtype's kind.
  /// For example, floating point types are 'f' and integral types are 'i'.
  char kind() const {
    auto* descr = reinterpret_cast<PyArray_Descr*>(ptr());
    return descr->kind;
  }
};

class nb_numpy_ndarray : public nanobind::object {
 public:
  NB_OBJECT_DEFAULT(nb_numpy_ndarray, object, "ndarray",
                    PyArray_Check);  // NOLINT

  nb_numpy_ndarray(nb_dtype dtype, absl::Span<ssize_t const> shape,
                   absl::Span<ssize_t const> strides, const void* ptr = nullptr,
                   nanobind::handle base = nanobind::handle());

  // Ensures that the given handle is a numpy array. If provided,
  // extra_requirements flags (NPY_ARRAY_...) are passed to PyArray_FromAny.
  static nb_numpy_ndarray ensure(nanobind::handle h,
                                 int extra_requirements = 0);

  nb_dtype dtype() const;
  ssize_t ndim() const;
  const ssize_t* shape() const;
  ssize_t shape(ssize_t dim) const;
  const ssize_t* strides() const;
  ssize_t itemsize() const;
  ssize_t size() const;
  const void* data() const;
  int flags() const;
};

}  // namespace xla

#endif  // XLA_PYTHON_NB_NUMPY_H_
