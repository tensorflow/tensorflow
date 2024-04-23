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

#include <cstdint>
#include <optional>
#include <string_view>

#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/tsl/python/lib/core/numpy.h"

#if NPY_ABI_VERSION < 0x02000000
#define PyDataType_ELSIZE(descr) ((descr)->elsize)
#endif

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
    return PyDataType_ELSIZE(descr);
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

  nb_numpy_ndarray(nb_dtype dtype, absl::Span<int64_t const> shape,
                   std::optional<absl::Span<int64_t const>> strides,
                   const void* ptr = nullptr,
                   nanobind::handle base = nanobind::handle());

  // Ensures that the given handle is a numpy array. If provided,
  // extra_requirements flags (NPY_ARRAY_...) are passed to PyArray_FromAny.
  // In case of an error, nullptr is returned and the Python error is cleared.
  static nb_numpy_ndarray ensure(nanobind::handle h,
                                 int extra_requirements = 0);

  // Constructs a numpy ndarray via the PyArray_From Any API. This throws an
  // error if an exception occurs.
  static nb_numpy_ndarray from_any(nanobind::handle h, int extra_requirements);

  nb_dtype dtype() const;
  npy_intp ndim() const;
  const npy_intp* shape() const;
  npy_intp shape(npy_intp dim) const;
  const npy_intp* strides() const;
  npy_intp strides(npy_intp dim) const;
  npy_intp itemsize() const;
  npy_intp size() const;
  const void* data() const;
  void* mutable_data();
  int flags() const;
};

}  // namespace xla

#endif  // XLA_PYTHON_NB_NUMPY_H_
