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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_UTIL_H_

#include <memory>
#include <vector>

#include "absl/strings/str_format.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

// Backward compatibility for vectorcalls in Python 3.8. Remove this after
// dropping support for Python 3.8.
#if PY_VERSION_HEX < 0x03090000
#define JAX_PyObject_Vectorcall _PyObject_Vectorcall
#define JAX_TPFLAGS_HAVE_VECTORCALL _Py_TPFLAGS_HAVE_VECTORCALL
#else  // PY_VERSION_HEX < 0x30900000
#define JAX_PyObject_Vectorcall PyObject_Vectorcall
#define JAX_TPFLAGS_HAVE_VECTORCALL Py_TPFLAGS_HAVE_VECTORCALL
#endif  // PY_VERSION_HEX < 0x30900000

// Faster version of the pybind11 cast x.cast<T*>.
// pybind11's cast is fairly slow because it looks up the type information
// in a global hash table. It's not a particularly fast hash table and the
// lookup is pointless when we know the target type and can cache the lookup.
// This function does depend on a number of pybind11 internals;
// if it ever bitrots, one option is to replace it with a pybind11 cast.
// Return nullptr if the cast fails.
template <typename T>
T* fast_cast(pybind11::handle h) {
  static pybind11::detail::type_info* const type_info = []() {
    auto* type_info =
        pybind11::detail::get_type_info(typeid(T), /*throw_if_missing=*/false);
    CHECK(type_info);
    CHECK(type_info->simple_type);
    return type_info;
  }();
  PyTypeObject* srctype = Py_TYPE(h.ptr());
  auto reinterpret_cast_ok = [&]() {
    // Exact type match.
    if (srctype == type_info->type) {
      return true;
    }
    // If we have a subtype, then look for a base type that matches.
    if (PyType_IsSubtype(srctype, type_info->type)) {
      const auto& bases = pybind11::detail::all_type_info(srctype);
      for (auto* base : bases) {
        if (PyType_IsSubtype(base->type, type_info->type)) {
          return true;
        }
      }
    }
    return false;
  };
  if (!reinterpret_cast_ok()) {
    // Fall back to pybind11's usual cast.
    return h.cast<T*>();
  }
  auto* instance = reinterpret_cast<pybind11::detail::instance*>(h.ptr());
  if (instance->simple_layout) {
    return reinterpret_cast<T*>(instance->simple_value_holder[0]);
  } else {
    return reinterpret_cast<T*>(
        pybind11::detail::values_and_holders(instance).begin()->value_ptr());
  }
}

// Issues a Python deprecation warning. Throws a C++ exception if issuing the
// Python warning causes a Python exception to be raised.
template <typename... Args>
void PythonDeprecationWarning(const absl::FormatSpec<Args...>& format,
                              const Args&... args) {
  if (PyErr_WarnEx(PyExc_DeprecationWarning,
                   absl::StrFormat(format, args...).c_str(), 1) < 0) {
    throw pybind11::error_already_set();
  }
}

// Requests if given buffers are ready, awaits for results and returns OK if
// all of the buffers are ready or the last non-ok status.
Status AwaitBuffersReady(ifrt::Array* ifrt_array);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_UTIL_H_
