/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PYBIND11_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PYBIND11_UTILS_H_

#include <Python.h>

#include <optional>
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace jax {

// This file contains utilities to write Python wrapers using the C API.
// It's used for performance critical code such as PyBuffer, jax.jit or
// jax.pmap.

// We do not use pybind11::class_ to build Python wrapper objects because
// creation, destruction, and casting of buffer objects is performance
// critical. By using hand-written Python classes, we can avoid extra C heap
// allocations, and we can avoid pybind11's slow cast<>() implementation
// during jit dispatch.

// We need to use heap-allocated type objects because we want to add
// additional methods dynamically.

template <typename BasePyObject>
xla::StatusOr<PyObject*> CreateHeapPythonBaseClass(std::string class_name) {
  pybind11::str name = pybind11::str(class_name);
  pybind11::str qualname = pybind11::str(class_name);
  PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  // Caution: we must not call any functions that might invoke the GC until
  // PyType_Ready() is called. Otherwise the GC might see a half-constructed
  // type object.
  if (!heap_type) {
    return xla::Internal("Unable to create heap type object");
  }
  heap_type->ht_name = name.release().ptr();
  heap_type->ht_qualname = qualname.release().ptr();
  PyTypeObject* type = &heap_type->ht_type;
  type->tp_name = class_name.c_str();
  type->tp_basicsize = sizeof(BasePyObject);
  type->tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE;
  TF_RET_CHECK(PyType_Ready(type) == 0);
  return reinterpret_cast<PyObject*>(type);
}

template <typename BufferLikePyObject>

xla::StatusOr<PyObject*> CreateDeviceArrayLike(
    std::string class_name, pybind11::object base_type, newfunc tp_new,
    destructor tp_dealloc,
    /* Functions to access object as input/output buffer. If defined from
     * Python, set it to nullptr. */
    PyBufferProcs* tp_as_buffer,  // new line
    PyGetSetDef* tp_getset = nullptr) {
  pybind11::tuple bases = pybind11::make_tuple(base_type);
  pybind11::str name = pybind11::str(class_name);
  pybind11::str qualname = pybind11::str(class_name);
  PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  // Caution: we must not call any functions that might invoke the GC until
  // PyType_Ready() is called below. Otherwise the GC might see a
  // half-constructed type object.
  if (!heap_type) {
    return xla::Internal("Unable to create heap type object");
  }
  heap_type->ht_name = name.release().ptr();
  heap_type->ht_qualname = qualname.release().ptr();
  PyTypeObject* type = &heap_type->ht_type;
  type->tp_name = class_name.c_str();
  type->tp_basicsize = sizeof(BufferLikePyObject);
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
  type->tp_bases = bases.release().ptr();
  type->tp_dealloc = tp_dealloc;
  type->tp_new = tp_new;
  if (tp_getset != nullptr) {
    type->tp_getset = tp_getset;
  }
  // Supported protocols
  type->tp_as_number = &heap_type->as_number;
  type->tp_as_sequence = &heap_type->as_sequence;
  type->tp_as_mapping = &heap_type->as_mapping;
  if (tp_as_buffer != nullptr) {
    type->tp_as_buffer = tp_as_buffer;
  }

  // Allow weak references to DeviceArray objects.
  type->tp_weaklistoffset = offsetof(BufferLikePyObject, weakrefs);

  TF_RET_CHECK(PyType_Ready(type) == 0);
  return reinterpret_cast<PyObject*>(type);
}

// Helpers for building Python properties
template <typename Func>
pybind11::object property_readonly(Func&& get) {
  pybind11::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property(pybind11::cpp_function(std::forward<Func>(get)),
                  pybind11::none(), pybind11::none(), "");
}

template <typename GetFunc, typename SetFunc>
pybind11::object property(GetFunc&& get, SetFunc&& set) {
  pybind11::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property(pybind11::cpp_function(std::forward<GetFunc>(get)),
                  pybind11::cpp_function(std::forward<SetFunc>(set)),
                  pybind11::none(), "");
}

template <typename Constructor>
pybind11::object def_static(Constructor&& constructor) {
  pybind11::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return pybind11::staticmethod(
      pybind11::cpp_function(std::forward<Constructor>(constructor)));
}

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PYBIND11_UTILS_H_
