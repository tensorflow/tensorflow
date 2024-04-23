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

#include "xla/python/nb_helpers.h"

#include <Python.h>

#include "third_party/nanobind/include/nanobind/nanobind.h"

namespace nb = nanobind;

namespace xla {

Py_hash_t nb_hash(nb::handle o) {
  Py_hash_t h = PyObject_Hash(o.ptr());
  if (h == -1) {
    throw nb::python_error();
  }
  return h;
}

bool nb_isinstance(nanobind::handle inst, nanobind::handle cls) {
  int ret = PyObject_IsInstance(inst.ptr(), cls.ptr());
  if (ret == -1) {
    throw nb::python_error();
  }
  return ret;
}
}  // namespace xla
