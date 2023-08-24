/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// The following headers need to be included in a specific order because of
// the numpy.h and Python.h headers.
// clang-format off
// NOLINTBEGIN
// Must be included first
#include "tensorflow/tsl/python/lib/core/numpy.h"

#include <locale>
#include <memory>
// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "tensorflow/tsl/python/lib/core/ml_dtypes.h"
// NOLINTEND
// clang-format on

namespace tsl {
namespace ml_dtypes {

namespace {

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

struct FloatTypes {
  PyObject* bfloat16 = nullptr;
  PyObject* float8_e4m3fn = nullptr;
  PyObject* float8_e4m3b11fnuz = nullptr;
  PyObject* float8_e5m2 = nullptr;

  int bfloat16_num = -1;
  int float8_e4m3fn_num = -1;
  int float8_e4m3b11fnuz_num = -1;
  int float8_e5m2_num = -1;

  bool initialized = false;
};

FloatTypes float_types_;  // Protected by the GIL.

bool Initialize() {
  if (float_types_.initialized) {
    return true;
  }

  auto init = []() {
    tsl::ImportNumpy();
    import_umath1(false);

    Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
    if (!numpy_str) {
      return false;
    }
    Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
    if (!numpy) {
      return false;
    }

    Safe_PyObjectPtr ml_dtypes_str =
        make_safe(PyUnicode_FromString("ml_dtypes"));
    if (!ml_dtypes_str) {
      return false;
    }
    Safe_PyObjectPtr ml_dtypes =
        make_safe(PyImport_Import(ml_dtypes_str.get()));
    if (!ml_dtypes) {
      return false;
    }
    float_types_.bfloat16 = PyObject_GetAttrString(ml_dtypes.get(), "bfloat16");
    if (!float_types_.bfloat16) {
      return false;
    }
    float_types_.float8_e4m3fn =
        PyObject_GetAttrString(ml_dtypes.get(), "float8_e4m3fn");
    if (!float_types_.float8_e4m3fn) {
      return false;
    }
    float_types_.float8_e4m3b11fnuz =
        PyObject_GetAttrString(ml_dtypes.get(), "float8_e4m3b11");
    if (!float_types_.float8_e4m3b11fnuz) {
      return false;
    }
    float_types_.float8_e5m2 =
        PyObject_GetAttrString(ml_dtypes.get(), "float8_e5m2");
    if (!float_types_.float8_e5m2) {
      return false;
    }

    float_types_.bfloat16_num = PyArray_TypeNumFromName("bfloat16");
    if (float_types_.bfloat16_num == NPY_NOTYPE) {
      return false;
    }
    float_types_.float8_e4m3fn_num = PyArray_TypeNumFromName("float8_e4m3fn");
    if (float_types_.float8_e4m3fn_num == NPY_NOTYPE) {
      return false;
    }
    float_types_.float8_e4m3b11fnuz_num =
        PyArray_TypeNumFromName("float8_e4m3b11fnuz");
    if (float_types_.float8_e4m3b11fnuz_num == NPY_NOTYPE) {
      return false;
    }
    float_types_.float8_e5m2_num = PyArray_TypeNumFromName("float8_e5m2");
    if (float_types_.float8_e5m2_num == NPY_NOTYPE) {
      return false;
    }
    float_types_.initialized = true;
    return true;
  };
  if (float_types_.initialized) {
    return true;
  }
  bool ok = init();
  if (!ok) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load ml_dtypes module.");
    }
    PyErr_Print();
  }
  return ok;
}

}  // namespace

bool RegisterTypes() { return Initialize(); }

PyObject* GetBfloat16Dtype() { return float_types_.bfloat16; }
PyObject* GetFloat8E4m3b11fnuzDtype() {
  return float_types_.float8_e4m3b11fnuz;
}
PyObject* GetFloat8E4m3fnDtype() { return float_types_.float8_e4m3fn; }
PyObject* GetFloat8E5m2Dtype() { return float_types_.float8_e5m2; }

int GetBfloat16TypeNum() { return float_types_.bfloat16_num; }
int GetFloat8E4m3b11fnuzTypeNum() {
  return float_types_.float8_e4m3b11fnuz_num;
}
int GetFloat8E4m3fnTypeNum() { return float_types_.float8_e4m3fn_num; }
int GetFloat8E5m2TypeNum() { return float_types_.float8_e5m2_num; }

}  // namespace ml_dtypes
}  // namespace tsl
