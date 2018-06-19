/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/lib/core/py_util.h"

// Place `<locale>` before <Python.h> to avoid build failure in macOS.
#include <locale>
#include <Python.h>

#undef toupper
#undef tolower
#undef isspace
#undef isupper
#undef islower
#undef isalpha
#undef isalnum

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

// py.__class__.__name__
const char* ClassName(PyObject* py) {
/* PyPy doesn't have a separate C API for old-style classes. */
#if PY_MAJOR_VERSION < 3 && !defined(PYPY_VERSION)
  if (PyClass_Check(py))
    return PyString_AS_STRING(
        CHECK_NOTNULL(reinterpret_cast<PyClassObject*>(py)->cl_name));
  if (PyInstance_Check(py))
    return PyString_AS_STRING(CHECK_NOTNULL(
        reinterpret_cast<PyInstanceObject*>(py)->in_class->cl_name));
#endif
  if (Py_TYPE(py) == &PyType_Type) {
    return reinterpret_cast<PyTypeObject*>(py)->tp_name;
  }
  return Py_TYPE(py)->tp_name;
}

}  // end namespace

// Returns a PyObject containing a string, or null
void TryAppendTraceback(PyObject* ptype, PyObject* pvalue, PyObject* ptraceback,
                        string* out) {
  // The "traceback" module is assumed to be imported already by script_ops.py.
  PyObject* tb_module = PyImport_AddModule("traceback");

  if (!tb_module) {
    return;
  }

  PyObject* format_exception =
      PyObject_GetAttrString(tb_module, "format_exception");

  if (!format_exception) {
    return;
  }

  if (!PyCallable_Check(format_exception)) {
    Py_DECREF(format_exception);
    return;
  }

  PyObject* ret_val = PyObject_CallFunctionObjArgs(format_exception, ptype,
                                                   pvalue, ptraceback, nullptr);
  Py_DECREF(format_exception);

  if (!ret_val) {
    return;
  }

  if (!PyList_Check(ret_val)) {
    Py_DECREF(ret_val);
    return;
  }

  Py_ssize_t n = PyList_GET_SIZE(ret_val);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* v = PyList_GET_ITEM(ret_val, i);
#if PY_MAJOR_VERSION < 3
    strings::StrAppend(out, PyString_AS_STRING(v), "\n");
#else
    strings::StrAppend(out, PyUnicode_AsUTF8(v), "\n");
#endif
  }

  // Iterate through ret_val.
  Py_DECREF(ret_val);
}

string PyExceptionFetch() {
  CHECK(PyErr_Occurred())
      << "Must only call PyExceptionFetch after an exception.";
  PyObject* ptype;
  PyObject* pvalue;
  PyObject* ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
  string err = ClassName(ptype);
  if (pvalue) {
    PyObject* str = PyObject_Str(pvalue);

    if (str) {
#if PY_MAJOR_VERSION < 3
      strings::StrAppend(&err, ": ", PyString_AS_STRING(str), "\n");
#else
      strings::StrAppend(&err, ": ", PyUnicode_AsUTF8(str), "\n");
#endif
      Py_DECREF(str);
    } else {
      strings::StrAppend(&err, "(unknown error message)\n");
    }

    TryAppendTraceback(ptype, pvalue, ptraceback, &err);

    Py_DECREF(pvalue);
  }
  Py_DECREF(ptype);
  Py_XDECREF(ptraceback);
  return err;
}

}  // end namespace tensorflow
