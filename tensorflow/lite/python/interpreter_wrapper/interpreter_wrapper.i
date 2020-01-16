/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

%include "std_string.i"


%{
#define SWIG_FILE_WITH_INIT
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
%}


%typemap(in) TfLiteDelegate* {
  $1 = reinterpret_cast<TfLiteDelegate*>(PyLong_AsVoidPtr($input));
}

%typemap(out) TfLiteDelegate* {
  $result = PyLong_FromVoidPtr($1)
}

// Converts a Python list of str to a std::vector<std::string>, returns true
// if the conversion was successful.
%{
static bool PyListToStdVectorString(PyObject *list, std::vector<std::string> *strings) {
  // Make sure the list is actually a list.
  if (!PyList_Check(list)) return false;

  // Convert the Python list to a vector of strings.
  const int list_size = PyList_Size(list);
  strings->resize(list_size);
  for (int k = 0; k < list_size; k++) {
    PyObject *string_py = PyList_GetItem(list, k);
    if (PyString_Check(string_py)) {
      (*strings)[k] = PyString_AsString(string_py);
    } else if (PyUnicode_Check(string_py)) {
      // First convert the PyUnicode to a PyString.
      PyObject *utf8_string_py = PyUnicode_AsUTF8String(string_py);
      if (!utf8_string_py) return false;

      // Then convert it to a regular std::string.
      (*strings)[k] = PyString_AsString(utf8_string_py);
      Py_DECREF(utf8_string_py);
    } else {
      return false;
    }
  }
  return true;
}
%}
bool PyListToStdVectorString(PyObject *list, std::vector<std::string> *strings);

%include "tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h"

namespace tflite {
namespace interpreter_wrapper {
%extend InterpreterWrapper {

  // Version of the constructor that handles producing Python exceptions
  // that propagate strings.
  static PyObject* CreateWrapperCPPFromFile(
      const char* model_path,
      PyObject* registerers_py) {
    std::string error;
    std::vector<std::string> registerers;
    if (!PyListToStdVectorString(registerers_py, &registerers)) {
      PyErr_SetString(PyExc_ValueError, "Second argument is expected to be a list of strings.");
      return nullptr;
    }
    if(tflite::interpreter_wrapper::InterpreterWrapper* ptr =
        tflite::interpreter_wrapper::InterpreterWrapper
            ::CreateWrapperCPPFromFile(
        model_path, registerers, &error)) {
      return SWIG_NewPointerObj(
          ptr, SWIGTYPE_p_tflite__interpreter_wrapper__InterpreterWrapper, 1);
    } else {
      PyErr_SetString(PyExc_ValueError, error.c_str());
      return nullptr;
    }
  }

  // Version of the constructor that handles producing Python exceptions
  // that propagate strings.
  static PyObject* CreateWrapperCPPFromBuffer(
      PyObject* data ,
      PyObject* registerers_py) {
    std::string error;
    std::vector<std::string> registerers;
    if (!PyListToStdVectorString(registerers_py, &registerers)) {
      PyErr_SetString(PyExc_ValueError, "Second argument is expected to be a list of strings.");
      return nullptr;
    }
    if(tflite::interpreter_wrapper::InterpreterWrapper* ptr =
        tflite::interpreter_wrapper::InterpreterWrapper
            ::CreateWrapperCPPFromBuffer(
        data, registerers, &error)) {
      return SWIG_NewPointerObj(
          ptr, SWIGTYPE_p_tflite__interpreter_wrapper__InterpreterWrapper, 1);
    } else {
      PyErr_SetString(PyExc_ValueError, error.c_str());
      return nullptr;
    }
  }
}

}  // namespace interpreter_wrapper
}  // namespace tflite
