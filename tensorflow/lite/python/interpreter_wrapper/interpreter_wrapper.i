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
%}


%include "tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h"

namespace tflite {
namespace interpreter_wrapper {
%extend InterpreterWrapper {

  // Version of the constructor that handles producing Python exceptions
  // that propagate strings.
  static PyObject* CreateWrapperCPPFromFile(const char* model_path) {
    std::string error;
    if(tflite::interpreter_wrapper::InterpreterWrapper* ptr =
        tflite::interpreter_wrapper::InterpreterWrapper
            ::CreateWrapperCPPFromFile(
        model_path, &error)) {
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
      PyObject* data) {
    std::string error;
    if(tflite::interpreter_wrapper::InterpreterWrapper* ptr =
        tflite::interpreter_wrapper::InterpreterWrapper
            ::CreateWrapperCPPFromBuffer(
        data, &error)) {
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
