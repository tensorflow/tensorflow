/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/framework/py_context_manager.h"

#include <map>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

bool PyContextManager::Enter(PyObject* py_context_manager) {
  if (context_manager_) {
    PyErr_SetString(
        PyExc_ValueError,
        "tensorflow::PyContextManager::Enter must be called at most once.");
  }
  if (!py_context_manager) return false;
  context_manager_.reset(py_context_manager);
  static char _enter[] = "__enter__";
  var_.reset(PyObject_CallMethod(context_manager_.get(), _enter, nullptr));
  return var_ != nullptr;
}

PyContextManager::~PyContextManager() {
  if (var_) {
    static char _exit[] = "__exit__";
    static char _ooo[] = "OOO";
    if (PyErr_Occurred()) {
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);
      value = value ? value : Py_None;
      traceback = traceback ? traceback : Py_None;
      Safe_PyObjectPtr result(PyObject_CallMethod(
          context_manager_.get(), _exit, _ooo, type, value, traceback));
      if (result) {
        if (PyObject_IsTrue(result.get())) {
          PyErr_SetString(
              PyExc_ValueError,
              "tensorflow::PyContextManager::Enter does not support "
              "context managers that suppress exceptions.");
        } else {
          PyErr_Restore(type, value, traceback);
        }
      }
    } else {
      PyObject* result = PyObject_CallMethod(context_manager_.get(), _exit,
                                             _ooo, Py_None, Py_None, Py_None);
      if (result) {
        Py_DECREF(result);
      } else {
        LOG(ERROR)
            << "A context manager wrapped by tensorflow::PyContextManager "
               "raised a new exception from its __new__ method.  This behavior "
               "is not supported by PyContextManager, and the exception is "
               "being suppressed.";
        PyErr_Clear();
      }
    }
  }
}

}  // namespace tensorflow
