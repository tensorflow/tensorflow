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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PY_CONTEXT_MANAGER_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PY_CONTEXT_MANAGER_H_

#include <Python.h>

#include <string>

#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Class that wraps a Python context manager, and calls the `__enter__` and
// `__exit__` methods at appropriate times:
//
// * When `PyContextManager::Enter(cm)` is called, the context manager `cm`
//   is stored, and `cm.__enter__` is called.  The result can be retrieved
//   with `PyContextManager::var()`.
// * When the `PyContextManager` is destroyed, then `cm.__exit__` is called
//   (with information about any active exception).
// * `PyContextManager::Enter(cm)` may be called at most once. If
//   `PyContextManager::Enter()` is never called, then the destructor is a
//   no-op (i.e., `__exit__` is not called).
//
// PyContextManager places two restrictons on the wrapped context managers:
//
// 1. The context manager may not suppress exceptions -- i.e., `__exit__`
//    may not return a True value.  If it does, then a new exception will be
//    set, indicating that this is unuspported.
// 2. The context manager may not raise an exception from `__exit__` if the
//    an exception is not active when it is called.  If it does, then an error
//    message will be logged, indicating that this is unsupported, and the
//    exception will be suppressed.
//
// These restrictions are both intended to ensure that the state of
// PyErr_Occured is unchanged by PyContextManager's destructor.  This is
// important, because changing the state of PyErr_Occurred in the destructor
// would mean that we are returning a nullptr with no exception set, or
// returning a non-null value with an exception set (both of which are invalid).
class PyContextManager {
 public:
  // Calls `py_context_manager.__enter__()`, and stores the result in `var`.
  // Return true if `__enter__` succeeds, or false if `__enter__` raises an
  // exception.  (Also returns false if `py_context_manager` is nullptr.)
  //
  // Steals a reference to `py_context_manager`.  (This reference is deleted
  // when the destructor is called.)
  bool Enter(PyObject* py_context_manager);

  // Calls `py_context_manager.__exit__()`.
  ~PyContextManager();

  // Returns the variable returned by `context_manager.__enter__()`.
  // (This is the `var` bound by `with context_manager as var`.)
  // Returns a borrowed reference.
  PyObject* var() { return var_.get(); }

 protected:
  Safe_PyObjectPtr context_manager_;
  Safe_PyObjectPtr var_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PY_CONTEXT_MANAGER_H_
