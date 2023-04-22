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

#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_DISPATCHER_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_DISPATCHER_H_

#include <Python.h>

#include <string>
#include <vector>

#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Dispatch handler for Python APIs.
//
// A separate PythonAPIDispatcher object is created for each Python API, and
// keeps track of which parameters should be checked for dispatch.
//
// When PythonAPIDispatcher::Dispatch() is called with a tuple of
// canonicalized parameters, it checks the indicated parameters' values for
// `__tf_dispatch__` methods.  If found, then this method is called with the
// following arguments: `__tf_dispatch__(api_name, api_func, canon_args)`,
// where:
//
//   * `api_name` is the fully-qualified name of the python API (e.g.,
//     `"tf.math.sum"`).
//   * `api_func` is the function that implements the APIs for `Tensor` inputs.
//   * `canon_args` is the canonicalized argument list.
//
class PythonAPIDispatcher {
 public:
  // Information about an API parameter that supports dispatch.  `index` is the
  // parameter's index in the canonicalized parameter list, and `is_list` is
  // true if the parameter expects a list of values (e.g. the `values` parameter
  // to `tf.concat`).
  struct ParamInfo {
    int index;
    bool is_list;
  };

  // Constructs a PythonAPIDispatcher.
  //
  // Args:
  //   api_name: The fully qualified name of the API handled by this dispatcher.
  //   api_func: The python function for which implements the API for `Tensor`
  //       inputs.
  //   num_params: The number of canonical parameters that the API expects.
  //   right_to_left: If true, then the normal precedence rules (in which
  //       dispatchers are tried from left-to-right) are changed to try
  //       dispatchers from right-to-left instead.  This is used for operations
  //       such as `__radd__`, where the normal parameter order is reversed.
  PythonAPIDispatcher(const std::string& api_name, PyObject* api_func,
                      int num_params, bool right_to_left = false);

  // Initiliaze this PythonAPIDispatcher with information about which parameters
  // support dispatch.  Returns true on success, or sets a python exception and
  // returns false on error.
  bool Initialize(std::vector<ParamInfo> dispatchable_params);

  // Checks if any of the dispatchable parameters have a `__tf_dispatch__`
  // method, and if so, calls them.  In particular, this method:
  //
  // 1. Constructs an ordered list of dispatchable types.
  //
  //   * Checks each argument that support dispatch to see if its value(s) have
  //     a `__tf_dispatch__` method.
  //   * Arguments are checked left-to-right unless `right_to_left` was set to
  //     True in the constructor.  *Within* a list-valued parameter, elements
  //     are always checked left-to-right (even if `right_to_left` is True).
  //   * Duplicate types are removed (only the first occurrence of each type is
  //     kept).
  //   * If any type `T_sub` is a subtype of another type `T_super`, but occurs
  //     after `T_super` in the list of dispatchable types, then it is moved to
  //     just before `T_super`.
  //
  // 2. Tries calling each of the dispatchable types' `__tf_dispatch__` methods.
  //
  //    * Dispatch methods are called with the following arguments:
  //      `__tf_dispatch__(api_name, api_func, canon_args)`
  //    * Dispatch methods are tried in the order described above.
  //    * If a dispatch method returns a value, then `Dispatch()` returns a
  //      new reference to that value.
  //    * If a dispatch method raises an exception, then `Dispatch()` returns
  //      null (i.e., propogates the exception).
  //    * If a dispatch method returns `NotImplemented`, then the dispatcher
  //      moves on to the next type.
  //
  // 3. If no dispatchers for found, or all dispatchers returned
  //    `NotImplemented', then the dispatcher returns a *borrowed* reference
  //    to `Py_NotImplemented`.
  //
  // Args:
  //   params: A `PyTuple` containing the canonicalized parameters to the API.
  //     All `POSITIONAL_OR_KEYWORD` arguments must be converted to positional
  //     arguments (`KEYWORD_ONLY` arguments are not currently supported).  Any
  //     dispatchable parameter with `is_list=True` must have been converted to
  //     `PyList`.
  //
  // Returns:
  //   * If a `__tf_dispatch__` handler successfully handled the API:
  //     Returns a *new* reference to the handler's return value.
  //   * If no handler was found, or all handlers returned NotImplemented:
  //     Returns a *borrowed* reference to `Py_NotImplemented`.
  //   * On error: Sets an exception and returns `nullptr`.
  PyObject* Dispatch(PyObject* params) const;

 private:
  Safe_PyObjectPtr api_name_;
  Safe_PyObjectPtr api_func_;
  int num_params_;
  std::vector<ParamInfo> dispatchable_params_;
  bool right_to_left_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_DISPATCHER_H_
