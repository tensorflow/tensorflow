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
// Note: This library is only used by python_api_dispatcher_test.  It is
// not meant to be used in other circumstances.

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/python/framework/python_api_dispatcher.h"

namespace py = pybind11;

namespace {

tensorflow::PythonAPIDispatcher MakePythonAPIDispatcher(
    const std::string& api_name, py::handle api_func, int num_params,
    const std::vector<int>& dispatch_params,
    const std::vector<int>& dispatch_list_params, bool right_to_left) {
  std::vector<tensorflow::PythonAPIDispatcher::ParamInfo> dispatchable_params;
  dispatchable_params.reserve(dispatch_params.size() +
                              dispatch_list_params.size());
  for (int p : dispatch_params) {
    dispatchable_params.push_back({p, false});
  }
  for (int p : dispatch_list_params) {
    dispatchable_params.push_back({p, true});
  }

  auto dispatcher = tensorflow::PythonAPIDispatcher(api_name, api_func.ptr(),
                                                    num_params, right_to_left);
  if (!dispatcher.Initialize(dispatchable_params)) {
    throw py::error_already_set();
  }
  return dispatcher;
}

py::handle Dispatch(tensorflow::PythonAPIDispatcher* self, py::args args) {
  auto result = self->Dispatch(args.ptr());
  if (result == nullptr) {
    throw py::error_already_set();
  } else if (result == Py_NotImplemented) {
    Py_INCREF(result);
    return result;
  } else {
    return result;
  }
}

}  // namespace

PYBIND11_MODULE(_pywrap_python_api_dispatcher, m) {
  py::class_<tensorflow::PythonAPIDispatcher>(m, "PythonAPIDispatcher")
      .def(py::init(&MakePythonAPIDispatcher))
      .def("Dispatch", Dispatch);
}
