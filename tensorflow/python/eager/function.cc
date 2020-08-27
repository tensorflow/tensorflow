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

#include <Python.h>

#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

namespace py = pybind11;

static const py::object Tensor =
    py::module::import("tensorflow.python.framework.ops").attr("Tensor");
static const py::object BaseResourceVariable =
    py::module::import("tensorflow.python.ops.resource_variable_ops")
        .attr("BaseResourceVariable");
static const py::module* np = new py::module(py::module::import("numpy"));
static const py::object CompositeTensor =
    py::module::import("tensorflow.python.framework.composite_tensor")
        .attr("CompositeTensor");

namespace tensorflow {

struct PyConcreteFunction {
  PyConcreteFunction() {}
  py::object BuildCallOutputs(py::object result,
                              py::object structured_outputs,
                              bool _ndarrays_list, bool _ndarray_singleton);
};

// TODO(jlchu): Migrate Python characteristics to C++; call this version from
// function.py when performance is improved
py::object PyConcreteFunction::BuildCallOutputs(
    py::object result, py::object structured_outputs, bool _ndarrays_list,
    bool _ndarray_singleton) {
  static const py::module* nest =
      new py::module(py::module::import("tensorflow.python.util.nest"));
  // TODO(jlchu): Look into lazy loading of np_arrays module
  static const py::module* np_arrays = new py::module(
      py::module::import("tensorflow.python.ops.numpy_ops.np_arrays"));

  if (structured_outputs.is_none()) {
    return result;
  }

  // Implied invariant: result == None only if structured_outputs == None
  py::list list_result = (py::list)result;

  if (!list_result.empty()) {
    if (_ndarrays_list) {
      py::list ndarr_result(list_result.size());
      for (int i = 0; i < ndarr_result.size(); ++i) {
        ndarr_result[i] = np_arrays->attr("tensor_to_ndarray")(list_result[i]);
      }
      return ndarr_result;
    } else if (_ndarray_singleton) {
      return np_arrays->attr("tensor_to_ndarray")(list_result[0]);
    }
  }

  // Replace outputs with results, skipping over any 'None' values.
  py::list outputs_list = nest->attr("flatten")(structured_outputs, true);
  int j = 0;
  for (int i = 0; i < outputs_list.size(); ++i) {
    if (!outputs_list[i].is_none()) {
      outputs_list[i] = list_result[j];
      ++j;
    }
  }
  return nest->attr("pack_sequence_as")(structured_outputs, outputs_list, true);
}

py::object AsNdarray(py::object value) {
  // TODO(tomhennigan) Support __array_interface__ too.
  return value.attr("__array__")();
}

bool IsNdarray(py::object value) {
  // TODO(tomhennigan) Support __array_interface__ too.
  PyObject* value_ptr = value.ptr();
  return PyObject_HasAttr(value_ptr,
                          PyUnicode_DecodeUTF8("__array__", 9, "strict")) && !(
      PyObject_IsInstance(value_ptr, Tensor.ptr())
      || PyObject_IsInstance(value_ptr, BaseResourceVariable.ptr())
      || PyObject_HasAttr(value_ptr, PyUnicode_DecodeUTF8(
             "_should_act_as_resource_variable", 32, "strict"))
      // For legacy reasons we do not automatically promote Numpy strings.
      || PyObject_IsInstance(value_ptr, np->attr("str_").ptr())
      // NumPy dtypes have __array__ as unbound methods.
      || PyObject_IsInstance(value_ptr, (PyObject*) &PyType_Type)
      // CompositeTensors should be flattened instead.
      || PyObject_IsInstance(value_ptr, CompositeTensor.ptr()));
}

PYBIND11_MODULE(_concrete_function, m) {
  py::class_<PyConcreteFunction>(m, "ConcreteFunction")
      .def(py::init<>())
      .def("_build_call_outputs", &PyConcreteFunction::BuildCallOutputs);
  m.def("_as_ndarray", &AsNdarray,
        "Converts value to an ndarray, assumes _is_ndarray(value).");
  m.def(
      "_is_ndarray", &IsNdarray,
      "Tests whether the given value is an ndarray (and not a TF tensor/var).");
}

}  // namespace tensorflow
