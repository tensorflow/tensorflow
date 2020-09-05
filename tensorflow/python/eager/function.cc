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

#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/util/util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace py = pybind11;

static const py::module* nest = new py::module(
    py::module::import("tensorflow.python.util.nest"));
static const py::module* np = new py::module(py::module::import("numpy"));
static const py::object* np_str = new py::object(np->attr("str_"));
static const py::object* np_ndarray = new py::object(np->attr("ndarray"));
static const py::object* create_constant_tensor = new py::object(
    py::module::import("tensorflow.python.framework.constant_op")
        .attr("constant"));

namespace tensorflow {

class PyConcreteFunction {
 public:
  PyConcreteFunction() {}
  py::object BuildCallOutputs(py::object result,
                              py::object structured_outputs,
                              bool _ndarrays_list, bool _ndarray_singleton);
};

namespace {

std::string JoinVector(const char* delimiter, std::vector<std::string>& vec) {
  auto it = vec.begin();
  std::string result = *it;
  ++it;
  while (it != vec.end()) {
    tensorflow::strings::StrAppend(&result, delimiter, *it);
    ++it;
  }
  return result;
}

std::string JoinPyDict(const char* delimiter, py::dict dict) {
  auto it = dict.begin();
  std::string result = std::string((py::str) it->first);
  ++it;
  while (it != dict.end()) {
    tensorflow::strings::StrAppend(
        &result, delimiter, std::string((py::str) it->first));
    ++it;
  }
  return result;
}

std::string PyObjectToString(PyObject* o) {
  if (o == nullptr) {
    return "";
  }
  PyObject* str = PyObject_Str(o);
  if (str) {
#if PY_MAJOR_VERSION < 3
    std::string s(PyString_AS_STRING(str));
#else
    std::string s(PyUnicode_AsUTF8(str));
#endif
    Py_DECREF(str);
    return s;
  } else {
    return "<failed to execute str() on object>";
  }
}

} // namespace

// TODO(jlchu): Migrate Python characteristics to C++; call this version from
// function.py when performance is improved
py::object PyConcreteFunction::BuildCallOutputs(
    py::object result, py::object structured_outputs, bool _ndarrays_list,
    bool _ndarray_singleton) {
  // TODO(jlchu): Look into lazy loading of np_arrays module
  static const py::module* np_arrays = new py::module(
      py::module::import("tensorflow.python.ops.numpy_ops.np_arrays"));

  if (structured_outputs.is_none()) {
    return result;
  }

  // Implied invariant: result == None only if structured_outputs == None
  py::list list_result = static_cast<py::list>(result);

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
  py::list outputs_list = PyoOrThrow(
      swig::Flatten(structured_outputs.ptr(), true));
  int j = 0;
  for (int i = 0; i < outputs_list.size(); ++i) {
    if (!outputs_list[i].is_none()) {
      outputs_list[i] = list_result[j];
      ++j;
    }
  }
  return nest->attr("pack_sequence_as")(structured_outputs, outputs_list, true);
}

py::object AsNdarray(py::handle value) {
  // TODO(tomhennigan) Support __array_interface__ too.
  return value.attr("__array__")();
}

bool IsNdarray(py::handle value) {
  // TODO(tomhennigan) Support __array_interface__ too.
  PyObject* value_ptr = value.ptr();
  return py::hasattr(value, "__array__") && !(
      swig::IsTensor(value_ptr)
      || swig::IsBaseResourceVariable(value_ptr)
      || py::hasattr(value, "_should_act_as_resource_variable")
      // For legacy reasons we do not automatically promote Numpy strings.
      || PyObject_IsInstance(value_ptr, np_str->ptr())
      // NumPy dtypes have __array__ as unbound methods.
      || PyType_Check(value_ptr)
      // CompositeTensors should be flattened instead.
      || swig::IsCompositeTensor(value_ptr));
}

py::tuple ConvertNumpyInputs(py::object inputs) {
  // We assume that any CompositeTensors have already converted their components
  // from numpy arrays to Tensors, so we don't need to expand composites here
  // for the numpy array conversion. Instead, we do so because the flattened
  // inputs are eventually passed to ConcreteFunction()._call_flat, which
  // requires expanded composites.
  py::list flat_inputs = PyoOrThrow(swig::Flatten(inputs.ptr(), true));

  // Check for NumPy arrays in arguments and convert them to Tensors.
  // TODO(nareshmodi): Skip ndarray conversion to tensor altogether, perhaps
  // finding a way to store them directly in the cache key (currently not
  // possible since ndarrays are not hashable).
  bool need_packing = false;
  py::list filtered_flat_inputs;

  for (int i = 0; i < flat_inputs.size(); ++i) {
    py::handle value = flat_inputs[i].ptr();
    PyObject* value_ptr = value.ptr();
    if (swig::IsTensor(value_ptr) || swig::IsBaseResourceVariable(value_ptr)) {
      filtered_flat_inputs.append(flat_inputs[i]);
    } else if (py::hasattr(value, "__array__") && !(
        py::hasattr(value, "_should_act_as_resource_variable")
        || PyObject_IsInstance(value_ptr, np_str->ptr())
        || PyType_Check(value_ptr)
        || swig::IsCompositeTensor(value_ptr))) {
      // This case is equivalent to IsNdarray(value) == True
      py::object a = AsNdarray(flat_inputs[i]);
      if (!PyObject_IsInstance(a.ptr(), np_ndarray->ptr())) {
        throw py::type_error(strings::StrCat(
            "The output of __array__ must be an np.ndarray (got ",
            Py_TYPE(a.ptr())->tp_name, " from ",
            Py_TYPE(value_ptr)->tp_name, ")."));
      }
      flat_inputs[i] = (*create_constant_tensor)(a);
      filtered_flat_inputs.append(flat_inputs[i]);
      need_packing = true;
    }
  }

  if (need_packing) {
    return py::make_tuple(
        nest->attr("pack_sequence_as")(inputs, flat_inputs, true),
        flat_inputs, filtered_flat_inputs);
  } else {
    return py::make_tuple(inputs, flat_inputs, filtered_flat_inputs);
  }
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
  m.def("_convert_numpy_inputs", &ConvertNumpyInputs,
        "Convert numpy array inputs to tensors.");
}

}  // namespace tensorflow
