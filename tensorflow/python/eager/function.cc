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

#include <vector>
#include <map>
#include <unordered_map>

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

py::object AsNdarray(py::handle value);
bool IsNdarray(py::handle value);
py::tuple ConvertNumpyInputs(py::object inputs);

class PyConcreteFunction {
 public:
  PyConcreteFunction() {}
  py::object BuildCallOutputs(py::object result,
                              py::object structured_outputs,
                              bool _ndarrays_list, bool _ndarray_singleton);
};

// TODO(jlchu): Add Pickling support; see
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
class FunctionSpec {
 public:
  py::object fullargspec_;
  bool is_method_;
  bool is_pure_;
  bool experimental_follow_type_hints_;
  std::string name_;

  py::list arg_names_;
  py::list kwonlyargs_;
  py::dict kwonlydefaults_;
  py::dict annotations_;
  std::unordered_map<std::string, int> args_to_indices_;
  std::map<int, py::object> arg_indices_to_default_values_;

  bool input_signature_is_none_;
  py::tuple input_signature_;
  py::tuple flat_input_signature_;

  FunctionSpec() {}
  FunctionSpec(py::object fullargspec,
               bool is_method,
               py::handle input_signature,
               bool is_pure,
               bool experimental_follow_type_hints,
               py::str name);
  std::string SignatureSummary(bool default_values = false);
  py::tuple ConvertVariablesToTensors(py::tuple args, py::dict kwargs);
  py::tuple ConvertAnnotatedArgsToTensors(py::tuple oargs, py::dict kwargs);
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

FunctionSpec::FunctionSpec(py::object fullargspec,
                           bool is_method,
                           py::handle input_signature,
                           bool is_pure,
                           bool experimental_follow_type_hints,
                           py::str name)
    : fullargspec_(fullargspec),
      is_method_(is_method),
      is_pure_(is_pure),
      experimental_follow_type_hints_(experimental_follow_type_hints),
      // # TODO(edloper): Include name when serializing for SavedModel?
      name_(PyObject_IsTrue(name.ptr()) ? name : "f"),
      arg_names_(fullargspec.attr("args")),
      kwonlyargs_(fullargspec.attr("kwonlyargs")),
      annotations_(fullargspec.attr("annotations")),
      input_signature_is_none_(input_signature.is_none()) {
  if (is_method_) {
    // Remove `self`: default arguments shouldn't be matched to it.
    // TODO(b/127938157): Should this error out if there is no arg to
    // be removed?
    py::list sliced_args = py::list(arg_names_.size() - 1);
    for (int i = 0; i < sliced_args.size(); ++i) {
      sliced_args[i] = arg_names_[i + 1];
    }
    arg_names_ = sliced_args;
  }

  py::object kwonlydefaults_or_none = fullargspec.attr("kwonlydefaults");
  if (!kwonlydefaults_or_none.is_none()) {
    kwonlydefaults_ = kwonlydefaults_or_none;
  }

  // A cache mapping from argument name to index, for canonicalizing
  // arguments that are called in a keyword-like fashion.
  for (int i = 0; i < arg_names_.size(); ++i) {
    std::string argname_string = PyObjectToString(arg_names_[i].ptr());
    args_to_indices_[argname_string] = i;
  }

  // A cache mapping from arg index to default value, for canonicalization.
  py::object defaults_or_none = fullargspec.attr("defaults");
  if (!defaults_or_none.is_none()) {
    py::tuple default_values = py::cast<py::tuple>(defaults_or_none);
    int offset = arg_names_.size();
    if (!default_values.empty()) {
      offset -= default_values.size();
      for (int i = 0; i < default_values.size(); ++i) {
        arg_indices_to_default_values_[offset + i] = default_values[i];
      }
    }
  }

  if (!input_signature_is_none_) {
    py::set kwonly_nodefaults = py::set(kwonlyargs_);
    if (!kwonlydefaults_.empty()) {
      kwonly_nodefaults -= py::set(kwonlydefaults_);
    }
    if (!kwonly_nodefaults.empty()) {
      throw py::value_error("Cannot define a TensorFlow function from a Python "
                            "function with keyword-only arguments when "
                            "input_signature is provided.");
    }

    if (!py::isinstance<py::tuple>(input_signature) &&
        !py::isinstance<py::list>(input_signature)) {
      throw py::type_error(tensorflow::strings::StrCat(
          "input_signature must be either a tuple or a list, received ",
          Py_TYPE(input_signature.ptr())->tp_name));
    }

    input_signature_ = py::cast<py::tuple>(input_signature);
    flat_input_signature_ = py::tuple(
        PyoOrThrow(swig::Flatten(input_signature.ptr(), true)));
  }
}

std::string FunctionSpec::SignatureSummary(bool default_values) {
  std::vector<std::string> args;
  for (const auto& arg : arg_names_) {
    args.push_back(py::cast<py::str>(arg));
  }

  if (default_values) {
    for (const auto& arg : arg_indices_to_default_values_) {
      tensorflow::strings::StrAppend(
          &args[arg.first], "=", PyObjectToString(arg.second.ptr()));
    }
  }
  if (!kwonlyargs_.empty()) {
    args.push_back("*");
    for (const auto& arg_name : kwonlyargs_) {
      args.push_back(py::cast<py::str>(arg_name));
      if (default_values && kwonlydefaults_.contains(arg_name)) {
        tensorflow::strings::StrAppend(
            &args[args.size() - 1], "=",
            std::string(py::cast<py::str>(kwonlydefaults_[arg_name])));
      }
    }
  }

  return tensorflow::strings::StrCat(name_, "(", JoinVector(", ", args), ")");
}

py::tuple FunctionSpec::ConvertVariablesToTensors(py::tuple args,
                                                  py::dict kwargs) {
  static const py::object* convert_to_tensor = new py::object(
      py::module::import("tensorflow.python.framework.ops")
          .attr("convert_to_tensor"));

  py::list converted_args;
  for (const auto& item : args) {
    converted_args.append((*convert_to_tensor)(item));
  }
  py::dict converted_kwargs;
  for (const auto& item : kwargs) {
    converted_kwargs[item.first] = (*convert_to_tensor)(item.second);
  }
  return py::make_tuple(py::tuple(converted_args), converted_kwargs);
}

// TODO(jlchu): Simplify logic by looking for .get equivalent in PyBind or C API
py::tuple FunctionSpec::ConvertAnnotatedArgsToTensors(py::tuple oargs,
                                                      py::dict kwargs) {
  static const py::object* convert_to_tensor = new py::object(
      py::module::import("tensorflow.python.framework.ops")
          .attr("convert_to_tensor"));
  static const py::object Tensor =
      py::module::import("tensorflow.python.framework.ops").attr("Tensor");

  py::list args = py::list(oargs);

  for (int i = 0; i < args.size(); ++i) {
    // See
    // https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
    if (i < arg_names_.size()) {
      if (annotations_.contains(arg_names_[i])) {
        py::handle arg_annotation = annotations_[arg_names_[i]];
        // TODO(rahulkamat): Change to TensorLike (here ans below).
        if (arg_annotation.ptr() == Tensor.ptr()) {
          args[i] = (*convert_to_tensor)(args[i]);
        }
      }
    } else {
      py::object varargs_string = fullargspec_.attr("varargs");
      if (!varargs_string.is_none() && annotations_.contains(varargs_string)) {
        py::handle varargs_annotation = annotations_[varargs_string];
        if (varargs_annotation.ptr() == Tensor.ptr()) {
          args[i] = (*convert_to_tensor)(args[i]);
        }
      }
    }
  }

  py::set kwonlyargs_set = py::set(kwonlyargs_);
  py::set args_set = py::set(arg_names_);
  for (auto& item : kwargs) {
    if (kwonlyargs_set.contains(item.first)) {
      if (annotations_.contains(item.first)) {
        py::handle kwonlyarg_annotation = annotations_[item.first];
        if (kwonlyarg_annotation.ptr() == Tensor.ptr()) {
          kwargs[item.first] = (*convert_to_tensor)(item.second);
        }
      }
    } else {
      py::object varkw_string = fullargspec_.attr("varkw");
      if (!varkw_string.is_none()) {
        if (args_set.contains(item.first)) {
          if (annotations_.contains(item.first)) {
            py::handle arg_annotation = annotations_[item.first];
            if (arg_annotation.ptr() == Tensor.ptr()) {
              kwargs[item.first] = (*convert_to_tensor)(item.second);
            }
          }
        } else if (annotations_.contains(varkw_string)) {
          py::handle varkw_annotation = annotations_[varkw_string];
          if (varkw_annotation.ptr() == Tensor.ptr()) {
            kwargs[item.first] = (*convert_to_tensor)(item.second);
          }
        }
      }
    }
  }

  return py::make_tuple(py::tuple(args), kwargs);
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
  py::class_<FunctionSpec>(m, "FunctionSpec")
      .def(py::init<py::object, bool, py::handle, bool, bool, py::str>())
      .def("signature_summary", &FunctionSpec::SignatureSummary)
      .def("_convert_variables_to_tensors",
           &FunctionSpec::ConvertVariablesToTensors)
      .def("_convert_annotated_args_to_tensors",
           &FunctionSpec::ConvertAnnotatedArgsToTensors,
           "Attempts to autobox arguments annotated as tf.Tensor.");
  m.def("_as_ndarray", &AsNdarray,
        "Converts value to an ndarray, assumes _is_ndarray(value).");
  m.def(
      "_is_ndarray", &IsNdarray,
      "Tests whether the given value is an ndarray (and not a TF tensor/var).");
  m.def("_convert_numpy_inputs", &ConvertNumpyInputs,
        "Convert numpy array inputs to tensors.");
}

}  // namespace tensorflow
