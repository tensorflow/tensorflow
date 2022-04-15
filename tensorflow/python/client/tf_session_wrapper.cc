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

// Must be at top (before any system includes and Python.h).
// clang-format off
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
// clang-format on

#include "Python.h"
#include "absl/types/optional.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/python_api.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/version_info.h"
#include "tensorflow/python/client/tf_session_helper.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace pybind11 {
namespace detail {
// Convert between absl::optional and python.
//
// pybind11 supports std::optional, and absl::optional is meant to be a
// drop-in replacement for std::optional, so we can just use the built in
// implementation.
#ifndef ABSL_USES_STD_OPTIONAL
template <typename T>
struct type_caster<absl::optional<T>>
    : public optional_caster<absl::optional<T>> {};
template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

}  // namespace detail
}  // namespace pybind11

// TODO(amitpatankar): Consolidate Buffer methods into a separate header file.
TF_Buffer* ProtoStringToTFBuffer(PyObject* input) {
  // Convert a Python string object to TF_Buffer.
  char* c_string;
  Py_ssize_t py_size;
  // PyBytes_AsStringAndSize() does not copy but simply interprets the input
  if (PyBytes_AsStringAndSize(input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    throw py::error_already_set();
  }
  return TF_NewBufferFromString(static_cast<void*>(c_string),
                                static_cast<size_t>(py_size));
}

// Copied from tf_session.i
// We have to do convoluted logic of passing in a vector of py::bytes. If we
// pass in strings they are freed prior to the necessary function calls.
tensorflow::NameVector ConvertPyListToNameVector(
    const std::vector<py::bytes>& py_vector) {
  tensorflow::NameVector temp;
  for (size_t i = 0; i < py_vector.size(); ++i) {
    const char* string_elem = PyBytes_AsString(py_vector.at(i).ptr());
    temp.push_back(string_elem);
  }
  return temp;
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(TF_Graph);
PYBIND11_MAKE_OPAQUE(TF_Session);
PYBIND11_MAKE_OPAQUE(TF_Operation);
PYBIND11_MAKE_OPAQUE(TF_Buffer);
PYBIND11_MAKE_OPAQUE(TF_ImportGraphDefOptions);
PYBIND11_MAKE_OPAQUE(TF_ImportGraphDefResults);
PYBIND11_MAKE_OPAQUE(TF_DeprecatedSession);
PYBIND11_MAKE_OPAQUE(TF_OperationDescription);
PYBIND11_MAKE_OPAQUE(TF_Library);
PYBIND11_MAKE_OPAQUE(TF_SessionOptions);
PYBIND11_MAKE_OPAQUE(TF_ApiDefMap);
PYBIND11_MAKE_OPAQUE(TF_Server);
PYBIND11_MAKE_OPAQUE(TF_DeviceList);
PYBIND11_MAKE_OPAQUE(TF_Status);

PYBIND11_MODULE(_pywrap_tf_session, m) {
  // Numpy initialization code for array checks.
  tensorflow::ImportNumpy();

  py::class_<TF_Graph> TF_Graph_class(m, "TF_Graph");
  py::class_<TF_Operation> TF_Operation_class(m, "TF_Operation");

  py::class_<TF_Output>(m, "TF_Output")
      .def(py::init<>())
      .def_readwrite("oper", &TF_Output::oper)
      .def_readwrite("index", &TF_Output::index);

  py::class_<TF_Input>(m, "TF_Input")
      .def(py::init<>())
      .def_readwrite("oper", &TF_Input::oper)
      .def_readwrite("index", &TF_Input::index);

  py::class_<TF_ImportGraphDefOptions> TF_ImportGraphDefOptions_class(
      m, "TF_ImportGraphDefOptions");
  py::class_<TF_ImportGraphDefResults> TF_ImportGraphDefResults_class(
      m, "TF_ImportGraphDefResults");
  py::class_<TF_DeprecatedSession> TF_DeprecatedSession_class(
      m, "TF_DeprecatedSession");
  py::class_<TF_Session> TF_Session_class(m, "TF_Session");
  py::class_<TF_OperationDescription> TF_OperationDescription_class(
      m, "TF_OperationDescription");
  py::class_<TF_Library> TF_Library_class(m, "TF_Library");
  py::class_<TF_SessionOptions> TF_SessionOptions_class(m, "TF_SessionOptions");
  py::class_<TF_Buffer> TF_Buffer_class(m, "TF_Buffer");
  py::class_<TF_ApiDefMap> TF_ApiDefMap_class(m, "TF_ApiDefMap");
  py::class_<TF_Server> TF_Server_class(m, "TF_Server");
  py::class_<TF_Status> TF_Status_class(m, "TF_Status");

  // We only release the Python GIL for certain methods that are
  // not explicitly marked. We disable this behavior for some functions
  // because they uses Python method(s) that expect the GIL to be held
  // (at least PyArray_Return, maybe others).

  // Do not release GIL.
  m.def("TF_OperationGetControlInputs_wrapper",
        tensorflow::TF_OperationGetControlInputs_wrapper);
  // Do not release GIL.
  m.def("TF_OperationGetControlOutputs_wrapper",
        tensorflow::TF_OperationGetControlOutputs_wrapper);
  m.def("TF_OperationOutputConsumers_wrapper",
        tensorflow::TF_OperationOutputConsumers_wrapper);
  // Do not release GIL.
  m.def("GetOperationInputs", tensorflow::GetOperationInputs);
  // Do not release GIL.
  m.def("TF_ImportGraphDefOptionsSetValidateColocationConstraints",
        TF_ImportGraphDefOptionsSetValidateColocationConstraints);
  // Do not release GIL.
  m.def("TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper",
        tensorflow::TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper);
  m.def("TF_SessionMakeCallable",
        [](TF_Session* session, const TF_Buffer* callable_options) {
          int64_t out_handle;
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());

          // Release GIL.
          py::gil_scoped_release release;
          tensorflow::TF_SessionMakeCallable(session, callable_options,
                                             &out_handle, status.get());

          // Acquire GIL for returning int conversion.
          pybind11::gil_scoped_acquire acquire;
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return out_handle;
        });
  m.def("_TF_SetTarget", TF_SetTarget);
  m.def("_TF_SetConfig", [](TF_SessionOptions* options, py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
    TF_SetConfig(options, buf.get()->data, buf.get()->length, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("_TF_NewSessionOptions", TF_NewSessionOptions,
        py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteSessionOptions", TF_DeleteSessionOptions,
        py::call_guard<py::gil_scoped_release>());

  m.def("EqualGraphDefWrapper", tensorflow::EqualGraphDefWrapper,
        py::call_guard<py::gil_scoped_release>());
  m.def("EqualAttrValueWrapper", tensorflow::EqualAttrValueWrapper,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_GraphToFunction_wrapper",
      [](const TF_Graph* fn_body, const char* fn_name,
         bool append_hash_to_fn_name,
         absl::optional<std::vector<TF_Operation*>> opers_opt,
         const std::vector<TF_Output>& inputs,
         const std::vector<TF_Output>& outputs,
         const std::vector<py::bytes> output_names,
         const std::vector<TF_Operation*> control_outputs,
         const std::vector<py::bytes> control_output_names, py::none opts,
         const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());

        // TODO(b/147674626): Use pybind11 list_caster instead.
        tensorflow::NameVector output_names_name_vector =
            ConvertPyListToNameVector(output_names);

        // TODO(b/147674626): Use pybind11 list_caster instead.
        tensorflow::NameVector control_output_names_name_vector =
            ConvertPyListToNameVector(control_output_names);

        // Release GIL.
        py::gil_scoped_release release;
        auto output = tensorflow::TF_GraphToFunction_wrapper(
            fn_body, fn_name, append_hash_to_fn_name,
            opers_opt.has_value() ? &opers_opt.value() : nullptr, inputs,
            outputs, output_names_name_vector, &control_outputs,
            control_output_names_name_vector,
            /*opts=*/nullptr, description, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_GraphGetTensorShapeHelper", [](TF_Graph* graph, TF_Output output) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    bool unknown_shape;

    auto result = tensorflow::TF_GraphGetTensorShapeHelper(
        graph, output, status.get(), &unknown_shape);
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    // Create a python list from InlinedVector
    py::list py_list;
    for (size_t i = 0; i < result.size(); ++i) {
      py_list.append(py::cast(result[i]));
    }

    // Return a tuple.
    py::tuple result_tuple = py::make_tuple(py_list, py::cast(unknown_shape));
    return result_tuple;
  });

  m.def("TF_GraphSetTensorShape_wrapper",
        [](TF_Graph* graph, TF_Output output, const std::vector<int64_t>& dims,
           bool unknown_shape) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());

          // Release GIL.
          py::gil_scoped_release release;
          tensorflow::TF_GraphSetTensorShape_wrapper(
              graph, output, dims, unknown_shape, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_GraphGetTensorShape_wrapper",
        [](TF_Graph* graph, TF_Output output, const std::vector<int64_t>& dims,
           bool unknown_shape) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          tensorflow::TF_GraphSetTensorShape_wrapper(
              graph, output, dims, unknown_shape, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_GraphSetOutputHandleShapesAndTypes_wrapper",
        [](TF_Graph* graph, TF_Output output,
           const std::vector<absl::optional<std::vector<int64_t>>>& shapes,
           const std::vector<int>& ranks, py::handle& types) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());

          // Cast types
          std::vector<TF_DataType> types_local;
          PyObject* seq =
              PySequence_Fast(types.ptr(), "$symname: expected list");
          if (seq == nullptr) {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname: PySequence_Fast returned NULL.");
            throw py::error_already_set();
          }

          int size = PySequence_Fast_GET_SIZE(seq);
          if (size == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "$symname: shapes list must be non-empty");
            throw py::error_already_set();
          }

          for (int i = 0; i < size; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            types_local.push_back((TF_DataType)PyLong_AsLong(item));
          }

          // Convert shapes nested vector
          std::vector<std::vector<int64_t>> shapes_local;
          for (size_t i = 0; i < shapes.size(); ++i) {
            std::vector<int64_t> dims;
            std::vector<int64_t> item =
                shapes[i].has_value() ? shapes[i].value() : dims;
            shapes_local.push_back(item);
          }

          Py_DECREF(seq);

          tensorflow::TF_GraphSetOutputHandleShapesAndTypes_wrapper(
              graph, output, shapes_local, ranks, types_local, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  // Do not release GIL.
  m.def("TF_CreatePlaceholders",
        [](TF_Graph* graph, py::handle& dtypes, const char* prefix) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          auto output = tensorflow::TF_CreatePlaceholders(graph, dtypes.ptr(),
                                                          prefix, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def(
      "TF_NewSession",
      [](TF_Graph* graph, const TF_SessionOptions* opts) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_NewSession(graph, opts, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_NewSessionRef",
      [](TF_Graph* graph, const TF_SessionOptions* opts) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = tensorflow::TF_NewSessionRef(graph, opts, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_CloseSession", [](TF_Session* session) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());

    // Release GIL.
    py::gil_scoped_release release;
    TF_CloseSession(session, status.get());

    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_DeleteSession", [](TF_Session* session) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_DeleteSession(session, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("SetRequireShapeInferenceFns", tensorflow::SetRequireShapeInferenceFns);

  // Do not release GIL.
  m.def("TF_TryEvaluateConstant_wrapper",
        [](TF_Graph* graph, const TF_Output output) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          auto result = tensorflow::TF_TryEvaluateConstant_wrapper(
              graph, output, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return tensorflow::PyoOrThrow(result);
        });

  m.def("ExtendSession", [](TF_Session* session) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    pybind11::gil_scoped_release release;
    tensorflow::ExtendSession(session, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("GetHandleShapeAndType", [](TF_Graph* graph, TF_Output output) {
    std::string output_string =
        tensorflow::GetHandleShapeAndType(graph, output);
    // Override default py3 behavior of attempting to encode into Unicode as
    // the dependent functions expect bytes.
    return py::bytes(output_string);
  });

  m.def("SetHandleShapeAndType",
        [](TF_Graph* graph, TF_Output output, py::bytes proto) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          tensorflow::Safe_TF_BufferPtr buf =
              tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
          tensorflow::SetHandleShapeAndType(graph, output, buf.get()->data,
                                            buf.get()->length, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  // Do not release GIL.
  m.def("TF_SessionRun_wrapper", [](TF_Session* session, TF_Buffer* run_options,
                                    const py::handle& input_dict,
                                    const std::vector<TF_Output>& outputs,
                                    const std::vector<TF_Operation*>& targets,
                                    TF_Buffer* run_metadata) {
    // Convert inputs dictionary
    std::vector<TF_Output> inputs;
    std::vector<PyObject*> input_ndarrays;
    if (!PyDict_Check(input_dict.ptr())) {
      PyErr_SetString(
          PyExc_TypeError,
          "Expected a dictionary as an argument to TF_SessionRun_wrapper.");
      throw py::error_already_set();
    }
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_dict.ptr(), &pos, &key, &value)) {
      TF_Output item = py::cast<TF_Output>(key);
      inputs.push_back(item);

      // TODO(amitpatankar): Fix this PyArray check. (b/147855599)

      // if (!PyArray_Check(value)) {
      //   PyErr_SetString(
      //       PyExc_TypeError,
      //       "$symname: Expected all values in input dict to be ndarray.");
      //   throw py::error_already_set();
      // }
      input_ndarrays.push_back(value);
    }

    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    std::vector<PyObject*> py_outputs;
    tensorflow::TF_SessionRun_wrapper(session, run_options, inputs,
                                      input_ndarrays, outputs, targets,
                                      run_metadata, status.get(), &py_outputs);
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    // Create a Python list using the C API rather than py::list. b/147855599
    PyObject* result = PyList_New(py_outputs.size());
    if (result == nullptr) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create a list.");
      throw py::error_already_set();
    }
    for (size_t i = 0; i < py_outputs.size(); ++i) {
      PyList_SET_ITEM(result, i, py_outputs.at(i));
    }

    return tensorflow::PyoOrThrow(result);
  });

  // Do not release GIL.
  m.def("TF_SessionPRun_wrapper", [](TF_Session* session, const char* handle,
                                     const py::handle& input_dict,
                                     const std::vector<TF_Output>& outputs) {
    // Convert inputs dictionary
    std::vector<TF_Output> inputs;
    std::vector<PyObject*> input_ndarrays;
    if (!PyDict_Check(input_dict.ptr())) {
      PyErr_SetString(
          PyExc_TypeError,
          "Expected a dictionary as an argument to TF_SessionPRun_wrapper.");
      throw py::error_already_set();
    }
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_dict.ptr(), &pos, &key, &value)) {
      TF_Output item = py::cast<TF_Output>(key);
      inputs.push_back(item);

      // TODO(amitpatankar): Fix this PyArray check. (b/147855599)

      // if (!PyArray_Check(value)) {
      //   PyErr_SetString(
      //       PyExc_TypeError,
      //       "$symname: Expected all values in input dict to be ndarray.");
      //   throw py::error_already_set();
      // }
      input_ndarrays.push_back(value);
    }

    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    std::vector<PyObject*> py_outputs;
    tensorflow::TF_SessionPRun_wrapper(session, handle, inputs, input_ndarrays,
                                       outputs, status.get(), &py_outputs);
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    PyObject* result = PyList_New(py_outputs.size());
    if (result == nullptr) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create a list.");
      throw py::error_already_set();
    }
    for (size_t i = 0; i < py_outputs.size(); ++i) {
      PyList_SET_ITEM(result, i, py_outputs.at(i));
    }

    return tensorflow::PyoOrThrow(result);
  });

  // Do not release GIL.
  m.def("TF_SessionPRunSetup_wrapper",
        [](TF_Session* session, const std::vector<TF_Output>& inputs,
           const std::vector<TF_Output>& outputs,
           const std::vector<TF_Operation*>& targets) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          const char* out_handle;
          tensorflow::TF_SessionPRunSetup_wrapper(
              session, inputs, outputs, targets, &out_handle, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return out_handle;
        });

  // Do not release GIL.
  m.def("TF_SessionRunCallable", [](TF_Session* session, int64_t handle,
                                    py::object feed_values,
                                    TF_Buffer* run_metadata) {
    tensorflow::PyObjectVector out_values;
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::TF_SessionRunCallable(session, handle, feed_values.ptr(),
                                      &out_values, run_metadata, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    // Return out_values
    py::list py_list;
    for (size_t i = 0; i < out_values.size(); ++i) {
      py::object obj = tensorflow::Pyo(out_values.at(i));
      py_list.append(obj);
    }
    return py_list;
  });

  m.def("TF_SessionReleaseCallable", [](TF_Session* session, int64_t handle) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    tensorflow::TF_SessionReleaseCallable(session, handle, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_NewGraph", TF_NewGraph, py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteGraph", TF_DeleteGraph,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_GraphGetOpDef",
        [](TF_Graph* graph, const char* op_name, TF_Buffer* output_op_def) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_GraphGetOpDef(graph, op_name, output_op_def, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def(
      "TF_NewOperation",
      [](TF_Graph* graph, const char* op_type, const char* oper_name) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        TF_OperationDescription* output =
            TF_NewOperation(graph, op_type, oper_name);
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_FinishOperation",
      [](TF_OperationDescription* desc) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        TF_Operation* output = TF_FinishOperation(desc, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_OperationGetAttrInt",
        [](TF_Operation* oper, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          int64_t value;
          // Release GIL.
          py::gil_scoped_release release;
          TF_OperationGetAttrInt(oper, attr_name, &value, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
          // Convert TF_OperationGetAttrInt int64_t* out-argument to Python
          // bool.
          // Acquire GIL for returning output returning.
          pybind11::gil_scoped_acquire acquire;
          return tensorflow::Pyo(PyLong_FromLongLong(value));
        });

  m.def("TF_SetAttrValueProto", [](TF_OperationDescription* desc,
                                   const char* attr_name, py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
    TF_SetAttrValueProto(desc, attr_name, buf.get()->data, buf.get()->length,
                         status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });

  m.def("TF_OperationNumOutputs", TF_OperationNumOutputs,
        py::call_guard<py::gil_scoped_release>());

  // Convert types to ints
  m.def("TF_OperationInputType", TF_OperationInputType,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_OperationOutputType", TF_OperationOutputType,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_OperationName", TF_OperationName,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_OperationOpType", TF_OperationOpType,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_OperationDevice", TF_OperationDevice,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_AddInput", TF_AddInput);

  m.def("TF_OperationToNodeDef",
        [](TF_Operation* oper, TF_Buffer* output_node_def) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TF_OperationToNodeDef(oper, output_node_def, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  m.def("TF_OperationGetAttrValueProto",
        [](TF_Operation* oper, const char* attr_name,
           TF_Buffer* output_attr_value) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TF_OperationGetAttrValueProto(oper, attr_name, output_attr_value,
                                        status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  m.def("TF_OperationGetStackTrace", [](TF_Operation* oper) -> py::object {
    const std::shared_ptr<tensorflow::AbstractStackTrace> trace =
        oper->node.GetStackTrace();
    if (!trace) {
      return py::none();
    }
    return py::cast(*trace, py::return_value_policy::reference);
  });

  m.def("SetRequestedDevice", tensorflow::SetRequestedDevice);

  // TF_Buffer util methods
  // TODO(amitpatankar): Consolidate Buffer methods into a separate header file.
  m.def("TF_NewBuffer", TF_NewBuffer, py::return_value_policy::reference);
  m.def("TF_GetBuffer", [](TF_Buffer* buf) {
    TF_Buffer buffer = TF_GetBuffer(buf);
    return tensorflow::PyoOrThrow(PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(buffer.data), buffer.length));
  });
  m.def("TF_DeleteBuffer", &TF_DeleteBuffer);
  m.def(
      "TF_NewBufferFromString",
      [](py::bytes buffer_as_string) {
        tensorflow::Safe_TF_BufferPtr buf = tensorflow::make_safe(
            ProtoStringToTFBuffer(buffer_as_string.ptr()));
        return TF_NewBufferFromString(buf.get()->data, buf.get()->length);
      },
      py::return_value_policy::reference);

  m.def("SetAttr", [](TF_Graph* graph, TF_Operation* op, const char* attr_name,
                      TF_Buffer* attr_value_proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    tensorflow::SetAttr(graph, op, attr_name, attr_value_proto, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("ClearAttr",
        [](TF_Graph* graph, TF_Operation* op, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          tensorflow::ClearAttr(graph, op, attr_name, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  // Note: users should prefer using tf.cast or equivalent, and only when
  // it's infeasible to set the type via OpDef's type constructor and inference
  // function.
  m.def("SetFullType", [](TF_Graph* graph, TF_Operation* op,
                          const std::string& serialized_full_type) {
    tensorflow::FullTypeDef proto;
    proto.ParseFromString(serialized_full_type);
    tensorflow::SetFullType(graph, op, proto);
  });

  m.def(
      "TF_LoadLibrary",
      [](const char* library_filename) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TF_LoadLibrary(library_filename, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_LoadPluggableDeviceLibrary",
      [](const char* library_filename) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TF_LoadPluggableDeviceLibrary(library_filename, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_GetOpList", [](TF_Library* lib_handle) {
    TF_Buffer output_buffer = TF_GetOpList(lib_handle);
    return tensorflow::PyoOrThrow(PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(output_buffer.data),
        output_buffer.length));
  });

  m.def("TF_DeleteLibraryHandle", TF_DeleteLibraryHandle,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_PluggableDeviceLibraryHandle",
        TF_DeletePluggableDeviceLibraryHandle,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_AddControlInput", TF_AddControlInput);
  m.def(
      "TF_AddInputList", [](TF_OperationDescription* desc, py::handle& inputs) {
        std::vector<TF_Output> vec;
        size_t size = PyList_Size(inputs.ptr());
        for (size_t i = 0; i < size; ++i) {
          TF_Output item = py::cast<TF_Output>(PyList_GetItem(inputs.ptr(), i));
          vec.push_back(item);
        }
        TF_AddInputList(desc, vec.data(), vec.size());
      });

  m.def("UpdateEdge", [](TF_Graph* graph, TF_Output new_src, TF_Input dst) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    tensorflow::UpdateEdge(graph, new_src, dst, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("RemoveAllControlInputs", tensorflow::RemoveAllControlInputs,
        py::call_guard<py::gil_scoped_release>());
  m.def("AddControlInput", tensorflow::AddControlInput,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_NewImportGraphDefOptions", TF_NewImportGraphDefOptions,
        py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsSetPrefix", TF_ImportGraphDefOptionsSetPrefix,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsSetUniquifyNames",
        TF_ImportGraphDefOptionsSetUniquifyNames,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsRemapControlDependency",
        TF_ImportGraphDefOptionsRemapControlDependency,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsAddInputMapping",
        TF_ImportGraphDefOptionsAddInputMapping,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsAddReturnOperation",
        TF_ImportGraphDefOptionsAddReturnOperation,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsAddReturnOutput",
        TF_ImportGraphDefOptionsAddReturnOutput,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_GraphImportGraphDefWithResults",
      [](TF_Graph* graph, const TF_Buffer* graph_def,
         const TF_ImportGraphDefOptions* options) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TF_GraphImportGraphDefWithResults(graph, graph_def,
                                                        options, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_GraphNextOperation",
      [](TF_Graph* graph, size_t pos) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TF_GraphNextOperation(graph, &pos);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

        // Returns a (TF_Operation*, int pos) tuple.
        py::tuple result_tuple = py::make_tuple(
            py::cast(output), tensorflow::Pyo(PyLong_FromSize_t(pos)));
        return result_tuple;
      },
      py::return_value_policy::reference);

  // Python needs to own deletion of outputs
  m.def("TF_ImportGraphDefResultsReturnOutputs",
        [](TF_ImportGraphDefResults* results) {
          int num_outputs;
          TF_Output* outputs;
          TF_ImportGraphDefResultsReturnOutputs(results, &num_outputs,
                                                &outputs);
          py::list py_list;
          for (int i = 0; i < num_outputs; ++i) {
            TF_Output tf_output = TF_Output(outputs[i]);
            py_list.append(tf_output);
          }
          return py_list;
        });

  m.def(
      "TF_ImportGraphDefResultsReturnOperations",
      [](TF_ImportGraphDefResults* results) {
        int num_opers;
        TF_Operation** opers;
        TF_ImportGraphDefResultsReturnOperations(results, &num_opers, &opers);
        py::list py_list;
        for (int i = 0; i < num_opers; ++i) {
          py_list.append(opers[i]);
        }
        return py_list;
      },
      py::return_value_policy::reference);

  m.def("TF_GraphToGraphDef", [](TF_Graph* graph, TF_Buffer* output_graph_def) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_GraphToGraphDef(graph, output_graph_def, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_OperationNumInputs", TF_OperationNumInputs,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_GraphVersions", [](TF_Graph* graph, TF_Buffer* output_graph_def) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_GraphVersions(graph, output_graph_def, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_DeleteFunction", TF_DeleteFunction,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteImportGraphDefResults", TF_DeleteImportGraphDefResults,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteImportGraphDefOptions", TF_DeleteImportGraphDefOptions,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_FunctionSetAttrValueProto",
        [](TF_Function* func, const char* attr_name, py::bytes proto) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          tensorflow::Safe_TF_BufferPtr buf =
              tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
          // Release GIL.
          py::gil_scoped_release release;
          TF_FunctionSetAttrValueProto(func, attr_name, buf.get()->data,
                                       buf.get()->length, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_FunctionToFunctionDef",
        [](TF_Function* graph, TF_Buffer* output_func_def) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_FunctionToFunctionDef(graph, output_func_def, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_GraphCopyFunction",
        [](TF_Graph* graph, const TF_Function* func, const TF_Function* grad) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_GraphCopyFunction(graph, func, grad, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def(
      "TF_FunctionImportFunctionDef",
      [](py::bytes proto) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        tensorflow::Safe_TF_BufferPtr buf =
            tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));

        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_FunctionImportFunctionDef(
            buf.get()->data, buf.get()->length, status.get());

        // Acquire GIL for returning output returning.
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("EqualAttrValueWrapper", tensorflow::EqualAttrValueWrapper,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_GetAllRegisteredKernels",
      []() {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_GetAllRegisteredKernels(status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_GetRegisteredKernelsForOp",
      [](const char* name) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_GetRegisteredKernelsForOp(name, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_GetAllOpList", TF_GetAllOpList, py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_NewApiDefMap",
      [](TF_Buffer* op_list_buffer) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_NewApiDefMap(op_list_buffer, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_DeleteApiDefMap", TF_DeleteApiDefMap,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_ApiDefMapGet",
      [](TF_ApiDefMap* api_def_map, const char* name, size_t name_len) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output =
            TF_ApiDefMapGet(api_def_map, name, name_len, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_ApiDefMapPut",
        [](TF_ApiDefMap* api_def_map, const char* name, size_t name_len) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_ApiDefMapPut(api_def_map, name, name_len, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_OperationGetAttrType",
        [](TF_Operation* oper, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TF_DataType value;
          // Release GIL.
          py::gil_scoped_release release;
          TF_OperationGetAttrType(oper, attr_name, &value, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
          return value;
        });

  m.def(
      "TF_NewServer",
      [](py::bytes proto) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        tensorflow::Safe_TF_BufferPtr buf =
            tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
        TF_Server* output =
            TF_NewServer(buf.get()->data, buf.get()->length, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_ServerStart", [](TF_Server* server) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_ServerStart(server, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_ServerStop", [](TF_Server* server) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    py::gil_scoped_release release;
    TF_ServerStop(server, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_ServerJoin", [](TF_Server* server) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    py::gil_scoped_release release;
    TF_ServerJoin(server, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def(
      "TF_ServerTarget",
      [](TF_Server* server) { return TF_ServerTarget(server); },
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_SessionListDevices",
      [](TF_Session* session) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        TF_DeviceList* output = TF_SessionListDevices(session, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_DeviceListCount",
        [](const TF_DeviceList* list) { return TF_DeviceListCount(list); });

  m.def("TF_DeviceListName", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    const char* output = TF_DeviceListName(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_DeviceListType", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    const char* output = TF_DeviceListType(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_DeviceListMemoryBytes", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    int64_t output = TF_DeviceListMemoryBytes(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_DeviceListIncarnation", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    int64_t output = TF_DeviceListIncarnation(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_SetDevice", TF_SetDevice);

  m.def("TF_DeleteDeviceList", TF_DeleteDeviceList);

  m.def("TF_OperationGetAttrBool",
        [](TF_Operation* oper, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          unsigned char value;
          // Release GIL for threading.
          {
            py::gil_scoped_release release;
            TF_OperationGetAttrBool(oper, attr_name, &value, status.get());
            tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
          }
          return tensorflow::Pyo(PyBool_FromLong(value));
        });

  m.def("TF_NewStatus", TF_NewStatus, py::return_value_policy::reference);
  m.def("TF_DeleteStatus", TF_DeleteStatus);

  m.def("TF_DeleteDeviceList", TF_DeleteDeviceList);

  m.def("AddWhileInputHack",
        [](TF_Graph* graph, TF_Output new_src, TF_Operation* dst) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL for threading.
          py::gil_scoped_release release;
          tensorflow::AddWhileInputHack(graph, new_src, dst, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_Reset_wrapper", [](const TF_SessionOptions* opt,
                               const std::vector<py::bytes> containers) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    py::gil_scoped_release release;
    tensorflow::NameVector containers_name_vector =
        ConvertPyListToNameVector(containers);
    tensorflow::TF_Reset_wrapper(opt, containers_name_vector, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });
  m.def("TF_GetCode", TF_GetCode);

  m.def("TF_SetXlaAutoJitMode", TF_SetXlaAutoJitMode);
  m.def("TF_GetXlaAutoJitEnabled", TF_GetXlaAutoJitEnabled);
  m.def("TF_SetXlaEnableLazyCompilation", TF_SetXlaEnableLazyCompilation);
  m.def("TF_SetTfXlaCpuGlobalJit", TF_SetTfXlaCpuGlobalJit);
  m.def("TF_SetXlaMinClusterSize", TF_SetXlaMinClusterSize);
  m.def("TF_GetXlaConstantFoldingDisabled", TF_GetXlaConstantFoldingDisabled);
  m.def("TF_SetXlaConstantFoldingDisabled", TF_SetXlaConstantFoldingDisabled);

  // // Static constants are not working on Windows. b/145559202
  // // Creating getters instead.

  m.def("get_version", []() { return TF_VERSION_STRING; });
  m.def("get_git_version", []() { return TF_GIT_VERSION; });
  m.def("get_compiler_version", []() { return TF_COMPILER_VERSION; });
  m.def("get_cxx11_abi_flag", []() { return TF_CXX11_ABI_FLAG; });
  m.def("get_eigen_max_align_bytes", []() { return EIGEN_MAX_ALIGN_BYTES; });
  m.def("get_monolithic_build", []() { return TF_MONOLITHIC_BUILD; });
  m.def("get_graph_def_version", []() { return TF_GRAPH_DEF_VERSION; });
  m.def("get_graph_def_version_min_consumer",
        []() { return TF_GRAPH_DEF_VERSION_MIN_CONSUMER; });
  m.def("get_graph_def_version_min_producer",
        []() { return TF_GRAPH_DEF_VERSION_MIN_PRODUCER; });
  m.def("get_tensor_handle_key", []() {
    // TODO(amitpatankar): Look into a more elegant solution.
    // Since this is a shared object we will hard code the value from
    // third_party/tensorflow/core/common_runtime/session_state.cc because
    // the Windows import will not load the libraries necessarily
    // in order. b/145559202
    return "TensorHandle";
  });

  m.def("TF_RegisterFilesystemPlugin", [](const char* plugin_filename) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TF_RegisterFilesystemPlugin(plugin_filename, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });

  py::enum_<TF_DataType>(m, "TF_DataType")
      .value("TF_FLOAT", TF_FLOAT)
      .value("TF_DOUBLE", TF_DOUBLE)
      .value("TF_INT32", TF_INT32)
      .value("TF_UINT8", TF_UINT8)
      .value("TF_INT16", TF_INT16)
      .value("TF_INT8", TF_INT8)
      .value("TF_STRING", TF_STRING)
      .value("TF_COMPLEX64", TF_COMPLEX64)
      .value("TF_COMPLEX", TF_COMPLEX)
      .value("TF_INT64", TF_INT64)
      .value("TF_BOOL", TF_BOOL)
      .value("TF_QINT8", TF_QINT8)
      .value("TF_QUINT8", TF_QUINT8)
      .value("TF_QINT32", TF_QINT32)
      .value("TF_BFLOAT16", TF_BFLOAT16)
      .value("TF_QINT16", TF_QINT16)
      .value("TF_QUINT16", TF_QUINT16)
      .value("TF_UINT16", TF_UINT16)
      .value("TF_COMPLEX128", TF_COMPLEX128)
      .value("TF_HALF", TF_HALF)
      .value("TF_RESOURCE", TF_RESOURCE)
      .value("TF_VARIANT", TF_VARIANT)
      .value("TF_UINT32", TF_UINT32)
      .value("TF_UINT64", TF_UINT64)
      .export_values();

  py::enum_<TF_Code>(m, "TF_Code")
      .value("TF_OK", TF_OK)
      .value("TF_CANCELLED", TF_CANCELLED)
      .value("TF_UNKNOWN", TF_UNKNOWN)
      .value("TF_INVALID_ARGUMENT", TF_INVALID_ARGUMENT)
      .value("TF_DEADLINE_EXCEEDED", TF_DEADLINE_EXCEEDED)
      .value("TF_PERMISSION_DENIED", TF_PERMISSION_DENIED)
      .value("TF_UNAUTHENTICATED", TF_UNAUTHENTICATED)
      .value("TF_RESOURCE_EXHAUSTED", TF_RESOURCE_EXHAUSTED)
      .value("TF_FAILED_PRECONDITION", TF_FAILED_PRECONDITION)
      .value("TF_ABORTED", TF_ABORTED)
      .value("TF_OUT_OF_RANGE", TF_OUT_OF_RANGE)
      .value("TF_UNIMPLEMENTED", TF_UNIMPLEMENTED)
      .value("TF_INTERNAL", TF_INTERNAL)
      .value("TF_DATA_LOSS", TF_DATA_LOSS)
      .export_values();
};
