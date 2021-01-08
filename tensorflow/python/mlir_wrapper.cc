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

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/python/mlir.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace {
TFE_Context *GetContextHandle(PyObject *py_context) {
  tensorflow::Safe_PyObjectPtr py_context_handle(
      PyObject_GetAttrString(py_context, "_handle"));
  if (py_context_handle == nullptr) {
    // Current Python code makes sure this never happens. If it does, or
    // becomes hard to maintain, we can call the ensure_initialized() method
    // here.
    PyErr_SetString(PyExc_TypeError,
                    "Expected context to have a `_handle` attribute but it did "
                    "not. Was eager Context initialized?");
    return nullptr;
  }

  auto *ctx = reinterpret_cast<TFE_Context *>(
      PyCapsule_GetPointer(py_context_handle.get(), nullptr));
  if (ctx == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "Expected context._handle to contain a PyCapsule "
                        "encoded pointer to TFE_Context. Got ",
                        Py_TYPE(py_context_handle.get())->tp_name)
                        .c_str());
  }
  return ctx;
}
}  // namespace

PYBIND11_MODULE(_pywrap_mlir, m) {
  m.def("ImportGraphDef",
        [](const std::string &graphdef, const std::string &pass_pipeline,
           bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output = tensorflow::ImportGraphDef(
              graphdef, pass_pipeline, show_debug_info, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def("ImportFunction", [](const std::string &functiondef,
                             const std::string &functiondef_library,
                             const std::string &pass_pipeline,
                             bool show_debug_info) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    py::object obj = py::reinterpret_steal<py::object>(GetPyEagerContext());
    TFE_Context *context = GetContextHandle(obj.ptr());
    if (!context) throw py::error_already_set();
    std::string output = tensorflow::ImportFunction(
        functiondef, pass_pipeline, show_debug_info, context, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("ExperimentalConvertSavedModelToMlir",
        [](const std::string &saved_model_path,
           const std::string &exported_names, bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output = tensorflow::ExperimentalConvertSavedModelToMlir(
              saved_model_path, exported_names, show_debug_info, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def("ExperimentalConvertSavedModelV1ToMlirLite",
        [](const std::string &saved_model_path, const std::string &tags,
           bool upgrade_legacy, bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output =
              tensorflow::ExperimentalConvertSavedModelV1ToMlirLite(
                  saved_model_path, tags, upgrade_legacy, show_debug_info,
                  status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def("ExperimentalConvertSavedModelV1ToMlir",
        [](const std::string &saved_model_path, const std::string &tags,
           bool lift_variables, bool upgrade_legacy, bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output =
              tensorflow::ExperimentalConvertSavedModelV1ToMlir(
                  saved_model_path, tags, lift_variables, upgrade_legacy,
                  show_debug_info, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def("ExperimentalRunPassPipeline",
        [](const std::string &mlir_txt, const std::string &pass_pipeline,
           bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output = tensorflow::ExperimentalRunPassPipeline(
              mlir_txt, pass_pipeline, show_debug_info, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });
};
