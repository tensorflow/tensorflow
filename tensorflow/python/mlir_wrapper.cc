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
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

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

  m.def("ImportFunction",
        [](const py::handle &context, const std::string &functiondef,
           const std::string &pass_pipeline, bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          auto *ctxt = static_cast<TFE_Context *>(
              PyCapsule_GetPointer(context.ptr(), nullptr));
          if (!ctxt) throw py::error_already_set();
          std::string output = tensorflow::ImportFunction(
              functiondef, pass_pipeline, show_debug_info, ctxt, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def("ImportGraphDef",
        [](const std::string &graphdef, const std::string &pass_pipeline,
           bool show_debug_info, const std::string &input_names,
           const std::string &input_data_types,
           const std::string &input_data_shapes,
           const std::string &output_names) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output = tensorflow::ImportGraphDef(
              graphdef, pass_pipeline, show_debug_info, input_names,
              input_data_types, input_data_shapes, output_names, status.get());
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
        [](const std::string &saved_model_path,
           const std::string &exported_names_str, const std::string &tags,
           bool upgrade_legacy, bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output =
              tensorflow::ExperimentalConvertSavedModelV1ToMlirLite(
                  saved_model_path, exported_names_str, tags, upgrade_legacy,
                  show_debug_info, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def("ExperimentalConvertSavedModelV1ToMlir",
        [](const std::string &saved_model_path,
           const std::string &exported_names_str, const std::string &tags,
           bool lift_variables, bool upgrade_legacy, bool show_debug_info) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          std::string output =
              tensorflow::ExperimentalConvertSavedModelV1ToMlir(
                  saved_model_path, exported_names_str, tags, lift_variables,
                  upgrade_legacy, show_debug_info, status.get());
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
