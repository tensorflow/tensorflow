/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <pybind11/stl.h>

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

using tensorflow::calibrator::CalibratorSingleton;

PyObject* QuantizeQATModel(absl::string_view saved_model_path,
                           absl::string_view exported_names_str,
                           absl::string_view tags) {
  auto graph_def_or =
      mlir::quant::QuantizeQATModel(saved_model_path, exported_names_str, tags);
  if (!graph_def_or.ok()) {
    PyErr_Format(PyExc_ValueError,
                 graph_def_or.status().error_message().c_str());
    return nullptr;
  }

  std::string ret_str = graph_def_or.ValueOrDie().SerializeAsString();

  return tflite::python_utils::ConvertToPyString(ret_str.c_str(),
                                                 ret_str.size());
}

PyObject* QuantizePTQModelPreCalibration(absl::string_view saved_model_path,
                                         absl::string_view exported_names_str,
                                         absl::string_view tags) {
  auto graph_def_or = mlir::quant::QuantizePTQModelPreCalibration(
      saved_model_path, exported_names_str, tags);
  if (!graph_def_or.ok()) {
    PyErr_Format(PyExc_ValueError,
                 graph_def_or.status().error_message().c_str());
    return nullptr;
  }

  std::string ret_str = graph_def_or.ValueOrDie().SerializeAsString();

  return tflite::python_utils::ConvertToPyString(ret_str.c_str(),
                                                 ret_str.size());
}

PyObject* QuantizePTQModelPostCalibration(absl::string_view saved_model_path,
                                          absl::string_view exported_names_str,
                                          absl::string_view tags) {
  auto graph_def_or = mlir::quant::QuantizePTQModelPostCalibration(
      saved_model_path, exported_names_str, tags);
  if (!graph_def_or.ok()) {
    PyErr_Format(PyExc_ValueError,
                 graph_def_or.status().error_message().c_str());
    return nullptr;
  }

  std::string ret_str = graph_def_or.ValueOrDie().SerializeAsString();

  return tflite::python_utils::ConvertToPyString(ret_str.c_str(),
                                                 ret_str.size());
}

py::tuple GetMinMaxFromCalibrator(absl::string_view id) {
  absl::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax(id);
  if (!min_max.has_value()) {
    PyErr_Format(PyExc_ValueError, "No calibrated data for '%s'",
                 std::string{id}.c_str());
    throw py::error_already_set();
  }

  return py::make_tuple(min_max->first, min_max->second);
}

PYBIND11_MODULE(quantize_model_wrapper, m) {
  m.def(
      "clear_calibrator",
      []() { CalibratorSingleton::ClearCollectedInformation(); },
      R"pbdoc(
      Clears the collected metrics from the calibrator.
    )pbdoc");
  m.def(
      "clear_data_from_calibrator",
      [](absl::string_view id) { CalibratorSingleton::ClearData(id); },
      R"pbdoc(
      Clears the collected data of the given id from calibrator.
    )pbdoc");
  m.def(
      "get_min_max_from_calibrator",
      [](absl::string_view id) { return GetMinMaxFromCalibrator(id); },
      R"pbdoc(
      Return the tuple with the min and max values of the given id.
    )pbdoc");
  m.def(
      "quantize_qat_model",
      [](absl::string_view saved_model_path,
         absl::string_view exported_names_str, absl::string_view tags) {
        return tensorflow::PyoOrThrow(
            QuantizeQATModel(saved_model_path, exported_names_str, tags));
      },
      R"pbdoc(
      Returns a tf model graph def string.
    )pbdoc");
  m.def(
      "quantize_ptq_model_pre_calibration",
      [](absl::string_view saved_model_path,
         absl::string_view exported_names_str, absl::string_view tags) {
        return tensorflow::PyoOrThrow(QuantizePTQModelPreCalibration(
            saved_model_path, exported_names_str, tags));
      },
      R"pbdoc(
      Returns a tf model graph def string.
    )pbdoc");
  m.def(
      "quantize_ptq_model_post_calibration",
      [](absl::string_view saved_model_path,
         absl::string_view exported_names_str, absl::string_view tags) {
        return tensorflow::PyoOrThrow(QuantizePTQModelPostCalibration(
            saved_model_path, exported_names_str, tags));
      },
      R"pbdoc(
      Returns a tf model graph def string.
    )pbdoc");
}
