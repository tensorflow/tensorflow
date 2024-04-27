/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/min_max_value.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/type_casters.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace py = ::pybind11;

namespace {

using ::stablehlo::quantization::CalibrationOptions;
using ::stablehlo::quantization::MinMaxValue;
using ::tensorflow::SignatureDef;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::RepresentativeDatasetFile;

// A "trampoline" class that redirects virtual function calls to the python
// implementation.
//
// Reference:
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
class PyFunctionLibraryTrampoline : public PyFunctionLibrary {
 public:
  using PyFunctionLibrary::PyFunctionLibrary;

  std::optional<bool> SaveExportedModel(
      const absl::string_view dst_saved_model_path,
      const ExportedModel& exported_model,
      const absl::string_view src_saved_model_path,
      const std::unordered_set<std::string>& tags,
      const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map)
      const override {
    PYBIND11_OVERRIDE_PURE(std::optional<bool>, PyFunctionLibrary,
                           save_exported_model, dst_saved_model_path,
                           exported_model, src_saved_model_path, tags,
                           signature_def_map);
  }

  std::optional<bool> RunCalibration(
      const absl::string_view saved_model_path,
      const std::vector<std::string>& signature_keys,
      const std::unordered_set<std::string>& tags,
      const bool force_graph_mode_calibration,
      const absl::flat_hash_map<std::string, RepresentativeDatasetFile>&
          representative_dataset_file_map) const override {
    PYBIND11_OVERRIDE_PURE(std::optional<bool>, PyFunctionLibrary,
                           run_calibration, saved_model_path, signature_keys,
                           tags, force_graph_mode_calibration,
                           representative_dataset_file_map);
  }

  std::optional<MinMaxValue> GetCalibrationMinMaxValue(
      const CalibrationStatistics& calibration_statistics,
      const CalibrationOptions& calibration_options) const override {
    PYBIND11_OVERRIDE_PURE(std::optional<MinMaxValue>, PyFunctionLibrary,
                           get_calibration_min_max_value,
                           calibration_statistics, calibration_options);
  }
};

}  // namespace

PYBIND11_MODULE(pywrap_function_lib, m) {
  py::class_<PyFunctionLibrary, PyFunctionLibraryTrampoline>(
      m, "PyFunctionLibrary")
      .def(py::init<>())
      .def("save_exported_model", &PyFunctionLibrary::SaveExportedModel,
           py::arg("dst_saved_model_path"),
           py::arg("exported_model_serialized"),
           py::arg("src_saved_model_path"), py::arg("tags"),
           py::arg("serialized_signature_def_map"))
      .def("run_calibration", &PyFunctionLibrary::RunCalibration,
           py::arg("saved_model_path"), py::arg("signature_keys"),
           py::arg("tags"), py::arg("force_graph_mode_calibration"),
           py::arg("representative_dataset_file_map_serialized"))
      .def("get_calibration_min_max_value",
           &PyFunctionLibrary::GetCalibrationMinMaxValue,
           py::arg("calibration_statistics_serialized"),
           py::arg("calibration_options_serialized"));
}
