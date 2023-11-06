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
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11  // IWYU pragma: keep
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil   // IWYU pragma: keep
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/type_casters.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace {

using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibratorSingleton;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::QuantizationOptions;
using ::tensorflow::quantization::QuantizePtqDynamicRange;
using ::tensorflow::quantization::QuantizePtqModelPostCalibration;
using ::tensorflow::quantization::QuantizePtqModelPreCalibration;
using ::tensorflow::quantization::QuantizeQatModel;
using ::tensorflow::quantization::QuantizeWeightOnly;

// Retrieves collected statistics of a `CustomAggregator` node from the
// singleton. `id` is the identifier of the `CustomAggregator`.
CalibrationStatistics GetStatisticsFromCalibrator(const absl::string_view id) {
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(id);

  if (!statistics.has_value()) {
    throw py::value_error(absl::StrFormat(
        "Calibrated data does not exist. Cannot find statistics."
        "value for id: '%s'",
        id));
  }

  return *statistics;
}

}  // namespace

PYBIND11_MODULE(pywrap_quantize_model, m) {
  // Supports absl::StatusOr<T> type conversions.
  pybind11::google::ImportStatusModule();
  // TODO - b/308532051: Make protobuf objects work without serialization
  // overhead.
  pybind11_protobuf::ImportNativeProtoCasters();

  // Calibrator related functions.
  m.def(
      "clear_calibrator",
      [] { CalibratorSingleton::ClearCollectedInformation(); },
      R"pbdoc(
      Clears the collected metrics from the calibrator.
    )pbdoc");
  m.def(
      "clear_data_from_calibrator",
      [](const absl::string_view id) { CalibratorSingleton::ClearData(id); },
      R"pbdoc(
      Clears the collected data of the given id from calibrator.
    )pbdoc");
  m.def(
      "get_statistics_from_calibrator",
      [](const absl::string_view id) -> CalibrationStatistics {
        return GetStatisticsFromCalibrator(id);
      },
      R"pbdoc(
      Returns the proto CalibrationStatistics given id from calibrator.
    )pbdoc");

  // Quantization functions.
  m.def(
      "quantize_qat_model",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const QuantizationOptions& quant_opts,
         const absl::flat_hash_map<std::string, std::string>& function_aliases)
          -> absl::StatusOr<ExportedModel> {
        return QuantizeQatModel(saved_model_path, signature_keys, tags,
                                quant_opts, function_aliases);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata. The user should pass a serialized
      `QuantizationOptions` for the `quant_opts` argument.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_ptq_dynamic_range",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const QuantizationOptions& quant_opts,
         const absl::flat_hash_map<std::string, std::string>& function_aliases)
          -> absl::StatusOr<ExportedModel> {
        return QuantizePtqDynamicRange(saved_model_path, signature_keys, tags,
                                       quant_opts, function_aliases);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata. The user should pass a serialized
      `QuantizationOptions` for the `quant_opts` argument.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_weight_only",
      [](const absl::string_view saved_model_path,
         const QuantizationOptions& quant_opts,
         const absl::flat_hash_map<std::string, std::string>& function_aliases)
          -> absl::StatusOr<ExportedModel> {
        return QuantizeWeightOnly(saved_model_path, quant_opts,
                                  function_aliases);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata. The user should pass a serialized
      `QuantizationOptions` for the `quant_opts` argument.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_ptq_model_pre_calibration",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const QuantizationOptions& quant_opts,
         const absl::flat_hash_map<std::string, std::string>& function_aliases,
         const PyFunctionLibrary& py_function_lib)
          -> absl::StatusOr<ExportedModel> {
        const absl::StatusOr<ExportedModel> exported_model =
            QuantizePtqModelPreCalibration(saved_model_path, signature_keys,
                                           tags, quant_opts, function_aliases);
        if (!exported_model.ok()) {
          return exported_model.status();
        }

        return py_function_lib.AssignIdsToCustomAggregatorOps(*exported_model);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the model's GraphDef and
      metadata. The GraphDef contains extra ops required for calibration. The
      user should pass a serialized `QuantizationOptions` for the `quant_opts`
      argument.

      The argument `custom_aggregator_id_assigner` is an instance of
      `CustomAggregatorIdAssigner` whose virtual function `assign_ids` is
      implemented in python.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_ptq_model_post_calibration",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const QuantizationOptions& quant_opts,
         const absl::flat_hash_map<std::string, std::string>& function_aliases)
          -> absl::StatusOr<ExportedModel> {
        return QuantizePtqModelPostCalibration(saved_model_path, signature_keys,
                                               tags, quant_opts,
                                               function_aliases);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata. The user should pass a serialized
      `QuantizationOptions` for the `quant_opts` argument.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");
}
