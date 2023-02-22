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
#include <cstring>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace {

using ::tensorflow::calibrator::CalibratorSingleton;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::QuantizationOptions;
using ::tensorflow::quantization::QuantizePtqDynamicRange;
using ::tensorflow::quantization::QuantizePtqModelPostCalibration;
using ::tensorflow::quantization::QuantizePtqModelPreCalibration;
using ::tensorflow::quantization::QuantizeQatModel;

// Serializes an ExportedModel. Raises python ValueError if serialization fails.
std::string Serialize(const ExportedModel& exported_model) {
  const std::string exported_model_serialized =
      exported_model.SerializeAsString();

  // Empty string means it failed to serialize the protobuf with an error. See
  // the docstring for SerializeAsString for details.
  if (exported_model_serialized.empty()) {
    throw py::value_error("Failed to serialize ExportedModel.");
  }

  return exported_model_serialized;
}

// Retrieves collected min / max values of a `CustomAggregator` node from the
// singleton. `id` is the identifier of the `CustomAggregator`.
std::pair<float, float> GetCalibratorMinMax(const absl::string_view id) {
  std::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax(id);
  if (min_max == std::nullopt) {
    throw py::value_error(
        absl::StrFormat("Calibrated data does not exist. Cannot find min/max "
                        "value for id: '%s'",
                        id));
  }

  return *min_max;
}

}  // namespace

namespace pybind11 {
namespace detail {

// Converts `ExportedModel` (c++) to `bytes` (python). The resulting `bytes`
// object is a serialization of `ExportedModel`.
//
// See https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html for
// further details on how custom type conversions work for pybind11.
template <>
struct type_caster<ExportedModel> {
 public:
  PYBIND11_TYPE_CASTER(ExportedModel, const_name("ExportedModel"));

  // Constructs a `bytes` object after serializing `src`.
  static handle cast(ExportedModel&& src, return_value_policy policy,
                     handle parent) {
    // release() prevents the reference count from decreasing upon the
    // destruction of py::bytes and returns a raw python object handle.
    return py::bytes(Serialize(src)).release();
  }
};

// Python -> cpp conversion for `QuantizationOptions`. Accepts a serialized
// protobuf string and deserializes into an instance of `QuantizationOptions`.
template <>
struct type_caster<QuantizationOptions> {
 public:
  PYBIND11_TYPE_CASTER(QuantizationOptions, const_name("QuantizationOptions"));

  bool load(handle src, const bool convert) {
    auto caster = make_caster<absl::string_view>();
    // The user should have passed a valid python string.
    if (!caster.load(src, convert)) {
      return false;
    }

    const absl::string_view quantization_opts_serialized =
        cast_op<absl::string_view>(std::move(caster));

    // NOLINTNEXTLINE: Explicit std::string conversion required for OSS.
    return value.ParseFromString(std::string(quantization_opts_serialized));
  }
};

}  // namespace detail
}  // namespace pybind11

PYBIND11_MODULE(pywrap_quantize_model, m) {
  // Supports absl::StatusOr<T> type conversions.
  pybind11::google::ImportStatusModule();

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
      "get_min_from_calibrator",
      [](const absl::string_view id) -> float {
        const std::pair<float, float> min_max = GetCalibratorMinMax(id);
        return min_max.first;
      },
      R"pbdoc(
      Return the tuple with the min value of the given id.
    )pbdoc");
  m.def(
      "get_max_from_calibrator",
      [](const absl::string_view id) -> float {
        const std::pair<float, float> min_max = GetCalibratorMinMax(id);
        return min_max.second;
      },
      R"pbdoc(
      Return the tuple with the min value of the given id.
    )pbdoc");

  // Quantization functions.
  m.def(
      "quantize_qat_model",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const QuantizationOptions& quant_opts)
          -> absl::StatusOr<ExportedModel> {
        return QuantizeQatModel(saved_model_path, signature_keys, tags,
                                quant_opts);
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
         const QuantizationOptions& quant_opts)
          -> absl::StatusOr<ExportedModel> {
        return QuantizePtqDynamicRange(saved_model_path, signature_keys, tags,
                                       quant_opts);
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
         const absl::flat_hash_map<std::string, std::string>& function_aliases)
          -> absl::StatusOr<ExportedModel> {
        return QuantizePtqModelPreCalibration(saved_model_path, signature_keys,
                                              tags, quant_opts,
                                              function_aliases);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the model's GraphDef and
      metadata. The GraphDef contains extra ops required for calibration. The
      user should pass a serialized `QuantizationOptions` for the `quant_opts`
      argument.

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
