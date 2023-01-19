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
#include <string>
#include <unordered_set>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace {

using ::tensorflow::quantization::ExportedModel;

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

}  // namespace detail
}  // namespace pybind11

PYBIND11_MODULE(pywrap_quantize_model, m) {
  // Supports absl::StatusOr<T> type conversions.
  pybind11::google::ImportStatusModule();

  // Calibrator related functions.
  m.def(
      "clear_calibrator",
      [] {
        tensorflow::quantization::ClearCollectedInformationFromCalibrator();
      },
      R"pbdoc(
      Clears the collected metrics from the calibrator.
    )pbdoc");
  m.def(
      "clear_data_from_calibrator",
      [](const absl::string_view id) {
        tensorflow::quantization::ClearDataFromCalibrator(id);
      },
      R"pbdoc(
      Clears the collected data of the given id from calibrator.
    )pbdoc");
  m.def(
      "get_max_from_calibrator",
      [](const absl::string_view id) -> float {
        return tensorflow::quantization::GetMaxFromCalibrator(id);
      },
      R"pbdoc(
      Return the tuple with the min value of the given id.
    )pbdoc");
  m.def(
      "get_min_from_calibrator",
      [](const absl::string_view id) -> float {
        return tensorflow::quantization::GetMinFromCalibrator(id);
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
         const absl::string_view quant_opts_serialized)
          -> absl::StatusOr<ExportedModel> {
        return tensorflow::quantization::internal::QuantizeQatModel(
            saved_model_path, signature_keys, tags, quant_opts_serialized);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_ptq_dynamic_range",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const absl::string_view quant_opts_serialized)
          -> absl::StatusOr<ExportedModel> {
        return tensorflow::quantization::internal::QuantizePtqDynamicRange(
            saved_model_path, signature_keys, tags, quant_opts_serialized);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_ptq_model_pre_calibration",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const absl::string_view quant_opts_serialized)
          -> absl::StatusOr<ExportedModel> {
        return tensorflow::quantization::internal::
            QuantizePtqModelPreCalibration(saved_model_path, signature_keys,
                                           tags, quant_opts_serialized);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the model's GraphDef and
      metadata. The GraphDef contains extra ops required for calibration.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");

  m.def(
      "quantize_ptq_model_post_calibration",
      [](const absl::string_view saved_model_path,
         const std::vector<std::string>& signature_keys,
         const std::unordered_set<std::string>& tags,
         const absl::string_view quant_opts_serialized)
          -> absl::StatusOr<ExportedModel> {
        return tensorflow::quantization::internal::
            QuantizePtqModelPostCalibration(saved_model_path, signature_keys,
                                            tags, quant_opts_serialized);
      },
      R"pbdoc(
      Returns serialized ExportedModel that contains the quantized model's
      GraphDef and metadata.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
    )pbdoc");
}
