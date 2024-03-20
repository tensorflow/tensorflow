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
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11  // IWYU pragma: keep
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil   // IWYU pragma: keep
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/python/pywrap_quantization_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/type_casters.h"  // IWYU pragma: keep

namespace py = pybind11;

namespace {

using ::stablehlo::quantization::pywrap::PywrapExpandPresets;
using ::stablehlo::quantization::pywrap::PywrapPopulateDefaults;
using ::stablehlo::quantization::pywrap::PywrapQuantizeStaticRangePtq;
using ::stablehlo::quantization::pywrap::PywrapQuantizeWeightOnlyPtq;

}  // namespace

PYBIND11_MODULE(pywrap_quantization, m) {
  // Supports absl::Status type conversions.
  pybind11::google::ImportStatusModule();

  m.doc() = "StableHLO Quantization APIs.";

  // If the function signature changes, likely its corresponding .pyi type
  // hinting should also change.
  // LINT.IfChange(static_range_ptq)
  m.def("static_range_ptq", &PywrapQuantizeStaticRangePtq,
        R"pbdoc(
        Runs static-range post-training quantization (PTQ) on a SavedModel at
        `src_saved_model_path` and saves the resulting model to
        `dst_saved_model_path`.

        The user should pass a serialized `QuantizationConfig` for the
        `quantization_config_serialized` argument, and a signature key ->
        serialized `SignatureDef` mapping for the `signature_def_map_serialized`
        argument.

        Raises `StatusNotOk` exception if when the run was unsuccessful.
        )pbdoc",
        py::arg("src_saved_model_path"), py::arg("dst_saved_model_path"),
        py::arg("quantization_config_serialized"), py::kw_only(),
        py::arg("signature_keys"), py::arg("signature_def_map_serialized"),
        py::arg("py_function_library"));
  // LINT.ThenChange(pywrap_quantization.pyi:static_range_ptq)

  // If the function signature changes, likely its corresponding .pyi type
  // hinting should also change.
  // LINT.IfChange(weight_only_ptq)
  m.def("weight_only_ptq", &PywrapQuantizeWeightOnlyPtq,
        R"pbdoc(
        Runs weight-only Quantization on a SavedModel at `src_saved_model_path`
        and saves the resulting model to `dst_saved_model_path`.

        The user should pass a serialized `QuantizationConfig` for the
        `quantization_config_serialized` argument, and a signature key ->
        serialized `SignatureDef` mapping for the `signature_def_map_serialized`
        argument.

        Raises `StatusNotOk` exception if when the run was unsuccessful.
        )pbdoc",
        py::arg("src_saved_model_path"), py::arg("dst_saved_model_path"),
        py::arg("quantization_config_serialized"), py::kw_only(),
        py::arg("signature_keys"), py::arg("signature_def_map_serialized"),
        py::arg("py_function_library"));
  // LINT.ThenChange(pywrap_quantization.pyi:weight_only_ptq)

  // If the function signature changes, likely its corresponding .pyi type
  // hinting should also change.
  // LINT.IfChange(populate_default_configs)
  m.def("populate_default_configs", &PywrapPopulateDefaults,
        R"pbdoc(
        Populates `QuantizationConfig` with default values.

        Returns an updated `QuantizationConfig` (serialized) after populating
        default values to fields that the user did not explicitly specify.
        )pbdoc",
        py::arg("user_provided_config_serialized"));
  // LINT.ThenChange(pywrap_quantization.pyi:populate_default_configs)

  // If the function signature changes, likely its corresponding .pyi type
  // hinting should also change.
  // LINT.IfChange(expand_preset_configs)
  m.def("expand_preset_configs", &PywrapExpandPresets, R"pbdoc(
        Expands presets to other fields in `QuantizationConfig`.

        Each preset is expressible by other fields in `QuantizationConfig`.
        Returns a copy of `QuantizationConfig` (serialized) where the fields are
        expanded from presets. If no preset has been set, it is a no-op and
        returns the same copy of the input.
        )pbdoc",
        py::arg("quantization_config_serialized"));
  // LINT.ThenChange(pywrap_quantization.pyi:expand_preset_configs)
}
