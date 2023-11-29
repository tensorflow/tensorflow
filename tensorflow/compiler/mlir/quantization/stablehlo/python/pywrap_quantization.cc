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
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11  // IWYU pragma: keep
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil   // IWYU pragma: keep
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/graph_def.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/type_casters.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tsl/platform/env.h"

namespace py = pybind11;

namespace {

using ::stablehlo::quantization::MutateNodeDefs;
using ::stablehlo::quantization::io::CreateTmpDir;
using ::tensorflow::FunctionDef;
using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::DebuggerOptions;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::QuantizationOptions;

// TODO: b/312371048 - Factor out this function to a separate file.
// Enables debugging on `exported_model` by updating the `DumpTensor` ops.
//
// Saves the current model to `debugger_options.unquantized_dump_model_path()`
// if the debugger type is `DEBUGGER_TYPE_WHOLE_MODEL`. This is required because
// in whole-model debugging mode the `DumpTensor` ops for the unquantized
// tensors are only inserted in the unquantized model whereas `DumpTensor` ops
// for the quantized tensors are only inserted in the quantized model. Both
// models are required to be able to dump both quantized and unquantized tensors
// and compare them offline.
ExportedModel EnableDebugging(
    const ExportedModel& exported_model,
    const DebuggerOptions& debugger_options,
    const PyFunctionLibrary& py_function_library,
    const absl::string_view src_saved_model_path,
    const std::unordered_set<std::string>& tags,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map) {
  ExportedModel debugger_enabled_exported_model = exported_model;

  // Enable `DumpTensor` nodes in `graph_def`. DumpTensor is disabled by
  // default to avoid logging data during calibration.
  MutateNodeDefs(*debugger_enabled_exported_model.mutable_graph_def(),
                 [](NodeDef& node_def) {
                   if (node_def.op() == "DumpTensor") {
                     (*node_def.mutable_attr())["enabled"].set_b(true);
                   }
                 });

  if (debugger_options.debugger_type() ==
      DebuggerOptions::DEBUGGER_TYPE_WHOLE_MODEL) {
    // TODO: b/295139417 - Remove CustomAggregator op in unquantized dump model.
    // TODO: b/296916287 - Create a separate function for saving unquantized
    // dump model.
    py_function_library.SaveExportedModel(
        debugger_options.unquantized_dump_model_path(),
        debugger_enabled_exported_model, src_saved_model_path, tags,
        signature_def_map);

    // Update the `DumpTensor` ops' file name in `graph_def`.
    MutateNodeDefs(*debugger_enabled_exported_model.mutable_graph_def(),
                   [](NodeDef& node_def) {
                     if (node_def.op() == "DumpTensor") {
                       (*node_def.mutable_attr())["file_name"].set_s(
                           "quantized_tensor_data.pb");
                     }
                   });
  }

  return debugger_enabled_exported_model;
}

}  // namespace

PYBIND11_MODULE(pywrap_quantization, m) {
  // Supports absl::Status type conversions.
  pybind11::google::ImportStatusModule();

  m.doc() = "StableHLO Quantization APIs.";

  m.def(
      // If the function signature changes, likely its corresponding .pyi type
      // hinting should also change.
      // LINT.IfChange
      "static_range_ptq",
      [](const absl::string_view src_saved_model_path,
         const absl::string_view dst_saved_model_path,
         const QuantizationOptions& quantization_options,
         const std::vector<std::string>& signature_keys,
         const absl::flat_hash_map<std::string, SignatureDef>&
             signature_def_map,
         const absl::flat_hash_map<std::string, std::string>& function_aliases,
         const PyFunctionLibrary& py_function_library,
         py::object representative_dataset) -> absl::Status {
        // LINT.ThenChange(pywrap_quantization.pyi:static_range_ptq)
        std::unordered_set<std::string> tags;
        tags.insert(quantization_options.tags().begin(),
                    quantization_options.tags().end());

        const absl::StatusOr<ExportedModel> exported_model =
            QuantizePtqModelPreCalibration(src_saved_model_path, signature_keys,
                                           tags, quantization_options,
                                           function_aliases);
        if (!exported_model.ok()) return exported_model.status();

        const ExportedModel exported_model_ids_assigned =
            py_function_library.AssignIdsToCustomAggregatorOps(*exported_model);

        const absl::StatusOr<std::string> precalibrated_saved_model_dir =
            CreateTmpDir();
        if (!precalibrated_saved_model_dir.ok()) {
          throw py::value_error(absl::StrFormat(
              "Failed to create tmp dir for precalibrated saved model: %s",
              precalibrated_saved_model_dir.status().ToString()));
        }

        py_function_library.SaveExportedModel(
            *precalibrated_saved_model_dir, exported_model_ids_assigned,
            src_saved_model_path, tags, signature_def_map);

        ExportedModel calibrated_exported_model =
            py_function_library.RunCalibration(
                *precalibrated_saved_model_dir, signature_keys, tags,
                exported_model_ids_assigned,
                quantization_options.calibration_options(),
                quantization_options.force_graph_mode_calibration(),
                representative_dataset);

        if (quantization_options.has_debugger_options()) {
          calibrated_exported_model = EnableDebugging(
              calibrated_exported_model,
              quantization_options.debugger_options(), py_function_library,
              src_saved_model_path, tags, signature_def_map);
        }

        const absl::StatusOr<std::string> calibrated_saved_model_path =
            CreateTmpDir();
        if (!calibrated_saved_model_path.ok()) {
          throw py::value_error(absl::StrFormat(
              "Failed to create tmp dir for calibrated saved model: %s",
              calibrated_saved_model_path.status().ToString()));
        }

        py_function_library.SaveExportedModel(
            *calibrated_saved_model_path, calibrated_exported_model,
            src_saved_model_path, tags, signature_def_map);

        const absl::flat_hash_map<std::string, std::string>
            function_aliases_after_calibration(
                calibrated_exported_model.function_aliases().begin(),
                calibrated_exported_model.function_aliases().end());

        const absl::StatusOr<ExportedModel> post_calibrated_exported_model =
            QuantizePtqModelPostCalibration(
                *calibrated_saved_model_path, signature_keys, tags,
                quantization_options, function_aliases_after_calibration);
        if (!post_calibrated_exported_model.ok()) {
          return post_calibrated_exported_model.status();
        }

        py_function_library.SaveExportedModel(
            dst_saved_model_path, *post_calibrated_exported_model,
            *calibrated_saved_model_path, tags, signature_def_map);

        return absl::OkStatus();
      },
      R"pbdoc(
      Runs static-range post-training quantization (PTQ) on a SavedModel at
      `src_saved_model_path` and saves the resulting model to
      `dst_saved_model_path`.

      The user should pass a serialized `QuantizationOptions` for the
      `quantization_options_serialized` argument, and a signature key ->
      serialized `SignatureDef` mapping for the `signature_def_map_serialized`
      argument.

      `function_aliases` maps actual function names to the function aliases, as
      defined by the `MetaGraphDef::MetaInfoDef::function_aliases` from the
      input SavedModel.

      Raises `StatusNotOk` exception if when the run was unsuccessful.
      )pbdoc",
      py::arg("saved_model_path"), py::arg("dst_saved_model_path"),
      py::arg("quantization_options_serialized"), py::kw_only(),
      py::arg("signature_keys"), py::arg("signature_def_map_serialized"),
      py::arg("function_aliases"), py::arg("py_function_library"),
      py::arg("representative_dataset"));
}
