/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <Python.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil    // IWYU pragma: keep
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow_to_stablehlo/python/pywrap_tensorflow_to_stablehlo_lib.h"

namespace py = pybind11;

namespace {

using mlir::tensorflow_to_stablehlo::pywrap::PywrapSavedModelToStablehlo;
using mlir::tensorflow_to_stablehlo::pywrap::PywrapTfModuleToStablehlo;

}  // namespace

PYBIND11_MODULE(pywrap_tensorflow_to_stablehlo, m) {
  m.doc() = "TensorFlow to StableHLO APIs.";

  // LINT.IfChange(savedmodel_to_stablehlo)
  m.def(
      "savedmodel_to_stablehlo",
      [](absl::string_view input_path,
         const std::vector<std::string>& exported_model_signatures =
             {"serving_default"},
         const std::vector<std::string>& tag_names = {"serve"},
         absl::string_view input_arg_shapes_str = "") -> py::bytes {
        auto module_bytecode =
            PywrapSavedModelToStablehlo(input_path, exported_model_signatures,
                                        tag_names, input_arg_shapes_str);
        if (!module_bytecode.ok()) {
          PyErr_SetString(PyExc_ValueError,
                          module_bytecode.status().ToString().c_str());
          throw py::error_already_set();
        }
        return py::bytes(module_bytecode.value());
      },
      R"pbdoc(
        Converts a TensorFlow SavedModel into StableHLO bytecode.

        * input-path: The path to the input TensorFlow SavedModel.
        * exported-model-signatures: Comma-separated list of exported model
          signatures to convert.
        * tag_names: Comma-separated list of tags for loading SavedModel.
        * input-arg-shapes: A string representation of input argument shapes for
          'main' entry-point, separating tensors with ':', dimension with ',', and
          using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?'
          expresses argument shapes [1,2], [] and [1,?].
        )pbdoc",
      py::arg("input_path"),
      py::arg("exported_model_signatures") =
          std::vector<std::string>{"serving_default"},
      py::arg("tag_names") = std::vector<std::string>{"serve"},
      py::arg("input_arg_shapes_str") = "");
  // LINT.ThenChange(pywrap_tensorflow_to_stablehlo.pyi:savedmodel_to_stablehlo)
  //
  // LINT.IfChange(tensorflow_module_to_stablehlo)
  m.def(
      "tensorflow_module_to_stablehlo",
      [](absl::string_view module_op_str,
         absl::string_view input_arg_shapes_str) -> py::bytes {
        auto module_bytecode =
            PywrapTfModuleToStablehlo(module_op_str, input_arg_shapes_str);
        if (!module_bytecode.ok()) {
          PyErr_SetString(PyExc_ValueError,
                          module_bytecode.status().ToString().c_str());
          throw py::error_already_set();
        }
        return py::bytes(module_bytecode.value());
      },
      R"pbdoc(
        Converts a TensorFlow MLIR module string into StableHLO bytecode.

        * module: TensorFlow MLIR module string.
        * input-arg-shapes: A string representation of input argument shapes for
          'main' entry-point, separating tensors with ':', dimension with ',', and
          using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?'
          expresses argument shapes [1,2], [] and [1,?].
        )pbdoc",
      py::arg("module"), py::arg("input_arg_shapes_str") = "");
  // LINT.ThenChange(pywrap_tensorflow_to_stablehlo.pyi:tensorflow_module_to_stablehlo)
}
