/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/python/toco_python_api.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_toco_api, m) {
  m.def(
      "TocoConvert",
      [](py::object model_flags_proto_txt_raw,
         py::object toco_flags_proto_txt_raw, py::object input_contents_txt_raw,
         bool extended_return, py::object debug_info_txt_raw,
         bool enable_mlir_converter) {
        return tensorflow::PyoOrThrow(toco::TocoConvert(
            model_flags_proto_txt_raw.ptr(), toco_flags_proto_txt_raw.ptr(),
            input_contents_txt_raw.ptr(), extended_return,
            debug_info_txt_raw.ptr(), enable_mlir_converter));
      },
      py::arg("model_flags_proto_txt_raw"), py::arg("toco_flags_proto_txt_raw"),
      py::arg("input_contents_txt_raw"), py::arg("extended_return") = false,
      py::arg("debug_info_txt_raw") = py::none(),
      py::arg("enable_mlir_converter") = false,
      R"pbdoc(
      Convert a model represented in `input_contents`. `model_flags_proto`
      describes model parameters. `toco_flags_proto` describes conversion
      parameters (see relevant .protos for more information). Returns a string
      representing the contents of the converted model. When extended_return
      flag is set to true returns a dictionary that contains string representation
      of the converted model and some statistics like arithmetic ops count.
      `debug_info_str` contains the `GraphDebugInfo` proto. When
      `enable_mlir_converter` is True, tuse MLIR-based conversion instead of
      TOCO conversion.
    )pbdoc");
  m.def(
      "TocoGetPotentiallySupportedOps",
      []() {
        return tensorflow::PyoOrThrow(toco::TocoGetPotentiallySupportedOps());
      },
      R"pbdoc(
      Returns a list of names of all ops potentially supported by tflite.
    )pbdoc");
  m.def(
      "ExperimentalMlirQuantizeModel",
      [](py::object input_contents_txt_raw, bool disable_per_channel,
         bool fully_quantize) {
        return tensorflow::PyoOrThrow(toco::MlirQuantizeModel(
            input_contents_txt_raw.ptr(), disable_per_channel, fully_quantize));
      },
      py::arg("input_contents_txt_raw"), py::arg("disable_per_channel") = false,
      py::arg("fully_quantize") = true,
      R"pbdoc(
      Returns a quantized model.
    )pbdoc");
  m.def(
      "ExperimentalMlirSparsifyModel",
      [](py::object input_contents_txt_raw) {
        return tensorflow::PyoOrThrow(
            toco::MlirSparsifyModel(input_contents_txt_raw.ptr()));
      },
      py::arg("input_contents_txt_raw"),
      R"pbdoc(
      Returns a sparsified model.
    )pbdoc");
}
