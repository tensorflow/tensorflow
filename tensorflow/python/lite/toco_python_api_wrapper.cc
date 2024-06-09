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

#include <string>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/lite/toco/python/toco_python_api.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_toco_api, m) {
  m.def(
      "TocoConvert",
      [](py::object model_flags_proto_txt_raw,
         py::object toco_flags_proto_txt_raw, py::object input_contents_txt_raw,
         bool extended_return, py::object debug_info_txt_raw,
         bool enable_mlir_converter,
         const tensorflow::quantization::PyFunctionLibrary*
             quantization_py_function_library) {
        return tensorflow::PyoOrThrow(toco::TocoConvert(
            model_flags_proto_txt_raw.ptr(), toco_flags_proto_txt_raw.ptr(),
            input_contents_txt_raw.ptr(), extended_return,
            debug_info_txt_raw.ptr(), enable_mlir_converter,
            quantization_py_function_library));
      },
      py::arg("model_flags_proto_txt_raw"), py::arg("toco_flags_proto_txt_raw"),
      py::arg("input_contents_txt_raw"), py::arg("extended_return") = false,
      py::arg("debug_info_txt_raw") = py::none(),
      py::arg("enable_mlir_converter") = false,
      py::arg("quantization_py_function_library") = py::none(),
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
      "ExperimentalMlirQuantizeModel",
      [](py::object input_contents_txt_raw, bool disable_per_channel,
         bool fully_quantize, int inference_type, int input_data_type,
         int output_data_type, bool enable_numeric_verify,
         bool enable_whole_model_verify, py::object op_blocklist,
         py::object node_blocklist, bool enable_variable_quantization,
         bool disable_per_channel_for_dense_layers,
         py::object debug_options_proto_txt_raw) {
        return tensorflow::PyoOrThrow(toco::MlirQuantizeModel(
            input_contents_txt_raw.ptr(), disable_per_channel, fully_quantize,
            inference_type, input_data_type, output_data_type,
            enable_numeric_verify, enable_whole_model_verify,
            op_blocklist.ptr(), node_blocklist.ptr(),
            enable_variable_quantization, disable_per_channel_for_dense_layers,
            debug_options_proto_txt_raw.ptr()));
      },
      py::arg("input_contents_txt_raw"), py::arg("disable_per_channel") = false,
      py::arg("fully_quantize") = true, py::arg("inference_type") = 9,
      py::arg("input_data_type") = 0, py::arg("output_data_type") = 0,
      py::arg("enable_numeric_verify") = false,
      py::arg("enable_whole_model_verify") = false,
      py::arg("op_blocklist") = py::none(),
      py::arg("node_blocklist") = py::none(),
      py::arg("enable_variable_quantization") = false,
      py::arg("disable_per_channel_for_dense_layers") = false,
      py::arg("debug_options_proto_txt_raw") = nullptr,
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
  m.def(
      "RegisterCustomOpdefs",
      [](py::object custom_opdefs_txt_raw) {
        return tensorflow::PyoOrThrow(
            toco::RegisterCustomOpdefs(custom_opdefs_txt_raw.ptr()));
      },
      py::arg("custom_opdefs_txt_raw"),
      R"pbdoc(
      Registers the given custom opdefs to the TensorFlow global op registry.
    )pbdoc");
  m.def(
      "RetrieveCollectedErrors",
      []() {
        std::vector<std::string> collected_errors =
            toco::RetrieveCollectedErrors();
        pybind11::list serialized_message_list(collected_errors.size());
        int i = 0;
        for (const auto& error_data : collected_errors) {
          serialized_message_list[i++] = pybind11::bytes(error_data);
        }
        return serialized_message_list;
      },
      R"pbdoc(
      Returns and clears the list of collected errors in ErrorCollector.
    )pbdoc");
  m.def(
      "FlatBufferToMlir",
      [](const std::string& model, bool input_is_filepath) {
        return toco::FlatBufferFileToMlir(model, input_is_filepath);
      },
      R"pbdoc(
      Returns MLIR dump of the given TFLite model.
    )pbdoc");
}
