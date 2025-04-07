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

#include <Python.h>

#include <string>
#include <vector>

#include "nanobind/nanobind.h"  // from @nanobind
#include "tensorflow/compiler/mlir/lite/python/converter_python_api.h"

namespace nb = nanobind;

namespace {

inline nb::object PyoOrThrow(PyObject* ptr) {
  if (PyErr_Occurred() || ptr == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::object>(ptr);
}

}  // namespace

NB_MODULE(_pywrap_converter_api, m) {
  m.def(
      "Convert",
      [](nb::object model_flags_proto_txt_raw,
         nb::object converter_flags_proto_txt_raw,
         nb::object input_contents_txt_raw, bool extended_return,
         nb::object debug_info_txt_raw) {
        return PyoOrThrow(tflite::Convert(
            model_flags_proto_txt_raw.ptr(),
            converter_flags_proto_txt_raw.ptr(), input_contents_txt_raw.ptr(),
            extended_return, debug_info_txt_raw.ptr()));
      },
      nb::arg("model_flags_proto_txt_raw"),
      nb::arg("converter_flags_proto_txt_raw"),
      nb::arg("input_contents_txt_raw") = nb::none(),
      nb::arg("extended_return") = false,
      nb::arg("debug_info_txt_raw") = nb::none(),
      R"pbdoc(
      Convert a model represented in `input_contents`. `model_flags_proto`
      describes model parameters. `flags_proto` describes conversion
      parameters (see relevant .protos for more information). Returns a string
      representing the contents of the converted model. When extended_return
      flag is set to true returns a dictionary that contains string representation
      of the converted model and some statistics like arithmetic ops count.
      `debug_info_str` contains the `GraphDebugInfo` proto.
    )pbdoc");
  m.def(
      "ExperimentalMlirQuantizeModel",
      [](nb::object input_contents_txt_raw, bool disable_per_channel,
         bool fully_quantize, int inference_type, int input_data_type,
         int output_data_type, bool enable_numeric_verify,
         bool enable_whole_model_verify, nb::object op_blocklist,
         nb::object node_blocklist, bool enable_variable_quantization,
         bool disable_per_channel_for_dense_layers,
         nb::object debug_options_proto_txt_raw) {
        return PyoOrThrow(tflite::MlirQuantizeModel(
            input_contents_txt_raw.ptr(), disable_per_channel, fully_quantize,
            inference_type, input_data_type, output_data_type,
            enable_numeric_verify, enable_whole_model_verify,
            op_blocklist.ptr(), node_blocklist.ptr(),
            enable_variable_quantization, disable_per_channel_for_dense_layers,
            debug_options_proto_txt_raw.ptr()));
      },
      nb::arg("input_contents_txt_raw"), nb::arg("disable_per_channel") = false,
      nb::arg("fully_quantize") = true, nb::arg("inference_type") = 9,
      nb::arg("input_data_type") = 0, nb::arg("output_data_type") = 0,
      nb::arg("enable_numeric_verify") = false,
      nb::arg("enable_whole_model_verify") = false,
      nb::arg("op_blocklist") = nb::none(),
      nb::arg("node_blocklist") = nb::none(),
      nb::arg("enable_variable_quantization") = false,
      nb::arg("disable_per_channel_for_dense_layers") = false,
      nb::arg("debug_options_proto_txt_raw") = nullptr,
      R"pbdoc(
      Returns a quantized model.
    )pbdoc");
  m.def(
      "ExperimentalMlirSparsifyModel",
      [](nb::object input_contents_txt_raw) {
        return PyoOrThrow(
            tflite::MlirSparsifyModel(input_contents_txt_raw.ptr()));
      },
      nb::arg("input_contents_txt_raw"),
      R"pbdoc(
      Returns a sparsified model.
    )pbdoc");
  m.def(
      "RegisterCustomOpdefs",
      [](nb::object custom_opdefs_txt_raw) {
        return PyoOrThrow(
            tflite::RegisterCustomOpdefs(custom_opdefs_txt_raw.ptr()));
      },
      nb::arg("custom_opdefs_txt_raw"),
      R"pbdoc(
      Registers the given custom opdefs to the TensorFlow global op registry.
    )pbdoc");
  m.def(
      "RetrieveCollectedErrors",
      []() {
        std::vector<std::string> collected_errors =
            tflite::RetrieveCollectedErrors();
        nb::list serialized_message_list;
        int i = 0;
        for (const auto& error_data : collected_errors) {
          serialized_message_list.append(
              nb::bytes(error_data.data(), error_data.size()));
        }
        return serialized_message_list;
      },
      R"pbdoc(
      Returns and clears the list of collected errors in ErrorCollector.
    )pbdoc");
  m.def(
      "FlatBufferToMlir",
      [](const std::string& model, bool input_is_filepath) {
        return tflite::FlatBufferFileToMlir(model, input_is_filepath);
      },
      R"pbdoc(
      Returns MLIR dump of the given TFLite model.
    )pbdoc");
}
