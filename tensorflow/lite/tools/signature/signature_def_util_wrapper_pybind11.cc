/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/tools/signature/signature_def_util.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

py::bytes WrappedSetSignatureDefMap(
    const std::vector<uint8_t>& model_buffer,
    const std::map<std::string, std::string>& serialized_signature_def_map) {
  auto flatbuffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buffer.data()), model_buffer.size());
  auto* model = flatbuffer_model->GetModel();
  if (!model) {
    throw std::invalid_argument("Invalid model");
  }
  std::string data;
  std::map<std::string, tensorflow::SignatureDef> signature_def_map;
  for (const auto& entry : serialized_signature_def_map) {
    tensorflow::SignatureDef signature_def;
    if (!signature_def.ParseFromString(entry.second)) {
      throw std::invalid_argument("Cannot parse signature def");
    }
    signature_def_map[entry.first] = signature_def;
  }
  auto status = tflite::SetSignatureDefMap(model, signature_def_map, &data);
  if (status != ::tensorflow::OkStatus()) {
    throw std::invalid_argument(status.error_message());
  }
  return py::bytes(data);
}

std::map<std::string, py::bytes> WrappedGetSignatureDefMap(
    const std::vector<uint8_t>& model_buffer) {
  auto flatbuffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buffer.data()), model_buffer.size());
  auto* model = flatbuffer_model->GetModel();
  if (!model) {
    throw std::invalid_argument("Invalid model");
  }
  std::string content;
  std::map<std::string, tensorflow::SignatureDef> signature_def_map;
  auto status = tflite::GetSignatureDefMap(model, &signature_def_map);
  if (status != ::tensorflow::OkStatus()) {
    throw std::invalid_argument("Cannot parse signature def");
  }
  std::map<std::string, py::bytes> serialized_signature_def_map;
  for (const auto& entry : signature_def_map) {
    serialized_signature_def_map[entry.first] =
        py::bytes(entry.second.SerializeAsString());
  }
  return serialized_signature_def_map;
}

py::bytes WrappedClearSignatureDefs(const std::vector<uint8_t>& model_buffer) {
  auto flatbuffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buffer.data()), model_buffer.size());
  auto* model = flatbuffer_model->GetModel();
  if (!model) {
    throw std::invalid_argument("Invalid model");
  }
  std::string content;
  auto status = tflite::ClearSignatureDefMap(model, &content);
  if (status != ::tensorflow::OkStatus()) {
    throw std::invalid_argument("An unknown error occurred");
  }
  return py::bytes(content);
}

PYBIND11_MODULE(_pywrap_signature_def_util_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_signature_def_util_wrapper
    -----
  )pbdoc";

  m.def("SetSignatureDefMap", &WrappedSetSignatureDefMap);

  m.def("GetSignatureDefMap", &WrappedGetSignatureDefMap);

  m.def("ClearSignatureDefs", &WrappedClearSignatureDefs);
}
