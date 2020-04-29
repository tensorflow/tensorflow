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

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace tflite {
namespace support {

PYBIND11_MODULE(_pywrap_flatbuffers, m) {
  pybind11::class_<flatbuffers::IDLOptions>(m, "IDLOptions")
      .def(pybind11::init<>())
      .def_readwrite("strict_json", &flatbuffers::IDLOptions::strict_json);
  pybind11::class_<flatbuffers::Parser>(m, "Parser")
      .def(pybind11::init<const flatbuffers::IDLOptions&>())
      .def("parse",
           [](flatbuffers::Parser* self, const std::string& source) {
             return self->Parse(source.c_str());
           })
      .def_readonly("builder", &flatbuffers::Parser::builder_)
      .def_readonly("error", &flatbuffers::Parser::error_);
  pybind11::class_<flatbuffers::FlatBufferBuilder>(m, "FlatBufferBuilder")
      .def("clear", &flatbuffers::FlatBufferBuilder::Clear)
      .def("push_flat_buffer", [](flatbuffers::FlatBufferBuilder* self,
                                  const std::string& contents) {
        self->PushFlatBuffer(reinterpret_cast<const uint8_t*>(contents.c_str()),
                             contents.length());
      });
  m.def("generate_text_file", &flatbuffers::GenerateTextFile);
  m.def(
      "generate_text",
      [](const flatbuffers::Parser& parser,
         const std::string& buffer) -> std::string {
        std::string text;
        if (!flatbuffers::GenerateText(
                parser, reinterpret_cast<const void*>(buffer.c_str()), &text)) {
          return "";
        }
        return text;
      });
}

}  // namespace support
}  // namespace tflite
