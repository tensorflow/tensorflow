/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_proto.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_events_writer, m) {
  py::class_<tensorflow::EventsWriter> events_writer_class(m, "EventsWriter");
  events_writer_class.def(py::init<const std::string&>())
      .def("InitWithSuffix",
           [](tensorflow::EventsWriter& self, const std::string& suffix) {
             return self.InitWithSuffix(suffix);
           })
      .def("FileName",
           [](tensorflow::EventsWriter& self) { return self.FileName(); })
      .def("_WriteSerializedEvent",
           [](tensorflow::EventsWriter& self,
              const absl::string_view event_str) {
             self.WriteSerializedEvent(event_str);
           })
      .def("Flush", [](tensorflow::EventsWriter& self) { return self.Flush(); })
      .def("Close", [](tensorflow::EventsWriter& self) { return self.Close(); })
      .def("WriteEvent",
           [](tensorflow::EventsWriter& self, const py::object obj) {
             // Verify the proto type is an event prior to writing.
             tensorflow::CheckProtoType(obj, "tensorflow.Event");
             self.WriteSerializedEvent(
                 obj.attr("SerializeToString")().cast<std::string>());
           });
};
