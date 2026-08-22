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

#include <string>
#include <utility>

#include "Python.h"

#include "absl/status/status.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_proto.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;


#ifdef Py_GIL_DISABLED
namespace {

class ScopedPyObjectCriticalSection {
 public:
  explicit ScopedPyObjectCriticalSection(PyObject* object) {
    PyCriticalSection_Begin(&critical_section_, object);
  }

  ~ScopedPyObjectCriticalSection() {
    PyCriticalSection_End(&critical_section_);
  }

 private:
  PyCriticalSection critical_section_;
};

}  // namespace
#endif

template <typename F>
decltype(auto) RunEventsWriterMethod(py::handle self, F&& fn) {
#ifdef Py_GIL_DISABLED
  ScopedPyObjectCriticalSection critical_section(self.ptr());
#else
  (void)self;
#endif
  return std::forward<F>(fn)();
}

PYBIND11_MODULE(
    _pywrap_events_writer, m, pybind11::mod_gil_not_used()) {
  py::class_<absl::Status> Status(m, "Status", py::module_local());
  py::class_<tensorflow::EventsWriter> events_writer_class(m, "EventsWriter");

  events_writer_class.def(py::init<const std::string&>())
      .def("InitWithSuffix",
           [](py::object self_obj, const std::string& suffix) {
             auto* self = self_obj.cast<tensorflow::EventsWriter*>();
             if (self == nullptr) {
               throw py::value_error("EventsWriter is not initialized");
             }
             return RunEventsWriterMethod(
                 self_obj, [&]() { return self->InitWithSuffix(suffix); });
           })
      .def("FileName", [](py::object self_obj) {
        auto* self = self_obj.cast<tensorflow::EventsWriter*>();
        if (self == nullptr) {
          throw py::value_error("EventsWriter is not initialized");
        }
        return RunEventsWriterMethod(
            self_obj, [&]() { return self->FileName(); });
      })
      .def("_WriteSerializedEvent",
           [](py::object self_obj, const std::string& event_str) {
             auto* self = self_obj.cast<tensorflow::EventsWriter*>();
             if (self == nullptr) {
               throw py::value_error("EventsWriter is not initialized");
             }
             RunEventsWriterMethod(
                 self_obj, [&]() { self->WriteSerializedEvent(event_str); });
           })
      .def("Flush", [](py::object self_obj) {
        auto* self = self_obj.cast<tensorflow::EventsWriter*>();
        if (self == nullptr) {
          throw py::value_error("EventsWriter is not initialized");
        }
        return RunEventsWriterMethod(
            self_obj, [&]() { return self->Flush(); });
      })
      .def("Close", [](py::object self_obj) {
        auto* self = self_obj.cast<tensorflow::EventsWriter*>();
        if (self == nullptr) {
          throw py::value_error("EventsWriter is not initialized");
        }
        return RunEventsWriterMethod(
            self_obj, [&]() { return self->Close(); });
      })
      .def("WriteEvent",
           [](py::object self_obj, const py::object obj) {
             auto* self = self_obj.cast<tensorflow::EventsWriter*>();
             if (self == nullptr) {
               throw py::value_error("EventsWriter is not initialized");
             }

             tensorflow::CheckProtoType(obj, "tensorflow.Event");

             // Python conversion must happen outside the native object's
             // critical section to avoid re-entrant Python execution while
             // the object is locked.
             std::string serialized_event =
                 obj.attr("SerializeToString")().cast<std::string>();

             RunEventsWriterMethod(self_obj, [&]() {
               self->WriteSerializedEvent(serialized_event);
             });
           });
};
