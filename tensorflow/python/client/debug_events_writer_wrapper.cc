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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/debug_events_writer.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_proto.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

PYBIND11_MODULE(_pywrap_debug_events_writer, m) {
  namespace py = pybind11;
  using namespace tensorflow;         // NOLINT(build/namespaces)
  using namespace tensorflow::tfdbg;  // NOLINT(build/namespaces)

  m.def("Init",
        [](const std::string& dump_root, const int64 circular_buffer_size) {
          DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
              dump_root, circular_buffer_size);
          if (!writer->Init().ok()) {
            throw py::value_error(tensorflow::strings::Printf(
                "Failed to initialize debug events writer at: %s",
                dump_root.c_str()));
          }
        });
  m.def("WriteSourceFile",
        [](const std::string& dump_root, const py::object obj) {
          CheckProtoType(obj, "tensorflow.DebugEvent");
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          writer->WriteSerializedNonExecutionDebugEvent(
              obj.attr("SerializeToString")().cast<std::string>(),
              tfdbg::DebugEventFileType::SOURCE_FILES);
        });
  m.def("WriteStackFrameWithId",
        [](const std::string& dump_root, const py::object& obj) {
          CheckProtoType(obj, "tensorflow.DebugEvent");
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          writer->WriteSerializedNonExecutionDebugEvent(
              obj.attr("SerializeToString")().cast<std::string>(),
              tfdbg::DebugEventFileType::STACK_FRAMES);
        });
  m.def("WriteGraphOpCreation",
        [](const std::string& dump_root, const py::object& obj) {
          CheckProtoType(obj, "tensorflow.DebugEvent");
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          writer->WriteSerializedNonExecutionDebugEvent(
              obj.attr("SerializeToString")().cast<std::string>(),
              tfdbg::DebugEventFileType::GRAPHS);
        });
  m.def("WriteDebuggedGraph",
        [](const std::string& dump_root, const py::object& obj) {
          CheckProtoType(obj, "tensorflow.DebugEvent");
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          writer->WriteSerializedNonExecutionDebugEvent(
              obj.attr("SerializeToString")().cast<std::string>(),
              tfdbg::DebugEventFileType::GRAPHS);
        });
  m.def("WriteExecution",
        [](const std::string& dump_root, const py::object& obj) {
          CheckProtoType(obj, "tensorflow.DebugEvent");
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          writer->WriteSerializedExecutionDebugEvent(
              obj.attr("SerializeToString")().cast<std::string>(),
              tfdbg::DebugEventFileType::EXECUTION);
        });
  m.def("WriteGraphExecutionTrace",
        [](const std::string& dump_root, const py::object& obj) {
          CheckProtoType(obj, "tensorflow.DebugEvent");
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          writer->WriteSerializedExecutionDebugEvent(
              obj.attr("SerializeToString")().cast<std::string>(),
              tfdbg::DebugEventFileType::GRAPH_EXECUTION_TRACES);
        });
  m.def("RegisterDeviceAndGetId",
        [](const std::string& dump_root, const std::string& device_name) {
          DebugEventsWriter* writer =
              DebugEventsWriter::GetDebugEventsWriter(dump_root);
          return writer->RegisterDeviceAndGetId(device_name);
        });
  m.def("FlushNonExecutionFiles", [](const std::string& dump_root) {
    DebugEventsWriter* writer =
        DebugEventsWriter::GetDebugEventsWriter(dump_root);
    writer->FlushNonExecutionFiles();
  });
  m.def("FlushExecutionFiles", [](const std::string& dump_root) {
    DebugEventsWriter* writer =
        DebugEventsWriter::GetDebugEventsWriter(dump_root);
    writer->FlushExecutionFiles();
  });
  m.def("Close", [](const std::string& dump_root) {
    DebugEventsWriter* writer =
        DebugEventsWriter::GetDebugEventsWriter(dump_root);
    writer->Close();
  });
};
