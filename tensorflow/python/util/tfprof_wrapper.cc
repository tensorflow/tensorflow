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

#include <mutex>
#include <string>

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/profiler/internal/print_model_analysis.h"

namespace py = pybind11;

namespace {

// The legacy multi-step tfprof API owns one process-global TFStats pointer.
// Serialize its lifecycle and all accesses from free-threaded Python.
std::mutex tfprof_mu;

}  // namespace

PYBIND11_MODULE(_pywrap_tfprof, m, pybind11::mod_gil_not_used()) {
  m.def("PrintModelAnalysis",
        [](const std::string* graph, const std::string* run_meta,
           const std::string* op_log, const std::string* command,
           const std::string* options) {
          std::string temp = tensorflow::tfprof::PrintModelAnalysis(
              graph, run_meta, op_log, command, options);
          return py::bytes(temp);
        });
  m.def("NewProfiler", [](const std::string* graph, const std::string* op_log) {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    return tensorflow::tfprof::NewProfiler(graph, op_log);
  });
  m.def("ProfilerFromFile", [](const std::string* filename) {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    tensorflow::tfprof::ProfilerFromFile(filename);
  });
  m.def("DeleteProfiler", []() {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    tensorflow::tfprof::DeleteProfiler();
  });
  m.def("AddStep", [](int64_t step, const std::string* graph,
                      const std::string* run_meta, const std::string* op_log) {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    return tensorflow::tfprof::AddStep(step, graph, run_meta, op_log);
  });
  m.def("SerializeToString", []() {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    std::string temp = tensorflow::tfprof::SerializeToString();
    return py::bytes(temp);
  });
  m.def("WriteProfile", [](const std::string* filename) {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    tensorflow::tfprof::WriteProfile(filename);
  });
  m.def("Profile", [](const std::string* command, const std::string* options) {
    std::lock_guard<std::mutex> lock(tfprof_mu);
    std::string temp = tensorflow::tfprof::Profile(command, options);
    return py::bytes(temp);
  });
}
