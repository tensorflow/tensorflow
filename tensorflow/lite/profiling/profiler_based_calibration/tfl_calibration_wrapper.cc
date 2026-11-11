/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_calibration_utils.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace {
namespace py = pybind11;
using tflite::profiling::memory::MemoryUsage;
}  // namespace

PYBIND11_MODULE(_pywrap_tfl_calibration, m) {
  py::class_<odml::Range>(m, "Range", R"pbdoc(
Min/max range for a tensor.
Attributes:
    min: min value.
    max: max value.
)pbdoc")
      .def_readonly("min", &odml::Range::min)
      .def_readonly("max", &odml::Range::max);

  py::class_<odml::CalibrationOptions>(m, "CalibrationOptions", R"pbdoc(
Options for profiler-based calibration.
Attributes:
    valid_min: Exclusive minimum valid float value.
    valid_max: Exclusive maximum valid float value.
    collect_histogram: Whether to collect dynamic histograms across invocations.
    num_bins: Maximum number of bins per tensor histogram.
    histogram_tensor_names: Optional set of tensor names to collect histograms for.
)pbdoc")
      .def(py::init([](float valid_min, float valid_max, bool collect_histogram,
                       int num_bins,
                       std::optional<std::unordered_set<std::string>>
                           histogram_tensor_names) {
             return odml::CalibrationOptions{
                 .valid_min = valid_min,
                 .valid_max = valid_max,
                 .collect_histogram = collect_histogram,
                 .num_bins = num_bins,
                 .histogram_tensor_names = std::move(histogram_tensor_names),
             };
           }),
           py::arg("valid_min") = -3e38f, py::arg("valid_max") = 3e38f,
           py::arg("collect_histogram") = false, py::arg("num_bins") = 2048,
           py::arg("histogram_tensor_names") = std::nullopt)
      .def_readwrite("valid_min", &odml::CalibrationOptions::valid_min)
      .def_readwrite("valid_max", &odml::CalibrationOptions::valid_max)
      .def_readwrite("collect_histogram",
                     &odml::CalibrationOptions::collect_histogram)
      .def_readwrite("num_bins", &odml::CalibrationOptions::num_bins)
      .def_readwrite("histogram_tensor_names",
                     &odml::CalibrationOptions::histogram_tensor_names);

  py::class_<odml::TensorHistogram>(m, "TensorHistogram", R"pbdoc(
Accumulated dynamic histogram for a single tensor.
Attributes:
    counts: 1D int64 numpy array of bin counts.
    lower_bound: Lower bound of the first bin.
    bin_width: Width of each bin.
    min: Global minimum value observed for this tensor.
    max: Global maximum value observed for this tensor.
)pbdoc")
      .def_property_readonly("counts",
                             [](const odml::TensorHistogram& self) {
                               return py::array_t<int64_t>(
                                   static_cast<py::ssize_t>(self.counts.size()),
                                   self.counts.data());
                             })
      .def_readonly("lower_bound", &odml::TensorHistogram::lower_bound)
      .def_readonly("bin_width", &odml::TensorHistogram::bin_width)
      .def_readonly("min", &odml::TensorHistogram::min)
      .def_readonly("max", &odml::TensorHistogram::max);

  py::class_<odml::CalibrationSession>(m, "CalibrationSession", R"pbdoc(
Stateful calibration session that tracks per-invocation min/max and optionally
accumulates dynamic histograms across invocations.
)pbdoc")
      .def(py::init<odml::CalibrationOptions>(),
           py::arg("options") = odml::CalibrationOptions{})
      .def(
          "invoke",
          [](odml::CalibrationSession& self, py::object interpreter_handle,
             int subgraph_index) {
            auto* interpreter = reinterpret_cast<tflite::Interpreter*>(
                interpreter_handle.cast<intptr_t>());
            py::gil_scoped_release release;
            auto status_or_map =
                self.Invoke(interpreter, subgraph_index,
                            static_cast<tflite::StatefulErrorReporter*>(
                                interpreter->error_reporter()));
            if (!status_or_map.ok()) {
              throw std::runtime_error(
                  std::string(status_or_map.status().message()));
            }
            return status_or_map.value();
          },
          py::arg("interpreter_handle"), py::arg("subgraph_index"))
      .def("get_histograms", &odml::CalibrationSession::GetHistograms)
      .def("reset", &odml::CalibrationSession::Reset);

  py::class_<MemoryUsage>(m, "MemoryUsage", R"pbdoc(
Memory usage statistics.
Attributes:
    mem_footprint_kb: Max resident set size in KB.
    total_allocated_bytes: Total non-mmapped heap space allocated in bytes.
    in_use_allocated_bytes: Total heap bytes in use in bytes.
    private_footprint_bytes: Private footprint in bytes.
)pbdoc")
      .def_readonly("mem_footprint_kb", &MemoryUsage::mem_footprint_kb)
      .def_readonly("total_allocated_bytes",
                    &MemoryUsage::total_allocated_bytes)
      .def_readonly("in_use_allocated_bytes",
                    &MemoryUsage::in_use_allocated_bytes)
      .def_readonly("private_footprint_bytes",
                    &MemoryUsage::private_footprint_bytes);
  m.def(
      "get_memory_usage",
      []() {
        py::gil_scoped_release release;
        return odml::GetMemoryUsage();
      },
      R"pbdoc(
        Returns memory usage stats.
      )pbdoc");
  m.def(
      "InvokeWithCalibration",
      [](py::object interpreter_handle, int subgraph_index) {
        auto* interpreter = reinterpret_cast<tflite::Interpreter*>(
            interpreter_handle.cast<intptr_t>());
        py::gil_scoped_release release;
        auto status_or_map = odml::InvokeWithCalibration(
            interpreter, subgraph_index,
            static_cast<tflite::StatefulErrorReporter*>(
                interpreter->error_reporter()));
        if (!status_or_map.ok())
          throw std::runtime_error(
              std::string(status_or_map.status().message()));
        return status_or_map.value();
      },
      R"pbdoc(
        Invoke the given ``tf.lite.Interpreter`` for calibration. Assumes
        input tensors are already set.
        Args:
          interpreter: The ``tf.lite:Interpreter`` to invoke.
          subgraph_index: The subgraph index to run.
        Returns:
          dict: A map of tensor names to Range objects containing 'min' and 'max' values.
      )pbdoc");
}
