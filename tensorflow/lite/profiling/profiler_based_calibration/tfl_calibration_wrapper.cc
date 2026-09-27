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
#include <stdexcept>
#include <string>

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
