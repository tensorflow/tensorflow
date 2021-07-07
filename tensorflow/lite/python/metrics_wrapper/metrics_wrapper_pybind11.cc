/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/python/metrics_wrapper/metrics_wrapper.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;
using tflite::metrics_wrapper::MetricsWrapper;

PYBIND11_MODULE(_pywrap_tensorflow_lite_metrics_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_tensorflow_lite_metrics_wrapper
    -----
  )pbdoc";

  py::class_<MetricsWrapper>(m, "MetricsWrapper")
      .def(py::init([](const std::string& session_id) {
        auto* wrapper = MetricsWrapper::CreateMetricsWrapper(session_id);
        if (!wrapper) {
          throw std::invalid_argument("Failed to created MetricsWrapper");
        }
        return wrapper;
      }))
      .def("ExportMetrics", [](MetricsWrapper& self) {
        return tensorflow::PyoOrThrow(self.ExportMetrics());
      });
}
