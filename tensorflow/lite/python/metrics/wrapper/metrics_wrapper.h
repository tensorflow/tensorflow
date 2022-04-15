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
#ifndef TENSORFLOW_LITE_PYTHON_METRICS_WRAPPER_H_
#define TENSORFLOW_LITE_PYTHON_METRICS_WRAPPER_H_

#include <memory>
#include <string>

// Place `<locale>` before <Python.h> to avoid build failures in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

// We forward declare TFLite classes here to avoid exposing them to SWIG.
namespace tensorflow {
namespace monitoring {
class MetricsExporter;
}  // namespace monitoring
}  // namespace tensorflow

namespace tflite {
namespace metrics_wrapper {

class MetricsWrapper {
 public:
  using MetricsExporter = tensorflow::monitoring::MetricsExporter;
  // SWIG caller takes ownership of pointer.
  static MetricsWrapper* CreateMetricsWrapper(const std::string& session_id);

  ~MetricsWrapper();

  // Export metrics with Streamz.
  PyObject* ExportMetrics();

 private:
  explicit MetricsWrapper(std::unique_ptr<MetricsExporter> exporter);

  const std::unique_ptr<MetricsExporter> exporter_;
};

}  // namespace metrics_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_METRICS_WRAPPER_H_
