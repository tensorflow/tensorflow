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

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/framework/metrics.h"

namespace {

// Read an internal TensorFlow counter that has been exposed for testing.
int64_t test_counter_value(std::string name, std::string label) {
  return tensorflow::metrics::TestCounter(name, label)->value();
}

}  // anonymous namespace

PYBIND11_MODULE(_test_metrics_util, m) {
  m.def("test_counter_value", &test_counter_value);
};
