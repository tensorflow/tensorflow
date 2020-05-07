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

#include "pybind11/pybind11.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/python/util/stack_trace.h"

PYBIND11_MODULE(_stack_trace_binding_for_test, m) {
  m.def("to_string",
        []() { return tensorflow::StackTrace::Capture().ToString(); });
  m.def("stack_trace_n_times", [](int n) {
    for (int i = 0; i < n; ++i) {
      auto stack_trace = tensorflow::StackTrace::Capture();
      tensorflow::testing::DoNotOptimize(stack_trace);
    }
  });
}
