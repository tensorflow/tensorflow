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

#include <Python.h>

#include <queue>

#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

struct CallCounter;  // Forward declaration.

namespace tensorflow {

namespace py = pybind11;

// Class keeping track of how many recent calls triggered tracing.
struct CallCounter {
  int call_count;

  CallCounter(int max_call_history);
  void called_with_tracing();
  void called_without_tracing();
  int get_tracing_count();

 private:
  int max_call_history;
  std::queue<int> calls_per_tracings;
};

CallCounter::CallCounter(int max_call_history)
    : max_call_history(max_call_history), call_count(0) {}

void CallCounter::called_with_tracing() {
  ++call_count;
  calls_per_tracings.push(1);

  while (!calls_per_tracings.empty()) {
    if (call_count - calls_per_tracings.front() > max_call_history) {
      call_count -= calls_per_tracings.front();
      calls_per_tracings.pop();
    } else {
      break;
    }
  }
}

void CallCounter::called_without_tracing() {
  // We don't count tracing when users load a concrete function directly or
  // call get_concrete_function, so the first call can be not a tracing call.
  if (calls_per_tracings.empty()) {
    calls_per_tracings.push(0);
  }
  ++calls_per_tracings.back();
  ++call_count;
}

int CallCounter::get_tracing_count() {
  return calls_per_tracings.size();
}

PYBIND11_MODULE(_call_counter, m) {
  py::class_<CallCounter>(m, "CallCounter")
      .def(py::init<int>())
      .def("called_with_tracing", &CallCounter::called_with_tracing)
      .def("called_without_tracing", &CallCounter::called_without_tracing)
      .def("get_tracing_count", &CallCounter::get_tracing_count)
      .def_readonly("call_count", &CallCounter::call_count);
}

} // namespace tensorflow
