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

namespace tensorflow {

namespace py = pybind11;

// Class keeping track of how many recent calls triggered tracing.
class CallCounter {
 public:
  int call_count_;

  CallCounter(int max_call_history);
  void CalledWithTracing();
  void CalledWithoutTracing();
  int GetTracingCount();

 private:
  int max_call_history_;
  std::queue<int> calls_per_tracings_;
};

CallCounter::CallCounter(int max_call_history)
    : max_call_history_(max_call_history), call_count_(0) {}

void CallCounter::CalledWithTracing() {
  ++call_count_;
  calls_per_tracings_.push(1);

  while (!calls_per_tracings_.empty()) {
    if (call_count_ - calls_per_tracings_.front() > max_call_history_) {
      call_count_ -= calls_per_tracings_.front();
      calls_per_tracings_.pop();
    } else {
      break;
    }
  }
}

void CallCounter::CalledWithoutTracing() {
  // We don't count tracing when users load a concrete function directly or
  // call get_concrete_function, so the first call can be not a tracing call.
  if (calls_per_tracings_.empty()) {
    calls_per_tracings_.push(0);
  }
  ++calls_per_tracings_.back();
  ++call_count_;
}

int CallCounter::GetTracingCount() {
  return calls_per_tracings_.size();
}

PYBIND11_MODULE(_call_counter, m) {
  py::class_<CallCounter>(m, "CallCounter")
      .def(py::init<int>())
      .def("called_with_tracing", &CallCounter::CalledWithTracing)
      .def("called_without_tracing", &CallCounter::CalledWithoutTracing)
      .def("get_tracing_count", &CallCounter::GetTracingCount)
      .def_readonly("call_count", &CallCounter::call_count_);
}

} // namespace tensorflow
