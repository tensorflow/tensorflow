/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_UTIL_H_

#include <memory>
#include <vector>

#include "absl/strings/str_format.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

// Issues a Python deprecation warning. Throws a C++ exception if issuing the
// Python warning causes a Python exception to be raised.
template <typename... Args>
void PythonDeprecationWarning(const absl::FormatSpec<Args...>& format,
                              const Args&... args) {
  if (PyErr_WarnEx(PyExc_DeprecationWarning,
                   absl::StrFormat(format, args...).c_str(), 1) < 0) {
    throw pybind11::error_already_set();
  }
}

// Requests if given buffers are ready, awaits for results and returns OK if
// all of the buffers are ready or the last non-ok status.
Status AwaitBuffersReady(
    const std::vector<std::shared_ptr<PjRtBuffer>>& buffers);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_UTIL_H_
