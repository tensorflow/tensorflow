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

#include <Python.h>

#include <string>

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/common_runtime/quantize_training.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

namespace tensorflow {
static PyObject* DoQuantizeTrainingOnGraphDefHelper(const string& input_graph,
                                                    int num_bits) {
  string result;
  // TODO(suharshs): Make the QuantizeAndDequantizeV2 configurable.
  tensorflow::MaybeRaiseFromStatus(
      tensorflow::DoQuantizeTrainingOnSerializedGraphDef(
          input_graph, num_bits, "QuantizeAndDequantizeV2", &result));

  PyObject* py_str = PyBytes_FromStringAndSize(result.data(), result.size());
  if (!py_str) {
    tensorflow::MaybeRaiseFromStatus(tensorflow::errors::Internal(
        "Failed to generate serialized string of the rewritten graph."));
  }
  return py_str;
}
}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_quantize_training, m) {
  m.def("DoQuantizeTrainingOnGraphDefHelper",
        [](const py::object input_graph, int num_bits) {
          return tensorflow::PyoOrThrow(
              tensorflow::DoQuantizeTrainingOnGraphDefHelper(
                  input_graph.cast<std::string>(), num_bits));
        });
};
