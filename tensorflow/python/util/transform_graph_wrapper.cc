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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/tools/graph_transforms/transform_graph.h"

namespace py = pybind11;

namespace tensorflow {

string TransformGraphWithStringInputs(string graph_def_string,
                                      string inputs_string,
                                      string outputs_string,
                                      string transforms_string) {
  GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    MaybeRaiseFromStatus(
        errors::InvalidArgument("Couldn't interpret input as a GraphDef"));
  }

  graph_transforms::TransformParameters params_list;
  absl::Status parse_status = graph_transforms::ParseTransformParameters(
      transforms_string, &params_list);
  if (!parse_status.ok()) {
    MaybeRaiseFromStatus(parse_status);
  }
  std::vector<string> inputs = str_util::Split(inputs_string, ',');
  std::vector<string> outputs = str_util::Split(outputs_string, ',');

  absl::Status transform_status = graph_transforms::TransformGraph(
      inputs, outputs, params_list, &graph_def);
  if (!transform_status.ok()) {
    MaybeRaiseFromStatus(transform_status);
  }
  string result;
  if (!graph_def.SerializeToString(&result)) {
    MaybeRaiseFromStatus(
        errors::InvalidArgument("Couldn't serialize output as a GraphDef"));
  }
  return result;
}

}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_transform_graph, m) {
  m.def(
      "TransformGraphWithStringInputs",
      [](const py::object graph_def_string, const py::object inputs_string,
         const py::object outputs_string, const py::object transforms_string) {
        return py::bytes(tensorflow::TransformGraphWithStringInputs(
            graph_def_string.cast<std::string>(),
            inputs_string.cast<std::string>(),
            outputs_string.cast<std::string>(),
            transforms_string.cast<std::string>()));
      });
};
