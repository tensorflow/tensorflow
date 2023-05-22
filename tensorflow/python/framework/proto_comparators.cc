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

#include <memory>
#include <string>

#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

struct ProtoComparisonOptions;  // Forward declaration

namespace tensorflow {

namespace {

namespace py = pybind11;
namespace tf = tensorflow;

struct ProtoComparisonOptions {
  bool treat_nan_as_equal;
};

bool EqualsGraphDef(string graphdef_string1, string graphdef_string2,
                    const ProtoComparisonOptions& options) {
  GraphDef graph_def_1;
  if (!graph_def_1.ParseFromString(graphdef_string1)) {
    MaybeRaiseFromStatus(errors::InvalidArgument(
        "Couldn't interpret first argument as a GraphDef"));
  }
  GraphDef graph_def_2;
  if (!graph_def_2.ParseFromString(graphdef_string2)) {
    MaybeRaiseFromStatus(errors::InvalidArgument(
        "Couldn't interpret second argument as a GraphDef"));
  }
  tf::protobuf::util::MessageDifferencer differencer;
  // Order doesnt matter in node defs, or functions in the function library and
  // their nested nodes.
  differencer.TreatAsSet(GraphDef::descriptor()->FindFieldByName("node"));
  differencer.TreatAsSet(
      FunctionDefLibrary::descriptor()->FindFieldByName("function"));
  differencer.TreatAsSet(
      FunctionDefLibrary::descriptor()->FindFieldByName("gradient"));
  differencer.TreatAsSet(
      FunctionDef::descriptor()->FindFieldByName("node_def"));
  tf::protobuf::util::DefaultFieldComparator comparator;
  comparator.set_treat_nan_as_equal(options.treat_nan_as_equal);
  differencer.set_field_comparator(&comparator);
  return differencer.Compare(graph_def_1, graph_def_2);
}

PYBIND11_MODULE(_proto_comparators, m) {
  py::class_<tensorflow::ProtoComparisonOptions>(m, "ProtoComparisonOptions")
      .def(py::init<const bool&>());
  m.def("EqualsGraphDef", &EqualsGraphDef,
        "GraphDef equality test taking comparison options.");
}

}  // anonymous namespace

}  // namespace tensorflow
