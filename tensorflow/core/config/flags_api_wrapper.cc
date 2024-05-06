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

#include <memory>
#include <string>

#include "absl/types/optional.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"

namespace py = pybind11;
namespace flags = tensorflow::flags;

using tensorflow::config::Flag;
using tensorflow::flags::Flags;

// Macro to expose the declared flag to python.
#define TF_PY_DECLARE_FLAG(flag_name) \
  flags.def_readwrite(#flag_name, &Flags::flag_name);

PYBIND11_MODULE(flags_pybind, m) {
  py::class_<Flag>(m, "Flag")
      .def("value", &Flag::value)
      .def("reset", &Flag::reset);

  py::class_<Flags, std::unique_ptr<Flags, py::nodelete>> flags(m, "Flags");
  flags.def(py::init(
      []() { return std::unique_ptr<Flags, py::nodelete>(&flags::Global()); }));
  // LINT.IfChange
  TF_PY_DECLARE_FLAG(test_only_experiment_1);
  TF_PY_DECLARE_FLAG(test_only_experiment_2);
  TF_PY_DECLARE_FLAG(enable_nested_function_shape_inference);
  TF_PY_DECLARE_FLAG(enable_quantized_dtypes_training);
  TF_PY_DECLARE_FLAG(graph_building_optimization);
  TF_PY_DECLARE_FLAG(op_building_optimization);
  TF_PY_DECLARE_FLAG(saved_model_fingerprinting);
  TF_PY_DECLARE_FLAG(tf_shape_default_int64);
  TF_PY_DECLARE_FLAG(more_stack_traces);
  TF_PY_DECLARE_FLAG(publish_function_graphs);
  TF_PY_DECLARE_FLAG(enable_aggressive_constant_replication);
  TF_PY_DECLARE_FLAG(enable_colocation_key_propagation_in_while_op_lowering);
  TF_PY_DECLARE_FLAG(enable_tf2min_ici_weight)
  // LINT.ThenChange(//tensorflow/core/config/flag_defs.h)
};
