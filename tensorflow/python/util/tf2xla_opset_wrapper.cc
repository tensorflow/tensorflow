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

#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/tf2xla/tf2xla_opset.h"

using tensorflow::GetRegisteredXlaOpsForDevice;

PYBIND11_MODULE(pywrap_xla_ops, m) {
  pybind11::google::ImportStatusModule();
  m.def(
      "get_gpu_kernel_names",
      []() -> absl::StatusOr<std::vector<std::string>> {
        return GetRegisteredXlaOpsForDevice("XLA_GPU_JIT");
      },
      R"pbdoc(
     Returns list of names of gpu ops that can be compiled.
    )pbdoc");
  m.def(
      "get_cpu_kernel_names",
      []() -> absl::StatusOr<std::vector<std::string>> {
        return GetRegisteredXlaOpsForDevice("XLA_CPU_JIT");
      },
      R"pbdoc(
     Returns list of names of cpu ops that can be compiled.
    )pbdoc");
};
