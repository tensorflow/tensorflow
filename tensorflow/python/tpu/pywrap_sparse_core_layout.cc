/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <pybind11/pybind11.h>

#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/core/tpu/kernels/sparse_core_layout.h"
#include "tensorflow/core/tpu/kernels/sparse_core_layout.pb.h"

namespace tensorflow::tpu {

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_sparse_core_layout, m) {
  py::class_<SparseCoreLayoutStacker>(m, "SparseCoreLayoutStacker")
      .def(py::init<int, int>(), py::arg("num_partitions"),
           py::arg("sparse_cores_per_partition"))
      .def("SetActivationMemoryBytesLimit",
           &SparseCoreLayoutStacker::SetActivationMemoryBytesLimit)
      .def("SetVariableShardBytesLimit",
           &SparseCoreLayoutStacker::SetVariableShardBytesLimit)
      .def("SetStackingEnabled", &SparseCoreLayoutStacker::SetStackingEnabled)
      .def("AddTable", &SparseCoreLayoutStacker::AddTable,
           py::arg("table_name"), py::arg("table_height"),
           py::arg("table_width"), py::arg("group"), py::arg("output_samples"))
      .def("GetLayouts", &SparseCoreLayoutStacker::GetLayouts);
}

}  // namespace tensorflow::tpu
