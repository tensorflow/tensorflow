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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/core/tpu/kernels/sparse_core_layout.h"
#include "tensorflow/core/tpu/kernels/sparse_core_layout.pb.h"

namespace tensorflow::tpu {

namespace py = pybind11;

// Serializes operations on a SparseCoreLayoutStacker instance exposed to
// Python. Under free-threaded Python, multiple threads may call methods on
// the same pybind object concurrently.
class PySparseCoreLayoutStacker {
 public:
  PySparseCoreLayoutStacker(int num_partitions, bool disable_table_stacking,
                            int sparse_cores_per_partition)
      : stacker_(num_partitions, disable_table_stacking,
                 sparse_cores_per_partition) {}

  void SetActivationMemoryBytesLimit(int64_t activation_mem_bytes_limit) {
    absl::MutexLock lock(mu_);
    stacker_.SetActivationMemoryBytesLimit(activation_mem_bytes_limit);
  }

  void SetVariableShardBytesLimit(int64_t variable_shard_bytes_limit) {
    absl::MutexLock lock(mu_);
    stacker_.SetVariableShardBytesLimit(variable_shard_bytes_limit);
  }

  void SetStackingEnabled(bool stacking_enabled) {
    absl::MutexLock lock(mu_);
    stacker_.SetStackingEnabled(stacking_enabled);
  }

  absl::Status AddTable(const std::string& table_name, int64_t table_height,
                        int64_t table_width, const std::string& group,
                        int64_t output_samples, int64_t num_features) {
    absl::MutexLock lock(mu_);
    return stacker_.AddTable(table_name, table_height, table_width, group,
                             output_samples, num_features);
  }

  absl::StatusOr<SparseCoreTableLayouts> GetLayouts() {
    absl::MutexLock lock(mu_);
    return stacker_.GetLayouts();
  }

 private:
  absl::Mutex mu_;
  SparseCoreLayoutStacker stacker_;
};

PYBIND11_MODULE(_pywrap_sparse_core_layout, m, py::mod_gil_not_used()) {
  py::class_<PySparseCoreLayoutStacker>(m, "SparseCoreLayoutStacker")
      .def(py::init<int, bool, int>(), py::arg("num_partitions"),
           py::arg("disable_table_stacking"),
           py::arg("sparse_cores_per_partition"))
      .def("SetActivationMemoryBytesLimit",
           &PySparseCoreLayoutStacker::SetActivationMemoryBytesLimit)
      .def("SetVariableShardBytesLimit",
           &PySparseCoreLayoutStacker::SetVariableShardBytesLimit)
      .def("SetStackingEnabled", &PySparseCoreLayoutStacker::SetStackingEnabled)
      .def("AddTable", &PySparseCoreLayoutStacker::AddTable,
           py::arg("table_name"), py::arg("table_height"),
           py::arg("table_width"), py::arg("group"), py::arg("output_samples"),
           py::arg("num_features"))
      .def("GetLayouts", &PySparseCoreLayoutStacker::GetLayouts);
}

}  // namespace tensorflow::tpu
