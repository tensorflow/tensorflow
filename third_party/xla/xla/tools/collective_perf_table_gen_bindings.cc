/* Copyright 2025 The OpenXLA Authors.

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

#include "absl/log/check.h"
#include "nanobind/nanobind.h"
#include "nanobind/nb_defs.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/tools/collective_perf_table_gen.h"

namespace nb = nanobind;

NB_MODULE(collective_perf_table_gen_bindings, m) {
  // Bind the Config struct
  nb::class_<xla::gpu::CollectivePerfTableGen::Config>(m, "Config")
      .def(nb::init<>())
      .def_rw("tensor_size_bytes_spec",
              &xla::gpu::CollectivePerfTableGen::Config::tensor_size_bytes_spec)
      .def_rw("collective_types",
              &xla::gpu::CollectivePerfTableGen::Config::collective_types)
      .def_rw("replica_groups_list",
              &xla::gpu::CollectivePerfTableGen::Config::replica_groups_list)
      .def_rw("dry_run", &xla::gpu::CollectivePerfTableGen::Config::dry_run)
      .def_rw("output", &xla::gpu::CollectivePerfTableGen::Config::output)
      .def_rw("coordinator_address",
              &xla::gpu::CollectivePerfTableGen::Config::coordinator_address)
      .def_rw("connection_timeout",
              &xla::gpu::CollectivePerfTableGen::Config::connection_timeout)
      .def_rw("num_nodes", &xla::gpu::CollectivePerfTableGen::Config::num_nodes)
      .def_rw("task_id", &xla::gpu::CollectivePerfTableGen::Config::task_id);

  // Bind the StepSpec struct
  nb::class_<xla::gpu::CollectivePerfTableGen::StepSpec>(m, "StepSpec")
      .def(nb::init<>())
      .def_rw("start", &xla::gpu::CollectivePerfTableGen::StepSpec::start)
      .def_rw("stop", &xla::gpu::CollectivePerfTableGen::StepSpec::stop)
      .def_rw("step", &xla::gpu::CollectivePerfTableGen::StepSpec::step)
      .def_rw("factor", &xla::gpu::CollectivePerfTableGen::StepSpec::factor);

  // Bind the CollectiveType enum
  nb::enum_<xla::gpu::CollectivePerfTableGen::CollectiveType>(m,
                                                              "CollectiveType")
      .value("UNSPECIFIED",
             xla::gpu::CollectivePerfTableGen::CollectiveType::UNSPECIFIED)
      .value("ALL_REDUCE",
             xla::gpu::CollectivePerfTableGen::CollectiveType::ALL_REDUCE)
      .value("REDUCE_SCATTER",
             xla::gpu::CollectivePerfTableGen::CollectiveType::REDUCE_SCATTER)
      .value("ALL_GATHER",
             xla::gpu::CollectivePerfTableGen::CollectiveType::ALL_GATHER)
      .export_values();

  m.def("run", [](xla::gpu::CollectivePerfTableGen::Config config) -> void {
    std::unique_ptr<xla::gpu::CollectivePerfTableGen> gen =
        xla::gpu::CollectivePerfTableGen::Create(config);
    auto table = gen->ComputeTable();
    CHECK_OK(gen->Dump(table));
  });
}
