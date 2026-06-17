/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_CPU_COLLECTIVES_MPI_COLLECTIVES_H_
#define XLA_BACKENDS_CPU_COLLECTIVES_MPI_COLLECTIVES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/runtime/device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class MpiCollectives : public CpuCollectives {
 public:
  /*
  The user has to explicitly call Init() and Finalize() before and
  after use.
  For example, using the Python client, this can be achieved with:

  collectives = xla_client._xla.make_mpi_collectives()
  collectives.Init()
  atexit.register(collectives.Finalize)
  */
  void Init();
  void Finalize();

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) final;

 private:
  absl::Status ExchangeGlobalDeviceIds(
      absl::Span<GlobalDeviceId const> global_devices, int rank);

  int mpi_world_rank_;
  int mpi_world_size_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_COLLECTIVES_MPI_COLLECTIVES_H_
