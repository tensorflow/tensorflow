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

#include "xla/backends/cpu/collectives/mpi_collectives.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mpi.h"
#include "xla/backends/cpu/collectives/mpi_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

void MpiCollectives::Init() {
  int provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size_);
  VLOG(1) << "MPI rank=" << mpi_world_rank_ << " size=" << mpi_world_size_;
}

void MpiCollectives::Finalize() { MPI_Finalize(); }

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
MpiCollectives::CreateCommunicators(const CliqueKey& clique_key,
                                    const std::optional<CliqueIds>& clique_ids,
                                    absl::Span<const DeviceRank> ranks,
                                    const Config& config) {
  int flag;
  MPI_Is_thread_main(&flag);
  if (!flag) {
    return absl::UnknownError(
        "MPI: Communicator requested from a thread that is not "
        "the one MPI was initialized from. Multiple "
        "threads/devices per process are not yet supported.");
  }

  std::vector<std::unique_ptr<Communicator>> communicators;
  for (auto& device_rank : ranks) {
    size_t rank = device_rank.rank.value();
    int color;
    int key = 0;
    if (clique_key.num_devices() > 0) {
      color = static_cast<int>(clique_key.devices().at(0).value());
      key = rank;
    } else {
      color = MPI_UNDEFINED;
    }
    communicators.push_back(std::make_unique<MpiCommunicator>(color, key));
  }

  return communicators;
}

}  // namespace xla::cpu
