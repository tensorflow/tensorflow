/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/util.h"

namespace xla::gpu {

// A stub for GPU collectives when XLA:GPU compiled without collectives support.
class GpuCollectivesStub : public GpuCollectives {
 public:
  bool IsImplemented() const final { return false; }
  bool IsGlobalConfig() const final { return false; }

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final {
    return UnimplementedError();
  }

  absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback*, bool) final {
    return UnimplementedError();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey&, const std::optional<CliqueIds>&,
                      absl::Span<const DeviceRank>,
                      const Collectives::Config&) final {
    return UnimplementedError();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const>, int32_t, absl::Span<const RankId>,
      const Collectives::Config&) final {
    return UnimplementedError();
  }

  absl::StatusOr<void*> Allocate(uint64_t bytes) final {
    return UnimplementedError();
  }

  absl::Status Deallocate(void* buffer) final { return UnimplementedError(); }

  absl::Status InitializeTopology(Topology topology) final {
    return UnimplementedError();
  }

  absl::StatusOr<std::unique_ptr<Communicator>> CreateCommunicator() final {
    return UnimplementedError();
  }

 protected:
  static absl::Status UnimplementedError() {
    return Unimplemented("XLA compiled without GPU collectives support");
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_
