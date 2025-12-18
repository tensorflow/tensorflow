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
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/nvshmem/nvshmem.h"   // IWYU pragma: keep
#include "third_party/nvshmem/nvshmemx.h"  // IWYU pragma: keep
#include "xla/backends/gpu/collectives/nvshmem_communicator.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/stream_executor/cuda/nvshmem.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

namespace xla::gpu {

NvshmemCollectives::~NvshmemCollectives() {
  if (se::gpu::nvshmem::IsInitialized()) {
    se::gpu::nvshmem::Finalize();
  }
}

bool NvshmemCollectives::IsInitialized() const {
  return se::gpu::nvshmem::IsInitialized();
}

NvshmemCollectives* NvshmemCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Get("CUDA", "nvshmem");
  CHECK_OK(collectives) << "Failed to get NVSHMEM collectives";  // Crash OK

  if (auto* nvshmem_collectives =
          tsl::down_cast<NvshmemCollectives*>(*collectives)) {
    return nvshmem_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for NVSHMEM";
}

absl::Status NvshmemCollectives::InitializeTopology(Topology topology) {
  se::gpu::nvshmem::SetEnvInfo(topology.node_id, topology.num_nodes,
                               topology.device_count_per_process,
                               topology.kv_store);
  return absl::OkStatus();
}

absl::StatusOr<void*> NvshmemCollectives::Allocate(uint64_t bytes) {
  TF_RETURN_IF_ERROR(se::gpu::nvshmem::InitializeOnce());
  VLOG(3) << absl::StreamFormat(
      "Start allocation of %s (%llu bytes) for NVSHMEM",
      tsl::strings::HumanReadableNumBytes(bytes), bytes);
  void* buffer = nvshmem_malloc(bytes);
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from NVSHMEM memory",
        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  return buffer;
}

absl::Status NvshmemCollectives::Deallocate(void* buffer) {
  TF_RETURN_IF_ERROR(se::gpu::nvshmem::InitializeOnce());
  VLOG(3) << absl::StreamFormat("Start de-allocation for NVSHMEM buffer: %p",
                                buffer);
  nvshmem_free(buffer);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Communicator>>
NvshmemCollectives::CreateCommunicator() {
  auto comm = std::make_unique<NvshmemCommunicator>(this);
  return comm;
}

}  // namespace xla::gpu

// NvshmemCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("CUDA", "nvshmem", -100,
                         std::make_unique<xla::gpu::NvshmemCollectives>());
