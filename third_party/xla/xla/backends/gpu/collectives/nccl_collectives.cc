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

#include "xla/backends/gpu/collectives/nccl_collectives.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

absl::StatusOr<CliqueId> NcclCollectives::CreateUniqueCliqueId() const {
  VLOG(3) << "Create NCCL unique clique id";
  ncclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return CliqueId(std::string_view(id.internal, NCCL_UNIQUE_ID_BYTES));
}

bool NcclCollectives::IsGlobalConfig() const {
  static const char* const nccl_comm_id = std::getenv("NCCL_COMM_ID");
  return nccl_comm_id != nullptr;
}

absl::StatusOr<const NcclCollectives::CliqueIdCallback*>
NcclCollectives::GetCliqueIdCallback(const CliqueIdCallback* clique_id_callback,
                                     bool is_local) {
  if (clique_id_callback != nullptr) return clique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the clique_id_callback must be provided by the client.";

  static auto* local_callback = new CliqueIdCallback(
      [this](const CliqueKey&) { return CreateUniqueCliqueId(); });
  return local_callback;
}

static ncclConfig_t AsNcclConfig(const CliqueId& clique_id,
                                 const GpuCollectives::Config& config) {
  ncclConfig_t comm_config = NCCL_CONFIG_INITIALIZER;
#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION > 50700
  comm_config.splitShare = config.split_share;
#endif
  if (config.max_nchannels > 0) {
    comm_config.maxCTAs = config.max_nchannels;
    VLOG(1) << "Maximum number of channels for fingerprint(id)="
            << clique_id.fingerprint() << " is set to: " << comm_config.maxCTAs;
  }
  return comm_config;
}

static absl::StatusOr<ncclUniqueId> AsNcclUniqueId(const CliqueId& clique_id) {
  if (clique_id.size() != NCCL_UNIQUE_ID_BYTES) {
    return Internal(
        "CliqueId size is not equal to NCCL_UNIQUE_ID_BYTES: %d vs %d",
        clique_id.size(), NCCL_UNIQUE_ID_BYTES);
  }
  ncclUniqueId id;
  absl::c_copy(clique_id.data(), id.internal);
  return id;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
NcclCollectives::CreateCommunicators(int32_t nranks,
                                     const CliqueKey& clique_key,
                                     const std::optional<CliqueId>& clique_id,
                                     absl::Span<const DeviceRank> ranks,
                                     const Collectives::Config& config) {
  // With NCCL backend we rely on host to exchange unique clique id.
  if (!clique_id.has_value()) {
    return InvalidArgument("CliqueId is required to create NCCL communicators");
  }

  VLOG(1) << "Initialize NCCL communicator for " << ranks.size() << " devices"
          << "; fingerprint(id)=" << clique_id->fingerprint();

  TF_ASSIGN_OR_RETURN(auto* gpu_config, TryCast(&config));
  ncclConfig_t comm_config = AsNcclConfig(*clique_id, *gpu_config);

  std::vector<ncclComm_t> comm_handles;
  std::vector<std::unique_ptr<Communicator>> comms;

  comm_handles.resize(ranks.size(), nullptr);
  comms.reserve(ranks.size());

  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < ranks.size(); ++i) {
    VLOG(1) << "Initialize NCCL communicator for rank #" << ranks[i].rank
            << " of " << nranks
            << "; fingerprint(id)=" << clique_id->fingerprint();
    TF_ASSIGN_OR_RETURN(auto* device, TryCast(ranks[i].device));
    auto activate_context = device->stream_executor()->Activate();

    TF_ASSIGN_OR_RETURN(auto nccl_unique_id, AsNcclUniqueId(*clique_id));
    XLA_NCCL_RETURN_IF_ERROR(
        ncclCommInitRankConfig(&comm_handles[i], nranks, nccl_unique_id,
                               ranks[i].rank.value(), &comm_config));
  }
  TF_RETURN_IF_ERROR(GroupEnd());

  for (ncclComm_t comm_handle : comm_handles) {
    comms.emplace_back(std::make_unique<NcclCommunicator>(comm_handle));
  }

  return comms;
}

absl::Status NcclCollectives::GroupStart() {
  VLOG(5) << "Start NCCL group";
  return XLA_NCCL_STATUS(ncclGroupStart());
}

absl::Status NcclCollectives::GroupEnd() {
  VLOG(5) << "End NCCL group";
  return XLA_NCCL_STATUS(ncclGroupEnd());
}

}  // namespace xla::gpu
