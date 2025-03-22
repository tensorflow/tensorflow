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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/debugging/leak_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

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

namespace {

// Thread-local information about a NCCL group.
//
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
struct GroupInfo {
  // NCCL groups can be nested. For example:
  //
  //     ncclGroupStart();
  //       ncclGroupStart();
  //         ncclGroupStart();
  //         ncclGroupEnd();
  //       ncclGroupEnd();
  //       ncclGroupStart();
  //       ncclGroupEnd();
  //     ncclGroupEnd();
  //
  // nesting_level is the current nesting level, or depth, of the group.
  int nesting_level = 0;

  // comms includes all communicators that have performed an operation in the
  // group at any nesting level.
  std::vector<ncclComm_t> comms;
};

// Returns the thread-local group information. NCCL groups are thread-local, so
// the group information is forced to be thread-local as well.
GroupInfo& ThreadLocalGroupInfo() {
  // As of March 2025, absl::NoDestructor is not available in OSS XLA.
  static thread_local GroupInfo* g = new GroupInfo();
  absl::IgnoreLeak(g);
  return *g;
}

}  // namespace

static ncclComm_t Cast(const Communicator* comm) {
  auto* nccl_communicator = tsl::down_cast<const NcclCommunicator*>(comm);
  CHECK(nccl_communicator != nullptr) << "Unsupported XLA communicator";
  return nccl_communicator->comm();
}

absl::StatusOr<CliqueId> NcclCollectives::CreateUniqueCliqueId() const {
  VLOG(3) << "Create NCCL unique clique id";
  ncclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return CliqueId(absl::string_view(id.internal, NCCL_UNIQUE_ID_BYTES));
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

static ncclConfig_t AsNcclConfig(const GpuCollectives::Config& config) {
  ncclConfig_t comm_config = NCCL_CONFIG_INITIALIZER;
  // Use non-blocking communicators. This allows us to ncclCommAbort stuck
  // collectives. It is unsafe to use ncclCommAbort with blocking communicators.
  //
  // TODO(mwhittaker): We still use blocking communicators. Other code changes
  // are needed to make non-blocking communicators work properly.
  comm_config.blocking = 1;
#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION > 50700
  comm_config.splitShare = config.split_share;
#endif
  if (config.max_nchannels > 0) {
    VLOG(1) << "Maximum number of channels is set to: " << comm_config.maxCTAs;
    comm_config.maxCTAs = config.max_nchannels;
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
NcclCollectives::CreateCommunicators(const CliqueKey& clique_key,
                                     const std::optional<CliqueIds>& clique_ids,
                                     absl::Span<const DeviceRank> ranks,
                                     const Collectives::Config& config) {
  // With NCCL backend we rely on host to exchange unique clique ids.
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return InvalidArgument("CliqueId is required to create NCCL communicators");
  }

  VLOG(1) << "Initialize NCCL communicator for " << ranks.size() << " devices"
          << "; fingerprint(id)=" << clique_ids->fingerprint();

  TF_ASSIGN_OR_RETURN(auto* gpu_config, TryCast(&config));
  ncclConfig_t comm_config = AsNcclConfig(*gpu_config);

  std::vector<ncclComm_t> comm_handles;
  std::vector<std::unique_ptr<Communicator>> comms;
  std::vector<NcclCommunicator*> nccl_comms;

  comm_handles.resize(ranks.size(), nullptr);
  comms.reserve(ranks.size());
  nccl_comms.reserve(ranks.size());

  if (clique_ids->data().size() != 1) {
    return InvalidArgument(
        "CliqueIds size must be 1 for NCCL communicator initialization");
  }

  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < ranks.size(); ++i) {
    VLOG(1) << "Initialize NCCL communicator for rank #" << ranks[i].rank
            << " of " << clique_key.num_devices()
            << "; fingerprint(id)=" << clique_ids->fingerprint()
            << "; size(id)=" << clique_ids->data().size();
    TF_ASSIGN_OR_RETURN(auto* device, TryCast(ranks[i].device));
    auto activate_context = device->stream_executor()->Activate();

    TF_ASSIGN_OR_RETURN(auto nccl_unique_id, AsNcclUniqueId(clique_ids->at(0)));
    XLA_NCCL_RETURN_IF_ERROR(ncclCommInitRankConfig(
        &comm_handles[i], clique_key.num_devices(), nccl_unique_id,
        ranks[i].rank.value(), &comm_config));
    auto comm = std::make_unique<NcclCommunicator>(this, comm_handles[i]);
    JoinGroup(comm.get());
    nccl_comms.push_back(comm.get());
    comms.push_back(std::move(comm));
  }
  TF_RETURN_IF_ERROR(GroupEnd());

  for (NcclCommunicator* comm : nccl_comms) {
    if (!JoinGroup(comm)) {
      TF_RETURN_IF_ERROR(PollUntilDone(comm->comm()));
    }
  }

  return comms;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
NcclCollectives::SplitCommunicators(absl::Span<const Communicator* const> comms,
                                    int32_t color,
                                    absl::Span<const RankId> keys,
                                    const Collectives::Config& config) {
  auto rank_formatter = [](std::string* str, RankId rank) {
    absl::StrAppend(str, rank.value());
  };

  VLOG(1) << absl::StreamFormat(
      "Split %d NCCL communicators using color %d and keys: [%s]", comms.size(),
      color, absl::StrJoin(keys, ",", rank_formatter));

  if (keys.size() != comms.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Comms and keys must have the same size, but %d != %d",
                        comms.size(), keys.size()));
  }

  TF_ASSIGN_OR_RETURN(auto* gpu_config, TryCast(&config));
  ncclConfig_t comm_config = AsNcclConfig(*gpu_config);

  // In contrast to grouped initialization communicator splitting initializes
  // communicators only after a successful call to `GroupEnd`, so we keep a
  // vector of handles and after successful splitting convert to RAII wrappers.
  std::vector<ncclComm_t> split_comms_handles;
  std::vector<std::unique_ptr<Communicator>> split_comms;
  std::vector<NcclCommunicator*> nccl_comms;
  split_comms_handles.resize(comms.size(), nullptr);
  split_comms.reserve(comms.size());
  nccl_comms.reserve(comms.size());

#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION >= 60000
  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < comms.size(); ++i) {
    VLOG(1) << "Split NCCL communicator " << comms[i] << " with color " << color
            << " and key " << keys[i];
    XLA_NCCL_RETURN_IF_ERROR(
        ncclCommSplit(Cast(comms[i]), color, keys[i].value(),
                      &split_comms_handles[i], &comm_config));
    auto comm =
        std::make_unique<NcclCommunicator>(this, split_comms_handles[i]);
    JoinGroup(comms[i]);
    JoinGroup(comm.get());
    nccl_comms.push_back(comm.get());
    split_comms.push_back(std::move(comm));
  }
  TF_RETURN_IF_ERROR(GroupEnd());

  for (size_t i = 0; i < split_comms_handles.size(); ++i) {
    if (!JoinGroup(comms[i])) {
      TF_RETURN_IF_ERROR(PollUntilDone(Cast(comms[i])));
    }
    if (!JoinGroup(nccl_comms[i])) {
      TF_RETURN_IF_ERROR(PollUntilDone(nccl_comms[i]->comm()));
    }
  }
  return split_comms;
#else
  return absl::UnimplementedError(
      absl::StrFormat("%s:%d: NCCL operation ncclCommSplit not implemented",
                      __FILE__, __LINE__));
#endif  // !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION >= 60000
}

absl::Status NcclCollectives::GroupStart() {
  VLOG(5) << "Start NCCL group";
  XLA_NCCL_RETURN_IF_ERROR(ncclGroupStart());
  GroupInfo& g = ThreadLocalGroupInfo();
  g.nesting_level++;
  VLOG(5) << "NCCL group nesting level = " << g.nesting_level;
  return absl::OkStatus();
}

absl::Status NcclCollectives::GroupEnd() {
  VLOG(5) << "End NCCL group";
  XLA_NCCL_RETURN_IF_ERROR(ncclGroupEnd());
  GroupInfo& g = ThreadLocalGroupInfo();
  g.nesting_level--;
  CHECK_GE(g.nesting_level, 0);
  VLOG(5) << "NCCL group nesting level = " << g.nesting_level;

  if (g.nesting_level > 0) {
    // Though NCCL allows groups to be nested, no operations are actually
    // performed until the outermost group ends. The inner calls to GroupStart()
    // and GroupEnd() are effectively noops.
    //
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
    return absl::OkStatus();
  }

  // Make sure to clear g.comms, even if we encounter an error.
  absl::Cleanup clear_comms = [&g]() {
    // The heap leak checker doesn't like g.comms.clear(), which may not free
    // the underlying heap-allocated memory, so we assign a new vector.
    g.comms = std::vector<ncclComm_t>();
  };

  // Wait for every communicator in the group to finish.
  for (ncclComm_t comm : g.comms) {
    TF_RETURN_IF_ERROR(PollUntilDone(comm));
  }

  return absl::OkStatus();
}

bool NcclCollectives::JoinGroup(const Communicator* communicator) {
  ncclComm_t comm = Cast(communicator);
  GroupInfo& g = ThreadLocalGroupInfo();
  if (g.nesting_level > 0) {
    VLOG(5) << "Adding NCCL communicator " << comm << " to group";
    g.comms.push_back(comm);
    return true;
  }
  return false;
}

}  // namespace xla::gpu

XLA_COLLECTIVES_REGISTER("gpu", "nccl", 1,
                         std::make_unique<xla::gpu::NcclCollectives>());
