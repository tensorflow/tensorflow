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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

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

  static auto* const local_callback = new CliqueIdCallback(
      [this](const CliqueKey&) { return CreateUniqueCliqueId(); });
  return local_callback;
}

static ncclConfig_t AsNcclConfig(const GpuCollectives::Config& config) {
  ncclConfig_t comm_config = NCCL_CONFIG_INITIALIZER;
  comm_config.blocking = config.blocking_communicators ? 1 : 0;
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
  // Validate clique ids. With the NCCL backend, we rely on the host to exchange
  // unique clique ids.
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return InvalidArgument("CliqueId is required to create NCCL communicators");
  }
  if (clique_ids->data().size() != 1) {
    return InvalidArgument(
        "CliqueIds size must be 1 for NCCL communicator initialization");
  }
  VLOG(1) << "Initialize NCCL communicator for " << ranks.size() << " devices"
          << "; fingerprint(id)=" << clique_ids->fingerprint();

  // Validate collectives config.
  TF_ASSIGN_OR_RETURN(const GpuCollectives::Config* gpu_config,
                      TryCast(&config));
  if (!gpu_config->blocking_communicators && !gpu_config->async_execution) {
    return FailedPrecondition(
        "GpuCollectives::Config blocking_communicators is false, but "
        "async_execution is false. Non-blocking communicators require "
        "asynchronous execution.");
  }
  ncclConfig_t comm_config = AsNcclConfig(*gpu_config);

  // make_comm returns a new ncclComm_t.
  auto make_comm = [&](int i) -> absl::StatusOr<ncclComm_t> {
    VLOG(1) << "Initialize NCCL communicator for rank #" << ranks[i].rank
            << " of " << clique_key.num_devices()
            << "; fingerprint(id)=" << clique_ids->fingerprint()
            << "; size(id)=" << clique_ids->data().size();
    TF_ASSIGN_OR_RETURN(auto* device, TryCast(ranks[i].device));
    auto activate_context = device->stream_executor()->Activate();
    TF_ASSIGN_OR_RETURN(auto nccl_unique_id, AsNcclUniqueId(clique_ids->at(0)));
    ncclComm_t comm;
    XLA_NCCL_RETURN_IF_ERROR(
        ncclCommInitRankConfig(&comm, clique_key.num_devices(), nccl_unique_id,
                               ranks[i].rank.value(), &comm_config));
    return comm;
  };

  // Create all communicators. Each communicator is created on its own thread.
  std::vector<std::unique_ptr<Communicator>> comms(ranks.size());
  std::vector<absl::Status> statuses(ranks.size());
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateCommunicators",
                                 ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
      pool.Schedule([&, i]() {
        absl::StatusOr<std::unique_ptr<NcclCommunicator>> comm =
            NcclCommunicator::Create(std::bind(make_comm, i),
                                     gpu_config->async_execution);
        if (!comm.ok()) {
          statuses[i] = comm.status();
          return;
        }
        comms[i] = *std::move(comm);
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  for (const absl::Status& s : statuses) {
    TF_RETURN_IF_ERROR(s);
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

#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION >= 60000
  auto make_comm = [&](int i) -> absl::StatusOr<ncclComm_t> {
    VLOG(1) << "Split NCCL communicator " << comms[i] << " with color " << color
            << " and key " << keys[i];
    ncclComm_t split_comm;
    XLA_NCCL_RETURN_IF_ERROR(ncclCommSplit(
        Cast(comms[i]), color, keys[i].value(), &split_comm, &comm_config));
    return split_comm;
  };

  std::vector<absl::Status> statuses(comms.size());
  std::vector<std::unique_ptr<Communicator>> split_comms(comms.size());
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "SplitCommunicators",
                                 comms.size());
    for (size_t i = 0; i < comms.size(); ++i) {
      pool.Schedule([&, i]() {
        absl::StatusOr<std::unique_ptr<NcclCommunicator>> comm =
            NcclCommunicator::Create(std::bind(make_comm, i),
                                     gpu_config->async_execution);
        if (!comm.ok()) {
          statuses[i] = comm.status();
          return;
        }
        split_comms[i] = *std::move(comm);
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  for (const absl::Status& s : statuses) {
    TF_RETURN_IF_ERROR(s);
  }
  return split_comms;
#else
  return absl::UnimplementedError(
      absl::StrFormat("%s:%d: NCCL operation ncclCommSplit not implemented",
                      __FILE__, __LINE__));
#endif  // !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION >= 60000
}

static absl::StatusOr<xla::gpu::GpuCollectives*> GetNvshmemCollectives() {
  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Get("gpu", "nvshmem"));
  xla::gpu::GpuCollectives* nvshmem_collectives =
      tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);
  if (nvshmem_collectives == nullptr) {
    return absl::InternalError("Failed to get NVSHMEM collectives");
  }

  return nvshmem_collectives;
}

absl::StatusOr<void*> NcclCollectives::Allocate(uint64_t bytes) {
  if (xla::GetDebugOptionsFromFlags().xla_gpu_experimental_enable_nvshmem()) {
    TF_ASSIGN_OR_RETURN(auto* nvshmem_collectives, GetNvshmemCollectives());
    return nvshmem_collectives->Allocate(bytes);
  }

  void* ptr = nullptr;
  ncclResult_t res = ncclMemAlloc(&ptr, bytes);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to allocate %s (%llu bytes) from device collective memory: %s, "
        "Last NCCL warning(error) log entry (may be unrelated): %s",
        tsl::strings::HumanReadableNumBytes(bytes), bytes,
        ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }
  VLOG(2) << "Allocated collective memory " << ptr << " of " << bytes
          << " bytes";
  return ptr;
}

absl::Status NcclCollectives::Deallocate(void* location) {
  if (xla::GetDebugOptionsFromFlags().xla_gpu_experimental_enable_nvshmem()) {
    TF_ASSIGN_OR_RETURN(auto* nvshmem_collectives, GetNvshmemCollectives());
    return nvshmem_collectives->Deallocate(location);
  }

  ncclResult_t res = ncclMemFree(location);
  if (res != ncclSuccess) {
    return absl::InternalError(absl::StrFormat(
        "failed to free device collective memory at %p; result: %s, Last NCCL "
        "warning(error) log entry (may be unrelated): %s",
        location, ncclGetErrorString(res), ncclGetLastError(nullptr)));
  }

  VLOG(2) << "Deallocated collective memory " << location;
  return absl::OkStatus();
}

class NcclIdStore {
 public:
  NcclIdStore(int node_id,
              absl::flat_hash_map<GlobalDeviceId, int> device_to_node,
              std::shared_ptr<KeyValueStoreInterface> kv_store)
      : node_id_(node_id),
        device_to_node_(std::move(device_to_node)),
        kv_store_(std::move(kv_store)) {}

  absl::StatusOr<CliqueId> GetNcclUniqueId(const CliqueKey& key) {
    auto* gpu_key = tsl::down_cast<const gpu::GpuCliqueKey*>(&key);
    if (gpu_key == nullptr) {
      return InvalidArgument("Expected GPU clique key");
    }

    // The caller must ensure that threads calling this method concurrently have
    // unique keys, otherwise the global key-value store may hold the wrong
    // value.
    {
      absl::MutexLock lock(&mu_);
      auto it = cache_.find(*gpu_key);
      if (it != cache_.end()) {
        return it->second;
      }
    }
    CliqueId clique_id;
    int primary_node_id = device_to_node_.at(gpu_key->root_device());
    if (node_id_ == primary_node_id) {
      TF_ASSIGN_OR_RETURN(
          clique_id, gpu::GpuCollectives::Default()->CreateUniqueCliqueId());
      TF_RETURN_IF_ERROR(
          kv_store_->Set(gpu_key->ToString(), clique_id.ToString()));
    } else {
      TF_ASSIGN_OR_RETURN(
          std::string id_str,
          kv_store_->Get(gpu_key->ToString(), absl::Minutes(10)));
      clique_id = CliqueId(id_str);
    }
    absl::MutexLock lock(&mu_);
    auto result = cache_.emplace(*gpu_key, std::move(clique_id));
    TF_RET_CHECK(result.second) << "Unique ID already in cache.";
    return result.first->second;
  }

 private:
  const int node_id_;
  const absl::flat_hash_map<GlobalDeviceId, int> device_to_node_;
  const std::shared_ptr<KeyValueStoreInterface> kv_store_;

  absl::Mutex mu_;
  absl::flat_hash_map<gpu::GpuCliqueKey, CliqueId> cache_ ABSL_GUARDED_BY(mu_);
};

absl::Status NcclCollectives::InitializeTopology(
    NcclCollectives::Topology topology) {
  if (xla::GetDebugOptionsFromFlags().xla_gpu_experimental_enable_nvshmem()) {
    TF_ASSIGN_OR_RETURN(auto* nvshmem_collectives, GetNvshmemCollectives());
    TF_RETURN_IF_ERROR(nvshmem_collectives->InitializeTopology(topology));
  }

  if (topology.num_nodes > 1) {
    auto nccl_id_store = std::make_shared<NcclIdStore>(
        topology.node_id, topology.device_id_to_node_id,
        std::move(topology.kv_store));
    topology.gpu_executable_run_options->set_clique_id_callback(
        [nccl_id_store](const CliqueKey& key) {
          return nccl_id_store->GetNcclUniqueId(key);
        });
  }
  return absl::OkStatus();
}
}  // namespace xla::gpu

XLA_COLLECTIVES_REGISTER("gpu", "nccl", 1,
                         std::make_unique<xla::gpu::NcclCollectives>());
