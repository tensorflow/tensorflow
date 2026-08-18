/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/backends/gpu/collectives/mori_collectives.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/mori_communicator.h"
#include "xla/backends/gpu/collectives/mori_stub.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

namespace shmem = ::mori::shmem;

namespace xla::gpu {

#define XLA_MORI_RETURN_IF_ERROR(expr)                           \
  do {                                                           \
    auto status = (expr);                                        \
    if (status != 0) {                                           \
      return absl::InternalError(                                \
          absl::StrFormat("MORI operation failed: %d", status)); \
    }                                                            \
  } while (0)

MoriCollectives::~MoriCollectives() {
  // NOTE this is most probably wrong since we need to call finalize
  // for all threads !
  if (initialized_) {
    Finalize();
  }
}

absl::StatusOr<CliqueId> MoriCollectives::CreateUniqueCliqueId() const {
  VLOG(3) << "Create MORI unique clique id";
  shmem::mori_shmem_uniqueid_t id;
  XLA_MORI_RETURN_IF_ERROR(shmem::ShmemGetUniqueId(&id));
  return CliqueId(absl::string_view(reinterpret_cast<char*>(id.data()),
                                    MORI_SHMEM_UNIQUE_ID_BYTES));
}

static absl::StatusOr<shmem::mori_shmem_uniqueid_t> AsMoriUniqueId(
    const CliqueId& clique_id) {
  if (clique_id.size() != MORI_SHMEM_UNIQUE_ID_BYTES) {
    return Internal(
        "CliqueId size is not equal to MORI_SHMEM_UNIQUE_ID_BYTES: %d vs %d",
        clique_id.size(), MORI_SHMEM_UNIQUE_ID_BYTES);
  }
  shmem::mori_shmem_uniqueid_t id;
  absl::c_copy(clique_id.data(), id.data());
  return id;
}

void MoriCollectives::Finalize() {
  VLOG(3) << "Finilizing MORI";
  shmem::ShmemFinalize();
}

absl::Status MoriCollectives::InitPe(size_t rank, size_t nranks,
                                     const CliqueId& clique_id,
                                     se::StreamExecutor* executor) {
  // ShmemInitAttr keys the per-device MORI state off the calling thread's
  // active HIP device, so we must activate `executor`'s context here.
  auto activate_context = executor->Activate();
  ABSL_ASSIGN_OR_RETURN(auto uid, AsMoriUniqueId(clique_id));
  shmem::mori_shmem_init_attr_t init_attr;
  XLA_MORI_RETURN_IF_ERROR(shmem::ShmemSetAttrUniqueIdArgs(
      static_cast<int32_t>(rank), static_cast<int32_t>(nranks), &uid,
      &init_attr));
  XLA_MORI_RETURN_IF_ERROR(
      shmem::ShmemInitAttr(shmem::MORI_SHMEM_INIT_WITH_UNIQUEID, &init_attr));
  VLOG(1) << "Initialized MORI PE rank " << rank << " of " << nranks;
  return absl::OkStatus();
}

absl::StatusOr<void*> MoriCollectives::Allocate(uint64_t bytes) {
  void* buffer = shmem::ShmemMalloc(bytes);  // ShmemMallocAlign
  if (buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to allocate %s (%llu bytes) from MORI memory",
                        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  VLOG(3) << absl::StreamFormat("Allocated %s (%llu bytes) for MORI: %p",
                                tsl::strings::HumanReadableNumBytes(bytes),
                                bytes, buffer);
  return buffer;
}

absl::Status MoriCollectives::Deallocate(void* buffer) {
  VLOG(3) << absl::StreamFormat("Start de-allocation for MORI buffer: %p",
                                buffer);
  shmem::ShmemFree(buffer);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
MoriCollectives::CreateCommunicatorsWithCancel(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config,
    std::shared_ptr<CancellationToken> cancel) {
  // Validate clique ids. With the MORI backend, we rely on the host to exchange
  // unique clique ids.
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return InvalidArgument("CliqueId is required to create MORI communicators");
  }
  if (clique_ids->data().size() != 1) {
    return InvalidArgument(
        "CliqueIds size must be 1 for MORI communicator initialization");
  }
  VLOG(1) << "Initialize MORI communicator for " << ranks.size() << " devices"
          << "; fingerprint(id)=" << clique_ids->fingerprint();

  const auto& gpu_config =
      tsl::down_cast<const GpuCollectives::Config&>(config);
  if (!gpu_config.blocking_communicators && !gpu_config.async_execution) {
    return FailedPrecondition(
        "GpuCollectives::Config blocking_communicators is false, but "
        "async_execution is false. Non-blocking communicators require "
        "asynchronous execution.");
  }

  // make_comm returns a new ncclComm_t.
  auto make_comm =
      [&, this](int i) -> absl::StatusOr<std::unique_ptr<MoriCommunicator>> {
    VLOG(1) << "Initialize MORI communicator for rank #" << ranks[i].rank
            << " of " << clique_key.num_devices()
            << "; fingerprint(id)=" << clique_ids->fingerprint()
            << "; size(id)=" << clique_ids->data().size();
    auto* device = tsl::down_cast<GpuCollectives::Device*>(ranks[i].device);
    // TF_RET_CHECK(device != nullptr);

    // When MORI was already initialized eagerly (see InitializeTopology), we
    // only build the communicator wrapper. Otherwise (e.g. unit tests that
    // bypass InitializeTopology) we lazily initialize this PE here.
    auto activate_context = device->stream_executor()->Activate();
    if (!initialized_) {
      ABSL_RETURN_IF_ERROR(InitPe(ranks[i].rank.value(), clique_key.num_devices(),
                             clique_ids->at(0), device->stream_executor()));
    }

    // Map each collective rank to its global MORI PE. In the single-process
    // eager-init path the MORI PE equals the global device ordinal.
    std::vector<int> rank_to_pe;
    rank_to_pe.reserve(clique_key.devices().size());
    for (GlobalDeviceId d : clique_key.devices()) {
      rank_to_pe.push_back(static_cast<int>(d.value()));
    }
    return MoriCommunicator::Create(this, cancel, ranks[i].rank.value(),
                                    rank_to_pe);
  };

  // Create all communicators. Each communicator is created on its own thread.
  std::vector<std::unique_ptr<Communicator>> comms(ranks.size());
  absl::Status status;
  absl::once_flag once;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateCommunicators",
                                 ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
      pool.Schedule([&, i]() {
        auto status_or_comm = make_comm(i);
        if (!status_or_comm.ok()) {
          absl::call_once(once, [&] { status = status_or_comm.status(); });
          return;
        }
        comms[i] = std::move(status_or_comm.value());
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  ABSL_RETURN_IF_ERROR(status);
  initialized_ = true;
  return comms;
}

absl::Status MoriCollectives::EagerInitLocalPes(ProcessId process_id,
                                                size_t nranks_local,
                                                size_t nranks_total,
                                                const CliqueId& clique_id) {
  if (nranks_local == 0) {
    return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName("ROCM"));

  // XLA:GPU assumes IOTA device assignment, so the global PE id of a local
  // device is `process_id * ndevs_per_process + dev_ord`.
  // dev_ord is needed for the executor to be activated.
  // pe is needed for the InitPe function.
  absl::Status status;
  absl::once_flag once;
  {
    // All PEs must initialize concurrently, so ShmemInitAttr's socket bootstrap
    // collective can rendezvous.
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "MoriEagerInit",
                                 nranks_local);
    for (size_t dev_ord = 0; dev_ord < nranks_local; dev_ord++) {
      const size_t pe = process_id.value() * nranks_local + dev_ord;
      pool.Schedule([&, dev_ord, pe]() {
        auto executor_or = platform->ExecutorForDevice(dev_ord);
        if (!executor_or.ok()) {
          absl::call_once(once, [&] { status = executor_or.status(); });
          return;
        }
        if (auto s = InitPe(pe, nranks_total, clique_id, *executor_or);
            !s.ok()) {
          absl::call_once(once, [&] { status = s; });
        }
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  ABSL_RETURN_IF_ERROR(status);
  initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<GpuCollectives::CliqueIdCallback>
MoriCollectives::InitializeTopology(const Topology& topology) {
  VLOG(1) << "InitializeTopology: num_processes=" << topology.num_processes
          << " device_count_per_process=" << topology.device_count_per_process
          << " kv_store=" << (topology.kv_store != nullptr);

  const size_t nranks_local = topology.device_count_per_process;
  if (topology.num_processes <= 1) {
    // Single process: eagerly initialize MORI for every local device as a
    // single global clique so that collective-memory allocations can use MORI's
    // static heap (ShmemMalloc) before any executable runs. The MORI PE equals
    // the local device ordinal.
    if (nranks_local == 0) {
      return absl::InvalidArgumentError("Wrong device_count_per_process");
    }
    ABSL_ASSIGN_OR_RETURN(CliqueId clique_id, CreateUniqueCliqueId());
    ABSL_RETURN_IF_ERROR(EagerInitLocalPes(topology.process_id, nranks_local,
                                      nranks_local, clique_id));
    VLOG(1) << "Eagerly initialized MORI for " << nranks_local
            << " local devices";
    return nullptr;
  }

  // Multi-process: eagerly initialize every local device as its global PE over
  // the full world. One root process generates the unique id and publishes it
  // via the key-value store; all other processes fetch it. Every PE across all
  // processes then initializes concurrently so MORI's socket bootstrap can
  // rendezvous, making the symmetric heap (ShmemMalloc/Allocate) available
  // before any executable runs.
  if (topology.kv_store == nullptr) {
    return InvalidArgument(
        "A key-value store is required for multi-process MORI initialization");
  }
  const size_t nranks_total = nranks_local * topology.num_processes;
  if (nranks_total == 0 || nranks_local == 0) {
    return InvalidArgument("Wrong nranks_total or nranks_local");
  }

  // Process 0 is the root that generates and publishes the unique
  // id; all other processes agree on this without extra coordination.
  const ProcessId root_process(0);

  static constexpr absl::string_view kMoriUidKey = "mori_shmem_global_uid";
  CliqueId clique_id;
  if (topology.process_id == root_process) {
    ABSL_ASSIGN_OR_RETURN(clique_id, CreateUniqueCliqueId());
    ABSL_RETURN_IF_ERROR(topology.kv_store->Set(kMoriUidKey, clique_id.ToString()));
  } else {
    ABSL_ASSIGN_OR_RETURN(std::string id_str,
                     topology.kv_store->Get(kMoriUidKey, absl::Minutes(10)));
    clique_id = CliqueId(id_str);
  }

  ABSL_RETURN_IF_ERROR(EagerInitLocalPes(topology.process_id, nranks_local,
                                    nranks_total, clique_id));
  VLOG(1) << "Eagerly initialized MORI: " << nranks_local << " local PEs of "
          << nranks_total << " global";
  return [clique_id](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
    return CliqueIds(clique_id);
  };
}

}  // namespace xla::gpu

// MoriCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("ROCM", "mori", -100,
                         std::make_unique<xla::gpu::MoriCollectives>());
