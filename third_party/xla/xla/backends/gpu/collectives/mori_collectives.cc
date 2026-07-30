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
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
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

//===----------------------------------------------------------------------===//
// MoriIdStore
//===----------------------------------------------------------------------===//

namespace {
class MoriIdStore {
 public:
  MoriIdStore(ProcessId process_id,
              absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
              std::shared_ptr<KeyValueStoreInterface> kv_store)
      : process_id_(process_id),
        device_to_process_(std::move(device_to_process)),
        kv_store_(std::move(kv_store)) {}

  absl::StatusOr<CliqueIds> GetCliqueIds(const CliqueKey& key,
                                         MoriCollectives& mori_collectives) {
    auto* gpu_key = tsl::down_cast<const gpu::GpuCliqueKey*>(&key);
    if (gpu_key == nullptr) {
      return InvalidArgument("Expected GPU clique key");
    }

    // The caller must ensure that threads calling this method concurrently have
    // unique keys, otherwise the global key-value store may hold the wrong
    // value.
    {
      absl::MutexLock lock(mu_);
      auto it = cache_.find(*gpu_key);
      if (it != cache_.end()) {
        return CliqueIds(it->second);
      }
    }

    CliqueId clique_id;
    if (process_id_ == device_to_process_.at(gpu_key->devices().front())) {
      ASSIGN_OR_RETURN(clique_id, mori_collectives.CreateUniqueCliqueId());
      RETURN_IF_ERROR(
          kv_store_->Set(gpu_key->ToString(), clique_id.ToString()));
    } else {
      ASSIGN_OR_RETURN(std::string id_str,
                       kv_store_->Get(gpu_key->ToString(), absl::Minutes(10)));
      clique_id = CliqueId(id_str);
    }

    absl::MutexLock lock(mu_);
    auto result = cache_.emplace(*gpu_key, std::move(clique_id));
    TF_RET_CHECK(result.second) << "Unique ID already in cache.";
    return CliqueIds(result.first->second);
  }

 private:
  ProcessId process_id_;
  const absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process_;
  const std::shared_ptr<KeyValueStoreInterface> kv_store_;

  absl::Mutex mu_;
  absl::flat_hash_map<gpu::GpuCliqueKey, CliqueId> cache_ ABSL_GUARDED_BY(mu_);
};
}  // namespace

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

absl::Status MoriCollectives::InitPe(int32_t rank, int32_t nranks,
                                     const CliqueId& clique_id,
                                     se::StreamExecutor* executor) {
  // ShmemInitAttr keys the per-device MORI state off the calling thread's
  // active HIP device, so we must activate `executor`'s context here.
  auto activate_context = executor->Activate();
  ASSIGN_OR_RETURN(auto uid, AsMoriUniqueId(clique_id));
  shmem::mori_shmem_init_attr_t init_attr;
  XLA_MORI_RETURN_IF_ERROR(
      shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &init_attr));
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
    // ShmemInitAttr is idempotent, but the initialized_ gate avoids redundant
    // uid setup.
    auto activate_context = device->stream_executor()->Activate();
    if (!initialized_) {
      RETURN_IF_ERROR(InitPe(ranks[i].rank.value(), clique_key.num_devices(),
                             clique_ids->at(0), device->stream_executor()));
    }

    // Map each collective rank to its global MORI PE. In the single-process
    // eager-init path the MORI PE equals the global device ordinal, which is
    // exactly clique_key.devices()[rank]. Passing this (rather than relying on
    // ShmemNPes()/ShmemMyPe(), which describe the whole local clique) lets the
    // collective run correctly over an arbitrary device subset.
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
  RETURN_IF_ERROR(status);
  initialized_ = true;
  return comms;
}

absl::StatusOr<GpuCollectives::CliqueIdCallback>
MoriCollectives::InitializeTopology(const Topology& topology) {
  VLOG(1) << "InitializeTopology: num_processes=" << topology.num_processes
          << " device_count_per_process=" << topology.device_count_per_process
          << " kv_store=" << (topology.kv_store != nullptr);

  if (topology.num_processes > 1) {
    // Multi-process eager init is not implemented yet; fall back to the lazy
    // per-clique initialization performed in CreateCommunicatorsWithCancel
    // (which keeps the previous hipMalloc + ShmemSymmetricRegister behavior).
    auto mori_id_store = std::make_shared<MoriIdStore>(
        topology.process_id, topology.device_to_process,
        std::move(topology.kv_store));
    return [mori_id_store, this](const CliqueKey& key) {
      return mori_id_store->GetCliqueIds(key, *this);
    };
  }

  // Single process: eagerly initialize MORI for every local device as a single
  // global clique so that collective-memory allocations can use MORI's static
  // heap (ShmemMalloc) before any executable runs.
  const int32_t nranks =
      static_cast<int32_t>(topology.device_count_per_process);
  if (nranks <= 0) {
    return nullptr;
  }

  ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName("ROCM"));

  // All local PEs share one unique id and must initialize concurrently, so
  // ShmemInitAttr's bootstrap collective can complete.
  ASSIGN_OR_RETURN(CliqueId clique_id, CreateUniqueCliqueId());

  absl::Status status;
  absl::once_flag once;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "MoriEagerInit", nranks);
    for (int32_t rank = 0; rank < nranks; ++rank) {
      pool.Schedule([&, rank]() {
        auto executor_or = platform->ExecutorForDevice(rank);
        if (!executor_or.ok()) {
          absl::call_once(once, [&] { status = executor_or.status(); });
          return;
        }
        if (auto s = InitPe(rank, nranks, clique_id, *executor_or); !s.ok()) {
          absl::call_once(once, [&] { status = s; });
        }
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  RETURN_IF_ERROR(status);

  initialized_ = true;
  VLOG(1) << "Eagerly initialized MORI for " << nranks << " local devices";
  return nullptr;
}

}  // namespace xla::gpu

// MoriCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("ROCM", "mori", -100,
                         std::make_unique<xla::gpu::MoriCollectives>());
