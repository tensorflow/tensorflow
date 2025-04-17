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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "third_party/nvshmem/nvshmem.h"   // IWYU pragma: keep
#include "third_party/nvshmem/nvshmemx.h"  // IWYU pragma: keep
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

namespace xla::gpu {

NvshmemCollectives::~NvshmemCollectives() {
  if (initialized_) Finalize();
}

NvshmemCollectives* NvshmemCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Get("gpu", "nvshmem");
  CHECK_OK(collectives) << "Failed to get NVSHMEM collectives";  // Crash OK

  if (auto* nvshmem_collectives =
          tsl::down_cast<NvshmemCollectives*>(*collectives)) {
    return nvshmem_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for NVSHMEM";
}

absl::Status NvshmemCollectives::InitializeTopology(Topology topology) {
  SetEnvInfo(topology.node_id, topology.num_nodes,
             topology.device_count_per_process, topology.kv_store);
  return absl::OkStatus();
}

void NvshmemCollectives::SetEnvInfo(
    int process_id, size_t num_processes, size_t device_count_per_process,
    std::weak_ptr<KeyValueStoreInterface> kv_store) {
  process_id_ = process_id;
  num_processes_ = num_processes;
  device_count_per_process_ = device_count_per_process;
  kv_store_ = kv_store;
}

absl::Status NvshmemCollectives::InitializeOnce() {
  auto init_fn = [this]() -> absl::Status {
    if (process_id_ == -1) {
      LOG(FATAL)
          << "NvshmemCollectives::SetEnvInfo was not called before using "
             "NVSHMEM API";
    }
    if (device_count_per_process_ != 1) {
      LOG(FATAL) << "NVSHMEM API is only supported with one device per process";
    }
    nvshmemx_init_attr_t nvshmem_init_attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t nvshmem_id = NVSHMEMX_UNIQUEID_INITIALIZER;

    // Initialize NVSHMEM
    if (std::shared_ptr<KeyValueStoreInterface> kv_store = kv_store_.lock()) {
      if (process_id_ == 0) {
        if (nvshmemx_get_uniqueid(&nvshmem_id) != 0) {
          return absl::InternalError("nvshmemx_get_uniqueid failed.");
        }
        char buf[sizeof(nvshmemx_uniqueid_t)];
        std::memcpy(buf, &nvshmem_id, sizeof(nvshmemx_uniqueid_t));
        absl::string_view nvshmem_id_str{buf, sizeof(buf)};
        TF_RETURN_IF_ERROR(kv_store->Set(kKvStoreKey, nvshmem_id_str));
      } else {
        TF_ASSIGN_OR_RETURN(std::string id_str,
                            kv_store->Get(kKvStoreKey, absl::Minutes(10)));
        CHECK(id_str.size() >= sizeof(nvshmemx_uniqueid_t));
        std::memcpy(&nvshmem_id, id_str.data(), sizeof(nvshmemx_uniqueid_t));
      }
    } else {
      return absl::InternalError(
          "KV store is not available for nvshmem initialization.");
    }

    if (nvshmemx_set_attr_uniqueid_args(process_id_, num_processes_,
                                        &nvshmem_id, &nvshmem_init_attr) != 0) {
      return absl::InternalError("nvshmemx_set_attr_uniqueid_args failed.");
    }
    if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID,
                                   &nvshmem_init_attr) != 0) {
      return absl::InternalError("nvshmemx_hostlib_init_attr failed.");
    }

    VLOG(3) << absl::StreamFormat(
        "Initialized NVSHMEM on process %d; num_processes=%llu", process_id_,
        num_processes_);
    return absl::OkStatus();
  };

  static absl::once_flag once_flag;
  absl::Status status = absl::OkStatus();
  absl::call_once(once_flag, [&]() {
    status = init_fn();
    initialized_ = true;
  });
  return status;
}

void NvshmemCollectives::Finalize() {
  VLOG(3) << absl::StreamFormat(
      "Finilizing NVSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  nvshmemx_hostlib_finalize();
}

absl::StatusOr<void*> NvshmemCollectives::Allocate(uint64_t bytes) {
  TF_RETURN_IF_ERROR(InitializeOnce());
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
  TF_RETURN_IF_ERROR(InitializeOnce());
  VLOG(3) << absl::StreamFormat("Start de-allocation for NVSHMEM buffer: %p",
                                buffer);
  nvshmem_free(buffer);
  return absl::OkStatus();
}

}  // namespace xla::gpu

// NvshmemCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("gpu", "nvshmem", -100,
                         std::make_unique<xla::gpu::NvshmemCollectives>());
