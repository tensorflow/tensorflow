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

#include "xla/stream_executor/cuda/nvshmem.h"

#include <cstring>
#include <memory>
#include <string>

#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "third_party/nvshmem/nvshmem.h"   // IWYU pragma: keep
#include "third_party/nvshmem/nvshmemx.h"  // IWYU pragma: keep
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/process_id.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu::nvshmem {

// NVSHMEM environment information is stored per process in a static variable.
namespace {
struct EnvInfo {
  xla::ProcessId process_id = xla::ProcessId{-1};
  size_t num_processes = 0;
  size_t device_count_per_process = 0;
  std::weak_ptr<xla::KeyValueStoreInterface> kv_store;
  bool initialized = false;
};

static absl::NoDestructor<EnvInfo> env;
}  // namespace

void SetEnvInfo(xla::ProcessId process_id, size_t num_processes,
                size_t device_count_per_process,
                std::weak_ptr<xla::KeyValueStoreInterface> kv_store) {
  env->process_id = process_id;
  env->num_processes = num_processes;
  env->device_count_per_process = device_count_per_process;
  env->kv_store = kv_store;
}

bool IsInitialized() { return env->initialized; }

absl::Status InitializeOnce() {
  static constexpr absl::string_view kKvStoreKey = "nvshmem_global_init";

  auto init_fn = []() -> absl::Status {
    VLOG(2) << "Initializing NVSHMEM: process_id=" << env->process_id
            << ", num_processes=" << env->num_processes
            << ", device_count_per_process=" << env->device_count_per_process;

    if (env->process_id == -1) {
      LOG(FATAL)
          << "NvshmemCollectives::SetEnvInfo was not called before using "
             "NVSHMEM API";
    }
    if (env->device_count_per_process != 1) {
      LOG(FATAL) << "NVSHMEM API is only supported with one device per process";
    }
    nvshmemx_init_attr_t nvshmem_init_attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t nvshmem_id = NVSHMEMX_UNIQUEID_INITIALIZER;

    // Initialize NVSHMEM
    if (std::shared_ptr<xla::KeyValueStoreInterface> kv_store =
            env->kv_store.lock()) {
      if (env->process_id == 0) {
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

    if (nvshmemx_set_attr_uniqueid_args(env->process_id.value(),
                                        env->num_processes, &nvshmem_id,
                                        &nvshmem_init_attr) != 0) {
      return absl::InternalError("nvshmemx_set_attr_uniqueid_args failed.");
    }
    if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID,
                                   &nvshmem_init_attr) != 0) {
      return absl::InternalError("nvshmemx_hostlib_init_attr failed.");
    }

    VLOG(3) << absl::StreamFormat(
        "Initialized NVSHMEM on process %v; num_processes=%llu",
        env->process_id, env->num_processes);
    return absl::OkStatus();
  };

  static absl::once_flag once_flag;
  absl::Status status = absl::OkStatus();
  absl::call_once(once_flag, [&]() {
    status = init_fn();
    env->initialized = true;
  });
  return status;
}

void Finalize() {
  VLOG(3) << absl::StreamFormat(
      "Finilizing NVSHMEM on process %v; num_processes=%llu", env->process_id,
      env->num_processes);
  nvshmemx_hostlib_finalize();
}

}  // namespace stream_executor::gpu::nvshmem
