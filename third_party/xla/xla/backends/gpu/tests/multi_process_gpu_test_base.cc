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

#include "xla/backends/gpu/tests/multi_process_gpu_test_base.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"

namespace xla {
namespace {

inline constexpr size_t kDefaultGpuSystemMemoryBytes = 128ULL * 1024 * 1024;

struct WorkerState {
  absl::Mutex mu;
  int worker_id = -1;
  int num_workers = 2;
  int gpus_per_worker = 1;
  std::string coordinator_address;
  std::shared_ptr<DistributedRuntimeClient> distributed_client;
  std::shared_ptr<KeyValueStoreInterface> kv_store;
  std::unique_ptr<PjRtClient> shared_client;
  std::atomic<int> custom_client_counter{0};
  std::atomic<int> next_test_invocation_id{0};
  std::atomic<int> current_test_invocation_id{0};
};

WorkerState& GetWorkerState() {
  static absl::NoDestructor<WorkerState> state;
  return *state;
}

}  // namespace

void ConfigureMultiProcessGpuParent(int num_workers, int gpus_per_worker) {
  WorkerState& state = GetWorkerState();
  absl::MutexLock lock(state.mu);
  state.worker_id = -1;
  state.num_workers = num_workers;
  state.gpus_per_worker = gpus_per_worker;
}

void InitializeMultiProcessGpuWorker(int worker_id, int num_workers,
                                     int gpus_per_worker,
                                     absl::string_view coordinator_address) {
  WorkerState& state = GetWorkerState();
  absl::MutexLock lock(state.mu);
  state.worker_id = worker_id;
  state.num_workers = num_workers;
  state.gpus_per_worker = gpus_per_worker;
  state.coordinator_address = std::string(coordinator_address);

  DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = worker_id;
  distributed_options.init_timeout = absl::Seconds(120);
  distributed_options.shutdown_timeout = absl::Seconds(10);
  state.distributed_client = GetDistributedRuntimeClient(
      state.coordinator_address, distributed_options);
  CHECK_OK(state.distributed_client->Connect())
      << "rank_" << worker_id << ": failed to connect DistributedRuntimeClient";
  state.kv_store = GetDistributedKeyValueStore(state.distributed_client,
                                               /*key_prefix=*/"shared_gpu:");
}

void ShutDownMultiProcessGpuWorker() {
  WorkerState& state = GetWorkerState();
  absl::MutexLock lock(state.mu);
  state.shared_client.reset();
  state.kv_store.reset();
  if (state.distributed_client != nullptr) {
    state.distributed_client->Shutdown().IgnoreError();
    state.distributed_client.reset();
  }
}

void MultiProcessGpuTest::SetUp() {
  WorkerState& state = GetWorkerState();
  state.current_test_invocation_id.store(
      state.next_test_invocation_id.fetch_add(1));
  if (rank() < 0) {
    GTEST_SKIP() << "Test requires at least " << num_ranks() * gpus_per_worker()
                 << " GPU devices.";
  }
  ASSERT_OK(distributed_client()->WaitAtBarrier(
      absl::StrCat(CurrentTestId(), "_start"), absl::Seconds(120),
      std::nullopt));
}

void MultiProcessGpuTest::TearDown() {
  if (rank() >= 0 && distributed_client() != nullptr) {
    const bool has_failure = HasFailure();
    const bool has_fatal_failure = HasFatalFailure();
    const std::string fail_key = absl::StrCat(CurrentTestId(), "_failed");
    if (has_failure) {
      distributed_client()
          ->KeyValueSet(fail_key, absl::StrCat(rank()),
                        /*allow_overwrite=*/true)
          .IgnoreError();
    }
    absl::Status barrier_status = distributed_client()->WaitAtBarrier(
        absl::StrCat(CurrentTestId(), "_end"),
        has_fatal_failure ? absl::Seconds(5) : absl::Seconds(120),
        std::nullopt);
    if (!barrier_status.ok() && has_fatal_failure) {
      LOG(ERROR) << "rank_" << rank()
                 << ": TearDown barrier timed out after fatal test failure ("
                 << barrier_status
                 << "); exiting immediately so orchestrator can terminate "
                    "stuck peers.";
      std::_Exit(1);
    }
    EXPECT_OK(barrier_status);
    if (!has_failure && distributed_client()->KeyValueTryGet(fail_key).ok()) {
      ADD_FAILURE() << "Peer rank failed in multi-process test.";
    }
  }
}

int MultiProcessGpuTest::rank() const { return GetWorkerState().worker_id; }

int MultiProcessGpuTest::num_ranks() const {
  return GetWorkerState().num_workers;
}

int MultiProcessGpuTest::gpus_per_worker() const {
  return GetWorkerState().gpus_per_worker;
}

std::shared_ptr<DistributedRuntimeClient>
MultiProcessGpuTest::distributed_client() const {
  return GetWorkerState().distributed_client;
}

std::shared_ptr<KeyValueStoreInterface> MultiProcessGpuTest::kv_store() const {
  return GetWorkerState().kv_store;
}

GpuClientOptions MultiProcessGpuTest::DefaultGpuClientOptions() const {
  GpuClientOptions options;
  options.node_id = rank();
  options.num_nodes = num_ranks();
  std::set<int> allowed_devices;
  for (int i = 0; i < gpus_per_worker(); ++i) {
    allowed_devices.insert(rank() * gpus_per_worker() + i);
  }
  options.allowed_devices = std::move(allowed_devices);
  options.distributed_client = distributed_client();
  options.kv_store = kv_store();
  options.allocator_config.kind = GpuAllocatorConfig::Kind::kBFC;
  options.allocator_config.gpu_system_memory_size =
      kDefaultGpuSystemMemoryBytes;
  options.allocator_config.preallocate = false;
  options.allocator_config.collective_memory_size = 0;
  return options;
}

PjRtClient* MultiProcessGpuTest::client() const {
  WorkerState& state = GetWorkerState();
  absl::MutexLock lock(state.mu);
  if (state.shared_client == nullptr) {
    absl::StatusOr<std::unique_ptr<PjRtClient>> shared_client =
        GetXlaPjrtGpuClient(DefaultGpuClientOptions());
    CHECK_OK(shared_client.status())
        << "rank_" << state.worker_id << ": failed to create shared PjRtClient";
    state.shared_client = *std::move(shared_client);
  }
  return state.shared_client.get();
}

absl::StatusOr<std::unique_ptr<PjRtClient>>
MultiProcessGpuTest::CreateCustomPjRtClient(GpuClientOptions options) const {
  WorkerState& state = GetWorkerState();
  options.node_id = rank();
  options.num_nodes = num_ranks();
  if (!options.allowed_devices.has_value()) {
    std::set<int> allowed_devices;
    for (int i = 0; i < gpus_per_worker(); ++i) {
      allowed_devices.insert(rank() * gpus_per_worker() + i);
    }
    options.allowed_devices = std::move(allowed_devices);
  }
  if (options.distributed_client == nullptr) {
    options.distributed_client = distributed_client();
  }
  if (options.kv_store == nullptr) {
    int custom_id = state.custom_client_counter.fetch_add(1);
    options.kv_store = CreateScopedKeyValueStore(
        absl::StrFormat("custom_client_%d", custom_id));
  }
  if (!options.allocator_config.gpu_system_memory_size.has_value()) {
    options.allocator_config.gpu_system_memory_size =
        kDefaultGpuSystemMemoryBytes;
    options.allocator_config.preallocate = false;
  }
  return GetXlaPjrtGpuClient(std::move(options));
}

std::shared_ptr<KeyValueStoreInterface>
MultiProcessGpuTest::CreateScopedKeyValueStore(absl::string_view scope) const {
  return GetDistributedKeyValueStore(
      distributed_client(),
      absl::StrFormat("test_%s_%s:", CurrentTestId(), scope));
}

std::string MultiProcessGpuTest::CurrentTestId() const {
  const int run_id = GetWorkerState().current_test_invocation_id.load();
  const ::testing::TestInfo* test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  if (test_info == nullptr) {
    return absl::StrFormat("unknown_test_run_%d", run_id);
  }
  return absl::StrFormat("%s.%s_run_%d", test_info->test_suite_name(),
                         test_info->name(), run_id);
}

}  // namespace xla
