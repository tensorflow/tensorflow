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

#ifndef XLA_BACKENDS_GPU_TESTS_MULTI_PROCESS_GPU_TEST_BASE_H_
#define XLA_BACKENDS_GPU_TESTS_MULTI_PROCESS_GPU_TEST_BASE_H_

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"

namespace xla {

// Internal hooks called by multi_process_gpu_test_main.
void ConfigureMultiProcessGpuParent(int num_workers, int gpus_per_worker);
void InitializeMultiProcessGpuWorker(int worker_id, int num_workers,
                                     int gpus_per_worker,
                                     absl::string_view coordinator_address);
void ShutDownMultiProcessGpuWorker();

// Base fixture for multi-process GPU tests.
//
// When linked with `multi_process_gpu_test_main`, the test binary acts as an
// orchestrator that starts a `DistributedRuntimeService` and spawns N worker
// subprocesses (one per GPU rank by default). Each worker executes the same
// GoogleTest suite in lockstep, synchronized via barriers in `SetUp()` and
// `TearDown()`.
class MultiProcessGpuTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // This worker's rank in [0, num_ranks()). Returns -1 if running in the
  // parent process (e.g. when skipping due to insufficient GPUs).
  int rank() const;
  int node_id() const { return rank(); }

  // Total number of worker processes.
  int num_ranks() const;
  int num_nodes() const { return num_ranks(); }

  // Number of GPUs assigned to each worker process.
  int gpus_per_worker() const;

  // Returns a process-wide shared PjRtClient initialized for this worker's
  // rank and device assignment. Lazily created on first call.
  PjRtClient* client() const;

  // Returns the process-wide DistributedRuntimeClient connected to the parent
  // orchestrator's coordination service.
  std::shared_ptr<DistributedRuntimeClient> distributed_client() const;

  // Returns the distributed KeyValueStore used by the shared PjRtClient.
  std::shared_ptr<KeyValueStoreInterface> kv_store() const;

  // Returns default GpuClientOptions configured for this worker rank.
  GpuClientOptions DefaultGpuClientOptions() const;

  // Creates a new PjRtClient with custom options, automatically populating
  // `node_id`, `num_nodes`, `allowed_devices`, `distributed_client`, and an
  // isolated `kv_store` if not already set on `options`.
  absl::StatusOr<std::unique_ptr<PjRtClient>> CreateCustomPjRtClient(
      GpuClientOptions options = GpuClientOptions()) const;

  // Creates a distributed KeyValueStore scoped to the current test and `scope`.
  std::shared_ptr<KeyValueStoreInterface> CreateScopedKeyValueStore(
      absl::string_view scope) const;

  // Returns a unique identifier string for the currently running test case.
  std::string CurrentTestId() const;
};

template <typename T>
class MultiProcessGpuTestWithParam : public MultiProcessGpuTest,
                                     public ::testing::WithParamInterface<T> {};

}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TESTS_MULTI_PROCESS_GPU_TEST_BASE_H_
