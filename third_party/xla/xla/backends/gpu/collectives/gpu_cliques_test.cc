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

#include "xla/backends/gpu/collectives/gpu_cliques.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);
static constexpr GlobalDeviceId kD2(2);
static constexpr GlobalDeviceId kD3(3);

static GpuCollectives::CliqueIdCallback DefaultCliqueId() {
  return [&](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
    GpuCollectives* collectives = GpuCollectives::Default("GPU");
    TF_ASSIGN_OR_RETURN(auto id, collectives->CreateUniqueCliqueId());
    return CliqueIds(id);
  };
}

static absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n) {
  std::vector<se::StreamExecutor*> executors(n);
  for (size_t d = 0; d < n; ++d) {
    TF_ASSIGN_OR_RETURN(executors[d], platform->ExecutorForDevice(d));
  }
  return executors;
}

// Acquire cliques for all executors.
static std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>>
AcquireCliques(tsl::Executor& exec,
               absl::Span<se::StreamExecutor* const> executors,
               const GpuCliqueKey& clique,
               std::vector<std::vector<GlobalDeviceId>> device_groups,
               absl::Span<const AcquiredCliquesMap> acquired_cliques) {
  CHECK_EQ(executors.size(), acquired_cliques.size());

  RunId run_id(0);
  GpuCollectives* collectives = GpuCollectives::Default("GPU");

  size_t n = executors.size();
  std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>> futures(n);
  for (size_t i = 0; i < n; ++i) {
    futures[i] = MakeFutureOn(exec, [=] {
      return AcquireGpuClique(collectives, executors.at(i), run_id, clique,
                              device_groups, DefaultCliqueId(), RankId(i),
                              acquired_cliques.at(i));
    });
  }

  return futures;
}

// Wait for completion of all futures and collect cliques.
static absl::StatusOr<std::vector<std::shared_ptr<LockableGpuClique::Lock>>>
WaitCliques(std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>> fs) {
  std::vector<std::shared_ptr<LockableGpuClique::Lock>> cliques(fs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(cliques[i], fs[i].Await());
  }
  return cliques;
}

TEST(GpuCliquesTest, AcquireCliques) {
  auto cleanup = absl::MakeCleanup([] { internal::DestroyAcquiredCliques(); });

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  RunId run_id(0);
  GpuCliqueKey key01({kD0, kD1}, 2);
  std::vector<std::vector<GlobalDeviceId>> group01 = {{kD0, kD1}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  std::vector<AcquiredCliquesMap> acquired_cliques(2);

  // Acquire GPU cliques for devices [0,1].
  {
    auto futures =
        AcquireCliques(exec, executors, key01, group01, acquired_cliques);
    ASSERT_OK_AND_ASSIGN(auto cliques, WaitCliques(std::move(futures)));
  }
}

TEST(GpuCliquesTest, SplitCliques) {
  auto cleanup = absl::MakeCleanup([] { internal::DestroyAcquiredCliques(); });

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 4) {
    GTEST_SKIP() << "Test requires at least 4 GPUs";
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 4);
  tsl::Executor& exec = *pool.AsExecutor();

  RunId run_id(0);

  GpuCliqueKey key0123({kD0, kD1, kD2, kD3}, 4);
  GpuCliqueKey key01({kD0, kD1}, 2);
  GpuCliqueKey key23({kD2, kD3}, 2);

  std::vector<std::vector<GlobalDeviceId>> group0123 = {{kD0, kD1, kD2, kD3}};
  std::vector<std::vector<GlobalDeviceId>> group01_23 = {{kD0, kD1},
                                                         {kD2, kD3}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors_vec,
                       CreateExecutors(platform, 4));
  absl::Span<se::StreamExecutor*> executors(executors_vec);

  std::vector<AcquiredCliquesMap> acquired_cliques_vec(4);
  absl::Span<AcquiredCliquesMap> acquired_cliques(acquired_cliques_vec);

  {  // Acquire clique that spans all 4 devices.
    auto futures =
        AcquireCliques(exec, executors, key0123, group0123, acquired_cliques);
    ASSERT_OK_AND_ASSIGN(auto cliques, WaitCliques(std::move(futures)));

    // Put acquired cliques into acquired cliques map for all devices.
    for (size_t i = 0; i < 4; ++i) {
      acquired_cliques.at(i).emplace(key0123, cliques.at(i));
    }
  }

  {  // Acquire smaller cliques by splitting earlier acquired clique.
    auto futures0 = AcquireCliques(exec, executors.first(2), key01, group01_23,
                                   acquired_cliques.first(2));
    auto futures1 = AcquireCliques(exec, executors.last(2), key23, group01_23,
                                   acquired_cliques.last(2));

    ASSERT_OK_AND_ASSIGN(auto cliques1, WaitCliques(std::move(futures0)));
    ASSERT_OK_AND_ASSIGN(auto cliques2, WaitCliques(std::move(futures1)));
  }
}

TEST(GpuCliquesTest, SplitCliquesNoDeadlock0) {
  auto cleanup = absl::MakeCleanup([] { internal::DestroyAcquiredCliques(); });

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 4) {
    GTEST_SKIP() << "Test requires at least 4 GPUs";
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 4);
  tsl::Executor& exec = *pool.AsExecutor();

  RunId run_id(0);

  GpuCliqueKey key0123({kD0, kD1, kD2, kD3}, 4);
  GpuCliqueKey key01({kD0, kD1}, 2);
  GpuCliqueKey key23({kD2, kD3}, 2);

  std::vector<std::vector<GlobalDeviceId>> group01 = {{kD0, kD1}};
  std::vector<std::vector<GlobalDeviceId>> group0123 = {{kD0, kD1, kD2, kD3}};
  std::vector<std::vector<GlobalDeviceId>> group01_23 = {{kD0, kD1},
                                                         {kD2, kD3}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors_vec,
                       CreateExecutors(platform, 4));
  absl::Span<se::StreamExecutor*> executors(executors_vec);

  std::vector<AcquiredCliquesMap> acquired_cliques_vec(4);
  absl::Span<AcquiredCliquesMap> acquired_cliques(acquired_cliques_vec);

  {  // Acquire clique for devices [0,1].
    auto futures = AcquireCliques(exec, executors.first(2), key01, group01,
                                  acquired_cliques.first(2));
    ASSERT_OK_AND_ASSIGN(auto cliques, WaitCliques(std::move(futures)));
  }

  {  // Acquire clique for devices [0,1,2,3].
    auto futures =
        AcquireCliques(exec, executors, key0123, group0123, acquired_cliques);
    ASSERT_OK_AND_ASSIGN(auto cliques, WaitCliques(std::move(futures)));

    // Put acquired cliques into acquired cliques map for all devices.
    for (size_t i = 0; i < 4; ++i) {
      acquired_cliques.at(i).emplace(key0123, cliques.at(i));
    }
  }

  {  // Acquire cliques for devices [0,1] and [2,3].
    auto futures0 = AcquireCliques(exec, executors.first(2), key01, group01_23,
                                   acquired_cliques.first(2));
    auto futures1 = AcquireCliques(exec, executors.last(2), key23, group01_23,
                                   acquired_cliques.last(2));
    ASSERT_OK_AND_ASSIGN(auto cliques0, WaitCliques(std::move(futures0)));
    ASSERT_OK_AND_ASSIGN(auto cliques1, WaitCliques(std::move(futures1)));
  }
}

TEST(GpuCliquesTest, SplitCliquesNoDeadlock1) {
  auto cleanup = absl::MakeCleanup([] { internal::DestroyAcquiredCliques(); });

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 4) {
    GTEST_SKIP() << "Test requires at least 4 GPUs";
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 4);
  tsl::Executor& exec = *pool.AsExecutor();

  RunId run_id(0);

  GpuCliqueKey key0123({kD0, kD1, kD2, kD3}, 4);
  GpuCliqueKey key01({kD0, kD1}, 2);
  GpuCliqueKey key23({kD2, kD3}, 2);

  std::vector<std::vector<GlobalDeviceId>> group01 = {{kD0, kD1}};
  std::vector<std::vector<GlobalDeviceId>> group0123 = {{kD0, kD1, kD2, kD3}};
  std::vector<std::vector<GlobalDeviceId>> group01_23 = {{kD0, kD1},
                                                         {kD2, kD3}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors_vec,
                       CreateExecutors(platform, 4));
  absl::Span<se::StreamExecutor*> executors(executors_vec);

  std::vector<AcquiredCliquesMap> acquired_cliques_vec(4);
  absl::Span<AcquiredCliquesMap> acquired_cliques(acquired_cliques_vec);

  {  // Acquire clique for devices [0,1,2,3].
    auto futures =
        AcquireCliques(exec, executors, key0123, group0123, acquired_cliques);
    ASSERT_OK_AND_ASSIGN(auto cliques, WaitCliques(std::move(futures)));

    // Put acquired cliques into acquired cliques map for all devices.
    for (size_t i = 0; i < 4; ++i) {
      acquired_cliques.at(i).emplace(key0123, cliques.at(i));
    }
  }

  {  // Acquire clique for devices [0,1]. This will not split because device
     // group doesn't cover [0,1,2,4] devices.
    auto futures = AcquireCliques(exec, executors.first(2), key01, group01,
                                  acquired_cliques.first(2));
    ASSERT_OK_AND_ASSIGN(auto cliques, WaitCliques(std::move(futures)));
  }

  {  // Acquire cliques for devices [0,1] and [2,3]. This must split from the
    // clique [0,1,2,3] and correctly ignore clique [0,1] acquired above.
    auto futures0 = AcquireCliques(exec, executors.first(2), key01, group01_23,
                                   acquired_cliques.first(2));
    auto futures1 = AcquireCliques(exec, executors.last(2), key23, group01_23,
                                   acquired_cliques.last(2));
    ASSERT_OK_AND_ASSIGN(auto cliques0, WaitCliques(std::move(futures0)));
    ASSERT_OK_AND_ASSIGN(auto cliques1, WaitCliques(std::move(futures1)));
  }
}

}  // namespace xla::gpu
