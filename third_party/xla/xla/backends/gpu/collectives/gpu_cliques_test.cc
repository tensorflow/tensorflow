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
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);
static constexpr GlobalDeviceId kD2(2);
static constexpr GlobalDeviceId kD3(3);
static constexpr GlobalDeviceId kD4(4);
static constexpr GlobalDeviceId kD5(5);
static constexpr GlobalDeviceId kD6(6);
static constexpr GlobalDeviceId kD7(7);

using DeviceGroups = std::vector<std::vector<GlobalDeviceId>>;

static GpuCollectives::CliqueIdCallback DefaultCliqueId() {
  return [&](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
    GpuCollectives* collectives = GpuCollectives::Default("GPU");
    ASSIGN_OR_RETURN(auto id, collectives->CreateUniqueCliqueId());
    return CliqueIds(id);
  };
}

static absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n) {
  std::vector<se::StreamExecutor*> executors(n);
  for (size_t d = 0; d < n; ++d) {
    ASSIGN_OR_RETURN(executors[d], platform->ExecutorForDevice(d));
  }
  return executors;
}

// Acquire cliques for all executors using a specific collectives
// implementation.
static std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>>
AcquireCliquesWithCollectives(
    GpuCollectives* collectives, tsl::Executor& exec,
    absl::Span<se::StreamExecutor* const> executors, const GpuCliqueKey& clique,
    DeviceGroups device_groups,
    absl::Span<const AcquiredCliquesMap> acquired_cliques) {
  CHECK_EQ(executors.size(), acquired_cliques.size());

  RunId run_id(0);

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

// Acquire cliques for all executors using the default collectives.
static std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>>
AcquireCliques(tsl::Executor& exec,
               absl::Span<se::StreamExecutor* const> executors,
               const GpuCliqueKey& clique, DeviceGroups device_groups,
               absl::Span<const AcquiredCliquesMap> acquired_cliques) {
  GpuCollectives* collectives = GpuCollectives::Default("GPU");
  return AcquireCliquesWithCollectives(collectives, exec, executors, clique,
                                       device_groups, acquired_cliques);
}

// Wait for completion of all futures and collect cliques.
static absl::StatusOr<std::vector<std::shared_ptr<LockableGpuClique::Lock>>>
WaitCliques(std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>> fs) {
  std::vector<std::shared_ptr<LockableGpuClique::Lock>> cliques(fs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    ASSIGN_OR_RETURN(cliques[i], fs[i].Await());
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
  DeviceGroups group01 = {{kD0, kD1}};

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

  DeviceGroups group0123 = {{kD0, kD1, kD2, kD3}};
  DeviceGroups group01_23 = {{kD0, kD1}, {kD2, kD3}};

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

  DeviceGroups group01 = {{kD0, kD1}};
  DeviceGroups group0123 = {{kD0, kD1, kD2, kD3}};
  DeviceGroups group01_23 = {{kD0, kD1}, {kD2, kD3}};

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

  DeviceGroups group01 = {{kD0, kD1}};
  DeviceGroups group0123 = {{kD0, kD1, kD2, kD3}};
  DeviceGroups group01_23 = {{kD0, kD1}, {kD2, kD3}};

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

// Verifies that when sub-cliques were split from a larger parent (8-device),
// re-acquiring them with a smaller split_from (4-device) correctly identifies
// the parent as a superset and reuses the existing cliques without unnecessary
// abandon.
TEST(GpuCliquesTest, ParentSupersetSkipsAbandon) {
  auto cleanup = absl::MakeCleanup([] { internal::DestroyAcquiredCliques(); });

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 8) {
    GTEST_SKIP() << "Test requires at least 8 GPUs";
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 8);
  tsl::Executor& exec = *pool.AsExecutor();

  GpuCliqueKey key01234567({kD0, kD1, kD2, kD3, kD4, kD5, kD6, kD7}, 8);
  GpuCliqueKey key0123({kD0, kD1, kD2, kD3}, 4);
  GpuCliqueKey key4567({kD4, kD5, kD6, kD7}, 4);
  GpuCliqueKey key01({kD0, kD1}, 2);
  GpuCliqueKey key23({kD2, kD3}, 2);
  GpuCliqueKey key45({kD4, kD5}, 2);
  GpuCliqueKey key67({kD6, kD7}, 2);

  DeviceGroups group_all = {{kD0, kD1, kD2, kD3, kD4, kD5, kD6, kD7}};
  DeviceGroups group_0123_4567 = {{kD0, kD1, kD2, kD3}, {kD4, kD5, kD6, kD7}};
  DeviceGroups group_01_23 = {{kD0, kD1}, {kD2, kD3}};
  DeviceGroups group_01_23_45_67 = {
      {kD0, kD1}, {kD2, kD3}, {kD4, kD5}, {kD6, kD7}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors_vec,
                       CreateExecutors(platform, 8));
  absl::Span<se::StreamExecutor*> executors(executors_vec);

  std::vector<AcquiredCliquesMap> acquired_cliques_vec(8);
  absl::Span<AcquiredCliquesMap> acquired_cliques(acquired_cliques_vec);

  // Step 1: Create parent [0,1,2,3,4,5,6,7].
  {
    auto futures = AcquireCliques(exec, executors, key01234567, group_all,
                                  acquired_cliques);
    ASSERT_OK_AND_ASSIGN(auto parent_locks, WaitCliques(std::move(futures)));
    for (size_t i = 0; i < 8; ++i) {
      acquired_cliques.at(i).emplace(key01234567, parent_locks.at(i));
    }

    // Step 2: Split [0,1],[2,3],[4,5],[6,7] from [0..7].
    // All 8 ranks participate in ncclCommSplit. Each 2-device sub-clique
    // gets parent_=[0,1,2,3,4,5,6,7].
    auto f01 =
        AcquireCliques(exec, executors.subspan(0, 2), key01, group_01_23_45_67,
                       acquired_cliques.subspan(0, 2));
    auto f23 =
        AcquireCliques(exec, executors.subspan(2, 2), key23, group_01_23_45_67,
                       acquired_cliques.subspan(2, 2));
    auto f45 =
        AcquireCliques(exec, executors.subspan(4, 2), key45, group_01_23_45_67,
                       acquired_cliques.subspan(4, 2));
    auto f67 =
        AcquireCliques(exec, executors.subspan(6, 2), key67, group_01_23_45_67,
                       acquired_cliques.subspan(6, 2));
    ASSERT_OK_AND_ASSIGN(auto c01, WaitCliques(std::move(f01)));
    ASSERT_OK_AND_ASSIGN(auto c23, WaitCliques(std::move(f23)));
    ASSERT_OK_AND_ASSIGN(auto c45, WaitCliques(std::move(f45)));
    ASSERT_OK_AND_ASSIGN(auto c67, WaitCliques(std::move(f67)));

    // Step 3: Also split [0,1,2,3] and [4,5,6,7] from [0..7] — we need
    // [0,1,2,3] as a lockable clique to serve as split_from in step 5.
    auto f0123 = AcquireCliques(exec, executors.first(4), key0123,
                                group_0123_4567, acquired_cliques.first(4));
    auto f4567 = AcquireCliques(exec, executors.last(4), key4567,
                                group_0123_4567, acquired_cliques.last(4));
    ASSERT_OK_AND_ASSIGN(auto c0123, WaitCliques(std::move(f0123)));
    ASSERT_OK_AND_ASSIGN(auto c4567, WaitCliques(std::move(f4567)));
  }

  // All locks released. Cliques persist in global state:
  //   [0,1], [2,3], [4,5], [6,7] all have parent_=[0..7]
  //   [0,1,2,3], [4,5,6,7] also exist

  // Step 4: Acquire [0,1,2,3] with no split_from (for use in acquired_cliques).
  std::vector<std::shared_ptr<LockableGpuClique::Lock>> lock0123;
  {
    absl::c_for_each(acquired_cliques_vec, [](auto& ac) { ac.clear(); });
    auto futures = AcquireCliques(exec, executors.first(4), key0123,
                                  group_0123_4567, acquired_cliques.first(4));
    ASSERT_OK_AND_ASSIGN(lock0123, WaitCliques(std::move(futures)));
  }

  // Step 5: Re-acquire [0,1] and [2,3] with split_from=[0,1,2,3].
  //
  // These sub-cliques have parent_=[0..7] (8-device, from step 2).
  // The split_from candidate is [0,1,2,3] (4-device). Since [0..7] is a
  // strict superset of [0,1,2,3], IsParentSupersetOf returns true and
  // the existing cliques are reused without unnecessary abandon.
  {
    absl::c_for_each(acquired_cliques_vec, [](auto& ac) { ac.clear(); });
    for (size_t i = 0; i < 4; ++i) {
      acquired_cliques.at(i).emplace(key0123, lock0123.at(i));
    }

    auto f01 = AcquireCliques(exec, executors.subspan(0, 2), key01, group_01_23,
                              acquired_cliques.subspan(0, 2));
    auto f23 = AcquireCliques(exec, executors.subspan(2, 2), key23, group_01_23,
                              acquired_cliques.subspan(2, 2));
    ASSERT_OK_AND_ASSIGN(auto c01, WaitCliques(std::move(f01)));
    ASSERT_OK_AND_ASSIGN(auto c23, WaitCliques(std::move(f23)));
  }
}

// Verifies that cliques acquired with different collectives implementations
// (e.g. NCCL vs loopback) are cached separately and produce different
// communicators for the same clique key.
TEST(GpuCliquesTest, DifferentCollectivesProduceDifferentCliques) {
  auto cleanup = absl::MakeCleanup([] { internal::DestroyAcquiredCliques(); });

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  GpuCliqueKey key01({kD0, kD1}, 2);
  DeviceGroups group01 = {{kD0, kD1}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));
  std::vector<AcquiredCliquesMap> acquired_cliques(2);

  // Acquire clique with the default collectives.
  GpuCollectives* default_collectives = GpuCollectives::Default("GPU");
  std::vector<std::shared_ptr<LockableGpuClique::Lock>> default_cliques;
  {
    auto futures = AcquireCliquesWithCollectives(
        default_collectives, exec, executors, key01, group01, acquired_cliques);
    ASSERT_OK_AND_ASSIGN(default_cliques, WaitCliques(std::move(futures)));
  }

  // Acquire clique with loopback collectives for the same key.
  ASSERT_OK_AND_ASSIGN(Collectives * loopback_base,
                       CollectivesRegistry::Get("GPU", "loopback"));
  auto* loopback = absl::down_cast<GpuCollectives*>(loopback_base);

  std::vector<std::shared_ptr<LockableGpuClique::Lock>> loopback_cliques;
  {
    auto futures = AcquireCliquesWithCollectives(
        loopback, exec, executors, key01, group01, acquired_cliques);
    ASSERT_OK_AND_ASSIGN(loopback_cliques, WaitCliques(std::move(futures)));
  }

  // The communicators should be different objects — the cache must not conflate
  // default and loopback cliques for the same GpuCliqueKey.
  Communicator* default_comm = *(*default_cliques[0])->comm(RankId(0));
  Communicator* loopback_comm = *(*loopback_cliques[0])->comm(RankId(0));
  EXPECT_NE(default_comm, loopback_comm);
}

}  // namespace xla::gpu
