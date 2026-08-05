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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);

coordination::TaskInfo MakeConnectedTask(int task_id, uint64_t incarnation) {
  coordination::TaskInfo info;
  info.set_task_id(task_id);
  info.set_state(coordination::TaskState::CONNECTED);
  info.set_incarnation(incarnation);
  return info;
}

// Clears process-local task state so tests do not leak across cases.
// UpdateGlobalProcessInfo may return FailedPrecondition when shrinking state;
// the assignment still replaces task_state_infos.
void ResetProcessTaskState() {
  std::vector<coordination::TaskInfo> empty;
  (void)UpdateGlobalProcessInfo(absl::MakeSpan(empty));
}

TEST(AbortCollectivesOnTaskFailureTest, EmptyTaskStateIsNoOp) {
  auto cleanup = absl::MakeCleanup([] {
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  // With no prior UpdateGlobalProcessInfo, AbortCollectivesOnTaskFailure must
  // be a no-op (same as HangWatchdog firing before task state is published).
  EXPECT_OK(AbortCollectivesOnTaskFailure(
      /*failed_task_id=*/0, absl::DeadlineExceededError("timeout")));
}

TEST(AbortCollectivesOnTaskFailureTest, UnknownTaskReturnsNotFound) {
  auto cleanup = absl::MakeCleanup([] {
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  std::vector<coordination::TaskInfo> infos = {
      MakeConnectedTask(/*task_id=*/0, /*incarnation=*/10),
      MakeConnectedTask(/*task_id=*/1, /*incarnation=*/11),
  };
  ASSERT_OK(UpdateGlobalProcessInfo(absl::MakeSpan(infos)));

  EXPECT_THAT(
      AbortCollectivesOnTaskFailure(
          /*failed_task_id=*/99, absl::DeadlineExceededError("timeout")),
      StatusIs(absl::StatusCode::kNotFound));
}

// After AbortCollectivesOnTaskFailure, cliques that include the failed task
// incarnation must be treated as stale so future collective acquisition /
// progress checks fail instead of hanging forever.
TEST(AbortCollectivesOnTaskFailureTest, MarksFailedTaskAndMakesCliqueKeyStale) {
  auto cleanup = absl::MakeCleanup([] {
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  std::vector<coordination::TaskInfo> infos = {
      MakeConnectedTask(/*task_id=*/0, /*incarnation=*/10),
      MakeConnectedTask(/*task_id=*/1, /*incarnation=*/11),
  };
  ASSERT_OK(UpdateGlobalProcessInfo(absl::MakeSpan(infos)));

  GpuCliqueKey key({kD0, kD1}, /*num_local_participants=*/2, CommunicationId(0),
                   /*incarnations=*/{IncarnationId(10), IncarnationId(11)});
  EXPECT_OK(CheckCliqueIsNotStale(key));

  ASSERT_OK(AbortCollectivesOnTaskFailure(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("XLA GPU execution timed out")));

  EXPECT_THAT(CheckCliqueIsNotStale(key),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("not connected")));
}

static GpuCollectives::CliqueIdCallback DefaultCliqueId() {
  return [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
    GpuCollectives* collectives = GpuCollectives::Default("GPU");
    ABSL_ASSIGN_OR_RETURN(auto id, collectives->CreateUniqueCliqueId());
    return CliqueIds(id);
  };
}

static absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n) {
  std::vector<se::StreamExecutor*> executors(n);
  for (size_t d = 0; d < n; ++d) {
    ABSL_ASSIGN_OR_RETURN(executors[d], platform->ExecutorForDevice(d));
  }
  return executors;
}

// End-to-end local unwind: acquire a live GPU clique, report a task failure
// through AbortCollectivesOnTaskFailure (the HangWatchdog handler path), and
// verify the clique is aborted/removed so collective progress cannot continue.
TEST(AbortCollectivesOnTaskFailureTest, AbortsAcquiredGpuCliqueOnTaskFailure) {
  auto cleanup = absl::MakeCleanup([] {
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));
  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  constexpr uint64_t kInc0 = 100;
  constexpr uint64_t kInc1 = 101;
  std::vector<coordination::TaskInfo> infos = {
      MakeConnectedTask(/*task_id=*/0, kInc0),
      MakeConnectedTask(/*task_id=*/1, kInc1),
  };
  ASSERT_OK(UpdateGlobalProcessInfo(absl::MakeSpan(infos)));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "abort-collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  GpuCliqueKey key(
      {kD0, kD1}, /*num_local_participants=*/2, CommunicationId(0),
      /*incarnations=*/{IncarnationId(kInc0), IncarnationId(kInc1)});
  std::vector<std::vector<GlobalDeviceId>> groups = {{kD0, kD1}};

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));
  std::vector<AcquiredCliquesMap> acquired_cliques(2);

  GpuCollectives* collectives = GpuCollectives::Default("GPU");
  std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>> futures(2);
  for (size_t i = 0; i < 2; ++i) {
    futures[i] = MakeFutureOn(exec, [=, &acquired_cliques] {
      return AcquireGpuClique(collectives, executors.at(i), RunId(0), key,
                              groups, DefaultCliqueId(), RankId(i),
                              acquired_cliques.at(i));
    });
  }

  std::vector<std::shared_ptr<LockableGpuClique::Lock>> locks(2);
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(locks[i], futures[i].Await());
  }
  EXPECT_OK(CheckCliqueIsNotStale(key));

  // Drop locks before abort so AbortCliquesWithIncarnations can take exclusive
  // access and tear the clique down (same as healthy ranks releasing after
  // cancel notification in production).
  locks.clear();
  futures.clear();
  acquired_cliques.clear();

  ASSERT_OK(AbortCollectivesOnTaskFailure(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated execution hang timeout")));

  EXPECT_THAT(CheckCliqueIsNotStale(key),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Re-acquiring the same failed-incarnation key must fail as stale. Both local
  // ranks must join AcquireGpuClique (num_local_participants=2) or rendezvous
  // hangs waiting for the missing rank.
  std::vector<AcquiredCliquesMap> reacquire_maps(2);
  std::vector<Future<std::shared_ptr<LockableGpuClique::Lock>>> reacquire(2);
  for (size_t i = 0; i < 2; ++i) {
    reacquire[i] = MakeFutureOn(exec, [=, &reacquire_maps] {
      return AcquireGpuClique(collectives, executors.at(i), RunId(1), key,
                              groups, DefaultCliqueId(), RankId(i),
                              reacquire_maps.at(i));
    });
  }
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_THAT(reacquire[i].Await().status(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

}  // namespace
}  // namespace xla::gpu
