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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/mock_gpu_collectives.h"
#include "xla/backends/gpu/collectives/mock_gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::NiceMock;
using ::testing::Return;

namespace se = ::stream_executor;

// Blocks inside CreateCommunicatorsWithCancel or SplitCommunicatorsWithCancel
// until the test calls Release, then reports whether that call's token was
// cancelled. MockGpuCollectives has no place to keep this wait state.
struct PendingClique {
  absl::Notification entered;
  absl::Notification release;
  mutable absl::Mutex mu;
  std::shared_ptr<CancellationToken> cancel ABSL_GUARDED_BY(mu);

  bool WaitUntilEntered(absl::Duration timeout) {
    return entered.WaitForNotificationWithTimeout(timeout);
  }

  std::shared_ptr<CancellationToken> token() const {
    absl::MutexLock lock(mu);
    return cancel;
  }

  void Release() {
    if (!release.HasBeenNotified()) {
      release.Notify();
    }
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> Block(
      std::shared_ptr<CancellationToken> cancel_token, absl::string_view what) {
    {
      absl::MutexLock lock(mu);
      cancel = cancel_token;
    }
    entered.Notify();
    release.WaitForNotification();
    if (cancel_token->IsCancelled()) {
      return absl::CancelledError(absl::StrCat(what, " cancelled"));
    }
    return absl::InternalError(
        absl::StrCat(what, " was not cancelled by task failure"));
  }
};

// Parent communicators a later split can abort. The counter outlives the
// clique: a later abort destroys the communicator after Abort().
std::vector<std::unique_ptr<Communicator>> MakeParentComms(
    size_t n, std::atomic<int>* abort_count, se::StreamExecutor* executor) {
  std::vector<std::unique_ptr<Communicator>> comms;
  comms.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto comm = std::make_unique<NiceMock<MockGpuCommunicator>>();
    ON_CALL(*comm, Abort).WillByDefault([abort_count] {
      abort_count->fetch_add(1);
      return absl::OkStatus();
    });
    ON_CALL(*comm, stream_executor).WillByDefault(Return(executor));
    comms.push_back(std::move(comm));
  }
  return comms;
}

coordination::TaskInfo MakeConnectedTask(int task_id, uint64_t incarnation) {
  coordination::TaskInfo info;
  info.set_task_id(task_id);
  info.set_state(coordination::TaskState::CONNECTED);
  info.set_incarnation(incarnation);
  return info;
}

void ResetProcessTaskState() {
  std::vector<coordination::TaskInfo> empty;
  (void)UpdateGlobalProcessInfo(absl::MakeSpan(empty));
}

// A task failure during CreateCommunicatorsWithCancel must cancel the token
// stored in pending_cliques. LockableGpuClique does not exist yet. No GPU and
// no NCCL are involved.
TEST(PendingCliqueInitTest, TaskFailureCancelsInProgressInit) {
  PendingClique pending;
  NiceMock<MockGpuCollectives> collectives;
  ON_CALL(collectives, CreateCommunicatorsWithCancel)
      .WillByDefault([&](const CliqueKey&, const std::optional<CliqueIds>&,
                         absl::Span<const Collectives::DeviceRank>,
                         const Collectives::Config&,
                         std::shared_ptr<CancellationToken> cancel) {
        return pending.Block(std::move(cancel), "pending clique init");
      });
  auto cleanup = absl::MakeCleanup([&] {
    pending.Release();
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  constexpr uint64_t kIncarnation = 10;
  std::vector<coordination::TaskInfo> infos = {
      MakeConnectedTask(/*task_id=*/0, kIncarnation),
  };
  ASSERT_OK(UpdateGlobalProcessInfo(absl::MakeSpan(infos)));

  NiceMock<se::MockStreamExecutor> executor;
  ON_CALL(executor, SynchronizeAllActivity).WillByDefault(Return(true));
  ON_CALL(executor, device_ordinal).WillByDefault(Return(0));

  GpuCliqueKey key({GlobalDeviceId(0)}, /*num_local_participants=*/1,
                   CommunicationId(0),
                   /*incarnations=*/{IncarnationId(kIncarnation)});
  std::vector<std::vector<GlobalDeviceId>> groups = {{GlobalDeviceId(0)}};
  AcquiredCliquesMap acquired;

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "pending-clique-init",
                               /*num_threads=*/1);
  tsl::Executor& exec = *pool.AsExecutor();
  auto acquire_future = MakeFutureOn(exec, [&] {
    return AcquireClique(
        &collectives, &executor, RunId(0), key, groups,
        [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
          return CliqueIds(CliqueId("pending-init"));
        },
        RankId(0), acquired);
  });
  auto join_worker = absl::MakeCleanup([&] {
    pending.Release();
    (void)acquire_future.Await();
  });

  ASSERT_TRUE(pending.WaitUntilEntered(absl::Seconds(30)))
      << "Init never reached CreateCommunicatorsWithCancel.";

  ASSERT_OK(AbortTaskCliques(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated task death")));

  std::shared_ptr<CancellationToken> token = pending.token();
  ASSERT_NE(token, nullptr);
  const bool cancelled = token->IsCancelled();

  // Unblock init before asserting so a red result still exits.
  pending.Release();
  absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> acquire_result =
      acquire_future.Await();

  EXPECT_TRUE(cancelled)
      << "Task failure did not cancel the pending clique init token. "
         "Acquire finished with: "
      << acquire_result.status();
}

// A task failure during SplitCommunicatorsWithCancel must cancel the pending
// token and abort the parent once, even while parent_lock is still held. A
// later abort of the same task must not abort that parent again. No GPU and
// no NCCL are involved.
TEST(PendingCliqueInitTest, TaskFailureCancelsInProgressSplit) {
  std::atomic<int> abort_count{0};
  PendingClique pending;

  constexpr uint64_t kIncarnation = 10;
  NiceMock<se::MockStreamExecutor> executor;
  ON_CALL(executor, SynchronizeAllActivity).WillByDefault(Return(true));
  ON_CALL(executor, device_ordinal).WillByDefault(Return(0));

  const std::vector<IncarnationId> incarnations = {IncarnationId(kIncarnation)};
  GpuCliqueKey parent_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                          /*num_local_participants=*/1, CommunicationId(0),
                          incarnations);
  GpuCliqueKey child_key({GlobalDeviceId(0)}, /*num_local_participants=*/1,
                         CommunicationId(0), incarnations);
  // device_groups must cover every device of the parent, or the child is
  // created instead of split.
  std::vector<std::vector<GlobalDeviceId>> parent_devices = {
      {GlobalDeviceId(0), GlobalDeviceId(1)}};
  // Declared before the cleanups that wait on the background acquire. An
  // ASSERT_ return destroys locals in reverse order, so these must still be
  // alive while AcquireClique is running.
  AcquiredCliquesMap acquired;
  std::shared_ptr<LockableGpuClique::Lock> parent_lock;
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "pending-clique-split",
                               /*num_threads=*/1);
  tsl::Executor& exec = *pool.AsExecutor();
  Future<std::shared_ptr<LockableGpuClique::Lock>> acquire_future;
  absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> acquire_result;
  NiceMock<MockGpuCollectives> collectives;
  ON_CALL(collectives, CreateCommunicatorsWithCancel)
      .WillByDefault([&](const CliqueKey&, const std::optional<CliqueIds>&,
                         absl::Span<const Collectives::DeviceRank> ranks,
                         const Collectives::Config&,
                         std::shared_ptr<CancellationToken>) {
        return MakeParentComms(ranks.size(), &abort_count, &executor);
      });
  ON_CALL(collectives, SplitCommunicatorsWithCancel)
      .WillByDefault([&](absl::Span<const Communicator* const>, int32_t,
                         absl::Span<const RankId>, const Collectives::Config&,
                         absl::Span<const Collectives::DeviceRank>,
                         std::shared_ptr<CancellationToken> cancel) {
        return pending.Block(std::move(cancel), "pending clique split");
      });
  auto cleanup = absl::MakeCleanup([&] {
    pending.Release();
    if (acquire_future) {
      (void)acquire_future.Await();
      acquire_future = Future<std::shared_ptr<LockableGpuClique::Lock>>();
    }
    acquire_result = absl::CancelledError("cleanup");
    parent_lock.reset();
    acquired.clear();
    internal::DestroyAcquiredCliques();
    ResetProcessTaskState();
  });
  ResetProcessTaskState();

  std::vector<coordination::TaskInfo> infos = {
      MakeConnectedTask(/*task_id=*/0, kIncarnation),
  };
  ASSERT_OK(UpdateGlobalProcessInfo(absl::MakeSpan(infos)));

  ASSERT_OK_AND_ASSIGN(
      parent_lock,
      AcquireClique(
          &collectives, &executor, RunId(0), parent_key, parent_devices,
          [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
            return CliqueIds(CliqueId("pending-split-parent"));
          },
          RankId(0), acquired));
  acquired.emplace(parent_key, parent_lock);

  acquire_future = MakeFutureOn(exec, [&] {
    return AcquireClique(
        &collectives, &executor, RunId(1), child_key, parent_devices,
        [](const CliqueKey&) -> absl::StatusOr<CliqueIds> {
          return CliqueIds(CliqueId("pending-split-child"));
        },
        RankId(0), acquired);
  });
  // Same role as join_worker in TaskFailureCancelsInProgressInit: unblock and
  // wait before any later local is destroyed.
  auto join_worker = absl::MakeCleanup([&] {
    pending.Release();
    (void)acquire_future.Await();
  });

  ASSERT_TRUE(pending.WaitUntilEntered(absl::Seconds(30)))
      << "Split never reached SplitCommunicatorsWithCancel.";

  ASSERT_OK(AbortTaskCliques(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated task death")));

  std::shared_ptr<CancellationToken> token = pending.token();
  ASSERT_NE(token, nullptr);
  EXPECT_TRUE(token->IsCancelled())
      << "Task failure did not cancel the pending clique split token.";
  // The split has not returned, so the parent lock is still held. Abort must
  // still run once.
  EXPECT_EQ(abort_count.load(), 1)
      << "Parent communicator was not aborted once while its lock was held.";

  pending.Release();
  acquire_result = acquire_future.Await();
  parent_lock.reset();
  acquired.clear();

  EXPECT_THAT(acquire_result, StatusIs(absl::StatusCode::kCancelled));

  // The parent was already aborted and erased. A later abort of the same task
  // must not abort it again.
  ASSERT_OK(AbortTaskCliques(
      /*failed_task_id=*/0,
      absl::DeadlineExceededError("simulated task death")));
  EXPECT_EQ(abort_count.load(), 1)
      << "Parent communicator was aborted more than once.";
}

}  // namespace
}  // namespace xla::gpu
