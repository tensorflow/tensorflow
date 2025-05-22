/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/distributed_runtime/preemption/preemption_sync_manager.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_notifier.h"
#include "xla/tsl/lib/monitoring/gauge.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedTask;
using tensorflow::KeyValueEntry;

constexpr char kPreemptionNoticeKey[] = "RECEIVED_PREEMPTION_NOTICE";
constexpr char kPreemptionCounterDirKey[] = "PREEMPTION_CURRENT_COUNTER/";
constexpr char kPreemptionBarrier[] = "PREEMPTION_SYNC_BARRIER";
constexpr absl::Duration kPreemptionBarrierTimeout = absl::Minutes(3);

auto* sync_usage_metric = monitoring::Gauge<bool, 0>::New(
    "/coordination_service/preempt_manager/reached_sync_point_usage",
    "Records if preempt sync manager's ReachSyncPoint() was called at least "
    "once.");

auto* notified_metric = monitoring::Gauge<bool, 0>::New(
    "/coordination_service/preempt_manager/notified",
    "Records receipt of preemption notification.");

auto* set_sync_point_metric = monitoring::Gauge<bool, 0>::New(
    "/coordination_service/preempt_manager/set_sync_point",
    "Records that sync point is set.");

auto* reached_sync_point_metric = monitoring::Gauge<bool, 0>::New(
    "/coordination_service/preempt_manager/reached_sync_point",
    "Records that sync point is reached.");

// Only start protocol if death time is within `kProtocolDuration`, so that we
// don't synchronize too early.
// TODO(b/230630494): Make this configurable so that users can extend this to
// accommodate higher checkpoint durations.
constexpr absl::Duration kProtocolDuration = absl::Minutes(15);

}  // namespace

absl::Status PreemptionSyncManager::Initialize(
    CoordinationServiceAgent* agent) {
  return Initialize(agent, "sigterm");
}

absl::Status PreemptionSyncManager::Initialize(
    CoordinationServiceAgent* agent,
    const std::string& preemption_notifier_type) {
  TF_ASSIGN_OR_RETURN(Env * env, agent->GetEnv());
  return Initialize(agent, PreemptionNotifier::CreatePreemptionNotifier(
                               preemption_notifier_type, env));
}

absl::Status PreemptionSyncManager::Initialize(
    CoordinationServiceAgent* agent,
    std::unique_ptr<PreemptionNotifier> notifier) {
  {
    absl::MutexLock l(&mu_);
    CHECK(!shut_down_);
  }

  TF_ASSIGN_OR_RETURN(Env * env, agent->GetEnv());
  env_ = env;
  agent_ = agent;
  preemption_notifier_ = std::move(notifier);
  TF_ASSIGN_OR_RETURN(CoordinatedTask own_task, agent->GetOwnTask());
  const std::string task_name =
      absl::StrCat("/job:", own_task.job_name(), "/task:", own_task.task_id());
  current_call_counter_key_ = absl::StrCat(kPreemptionCounterDirKey, task_name);

  /* Listen for preemption notice within this task, then notify coordination
   * service when death time is within kProtocolDuration.
   */
  preemption_notifier_->WillBePreemptedAtAsync(
      [agent = agent_, task_name](absl::StatusOr<absl::Time> death_time) {
        if (!death_time.ok()) {
          // The preemption notifier invokes callback with Cancelled error when
          // its being destructed.
          if (absl::IsCancelled(death_time.status())) {
            LOG(INFO) << "Preemption sync protocol cancelled by notifier: "
                      << death_time.status()
                      << ". This is expected during program shutdown.";
          } else {
            LOG(ERROR) << "Error from preemption notifier: "
                       << death_time.status();
          }
          return;
        }
        notified_metric->GetCell()->Set(true);
        // Notify coordination service about preemption notice.
        const absl::Status s = agent->InsertKeyValue(
            kPreemptionNoticeKey, absl::FormatTime(*death_time));
        LOG(INFO) << "Notified coordination service that this task will "
                     "be preempted at "
                  << *death_time << ". absl::Status: " << s;
      });

  /* Listen for preemption notice (death time) from coordination service, which
   * triggers the sync protocol.
   */
  call_opts_ = agent_->GetKeyValueAsync(
      kPreemptionNoticeKey,
      [this, agent = agent_](absl::StatusOr<std::string> status_or_death_time) {
        if (absl::IsCancelled(status_or_death_time.status())) {
          // The agent cancels pending GetKeyValue RPCs because of shutdown,
          // so simply log and return.
          LOG(INFO) << "Cancelled call to retrieve preemption notice. This is "
                       "expected upon program shutdown.";
          return;
        }
        if (!status_or_death_time.ok()) {
          LOG(WARNING)
              << "Failed to retrieve preemption notice from "
                 "coordination service: "
              << status_or_death_time.status()
              << ". This is only expected if one of the tasks is unhealthy."
                 " Check the logs for the actual root cause.";
          // Notify other tasks to not wait at the barrier. Note:
          // CancelPreemptionBarrier() cannot be used because this may be
          // triggered after preemption sync manager has been destroyed.
          agent->CancelBarrierAsync(
              kPreemptionBarrier, [](const absl::Status& status) {
                if (!status.ok()) {
                  LOG(ERROR)
                      << "Failed to cancel preemption barrier: " << status;
                }
              });
          return;
        }
        std::string err;
        absl::Time death_time;
        if (absl::ParseTime(absl::RFC3339_full, *status_or_death_time,
                            &death_time, &err)) {
          LOG(INFO) << "Received preemption notice with death_time "
                    << death_time;
        } else {
          LOG(ERROR) << "Unable to parse preemption notice's death time: "
                     << err;
          CancelPreemptionBarrier();
          return;
        }

        // Trigger protocol in a separate thread: compute max call counter.
        {
          absl::MutexLock l(&mu_);
          if (shut_down_) {
            return;
          }
          sync_protocol_thread_ = absl::WrapUnique(env_->StartThread(
              {}, "PreemptionSyncManager_SyncProtocol",
              std::bind(&PreemptionSyncManager::ComputeSyncCallCounter, this,
                        death_time)));
        }
      });

  return absl::OkStatus();
}

void PreemptionSyncManager::Shutdown() {
  absl::MutexLock l(&mu_);
  if (shut_down_) {
    LOG(INFO) << "PreemptionSyncManager already shut down";
    return;
  }
  shut_down_ = true;

  LOG(INFO) << "Shutting down PreemptionSyncManager...";
  shutdown_.Notify();
  if (call_opts_) {
    call_opts_->StartCancel();
  }
  if (sync_protocol_thread_) {
    sync_protocol_thread_.reset();
  }
  LOG(INFO) << "PreemptionSyncManager shut down.";
}

void PreemptionSyncManager::ComputeSyncCallCounter(absl::Time death_time) {
  // 1. If death time is in the distant future, sleep until there's
  // `kProtocolDuration` left until death time before we begin the protocol.
  const absl::Duration remaining_time = death_time - absl::Now();
  if (remaining_time > kProtocolDuration) {
    LOG(INFO) << "Will begin preemption sync protocol in " << remaining_time;
    const absl::Duration sleep_time = remaining_time - kProtocolDuration;

    if (shutdown_.WaitForNotificationWithTimeout(sleep_time)) {
      // If shutdown is triggered midway, exit thread immediately.
      LOG(WARNING)
          << "Shutdown is triggered before preemption sync protocol has begun.";
      CancelPreemptionBarrier();
      return;
    }
  }

  // 2. Send coordination service the task's current call counter. Hold the lock
  // to prevent updates to `call_counter_` until the protocol completes and this
  // function exits, implying that we have decided on a new
  // `preemption_sync_counter_` or the protocol failed. This ensures correctness
  // of the preemption sync protocol.
  absl::MutexLock l(&mu_);
  const absl::Status notified_status = agent_->InsertKeyValue(
      current_call_counter_key_, std::to_string(call_counter_));
  if (!notified_status.ok()) {
    LOG(ERROR) << "Preemption sync failed - could not inform service of "
                  "current call counter: "
               << notified_status;
    CancelPreemptionBarrier();
    return;
  }

  // 3. Impose a barrier to wait until everybody sends their current call
  // counter.
  const absl::Status barrier_status =
      agent_->WaitAtBarrier(kPreemptionBarrier, kPreemptionBarrierTimeout, {});
  if (!barrier_status.ok()) {
    LOG(ERROR) << "Preemption sync barrier failed: " << barrier_status;
    return;
  }

  // 4. Retrieve every task's current call counter.
  absl::StatusOr<std::vector<KeyValueEntry>> all_counters =
      agent_->GetKeyValueDir(kPreemptionCounterDirKey);
  if (!all_counters.ok()) {
    LOG(ERROR) << "Preemption sync failed - unable to retrieve call counters: "
               << all_counters.status();
    return;
  }

  // 5. Compute the fastest task's call counter.
  // Note: Each task should retrieve the same set of call counters and arrive at
  // the same maximum. We have to calculate this max within each task because
  // coordination service does not provide GetMaxKeyValue().
  int64_t max_counter = kPreemptionSyncUnsetCounter;
  for (const auto& kv : *all_counters) {
    int64_t call_counter;
    if (!absl::SimpleAtoi(kv.value(), &call_counter)) {
      LOG(ERROR) << "Preemption sync failed - failed to parse preemption call "
                    "counter: "
                 << kv.DebugString();
      return;
    }
    max_counter = std::max(max_counter, call_counter);
  }

  if (max_counter == kPreemptionSyncUnsetCounter) {
    LOG(ERROR) << "Preemption sync failed - no call counters found.";
    return;
  }

  // 6. Set sync point to be the next possible call counter of the fastest task.
  preemption_sync_counter_ = max_counter + 1;
  LOG(INFO) << "Preemption sync counter is set: " << preemption_sync_counter_;
  set_sync_point_metric->GetCell()->Set(true);
}

void PreemptionSyncManager::CancelPreemptionBarrier() {
  agent_->CancelBarrierAsync(
      kPreemptionBarrier, [](const absl::Status& status) {
        if (!status.ok()) {
          LOG(ERROR) << "Failed to cancel preemption barrier: " << status;
        }
      });
}

bool PreemptionSyncManager::ReachedSyncPoint(int step_counter) {
  // Record that this API was called at least once.
  sync_usage_metric->GetCell()->Set(true);
  // Note: if a preemption notice has been received and ComputeSyncCallCounter()
  // is ongoing , this method will be blocked until it acquires the lock. This
  // prevents updates to `call_counter_` while `preemption_sync_counter_` is
  // being computed, which ensures correctness of the preemption sync protocol.
  absl::MutexLock l(&mu_);
  CHECK(!shut_down_);
  // Track current call.
  call_counter_ = step_counter;
  VLOG(3) << "Current call counter: " << call_counter_
          << ", Preemption sync point: " << preemption_sync_counter_;

  const bool reached_sync_point = preemption_sync_counter_ == call_counter_;
  if (reached_sync_point) {
    // Record that this job reached the sync point.
    reached_sync_point_metric->GetCell()->Set(true);
  }
  return reached_sync_point;
}

std::unique_ptr<PreemptionSyncManager> CreatePreemptionSyncManager() {
  return std::make_unique<PreemptionSyncManager>();
}
}  // namespace tsl
