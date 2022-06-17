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
#include "tensorflow/core/distributed_runtime/preemption/preemption_sync_manager.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/preemption/preemption_notifier.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
#include "learning/brain/runtime/preemption/borg_preemption_notifier.h"
#endif  // PLATFORM_GOOGLE && !LIBTPU_ON_GCE

namespace tensorflow {
namespace {
constexpr int64_t kPreemptionSyncUnsetCounter = -1;
constexpr char kPreemptionNoticeKey[] = "RECEIVED_PREEMPTION_NOTICE";
constexpr char kPreemptionCounterDirKey[] = "PREEMPTION_CURRENT_COUNTER/";
constexpr char kPreemptionBarrier[] = "PREEMPTION_SYNC_BARRIER";
constexpr absl::Duration kPreemptionBarrierTimeout = absl::Minutes(3);

// Only start protocol if death time is within `kProtocolDuration`, so that we
// don't synchronize too early.
// TODO(b/230630494): Make this configurable so that users can extend this to
// accommodate higher checkpoint durations.
constexpr absl::Duration kProtocolDuration = absl::Minutes(15);

class PreemptionSyncManagerImpl : public PreemptionSyncManager {
 public:
  PreemptionSyncManagerImpl() = default;
  ~PreemptionSyncManagerImpl() override {
    call_opts_->StartCancel();
    call_opts_->ClearCancelCallback();
    shutdown_.Notify();
  }
  Status Initialize(CoordinationServiceAgent* agent) override;
#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
  Status InitWithBorgPreemptionNotifier(
      CoordinationServiceAgent* agent) override;
#endif
  Status Initialize(CoordinationServiceAgent* agent,
                    std::unique_ptr<PreemptionNotifier> notifier) override;
  bool ReachedSyncPoint(int step_counter) override;

 private:
  // Determine the sync point upon receipt of preemption notice (death time).
  void ComputeSyncCallCounter(absl::Time death_time);
  // Notify other tasks to not wait at the barrier if the sync protocol failed
  // midway.
  void CancelPreemptionBarrier();

  mutex mu_;
  // Tracks the last step_counter passed into ReachedSyncPoint();
  int64_t call_counter_ TF_GUARDED_BY(mu_) = 0;
  // If set, determines the sync point.
  int64_t preemption_sync_counter_ TF_GUARDED_BY(mu_) =
      kPreemptionSyncUnsetCounter;
  std::string current_call_counter_key_;

  Env* env_;                         // Not owned;
  CoordinationServiceAgent* agent_;  // Not owned.
  absl::Notification shutdown_;
  std::unique_ptr<Thread> sync_protocol_thread_;
  std::unique_ptr<PreemptionNotifier> preemption_notifier_;
  std::shared_ptr<CallOptions> call_opts_;
};

Status PreemptionSyncManagerImpl::Initialize(CoordinationServiceAgent* agent) {
  TF_ASSIGN_OR_RETURN(Env * env, agent->GetEnv());
  return Initialize(agent, CreateSigtermNotifier(env));
}

#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
Status PreemptionSyncManagerImpl::InitWithBorgPreemptionNotifier(
    CoordinationServiceAgent* agent) {
  TF_ASSIGN_OR_RETURN(Env * env, agent->GetEnv());
  return Initialize(agent, CreateBorgPreemptionNotifier(env));
}
#endif

Status PreemptionSyncManagerImpl::Initialize(
    CoordinationServiceAgent* agent,
    std::unique_ptr<PreemptionNotifier> notifier) {
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
      [agent = agent_, task_name](StatusOr<absl::Time> death_time) {
        if (!death_time.ok()) {
          // This usually happens when the preemption notifier dtor is called
          // and blocking calls are cancelled.
          LOG(ERROR) << "Error from preemption notifier: "
                     << death_time.status();
          return;
        }
        // Notify coordination service about preemption notice.
        const Status s = agent->InsertKeyValue(kPreemptionNoticeKey,
                                               absl::FormatTime(*death_time));
        LOG(INFO) << "Notified coordination service that this task will "
                     "be preempted at "
                  << *death_time << ". Status: " << s;
      });

  /* Listen for preemption notice (death time) from coordination service, which
   * triggers the sync protocol.
   */
  call_opts_ = agent_->GetKeyValueAsync(
      kPreemptionNoticeKey,
      [this, agent = agent_](StatusOr<std::string> status_or_death_time) {
        // Retrieve preemption notice and parse death time.
        if (!status_or_death_time.ok()) {
          LOG(ERROR) << "Failed to retrieve preemption notice from "
                        "coordination service: "
                     << status_or_death_time.status();
          // Notify other tasks to not wait at the barrier. Note:
          // CancelPreemptionBarrier() cannot be used because this may be
          // triggered after preemption sync manager has been destroyed.
          agent->CancelBarrierAsync(
              kPreemptionBarrier, [](const Status& status) {
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

        LOG(INFO) << "Received preemption notice with death time: "
                  << death_time;

        // Trigger protocol in a separate thread: compute max call counter.
        sync_protocol_thread_ = absl::WrapUnique(env_->StartThread(
            {}, "PreemptionSyncManager_SyncProtocol",
            std::bind(&PreemptionSyncManagerImpl::ComputeSyncCallCounter, this,
                      death_time)));
      });

  return Status::OK();
}

void PreemptionSyncManagerImpl::ComputeSyncCallCounter(absl::Time death_time) {
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
  mutex_lock l(mu_);
  const Status notified_status = agent_->InsertKeyValue(
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
  const Status barrier_status =
      agent_->WaitAtBarrier(kPreemptionBarrier, kPreemptionBarrierTimeout, {});
  if (!barrier_status.ok()) {
    LOG(ERROR) << "Preemption sync barrier failed: " << barrier_status;
    return;
  }

  // 4. Retrieve every task's current call counter.
  StatusOr<std::vector<KeyValueEntry>> all_counters =
      agent_->GetKeyValueDir(kPreemptionCounterDirKey);
  if (!all_counters.ok()) {
    LOG(ERROR) << "Preemption sync failed - unable to retrieve call counters : "
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
}

void PreemptionSyncManagerImpl::CancelPreemptionBarrier() {
  agent_->CancelBarrierAsync(kPreemptionBarrier, [](const Status& status) {
    if (!status.ok()) {
      LOG(ERROR) << "Failed to cancel preemption barrier: " << status;
    }
  });
}

bool PreemptionSyncManagerImpl::ReachedSyncPoint(int step_counter) {
  // Note: if a preemption notice has been received and ComputeSyncCallCounter()
  // is ongoing , this method will be blocked until it acquires the lock. This
  // prevents updates to `call_counter_` while `preemption_sync_counter_` is
  // being computed, which ensures correctness of the preemption sync protocol.
  mutex_lock l(mu_);
  // Track current call.
  call_counter_ = step_counter;
  VLOG(3) << "Current call counter: " << call_counter_
          << ", Preemption sync point: " << preemption_sync_counter_;

  // Check if we have reached the sync point.
  return preemption_sync_counter_ == call_counter_;
}
}  // namespace
std::unique_ptr<PreemptionSyncManager> CreatePreemptionSyncManager() {
  return std::make_unique<PreemptionSyncManagerImpl>();
}
}  // namespace tensorflow
