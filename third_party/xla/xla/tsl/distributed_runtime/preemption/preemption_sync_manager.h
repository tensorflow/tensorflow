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
#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_SYNC_MANAGER_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_SYNC_MANAGER_H_

#include <memory>
#include <string>

#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_notifier.h"
#include "tsl/platform/status.h"

namespace tsl {

// Enables multiple tasks to coordinate on a safe sync point if any of the tasks
// receive a preemption notice. Example: tasks agree on a safe checkpointing
// step after a preemption notice so that training can resume with minimal
// disruption after the preemption.
// Note: the sync point can only be set once whenever the first preemption
// occurs.
// TODO(b/230630494): Add Reset() to allow multiple sync points to be set.
class PreemptionSyncManager {
 public:
  virtual ~PreemptionSyncManager() = default;

  virtual absl::Status Initialize(CoordinationServiceAgent* agent) = 0;
  virtual absl::Status Initialize(
      CoordinationServiceAgent* agent,
      const std::string& preemption_notifier_type) = 0;
  virtual absl::Status Initialize(
      CoordinationServiceAgent* agent,
      std::unique_ptr<PreemptionNotifier> notifier) = 0;

  // Check if the synchronized point has been reached. When a task has been
  // preempted, a safe sync point will be determined by using the fastest task's
  // next possible sync point, which is then propagated to all tasks via this
  // method.
  // Notes:
  // 1) This must be called during every possible sync point so that the library
  //    is aware of each task's progress.
  // 2) This assumes that each task begins from the same point.
  //    Internally, it updates a counter to track the last `step_counter` passed
  //    in as argument to record each task's current progress.
  // Example use case: this can be called during every training step for every
  // task. Once a preemption notice is received, all tasks will agree on a safe
  // step to pause training and handle the preemption (e.g. save checkpoint and
  // exit, or wait for preempted task to restart, then resume training).
  virtual bool ReachedSyncPoint(int step_counter) = 0;
};

std::unique_ptr<PreemptionSyncManager> CreatePreemptionSyncManager();

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_SYNC_MANAGER_H_
