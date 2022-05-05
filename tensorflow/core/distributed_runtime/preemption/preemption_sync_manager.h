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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_SYNC_MANAGER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_SYNC_MANAGER_H_

#include <memory>

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/preemption/preemption_notifier.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

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

  // TODO(b/230630494): Allow init with PjRT distributed client.
  virtual Status Initialize(Env* env, CoordinationServiceAgent* agent) = 0;
  virtual Status Initialize(Env* env, CoordinationServiceAgent* agent,
                            std::unique_ptr<PreemptionNotifier> notifier) = 0;

  // Check if the synchronized point has been reached. When a task has been
  // preempted, a safe sync point will be determined by using the fastest task's
  // next possible sync point, which is then propagated to all tasks via this
  // method.
  // Notes:
  // 1) This must be called during every possible sync point so that the library
  //    is aware of each task's progress.
  // 2) This assumes that each task begins from the same point.
  //    Internally, we use a counter to track the number of calls that have been
  //    made to record each task's current progress.
  // Example use case: this can be called during every training step for every
  // task. Once a preemption notice is received, all tasks will agree on a safe
  // step to pause training and handle the preemption (e.g. save checkpoint and
  // exit, or wait for preempted task to restart, then resume training).
  virtual bool ReachedSyncPoint() = 0;
};

std::unique_ptr<PreemptionSyncManager> CreatePreemptionSyncManager();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_SYNC_MANAGER_H_
