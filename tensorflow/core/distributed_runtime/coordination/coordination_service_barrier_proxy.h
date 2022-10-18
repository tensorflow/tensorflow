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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_BARRIER_PROXY_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_BARRIER_PROXY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// A local proxy connecting the coordination service's barrier.
// The barrier provided by coordination service can only block at tasks (i.e.,
// TPU workers), but sometimes we need a barrier that can block at different
// threads. The proxy first waits at threads on a participating
// task and then issues a barrier wait to the coordination service once all the
// threads at that task have arrived.
// Usage:
//   // Main thread creates a `BarrierProxy`:
//   barrier = new BarrierProxy(agent, tasks, key, num_local_threads);
//
//   // Each participating thread could then call:
//   auto [status, last_exit] = barrier.Wait();
//   // The last exited thread is responsible for deleting the barrier.
//   if (last_exit) {
//     delete barrier;
//   }
class BarrierProxy {
 public:
  TF_DISALLOW_COPY_AND_ASSIGN(BarrierProxy);
  // Construct a BarrierProxy connected to the coordination service via `agent`.
  // `tasks` specifies all participating coordinated tasks and
  // `num_local_threads` specifies the number of threads in this task to
  // particiate. If no tasks are specified, the barrier will block for all the
  // connected tasks.
  BarrierProxy(CoordinationServiceAgent* agent,
               std::vector<CoordinatedTask> tasks, int num_local_threads,
               absl::string_view key, absl::Duration timeout)
      : key_(key),
        agent_(agent),
        tasks_(std::move(tasks)),
        timeout_(timeout),
        num_local_threads_(num_local_threads) {}

  ~BarrierProxy() = default;

  // Waits at the barrier. The first return value is the status when exiting the
  // barrier and the second returns `true` for precisely one caller, which may
  // then destroy the barrier.
  std::pair<Status, bool> Wait();

 private:
  const std::string key_;
  CoordinationServiceAgent* agent_;
  const std::vector<CoordinatedTask> tasks_;
  absl::Duration timeout_;

  mutex mu_;
  condition_variable cv_ TF_GUARDED_BY(mu_);
  const int num_local_threads_;
  int num_entered_ TF_GUARDED_BY(mu_) = 0;
  int num_to_exit_ TF_GUARDED_BY(mu_) = 0;
  Status status_ TF_GUARDED_BY(mu_);
  bool status_set_ TF_GUARDED_BY(mu_) = false;
};

// Manages the life cycle of BarrierProxies automatically.
// Usage:
//   // Main thread creates a `BarrierProxy`:
//   BarrierProxyManager barrier_mgr;
//
//   // Exactly `num_local_threads` threads call:
//   Status s = barrier_mgr.Wait(agent, task, num_local_threads, key, timeout);
class BarrierProxyManager {
 public:
  TF_DISALLOW_COPY_AND_ASSIGN(BarrierProxyManager);
  BarrierProxyManager() = default;
  ~BarrierProxyManager() = default;

  // Waits at the barrier backed by the coord service `agent` and keyed by
  // `key`. `tasks` specifies all participating coordinated tasks and
  // `num_local_threads` specifies the number of threads in this task to
  // participate. If no tasks are specified, the barrier will block for all the
  // connected tasks.
  Status Wait(CoordinationServiceAgent* agent,
              const std::vector<CoordinatedTask>& tasks, int num_local_threads,
              absl::string_view key, absl::Duration timeout);
  // The number of active BarrierProxies.
  size_t size() const;

 private:
  mutable mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<BarrierProxy>> barriers_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_BARRIER_PROXY_H_
