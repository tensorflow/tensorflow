/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_DISTRIBUTED_MANAGER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_DISTRIBUTED_MANAGER_H_

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_distributed_manager.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/status.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "tsl/distributed_runtime/preemption/preemption_notifier.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {
#if !defined(IS_MOBILE_PLATFORM)
class EagerContext;
class ServerDef;

class EagerContextDistributedManager
    : public ImmediateExecutionDistributedManager {
 public:
  explicit EagerContextDistributedManager(EagerContext* context)
      : context_(context) {}

  // When running in a distributed context, `init_timeout_in_ms` requests the
  // amount of time to wait for remote workers to respond.

  Status SetOrUpdateServerDef(const ServerDef& server_def, bool reset_context,
                              int keep_alive_secs, int64_t init_timeout_in_ms,
                              int retries,
                              bool clear_existing_contexts = false) override;

  Status InitializeLocalOnlyContext(const ServerDef& server_def,
                                    int keep_alive_secs) override;

  Status EnableCollectiveOps(const ServerDef& server_def) override;

  Status CheckRemoteAlive(const std::string& remote_task_name,
                          bool* is_alive) override;

  tsl::CoordinationServiceAgent* GetCoordinationServiceAgent() override {
    return coordination_service_agent_;
  }
  void SetCoordinationServiceAgent(tsl::CoordinationServiceAgent* agent) {
    coordination_service_agent_ = agent;
  }
  void SetPreemptionNotifier(
      std::unique_ptr<tsl::PreemptionNotifier> notifier) {
    preemption_notifier_ = std::move(notifier);
  }

 private:
  EagerContext* context_;
  // Owned by context_->GetServer()->worker_env()->session_mgr.
  tsl::CoordinationServiceAgent* coordination_service_agent_ = nullptr;
  std::unique_ptr<tsl::PreemptionNotifier> preemption_notifier_;
};
#endif  // !IS_MOBILE_PLATFORM
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_DISTRIBUTED_MANAGER_H_
