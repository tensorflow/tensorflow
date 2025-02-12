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

#ifndef TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_DISTRIBUTED_MANAGER_H_
#define TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_DISTRIBUTED_MANAGER_H_

#include <cstdint>
#include <string>

#include "tensorflow/core/platform/status.h"

namespace tsl {
class CoordinationServiceAgent;
}

namespace tensorflow {
class ImmediateExecutionContext;
class ServerDef;
class WorkerEnv;
class WorkerCacheInterface;

class ImmediateExecutionDistributedManager {
 public:
  virtual ~ImmediateExecutionDistributedManager() {}

  // Set up distributed execution environment on local and remote tasks.
  // When `reset_context` is true, initialize new cluster context state based
  // on cluster configurations provided in `server_def`; otherwise, update
  // existing context state with the provided `server_def`. Contexts created
  // on remote tasks will be considered stale and garbage collected after
  // `keep_alive_secs` of inactivity.
  virtual absl::Status SetOrUpdateServerDef(
      const ServerDef& server_def, bool reset_context, int keep_alive_secs,
      int64_t init_timeout_in_ms, int retries,
      bool clear_existing_contexts = false) = 0;

  // Initializes context for the local worker and no contexts will be created
  // for remote workers. Currently this only works for resetting context.
  // TODO(b/289445025): Consider removing this when we find a proper fix.
  virtual absl::Status InitializeLocalOnlyContext(const ServerDef& server_def,
                                                  int keep_alive_secs) = 0;

  // Set up a multi-client distributed execution environment. Must be called
  // on all tasks in the cluster. This call internally coordinates with other
  // tasks to initialize the eager context and TF server for multi-client
  // execution.
  virtual absl::Status EnableCollectiveOps(const ServerDef& server_def) = 0;

  // Check if the remote task is alive.
  virtual absl::Status CheckRemoteAlive(const std::string& remote_task_name,
                                        bool* is_alive) = 0;

  // Get pointer to the coordination service agent instance.
  virtual tsl::CoordinationServiceAgent* GetCoordinationServiceAgent() = 0;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_DISTRIBUTED_MANAGER_H_
