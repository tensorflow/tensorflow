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

#ifndef TENSORFLOW_C_EAGER_immediate_execution_distributed_manager_H_
#define TENSORFLOW_C_EAGER_immediate_execution_distributed_manager_H_

#include "tensorflow/core/platform/status.h"

namespace tensorflow {
class CoordinationServiceAgent;
class ImmediateExecutionContext;
class ServerDef;
class WorkerEnv;
class WorkerCacheInterface;

class ImmediateExecutionDistributedManager {
 public:
  virtual ~ImmediateExecutionDistributedManager() {}

  // Set up distributed execution environment on local and remote tasks.
  // When `reset_context` is true, initialize new cluster context state based on
  // cluster configurations provided in `server_def`; otherwise, update existing
  // context state with the provided `server_def`.
  // Contexts created on remote tasks will be considered stale and garbage
  // collected after `keep_alive_secs` of inactivity.
  virtual Status SetOrUpdateServerDef(const ServerDef& server_def,
                                      bool reset_context,
                                      int keep_alive_secs) = 0;

  // Set up a multi-client distributed execution environment. Must be called on
  // all tasks in the cluster.
  // This call internally coordinates with other tasks to initialize the eager
  // context and TF server for multi-client execution.
  virtual Status EnableCollectiveOps(const ServerDef& server_def) = 0;

  // Enable coordination service instance for the distributed cluster. The
  // service is owned by the current distributed manager.
  // See CoordinationServiceInterface for details.
  virtual Status EnableCoordinationService(
      const std::string& service_type, const WorkerEnv* worker_env,
      const ServerDef& server_def, WorkerCacheInterface* worker_cache) = 0;

  // Check if the remote task is alive.
  virtual Status CheckRemoteAlive(const std::string& remote_task_name,
                                  bool* is_alive) = 0;

  // Get pointer to the coordination service agent instance.
  virtual CoordinationServiceAgent* GetCoordinationServiceAgent() = 0;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_immediate_execution_distributed_manager_H_
