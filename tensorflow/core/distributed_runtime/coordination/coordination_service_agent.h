/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_

#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
class DeviceAttributes;
class WorkerEnv;
class ServerDef;

// CoordinationServiceAgent defines the interface for tasks to communicate with
// the coordination service instance (which implements
// CoordinationServiceInterface). One instance of the agent should be deployed
// on each task for it to send various requests and stores / retrieves config
// key-value data to the service.
//
// See CoordinationServiceInterface for more details on coordination service.
//
// Experimental feature. Not yet implemented in open source.
class CoordinationServiceAgent {
 public:
  using StatusOrValueCallback =
      std::function<void(const StatusOr<std::string>&)>;
  using ChangedKeyValuesCallback =
      std::function<void(const std::map<std::string, std::string>&)>;

  virtual ~CoordinationServiceAgent() {}

  // Initialize coordination service agent.
  virtual Status Initialize(
      const WorkerEnv* worker_env, const ServerDef& server_def,
      std::unique_ptr<CoordinationClientCache> client_cache,
      StatusCallback error_fn) = 0;
  // Return true if the coordination service agent has been initialized.
  virtual bool IsInitialized() = 0;

  // Connect to coordination service with the following steps:
  //   - connect to service address specified in the config of `server_def`
  //   - register itself as a worker to the service
  //   - start a thread to periodically send heartbeat message with the service
  virtual Status Connect() = 0;

  // Wait for all tasks to be up and registered. The call blocks until all tasks
  // in the cluster are up, or some error occurs.
  virtual Status WaitForAllTasks() = 0;

  // Get the device attributes of tasks from remote tasks in the cluster.
  virtual const std::vector<DeviceAttributes>& GetClusterDeviceAttributes() = 0;

  // State transition in coordination service agent:
  //
  //                 Init              Connect         SetError
  //   UNINITIALIZED ---> DISCONNECTED ------> RUNNING -------> ERROR
  //                           ^                                  |
  //                           |__________________________________|
  //                                         Reset
  enum class TaskState {
    UNINITIALIZED,
    DISCONNECTED,
    RUNNING,
    ERROR,
  };

  // Get status of a remote task.
  virtual StatusOr<TaskState> GetTaskStatus(const std::string& job_name,
                                            const int task_id) = 0;

  // Report error to coordination service. This will invoke the error callback.
  virtual Status ReportError(const Status& error) = 0;

  // Disconnect from the service, and clean up the internal error status.
  virtual Status Reset() = 0;

  // Get config key-value from the service.
  virtual StatusOr<std::string> GetKeyValue(const std::string& key) = 0;
  virtual void GetKeyValueAsync(const std::string& key,
                                StatusOrValueCallback done) = 0;

  // Insert config key-value to the service. Return error if key is already set.
  virtual Status InsertKeyValue(const std::string& key,
                                const std::string& value) = 0;

  // Delete config keys in the coordination service.
  virtual Status DeleteKeyValue(const std::string& key) = 0;

  // Update the value of a config key.
  virtual Status UpdateKeyValue(const std::string& key,
                                const std::string& value) = 0;

  // Register a callback that will be invoked when the key or keys under the key
  // directory are changed (inserted, deleted, or updated).
  virtual Status StartWatchKey(const std::string& key,
                               ChangedKeyValuesCallback on_change) = 0;
  virtual Status StopWatchKey(const std::string& key) = 0;

 protected:
  // Set the service agent to error status and invoke the error callback.
  // Note: different from ReportError, this does not report the error status to
  // remote coordination service.
  virtual void SetError(const Status& error) = 0;

  // Activate the key-value callback watch.
  virtual Status ActivateWatch(const std::string& key,
                               const std::map<std::string, std::string>&) = 0;

 private:
  friend class CoordinationServiceRpcHandler;
};

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
