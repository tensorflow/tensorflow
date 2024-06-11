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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "tsl/platform/status.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace tensorflow {
class CoordinationServiceConfig;
};  // namespace tensorflow

namespace tsl {
class Env;

// CoordinationServiceAgent defines the interface for tasks to communicate with
// the coordination service instance (which implements
// CoordinationServiceInterface). One instance of the agent should be deployed
// on each task for it to send various requests and stores / retrieves config
// key-value data to the service.
//
// See CoordinationServiceInterface for more details on coordination service.
//
// All coordination service errors will have an additional
// CoordinationServiceError payload to distinguish themselves from RPC failures.
// The payload can optionally specify the error origin, and if the error is
// reported by the user via `agent->ReportError()`.
//
// Possible service errors:
//    - Internal: Coordination service has shut down or has not been enabled.
//    - Aborted: Incarnation mismatch during heartbeat (either remote
//                       task or coordination service has restarted).
//    - Unavailable: Heartbeat timeout from remote task (failed,
//                           crashed or got preempted).
//    - InvalidArgument: Unexpected heartbeat from remote task (not
//                               registered or wrong config).
class CoordinationServiceAgent {
 public:
  using StatusOrValueCallback =
      std::function<void(const absl::StatusOr<std::string>&)>;
  // Collection of key-value pairs in the same directory.
  using StatusOrValueDirCallback = std::function<void(
      const absl::StatusOr<std::vector<tensorflow::KeyValueEntry>>&)>;
  using ChangedKeyValuesCallback =
      std::function<void(const std::map<std::string, std::string>&)>;

  virtual ~CoordinationServiceAgent() = default;

  // Initialize coordination service agent.
  virtual absl::Status Initialize(
      tsl::Env* env, std::string_view job_name, int task_id,
      const tensorflow::CoordinationServiceConfig& configs,
      std::unique_ptr<CoordinationClient> leader_client,
      StatusCallback error_fn) = 0;
  virtual absl::Status Initialize(
      tsl::Env* env, const tensorflow::CoordinatedTask& task,
      const tensorflow::CoordinationServiceConfig& configs,
      std::unique_ptr<CoordinationClient> leader_client,
      StatusCallback error_fn) = 0;

  // Return true if the coordination service agent has been initialized.
  virtual bool IsInitialized() = 0;

  // Return true if the coordination service agent has successfully connected
  // with the Coordination Service
  virtual bool IsConnected() = 0;

  // Return true if the coordination service agent has an error state.
  virtual bool IsError() = 0;

  // Connect to coordination service with the following steps:
  //   - connect to service address specified in the config of `server_def`
  //   - register itself as a task to the service
  //   - start a thread to periodically send heartbeat message with the service
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Agent is not in DISCONNECTED state.
  //   - InvalidArgument: Unexpected task registration
  //   - Aborted: Duplicate task registration (agent will retry connecting until
  //              the configured timeout)
  virtual absl::Status Connect() = 0;

  // Wait for all tasks to be up and registered. The call blocks until all tasks
  // in the cluster are up, or some error occurs.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Agent is not in CONNECTED state.
  //   - InvalidArgument: Unexpected task request
  virtual absl::Status WaitForAllTasks(
      const tensorflow::DeviceInfo& local_devices) = 0;

  // Get the device attributes of tasks from remote tasks in the cluster.
  virtual const tensorflow::DeviceInfo& GetClusterDeviceInfo() = 0;

  // State transition in coordination service agent:
  //
  //                 Init              Connect           SetError
  //   UNINITIALIZED ---> DISCONNECTED ------> CONNECTED -------> ERROR
  //                           ^                                  |
  //                           |__________________________________|
  //                                         Reset

  // Get task associated with this agent.
  virtual absl::StatusOr<tensorflow::CoordinatedTask> GetOwnTask() = 0;

  // Get status of a remote task.
  virtual absl::StatusOr<std::vector<tensorflow::CoordinatedTaskStateInfo>>
  GetTaskState(const std::vector<tensorflow::CoordinatedTask>& task) = 0;

  // Report error to coordination service. This will invoke the error callback.
  // Note that the error payload will set `is_reported_error` to true, to
  // distinguish user-specified errors from internal service or RPC failures.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Uninitialized/disconnected/already in error state.
  //   - InvalidArgument: Unexpected task request
  virtual absl::Status ReportError(const absl::Status& error) = 0;

  // Shuts down by disconnecting from the service. Should only be called if
  // agent is connected and no further agent calls (except the destructor) are
  // expected. If `shutdown_barrier_timeout_in_ms` is specified in the config,
  // blocks until all tasks reach the barrier before shutting down together. If
  // the barrier times out, this agent will still disconnect, while an error is
  // reported to other agents that did not reach the barrier on time.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: Task was in error state (note: agent is still
  //                         shut down forcefully).
  virtual absl::Status Shutdown() = 0;

  // Disconnect from the service, and clean up the internal error status.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task is not in error state/has already
  //       disconnected.
  virtual absl::Status Reset() = 0;

  // Key-value store API.
  // The agent does not need to be connected to utilize the key-value store.
  // There are no concurrency guarantees. To avoid a race / impose an ordering
  // on potentially concurrent ops (e.g. set, delete), use WaitAtBarrier().

  // Get config key-value from the service.
  // If the key-value is not inserted yet, this is a blocking call that waits
  // until the corresponding key is inserted.
  //   - DeadlineExceeded: timed out waiting for key.
  virtual absl::StatusOr<std::string> GetKeyValue(std::string_view key) = 0;
  virtual absl::StatusOr<std::string> GetKeyValue(std::string_view key,
                                                  absl::Duration timeout) = 0;
  // Note: Cancel the underlying RPC call with `call_opts->StartCancel()` and
  // `call_opts->ClearCancelCallback()`.
  virtual std::shared_ptr<CallOptions> GetKeyValueAsync(
      std::string_view, StatusOrValueCallback done) = 0;

  // Get config key-value from the service.
  //   - NotFound: the requested key does not exist.
  virtual absl::StatusOr<std::string> TryGetKeyValue(std::string_view key) = 0;

  // Get all values under a directory (key).
  // A value is considered to be in the directory if its key is prefixed with
  // the directory.
  // This is not a blocking call. If no keys are found, an empty vector is
  // returned immediately.
  virtual absl::StatusOr<std::vector<tensorflow::KeyValueEntry>> GetKeyValueDir(
      std::string_view key) = 0;
  virtual void GetKeyValueDirAsync(std::string_view key,
                                   StatusOrValueDirCallback done) = 0;

  // Insert config key-value to the service.
  //   - AlreadyExists: key is already set.
  virtual absl::Status InsertKeyValue(std::string_view key,
                                      std::string_view value) = 0;

  virtual absl::Status InsertKeyValue(std::string_view key,
                                      std::string_view value,
                                      bool allow_overwrite) = 0;

  // Delete config keys in the coordination service.
  virtual absl::Status DeleteKeyValue(std::string_view key) = 0;

  // Update the value of a config key.
  virtual absl::Status UpdateKeyValue(std::string_view key,
                                      std::string_view value) = 0;

  // Register a callback that will be invoked when the key or keys under the key
  // directory are changed (inserted, deleted, or updated).
  virtual absl::Status StartWatchKey(std::string_view key,
                                     ChangedKeyValuesCallback on_change) = 0;
  virtual absl::Status StopWatchKey(std::string_view key) = 0;

  // Blocks until all (or a subset of) tasks are at the barrier or the barrier
  // fails.
  //
  // `barrier_id` should be unique across barriers.
  //
  // The first WaitAtBarrier() call received by the service for a particular
  // barrier_id is special in that it determines the barrier deadline based on
  // timeout duration.
  // However, if subsequent calls by different agents specify a different set of
  // `tasks` for the same `barrier_id`, the barrier will fail instantly.
  // For example,
  //   agent_1->WaitAtBarrier(“barrier”, 10min, <<”worker”, 1>, <”worker”, 2>>);
  //   agent_2->WaitAtBarrier(“barrier”, 10min, <<”worker”, 2>, <”worker”, 3>>);
  // Barrier fails after agent_2’s call because it specifies a different set of
  // participating tasks.
  //
  // If no tasks are specified (default), the barrier will block for all the
  // connected tasks.
  //
  // Possible service errors:
  //   - DeadlineExceeded: Timed out waiting for specified tasks at the barrier.
  //      Deadline is determined by the server timestamp when it receives the
  //      first WaitAtBarrier() + timeout duration.
  //   - Cancelled: One of the tasks called CancelBarrier().
  //   - Aborted: Service is shutting down.
  //   - Internal: Any participating task is in ERROR state, or service has shut
  //     down.
  //   - InvalidArgument: (1) Conflicting tasks specified by different agents
  //       for the same barrier, (2) one of the participating tasks is not in
  //       the cluster, or (3) task making the request is not included in the
  //       list of participating tasks.
  //   - FailedPrecondition: Agent is in UNINITIALIZED or ERROR state. Or the
  //       same barrier_id was already used previously.
  virtual absl::Status WaitAtBarrier(
      std::string_view barrier_id, absl::Duration timeout,
      const std::vector<tensorflow::CoordinatedTask>& tasks) = 0;

  virtual void WaitAtBarrierAsync(
      std::string_view barrier_id, absl::Duration timeout,
      const std::vector<tensorflow::CoordinatedTask>& tasks,
      StatusCallback done) = 0;

  // Aborts the barrier if it is ongoing.
  // Current and future WaitAtBarrier() calls with the same id will return a
  // CANCELLED error status.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Barrier has already been passed.
  virtual absl::Status CancelBarrier(std::string_view barrier_id) = 0;
  virtual void CancelBarrierAsync(std::string_view barrier_id,
                                  StatusCallback done) = 0;

  // Get unowned Env* that the agent was initialized with.
  virtual absl::StatusOr<tsl::Env*> GetEnv() = 0;

 protected:
  // Set the service agent to error status and invoke the error callback.
  // Note: different from ReportError, this does not report the error status to
  // remote coordination service.
  virtual void SetError(const absl::Status& error) = 0;

  // Activate the key-value callback watch.
  virtual absl::Status ActivateWatch(
      std::string_view, const std::map<std::string, std::string>&) = 0;

 private:
  friend class CoordinationServiceRpcHandler;
};

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent();

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
