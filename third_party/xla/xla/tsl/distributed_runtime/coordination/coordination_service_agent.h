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

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/framework/cancellation.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/random.h"

namespace tsl {

// CoordinationServiceAgent defines the interface for tasks to communicate with
// the coordination service instance (which implements
// CoordinationService). One instance of the agent should be deployed
// on each task for it to send various requests and stores / retrieves config
// key-value data to the service.
//
// See CoordinationService for more details on coordination service.
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

  // A JobStateCallback is a callback that receives the current and previous job
  // state. If there is no previous job state, previous_state is empty. The
  // provided states are only valid for the duration of the callback.
  struct JobStateUpdate {
    absl::Span<const tensorflow::CoordinatedTaskStateInfo> previous_state;
    absl::Span<const tensorflow::CoordinatedTaskStateInfo> current_state;
  };
  using JobStateCallback = absl::AnyInvocable<void(const JobStateUpdate&)>;

  CoordinationServiceAgent() = default;

  virtual ~CoordinationServiceAgent() {
    absl::Status s = ShutdownInternal();
    VLOG(3) << "Coordination agent dtor failed with status: " << s;
  }

  absl::Status Initialize(Env* env, absl::string_view job_name, int task_id,
                          const tensorflow::CoordinationServiceConfig& configs,
                          std::unique_ptr<CoordinationClient> leader_client,
                          StatusCallback error_fn, bool recoverable);
  absl::Status Initialize(Env* env, absl::string_view job_name, int task_id,
                          const tensorflow::CoordinationServiceConfig& configs,
                          std::unique_ptr<CoordinationClient> leader_client,
                          StatusCallback error_fn);
  absl::Status Initialize(Env* env, const tensorflow::CoordinatedTask& task,
                          const tensorflow::CoordinationServiceConfig& configs,
                          std::unique_ptr<CoordinationClient> leader_client,
                          StatusCallback error_fn);

  // Return true if the coordination service agent has been initialized.
  bool IsInitialized();

  // Return true if the coordination service agent has successfully connected
  // with the Coordination Service
  bool IsConnected();

  // Return true if the coordination service agent has an error state.
  bool IsError();

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
  absl::Status Connect();

  // Wait for all tasks to be up and registered. The call blocks until all tasks
  // in the cluster are up, or some error occurs.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Agent is not in CONNECTED state.
  //   - InvalidArgument: Unexpected task request
  absl::Status WaitForAllTasks(const tensorflow::DeviceInfo& local_devices);

  // Get the device attributes of tasks from remote tasks in the cluster.
  const tensorflow::DeviceInfo& GetClusterDeviceInfo();

  // State transition in coordination service agent:
  //
  //                 Init              Connect           SetError
  //   UNINITIALIZED ---> DISCONNECTED ------> CONNECTED -------> ERROR
  //                           ^                                  |
  //                           |__________________________________|
  //                                         Reset

  // Get task associated with this agent.
  absl::StatusOr<tensorflow::CoordinatedTask> GetOwnTask();

  // Get status of a remote task.
  absl::StatusOr<std::vector<tensorflow::CoordinatedTaskStateInfo>>
  GetTaskState(const std::vector<tensorflow::CoordinatedTask>& task);

  // Gets status of a remote job.
  absl::StatusOr<std::vector<tensorflow::CoordinatedTaskStateInfo>> GetJobState(
      absl::string_view job_name);

  // Report error to coordination service. This will invoke the error callback.
  // Note that the error payload will set `is_reported_error` to true, to
  // distinguish user-specified errors from internal service or RPC failures.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Uninitialized/disconnected/already in error state.
  //   - InvalidArgument: Unexpected task request
  absl::Status ReportError(const absl::Status& error);

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
  absl::Status Shutdown();

  // Disconnect from the service, and clean up the internal error status.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task is not in error state/has already
  //       disconnected.
  absl::Status Reset();

  // Key-value store API.
  // The agent does not need to be connected to utilize the key-value store.
  // There are no concurrency guarantees. To avoid a race / impose an ordering
  // on potentially concurrent ops (e.g. set, delete), use WaitAtBarrier().

  // Get config key-value from the service.
  // If the key-value is not inserted yet, this is a blocking call that waits
  // until the corresponding key is inserted.
  //   - DeadlineExceeded: timed out waiting for key.
  absl::StatusOr<std::string> GetKeyValue(absl::string_view key);
  absl::StatusOr<std::string> GetKeyValue(absl::string_view key,
                                          absl::Duration timeout);

  // Note: Cancel the underlying RPC call with `call_opts->StartCancel()` and
  // `call_opts->ClearCancelCallback()`.
  std::shared_ptr<CallOptions> GetKeyValueAsync(absl::string_view key,
                                                StatusOrValueCallback done);

  // Get config key-value from the service.
  //   - NotFound: the requested key does not exist.
  absl::StatusOr<std::string> TryGetKeyValue(absl::string_view key);

  // Get all values under a directory (key).
  // A value is considered to be in the directory if its key is prefixed with
  // the directory.
  // This is not a blocking call. If no keys are found, an empty vector is
  // returned immediately.
  absl::StatusOr<std::vector<tensorflow::KeyValueEntry>> GetKeyValueDir(
      absl::string_view key);
  void GetKeyValueDirAsync(absl::string_view key,
                           StatusOrValueDirCallback done);

  // Insert config key-value to the service.
  //   - AlreadyExists: key is already set.
  absl::Status InsertKeyValue(absl::string_view key, absl::string_view value);
  absl::Status InsertKeyValue(absl::string_view key, absl::string_view value,
                              bool allow_overwrite);

  // Delete config keys in the coordination service.
  absl::Status DeleteKeyValue(absl::string_view key);

  // Update the value of a config key.
  absl::Status UpdateKeyValue(absl::string_view key, absl::string_view value);

  // Register a callback that will be invoked when the key or keys under the key
  // directory are changed (inserted, deleted, or updated).
  absl::Status StartWatchKey(absl::string_view key,
                             ChangedKeyValuesCallback on_change);
  absl::Status StopWatchKey(absl::string_view key);

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
  //   - FailedPrecondition: Agent is in UNINITIALIZED or ERROR state, or the
  //       same barrier id is still being invoked.
  virtual absl::Status WaitAtBarrier(
      absl::string_view barrier_id, absl::Duration timeout,
      const std::vector<tensorflow::CoordinatedTask>& tasks);

  void WaitAtBarrierAsync(absl::string_view barrier_id, absl::Duration timeout,
                          const std::vector<tensorflow::CoordinatedTask>& tasks,
                          StatusCallback done);

  // Aborts the barrier if it is ongoing.
  // Current and future WaitAtBarrier() calls with the same id will return a
  // CANCELLED error status.
  // Possible service errors:
  //   - Internal: Coordination service has shut down.
  //   - FailedPrecondition: Barrier is non-existent or not ongoing.
  virtual absl::Status CancelBarrier(absl::string_view barrier_id);
  void CancelBarrierAsync(absl::string_view barrier_id, StatusCallback done);

  // Returns the set of currently alive tasks. More specifically, given a set of
  // tasks T, GetAliveTasks(T) returns the subset T of alive tasks.
  //
  // # Barrier Semantics
  //
  // If multiple tasks call GetAliveTasks concurrently, it's important that they
  // all agree on which tasks are alive. Otherwise, the tasks' behavior might
  // diverge. For example, imagine a set of tasks trying to run an AllGather,
  // but they all disagree on which tasks should be participating in the
  // AllGather. This is buggy.
  //
  // To ensure that every task agrees on which tasks are alive, the
  // GetAliveTasks RPC has barrier-like semantics. Consider an invocation
  // GetAliveTasks(T) for a set of tasks T. The invocation acts as a barrier,
  // waiting for every task in T to call GetAliveTasks(T). Afterwards,
  // GetAliveTasks returns the same set of alive tasks A to all the tasks in T.
  // This ensures that every task agrees which tasks are alive.
  //
  // One small correction. GetAliveTasks doesn't act as a barrier for *every*
  // task in T. Some tasks in T might have failed, so we should not wait for
  // them. Instead, the GetAliveTasks RPC waits only for the returned tasks A.
  //
  // # An Example
  //
  // Imagine we have four tasks: A, B, C, and D. Further imagine that task D
  // has failed and that every task calls GetAliveTasks([A, B, C, D]). The
  // invocation will return tasks [A, B, C]. The GetAliveTasks call acts as a
  // barrier across tasks A, B, and C. Task D, which failed, is ignored.
  absl::StatusOr<std::vector<tensorflow::CoordinatedTask>> GetAliveTasks(
      const std::vector<tensorflow::CoordinatedTask>& tasks);

  // Registers a JobStateCallback that will be invoked when the state of the job
  // changes. Multiple changes to the job state may be coalesced into a single
  // call to the provided callback. Callback invocations may also be delayed.
  void AddJobStateCallback(JobStateCallback callback);

  // Get unowned Env* that the agent was initialized with.
  absl::StatusOr<Env*> GetEnv();

 protected:
  // Set the service agent to error status and invoke the error callback.
  // Note: different from ReportError, this does not report the error status to
  // remote coordination service.
  void SetError(const absl::Status& error);

  // Activate the key-value callback watch.
  absl::Status ActivateWatch(absl::string_view key,
                             const std::map<std::string, std::string>&);

  // Returns an error if agent is not running. If `allow_disconnected` is true,
  // returns OK even if the agent is in DISCONNECTED state.
  absl::Status ValidateRunningAgent(bool allow_disconnected = false);
  void StopHeartbeat();

 private:
  friend class CoordinationServiceRpcHandler;

  absl::Status ShutdownInternal();
  // Starts sending heartbeats to the coordination service.
  void StartSendingHeartbeats();
  // Use long polling to get error from the coordination service.
  void PollForErrorAsync(StatusCallback done);

  // Starts polling for error from the coordination service.
  void StartPollingForError();
  // Cancels the error polling request and stops the error polling thread.
  void StopErrorPolling();
  // Resets the cancellation manager for error polling.
  void ResetCancellationManager();

  // Watches the state of this job.
  void WatchJobState();
  // Stops watching the state of this job.
  void StopWatchingJobState();

  Env* env_ = nullptr;  // Not owned.
  const uint64_t incarnation_id_ = random::New64();
  tensorflow::CoordinatedTask task_;
  tensorflow::CoordinationServiceConfig configs_;
  StatusCallback error_fn_;

  mutable absl::Mutex state_mu_;
  tensorflow::CoordinatedTaskState state_ ABSL_GUARDED_BY(state_mu_) =
      tensorflow::CoordinatedTaskState::TASKSTATE_UNINITIALIZED;
  absl::Status status_ ABSL_GUARDED_BY(state_mu_) = absl::OkStatus();
  // Tracks the number of times a barrier has been used, keyed by id.
  absl::flat_hash_map<std::string, int64_t> barrier_counter_
      ABSL_GUARDED_BY(state_mu_);
  absl::flat_hash_set<std::string> ongoing_barriers_ ABSL_GUARDED_BY(state_mu_);

  uint64_t leader_incarnation_ = 0;
  tensorflow::DeviceInfo cluster_devices_;

  absl::Mutex shutdown_mu_;
  bool shutting_down_ ABSL_GUARDED_BY(shutdown_mu_) = false;
  std::unique_ptr<Thread> heartbeat_thread_;

  absl::Mutex job_state_watcher_mu_;
  std::vector<JobStateCallback> job_state_callbacks_
      ABSL_GUARDED_BY(job_state_watcher_mu_);
  std::unique_ptr<Thread> job_state_watcher_thread_
      ABSL_GUARDED_BY(job_state_watcher_mu_);

  // Must outlive coordination client which may need to access it within
  // GetKeyValueAsync() callbacks.
  CancellationManager cancellation_manager_;
  std::unique_ptr<CancellationManager> error_polling_cancellation_manager_ =
      std::make_unique<CancellationManager>();
  std::unique_ptr<CoordinationClient> leader_client_;

  CoordinationServiceAgent(const CoordinationServiceAgent&) = delete;
  void operator=(const CoordinationServiceAgent&) = delete;
};

std::unique_ptr<CoordinationServiceAgent> CreateCoordinationServiceAgent();

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_AGENT_H_
