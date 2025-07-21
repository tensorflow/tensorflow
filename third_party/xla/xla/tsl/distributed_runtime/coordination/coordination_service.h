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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/key_value_store.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/random.h"

namespace tsl {

TSL_LIB_GTL_DEFINE_INT_TYPE(IncarnationId, uint64_t);

// Coordination service is used for controlling and coordinating distributed
// execution in a cluster of multiple tasks.
//
// When enabled, the service keeps track of cluster configurations and the state
// of cluster members. TF runtime and libraries can use it to orchestrate
// cluster initialization, check the healthiness of tasks, and propagate error
// messages to the cluster.
//
// Normally, the service should first Start(), then perform the supported
// coordination operations, and finally Stop(). When service runs into error or
// SetError() is called, all subsequent operations will be in error state.
//
// CoordinationServiceInterface defines the service interface for distributed
// coordination. One instance of the service should be deployed in a cluster,
// handling various requests and stores configuration key-value data for the
// tasks. Each task interacts with the service through CoordinationServiceAgent.
class CoordinationService {
 public:
  using StatusOrValueCallback =
      std::function<void(const absl::StatusOr<absl::string_view>&)>;
  using BarrierCallback = std::function<void(const absl::Status&, int64_t)>;
  using GetAliveTasksCallback = std::function<void(
      const absl::Status&, const std::vector<tensorflow::CoordinatedTask>&,
      const std::vector<IncarnationId> incarnations)>;

  // Convenience structs to allow using CoordinatedTask as container keys.
  struct CoordinatedTaskHash {
    uint64_t operator()(const tensorflow::CoordinatedTask& task) const {
      return absl::HashOf(task.job_name(), task.task_id());
    }
  };
  struct CoordinatedTaskEqual {
    bool operator()(const tensorflow::CoordinatedTask& lhs,
                    const tensorflow::CoordinatedTask& rhs) const {
      return lhs.job_name() == rhs.job_name() && lhs.task_id() == rhs.task_id();
    }
  };

  using CoordinatedTaskSet =
      absl::flat_hash_set<tensorflow::CoordinatedTask, CoordinatedTaskHash,
                          CoordinatedTaskEqual>;

  static std::unique_ptr<CoordinationService> Create(
      Env* env, const tensorflow::CoordinationServiceConfig& config,
      std::unique_ptr<CoordinationClientCache> cache) {
    return std::make_unique<CoordinationService>(env, config, std::move(cache));
  }

  CoordinationService(Env* env,
                      const tensorflow::CoordinationServiceConfig& config,
                      std::unique_ptr<CoordinationClientCache> client_cache);

  ~CoordinationService() {
    absl::MutexLock lock(&state_mu_);
    Stop();
  }

  // This function is invoked after each task's local devices are appended in a
  // deterministic order during WaitForAllTasks(). This is useful to convert the
  // result into another message, or set global device ids.
  void SetDeviceAggregationFunction(std::function<tensorflow::DeviceInfo(
                                        const tensorflow::DeviceInfo& devices)>
                                        post_aggregate_device_fn);

  // Register a task to the service.
  // Possible service errors:
  //   - Internal: Service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - Aborted: (1) task is in error state, or (2) task is in connected state
  //       with a different incarnation, indicating that it restarted.
  //   - DeadlineExceeded: waited too long for straggler tasks to register.
  absl::Status RegisterTask(const tensorflow::CoordinatedTask& task,
                            IncarnationId incarnation);
  void RegisterTaskAsync(const tensorflow::CoordinatedTask& task,
                         IncarnationId incarnation, StatusCallback done);

  // Wait for all tasks to be up and running, and register local device
  // info. The callback is invoked when all tasks are up and registered, or some
  // error occurs.
  // Each task's local devices will be appended in a deterministic order, and
  // post-processed by the callback in SetDeviceAggregationFunction() (if set).
  void WaitForAllTasks(const tensorflow::CoordinatedTask& task,
                       const tensorflow::DeviceInfo& devices,
                       StatusCallback done);

  // Disconnects task from the service. If `shutdown_barrier_timeout_in_ms` is
  // specified in the config, blocks until all tasks reach the barrier before
  // disconnecting together.
  // Possible service errors:
  //   - Internal: Service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task has already disconnected.
  void ShutdownTaskAsync(const tensorflow::CoordinatedTask& task,
                         StatusCallback done);

  // Disconnects task from the service and cleans up its internal error state.
  // Possible service errors:
  //   - Internal: Service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task has already disconnected.
  absl::Status ResetTask(const tensorflow::CoordinatedTask& task);

  // Update the heartbeat timestamp of a task. This should only be invoked on
  // the leader of the cluster.
  //   - Internal: Service has shut down.
  absl::Status RecordHeartbeat(const tensorflow::CoordinatedTask& task,
                               IncarnationId incarnation);

  // Set a task in error state permanently.
  absl::Status ReportTaskError(const tensorflow::CoordinatedTask& task,
                               const absl::Status& error);

  // Get the state and the error status of the tasks.
  std::vector<tensorflow::CoordinatedTaskStateInfo> GetTaskState(
      const std::vector<tensorflow::CoordinatedTask>& task);

  // Watches the state and the error status of the job.
  using WatchJobStateCallback = absl::AnyInvocable<void(
      std::vector<tensorflow::CoordinatedTaskStateInfo>, int64_t)>;
  void WatchJobState(absl::string_view job_name,
                     std::optional<int64_t> version_number,
                     WatchJobStateCallback);

  // Insert a configuration key-value in the coordination service.
  // For now, a key-value can only be inserted once and cannot be updated.
  // The key-values are not persisted and will be lost if the leader fails.
  absl::Status InsertKeyValue(absl::string_view key, absl::string_view value);
  absl::Status InsertKeyValue(absl::string_view key, absl::string_view value,
                              bool allow_overwrite);

  // Get a configuration key-value from the coordination service. The `done`
  // callback is invoked when the key-value becomes available.
  void GetKeyValueAsync(absl::string_view key, StatusOrValueCallback done);

  // Get a configuration key-value from the coordination service. If the key
  // does not exist, return NotFound error.
  absl::StatusOr<std::string> TryGetKeyValue(absl::string_view key);

  // Gets all values under a directory (key).
  // A value is considered to be in the directory if its key is prefixed with
  // the directory. This is not a blocking call. Agent does not need to be
  // connected to utilize the distributed key-value store.
  std::vector<tensorflow::KeyValueEntry> GetKeyValueDir(
      absl::string_view directory_key);

  // Delete configuration key-value. If key is a directory, recursively clean
  // up all key-values under the directory.
  absl::Status DeleteKeyValue(absl::string_view key);

  // Blocks until all (or a subset of) tasks are at the barrier or the barrier
  // fails.
  //
  // `barrier_id` should be unique across barriers. Once the barrier has passed
  // or failed, subsequent calls will not block, and immediately respond with
  // the previous response.
  //
  // The first WaitAtBarrier() call received by the service for a particular
  // barrier id is special in that it determines the barrier deadline based on
  // timeout duration.
  // However, if subsequent calls by different agents specify a different set of
  // `participating_tasks` for the same `barrier_id`, the barrier will fail
  // instantly.
  //
  // If no tasks are specified (default), the barrier will block for all the
  // connected tasks.
  //
  // Possible service errors:
  //   - DeadlineExceeded: Timed out waiting for specified tasks at the barrier.
  //       Deadline is determined by the server timestamp when it receives the
  //       first WaitAtBarrier() + timeout duration.
  //   - Cancelled: One of the tasks called CancelBarrier().
  //   - Aborted: Service is shutting down.
  //   - Internal: (1) Any participating task is in ERROR state, (2)
  //       coordination service has shut down, or (3) the barrier request has a
  //       mismatched counter, indicating that somebody unexpectedly restarted.
  //   - InvalidArgument: (1) Conflicting tasks specified by different agents
  //       for the same barrier, (2) one of the participating tasks is not in
  //       the cluster, or (3) task making the request is not included in the
  //       list of participating tasks.
  //   - FailedPrecondition: Agent is in UNINITIALIZED or ERROR state.
  // TODO(b/342448688): Allow re-use of ids by specifying different counters.
  // The counter field is mostly ignored at the moment with no user-facing
  // effect.
  void BarrierAsync(
      std::string barrier_id, int64_t counter, absl::Duration timeout,
      const tensorflow::CoordinatedTask& task,
      const std::vector<tensorflow::CoordinatedTask>& participating_tasks,
      BarrierCallback done);

  // Aborts the barrier if it is ongoing.
  // Current and future WaitAtBarrier() calls with the same id will return a
  // CANCELLED error status.
  // Possible service errors:
  //   - FailedPrecondition: Barrier has already been passed.
  // TODO(b/342448688): Allow re-use of ids by specifying different counters.
  // The counter field is mostly ignored at the moment with no user-facing
  // effect.
  absl::Status CancelBarrier(std::string barrier_id, int64_t counter,
                             const tensorflow::CoordinatedTask& task);

  // Returns the set of currently alive tasks. More specifically, given a set of
  // tasks T, GetAliveTasks(T) returns the subset T of alive tasks. Note that
  // `tasks` must include `requesting_task`.
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
  void GetAliveTasksAsync(const tensorflow::CoordinatedTask& requesting_task,
                          const std::vector<tensorflow::CoordinatedTask>& tasks,
                          GetAliveTasksCallback done);

  // Gets error from the coordination service. Block until the service
  // returns an error or the task/service is shutdown. This should never be used
  // when there is service to client connection (i.e. `CoordinationClientCache`
  // is passed in during construction).
  //
  // The first call to this function will trigger the error polling mode in the
  // coordination service, so once an error occurs after the first call, the
  // service will use the error polling mode to propagate the error to all
  // connected tasks instead of simply shutting down.
  void PollForErrorAsync(const tensorflow::CoordinatedTask& task,
                         StatusCallback done);

 private:
  friend class CoordinationServiceRpcHandler;
  friend class CoordinationServiceTest_ListClusterDevices_TfDevice_Test;
  friend class CoordinationServiceTest_ListClusterDevices_XlaDevice_Test;
  friend class
      CoordinationServiceTest_ListClusterDevices_DevicesAreNotAddedTwice_Test;

  void LogConnectStatusLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  const tensorflow::DeviceInfo& ListClusterDevices()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  IncarnationId GetServiceIncarnation();
  void BarrierAsyncLocked(
      absl::string_view barrier_id, int64_t counter, absl::Duration timeout,
      const tensorflow::CoordinatedTask& task,
      const std::vector<tensorflow::CoordinatedTask>& participating_tasks,
      BarrierCallback done) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  BarrierCallback ConnectAfterBarrierPasses(absl::string_view task_name,
                                            IncarnationId incarnation,
                                            StatusCallback done);
  // Connects a task to the service, and leaves any previously ongoing barriers
  // for recoverable tasks.
  void ConnectTask(const tensorflow::CoordinatedTask& task,
                   IncarnationId incarnation)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Checks if any task has stopped sending heartbeats.
  void CheckHeartbeatTimeout();
  // Checks if any barrier has timed out.
  void CheckBarrierTimeout();
  // Checks both heartbeat and barrier timeouts. Use a single function so they
  // can be run in the same thread as threads are a constrained resource.
  void CheckStaleness();
  // Starts a thread to check staleness.
  void StartCheckStaleness();
  void Stop() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool ServiceHasStopped() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Report error from a task to all other connected tasks if the task is not
  // recoverable.
  // Note: SetTaskError() must be called before propagating its error.
  void PropagateError(
      const absl::Status& error,
      const std::vector<tensorflow::CoordinatedTask>& source_tasks,
      bool is_reported_by_task = false)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void PropagateError(const absl::Status& error,
                      const std::vector<absl::string_view>& source_task_names,
                      bool is_reported_by_task = false)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Checks if all tasks are from recoverable jobs.
  bool AllTasksAreRecoverable(
      const std::vector<tensorflow::CoordinatedTask>& tasks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void SetTaskError(absl::string_view task_name, const absl::Status& error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Used for cluster-wide errors (e.g. register or shutdown barrier fails).
  void SetAllTasksError(const absl::Status& error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  absl::Status DisconnectTask(const tensorflow::CoordinatedTask& task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void DisconnectAllNonRecoverableTasks()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  std::vector<tensorflow::CoordinatedTask> GetTasksForShutdownBarrier();

  struct BarrierState {
    std::string id = "";
    // Counter is incremented for each new barrier using the same id.
    // No two barriers with the same id (and different counters) can be ongoing
    // at the same time.
    int64_t counter = 0;
    bool passed = false;
    absl::Status result = absl::UnknownError(
        "Invalid barrier result.");  // Only valid if `passed` is true.
    uint64_t deadline_in_micros = 0;
    int num_pending_tasks = 0;
    // Specifies which tasks have called the barrier so far.
    absl::flat_hash_map<tensorflow::CoordinatedTask, bool, CoordinatedTaskHash,
                        CoordinatedTaskEqual>
        tasks_at_barrier;
    absl::flat_hash_set<tensorflow::CoordinatedTask, CoordinatedTaskHash,
                        CoordinatedTaskEqual>
        recoverable_tasks_restarted_during_barrier;
    absl::flat_hash_map<tensorflow::CoordinatedTask, BarrierCallback,
                        CoordinatedTaskHash, CoordinatedTaskEqual>
        done_callbacks;
    // Specifies the task that initiated the barrier (the first task to call the
    // barrier).
    tensorflow::CoordinatedTask initiating_task;
  };
  bool BarrierIsUninitialized(const BarrierState& barrier) {
    return barrier.id.empty() && barrier.counter == 0 && !barrier.passed &&
           barrier.deadline_in_micros == 0 && barrier.num_pending_tasks == 0;
  }
  std::string BarrierName(absl::string_view barrier_id, int64_t counter) {
    return absl::StrCat(barrier_id, "::", counter);
  }
  std::string BarrierName(const BarrierState& barrier) {
    return BarrierName(barrier.id, barrier.counter);
  }
  // Initializes a new barrier. Returns false if the barrier should fail
  // immediately.
  bool InitializeBarrier(
      BarrierState* barrier, absl::string_view barrier_id, int64_t counter,
      absl::Duration timeout, const tensorflow::CoordinatedTask& task,
      const std::vector<tensorflow::CoordinatedTask>& participating_tasks,
      BarrierCallback done) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Initialize `BarrierState`'s tasks_at_barrier map.
  bool InitializeTasksAtBarrier(
      BarrierState* barrier,
      const std::vector<tensorflow::CoordinatedTask>& participating_tasks,
      BarrierCallback done) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Adds a callback to be called when the barrier is done.
  // If there is an existing callback for that task, it will be overwritten,
  // cancelling the previous callback.
  void AddBarrierCallback(BarrierState* barrier,
                          const tensorflow::CoordinatedTask& task,
                          BarrierCallback done)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Ends the barrier with a result (ok or error).
  void PassBarrier(BarrierState* barrier, const absl::Status& result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // A task reaches the barrier.
  void ReachBarrier(BarrierState* barrier,
                    const tensorflow::CoordinatedTask& task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void FailBarrierWithCounterMismatch(BarrierState* barrier, int64_t counter)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Propagates same result back to task.
  void RepeatBarrierResult(BarrierState* barrier,
                           const tensorflow::CoordinatedTask& task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Leaves any ongoing barriers.
  // If the task is non-recoverable, the barrier exits with an error.
  // If the task is recoverable, the barrier will 'unregister' a task and allow
  // it to join back again later before the timeout.
  void LeaveOngoingBarriers(const tensorflow::CoordinatedTask& task,
                            absl::string_view reason)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Post-barrier hook to connect all tasks.
  void ConnectAllTasks() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Post-barrier hook to aggregate device info.
  void AggregateClusterDevices() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Post-shutdown barrier hook to disconnect tasks that acked and propagate
  // errors to those that have not.
  void CompleteShutdownAfterBarrier(const absl::Status& result,
                                    BarrierState* barrier)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Checks if the participating tasks are specified correctly across barrier
  // calls and that the caller task is one of the participating tasks.
  bool ValidateTaskArgs(
      BarrierState* barrier, const tensorflow::CoordinatedTask& caller_task,
      const std::vector<tensorflow::CoordinatedTask>& tasks_args)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool isRecoverableJob(absl::string_view task_name) const;
  // Sends responses to error polling requests when an error is encountered.
  void SendErrorPollingResponse(const absl::Status& error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Responds to error polling or fails all tasks when an error is
  // encountered. Should only be called when there is no service to client
  // connection.
  void SendErrorPollingResponseOrFailAllTasks(const absl::Status& error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  // Returns whether the clients are polling for error from the service. If the
  // clients are not polling for error from the service, the service should stop
  // when there is an error. Otherwise, the service should not stop.
  bool IsClientPollingForError() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Checks if the barrier can be passed, if recoverable tasks reconnected or
  // disconnected to the service while barrier is ongoing.
  // This is only applicable if leave_barriers_on_recoverable_agent_restart flag
  // is set to true.
  void CheckBarrierStatusWithRecoverableTasks();

  // Returns a map of ongoing barriers to count of unsynced tasks waiting on
  // other barriers.
  absl::flat_hash_map<std::string, int> GetCountOfOutOfSyncTasksPerBarrier();

  class ErrorPollingState {
   public:
    // Returns whether the error polling requests have been responded.
    bool Responded() const { return responded_; }
    // Sets the error and executes the status callbacks.
    void SetError(const absl::Status& error);
    // Gets the error that is propagated to the agents.
    const absl::Status& GetError() const { return error_; }
    // Returns true if the task has sent request to poll for error from the
    // service.
    bool IsTaskPolling(absl::string_view task_name) const {
      return polling_task_names_.contains(task_name);
    }
    // Adds a task to the error polling state.
    void AddTask(const tensorflow::CoordinatedTask& task,
                 StatusCallback&& done);

    // Removes a task from the error polling state.
    // If an existing polling request is present, we will invoke the callback
    // with the `reason` argument.
    // Note: for disconnected tasks, this does not actually propagate the error
    // back, but prevents memory leaks by removing stale callbacks.
    void RemoveTask(const tensorflow::CoordinatedTask& task,
                    absl::string_view reason);

   private:
    bool responded_ = false;
    absl::Status error_ = absl::OkStatus();
    absl::flat_hash_map<tensorflow::CoordinatedTask, StatusCallback,
                        CoordinatedTaskHash, CoordinatedTaskEqual>
        done_callbacks_;
    absl::flat_hash_set<std::string> polling_task_names_;
  };

  class TaskState {
   public:
    // Task state maintained on the coordination service side.
    // State transition:
    //                Register           Heartbeat
    //   DISCONNECTED -------> CONNECTED --------> ERROR (timeout)
    //                              |   ReportError
    //                              +--------------> ERROR
    //
    // When task state becomes ERROR, propagate this status to other CONNECTED
    // tasks in the cluster.

    explicit TaskState(absl::string_view task) { task_name_ = task; }

    tensorflow::CoordinatedTaskState GetState() const { return state_; }
    absl::Status GetStatus() const { return status_; }
    bool IsRecoverable() const { return recoverable_; }
    void SetRecoverable(bool recoverable) { recoverable_ = recoverable; }
    IncarnationId GetTaskIncarnation() const { return task_incarnation_; }
    void SetTaskIncarnation(IncarnationId task_incarnation) {
      task_incarnation_ = task_incarnation;
    }
    void Connect() {
      SetConnected(task_incarnation_);
      LOG(INFO) << task_name_
                << " has connected to coordination service. Incarnation: "
                << task_incarnation_;
    }
    void SetConnected(IncarnationId task_incarnation);
    void Disconnect(uint64_t grace_period_duration_us);
    absl::Status RecordHeartbeat(IncarnationId task_incarnation);
    int64_t TimeSinceLastHeartbeatMs();
    // Sets the error and returns true if the task state is not ERROR.
    // Otherwise, don't overwrite the error and return false.
    bool SetError(const absl::Status& status);
    tensorflow::DeviceInfo GetDeviceInfo() { return devices_; }
    void CollectDeviceInfo(const tensorflow::DeviceInfo& devices) {
      devices_ = devices;
    }
    // Checks if task has called WaitForAllTasks() previously, which gathers the
    // local device info.
    bool DeviceInfoIsCollected() { return !devices_.device().empty(); }

    // This is used to propagate state changes (disconnect, error) to ongoing
    // barriers.
    absl::flat_hash_set<std::string> GetOngoingBarriers();
    // The task has a new ongoing barrier. This does not mean that it has
    // reached the barrier.
    void JoinBarrier(absl::string_view barrier_id);
    // The task has exited a barrier (because a barrier has passed).
    void ExitBarrier(absl::string_view barrier_id);
    // Returns true if the task has been disconnected beyond the grace period
    // and no further agent requests are expected. Note that the grace period
    // accounts for the lag time between the service recording the state change
    // and the agent stopping heartbeats/error polling.
    bool IsDisconnectedBeyondGracePeriod();

   private:
    std::string task_name_;
    // Incarnation ID for CPU:0 on remote task.
    IncarnationId task_incarnation_{0};

    tensorflow::CoordinatedTaskState state_ =
        tensorflow::CoordinatedTaskState::TASKSTATE_DISCONNECTED;
    absl::Status status_;
    absl::Mutex last_heartbeat_mu_;
    uint64_t last_heartbeat_us_ ABSL_GUARDED_BY(last_heartbeat_mu_);
    // This denotes the deadline after which we stop accepting heartbeats or
    // error polling requests from a disconnected task. This grace period
    // accounts for the lag time between the service recording the state change
    // and the agent stopping heartbeats/error polling.
    uint64_t disconnect_grace_period_us_ = 0;
    tensorflow::DeviceInfo devices_;
    // For now, we assume there won't be many simultaneous barriers so we simply
    // use a set.
    absl::flat_hash_set<std::string> ongoing_barriers_for_task_;
    // TODO(b/342448688): Re-use config's recoverable jobs instead.
    bool recoverable_ = false;
  };

  // AlivenessState tracks the state of pending GetAliveTasks calls.
  struct AlivenessState {
    // All tasks that can participate in the GetAliveTasks barrier.
    CoordinatedTaskSet tasks;
    // All tasks currently blocked on the barrier.
    CoordinatedTaskSet in_barrier;
    // Done callbacks for the tasks blocked on the barrier.
    std::vector<GetAliveTasksCallback> dones;
  };

  // Returns the set of alive tasks drawn from the provided set of tasks.
  CoordinatedTaskSet AliveTasks(const CoordinatedTaskSet& tasks) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Returns the incarnation ids of the provided tasks, in the same order.
  std::vector<IncarnationId> IncarnationIds(
      absl::Span<const tensorflow::CoordinatedTask> tasks) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Refreshes the AlivenessStates of all pending GetAliveTasks call,
  // potentially finishing some of the pending calls. The AlivenessStates should
  // be refreshed, for example, after a task has failed.
  void RefreshAliveness() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  static tensorflow::CoordinatedTaskStateInfo CreateTaskStateInfo(
      const tensorflow::CoordinatedTask& task, const TaskState& state);

  // Gets the task states for the provided job.
  std::vector<tensorflow::CoordinatedTaskStateInfo> GetJobState(
      absl::string_view job) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Notifies all callbacks registered via WatchJobState.
  void NotifyWatchJobStateCallbacks() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // This method should be called whenever the cluster state changes in a way
  // such that NotifyWatchJobStateCallbacks should be called.
  void ClusterStateUpdated() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  std::unique_ptr<CoordinationClientCache> client_cache_;
  Env& env_;
  const IncarnationId service_incarnation_{random::New64()};
  const uint64_t heartbeat_timeout_ms_;
  bool cluster_register_with_barrier_ = false;
  const absl::Duration cluster_register_timeout_;
  const absl::Duration shutdown_barrier_timeout_;
  // If a task restarts with a new incarnation, we may allow it to reconnect
  // silently if configured. This is useful when we know that a task can
  // immediately resume work upon re-connecting to the service.
  bool allow_new_incarnation_to_reconnect_ = false;

  std::function<tensorflow::DeviceInfo(const tensorflow::DeviceInfo& devices)>
      post_aggregate_device_fn_;

  const std::string device_propagation_barrier_id_ =
      absl::StrCat("WaitForAllTasks::", service_incarnation_.value());
  const std::string shutdown_barrier_id_ =
      absl::StrCat("Shutdown::", service_incarnation_.value());
  std::vector<tensorflow::CoordinatedTask> shutdown_barrier_tasks_
      ABSL_GUARDED_BY(state_mu_);

  absl::Mutex state_mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<TaskState>> cluster_state_
      ABSL_GUARDED_BY(state_mu_);
  int64_t cluster_state_version_number_ ABSL_GUARDED_BY(state_mu_) = 0;
  std::vector<std::tuple<std::string, WatchJobStateCallback>>
      watch_job_state_callbacks_ ABSL_GUARDED_BY(state_mu_);
  tensorflow::DeviceInfo cluster_devices_ ABSL_GUARDED_BY(state_mu_);

  KeyValueStore store_;

  absl::flat_hash_map<std::string, BarrierState> barriers_
      ABSL_GUARDED_BY(state_mu_);
  // For now, we assume there won't be many simultaneous barriers so we simply
  // use a set.
  absl::flat_hash_set<std::string> ongoing_barriers_ ABSL_GUARDED_BY(state_mu_);

  // The state of all pending GetAliveTasks calls.
  std::vector<AlivenessState> aliveness_states_ ABSL_GUARDED_BY(state_mu_);

  absl::flat_hash_set<std::string> recoverable_jobs_;

  // When the tasks connect to coordination service after cluster initialization
  // is done, they will be added to this set.
  // Tasks connecting after cluster initialization indicate that they
  // reconnected to the service due to preemption or restart.
  // Unsynced recoverable tasks will be excluded from the barrier check after
  // the first cluster initialization.
  // The service will remove them from the set when the tasks pass a
  // barrier with other tasks.
  absl::flat_hash_set<std::string> unsynced_recoverable_jobs_
      ABSL_GUARDED_BY(state_mu_);
  // Whether the agents are polling for error from the service. It will be set
  // to true when the service sees the first error polling request. Once set to
  // true, the value will never change back to false.
  bool client_polling_for_error_ ABSL_GUARDED_BY(state_mu_) = false;
  ErrorPollingState error_polling_state_ ABSL_GUARDED_BY(state_mu_);

  absl::CondVar check_staleness_thread_cv_;
  bool shutting_down_ ABSL_GUARDED_BY(state_mu_) = false;
  // Note: sequence matters here, we must destroy the staleness thread before
  // the other state related to barriers and heartbeats to prevent illegal
  // memory access.
  std::unique_ptr<Thread> check_staleness_thread_;

  CoordinationService(const CoordinationService&) = delete;
  void operator=(const CoordinationService&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
