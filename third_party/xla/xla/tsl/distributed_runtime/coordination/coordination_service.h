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
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/protobuf/coordination_config.pb.h"

namespace tsl {
class Env;

// Static registration for coordination service implementations.
#define REGISTER_COORDINATION_SERVICE(service_type_name, factory_fn)        \
  REGISTER_COORDINATION_SERVICE_UNIQ_HELPER(__COUNTER__, service_type_name, \
                                            factory_fn)
#define REGISTER_COORDINATION_SERVICE_UNIQ_HELPER(counter, service_type_name, \
                                                  factory_fn)                 \
  static bool static_coordination_service_##counter TF_ATTRIBUTE_UNUSED =     \
      []() {                                                                  \
        ::tsl::CoordinationServiceInterface::RegisterCoordinationService(     \
            service_type_name, std::move(factory_fn));                        \
        return true;                                                          \
      }()

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
class CoordinationServiceInterface {
 public:
  using CoordinationServiceFactory =
      std::function<std::unique_ptr<CoordinationServiceInterface>(
          Env* env, const tensorflow::CoordinationServiceConfig& config,
          std::unique_ptr<CoordinationClientCache> cache)>;

  using StatusOrValueCallback =
      std::function<void(const absl::StatusOr<std::string>&)>;

  virtual ~CoordinationServiceInterface() = default;

  static void RegisterCoordinationService(
      const std::string& service_type_name,
      CoordinationServiceFactory factory_fn) {
    auto factories = GetCoordinationServiceFactories();
    factories->emplace(service_type_name, factory_fn);
  }

  static std::unique_ptr<CoordinationServiceInterface>
  EnableCoordinationService(Env* env,
                            const tensorflow::CoordinationServiceConfig& config,
                            std::unique_ptr<CoordinationClientCache> cache) {
    const auto* factories = GetCoordinationServiceFactories();
    auto factories_iter = factories->find(config.service_type());
    if (factories_iter == factories->end()) {
      LOG(ERROR) << "No coordination service factory found for service type "
                 << config.service_type();
      return nullptr;
    }
    auto service = factories_iter->second(env, config, std::move(cache));
    if (service != nullptr) {
      *GetCoordinationServiceInstancePtr() = service.get();
    }
    return service;
  }

  static CoordinationServiceInterface* GetCoordinationServiceInstance() {
    return *GetCoordinationServiceInstancePtr();
  }

  // This function is invoked after each task's local devices are appended in a
  // deterministic order during WaitForAllTasks(). This is useful to convert the
  // result into another message, or set global device ids.
  virtual void SetDeviceAggregationFunction(
      std::function<
          tensorflow::DeviceInfo(const tensorflow::DeviceInfo& devices)>
          post_aggregate_device_fn) = 0;

  // Register a task to the service.
  // Possible service errors:
  //   - Internal: Service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - Aborted: (1) task is in error state, or (2) task is in connected state
  //       with a different incarnation, indicating that it restarted.
  virtual absl::Status RegisterTask(const tensorflow::CoordinatedTask& task,
                                    uint64_t incarnation) = 0;

  // Wait for all tasks to be up and running, and register local device
  // info. The callback is invoked when all tasks are up and registered, or some
  // error occurs.
  // Each task's local devices will be appended in a deterministic order, and
  // post-processed by the callback in SetDeviceAggregationFunction() (if set).
  virtual void WaitForAllTasks(const tensorflow::CoordinatedTask& task,
                               const tensorflow::DeviceInfo& devices,
                               StatusCallback done) = 0;

  // Disconnects task from the service. If `shutdown_barrier_timeout_in_ms` is
  // specified in the config, blocks until all tasks reach the barrier before
  // disconnecting together.
  // Possible service errors:
  //   - Internal: Service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task has already disconnected.
  virtual void ShutdownTaskAsync(const tensorflow::CoordinatedTask& task,
                                 StatusCallback done) = 0;

  // Disconnects task from the service and cleans up its internal error state.
  // Possible service errors:
  //   - Internal: Service has shut down.
  //   - InvalidArgument: Unexpected task request.
  //   - FailedPrecondition: task has already disconnected.
  virtual absl::Status ResetTask(const tensorflow::CoordinatedTask& task) = 0;

  // Update the heartbeat timestamp of a task. This should only be invoked on
  // the leader of the cluster.
  //   - Internal: Service has shut down.
  virtual absl::Status RecordHeartbeat(const tensorflow::CoordinatedTask& task,
                                       uint64_t incarnation) = 0;

  // Set a task in error state permanently.
  virtual absl::Status ReportTaskError(const tensorflow::CoordinatedTask& task,
                                       absl::Status error) = 0;

  // Get the state and the error status of the tasks.
  virtual std::vector<tensorflow::CoordinatedTaskStateInfo> GetTaskState(
      const std::vector<tensorflow::CoordinatedTask>& task) = 0;

  // Insert a configuration key-value in the coordination service.
  // For now, a key-value can only be inserted once and cannot be updated.
  // The key-values are not persisted and will be lost if the leader fails.
  virtual absl::Status InsertKeyValue(const std::string& key,
                                      const std::string& value) = 0;

  // Get a configuration key-value from the coordination service. The `done`
  // callback is invoked when the key-value becomes available.
  virtual void GetKeyValueAsync(const std::string& key,
                                StatusOrValueCallback done) = 0;

  // Get a configuration key-value from the coordination service. If the key
  // does not exist, return NotFound error.
  virtual absl::StatusOr<std::string> TryGetKeyValue(
      const std::string& key) = 0;

  // Gets all values under a directory (key).
  // A value is considered to be in the directory if its key is prefixed with
  // the directory. This is not a blocking call. Agent does not need to be
  // connected to utilize the distributed key-value store.
  virtual std::vector<tensorflow::KeyValueEntry> GetKeyValueDir(
      absl::string_view directory_key) = 0;

  // Delete configuration key-value. If key is a directory, recursively clean
  // up all key-values under the directory.
  virtual absl::Status DeleteKeyValue(const std::string& key) = 0;

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
  //   - Internal: Any participating task is in ERROR state, or coordination
  //   -   service has shut down.
  //   - InvalidArgument: (1) Conflicting tasks specified by different agents
  //       for the same barrier, (2) one of the participating tasks is not in
  //       the cluster, or (3) task making the request is not included in the
  //       list of participating tasks.
  //   - FailedPrecondition: Agent is in UNINITIALIZED or ERROR state.
  virtual void BarrierAsync(
      const std::string& barrier_id, absl::Duration timeout,
      const tensorflow::CoordinatedTask& task,
      const std::vector<tensorflow::CoordinatedTask>& participating_tasks,
      StatusCallback done) = 0;

  // Aborts the barrier if it is ongoing.
  // Current and future WaitAtBarrier() calls with the same id will return a
  // CANCELLED error status.
  // Possible service errors:
  //   - FailedPrecondition: Barrier has already been passed.
  virtual absl::Status CancelBarrier(
      const std::string& barrier_id,
      const tensorflow::CoordinatedTask& task) = 0;

 private:
  friend class CoordinationServiceRpcHandler;
  friend class CoordinationServiceTest_ListClusterDevices_TfDevice_Test;
  friend class CoordinationServiceTest_ListClusterDevices_XlaDevice_Test;
  friend class
      CoordinationServiceTest_ListClusterDevices_DevicesAreNotAddedTwice_Test;

  virtual const tensorflow::DeviceInfo& ListClusterDevices() = 0;
  virtual uint64_t GetServiceIncarnation() = 0;

  static std::unordered_map<std::string, CoordinationServiceFactory>*
  GetCoordinationServiceFactories() {
    static auto* coordination_service_factories =
        new std::unordered_map<std::string, CoordinationServiceFactory>();
    return coordination_service_factories;
  }

  static CoordinationServiceInterface** GetCoordinationServiceInstancePtr() {
    static CoordinationServiceInterface* instance = nullptr;
    return &instance;
  }
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
