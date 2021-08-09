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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_

#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
class DeviceAttributes;
class ServerDef;
class WorkerEnv;

// Static registration for coordination service implementations.
#define REGISTER_COORDINATION_SERVICE(service_type_name, factory_fn)        \
  REGISTER_COORDINATION_SERVICE_UNIQ_HELPER(__COUNTER__, service_type_name, \
                                            factory_fn)
#define REGISTER_COORDINATION_SERVICE_UNIQ_HELPER(counter, service_type_name, \
                                                  factory_fn)                 \
  static bool static_coordination_service_##counter TF_ATTRIBUTE_UNUSED =     \
      []() {                                                                  \
        ::tensorflow::CoordinationServiceInterface::                          \
            RegisterCoordinationService(service_type_name,                    \
                                        std::move(factory_fn));               \
        return true;                                                          \
      }()

// Coordination service is used for controlling and coordinating distributed
// execution in a cluster of multiple workers.
//
// When enabled, the service keeps track of cluster configurations and the state
// of cluster members. TF runtime and libraries can use it to orchastrate
// cluster initialization, check the healthiness of workers, and propagate error
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
//
// Experimental feature. Not yet implemented in open source.
class CoordinationServiceInterface {
 public:
  using CoordinationServiceFactory =
      std::function<std::unique_ptr<CoordinationServiceInterface>(
          const WorkerEnv* env, const ServerDef& server_def,
          std::unique_ptr<CoordinationClientCache> cache)>;

  using StatusOrValueCallback =
      std::function<void(const StatusOr<std::string>&)>;

  virtual ~CoordinationServiceInterface() {}

  static void RegisterCoordinationService(
      const std::string& service_type_name,
      CoordinationServiceFactory factory_fn) {
    auto factories = GetCoordinationServiceFactories();
    factories->emplace(service_type_name, factory_fn);
  }

  static std::unique_ptr<CoordinationServiceInterface>
  EnableCoordinationService(const std::string& service_type,
                            const WorkerEnv* env, const ServerDef& server_def,
                            std::unique_ptr<CoordinationClientCache> cache) {
    const auto* factories = GetCoordinationServiceFactories();
    auto factories_iter = factories->find(service_type);
    if (factories_iter == factories->end()) {
      LOG(ERROR) << "No coordination service factory found for service type "
                 << service_type;
      return nullptr;
    }
    auto service = factories_iter->second(env, server_def, std::move(cache));
    if (service != nullptr) {
      *GetCoordinationServiceInstancePtr() = service.get();
    }
    return service;
  }

  static CoordinationServiceInterface* GetCoordinationServiceInstance() {
    return *GetCoordinationServiceInstancePtr();
  }

  // Register a worker to the service.
  virtual void RegisterWorker(const std::string& job_name, const int task_id,
                              const uint64 incarnation,
                              std::vector<DeviceAttributes> devices,
                              StatusCallback done) = 0;

  // Wait for all tasks to be up and running. The callback is invoked when all
  // tasks are up and registered, or some error occurs.
  virtual void WaitForAllTasks(const std::string& job_name, const int task_id,
                               StatusCallback done) = 0;

  // Update the heartbeat timestamp of a task. This should only be invoked on
  // the leader of the cluster.
  virtual Status RecordHeartbeat(const std::string& job_name, const int task_id,
                                 const uint64 incarnation) = 0;

  // Set a task in error state permanently.
  virtual Status ReportTaskError(const std::string& job_name, const int task_id,
                                 Status error) = 0;

  // Insert a configuration key-value in the coordination service.
  // For now, a key-value can only be inserted once and cannot be updated.
  // The key-values are not persisted and will be lost if the leader fails.
  virtual Status InsertKeyValue(const std::string& key,
                                const std::string& value) = 0;

  // Get a configuration key-value from the coordination service. Block until
  // the key-value is available.
  virtual StatusOr<std::string> GetKeyValue(const std::string& key) = 0;
  // Get a configuration key-value from the coordination service. The `done`
  // callback is invoked when the key-value becomes available.
  virtual void GetKeyValueAsync(const std::string& key,
                                StatusOrValueCallback done) = 0;

  // Delete configuration key-value. If key is a directory, recursively clean
  // up all key-values under the directory.
  virtual Status DeleteKeyValue(const std::string& key) = 0;

 private:
  friend class CoordinationServiceRpcHandler;
  virtual const std::vector<DeviceAttributes>& ListClusterDevices() = 0;

  static std::unordered_map<std::string, CoordinationServiceFactory>*
  GetCoordinationServiceFactories() {
    static auto* coordination_service_factories =
        new std::unordered_map<std::string, CoordinationServiceFactory>();
    return coordination_service_factories;
  }

  // TODO(haoyuzhang): Remove singleton once we decide on how to access the
  // coordination service from op kernel.
  static CoordinationServiceInterface** GetCoordinationServiceInstancePtr() {
    static CoordinationServiceInterface* instance = nullptr;
    return &instance;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
