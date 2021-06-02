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

#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
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
// Experimental feature. Not yet implemented in open source.
class CoordinationServiceInterface {
 public:
  using CoordinationServiceFactory =
      std::function<std::unique_ptr<CoordinationServiceInterface>(
          WorkerEnv* env, const ServerDef& server_def,
          std::unique_ptr<CoordinationClientCache> cache,
          StatusCallback error_fn)>;

  virtual ~CoordinationServiceInterface() {}

  static void RegisterCoordinationService(
      const std::string& service_type_name,
      CoordinationServiceFactory factory_fn) {
    auto factories = GetCoordinationServiceFactories();
    factories->emplace(service_type_name, factory_fn);
  }

  static std::unique_ptr<CoordinationServiceInterface>
  EnableCoordinationService(const std::string& service_type, WorkerEnv* env,
                            const ServerDef& server_def,
                            std::unique_ptr<CoordinationClientCache> cache,
                            StatusCallback error_fn) {
    const auto* factories = GetCoordinationServiceFactories();
    auto factories_iter = factories->find(service_type);
    if (factories_iter == factories->end()) {
      LOG(ERROR) << "No coordination service factory found for service type "
                 << service_type;
      return nullptr;
    }
    return factories_iter->second(env, server_def, std::move(cache),
                                  std::move(error_fn));
  }

  // Start coordination service. This is a blocking call and will only return
  // when all member in the cluster have started, or some errors occur.
  virtual Status Start() = 0;

  // Stop the service and shutdown its internal threads. The service is then
  // ready to be deleted.
  virtual void Stop() = 0;

  // Set the service in error state permanently.
  virtual void SetError(Status error) = 0;

  // Register a worker to the leader. Should only be invoked on the leader of
  // the cluster.
  virtual void RegisterWorker(const std::string& job_name, const int task_id,
                              const uint64 incarnation,
                              StatusCallback done) = 0;

  // Update the heartbeat timestamp of a worker. This should only be invoked on
  // the leader of the cluster.
  virtual Status RecordHeartbeat(const std::string& job_name, const int task_id,
                                 const uint64 incarnation) = 0;

 private:
  static std::unordered_map<std::string, CoordinationServiceFactory>*
  GetCoordinationServiceFactories() {
    static auto* coordination_service_factories =
        new std::unordered_map<std::string, CoordinationServiceFactory>();
    return coordination_service_factories;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_H_
