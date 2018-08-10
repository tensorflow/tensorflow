/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_ENV_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_ENV_H_

#include <functional>
#include <vector>

#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class CollectiveExecutorMgrInterface;
class Device;
class DeviceSet;
class Env;
class MasterSession;
class OpRegistryInterface;

// Options passed to the worker_cache_factory function.
struct WorkerCacheFactoryOptions {
  const ClusterDef* cluster_def = nullptr;
  const string* job_name = nullptr;
  int task_index;
  const string* protocol = nullptr;

  WorkerCacheFactoryOptions() {}

  // Construct from a ServerDef proto.
  //
  // Note: server_def must outlive WorkerCacheFactoryOptions!
  WorkerCacheFactoryOptions(const ServerDef& server_def) {
    if (server_def.has_cluster() && !server_def.job_name().empty()) {
      cluster_def = &server_def.cluster();
      job_name = &server_def.job_name();
      task_index = server_def.task_index();
      protocol = &server_def.protocol();
    }
  }
};

// The master environment class, which holds a bag of pointers to
// per-master state.
//
// MasterEnv does not own its member pointers.
struct MasterEnv {
  Env* env = nullptr;

  // Object from which WorkerInterface instances can be obtained.
  WorkerCacheInterface* worker_cache = nullptr;

  // The operation definitions to use.  Must be filled before use.
  const OpRegistryInterface* ops = nullptr;

  // Local devices co-located with this master.  Devices are not owned
  // by the master service.
  //
  // REQUIRES: !local_devices.empty().
  std::vector<Device*> local_devices;

  // Factory for creating master sessions, given session options and a
  // vector of devices.
  //
  // The caller of the function takes ownership of the returned
  // `MasterSession`, which may not be null. Ownership of the
  // `MasterEnv*` is retained by the caller.
  std::function<MasterSession*(
      SessionOptions, MasterEnv*,
      std::unique_ptr<std::vector<std::unique_ptr<Device>>>,
      std::unique_ptr<WorkerCacheInterface>,
      std::unique_ptr<DeviceSet> device_set,
      std::vector<string> filtered_worker_list)>
      master_session_factory;

  std::function<Status(const WorkerCacheFactoryOptions&,
                       WorkerCacheInterface**)>
      worker_cache_factory;

  // Generates per-step CollectiveExecutors and has access to utilities
  // supporting collective operations.
  CollectiveExecutorMgrInterface* collective_executor_mgr = nullptr;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
