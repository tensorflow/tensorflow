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
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/protobuf/rpc_options.pb.h"

namespace tsl {
class Env;
}  // namespace tsl
namespace tensorflow {
using Env = tsl::Env;

class CollectiveExecutorMgrInterface;
class Device;
class DeviceSet;
class MasterSession;
class OpRegistryInterface;

// Options passed to the worker_cache_factory function.
struct WorkerCacheFactoryOptions {
  ClusterDef cluster_def;
  string job_name;
  int task_index;
  int replica_index = 0;
  RPCOptions rpc_options;

  explicit WorkerCacheFactoryOptions() = default;

  // Construct from a ServerDef proto.
  explicit WorkerCacheFactoryOptions(const ServerDef& server_def) {
    if (server_def.has_cluster() && !server_def.job_name().empty()) {
      cluster_def = server_def.cluster();
      job_name = server_def.job_name();
      task_index = server_def.task_index();
      rpc_options = server_def.default_session_config().rpc_options();
      replica_index = server_def.replica();
    }
  }
};

// The master environment class, which holds a bag of pointers to
// per-master state.
//
// MasterEnv does not own its member pointers.
struct MasterEnv {
  Env* env = nullptr;

  // Object from which WorkerInterface instances can be obtained. Not owned.
  WorkerCacheInterface* worker_cache = nullptr;

  // The operation definitions to use.  Must be filled before use.
  const OpRegistryInterface* ops = nullptr;

  // Local devices co-located with this master.  Devices are not owned
  // by the master service.
  //
  // REQUIRES: !local_devices.empty().
  std::vector<Device*> local_devices;

  // In large scaled distributed training, many singleton components (e.g.
  // Rendezvous) can becomes the bottleneck of the system. This field allows
  // us to shard the single components. This number will scale up with number
  // of tasks in this cluster. It is always greater than 1.
  int experimental_num_shards = 1;

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
  // supporting collective operations. Not owned.
  CollectiveExecutorMgrInterface* collective_executor_mgr = nullptr;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_ENV_H_
