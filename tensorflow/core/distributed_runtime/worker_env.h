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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_

#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace thread {
class ThreadPool;
}  // namespace thread

namespace eager {
class EagerClientCache;
}  // namespace eager

class CollectiveExecutorMgrInterface;
class Device;
class DeviceMgr;
class Env;
class RendezvousMgrInterface;
class SessionMgr;
class ServerDef;

typedef std::function<Status(const ServerDef&,
                             std::unique_ptr<eager::EagerClientCache>*)>
    EagerClientCacheFactory;

// The worker environment class, which holds a bag of pointers to
// per-worker singletons.
//
// WorkerEnv does not own its member pointers.
struct WorkerEnv {
  Env* env = nullptr;

  // session_mgr encapsulates state for each session.
  SessionMgr* session_mgr = nullptr;

  // The local devices of this worker. Devices are owned by the device_mgr.
  //
  // REQUIRES: !local_devices.empty().
  std::vector<Device*> local_devices;

  // device_mgr manages local devices (cpu and gpu). The WorkerService
  // is the network interface for managed devices.
  //
  // Note: Please use the device_mgr associated with your session if appropriate
  // instead of this one. Using this device_mgr does not support ClusterSpec
  // propagated sessions.
  DeviceMgr* device_mgr = nullptr;

  // A set of rendezvous keyed by step ids.
  RendezvousMgrInterface* rendezvous_mgr = nullptr;

  // Generates per-step CollectiveExecutors and has access to utilities
  // supporting collective operations.
  CollectiveExecutorMgrInterface* collective_executor_mgr = nullptr;

  // A pool of threads for scheduling compute work.
  thread::ThreadPool* compute_pool = nullptr;

  // A factory function to create eager client cache.
  EagerClientCacheFactory eager_client_cache_factory =
      [](const ServerDef& s, std::unique_ptr<eager::EagerClientCache>* c) {
        return errors::Unimplemented(
            "EagerClientCacheFactory unimplemented. "
            "It is probably because you didn't use GRPC. Right now "
            "EagerClient only supports GRPC.");
      };
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_
