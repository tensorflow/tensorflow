/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace thread {
class ThreadPool;
}  // namespace thread

class DeviceMgr;
class Env;
class GraphMgr;
class RendezvousMgrInterface;
class WorkerCacheInterface;

// The worker environment class, which holds a bag of pointers to
// per-worker singletons.
//
// WorkerEnv does not own its member pointers.
struct WorkerEnv {
  Env* env = nullptr;

  // The name of the worker. E.g., /job:mnist/replica:1/task:0.
  string worker_name;

  // Object from which WorkerInterface instances can be obtained.
  WorkerCacheInterface* worker_cache = nullptr;

  // device_mgr manages local devices (cpu and gpu). The WorkerService
  // is the network interface for managed devices.
  DeviceMgr* device_mgr = nullptr;

  // graph_mgr keeps track of registered graphs of this worker.
  GraphMgr* graph_mgr = nullptr;

  // A set of rendezvous keyed by step ids.
  RendezvousMgrInterface* rendezvous_mgr = nullptr;

  // A pool of threads for scheduling compute work.
  thread::ThreadPool* compute_pool = nullptr;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_ENV_H_
