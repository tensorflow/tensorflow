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

#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class Device;
class Env;
class MasterSession;
class OpRegistryInterface;
class WorkerCacheInterface;

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
  // `MasterEnv*` is retained by the caller. The callee takes
  // ownership of the `std::vector<Device*>*` argument, but does not
  // take ownership of the `Device*` objects in the vector.
  std::function<MasterSession*(const SessionOptions&, MasterEnv*,
                               std::vector<Device*>*)>
      master_session_factory;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
