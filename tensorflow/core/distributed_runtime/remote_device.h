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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_

#include <functional>
#include <string>
#include <vector>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Device;
class Env;
class WorkerCacheInterface;

// NewRemoteDevices discovers available devices on the
// 'remote_worker'.  The implementation uses 'channel_cache' to
// discover how to communicate with the 'remote_worker' (via gRPC, for
// example).
//
// NewRemoteDevices does not block.
//
// On success, the 'done' callback is given the OK status and a vector
// of Device*. The caller should take ownership of these devices.
//
// Otherwise, the 'done' callback is given an error status and the
// vector is empty.
typedef std::function<void(const Status&, std::vector<Device*>*)>
    NewRemoteDevicesDone;
void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache,
                      const string& remote_worker, NewRemoteDevicesDone done);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
