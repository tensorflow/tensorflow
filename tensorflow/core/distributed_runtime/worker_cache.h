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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_H_

#include <string>
#include <vector>

#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/device_attributes.pb.h"  // for DeviceLocality
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
typedef std::function<void(const Status&)> StatusCallback;

class ChannelCache;
class StepStats;

class WorkerCacheInterface {
 public:
  virtual ~WorkerCacheInterface() {}

  // Updates *workers with strings naming the remote worker tasks to
  // which open channels have been established.
  virtual void ListWorkers(std::vector<string>* workers) const = 0;
  virtual void ListWorkersInJob(const string& job_name,
                                std::vector<string>* workers) const = 0;

  // If "target" names a remote task for which an RPC channel exists
  // or can be constructed, returns a pointer to a WorkerInterface object
  // wrapping that channel. The returned value must be destroyed by
  // calling `this->ReleaseWorker(target, ret)`
  virtual WorkerInterface* GetOrCreateWorker(const string& target) = 0;

  // Release a worker previously returned by this->GetOrCreateWorker(target).
  //
  // TODO(jeff,sanjay): Consider moving target into WorkerInterface.
  // TODO(jeff,sanjay): Unify all worker-cache impls and factor out a
  //                    per-rpc-subsystem WorkerInterface creator.
  virtual void ReleaseWorker(const string& target, WorkerInterface* worker) {
    // Subclasses may override to reuse worker objects.
    delete worker;
  }

  // Set *locality with the DeviceLocality of the specified remote device
  // within its local environment.  Returns true if *locality
  // was set, using only locally cached data.  Returns false
  // if status data for that device was not available.  Never blocks.
  virtual bool GetDeviceLocalityNonBlocking(const string& device,
                                            DeviceLocality* locality) = 0;

  // Set *locality with the DeviceLocality of the specified remote device
  // within its local environment.  Callback gets Status::OK if *locality
  // was set.
  virtual void GetDeviceLocalityAsync(const string& device,
                                      DeviceLocality* locality,
                                      StatusCallback done) = 0;

  // TODO(b/189159585): Define a general client cache maker function to
  // construct client cache of different types sharing the same underling RPC
  // channels, to replace the eager and coordination cache function.
  // Build and return a EagerClientCache object wrapping that channel.
  virtual Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) = 0;

  // Build and return a CoordinationClientCache object wrapping that channel.
  virtual Status GetCoordinationClientCache(
      std::unique_ptr<CoordinationClientCache>* coordination_client_cache) = 0;

  // Start/stop logging activity.
  virtual void SetLogging(bool active) {}

  // Discard any saved log data.
  virtual void ClearLogs() {}

  // Return logs for the identified step in *ss.  Any returned data will no
  // longer be stored.
  virtual bool RetrieveLogs(int64_t step_id, StepStats* ss) { return false; }
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_H_
