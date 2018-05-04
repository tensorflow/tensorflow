/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_WRAPPER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_WRAPPER_H_

#include <string>
#include <vector>

#include "tensorflow/core/distributed_runtime/worker_cache.h"

namespace tensorflow {

class WorkerCacheWrapper : public WorkerCacheInterface {
 public:
  WorkerCacheWrapper(WorkerCacheInterface* wrapped) : wrapped_(wrapped) {}

  // Updates *workers with strings naming the remote worker tasks to
  // which open channels have been established.
  virtual void ListWorkers(std::vector<string>* workers) const {
    return wrapped_->ListWorkers(workers);
  }

  // If "target" names a remote task for which an RPC channel exists
  // or can be constructed, returns a pointer to a WorkerInterface object
  // wrapping that channel. The returned value must be destroyed by
  // calling `this->ReleaseWorker(target, ret)`
  // TODO(mrry): rename this to GetOrCreateWorker() or something that
  // makes it more obvious that this method returns a potentially
  // shared object.
  virtual WorkerInterface* CreateWorker(const string& target) {
    return wrapped_->CreateWorker(target);
  }

  // Release a worker previously returned by this->CreateWorker(target).
  //
  // TODO(jeff,sanjay): Consider moving target into WorkerInterface.
  // TODO(jeff,sanjay): Unify all worker-cache impls and factor out a
  //                    per-rpc-subsystem WorkerInterface creator.
  virtual void ReleaseWorker(const string& target, WorkerInterface* worker) {
    return wrapped_->ReleaseWorker(target, worker);
  }

  // Set *locality with the DeviceLocality of the specified remote device
  // within its local environment.  Returns true if *locality
  // was set, using only locally cached data.  Returns false
  // if status data for that device was not available.  Never blocks.
  virtual bool GetDeviceLocalityNonBlocking(const string& device,
                                            DeviceLocality* locality) {
    return wrapped_->GetDeviceLocalityNonBlocking(device, locality);
  }

  // Set *locality with the DeviceLocality of the specified remote device
  // within its local environment.  Callback gets Status::OK if *locality
  // was set.
  virtual void GetDeviceLocalityAsync(const string& device,
                                      DeviceLocality* locality,
                                      StatusCallback done) {
    return wrapped_->GetDeviceLocalityAsync(device, locality, std::move(done));
  }

  // Start/stop logging activity.
  virtual void SetLogging(bool active) { wrapped_->SetLogging(active); }

  // Discard any saved log data.
  virtual void ClearLogs() { wrapped_->ClearLogs(); }

  // Return logs for the identified step in *ss.  Any returned data will no
  // longer be stored.
  virtual bool RetrieveLogs(int64 step_id, StepStats* ss) {
    return wrapped_->RetrieveLogs(step_id, ss);
  }

 private:
  WorkerCacheInterface* wrapped_;  // Not owned.
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_WRAPPER_H_
