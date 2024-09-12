/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TEST_UTILS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TEST_UTILS_H_

#include <unordered_map>
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// Some utilities for testing distributed-mode components in a single process
// without RPCs.

// Implements the worker interface with methods that just respond with
// "unimplemented" status.  Override just the methods needed for
// testing.
class TestWorkerInterface : public WorkerInterface {
 public:
  void GetStatusAsync(CallOptions* opts, const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
    done(errors::Unimplemented("GetStatusAsync"));
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    done(errors::Unimplemented("CreateWorkerSessionAsync"));
  }

  void DeleteWorkerSessionAsync(CallOptions* opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
    done(errors::Unimplemented("DeleteWorkerSessionAsync"));
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    done(errors::Unimplemented("RegisterGraphAsync"));
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    done(errors::Unimplemented("DeregisterGraphAsync"));
  }

  void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    done(errors::Unimplemented("RunGraphAsync"));
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    done(errors::Unimplemented("CleanupGraphAsync"));
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    done(errors::Unimplemented("CleanupAllAsync"));
  }

  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("RecvTensorAsync"));
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    done(errors::Unimplemented("LoggingAsync"));
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    done(errors::Unimplemented("TracingAsync"));
  }

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("RecvBufAsync"));
  }

  void CompleteGroupAsync(CallOptions* opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    done(errors::Unimplemented("CompleteGroupAsync"));
  }

  void CompleteInstanceAsync(CallOptions* ops,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    done(errors::Unimplemented("CompleteInstanceAsync"));
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
    done(errors::Unimplemented("GetStepSequenceAsync"));
  }
};

class TestWorkerCache : public WorkerCacheInterface {
 public:
  virtual ~TestWorkerCache() {}

  void AddWorker(const string& target, WorkerInterface* wi) {
    workers_[target] = wi;
  }

  void AddDevice(const string& device_name, const DeviceLocality& dev_loc) {
    localities_[device_name] = dev_loc;
  }

  void ListWorkers(std::vector<string>* workers) const override {
    workers->clear();
    for (auto it : workers_) {
      workers->push_back(it.first);
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    workers->clear();
    for (auto it : workers_) {
      DeviceNameUtils::ParsedName device_name;
      CHECK(DeviceNameUtils::ParseFullName(it.first, &device_name));
      CHECK(device_name.has_job);
      if (job_name == device_name.job) {
        workers->push_back(it.first);
      }
    }
  }

  WorkerInterface* GetOrCreateWorker(const string& target) override {
    auto it = workers_.find(target);
    if (it != workers_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {}

  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
    return errors::Unimplemented("Unimplemented.");
  }

  Status GetCoordinationClientCache(
      std::unique_ptr<CoordinationClientCache>* coord_client_cache) override {
    return errors::Unimplemented("Unimplemented.");
  }

  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
    auto it = localities_.find(device);
    if (it != localities_.end()) {
      *locality = it->second;
      return true;
    }
    return false;
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
    auto it = localities_.find(device);
    if (it != localities_.end()) {
      *locality = it->second;
      done(absl::OkStatus());
      return;
    }
    done(errors::Internal("Device not found: ", device));
  }

 protected:
  std::unordered_map<string, WorkerInterface*> workers_;
  std::unordered_map<string, DeviceLocality> localities_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TEST_UTILS_H_
