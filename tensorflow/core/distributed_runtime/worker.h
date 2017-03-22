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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"

namespace tensorflow {

class CancellationManager;
class Device;
class WorkerEnv;

// A TensorFlow Worker runs registered graphs and supports worker-to-worker
// Tensor transfer.
//
// See `../protobuf/worker_service.proto` for more details about each method.
//
// This class may be subclassed to provide specialized implementations of
// particular methods for different transport mechanism. For example,
// `GrpcWorker` specializes the `RecvTensorAsync()` method to support a more
// efficient gRPC data structure for handling large binary data.
class Worker : public WorkerInterface {
 public:
  Worker(WorkerEnv* env);
  virtual ~Worker() {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override;

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override;

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override;

  void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override;

  MutableRunGraphRequestWrapper* CreateRunGraphRequest() override;

  MutableRunGraphResponseWrapper* CreateRunGraphResponse() override;

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override;

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override;

  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override;

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override;

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override;

 protected:
  WorkerEnv* const env_;  // Not owned.

  Status PrepareRecvTensor(const Rendezvous::ParsedKey& parsed,
                           Device** src_dev);

  void AbortStep(int64);

 private:
  mutex mu_;
  CancellationManager* cancellation_manager_ GUARDED_BY(mu_);

  struct PartialRunState {
    CancellationManager* cancellation_manager;

    bool executor_done = false;
    StatusCallback final_callback = nullptr;
    Status final_status;

    explicit PartialRunState(CancellationManager* cm)
        : cancellation_manager(cm) {}
  };
  struct PairHash {
    std::size_t operator()(std::pair<string, int> const& p) const {
      return Hash64Combine(std::hash<string>()(p.first),
                           std::hash<int>()(p.second));
    }
  };
  std::unordered_map<std::pair<string, int>, std::unique_ptr<PartialRunState>,
                     PairHash>
      partial_runs_ GUARDED_BY(mu_);

  PartialRunState* FindPartialRun(const string& graph_handle, int step_id);

  void InsertPartialRunLocked(const string& graph_handle, int step_id,
                              PartialRunState* partial_run_state)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void RemovePartialRun(const string& graph_handle, int step_id);

  void MaybeCallFinalCallback(const string& graph_handle, int step_id,
                              const Status& executor_status);

  void SetOrCallFinalCallback(const string& graph_handle, int step_id,
                              StatusCallback done, const Status& s);

  Status PrepareRunGraph(RunGraphRequestWrapper* req,
                         GraphMgr::NamedTensors* in,
                         GraphMgr::NamedTensors* out);

  void DoRunGraph(CallOptions* opts, RunGraphRequestWrapper* request,
                  MutableRunGraphResponseWrapper* response,
                  StatusCallback done);

  void DoPartialRunGraph(CallOptions* opts, RunGraphRequestWrapper* request,
                         MutableRunGraphResponseWrapper* response,
                         StatusCallback done);

  TF_DISALLOW_COPY_AND_ASSIGN(Worker);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
