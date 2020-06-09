/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace data {

// A TensorFlow DataService serves dataset elements over RPC.
class DataServiceWorkerImpl {
 public:
  explicit DataServiceWorkerImpl(const std::string& master_address,
                                 const std::string& protocol);
  ~DataServiceWorkerImpl();

  // Starts the worker. The worker needs to know its own address so that it can
  // register with the master.
  void Start(const std::string& worker_address);

  // See worker.proto for API documentation.

  /// Master-facing API.
  Status ProcessTask(const ProcessTaskRequest* request,
                     ProcessTaskResponse* response);

  /// Client-facing API.
  Status GetElement(const GetElementRequest* request,
                    GetElementResponse* response);

 private:
  // Sets master_stub_ if it isn't already set.
  Status EnsureMasterStubInitialized();
  // Registers the worker with the master.
  Status Register();
  // Sends task status to the master.
  Status SendTaskUpdate();
  // Creates an iterator to process a task.
  Status ProcessTaskInternal(const TaskDef& task);
  // A thread for updating the master with worker status.
  void HeartbeatThread();

  typedef struct Task {
    int64 id;
    // TODO(aaudibert): Have standalone::Iterator own a reference to
    // standalone::Dataset so that we don't need to store the dataset here.
    std::unique_ptr<standalone::Dataset> dataset;
    std::unique_ptr<standalone::Iterator> iterator;
  } Task;

  const std::string master_address_;
  // Protocol for communicating with the master.
  const std::string protocol_;
  // The worker's own address.
  std::string worker_address_;

  mutex mu_;
  int64 worker_id_ TF_GUARDED_BY(mu_);
  std::unique_ptr<MasterService::Stub> master_stub_ TF_GUARDED_BY(mu_);
  // Information about tasks, keyed by task ids.
  absl::flat_hash_map<int64, Task> tasks_ TF_GUARDED_BY(mu_);
  // List of completed tasks which haven't yet been communicated to the master.
  std::vector<int64> pending_completed_tasks_ TF_GUARDED_BY(mu_);
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  // Condition variable for notifying the heartbeat thread.
  condition_variable heartbeat_cv_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Thread> heartbeat_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataServiceWorkerImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
