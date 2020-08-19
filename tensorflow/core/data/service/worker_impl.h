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
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/data/experimental/service_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace data {

// A TensorFlow DataService serves dataset elements over RPC.
class DataServiceWorkerImpl {
 public:
  explicit DataServiceWorkerImpl(const experimental::WorkerConfig& config);
  ~DataServiceWorkerImpl();

  // Starts the worker. The worker needs to know its own address so that it can
  // register with the dispatcher. This is set in `Start` instead of in the
  // constructor because the worker may be binding to port `0`, in which case
  // the address isn't known until the worker has started and decided which port
  // to bind to.
  Status Start(const std::string& worker_address);

  // See worker.proto for API documentation.

  /// Dispatcher-facing API.
  Status ProcessTask(const ProcessTaskRequest* request,
                     ProcessTaskResponse* response);

  /// Client-facing API.
  Status GetElement(const GetElementRequest* request,
                    GetElementResponse* response);

 private:
  struct Task {
    explicit Task(TaskDef task_def) : task_def(std::move(task_def)) {}

    TaskDef task_def;
    mutex mu;
    bool initialized TF_GUARDED_BY(mu) = false;
    // TODO(aaudibert): Have standalone::Iterator own a reference to
    // standalone::Dataset so that we don't need to store the dataset here.
    std::unique_ptr<standalone::Dataset> dataset;
    std::unique_ptr<standalone::Iterator> iterator;
  };

  Status MakeDispatcherStub(std::unique_ptr<DispatcherService::Stub>* stub);
  // Registers the worker with the dispatcher.
  Status Register(DispatcherService::Stub* dispatcher) LOCKS_EXCLUDED(mu_);
  // Sends task status to the dispatcher and checks for dispatcher commands.
  Status SendTaskUpdates(DispatcherService::Stub* dispatcher)
      LOCKS_EXCLUDED(mu_);
  // Creates an iterator to process a task.
  Status ProcessTaskInternal(const TaskDef& task) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status EnsureTaskInitialized(Task& task);
  // A thread for doing async background processing not associated with a
  // specific RPC, such as reporting finished tasks. The thread takes
  // ownership of the passed dispatcher_ptr. We use a raw pointer instead of
  // unique_ptr since unique_ptr cannot be passed to std::function.
  void BackgroundThread(DispatcherService::Stub* dispatcher_ptr)
      LOCKS_EXCLUDED(mu_);

  const experimental::WorkerConfig config_;
  // The worker's own address.
  std::string worker_address_;

  mutex mu_;
  // Information about tasks, keyed by task ids.
  absl::flat_hash_map<int64, std::unique_ptr<Task>> tasks_ TF_GUARDED_BY(mu_);
  // Completed tasks which haven't yet been communicated to the dispatcher.
  absl::flat_hash_set<int64> pending_completed_tasks_ TF_GUARDED_BY(mu_);
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  // Whether the worker has registered with the dispatcher yet.
  bool registered_ TF_GUARDED_BY(mu_) = false;
  // Condition variable for notifying the background thread.
  condition_variable background_cv_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Thread> background_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataServiceWorkerImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_WORKER_IMPL_H_
