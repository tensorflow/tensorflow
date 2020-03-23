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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_MASTER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_MASTER_IMPL_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/master.pb.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace data {

// A service which coordinates a pool of workers to serve dataset elements over
// RPC.
//
// Glossary:
// * Dataset: A definition of how to generate a potentially large collection of
//   elements.
// * Epoch: A single pass over a dataset. There may be multiple epochs
//   for the same dataset, and they can be iterated over independently
// * Task: An epoch is broken into multiple tasks, which each represent
//   iterating over all of or part of the dataset. Workers process tasks. We
//   don't currently implement dataset splitting, so every task represents a
//   full iteration over the dataset. In the future, we will partition the data
//   across all tasks for the same epoch.
class DataServiceMasterImpl {
 public:
  explicit DataServiceMasterImpl(const std::string protocol);

  // See master.proto for API documentation.

  /// Worker-facing API.
  Status RegisterWorker(const RegisterWorkerRequest* request,
                        RegisterWorkerResponse* response);

  /// Client-facing API.
  Status GetOrRegisterDataset(const GetOrRegisterDatasetRequest* request,
                              GetOrRegisterDatasetResponse* response);
  Status BeginEpoch(const BeginEpochRequest* request,
                    BeginEpochResponse* response);
  Status GetTasks(const GetTasksRequest* request, GetTasksResponse* response);

 private:
  typedef struct WorkerInfo {
    std::string address;
    int64 id;
    std::unique_ptr<WorkerService::Stub> stub;

    std::string DebugString() {
      return absl::StrCat("id: ", id, "address: ", address);
    }
  } WorkerInfo;

  typedef struct Dataset {
    int64 id;
    int64 fingerprint;
    DatasetDef dataset_def;
  } Dataset;

  typedef struct Epoch {
    int64 id;
    int64 dataset_id;
    std::vector<int64> task_ids;
  } Epoch;

  typedef struct Task {
    int64 id;
    int64 dataset_id;
    std::string worker_address;
  } Task;

  // Registers a dataset with the given fingerprint, returning a new dataset id.
  int64 RegisterDataset(uint64 fingerprint, const DatasetDef& dataset);
  // Instructs a worker to begin processing a task.
  Status AllocateTaskToWorker(const Task& task_id, WorkerInfo* worker);

  // Protocol to use for communicating with workers.
  const std::string protocol_;

  mutex mu_;

  int64 next_worker_id_ TF_GUARDED_BY(mu_) = 0;
  int64 next_dataset_id_ TF_GUARDED_BY(mu_) = 0;
  int64 next_epoch_id_ TF_GUARDED_BY(mu_) = 0;
  int64 next_task_id_ TF_GUARDED_BY(mu_) = 0;

  // Registered workers.
  std::vector<WorkerInfo> workers_ TF_GUARDED_BY(mu_);
  // Registered datasets, keyed by dataset ids.
  absl::flat_hash_map<int64, std::shared_ptr<Dataset>> datasets_by_id_
      TF_GUARDED_BY(mu_);
  // Registered datasets, keyed by dataset fingerprints.
  absl::flat_hash_map<uint64, std::shared_ptr<Dataset>> datasets_by_fingerprint_
      TF_GUARDED_BY(mu_);
  // Information about epochs, keyed by epoch ids.
  absl::flat_hash_map<int64, Epoch> epochs_ TF_GUARDED_BY(mu_);
  // Information about tasks, keyed by task ids.
  absl::flat_hash_map<int64, Task> tasks_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DataServiceMasterImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_MASTER_IMPL_H_
