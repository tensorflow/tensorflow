/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

// Client for communicating with the tf.data service dispatcher.
class DataServiceDispatcherClient : public DataServiceClientBase {
 public:
  DataServiceDispatcherClient(const std::string& address,
                              const std::string& protocol)
      : DataServiceClientBase(address, protocol) {}

  // Sends a heartbeat to the dispatcher. If the worker wasn't already
  // registered with the dispatcher, this will register the worker. The
  // dispatcher will report which new tasks the worker should run, and which
  // tasks it should delete. This is stored into `new_tasks` and
  // `tasks_to_delete`.
  Status WorkerHeartbeat(const std::string& worker_address,
                         const std::string& transfer_address,
                         const std::vector<int64>& current_tasks,
                         std::vector<TaskDef>& new_tasks,
                         std::vector<int64>& tasks_to_delete);

  // Updates the dispatcher with information about the worker's state.
  Status WorkerUpdate(const std::string& worker_address,
                      std::vector<TaskProgress>& task_progress);

  // Gets a dataset definition for the given dataset id, and stores the
  // definition in `dataset_def`.
  Status GetDatasetDef(int64 dataset_id, DatasetDef& dataset_def);

  // Gets the next split for the specified job id, repetition, and split
  // provider index.
  Status GetSplit(int64 job_id, int64 repetition, int64 split_provider_index,
                  Tensor& split, bool& end_of_splits);

  // Registers a dataset with the tf.data service, and stores the generated
  // dataset id in `dataset_id`.
  Status RegisterDataset(const DatasetDef& dataset,
                         const absl::optional<std::string>& element_spec,
                         int64& dataset_id);

  // If `job_key` is set, looks up a job matching `job_key`. If `job_key` is
  // absent or no matching job is found, creates a new job. The resulting job
  // id is stored in `job_client_id`.
  Status GetOrCreateJob(int64 dataset_id, ProcessingMode processing_mode,
                        const absl::optional<JobKey>& job_key,
                        absl::optional<int64> num_consumers,
                        int64& job_client_id);

  // Releases a job client id, indicating that the id will no longer be used to
  // read from the job.
  Status ReleaseJobClient(int64 job_client_id);

  // Attempts to remove a task. The task is removed if all consumers try to
  // remove the task in the same round.
  Status MaybeRemoveTask(int64 task_id, int64 consumer_index, int64 round,
                         bool& removed);

  // Heartbeats to the dispatcher, getting back the tasks that should be
  // running, and whether the job is finished.
  Status ClientHeartbeat(ClientHeartbeatRequest& req,
                         ClientHeartbeatResponse& resp);

  // Queries the dispatcher for its registered workers. The worker info will be
  // stored in `workers`.
  Status GetWorkers(std::vector<WorkerInfo>& workers);

  // Returns element spec for the registered dataset.
  Status GetElementSpec(int64 dataset_id, std::string& element_spec);

 protected:
  Status EnsureInitialized() override;

 private:
  mutex mu_;
  // Initialization is guarded by `mu_`, but using the stub does not require
  // holding `mu_`
  std::unique_ptr<DispatcherService::Stub> stub_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_
