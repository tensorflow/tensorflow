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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"

namespace tensorflow {
namespace data {

// Client for communicating with the tf.data service dispatcher.
class DataServiceDispatcherClient : public DataServiceClientBase {
 public:
  DataServiceDispatcherClient(const std::string& address,
                              const std::string& protocol)
      : DataServiceClientBase(address, protocol) {}

  absl::Status Initialize() override;

  // Sends a heartbeat to the dispatcher. If the worker wasn't already
  // registered with the dispatcher, this will register the worker. The
  // dispatcher will report which new tasks the worker should run, and which
  // tasks it should delete.
  absl::StatusOr<WorkerHeartbeatResponse> WorkerHeartbeat(
      const WorkerHeartbeatRequest& request);

  // Updates the dispatcher with information about the worker's state.
  absl::Status WorkerUpdate(const std::string& worker_address,
                            std::vector<TaskProgress>& task_progress);

  // Gets a dataset definition for the given dataset id, and stores the
  // definition in `dataset_def`.
  absl::Status GetDatasetDef(const std::string& dataset_id,
                             DatasetDef& dataset_def);

  // Gets the next split for the specified iteration id, repetition, and split
  // provider index.
  absl::Status GetSplit(int64_t iteration_id, int64_t repetition,
                        int64_t split_provider_index, Tensor& split,
                        bool& end_of_splits);

  // Gets the next split for the specified source of a stream of the snapshot in
  // `base_path`. If `end_of_splits` returns true, then there are no more splits
  // to be processed for the specified stream source.
  virtual absl::Status GetSnapshotSplit(
      const std::string& worker_address, const std::string& base_path,
      int64_t stream_index, int64_t source_index, int64_t repetition_index,
      Tensor& split, int64_t& local_split_index, bool& end_of_splits);

  // Initiates the process of materializing `dataset`'s output to `path`.
  absl::Status Snapshot(
      const DatasetDef& dataset, const std::string& path,
      const experimental::DistributedSnapshotMetadata& metadata);

  // Registers a dataset with the tf.data service, and stores the generated
  // dataset id in `dataset_id`.
  absl::Status RegisterDataset(
      const DatasetDef& dataset, const DataServiceMetadata& metadata,
      const std::optional<std::string>& requested_dataset_id,
      std::string& dataset_id);

  // If `job_name` is set, looks up a job matching `job_name`.
  // If `job_name` is absent or no matching job is found, creates a
  // new job. The resulting job id is stored in `job_id`.
  absl::Status GetOrCreateJob(const std::string& dataset_id,
                              const ProcessingModeDef& processing_mode,
                              const std::optional<std::string>& job_name,
                              std::optional<int64_t> num_consumers,
                              bool use_cross_trainer_cache,
                              TargetWorkers target_workers, int64_t& job_id);

  // Looks up an iteration of a job, creating an iteration if one doesn't
  // already exist. The returned `iteration_client_id` can be used to query
  // information about the iteration. The client should call
  // `ReleaseIterationClient` when finished with the iteration, so that
  // resources can be reclaimed.
  absl::Status GetOrCreateIteration(int64_t job_id, int64_t repetition,
                                    int64_t& iteration_client_id);

  // Releases a iteration client id, indicating that the id will no longer be
  // used to read from the iteration.
  absl::Status ReleaseIterationClient(int64_t iteration_client_id);

  // Attempts to remove a task. The task is removed if all consumers try to
  // remove the task in the same round.
  absl::Status MaybeRemoveTask(int64_t task_id, int64_t consumer_index,
                               int64_t round, bool& removed);

  // Heartbeats to the dispatcher, getting back the tasks that should be
  // running, and whether the iteration is finished.
  absl::Status ClientHeartbeat(ClientHeartbeatRequest& req,
                               ClientHeartbeatResponse& resp);

  // Queries the dispatcher for its registered workers. The worker info will be
  // stored in `workers`.
  absl::Status GetWorkers(std::vector<WorkerInfo>& workers);

  // Returns data service metadata for the registered dataset.
  absl::Status GetDataServiceMetadata(const std::string& dataset_id,
                                      DataServiceMetadata& metadata);

  // Returns data service config of the data service cluster.
  absl::Status GetDataServiceConfig(DataServiceConfig& config);

  // Returns information about the decision to disable compression at runtime
  // for a given dataset.
  absl::Status DisableCompressionAtRuntime(
      const std::string& dataset_id, bool disable_compression_at_runtime,
      DisableCompressionAtRuntimeResponse& response);

 protected:
  absl::Status EnsureInitialized() override;

 private:
  mutex mu_;
  // Initialization is guarded by `mu_`, but using the stub does not require
  // holding `mu_`
  std::unique_ptr<DispatcherService::Stub> stub_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_
