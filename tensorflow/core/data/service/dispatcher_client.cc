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
#include "tensorflow/core/data/service/dispatcher_client.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

absl::Status DataServiceDispatcherClient::Initialize() {
  mutex_lock l(mu_);
  if (stub_) {
    return absl::OkStatus();
  }
  std::shared_ptr<grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol_, &credentials));
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
  auto channel = grpc::CreateCustomChannel(address_, credentials, args);
  stub_ = DispatcherService::NewStub(channel);
  GetVersionRequest req;
  GetVersionResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetVersion(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to get dispatcher version from dispatcher "
                     "running at ",
                     address_),
        s);
  }

  if (resp.version() != kDataServiceVersion) {
    return errors::FailedPrecondition(
        "Version mismatch with tf.data service server. The server is running "
        "version ",
        resp.version(), ", while the client is running version ",
        kDataServiceVersion,
        ". Please ensure that the client and server side are running the "
        "same version of TensorFlow. If you're running an MPM binary, make "
        "sure the server is running an up-to-date MPM.");
  }
  return absl::OkStatus();
}

absl::StatusOr<WorkerHeartbeatResponse>
DataServiceDispatcherClient::WorkerHeartbeat(
    const WorkerHeartbeatRequest& request) {
  WorkerHeartbeatResponse response;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->WorkerHeartbeat(&client_ctx, request, &response);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to perform worker heartbeat", status);
  }
  return response;
}

absl::Status DataServiceDispatcherClient::WorkerUpdate(
    const std::string& worker_address,
    std::vector<TaskProgress>& task_progress) {
  WorkerUpdateRequest req;
  req.set_worker_address(worker_address);
  for (const auto& update : task_progress) {
    *(req.add_updates()) = update;
  }
  WorkerUpdateResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->WorkerUpdate(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to send worker update", status);
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetDatasetDef(
    const std::string& dataset_id, DatasetDef& dataset_def) {
  GetDatasetDefRequest req;
  req.set_dataset_id(dataset_id);
  GetDatasetDefResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetDatasetDef(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get dataset def", status);
  }
  dataset_def = resp.dataset_def();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetSplit(int64_t iteration_id,
                                                   int64_t repetition,
                                                   int64_t split_provider_index,
                                                   Tensor& split,
                                                   bool& end_of_splits) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetSplitRequest req;
  req.set_iteration_id(iteration_id);
  req.set_repetition(repetition);
  req.set_split_provider_index(split_provider_index);
  GetSplitResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetSplit(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get split", status);
  }
  end_of_splits = resp.end_of_splits();
  if (!end_of_splits) {
    if (!split.FromProto(resp.split())) {
      return errors::Internal("Failed to parse split tensor proto");
    }
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::Snapshot(
    const DatasetDef& dataset, const std::string& path,
    const experimental::DistributedSnapshotMetadata& metadata) {
  TF_RETURN_IF_ERROR(EnsureInitialized());

  SnapshotRequest req;
  *req.mutable_dataset() = dataset;
  req.set_path(path);
  *req.mutable_metadata() = metadata;

  SnapshotResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->Snapshot(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to snapshot", status);
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetSnapshotSplit(
    const std::string& worker_address, const std::string& base_path,
    int64_t stream_index, int64_t source_index, int64_t repetition_index,
    Tensor& split, int64_t& local_split_index, bool& end_of_splits) {
  GetSnapshotSplitRequest req;
  req.set_worker_address(worker_address);
  req.set_base_path(base_path);
  req.set_stream_index(stream_index);
  req.set_source_index(source_index);
  req.set_repetition_index(repetition_index);

  GetSnapshotSplitResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetSnapshotSplit(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to get snapshot split", status);
  }
  local_split_index = resp.local_split_index();
  end_of_splits = resp.end_of_splits();
  if (end_of_splits) {
    return absl::OkStatus();
  }
  if (!split.FromProto(resp.split())) {
    return errors::Internal("Failed to parse split tensor proto: ",
                            resp.split().DebugString());
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::RegisterDataset(
    const DatasetDef& dataset, const DataServiceMetadata& metadata,
    const std::optional<std::string>& requested_dataset_id,
    std::string& dataset_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrRegisterDatasetRequest req;
  *req.mutable_dataset() = dataset;
  *req.mutable_metadata() = metadata;
  if (requested_dataset_id.has_value()) {
    req.set_dataset_id(*requested_dataset_id);
  }

  GetOrRegisterDatasetResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrRegisterDataset(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to register dataset", status);
  }
  dataset_id = resp.dataset_id();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetOrCreateJob(
    const std::string& dataset_id, const ProcessingModeDef& processing_mode,
    const std::optional<std::string>& job_name,
    std::optional<int64_t> num_consumers, bool use_cross_trainer_cache,
    TargetWorkers target_workers, int64_t& job_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrCreateJobRequest req;
  req.set_dataset_id(dataset_id);
  *req.mutable_processing_mode_def() = processing_mode;
  if (job_name.has_value()) {
    req.set_job_name(job_name.value());
  }
  if (num_consumers.has_value()) {
    req.set_num_consumers(num_consumers.value());
  }
  req.set_target_workers(target_workers);
  req.set_use_cross_trainer_cache(use_cross_trainer_cache);
  GetOrCreateJobResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrCreateJob(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to get or create job for dataset with id ",
                     dataset_id),
        status);
  }
  job_id = resp.job_id();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetOrCreateIteration(
    int64_t job_id, int64_t repetition, int64_t& iteration_client_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetOrCreateIterationRequest req;
  req.set_job_id(job_id);
  req.set_repetition(repetition);
  GetOrCreateIterationResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->GetOrCreateIteration(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to get or create iteration for job with id ",
                     job_id),
        status);
  }
  iteration_client_id = resp.iteration_client_id();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::ReleaseIterationClient(
    int64_t iteration_client_id) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  ReleaseIterationClientRequest req;
  req.set_iteration_client_id(iteration_client_id);
  ReleaseIterationClientResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->ReleaseIterationClient(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError(
        absl::StrCat("Failed to release iteration client with id ",
                     iteration_client_id),
        status);
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::MaybeRemoveTask(
    int64_t task_id, int64_t consumer_index, int64_t round, bool& removed) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  MaybeRemoveTaskRequest req;
  req.set_task_id(task_id);
  req.set_consumer_index(consumer_index);
  req.set_round(round);
  MaybeRemoveTaskResponse resp;
  grpc::ClientContext client_ctx;
  grpc::Status status = stub_->MaybeRemoveTask(&client_ctx, req, &resp);
  if (!status.ok()) {
    return grpc_util::WrapError("Failed to call MaybeRemoveTask", status);
  }
  removed = resp.removed();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::ClientHeartbeat(
    ClientHeartbeatRequest& req, ClientHeartbeatResponse& resp) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  grpc::ClientContext ctx;
  grpc::Status s = stub_->ClientHeartbeat(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetWorkers(
    std::vector<WorkerInfo>& workers) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetWorkersRequest req;
  GetWorkersResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetWorkers(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get workers", s);
  }
  workers.clear();
  for (auto& worker : resp.workers()) {
    workers.push_back(worker);
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetDataServiceMetadata(
    const std::string& dataset_id, DataServiceMetadata& metadata) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetDataServiceMetadataRequest req;
  req.set_dataset_id(dataset_id);
  GetDataServiceMetadataResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetDataServiceMetadata(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get data service metadata", s);
  }
  metadata = resp.metadata();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::GetDataServiceConfig(
    DataServiceConfig& config) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  GetDataServiceConfigRequest request;
  GetDataServiceConfigResponse response;
  grpc::ClientContext ctx;
  grpc::Status s = stub_->GetDataServiceConfig(&ctx, request, &response);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get data service config", s);
  }
  config = response.config();
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::DisableCompressionAtRuntime(
    const std::string& dataset_id, bool disable_compression_at_runtime,
    DisableCompressionAtRuntimeResponse& response) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  grpc::ClientContext ctx;
  DisableCompressionAtRuntimeRequest request;
  request.set_dataset_id(dataset_id);
  request.set_disable_compression_at_runtime(disable_compression_at_runtime);
  grpc::Status s = stub_->DisableCompressionAtRuntime(&ctx, request, &response);
  if (!s.ok()) {
    return grpc_util::WrapError(
        "Failed to get runtime compression disabling decision", s);
  }
  return absl::OkStatus();
}

absl::Status DataServiceDispatcherClient::EnsureInitialized() {
  return grpc_util::Retry([this] { return Initialize(); },
                          "Initialize dispatcher client",
                          /*deadline_micros=*/kint64max);
}

}  // namespace data
}  // namespace tensorflow
