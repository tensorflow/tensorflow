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

#include "tensorflow/core/data/service/server_lib.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/grpc_dispatcher_impl.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/grpc_worker_impl.h"
#include "tensorflow/core/data/service/worker_client.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {
namespace data {

namespace {

// See `WorkerConfig` docs.
constexpr char kPortPlaceholder[] = "%port%";
constexpr char kDataTransferPortPlaceholder[] = "%dts_port%";

}  // namespace

GrpcDataServerBase::GrpcDataServerBase(
    int port, const std::string& protocol, const std::string& server_type,
    std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> options)
    : requested_port_(port),
      protocol_(protocol),
      server_type_(server_type),
      bound_port_(port),
      server_options_(std::move(options)) {}

Status GrpcDataServerBase::Start() {
  if (stopped_) {
    return errors::FailedPrecondition(
        "Server cannot be started after it has been stopped.");
  }
  if (started_) {
    return absl::OkStatus();
  }
  ::grpc::ServerBuilder builder;
  for (std::unique_ptr<::grpc::ServerBuilderOption>& option : server_options_) {
    builder.SetOption(std::move(option));
  }
  server_options_.clear();

  std::shared_ptr<::grpc::ServerCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateServerCredentials(protocol_, &credentials));
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port_),
                           credentials, &bound_port_);
  builder.SetMaxReceiveMessageSize(-1);

  AddDataServiceToBuilder(builder);
  AddProfilerServiceToBuilder(builder);
  server_ = builder.BuildAndStart();
  if (!server_) {
    return errors::Internal("Could not start gRPC server");
  }

  TF_RETURN_IF_ERROR(StartServiceInternal());

  started_ = true;
  LOG(INFO) << "Started tf.data " << server_type_
            << " running at 0.0.0.0:" << BoundPort();
  return absl::OkStatus();
}

void GrpcDataServerBase::Stop() {
  if (stopped_) {
    return;
  }
  if (server_) {
    StopServiceInternal();
    server_->Shutdown();
    LOG(INFO) << "Shut down " << server_type_ << " server running at port "
              << BoundPort();
  }
  stopped_ = true;
}

void GrpcDataServerBase::Join() { server_->Wait(); }

int GrpcDataServerBase::BoundPort() { return bound_port(); }

void GrpcDataServerBase::AddProfilerServiceToBuilder(
    ::grpc::ServerBuilder& builder) {
  profiler_service_ = profiler::CreateProfilerService();
  builder.RegisterService(profiler_service_.get());
}

DispatchGrpcDataServer::DispatchGrpcDataServer(
    const experimental::DispatcherConfig& config,
    std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> options)
    : GrpcDataServerBase(config.port(), config.protocol(), "DispatchServer",
                         std::move(options)),
      config_(config) {}

DispatchGrpcDataServer::~DispatchGrpcDataServer() { delete service_; }

void DispatchGrpcDataServer::AddDataServiceToBuilder(
    ::grpc::ServerBuilder& builder) {
  service_ = std::make_unique<GrpcDispatcherImpl>(config_, builder).release();
}

Status DispatchGrpcDataServer::StartServiceInternal() {
  return service_->Start();
}

void DispatchGrpcDataServer::StopServiceInternal() { service_->Stop(); }

Status DispatchGrpcDataServer::NumWorkers(int* num_workers) {
  GetWorkersRequest req;
  GetWorkersResponse resp;
  ::grpc::ServerContext ctx;
  ::grpc::Status s = service_->GetWorkers(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get workers", s);
  }
  *num_workers = resp.workers_size();
  return absl::OkStatus();
}

Status DispatchGrpcDataServer::SnapshotStreams(
    const std::string& path, std::vector<SnapshotStreamInfoWrapper>* streams) {
  GetSnapshotStreamsRequest req;
  req.set_path(path);
  GetSnapshotStreamsResponse resp;
  ::grpc::ServerContext ctx;
  ::grpc::Status s = service_->GetSnapshotStreams(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get snapshot streams", s);
  }
  for (const auto& stream : resp.streams()) {
    streams->push_back(SnapshotStreamInfoWrapper(stream));
  }
  return absl::OkStatus();
}

size_t DispatchGrpcDataServer::NumActiveIterations() {
  return service_->NumActiveIterations();
}

ServerStateExport DispatchGrpcDataServer::ExportState() const {
  ServerStateExport server_state_export;
  *server_state_export.mutable_dispatcher_state_export() =
      service_->ExportState();
  return server_state_export;
}

WorkerGrpcDataServer::WorkerGrpcDataServer(
    const experimental::WorkerConfig& config,
    std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> options)
    : GrpcDataServerBase(config.port(), config.protocol(), "WorkerServer",
                         std::move(options)),
      config_(config) {}

WorkerGrpcDataServer::~WorkerGrpcDataServer() { delete service_; }

void WorkerGrpcDataServer::AddDataServiceToBuilder(
    ::grpc::ServerBuilder& builder) {
  service_ = std::make_unique<GrpcWorkerImpl>(config_, builder).release();
}

void WorkerGrpcDataServer::MaybeStartAlternativeDataTransferServer(
    std::vector<DataTransferServerInfo>& transfer_servers) {
  if (config_.data_transfer_protocol().empty() ||
      config_.data_transfer_protocol() == kGrpcTransferProtocol) {
    return;
  }
  Status s = DataTransferServer::Build(config_.data_transfer_protocol(),
                                       service_->get_element_getter(),
                                       &transfer_server_);
  if (!s.ok()) {
    LOG(ERROR) << "failed to build " << config_.data_transfer_protocol()
               << " server for worker " << config_.worker_address() << ": "
               << s;
    return;
  }
  s = transfer_server_->Start(config_);
  if (!s.ok()) {
    LOG(ERROR) << "failed to start " << config_.data_transfer_protocol()
               << " server for worker " << config_.worker_address() << ": "
               << s;
    return;
  }
  LOG(INFO) << "Data transfer server started at 0.0.0.0:"
            << transfer_server_->Port() << " for protocol "
            << config_.data_transfer_protocol() << " for worker "
            << config_.worker_address();
  DataTransferServerInfo alternative_transfer_server;
  alternative_transfer_server.set_protocol(config_.data_transfer_protocol());
  alternative_transfer_server.set_address(str_util::StringReplace(
      config_.data_transfer_address(), kDataTransferPortPlaceholder,
      absl::StrCat(transfer_server_->Port()),
      /*replace_all=*/false));
  absl::StatusOr<std::string> compatibility_info =
      transfer_server_->GetCompatibilityInfo();
  if (!compatibility_info.ok()) {
    LOG(ERROR)
        << "failed to populate compatibility information for worker server "
        << config_.worker_address() << " for protocol "
        << config_.data_transfer_protocol() << ": "
        << compatibility_info.status();
    return;
  }
  alternative_transfer_server.set_compatibility_info(*compatibility_info);
  transfer_servers.push_back(alternative_transfer_server);
}

Status WorkerGrpcDataServer::StartServiceInternal() {
  std::string base_address = config_.worker_address();
  if (base_address.empty()) {
    base_address = absl::StrCat("localhost:", kPortPlaceholder);
  }
  std::string worker_address = str_util::StringReplace(
      base_address, kPortPlaceholder, absl::StrCat(bound_port()),
      /*replace_all=*/false);
  DataTransferServerInfo grpc_transfer_server;
  grpc_transfer_server.set_protocol(kGrpcTransferProtocol);
  grpc_transfer_server.set_address(worker_address);
  std::vector<DataTransferServerInfo> transfer_servers = {grpc_transfer_server};
  MaybeStartAlternativeDataTransferServer(transfer_servers);
  TF_RETURN_IF_ERROR(service_->Start(worker_address, transfer_servers));
  return absl::OkStatus();
}

void WorkerGrpcDataServer::StopServiceInternal() { service_->Stop(); }

Status WorkerGrpcDataServer::NumTasks(int* num_tasks) {
  GetWorkerTasksRequest req;
  GetWorkerTasksResponse resp;
  ::grpc::ServerContext ctx;
  ::grpc::Status s = service_->GetWorkerTasks(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  *num_tasks = resp.tasks_size();
  return absl::OkStatus();
}

Status WorkerGrpcDataServer::SnapshotTaskProgresses(
    std::vector<SnapshotTaskProgressWrapper>* snapshot_task_progresses) {
  GetSnapshotTaskProgressesRequest req;
  GetSnapshotTaskProgressesResponse resp;
  ::grpc::ServerContext ctx;
  ::grpc::Status s = service_->GetSnapshotTaskProgresses(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  for (const auto& progress : resp.snapshot_task_progresses()) {
    snapshot_task_progresses->push_back(SnapshotTaskProgressWrapper(progress));
  }
  return absl::OkStatus();
}

ServerStateExport WorkerGrpcDataServer::ExportState() const {
  ServerStateExport server_state_export;
  *server_state_export.mutable_worker_state_export() = service_->ExportState();
  return server_state_export;
}

Status NewDispatchServer(const experimental::DispatcherConfig& config,
                         std::unique_ptr<DispatchGrpcDataServer>& out_server) {
  out_server = std::make_unique<DispatchGrpcDataServer>(config);
  return absl::OkStatus();
}

Status NewWorkerServer(const experimental::WorkerConfig& config,
                       std::unique_ptr<WorkerGrpcDataServer>& out_server) {
  out_server = std::make_unique<WorkerGrpcDataServer>(config);
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
