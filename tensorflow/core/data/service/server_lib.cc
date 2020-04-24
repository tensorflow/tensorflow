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

#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_master_impl.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/grpc_worker_impl.h"

namespace tensorflow {
namespace data {

GrpcDataServerBase::GrpcDataServerBase(int port, const std::string& protocol)
    : requested_port_(port), protocol_(protocol) {}

Status GrpcDataServerBase::Start() {
  ::grpc::ServerBuilder builder;
  std::shared_ptr<::grpc::ServerCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateServerCredentials(protocol_, &credentials));
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port_),
                           credentials, &bound_port_);
  builder.SetMaxReceiveMessageSize(-1);

  AddServiceToBuilder(&builder);
  server_ = builder.BuildAndStart();
  if (!server_) {
    return errors::Internal("Could not start gRPC server");
  }

  TF_RETURN_IF_ERROR(StartServiceInternal());

  LOG(INFO) << "Started data service running at " << Target();
  return Status::OK();
}

void GrpcDataServerBase::Stop() { server_->Shutdown(); }

void GrpcDataServerBase::Join() { server_->Wait(); }

std::string GrpcDataServerBase::Target() {
  return strings::StrCat(protocol_, "://localhost:", bound_port_);
}

MasterGrpcDataServer::MasterGrpcDataServer(int port,
                                           const std::string& protocol)
    : GrpcDataServerBase(port, protocol) {}

MasterGrpcDataServer::~MasterGrpcDataServer() { delete service_; }

void MasterGrpcDataServer::AddServiceToBuilder(grpc::ServerBuilder* builder) {
  auto service = absl::make_unique<GrpcMasterImpl>(builder, protocol_);
  service_ = service.release();
}

Status MasterGrpcDataServer::NumTasks(int* num_tasks) {
  GetTasksRequest req;
  GetTasksResponse resp;
  grpc::ServerContext ctx;
  grpc::Status s = service_->GetTasks(&ctx, &req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get num tasks", s);
  }
  *num_tasks = resp.task_info_size();
  return Status::OK();
}

WorkerGrpcDataServer::WorkerGrpcDataServer(int port,
                                           const std::string& protocol,
                                           const std::string& master_address)
    : GrpcDataServerBase(port, protocol), master_address_(master_address) {}

WorkerGrpcDataServer::~WorkerGrpcDataServer() { delete service_; }

void WorkerGrpcDataServer::AddServiceToBuilder(grpc::ServerBuilder* builder) {
  auto service =
      absl::make_unique<GrpcWorkerImpl>(builder, master_address_, protocol_);
  service_ = service.release();
}

Status WorkerGrpcDataServer::StartServiceInternal() {
  service_->Start(strings::StrCat("localhost:", bound_port()));
  return Status::OK();
}

Status NewMasterServer(int port, const std::string& protocol,
                       std::unique_ptr<MasterGrpcDataServer>* out_server) {
  *out_server = absl::make_unique<MasterGrpcDataServer>(port, protocol);
  return Status::OK();
}

Status NewWorkerServer(int port, const std::string& protocol,
                       const std::string& master_address,
                       std::unique_ptr<WorkerGrpcDataServer>* out_server) {
  *out_server =
      absl::make_unique<WorkerGrpcDataServer>(port, protocol, master_address);
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
