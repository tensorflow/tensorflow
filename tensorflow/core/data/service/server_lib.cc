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
#include "tensorflow/core/data/service/grpc_worker_impl.h"

namespace tensorflow {
namespace data {

GrpcDataServer::GrpcDataServer(int port, const std::string& protocol,
                               bool is_master,
                               const std::string& master_address)
    : requested_port_(port),
      protocol_(protocol),
      is_master_(is_master),
      master_address_(master_address) {}

Status GrpcDataServer::Start() {
  ::grpc::ServerBuilder builder;
  std::shared_ptr<::grpc::ServerCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateServerCredentials(protocol_, &credentials));
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port_),
                           credentials, &bound_port_);
  builder.SetMaxReceiveMessageSize(-1);

  if (is_master_) {
    service_ = absl::make_unique<GrpcMasterImpl>(&builder, protocol_);
  } else {
    service_ =
        absl::make_unique<GrpcWorkerImpl>(&builder, master_address_, protocol_);
  }

  server_ = builder.BuildAndStart();
  if (!server_) {
    return errors::Internal("Could not start gRPC server");
  }

  if (!is_master_) {
    static_cast<GrpcWorkerImpl*>(service_.get())
        ->Start(strings::StrCat("localhost:", bound_port_));
  }

  LOG(INFO) << "Started data service " << (is_master_ ? "master" : "worker")
            << " running at " << Target();
  return Status::OK();
}

void GrpcDataServer::Stop() { server_->Shutdown(); }

void GrpcDataServer::Join() { server_->Wait(); }

std::string GrpcDataServer::Target() {
  return strings::StrCat(protocol_, "://localhost:", bound_port_);
}

Status NewMasterServer(int port, const std::string& protocol,
                       std::unique_ptr<GrpcDataServer>* out_server) {
  *out_server = absl::make_unique<GrpcDataServer>(
      port, protocol, /*is_master=*/true, /*master_address=*/"");
  return Status::OK();
}

Status NewWorkerServer(int port, const std::string& protocol,
                       const std::string& master_address,
                       std::unique_ptr<GrpcDataServer>* out_server) {
  *out_server = absl::make_unique<GrpcDataServer>(
      port, protocol, /*is_master=*/false, master_address);
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
