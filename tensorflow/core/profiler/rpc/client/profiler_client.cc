/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"

#include <limits>

#include "grpcpp/grpcpp.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/rpc/grpc.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

inline Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? Status::OK()
                : Status(static_cast<error::Code>(s.error_code()),
                         s.error_message());
}

template <typename T>
std::unique_ptr<typename T::Stub> CreateStub(const std::string& service_addr) {
  ::grpc::ChannelArguments channel_args;
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
  // Default URI prefix is "dns:///" if not provided.
  auto channel = ::grpc::CreateCustomChannel(
      service_addr, ::grpc::InsecureChannelCredentials(), channel_args);
  if (!channel) {
    LOG(ERROR) << "Unable to create channel" << service_addr;
  }
  return T::NewStub(channel);
}

}  // namespace

Status ProfileGrpc(const std::string& service_addr,
                   const ProfileRequest& request, ProfileResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      CreateStub<grpc::ProfilerService>(service_addr);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Profile(&context, request, response)));
  return Status::OK();
}

Status NewSessionGrpc(const std::string& service_addr,
                      const NewProfileSessionRequest& request,
                      NewProfileSessionResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<grpc::ProfileAnalysis::Stub> stub =
      CreateStub<grpc::ProfileAnalysis>(service_addr);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->NewSession(&context, request, response)));
  return Status::OK();
}

Status MonitorGrpc(const std::string& service_addr,
                   const MonitorRequest& request, MonitorResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<grpc::ProfilerService::Stub> stub =
      CreateStub<grpc::ProfilerService>(service_addr);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Monitor(&context, request, response)));
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
