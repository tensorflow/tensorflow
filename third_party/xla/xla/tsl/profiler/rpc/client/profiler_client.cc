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
#include "xla/tsl/profiler/rpc/client/profiler_client.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/grpcpp.h"  // IWYU pragma: keep
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tsl/profiler/protobuf/profiler_analysis.grpc.pb.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::ContinuousProfilingResponse;
using tensorflow::GetSnapshotRequest;
using tensorflow::MonitorRequest;
using tensorflow::MonitorResponse;
using tensorflow::NewProfileSessionRequest;
using tensorflow::NewProfileSessionResponse;
using tensorflow::ProfileRequest;
using tensorflow::ProfileResponse;

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                               s.error_message());
}

template <typename T>
std::unique_ptr<typename T::Stub> CreateStub(
    const std::string& service_address) {
  ::grpc::ChannelArguments channel_args;
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32_t>::max());
  // Default URI prefix is "dns:///" if not provided.
  auto channel = ::grpc::CreateCustomChannel(
      service_address, ::grpc::InsecureChannelCredentials(), channel_args);
  if (!channel) {
    LOG(ERROR) << "Unable to create channel" << service_address;
    return nullptr;
  }
  return T::NewStub(channel);
}

}  // namespace

absl::Status ProfileGrpc(const std::string& service_address,
                         const ProfileRequest& request,
                         ProfileResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<tensorflow::grpc::ProfilerService::Stub> stub =
      CreateStub<tensorflow::grpc::ProfilerService>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Profile(&context, request, response)));
  return absl::OkStatus();
}

absl::Status ContinuousProfilingGrpc(const std::string& service_address,
                                     const ProfileRequest& request,
                                     ContinuousProfilingResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<tensorflow::grpc::ProfilerService::Stub> stub =
      CreateStub<tensorflow::grpc::ProfilerService>(service_address);
  TF_RETURN_IF_ERROR(FromGrpcStatus(
      stub->StartContinuousProfiling(&context, request, response)));
  return absl::OkStatus();
}

absl::Status GetSnapshotGrpc(const std::string& service_address,
                             const GetSnapshotRequest& request,
                             ProfileResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<tensorflow::grpc::ProfilerService::Stub> stub =
      CreateStub<tensorflow::grpc::ProfilerService>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->GetSnapshot(&context, request, response)));
  return absl::OkStatus();
}

absl::Status NewSessionGrpc(const std::string& service_address,
                            const NewProfileSessionRequest& request,
                            NewProfileSessionResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<tensorflow::grpc::ProfileAnalysis::Stub> stub =
      CreateStub<tensorflow::grpc::ProfileAnalysis>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->NewSession(&context, request, response)));
  return absl::OkStatus();
}

absl::Status MonitorGrpc(const std::string& service_address,
                         const MonitorRequest& request,
                         MonitorResponse* response) {
  ::grpc::ClientContext context;
  std::unique_ptr<tensorflow::grpc::ProfilerService::Stub> stub =
      CreateStub<tensorflow::grpc::ProfilerService>(service_address);
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub->Monitor(&context, request, response)));
  return absl::OkStatus();
}

/*static*/ std::unique_ptr<RemoteProfilerSession> RemoteProfilerSession::Create(
    const std::string& service_address, absl::Time deadline,
    const ProfileRequest& profile_request) {
  auto instance = absl::WrapUnique(
      new RemoteProfilerSession(service_address, deadline, profile_request));
  instance->ProfileAsync();
  return instance;
}

RemoteProfilerSession::RemoteProfilerSession(
    const std::string& service_address, absl::Time deadline,
    const ProfileRequest& profile_request)
    : response_(std::make_unique<ProfileResponse>()),
      service_address_(service_address),
      stub_(CreateStub<tensorflow::grpc::ProfilerService>(service_address_)),
      deadline_(deadline),
      profile_request_(profile_request) {
  response_->set_empty_trace(true);
}

RemoteProfilerSession::~RemoteProfilerSession() {
  absl::Status dummy;
  WaitForCompletion(dummy);
  grpc_context_.TryCancel();
}

void RemoteProfilerSession::ProfileAsync() {
  LOG(INFO) << "Asynchronous gRPC Profile() to " << service_address_;
  grpc_context_.set_deadline(absl::ToChronoTime(deadline_));
  VLOG(1) << "Deadline set to " << deadline_;
  rpc_ = stub_->AsyncProfile(&grpc_context_, profile_request_, &cq_);
  // Connection failure will create lame channel whereby grpc_status_ will be an
  // error.
  rpc_->Finish(response_.get(), &grpc_status_,
               static_cast<void*>(&status_on_completion_));
  VLOG(2) << "Asynchronous gRPC Profile() issued." << absl::Now();
}

std::unique_ptr<ProfileResponse> RemoteProfilerSession::WaitForCompletion(
    absl::Status& out_status) {
  if (!response_) {
    out_status = absl::FailedPreconditionError(
        "WaitForCompletion must only be called once.");
    return nullptr;
  }
  LOG(INFO) << "Waiting for completion.";

  void* got_tag = nullptr;
  bool ok = false;
  // Next blocks until there is a response in the completion queue. Expect the
  // completion queue to have exactly a single response because deadline is set
  // and completion queue is only drained once at destruction time.
  bool success = cq_.Next(&got_tag, &ok);
  if (!success || !ok || got_tag == nullptr) {
    out_status =
        absl::InternalError("Missing or invalid event from completion queue.");
    return nullptr;
  }

  VLOG(1) << "Writing out status.";
  // For the event read from the completion queue, expect that got_tag points to
  // the memory location of status_on_completion.
  DCHECK_EQ(got_tag, &status_on_completion_);
  // tagged status points to pre-allocated memory which is okay to overwrite.
  status_on_completion_.Update(FromGrpcStatus(grpc_status_));
  if (status_on_completion_.code() == error::DEADLINE_EXCEEDED) {
    LOG(WARNING) << status_on_completion_;
  } else if (!status_on_completion_.ok()) {
    LOG(ERROR) << status_on_completion_;
  }

  out_status = status_on_completion_;
  return std::move(response_);
}

}  // namespace profiler
}  // namespace tsl
