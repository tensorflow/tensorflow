/* Copyright 2020 Google LLC

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

#include "tensorflow/compiler/xla/pjrt/distributed/client.h"

#include <chrono>  // NOLINT

#include "tensorflow/compiler/xla/pjrt/distributed/protocol.h"
#include "tensorflow/compiler/xla/pjrt/distributed/util.h"

namespace xla {

DistributedRuntimeClient::DistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel)
    : stub_(grpc::DistributedRuntimeService::NewStub(std::move(channel))) {}
DistributedRuntimeClient::~DistributedRuntimeClient() = default;

xla::Status DistributedRuntimeClient::Connect(
    const LocalTopologyProto& local_topology,
    GlobalTopologyProto* global_topology) {
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + rpc_timeout_));
  ConnectRequest request;
  request.set_protocol_version(kDistributedRuntimeProtocolVersion);
  *request.mutable_local_topology() = local_topology;
  VLOG(10) << "Connect: " << request.DebugString();
  ConnectResponse response;
  ::grpc::Status status = stub_->Connect(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  VLOG(10) << "Connect() response: " << response.DebugString();
  response.mutable_global_topology()->Swap(global_topology);
  return xla::Status::OK();
}

xla::StatusOr<std::string> DistributedRuntimeClient::BlockingKeyValueGet(
    std::string key, absl::Duration timeout) {
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  KeyValueGetRequest request;
  request.set_key(std::move(key));
  timeout = std::min(timeout, absl::Minutes(10));  // Avoid overflow
  request.set_timeout_milliseconds(timeout / absl::Milliseconds(1));
  VLOG(10) << "BlockingKeyValueGet: " << request.DebugString();
  KeyValueGetResponse response;
  ::grpc::Status status = stub_->KeyValueGet(&ctx, request, &response);
  if (!status.ok()) {
    return FromGrpcStatus(status);
  }
  return response.value();
}

xla::Status DistributedRuntimeClient::KeyValueSet(std::string key,
                                                  std::string value) {
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(absl::ToChronoTime(absl::Now() + rpc_timeout_));
  KeyValueSetRequest request;
  request.set_key(std::move(key));
  request.set_value(std::move(value));
  VLOG(10) << "KeyValueSet: " << request.DebugString();
  KeyValueSetResponse response;
  ::grpc::Status status = stub_->KeyValueSet(&ctx, request, &response);
  return FromGrpcStatus(status);
}

}  // namespace xla
