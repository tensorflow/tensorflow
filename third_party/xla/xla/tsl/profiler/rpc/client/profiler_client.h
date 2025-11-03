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
// GRPC client to perform on-demand profiling

#ifndef XLA_TSL_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
#define XLA_TSL_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/completion_queue.h"
#include "xla/tsl/platform/types.h"
#include "tsl/profiler/protobuf/profiler_analysis.grpc.pb.h"  // IWYU pragma: keep
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace tsl {
namespace profiler {

// Note that tensorflow/tools/def_file_filter/symbols_pybind.txt is incompatible
// with absl::string_view.
absl::Status ProfileGrpc(const std::string& service_address,
                         const tensorflow::ProfileRequest& request,
                         tensorflow::ProfileResponse* response);

absl::Status NewSessionGrpc(const std::string& service_address,
                            const tensorflow::NewProfileSessionRequest& request,
                            tensorflow::NewProfileSessionResponse* response);

absl::Status MonitorGrpc(const std::string& service_address,
                         const tensorflow::MonitorRequest& request,
                         tensorflow::MonitorResponse* response);

class RemoteProfilerSession {
 public:
  // Creates an instance and starts a remote profiling session immediately.
  // This is a non-blocking call and does not wait for a response.
  // Response must outlive the instantiation.
  static std::unique_ptr<RemoteProfilerSession> Create(
      const std::string& service_address, absl::Time deadline,
      const tensorflow::ProfileRequest& profile_request,
      ::grpc::CompletionQueue* cq, int64_t client_id);

  // Not copyable or movable.
  RemoteProfilerSession(const RemoteProfilerSession&) = delete;
  RemoteProfilerSession& operator=(const RemoteProfilerSession&) = delete;

  ~RemoteProfilerSession();

  absl::string_view GetServiceAddress() const { return service_address_; }

  // Processes a completion event that has been received from the gRPC
  // completion queue. Updates internal state with the final status and returns
  // the response. Subsequent calls after the first will yield nullptr and an
  // error status.
  std::unique_ptr<tensorflow::ProfileResponse> HandleCompletion(
      absl::Status& out_status, void* got_tag, bool ok);

 private:
  // Constructs a new RemoteProfilerSession instance.
  // client_id is a unique identifier used as a tag for gRPC completion queue
  // operations.
  explicit RemoteProfilerSession(
      const std::string& service_addr, absl::Time deadline,
      const tensorflow::ProfileRequest& profile_request, int64_t client_id);

  // Starts a remote profiling session. This is a non-blocking call.
  // Will be called exactly once during instantiation.
  // RPC will write to response.profile_response eagerly. However, since
  // response.status requires a conversion from grpc::Status, it can only be
  //  evaluated lazily at WaitForCompletion() time.
  void ProfileAsync(::grpc::CompletionQueue* cq);

  absl::Status status_on_completion_;
  std::unique_ptr<tensorflow::ProfileResponse> response_;
  // Client address and connection attributes.
  std::string service_address_;
  std::unique_ptr<tensorflow::grpc::ProfilerService::Stub> stub_;
  absl::Time deadline_;
  ::grpc::ClientContext grpc_context_;
  std::unique_ptr<
      ::grpc::ClientAsyncResponseReader<tensorflow::ProfileResponse>>
      rpc_;
  ::grpc::Status grpc_status_ = ::grpc::Status::OK;

  int64_t client_id_;

  tensorflow::ProfileRequest profile_request_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
