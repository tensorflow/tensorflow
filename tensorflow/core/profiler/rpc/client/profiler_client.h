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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/profiler_analysis.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"

namespace tensorflow {
namespace profiler {

// Note that tensorflow/tools/def_file_filter/symbols_pybind.txt is incompatible
// with absl::string_view.
Status ProfileGrpc(const std::string& service_address,
                   const ProfileRequest& request, ProfileResponse* response);

Status NewSessionGrpc(const std::string& service_address,
                      const NewProfileSessionRequest& request,
                      NewProfileSessionResponse* response);

Status MonitorGrpc(const std::string& service_address,
                   const MonitorRequest& request, MonitorResponse* response);

class RemoteProfilerSession {
 public:
  // Creates an instance and starts a remote profiling session immediately.
  // This is a non-blocking call and does not wait for a response.
  // Response must outlive the instantiation.
  static std::unique_ptr<RemoteProfilerSession> Create(
      const std::string& service_address, absl::Time deadline,
      const ProfileRequest& profile_request);

  // Not copyable or movable.
  RemoteProfilerSession(const RemoteProfilerSession&) = delete;
  RemoteProfilerSession& operator=(const RemoteProfilerSession&) = delete;

  ~RemoteProfilerSession();

  absl::string_view GetServiceAddress() const { return service_address_; }

  // Blocks until a response has been received or until deadline expiry,
  // whichever is first. Subsequent calls after the first will yield nullptr and
  // an error status.
  std::unique_ptr<ProfileResponse> WaitForCompletion(Status& out_status);

 private:
  explicit RemoteProfilerSession(const std::string& service_addr,
                                 absl::Time deadline,
                                 const ProfileRequest& profile_request);

  // Starts a remote profiling session. This is a non-blocking call.
  // Will be called exactly once during instantiation.
  // RPC will write to response.profile_response eagerly. However, since
  // response.status requires a conversion from grpc::Status, it can only be
  //  evaluated lazily at WaitForCompletion() time.
  void ProfileAsync();

  Status status_on_completion_;
  std::unique_ptr<ProfileResponse> response_;
  // Client address and connection attributes.
  std::string service_address_;
  std::unique_ptr<grpc::ProfilerService::Stub> stub_;
  absl::Time deadline_;
  ::grpc::ClientContext grpc_context_;
  std::unique_ptr<::grpc::ClientAsyncResponseReader<ProfileResponse>> rpc_;
  ::grpc::Status grpc_status_ = ::grpc::Status::OK;

  // Asynchronous completion queue states.
  ::grpc::CompletionQueue cq_;

  ProfileRequest profile_request_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
