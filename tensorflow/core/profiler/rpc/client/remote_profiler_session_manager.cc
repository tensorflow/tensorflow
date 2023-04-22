/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.h"

#include <cstddef>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

/*static*/ std::unique_ptr<RemoteProfilerSessionManager>
RemoteProfilerSessionManager::Create(
    const RemoteProfilerSessionManagerOptions& options,
    const ProfileRequest& request, tensorflow::Status& out_status,
    AddressResolver resolver) {
  VLOG(1) << "Creating a RemoteProfilerSessionManager.";
  auto session_manager = absl::WrapUnique(
      new RemoteProfilerSessionManager(options, request, resolver));
  out_status = session_manager->Init();
  if (!out_status.ok()) {
    return nullptr;
  }
  return session_manager;
}

RemoteProfilerSessionManager::RemoteProfilerSessionManager(
    RemoteProfilerSessionManagerOptions options, ProfileRequest request,
    AddressResolver resolver)
    : options_(options), request_(request) {
  if (resolver) {
    resolver_ = resolver;
  } else {
    resolver_ = [](absl::string_view addr) { return std::string(addr); };
  }
}

RemoteProfilerSessionManager::~RemoteProfilerSessionManager() {
  VLOG(2) << "Destroying RemoteProfilerSessionManager.";
}

Status RemoteProfilerSessionManager::Init() {
  mutex_lock lock(mutex_);
  VLOG(1) << "SessionManager initializing.";

  const absl::Time session_created_ts =
      absl::FromUnixNanos(options_.session_creation_timestamp_ns());
  const absl::Time deadline =
      session_created_ts +
      absl::Milliseconds(options_.max_session_duration_ms());

  LOG(INFO) << "Deadline set to " << deadline
            << " because max_session_duration_ms was "
            << options_.max_session_duration_ms()
            << " and session_creation_timestamp_ns was "
            << options_.session_creation_timestamp_ns() << " ["
            << session_created_ts << "]";

  // Prepare a list of clients.
  clients_.reserve(options_.service_addresses_size());

  ProfileRequest request = request_;
  for (auto& service_address : options_.service_addresses()) {
    std::string resolved_service_address = resolver_(service_address);
    request.set_host_name(resolved_service_address);

    // Creation also issues Profile RPC asynchronously.
    auto client = RemoteProfilerSession::Create(resolved_service_address,
                                                deadline, request);
    clients_.push_back(std::move(client));
  }

  LOG(INFO) << "Issued Profile gRPC to " << clients_.size() << " clients";
  return Status::OK();
}

std::vector<RemoteProfilerSessionManager::Response>
RemoteProfilerSessionManager::WaitForCompletion() {
  mutex_lock lock(mutex_);
  std::vector<RemoteProfilerSessionManager::Response> remote_responses(
      clients_.size());

  for (int32 idx = 0; idx < clients_.size(); ++idx) {
    auto& remote_response = remote_responses[idx];
    auto* client = clients_[idx].get();
    remote_response.profile_response =
        client->WaitForCompletion(remote_response.status);
    remote_response.service_address = std::string(client->GetServiceAddress());
  }
  return remote_responses;
}

}  // namespace profiler
}  // namespace tensorflow
