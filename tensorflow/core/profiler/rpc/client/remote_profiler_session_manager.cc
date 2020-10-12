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
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr uint64 kMaxEvents = 1000000;

// TODO(yisitu) merge with the implementation in capture_profile.
void PopulateProfileRequest(const RemoteProfilerSessionManagerOptions& options,
                            absl::string_view session_id,
                            absl::string_view host_name,
                            ProfileRequest* request) {
  request->set_max_events(kMaxEvents);
  request->set_repository_root(options.profiler_options().repository_path());
  request->set_session_id(session_id.data(), session_id.size());
  request->add_tools("trace_viewer");
  request->add_tools("op_profile");
  request->add_tools("input_pipeline");
  request->add_tools("kernel_stats");
  request->add_tools("memory_viewer");
  request->add_tools("memory_profile");
  request->add_tools("overview_page");
  request->add_tools("pod_viewer");
  request->add_tools("tensorflow_stats");
  request->set_host_name(host_name.data(), host_name.size());
  *request->mutable_opts() = options.profiler_options();
  request->set_duration_ms(options.profiler_options().duration_ms());
}

}  // namespace

/*static*/ std::unique_ptr<RemoteProfilerSessionManager>
RemoteProfilerSessionManager::Create(
    const RemoteProfilerSessionManagerOptions& options,
    tensorflow::Status& out_status, AddressResolver resolver) {
  VLOG(1) << "Creating a RemoteProfilerSessionManager.";
  auto session_manager =
      absl::WrapUnique(new RemoteProfilerSessionManager(options, resolver));
  out_status = session_manager->Init();
  if (!out_status.ok()) {
    return nullptr;
  }
  return session_manager;
}

RemoteProfilerSessionManager::RemoteProfilerSessionManager(
    RemoteProfilerSessionManagerOptions options, AddressResolver resolver)
    : options_(std::move(options)) {
  if (resolver) {
    resolver_ = std::move(resolver);
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
  // TODO(b/169482824) Move validation to call site.
  Status status = ValidateOptionsLocked();
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }

  std::string session_id = GetCurrentTimeStampAsString();
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

  for (auto& service_addr : options_.service_addresses()) {
    std::string resolved_service_addr = resolver_(service_addr);
    ProfileRequest profile_request;
    PopulateProfileRequest(options_, session_id, resolved_service_addr,
                           &profile_request);

    // Creation also issues Profile RPC asynchronously.
    auto client = RemoteProfilerSession::Create(
        std::move(resolved_service_addr), deadline, std::move(profile_request));

    clients_.push_back(std::move(client));
  }

  LOG(INFO) << "Issued Profile gRPC to " << clients_.size() << " clients";
  return Status::OK();
}

Status RemoteProfilerSessionManager::ValidateOptionsLocked() {
  if (!options_.service_addresses_size()) {
    return errors::InvalidArgument("No service addresses specified.");
  }

  if (options_.profiler_options().duration_ms() == 0) {
    if (options_.max_session_duration_ms() != 0) {
      return errors::InvalidArgument(
          "If local profiler duration is unbounded, profiling session duration "
          "must be unbounded.");
    }
  }

  if (options_.max_session_duration_ms() <
      options_.profiler_options().duration_ms()) {
    return errors::InvalidArgument(
        "The maximum profiling session duration must be greater than or equal "
        "to the local profiler duration.");
  }
  return Status::OK();
}

std::vector<RemoteProfilerSessionManager::Response>
RemoteProfilerSessionManager::WaitForCompletion() {
  mutex_lock lock(mutex_);
  std::vector<RemoteProfilerSessionManager::Response> remote_responses;
  remote_responses.reserve(clients_.size());

  for (auto& client : clients_) {
    remote_responses.emplace_back();
    auto* profile_response = &remote_responses.back().profile_response;
    Status& status = remote_responses.back().status;
    std::string* service_addr = &remote_responses.back().service_addr;
    *profile_response = client->WaitForCompletion(status);
    *service_addr = std::string(client->GetServiceAddress());
  }
  return remote_responses;
}

}  // namespace profiler
}  // namespace tensorflow
