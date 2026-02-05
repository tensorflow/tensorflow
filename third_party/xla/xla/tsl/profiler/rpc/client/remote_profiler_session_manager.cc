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

#include "xla/tsl/profiler/rpc/client/remote_profiler_session_manager.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/rpc/client/profiler_client.h"

namespace tsl {
namespace profiler {

namespace {

using tensorflow::ProfileRequest;
using tensorflow::RemoteProfilerSessionManagerOptions;

// Parses "override_hostnames" from the configuration to save profile results.
// Validates that the hostnames match the count and order of `service_addresses`
// in `options`. The caller needs to ensure that order of `service_addresses`
// and `override_hostnames` is same.
absl::StatusOr<std::vector<std::string>> ParseAndValidateOverrideHostnames(
    const RemoteProfilerSessionManagerOptions& options,
    ProfileRequest& request) {
  const auto* override_hostnames = gtl::FindOrNull(
      request.opts().advanced_configuration(), "override_hostnames");
  if (override_hostnames == nullptr) {
    return std::vector<std::string>();
  }
  std::vector<std::string> override_hostnames_list =
      absl::StrSplit(override_hostnames->string_value(), ',');
  if (override_hostnames_list.size() != options.service_addresses().size()) {
    return absl::InvalidArgumentError(
        "The number of override hostnames must match the number of service "
        "addresses.");
  }
  request.mutable_opts()->mutable_advanced_configuration()->erase(
      "override_hostnames");
  return override_hostnames_list;
}

}  // namespace

/*static*/ std::unique_ptr<RemoteProfilerSessionManager>
RemoteProfilerSessionManager::Create(
    const RemoteProfilerSessionManagerOptions& options,
    const ProfileRequest& request, absl::Status& out_status,
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

absl::Status RemoteProfilerSessionManager::Init() {
  absl::MutexLock lock(mutex_);
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
  clients_.reserve(options_.service_addresses().size());

  ProfileRequest request_template = request_;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> override_hostnames_list,
      ParseAndValidateOverrideHostnames(options_, request_template));

  for (size_t i = 0; i < options_.service_addresses().size(); ++i) {
    const std::string& service_address = options_.service_addresses(i);
    std::string resolved_service_address = resolver_(service_address);
    ProfileRequest request = request_template;
    request.set_host_name(resolved_service_address);
    if (i < override_hostnames_list.size()) {
      request.mutable_opts()->set_override_hostname(override_hostnames_list[i]);
    }

    // Creation also issues Profile RPC asynchronously.
    auto client = RemoteProfilerSession::Create(resolved_service_address,
                                                deadline, request);
    clients_.push_back(std::move(client));
  }

  LOG(INFO) << "Issued Profile gRPC to " << clients_.size() << " clients";
  return absl::OkStatus();
}

std::vector<RemoteProfilerSessionManager::Response>
RemoteProfilerSessionManager::WaitForCompletion() {
  absl::MutexLock lock(mutex_);
  std::vector<RemoteProfilerSessionManager::Response> remote_responses(
      clients_.size());

  for (int32_t idx = 0; idx < clients_.size(); ++idx) {
    auto& remote_response = remote_responses[idx];
    auto* client = clients_[idx].get();
    remote_response.profile_response =
        client->WaitForCompletion(remote_response.status);
    remote_response.service_address = std::string(client->GetServiceAddress());
  }
  return remote_responses;
}

}  // namespace profiler
}  // namespace tsl
