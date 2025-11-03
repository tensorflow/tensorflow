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

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/rpc/client/profiler_client.h"

namespace tsl {
namespace profiler {

using tensorflow::ProfileRequest;
using tensorflow::RemoteProfilerSessionManagerOptions;

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
  cq_.Shutdown();
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
  clients_.reserve(options_.service_addresses_size());

  ProfileRequest request = request_;
  for (int32_t idx = 0; idx < options_.service_addresses_size(); ++idx) {
    auto& service_address = options_.service_addresses(idx);
    std::string resolved_service_address = resolver_(service_address);
    request.set_host_name(resolved_service_address);

    // Creation also issues Profile RPC asynchronously.
    auto client = RemoteProfilerSession::Create(resolved_service_address,
                                                deadline, request, &cq_, idx);
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

  for (int32_t req_cnt = 0; req_cnt < clients_.size(); ++req_cnt) {
    void* got_tag = nullptr;
    bool ok = false;
    bool success = cq_.Next(&got_tag, &ok);

    if (!success) {
      LOG(ERROR) << "Completion queue drained after processing " << req_cnt
                 << " of " << clients_.size() << " clients";
      break;
    }
    if (!ok) {
      remote_responses[req_cnt].status = absl::InternalError(absl::StrCat(
          "Missing or invalid event from completion queue, got_tag: ",
          reinterpret_cast<int64_t>(got_tag),
          " got_tag:0 means either nullptr or a client_id:0"));
      LOG(ERROR) << "Missing or invalid event from completion queue, got_tag: "
                 << reinterpret_cast<int64_t>(got_tag)
                 << ". got_tag:0 means either nullptr or a client_id:0";
      continue;
    }
    int64_t client_id = reinterpret_cast<int64_t>(got_tag);
    auto* client = clients_[client_id].get();
    remote_responses[req_cnt].profile_response = client->HandleCompletion(
        remote_responses[req_cnt].status, got_tag, true);
    remote_responses[req_cnt].service_address =
        std::string(client->GetServiceAddress());
  }
  LOG(INFO) << "Completed waiting for completion of " << clients_.size()
            << " clients";

  return remote_responses;
}

}  // namespace profiler
}  // namespace tsl
