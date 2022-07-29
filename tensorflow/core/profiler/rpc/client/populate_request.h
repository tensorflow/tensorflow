/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_POPULATE_REQUEST_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_POPULATE_REQUEST_H_

#include <vector>

#include "absl/strings/str_split.h"
#include "tensorflow/core/profiler/rpc/client/remote_profiler_session_manager.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::RemoteProfilerSessionManager;

constexpr uint64 kMaxEvents = 1000000;
const absl::string_view kXPlanePb = "xplane.pb";

MonitorRequest PopulateMonitorRequest(int duration_ms, int monitoring_level,
                                      bool timestamp) {
  MonitorRequest request;
  request.set_duration_ms(duration_ms);
  request.set_monitoring_level(monitoring_level);
  request.set_timestamp(timestamp);
  return request;
}

ProfileRequest PopulateProfileRequest(
    absl::string_view repository_root, absl::string_view session_id,
    absl::string_view host_name,
    const RemoteProfilerSessionManagerOptions& options) {
  ProfileRequest request;
  // TODO(b/169976117) Remove duration from request.
  request.set_duration_ms(options.profiler_options().duration_ms());
  request.set_max_events(kMaxEvents);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  request.set_host_name(host_name.data(), host_name.size());
  // These tools are only used by TPU profiler.
  request.add_tools("trace_viewer");
  request.add_tools("op_profile");
  request.add_tools("input_pipeline");
  request.add_tools("kernel_stats");
  request.add_tools("memory_viewer");
  request.add_tools("memory_profile");
  request.add_tools("overview_page");
  request.add_tools("pod_viewer");
  request.add_tools("tensorflow_stats");
  // XPlane tool is only used by OSS profiler and safely ignored by TPU
  // profiler.
  request.add_tools(kXPlanePb.data(), kXPlanePb.size());
  *request.mutable_opts() = options.profiler_options();
  return request;
}

NewProfileSessionRequest PopulateNewProfileSessionRequest(
    absl::string_view repository_root, absl::string_view session_id,
    const RemoteProfilerSessionManagerOptions& opts) {
  NewProfileSessionRequest request;
  std::vector<absl::string_view> parts =
      absl::StrSplit(opts.service_addresses(0), ':');
  DCHECK(!parts.empty());

  *request.mutable_request() =
      PopulateProfileRequest(repository_root, session_id, parts[0], opts);
  request.set_repository_root(repository_root.data(), repository_root.size());
  request.set_session_id(session_id.data(), session_id.size());
  for (const auto& hostname : opts.service_addresses()) {
    request.add_hosts(hostname);
  }
  return request;
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_POPULATE_REQUEST_H_
