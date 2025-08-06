/* Copyright 2023 The TensorFlow Authors All Rights Reserved.

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

#include "xla/tsl/profiler/utils/session_manager.h"

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::RemoteProfilerSessionManagerOptions;

// Profiler gives grace after profiling duration to terminate.
constexpr absl::Duration kMinSessionGraceTime = absl::Seconds(60);

// Helper template function to set integer options in ProfilerOptions.
template <typename T, typename Setter>
void SetOption(absl::string_view key,
               const std::variant<bool, int, std::string>& value, Setter setter,
               tensorflow::ProfileOptions* profiler_options) {
  if (std::holds_alternative<T>(value)) {
    auto int_value = std::get<T>(value);
    setter(profiler_options, int_value);
    VLOG(1) << key << " set to " << int_value;
  } else {
    LOG(WARNING) << key << " expects an " << typeid(T).name() << " value.";
  }
}

// Sets gRPC deadline to a grace period based on the profiling duration.
void UpdateMaxSessionDuration(RemoteProfilerSessionManagerOptions& options) {
  auto local_profiler_duration = options.profiler_options().duration_ms();
  auto session_creation_ts = options.session_creation_timestamp_ns();
  auto requested_start_ts = options.profiler_options().start_timestamp_ns();
  // User only needs to set maximal session duration if the profiling duration
  // is bounded.
  DCHECK_GT(local_profiler_duration, 0);
  VLOG(3) << "duration_ms was given as " << local_profiler_duration;
  // Max session duration is the profiling session with grace time.
  auto profile_duration = std::max(
      kMinSessionGraceTime, absl::Milliseconds(local_profiler_duration) * 2);
  absl::Duration delay_duration;
  // When requested start timestamp is 0, profiling starts immediately.
  if (requested_start_ts > 0) {
    delay_duration =
        absl::Nanoseconds(requested_start_ts - session_creation_ts);
  }

  auto max_session_duration = profile_duration + delay_duration;
  options.set_max_session_duration_ms(
      absl::ToInt64Milliseconds(max_session_duration));
  VLOG(1) << "max_session_duration set to " << max_session_duration;
}

// Receives a comma delimited list of service_addresses and adds them to
// RemoteProfilerSessionManagerOptions::service_addresses.
void AddServiceAddresses(absl::string_view service_addresses,
                         RemoteProfilerSessionManagerOptions* options) {
  for (absl::string_view server : absl::StrSplit(service_addresses, ',')) {
    options->add_service_addresses(server.data(), server.size());
  }
}

}  // namespace
// Takes profiler options in absl::flat_hash_map and returns a
// RemoteProfilerSessionManagerOptions.
RemoteProfilerSessionManagerOptions
GetRemoteSessionManagerOptionsLockedWithBoolOpts(
    absl::string_view logdir,
    const absl::flat_hash_map<std::string,
                              std::variant<bool, int, std::string>>& opts) {
  RemoteProfilerSessionManagerOptions options;
  *options.mutable_profiler_options() = tsl::ProfilerSession::DefaultOptions();
  // Store a timestamp of when this session was created. This will be the basis
  // of gRPC deadline afterwards.
  auto now = absl::Now();
  options.set_session_creation_timestamp_ns(absl::ToUnixNanos(now));
  VLOG(2) << "set_session_creation_timestamp_ns set to "
          << options.session_creation_timestamp_ns() << " [" << now << "]";

  // Set the path of where to store XSpaces.
  options.mutable_profiler_options()->set_repository_path(logdir.data(),
                                                          logdir.size());
  VLOG(2) << "repository_path set to "
          << options.profiler_options().repository_path();

  for (const auto& kw : opts) {
    absl::string_view key = kw.first;
    if (key == "host_tracer_level") {
      SetOption<int>(
          key, kw.second,
          [](tensorflow::ProfileOptions* options, int value) {
            options->set_host_tracer_level(value);
          },
          options.mutable_profiler_options());
    } else if (key == "device_tracer_level") {
      SetOption<int>(
          key, kw.second,
          [](tensorflow::ProfileOptions* options, int value) {
            options->set_device_tracer_level(value);
          },
          options.mutable_profiler_options());
    } else if (key == "python_tracer_level") {
      SetOption<int>(
          key, kw.second,
          [](tensorflow::ProfileOptions* options, int value) {
            options->set_python_tracer_level(value);
          },
          options.mutable_profiler_options());
    } else if (key == "delay_ms") {
      SetOption<int>(
          key, kw.second,
          [&options](tensorflow::ProfileOptions*, int value) {
            options.set_delay_ms(value);
          },
          nullptr);
    } else {
      LOG(WARNING) << "Unrecognised key: " << key;
    }
  }

  return options;
}

RemoteProfilerSessionManagerOptions
GetRemoteSessionManagerOptionsLockedWithBoolOpts(
    absl::string_view service_addresses, absl::string_view logdir,
    absl::string_view worker_list, bool include_dataset_ops,
    int32_t duration_ms,
    const absl::flat_hash_map<std::string,
                              std::variant<bool, int, std::string>>& opts,
    bool* is_cloud_tpu_session) {
  auto options = GetRemoteSessionManagerOptionsLockedWithBoolOpts(logdir, opts);

  // Remote profiling does not support any use cases where the following options
  // are set by `opts`. e.g. `opts['service_addrs']` will not happen.
  DCHECK(options.service_addresses().empty());
  // In remote profiling, duration is always passed by value explicitly and not
  // set in opts.
  DCHECK_EQ(options.profiler_options().duration_ms(), 0);
  // Because duration_ms is not set from opts, it follows that
  // max_session_duration_ms must be unset as well.
  DCHECK_EQ(options.max_session_duration_ms(), 0);

  // Worker_list is only used for TensorBoard TPU capture cases. For a TPU
  // cluster, service_address is the Master, which can already be found in the
  // list of workers. These sessions will be used with the ProfileAnalysis
  // service.
  *is_cloud_tpu_session = !worker_list.empty();
  AddServiceAddresses(*is_cloud_tpu_session ? worker_list : service_addresses,
                      &options);

  // Set local profiler duration and profiler session durations.
  options.mutable_profiler_options()->set_include_dataset_ops(
      include_dataset_ops);
  options.mutable_profiler_options()->set_duration_ms(duration_ms);
  UpdateMaxSessionDuration(options);

  for (int idx = 0; idx < options.service_addresses_size(); ++idx) {
    VLOG(1) << "service_addr " << idx << " set to "
            << options.service_addresses(idx);
  }
  VLOG(1) << "include_dataset_ops set to " << include_dataset_ops;
  VLOG(1) << "duration_ms set to " << duration_ms;

  return options;
}

RemoteProfilerSessionManagerOptions GetRemoteSessionManagerOptionsLocked(
    absl::string_view logdir,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        opts) {
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>>
      converted_opts;
  for (const auto& [key, value] : opts) {
    converted_opts[key] = std::visit(
        [](auto&& arg) -> std::variant<bool, int, std::string> { return arg; },
        value);
  }
  return GetRemoteSessionManagerOptionsLockedWithBoolOpts(logdir,
                                                          converted_opts);
}

RemoteProfilerSessionManagerOptions GetRemoteSessionManagerOptionsLocked(
    absl::string_view service_addresses, absl::string_view logdir,
    absl::string_view worker_list, bool include_dataset_ops,
    int32_t duration_ms,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options,
    bool* is_cloud_tpu_session) {
  absl::flat_hash_map<std::string, std::variant<bool, int, std::string>>
      converted_options;
  for (const auto& [key, value] : options) {
    converted_options[key] = std::visit(
        [](auto&& arg) -> std::variant<bool, int, std::string> { return arg; },
        value);
  }
  return GetRemoteSessionManagerOptionsLockedWithBoolOpts(
      service_addresses, logdir, worker_list, include_dataset_ops, duration_ms,
      converted_options, is_cloud_tpu_session);
}

absl::Status ValidateRemoteProfilerSessionManagerOptions(
    const RemoteProfilerSessionManagerOptions& options) {
  if (options.service_addresses().empty()) {
    return tsl::errors::InvalidArgument("No service address provided.");
  }

  if (options.profiler_options().duration_ms() == 0) {
    return tsl::errors::InvalidArgument(
        "duration_ms must be greater than zero.");
  }

  for (absl::string_view host_port : options.service_addresses()) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(host_port));
  }

  if (options.max_session_duration_ms() <
      options.profiler_options().duration_ms()) {
    return tsl::errors::InvalidArgument(
        "The maximum profiling session duration must be greater than or equal "
        "to the local profiler duration.");
  }

  return absl::OkStatus();
}

absl::Status ValidateHostPortPair(absl::string_view host_port) {
  tsl::uint32 port;
  std::vector<absl::string_view> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      absl::StrContains(parts[0], "/") || parts[0].empty()) {
    return tsl::errors::InvalidArgument("Could not interpret \"", host_port,
                                        "\" as a host-port pair.");
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tsl
