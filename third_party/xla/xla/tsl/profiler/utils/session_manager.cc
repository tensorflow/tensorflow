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
#include <climits>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
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

struct SetAdvancedOption {
  tensorflow::ProfileOptions* options;
  const std::string& key;
  void operator()(int value) {
    (*options->mutable_advanced_configuration())[key].set_int64_value(value);
  }
  void operator()(const std::string& value) {
    (*options->mutable_advanced_configuration())[key].set_string_value(value);
  }
  void operator()(bool value) {
    (*options->mutable_advanced_configuration())[key].set_bool_value(value);
  }
};

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
// If override_hostnames is also specified, the order of hosts in
// service_addresses and override_hostnames must match.
void AddServiceAddresses(absl::string_view service_addresses,
                         RemoteProfilerSessionManagerOptions* options) {
  for (absl::string_view server : absl::StrSplit(service_addresses, ',')) {
    options->add_service_addresses(server.data(), server.size());
  }
}

}  // namespace

RemoteProfilerSessionManagerOptions GetRemoteSessionManagerOptionsLocked(
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
    } else if (key == "session_id") {
      SetOption<std::string>(
          key, kw.second,
          [&options](tensorflow::ProfileOptions*, std::string value) {
            options.mutable_profiler_options()->set_session_id(value);
          },
          nullptr);
    } else if (key == "override_hostnames") {
      // A comma-separated list of hostnames that should be used to save the
      // profile results. The order of these hostnames must match the order of
      // service_addresses.
      SetOption<std::string>(
          key, kw.second,
          [&options](tensorflow::ProfileOptions*, std::string value) {
            (*options.mutable_profiler_options()
                  ->mutable_advanced_configuration())["override_hostnames"]
                .set_string_value(value);
          },
          nullptr);
    } else if (key == "tracemark_lower") {
      SetOption<int>(
          key, kw.second,
          [](tensorflow::ProfileOptions* options, int value) {
            if (value > INT_MAX) {
              LOG(WARNING) << "tracemark_lower value is too large: " << value
                           << ", setting to INT_MAX";
              value = INT_MAX;
            }
            (*options->mutable_advanced_configuration())["tracemark_lower"]
                .set_int64_value(value);
          },
          options.mutable_profiler_options());
    } else if (key == "tracemark_upper") {
      SetOption<int>(
          key, kw.second,
          [](tensorflow::ProfileOptions* options, int value) {
            if (value > INT_MAX) {
              LOG(WARNING) << "tracemark_upper value is too large: " << value
                           << ", setting to INT_MAX";
              value = INT_MAX;
            }
            (*options->mutable_advanced_configuration())["tracemark_upper"]
                .set_int64_value(value);
          },
          options.mutable_profiler_options());
    } else if (absl::StartsWith(key, "tpu_")) {
      std::visit(
          SetAdvancedOption{options.mutable_profiler_options(), kw.first},
          kw.second);
    } else {
      LOG(WARNING) << "Unrecognised key: " << key;
    }
  }

  return options;
}

RemoteProfilerSessionManagerOptions GetRemoteSessionManagerOptionsLocked(
    absl::string_view service_addresses, absl::string_view logdir,
    absl::string_view worker_list, bool include_dataset_ops,
    int32_t duration_ms,
    const absl::flat_hash_map<std::string,
                              std::variant<bool, int, std::string>>& opts,
    bool* is_cloud_tpu_session) {
  auto options = GetRemoteSessionManagerOptionsLocked(logdir, opts);

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

absl::Status ValidateRemoteProfilerSessionManagerOptions(
    const RemoteProfilerSessionManagerOptions& options) {
  if (options.service_addresses().empty()) {
    return absl::InvalidArgumentError("No service address provided.");
  }

  if (options.profiler_options().duration_ms() == 0) {
    return absl::InvalidArgumentError("duration_ms must be greater than zero.");
  }

  for (absl::string_view host_port : options.service_addresses()) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(host_port));
  }

  if (options.max_session_duration_ms() <
      options.profiler_options().duration_ms()) {
    return absl::InvalidArgumentError(
        "The maximum profiling session duration must be greater than or equal "
        "to the local profiler duration.");
  }

  return absl::OkStatus();
}

absl::Status ValidateHostPortPair(absl::string_view host_port) {
  uint32_t port;
  std::vector<absl::string_view> parts = absl::StrSplit(host_port, ':');
  // Must be host:port, port must be a number, host must not contain a '/',
  // host also must not be empty.
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[1], &port) ||
      absl::StrContains(parts[0], "/") || parts[0].empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not interpret \"", host_port, "\" as a host-port pair."));
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tsl
