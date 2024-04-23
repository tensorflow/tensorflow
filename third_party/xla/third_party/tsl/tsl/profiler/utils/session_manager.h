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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_SESSION_MANAGER_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_SESSION_MANAGER_H_

#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {

// Validate RemoteProfilerSessionManagerOptions.
absl::Status ValidateRemoteProfilerSessionManagerOptions(
    const tensorflow::RemoteProfilerSessionManagerOptions& options);

// Get RemoteSessionManagerOptions from logdir and opts.
tensorflow::RemoteProfilerSessionManagerOptions
GetRemoteSessionManagerOptionsLocked(
    absl::string_view logdir,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        opts);

// Get RemoteSessionManagerOptions from provided options.
tensorflow::RemoteProfilerSessionManagerOptions
GetRemoteSessionManagerOptionsLocked(
    absl::string_view service_addresses, absl::string_view logdir,
    absl::string_view worker_list, bool include_dataset_ops,
    int32_t duration_ms,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        opts,
    bool* is_cloud_tpu_session);

// Validate Host Port pair.
absl::Status ValidateHostPortPair(absl::string_view host_port);
}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_SESSION_MANAGER_H_
