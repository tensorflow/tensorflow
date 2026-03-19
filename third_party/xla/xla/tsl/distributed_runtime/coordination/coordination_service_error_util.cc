/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/regexp.h"

namespace tsl {
absl::Status TrimCoordinationErrorMessage(const absl::Status& s) {
  if (s.ok()) {
    return s;
  }
  auto status_message = std::string(s.message());
  auto additional_info_index = status_message.find("Additional GRPC");
  // This error didn't come from gRPC, so we don't need to trim it.
  if (additional_info_index == std::string::npos) {
    return s;
  }

  std::optional<absl::Cord> payload =
      s.GetPayload(CoordinationErrorPayloadKey());
  if (!payload.has_value() && absl::IsUnavailable(s)) {
    // This error is not provided by us, so it's probably an RPC layer error.
    auto prefix_message =
        "Failed to send RPC to coordination service. Either the leader task "
        "was preempted/died/restarted unexpectedly or this task is "
        "experiencing network issues. Check earlier logs from 1) this task, 2) "
        "the leader (usually slice 0 task 0), and 3) cluster scheduler to debug"
        " further.\n";
    status_message = absl::StrCat(
        prefix_message,
        // Replace the duplicated error message at the start with the prefix.
        status_message.substr(additional_info_index));
  } else {
    // Extract RPC called.
    std::string rpc_name;
    // Note: it is unfortunate that we have to keep the tensorflow prefix
    // because that's the RPC service proto namespace.
    RE2::PartialMatch(status_message,
                      "(/tensorflow.CoordinationService/(\\w+))", &rpc_name);
    // Erase duplicated error message.
    status_message = status_message.substr(0, additional_info_index);
    absl::StrAppend(&status_message, "\nRPC: ", rpc_name);
  }
  auto trimmed_status = absl::Status(s.code(), status_message);
  // Reattach payload.
  if (payload.has_value()) {
    trimmed_status.SetPayload(CoordinationErrorPayloadKey(), *payload);
  }
#if defined(PLATFORM_GOOGLE)
  // Reattach source locations.
  for (const auto& source_location : s.GetSourceLocations()) {
    trimmed_status.AddSourceLocation(source_location);
  }
#endif
  return trimmed_status;
}
}  // namespace tsl
