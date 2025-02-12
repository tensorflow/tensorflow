/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_ERROR_UTIL_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_ERROR_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"

namespace tsl {

constexpr absl::string_view CoordinationErrorPayloadKey() {
  return "type.googleapis.com/tensorflow.CoordinationServiceError";
}

constexpr absl::string_view BarrierErrorPayloadKey() {
  return "type.googleapis.com/tensorflow.BarrierError";
}

// Mark error as a coordination service error (as opposed to RPC
// errors).
inline absl::Status MakeCoordinationError(absl::Status s) {
  s.SetPayload(CoordinationErrorPayloadKey(), absl::Cord(""));
  return s;
}

inline absl::Status MakeBarrierError(absl::Status s,
                                     std::string_view barrier_id,
                                     int64_t counter) {
  tensorflow::BarrierError error;
  error.set_barrier_id(std::string(barrier_id));
  error.set_counter(counter);
  s.SetPayload(BarrierErrorPayloadKey(), absl::Cord(error.SerializeAsString()));
  return MakeCoordinationError(s);
}

inline int64_t GetBarrierCounterFromError(const absl::Status& s) {
  if (s.GetPayload(BarrierErrorPayloadKey()) == std::nullopt) {
    return -1;
  }
  tensorflow::BarrierError error;
  error.ParseFromString(
      std::string(s.GetPayload(BarrierErrorPayloadKey()).value()));
  return error.counter();
}

// Mark error as a coordination service error (as opposed to RPC
// errors), and indicate error origin.
// Errors reported via the agent API by the user should set `is_reported_error`
// to true.
inline absl::Status MakeCoordinationError(
    absl::Status s, const tensorflow::CoordinatedTask& origin,
    bool is_reported_error = false) {
  tensorflow::CoordinationServiceError error;
  *error.mutable_source_task() = origin;
  error.set_is_reported_error(is_reported_error);
  s.SetPayload(CoordinationErrorPayloadKey(),
               absl::Cord(error.SerializeAsString()));
  return s;
}

// Mark error as a coordination service error with payload.
inline absl::Status MakeCoordinationError(
    absl::Status s, const tensorflow::CoordinationServiceError& payload) {
  s.SetPayload(CoordinationErrorPayloadKey(),
               absl::Cord(payload.SerializeAsString()));
  return s;
}

// Trims the error message by replacing the `Additional GRPC error` part.
// Note: The duplicated error message is a quirk of the underlying gRPC code
// that we are using. Changing the shared code may hide important messages for
// other libraries, so we trim the error message for coordination service
// instead. See tsl/distributed_runtime/rpc/grpc_state.h for more details.
absl::Status TrimCoordinationErrorMessage(const absl::Status& s);

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_ERROR_UTIL_H_
