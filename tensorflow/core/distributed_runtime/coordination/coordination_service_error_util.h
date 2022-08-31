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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_ERROR_UTIL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_ERROR_UTIL_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {

constexpr absl::string_view CoordinationErrorPayloadKey() {
  return "type.googleapis.com/tensorflow.CoordinationServiceError";
}

// Mark error as a coordination service error (as opposed to RPC
// errors).
inline Status MakeCoordinationError(Status s) {
  s.SetPayload(CoordinationErrorPayloadKey(), "");
  return s;
}

// Mark error as a coordination service error (as opposed to RPC
// errors), and indicate error origin.
// Errors reported via the agent API by the user should set `is_reported_error`
// to true.
inline Status MakeCoordinationError(Status s, const CoordinatedTask& origin,
                                    bool is_reported_error = false) {
  CoordinationServiceError error;
  *error.mutable_source_task() = origin;
  error.set_is_reported_error(is_reported_error);
  s.SetPayload(CoordinationErrorPayloadKey(), error.SerializeAsString());
  return s;
}

// Mark error as a coordination service error with payload.
inline Status MakeCoordinationError(Status s,
                                    const CoordinationServiceError& payload) {
  s.SetPayload(CoordinationErrorPayloadKey(), payload.SerializeAsString());
  return s;
}
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_ERROR_UTIL_H_
