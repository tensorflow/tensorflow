/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DISTRIBUTED_UTIL_H_
#define XLA_PJRT_DISTRIBUTED_UTIL_H_

#include "absl/status/status.h"
#include "grpcpp/support/status.h"

namespace xla {

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return absl::OkStatus();
  } else {
    return absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                        s.error_message());
  }
}

inline ::grpc::Status ToGrpcStatus(const absl::Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          std::string(s.message()));
  }
}

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_UTIL_H_
