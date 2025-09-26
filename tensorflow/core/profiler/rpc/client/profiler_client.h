/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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
// GRPC client to perform on-demand profiling

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_

#include <string>

#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "xla/tsl/profiler/rpc/client/profiler_client.h"

namespace tensorflow {
namespace profiler {

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status MonitorGrpc(const std::string& service_address,
                                const tensorflow::MonitorRequest& request,
                                tensorflow::MonitorResponse* response) {
  return tsl::profiler::MonitorGrpc(service_address, request, response);
}

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NewSessionGrpc(
    const std::string& service_address,
    const tensorflow::NewProfileSessionRequest& request,
    tensorflow::NewProfileSessionResponse* response) {
  return tsl::profiler::NewSessionGrpc(service_address, request, response);
}

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ProfileGrpc(const std::string& service_address,
                                const tensorflow::ProfileRequest& request,
                                tensorflow::ProfileResponse* response) {
  return tsl::profiler::ProfileGrpc(service_address, request, response);
}

using RemoteProfilerSession ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::RemoteProfilerSession;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
