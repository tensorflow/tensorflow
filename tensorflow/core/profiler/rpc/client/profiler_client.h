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

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/profiler_analysis.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"

namespace tensorflow {
namespace profiler {

Status ProfileGrpc(const std::string& service_addr,
                   const ProfileRequest& request, ProfileResponse* response);

Status NewSessionGrpc(const std::string& service_addr,
                      const NewProfileSessionRequest& request,
                      NewProfileSessionResponse* response);

Status MonitorGrpc(const std::string& service_addr,
                   const MonitorRequest& request, MonitorResponse* response);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_PROFILER_CLIENT_H_
