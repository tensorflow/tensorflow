/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_RPC_PROFILER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_PROFILER_RPC_PROFILER_SERVICE_IMPL_H_

#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/grpc_services.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"

namespace tensorflow {

std::unique_ptr<grpc::ProfilerService::Service> CreateProfilerService(
    const ProfilerContext& profiler_context);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_PROFILER_SERVICE_IMPL_H_
