/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// Defines the configuration for a single task (typically a process)
// that is part of a gRPC-based TensorFlow cluster.
struct GrpcServerOptions {
  // This identity of the job to which this task belongs.  The names
  // of the devices in this task will be prefixed with
  // "/job:<job_name>/task:<task_index>"
  string job_name;
  int32 task_index = 0;

  // A channel specification, which defines (i) the set of jobs that
  // comprise the cluster, and (ii) within each job, the endpoints
  // exposed by each task. NOTE: This spec also defines the endpoint
  // on which this task will listen.
  GrpcChannelSpec channel_spec;

  // SessionOptions that will be used as defaults when configuring
  // sessions in this task. `default_session_options.target` is
  // ignored.
  SessionOptions default_session_options;
};

// Starts a gRPC-based TensorFlow server with the given options.
// This function will not return.
void StartTensorFlowServer(const GrpcServerOptions& options);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
