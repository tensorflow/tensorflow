/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_TPU_GLOBAL_INIT_H_
#define TENSORFLOW_CORE_TPU_TPU_GLOBAL_INIT_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"

namespace tensorflow {

// Initializes the TPU system globally. The state of initialization can then be
// shared by different sessions running on these TPUs, on the same process. This
// API is provided for multi-tenant usecases where multiple sessions in a
// process are using the same set of TPUs.
//
// Returns status errors if initialization is unsuccessful and returns the TPU
// TopologyProto as an output parameter.
//
// REQUIRES:
// * Call this API before any sessions using TPUs are run.
// * If you are using this API for initialization, please don't use the TPU
// configuration ops within your graph. This will cause errors to be returned
// from the API which is called second.
//
// DISTRIBUTED SETUP:
// To properly initialize a TPU topology that is beyond donut level, caller is
// required to provide correct following arguments:
//
// 1. job_name
// The name of the job under distributed settings. For example, if the job is
// '/job:tpu_worker/replica:0/task:0/...', the "tpu_worker" is the desired
// job_name here.
//
// 2. session_target
// The target string that will be used to create a Session and run the
// distributed TPU initialization graph. Generally this would be the master
// session from the cluster.
//
// 3.device_set
// The GLOBAL set of devices in the distributed setting, including proper
// "TPU_SYSTEM" devices across all tasks.
// For example, device_set should contain two "TPU_SYSTEM" devices on 2 tasks
// for a 4x2 (2 TPU workers) setup, and other non "TPU_SYSTEM" devices.
absl::Status InitializeTPUSystemGlobally(absl::string_view job_name,
                                         absl::string_view session_target,
                                         const DeviceSet& device_set, Env* env,
                                         tpu::TopologyProto* tpu_topology);

absl::Status InitializeTPUSystemGlobally(Env* env,
                                         tpu::TopologyProto* tpu_topology);

absl::Status InitializeTPUSystemGlobally();

}  // namespace tensorflow

// Many clients rely on ADL to lookup InitializeTPUSystemGlobally, now that Env
// moved to namespace tsl they are all broken without these forwarding
// declarations.
namespace tsl {
using tensorflow::InitializeTPUSystemGlobally;  // NOLINT
}

#endif  // TENSORFLOW_CORE_TPU_TPU_GLOBAL_INIT_H_
