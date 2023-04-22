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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
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
// LIMITATIONS:
// Due to some implementation details, this API won't initialize TPU pods, but
// only donuts or slices of donuts.
Status InitializeTPUSystemGlobally(Env* env, tpu::TopologyProto* tpu_topology);

Status InitializeTPUSystemGlobally();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_GLOBAL_INIT_H_
