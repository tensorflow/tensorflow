/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_REPLICATE_CONSTANTS_PASS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_REPLICATE_CONSTANTS_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

// Small constants are replicated to the hosts of their successors. This pass
// only applies when there are multiple successors.
//
// For example, the graph:
//   C -> {Op0, Op1, Op2, Op3}
//   C's assigned_device is /job:tpu_host_worker/replica:0/task:0/device:CPU:0
//   Op0's assigned_device is /job:tpu_host_worker/replica:0/task:0/device:TPU:0
//   Op1's assigned_device is /job:tpu_host_worker/replica:0/task:0/device:TPU:1
//   Op2's assigned_device is /job:tpu_host_worker/replica:0/task:1/device:TPU:0
//   Op3's assigned_device is /job:tpu_host_worker/replica:0/task:1/device:TPU:1
// is rewritten to:
//   C0 -> {Op0, Op1}
//   C1 -> {Op2, Op3}
//   C0's assigned_device is /job:tpu_host_worker/replica:0/task:0/device:CPU:0
//   C1's assigned_device is /job:tpu_host_worker/replica:0/task:1/device:CPU:0
//   Op0's assigned_device is /job:tpu_host_worker/replica:0/task:0/device:TPU:0
//   Op1's assigned_device is /job:tpu_host_worker/replica:0/task:0/device:TPU:1
//   Op2's assigned_device is /job:tpu_host_worker/replica:0/task:1/device:TPU:0
//   Op3's assigned_device is /job:tpu_host_worker/replica:0/task:1/device:TPU:1

namespace tensorflow {

class ReplicateConstantsPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_REPLICATE_CONSTANTS_PASS_H_
