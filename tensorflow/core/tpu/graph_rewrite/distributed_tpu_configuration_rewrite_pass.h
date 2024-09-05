/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Rewrites ConfigureDistributedTPU Op into a graph that configures each host.
//
// See the comment at the top of
// third_party/tensorflow/core/ops/tpu_configuration_ops.cc to see the
// sequence of Ops used to configure a distributed TPU system.

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_CONFIGURATION_REWRITE_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_CONFIGURATION_REWRITE_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Replaces dummy ConfigureDistributedTPU Ops assigned to TPU_SYSTEM
// devices with _ConfigureDistributedTPU and _WaitForDistributedTPU
// Ops on TPU_SYSTEM, and _InitializeHostForDistributedTPU on the CPU
// device of each host in the same job as the given TPU_SYSTEM device.
class DistributedTPUConfigurationRewritePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

// Replaces dummy ShutdownDistributedTPU Ops assigned to TPU_SYSTEM
// devices with _ShutdownDistributedTPU Ops on TPU_SYSTEM and
// _DisconnectHostFromDistributedTPUSystem on the CPU device of each
// host in the same job as the given TPU_SYSTEM device.
class DistributedTPUShutdownRewritePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_CONFIGURATION_REWRITE_PASS_H_
