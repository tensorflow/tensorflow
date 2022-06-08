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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TPU_CLUSTER_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TPU_CLUSTER_UTIL_H_

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

// For each TPU cluster in `module`, walk over all ops inside the cluster
// and reachable in the call graph from the cluster.
// For each op walked, `callback` is applied to the op, the root cluster, and
// the root cluster's host device. `callback` returning WasInterrupted
// indicates failure.
// The host device is null when the tpu_cluster HasModelParallelism: The
// HasModelParallelism case is currently unsupported in combination with
// outside compilation.
mlir::LogicalResult WalkReachableFromTpuCluster(
    ModuleOp module, std::function<WalkResult(Operation*, tf_device::ClusterOp,
                                              std::optional<std::string>)>
                         callback);

// Like above, except TPU clusters are not required to have a host device, and
// no host device is passed to `callback`.
mlir::LogicalResult WalkReachableFromTpuCluster(
    ModuleOp module,
    std::function<WalkResult(Operation*, tf_device::ClusterOp)> callback);

}  // namespace TFTPU
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TPU_CLUSTER_UTIL_H_
