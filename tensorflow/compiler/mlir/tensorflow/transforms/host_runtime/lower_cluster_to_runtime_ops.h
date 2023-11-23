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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_LOWER_CLUSTER_TO_RUNTIME_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_LOWER_CLUSTER_TO_RUNTIME_OPS_H_

#include "absl/base/attributes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/core/lib/core/status.h"
#include "tsl/framework/device_type.h"

namespace tensorflow {
namespace tfrt_compiler {

// Given a MLIR module with tf_device.cluster ops, insert specific Runtime ops
// such as TPUExecute or XlaExecute depending on the device type and specific
// host runtime. Also does some optimization. Will return an error if it fails.
// The output Runtime ops depends on both Device Type and Runtime Host.
//
// Input:
//     Tensorflow Dialect MLIR with tf_device.cluster ops and virtual devices.
//     xla_device_type - The device type that is being targeted.
// Output:
//     Tensorflow Dialect MLIR with Runtime specific ops. All tf_device.cluster
//     ops are removed. Physical devices are assigned to ops instead of virtual
//     devices.
tensorflow::Status RunLowerClusterToRuntimeOpsPassPipeline(
    mlir::ModuleOp module, tsl::DeviceType xla_device_type,
    llvm::StringRef module_name = llvm::StringRef());

// The same API as RunLowerClusterToRuntimeOpsPassPipeline but as an MLIR pass
// pipeline.
void RegisterTPULowerClusterToRuntimeOpsPassPipeline();
void RegisterNonTPULowerClusterToRuntimeOpsPassPipeline();

}  // namespace tfrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_LOWER_CLUSTER_TO_RUNTIME_OPS_H_
