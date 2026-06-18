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
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_TPU_METADATA_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_TPU_METADATA_UTILS_H_

#include <optional>

#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace mlir {
namespace TFTPU {

// Populates a TPUCompileMetadataProto from attributes of a
// `tf_device::ClusterFuncOp`. If any necessary attributes are missing from the
// op, a failure will be returned.
// TODO(lyandy): Support session handle and guaranteed consts.
LogicalResult SetMetadataProtoFromClusterFuncOp(
    tf_device::ClusterFuncOp op, int num_replicas, int num_cores_per_replica,
    std::optional<xla::DeviceAssignmentProto>&& xla_device_assignment,
    tensorflow::tpu::TPUCompileMetadataProto* metadata);
}  // namespace TFTPU
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_HOST_RUNTIME_TPU_METADATA_UTILS_H_
