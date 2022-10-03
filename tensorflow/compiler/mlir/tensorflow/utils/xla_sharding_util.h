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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_

#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {

inline constexpr absl::string_view kInputShardingAttr =
    "input_sharding_configuration";
inline constexpr absl::string_view kOutputShardingAttr =
    "output_sharding_configuration";

// Parses "input_sharding_configuration" attribute and returns a list where i-th
// element is a list of mlir::Value's which represent inputs for the TPU
// computation correponding to i-th logical device. If the attribute does not
// exist, the all inputs are placed on logical core 0.
mlir::LogicalResult ExtractInputsForLogicalDevices(
    const int num_cores_per_replica,
    mlir::tf_device::ClusterFuncOp cluster_func, mlir::OpBuilder* builder,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 4>>* input_list);

// Extracts a list of OpSharding that represent output sharding configuration of
// `tf_device.cluster`.
mlir::LogicalResult ParseAndValidateOutputSharding(
    const int num_cores_per_replica,
    mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::SmallVector<xla::OpSharding, 4>* output_sharding_list);

// Retrieves output types for TPUExecute op representing execution for provided
// logical device id. TPUExecute op for different logical device may have
// different outputs depending on the output sharding configuration.
mlir::LogicalResult GetOutputTypesForLogicalDeviceComputation(
    const int core_id, llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func,
    llvm::SmallVectorImpl<mlir::Type>* output_types,
    llvm::SmallVectorImpl<int>* cluster_to_core_index);

// Remaps outputs of `new_parallel_execute` op that represent concurrent
// execution of the `tf_device.cluster_func` at index `cluster_idx` of
// `old_parallel_execute` with its users.
// `num_results_pre_cluster` represent the # of outputs of
// `new_parallel_execute` which are from ops before `tf_device.cluster_func` op
mlir::LogicalResult RemapOutputsFromLogicalDevices(
    const mlir::Location& location,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    int num_results_pre_cluster,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute, int cluster_idx,
    mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    mlir::OpBuilder* builder);

// Determines each logical core argument to metadata argument index mapping,
// based on sharding. The return value is indexed first by logical core then by
// argument index.
llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> GetMetadataArgumentMapping(
    const tpu::TPUCompileMetadataProto& metadata);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_
