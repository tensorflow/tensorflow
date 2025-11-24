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

#include <stdbool.h>

#include <cstdint>
#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {

inline constexpr llvm::StringRef kInputShardingAttr =
    "input_sharding_configuration";
inline constexpr llvm::StringRef kOutputShardingAttr =
    "output_sharding_configuration";

inline constexpr llvm::StringRef kICIWeightDistributionMlirBridgeMarker =
    "_ici_weight_distribution_mlir_bridge_marker";

// Parses the sharding string. This sharding string can be binary (serialized)
// or human readable.
mlir::LogicalResult DecodeShardingAttribute(const std::string& shard_str,
                                            xla::OpSharding& sharding,
                                            bool report_error = true);

// Encodes the sharding in human readable form.
mlir::LogicalResult DecodeShardingAttribute(mlir::Attribute shard_attr,
                                            xla::OpSharding& sharding,
                                            bool report_error = true);

// Parses the sharding attr. This sharding attr can be binary (serialized)
// or human readable.
void EncodeSharding(mlir::Operation* op, llvm::StringRef shard_str);

// Parses "input_sharding_configuration" attribute and returns a list where i-th
// element is a list of mlir::Value's which represent inputs for the TPU
// computation corresponding to i-th logical device. If the attribute does not
// exist, the all inputs are placed on logical core 0.
mlir::LogicalResult ExtractInputsForLogicalDevices(
    int num_cores_per_replica, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::OpBuilder* builder,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 4>>* input_list);

// Same as above, except creates tf.XlaSplitND Op for split sharding if
// use_xla_nd_ops is true, otherwise creates tf.Split op.
mlir::LogicalResult ExtractInputsForLogicalDevices(
    int num_cores_per_replica, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::OpBuilder* builder, bool use_xla_nd_ops,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 4>>* input_list);

// Extracts a list of OpSharding that represent output sharding configuration of
// `tf_device.cluster`.
mlir::LogicalResult ParseAndValidateOutputSharding(
    int num_cores_per_replica, mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::SmallVector<xla::OpSharding, 4>* output_sharding_list);

// Retrieves output types for TPUExecute op representing execution for provided
// logical device id. TPUExecute op for different logical device may have
// different outputs depending on the output sharding configuration.
mlir::LogicalResult GetOutputTypesForLogicalDeviceComputation(
    int core_id, llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func,
    llvm::SmallVectorImpl<mlir::Type>* output_types,
    llvm::SmallVectorImpl<int>* cluster_to_core_index);

// Same as above, except creates tf.XlaSplitND Op for split sharding if
// use_xla_nd_ops is true, otherwise creates tf.Split op.
mlir::LogicalResult GetOutputTypesForLogicalDeviceComputation(
    int core_id, llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func,
    llvm::SmallVectorImpl<mlir::Type>* output_types, bool use_xla_nd_ops,
    llvm::SmallVectorImpl<int>* cluster_to_core_index);

// Remaps outputs of `new_parallel_execute` op that represent concurrent
// execution of the `tf_device.cluster_func` at index `cluster_idx` of
// `old_parallel_execute` with its users.
// `num_results_pre_cluster` represent the # of outputs of
// `new_parallel_execute` which are from ops before `tf_device.cluster_func` op.
mlir::LogicalResult RemapOutputsFromLogicalDevices(
    const mlir::Location& location,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    int num_results_pre_cluster,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute, int cluster_idx,
    mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    mlir::OpBuilder* builder);

// Same as above, except creates tf.XlaConcatNd Op for split sharding if
// use_xla_nd_ops is true, otherwise creates tf.Concat op.
mlir::LogicalResult RemapOutputsFromLogicalDevices(
    const mlir::Location& location,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    int num_results_pre_cluster,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute, int cluster_idx,
    mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    bool use_xla_nd_ops, mlir::OpBuilder* builder);

// Determines each logical core argument to metadata argument index mapping,
// based on sharding. The return value is indexed first by logical core then by
// argument index.
llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> GetMetadataArgumentMapping(
    const tpu::TPUCompileMetadataProto& metadata);

// Gets the proper tensor dimension from XLA OpSharding.
// "replicate_on_last_tile_dim" and "last_tile_dims" should be deducted from the
// real Tensor dimensions when tiled.
// For example:
// f32[8,512](sharding={devices=[1,1,2]0,1 last_tile_dims={REPLICATED})
// also means a replicated tensor over all devices.
//
// See xla_data.proto for detailed explanations on the fields.
int GetDimsFromXLAShardingTiled(const xla::OpSharding& xla_sharding);

// A sharding with OTHER type may be REPLICATED if:
// 'replicate_on_last_tile_dim' is true OR
// 'last_tile_dims' is not empty
// AND
// other than replicated last tile dims, all other dims are not sharded.
bool IsOtherReplicatedSharding(const xla::OpSharding& xla_sharding);

// Returns whether the sharding is split sharding. i.e. A sharding with OTHER
// type but not replicated.
bool IsSplitSharding(const xla::OpSharding& sharding);

// Returns whether the sharding is replicated. It includes sharding with
// REPLICATED type and replicated OTHER type.
bool IsReplicatedSharding(const xla::OpSharding& sharding);

// Returns whether the shape of inputs and outputs is statically known when
// split sharding is done on inputs or outputs.
bool AreInputOutputShapesStaticallyKnownForSplitSharding(
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func);

// Returns a map of dimension indices and number of splits for tiled sharding.
absl::StatusOr<std::map<int, int>> GetDimensionIndicesAndNumSplitsFromSharding(
    const xla::OpSharding& sharding);

// Verifies that the two sharding attributes are equivalent, by converting them
// to OpSharding and then HloSharding and performing a comparison. Returns
// failure if the attributes are not equivalent or if there is any problem
// in converting the attributes to OpSharding or HloSharding.
mlir::LogicalResult VerifyShardingEquivalent(
    const xla::OpSharding& sharding_proto1,
    const xla::OpSharding& sharding_proto2);

// Returns _XlaShardingV2 if it exists or _XlaSharding otherwise. When both
// _XlaSharding and _XlaShardingV2 exist, verifies that they are equivalent
// and returns an error status if they aren't equivalent.
absl::StatusOr<mlir::StringAttr> GetXlaShardingAttrFromShardingOp(
    mlir::TF::XlaShardingOp sharding);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_
