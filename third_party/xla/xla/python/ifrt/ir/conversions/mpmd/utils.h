/* Copyright 2025 The OpenXLA Authors.

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
#ifndef XLA_PYTHON_IFRT_IR_CONVERSIONS_MPMD_UTILS_H_
#define XLA_PYTHON_IFRT_IR_CONVERSIONS_MPMD_UTILS_H_

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/sharding_param.h"

namespace xla::ifrt::mpmd {

// Converts a MeshTensorType to a ShardingParam.
//
// The ShardingParam has: 1) `dim_shards` matching the rank of the tensor, with
// each entry representing the number of shards for the corresponding dimension
// 2) `axis_sizes` with the sizes of the mesh dimensions, and 3) `permutations`
// of the same length as `axis_sizes` telling how the shards are mapped over
// the axis in `minor_to_major` order.
//
// For example, a MeshTensor with <["x":range<2>,"y":range<1>], f32[4{1},2{0}]>
// is converted to a ShardingParam with `dim_shards` 1x2, `permutations` [1, 0].
absl::StatusOr<xla::ifrt::ShardingParam> MeshTensorTypeToShardingParam(
    mlir::mpmd::MeshTensorType mesh_tensor_type, mlir::sdy::MeshAttr mesh_attr);

// Converts a MeshTensorType to an HloSharding.
xla::HloSharding GetHloSharding(mlir::mpmd::MeshTensorType mesh_tensor_type,
                                mlir::sdy::MeshAttr sdy_mesh_attr);

// Returns true if the function is annotated with the ifrt.function attribute.
bool IsIfrtFunction(mlir::func::FuncOp func_op);

}  // namespace xla::ifrt::mpmd

#endif  // XLA_PYTHON_IFRT_IR_CONVERSIONS_MPMD_UTILS_H_
