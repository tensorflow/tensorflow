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
#include "xla/python/ifrt/ir/conversions/mpmd/utils.h"

#include <cstdint>
#include <functional>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

namespace xla::ifrt::mpmd {

namespace sdy = ::mlir::sdy;
using ::mlir::DenseSet;
using ::mlir::func::FuncOp;
using ::mlir::mpmd::MeshTensorType;

xla::HloSharding GetHloSharding(MeshTensorType mesh_tensor_type,
                                sdy::MeshAttr sdy_mesh_attr) {
  sdy::TensorShardingAttr sharding = mesh_tensor_type.getSharding();
  // If there is no sharding, it means the mesh_tensor_type is fully replicated
  // but we need to pass a non-null sharding to the conversion function.
  if (!sharding) {
    sharding = sdy::TensorShardingAttr::getFullyClosed(
        mesh_tensor_type.getContext(),
        mesh_tensor_type.getRankedTensorType().getRank(),
        xla::sdy::kGlobalMeshName);
  }
  return xla::sdy::convertToHloSharding(
      sharding, [&](sdy::TensorShardingAttr sharding) { return sdy_mesh_attr; },
      /*manualAxes=*/{});
}

// TODO: b/353920283 - Directly convert to ShardingParam from sdy sharding. This
// implementation now converts sdy sharding to hlo sharding and then to sharding
// param.
absl::StatusOr<xla::ifrt::ShardingParam> MeshTensorTypeToShardingParam(
    MeshTensorType mesh_tensor_type, sdy::MeshAttr mesh_attr) {
  return xla::ifrt::support::ToShardingParam(
      /*hlo_sharding=*/GetHloSharding(mesh_tensor_type, mesh_attr),
      /*rank=*/mesh_tensor_type.getRankedTensorType().getRank(),
      /*num_devices=*/
      absl::c_accumulate(mesh_attr.getAxes(), 1,
                         [](int64_t acc, sdy::MeshAxisAttr axis) {
                           return acc * axis.getSize();
                         }));
}

bool IsIfrtFunction(FuncOp func_op) {
  return func_op->hasAttr(xla::ifrt::kIfrtFunctionAttrName);
}

}  // namespace xla::ifrt::mpmd
