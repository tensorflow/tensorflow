/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/scatter_mlir.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;

}  // namespace

std::optional<IndexingMap> MlirScatterFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap> MlirScatterFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  auto* scatter =
      DynCast<HloScatterInstruction>(analysis_.fusion_heroes().front());
  int64_t scatter_operand_count = scatter->scatter_operand_count();
  // Scatter operands a packed in the following way:
  // Operand IDs [0, scatter_operand_count - 1] for `scatter operands`.
  // Operand ID  scatter_operand_count for `scatter indices`.
  // Operand IDs [scatter_operand_count + 1, 2 * scatter_operand_count] for
  // `scatter updates`.

  // For scatter operands we do not know the thread ID indexing.
  if (hero_operand_index < scatter_operand_count) {
    return std::nullopt;
  }
  // Compute thread id mapping based on the first update operand.
  Shape scatter_update_shape = scatter->scatter_updates().front()->shape();
  IndexingMap scatter_update_map = GetDefaultThreadIdToOutputIndexingMap(
      launch_dimensions(), config_.unroll_factor, scatter_update_shape, ctx);

  // For scatter indices we project indexing for scatter updates and take the
  // first result of the affine map only, because they coincide.
  if (hero_operand_index == scatter_operand_count) {
    Shape scatter_indices_shape = scatter->scatter_indices()->shape();
    CHECK_EQ(scatter_indices_shape.rank(), 2) << scatter->ToString();
    // Create a map from scatter update to scatter indices.
    IndexingMap updates_to_indices_map{
        mlir::AffineMap::get(
            /*dimCount=*/scatter_update_shape.rank(), /*symbolCount=*/1,
            {mlir::getAffineDimExpr(0, ctx), mlir::getAffineSymbolExpr(0, ctx)},
            ctx),
        /*dim_ranges=*/RangesFromTensorSizes(scatter_update_shape.dimensions()),
        /*symbol_ranges=*/
        RangesFromTensorSizes({scatter_indices_shape.dimensions(1)})};
    auto scatter_indices_map = scatter_update_map * updates_to_indices_map;
    scatter_indices_map.Simplify();
    return scatter_indices_map;
  }
  return scatter_update_map;
}

LaunchDimensions MlirScatterFusion::launch_dimensions() const {
  auto* scatter = analysis_.fusion_heroes().front();
  // Compute thread id mapping based on the shape of update operand.
  auto& shape = scatter->operands().back()->shape();
  return CalculateLaunchDimensions(shape, analysis_.device_info());
}

absl::Status MlirScatterFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError(
      "MlirScatterFusion::EmitMlir is not yet implemented");
}

}  // namespace gpu
}  // namespace xla
