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

#include "xla/codegen/tiling/tiled_hlo_schedule.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

// Helper to validate that an iteration space is compatible with a tile offsets
// indexing map.
absl::Status ValidateIterationSpace(const IterationSpace& iteration_space,
                                    const IndexingMap& tile_offsets_indexing) {
  if (iteration_space.size() != tile_offsets_indexing.GetDimVarsCount()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected iteration space to have exactly as many dimensions as there "
        "are parameters in the tile offsets indexing map, but iteration space "
        "has %d dimensions, and tile offsets indexing map has %d dimensions.",
        iteration_space.size(), tile_offsets_indexing.GetDimVarsCount()));
  }

  std::vector<int64_t> iteration_space_dims;
  iteration_space_dims.reserve(iteration_space.size());

  for (const auto& [dim_id, dim_size] : iteration_space) {
    if (dim_id >= tile_offsets_indexing.GetDimVarsCount() || dim_id < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dimension id %d is out of bounds for tile offsets indexing map with "
          "%d dimensions. This can happen if ",
          dim_id, tile_offsets_indexing.GetDimVarsCount()));
    }

    if (absl::c_linear_search(iteration_space_dims, dim_id)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Iteration space contains multiple dimensions with id %d.", dim_id));
    }
    iteration_space_dims.push_back(dim_id);
  }
  return absl::OkStatus();
}

absl::StatusOr<IndexingMap> MajorToMinorScheduleImpl(
    const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
    gpu::SymbolicExprContext* symbolic_expr_context) {
  mlir::MLIRContext* mlir_context = symbolic_expr_context->GetMLIRContext();
  mlir::AffineExpr program_id = mlir::getAffineDimExpr(0, mlir_context);

  std::vector<int64_t> iteration_space_sizes;
  iteration_space_sizes.reserve(iteration_space.size());
  for (const auto& dim_info : iteration_space) {
    iteration_space_sizes.push_back(dim_info.dimension_size);
  }

  std::vector<mlir::AffineExpr> tile_exprs(
      tile_offsets_indexing.GetDimVarsCount(),
      mlir::getAffineConstantExpr(0, mlir_context));

  for (auto [dim_info, tile_expr] : llvm::zip(
           iteration_space, DelinearizeIndex(iteration_space_sizes, program_id,
                                             symbolic_expr_context))) {
    tile_exprs[dim_info.dimension_id] = tile_expr;
  }
  std::vector<IndexingMap::Variable> dim_vars{
      {0, Product(iteration_space_sizes) - 1, "pid_0"}};
  IndexingMap program_id_to_output_dims{
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0, tile_exprs, mlir_context),
      dim_vars, /*range_vars=*/{}, /*rt_vars=*/{}};
  auto scheduled_indexing =
      ComposeIndexingMaps(program_id_to_output_dims, tile_offsets_indexing);
  scheduled_indexing.Simplify();
  scheduled_indexing.RescaleSymbols();
  scheduled_indexing.RemoveUnusedSymbols();
  return scheduled_indexing;
}
}  // namespace

absl::StatusOr<IndexingMap> MajorToMinorTiledHloSchedule::Schedule(
    const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
    gpu::SymbolicExprContext* ctx) const {
  TF_RETURN_IF_ERROR(
      ValidateIterationSpace(iteration_space, tile_offsets_indexing));
  return MajorToMinorScheduleImpl(tile_offsets_indexing, iteration_space, ctx);
}

absl::StatusOr<TransposedDotTiledHloSchedule>
TransposedDotTiledHloSchedule::Create(
    const TilingSpecification& tiling_specification) {
  const TilingSpecification::ParameterMapping& parameter_mapping =
      tiling_specification.parameter_mapping();
  CHECK(!parameter_mapping.empty());
  const HloDotInstruction* dot =
      ::xla::DynCast<HloDotInstruction>(parameter_mapping.front().instruction);
  if (dot == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("TransposedDotTiledHloSchedule expects its root to be a "
                     "dot instruction "
                     "but got ",
                     parameter_mapping.front().instruction->ToString()));
  }
  if (absl::c_any_of(absl::MakeSpan(parameter_mapping).subspan(1),
                     [](const auto& param) {
                       return param.instruction->opcode() == HloOpcode::kDot;
                     })) {
    return absl::InvalidArgumentError(
        "TransposedDotTiledHloSchedule is only supported for "
        "TilingSpecifications specifying tiling for a single dot "
        "instruction.");
  }

  int64_t num_lhs_non_contracting_dims =
      dot->operand(0)->shape().dimensions().size() -
      dot->dot_dimension_numbers().lhs_contracting_dimensions().size() -
      dot->dot_dimension_numbers().lhs_batch_dimensions().size();

  int64_t num_rhs_non_contracting_dims =
      dot->operand(1)->shape().dimensions().size() -
      dot->dot_dimension_numbers().rhs_contracting_dimensions().size() -
      dot->dot_dimension_numbers().rhs_batch_dimensions().size();

  constexpr absl::string_view kErrorFormat =
      "TransposedDotTiledHloSchedule is only supported for dot instructions "
      "with a single non-contracting dimension, but got %d non-contracting "
      "dimensions on the %s operand of %s.";

  if (num_lhs_non_contracting_dims != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        kErrorFormat, num_lhs_non_contracting_dims, "lhs", dot->ToString()));
  }

  if (num_rhs_non_contracting_dims != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        kErrorFormat, num_rhs_non_contracting_dims, "rhs", dot->ToString()));
  }

  // The shape of the dot's output is now known to always be of the form
  // [..., m, n]. This is because batch dimensions precede non-contracting
  // dimensions, the lhs non-contracting dimensions precede the rhs
  // non-contracting dimensions, and there is exactly one such dimension on
  // each side.
  //
  // Figure out the parameter index of the m and n dimensions within the op.
  int64_t m_local_parameter_index =
      parameter_mapping.front().num_tiling_parameters - 2;
  int64_t n_local_parameter_index =
      parameter_mapping.front().num_tiling_parameters - 1;

  // Using the local parameter index, we can compute the global parameter index
  // (i.e. the parameter index within the sequence of all tiling parameters).
  TF_ASSIGN_OR_RETURN(int64_t m_dim_id, tiling_specification.ParameterIndex(
                                            dot, m_local_parameter_index));
  TF_ASSIGN_OR_RETURN(int64_t n_dim_id, tiling_specification.ParameterIndex(
                                            dot, n_local_parameter_index));

  return TransposedDotTiledHloSchedule(tiling_specification, m_dim_id,
                                       n_dim_id);
}

absl::StatusOr<IndexingMap> TransposedDotTiledHloSchedule::Schedule(
    const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
    gpu::SymbolicExprContext* ctx) const {
  CHECK_EQ(iteration_space.size(), tiling_specification_.num_parameters());
  TF_RETURN_IF_ERROR(
      ValidateIterationSpace(iteration_space, tile_offsets_indexing));

  DimensionInfo m_dim_info = iteration_space[m_dim_id_];
  DimensionInfo n_dim_info = iteration_space[n_dim_id_];

  if (m_dim_info.dimension_id != m_dim_id_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected dimension at offset %d to have id %d but got %d.", m_dim_id_,
        m_dim_id_, m_dim_info.dimension_id));
  }
  if (n_dim_info.dimension_id != n_dim_id_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected dimension at offset %d to have id %d but got %d.", n_dim_id_,
        n_dim_id_, n_dim_info.dimension_id));
  }

  std::vector<DimensionInfo> transposed_iteration_space(iteration_space.begin(),
                                                        iteration_space.end());
  transposed_iteration_space[m_dim_id_] = n_dim_info;
  transposed_iteration_space[n_dim_id_] = m_dim_info;
  return MajorToMinorScheduleImpl(tile_offsets_indexing,
                                  transposed_iteration_space, ctx);
}

}  // namespace xla
