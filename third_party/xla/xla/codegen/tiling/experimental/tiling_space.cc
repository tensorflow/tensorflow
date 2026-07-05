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

#include "xla/codegen/tiling/experimental/tiling_space.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space_utils.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {
namespace {

std::string HloPtrToString(const HloInstruction* hlo) {
  return hlo == nullptr ? "nullptr" : hlo->ToString();
}

}  // namespace

std::string TilingSpace::DimensionInfo::ToString() const {
  std::stringstream ss;
  ss << id << " type: "
     << (type == DimensionSemantics::kParallel ? "parallel" : "sequential")
     << " size: " << dimension_size;
  if (tile_size.has_value()) {
    ss << " tile size: " << *tile_size;
  }
  ss << " dim ID:" << dim_position << " hlo: " << HloPtrToString(hlo);
  return ss.str();
}

void TilingSpace::AppendDimension(const HloInstruction* hlo,
                                  int64_t dim_position, int64_t dim_size,
                                  DimensionSemantics dim_type) {
  dimensions_.push_back(DimensionInfo{TiledDimId(dimensions_.size()), dim_size,
                                      dim_type, hlo, dim_position});
  hlo_to_dimension_[std::make_pair(hlo, dim_position)] = &dimensions_.back();
}

void TilingSpace::AppendRTVar(const HloInstruction* hlo, int64_t operand_id,
                              const HloInstruction* rt_var,
                              int64_t upper_bound) {
  rt_vars_.push_back(RTVarInfo{
      static_cast<int64_t>(rt_vars_.size()),
      Interval{0, upper_bound},
      rt_var,
  });
  hlo_to_rt_var_[std::make_pair(hlo, operand_id)] = &rt_vars_.back();
}

void TilingSpace::ProcessInstruction(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kDot:
    case HloOpcode::kScaledDot:
      ProcessDotLike(hlo);
      break;
    case HloOpcode::kReduce:
      ProcessReduce(hlo);
      break;
    case HloOpcode::kScan:
      ProcessScan(hlo);
      break;
    case HloOpcode::kDynamicSlice:
      ProcessDynamicSlice(hlo);
      break;
    case HloOpcode::kGetTupleElement:
      ProcessGetTupleElement(hlo);
      break;
    default:
      // TODO(goncharov): should have a explicit list of supported instructions?
      break;
  }
}

// Add dot contraction dimensions in the order of contracting dimensions.
void TilingSpace::ProcessDotLike(const HloInstruction& hlo) {
  const Shape& lhs_shape = hlo.operand(0)->shape();
  const DotDimensionNumbers& dim_numbers = hlo.dot_dimension_numbers();
  int64_t output_rank = hlo.shape().dimensions().size();
  for (auto [index, contracting_dim_id] :
       llvm::enumerate(dim_numbers.lhs_contracting_dimensions())) {
    AppendDimension(&hlo, output_rank + index,
                    lhs_shape.dimensions(contracting_dim_id),
                    DimensionSemantics::kSequential);
  }
}

// Add reduction dimensions.
void TilingSpace::ProcessReduce(const HloInstruction& hlo) {
  auto reduce = Cast<HloReduceInstruction>(&hlo);
  const Shape& input_shape = reduce->operand(0)->shape();
  int64_t output_rank = GetFirstShape(reduce).dimensions().size();
  for (auto [index, reduction_dim_id] : llvm::enumerate(reduce->dimensions())) {
    AppendDimension(&hlo, output_rank + index,
                    input_shape.dimensions(reduction_dim_id),
                    DimensionSemantics::kSequential);
  }
}

// Ensure scan dimensions are not tiled across CTAs.
void TilingSpace::ProcessScan(const HloInstruction& hlo) {
  auto scan = Cast<HloScanInstruction>(&hlo);
  int64_t scan_dim_idx = scan->scan_dimension();

  auto it = hlo_to_dimension_.find(std::make_pair(&hlo, scan_dim_idx));
  if (it == hlo_to_dimension_.end()) {
    // Without indexing maps, we cannot express constraints for intermediate
    // scan operations in TilingSpace.
    return;
  }

  int64_t global_dim_id = it->second->id.value();
  SymbolicExpr scan_dim_tile_size = CreateDimExpr(global_dim_id, mlir_context_);

  const Shape& shape = GetFirstShape(&hlo);
  SymbolicExpr scan_dim_bound =
      CreateSymbolicConstant(shape.dimensions(scan_dim_idx), mlir_context_);

  // Require that the tile size equals the dimension bound.
  ConstraintExpression::Constraint eq_constraint{
      scan_dim_bound - scan_dim_tile_size, Interval{0, 0}};
  constraint_ = constraint_ && eq_constraint;
}

// Add offsets of dynamic slice.
void TilingSpace::ProcessDynamicSlice(const HloInstruction& hlo) {
  auto ds = Cast<HloDynamicSliceInstruction>(&hlo);
  const int64_t first_index_num = ds->first_index_operand_number();
  CHECK(ds->operand(first_index_num)->shape().dimensions().empty())
      << "b/118437727: Old form, not supported.";

  const Shape& input_shape = ds->operand(0)->shape();
  for (auto [dim, slice_size] : llvm::enumerate(ds->dynamic_slice_sizes())) {
    AppendRTVar(&hlo, dim + first_index_num, ds->operand(dim + first_index_num),
                input_shape.dimensions(dim) - slice_size);
  }
}

const Shape& GetFirstShape(const HloInstruction* instr, int64_t index) {
  return instr->shape().IsTuple()
             ? ShapeUtil::GetSubshape(instr->shape(), {index})
             : instr->shape();
}

// Propagate dimensions from get-tuple-element to its operand.
void TilingSpace::ProcessGetTupleElement(const HloInstruction& hlo) {
  for (int64_t i = 0; i < hlo.shape().dimensions().size(); ++i) {
    auto it = hlo_to_dimension_.find(std::make_pair(&hlo, i));
    if (it != hlo_to_dimension_.end()) {
      hlo_to_dimension_[std::make_pair(hlo.operand(0), i)] = it->second;
    }
  }
}

std::string TilingSpace::ToString() const {
  std::stringstream ss;
  ss << "Dimensions:\n";
  for (const auto& dim : dimensions_) {
    ss << dim.ToString() << "\n";
  }
  if (!rt_vars_.empty()) {
    ss << "Runtime variables:\n";
    for (const auto& rt_var : rt_vars_) {
      ss << rt_var.id << " bounds: " << rt_var.bounds
         << " hlo: " << HloPtrToString(rt_var.hlo) << "\n";
    }
  }
  ss << "Root tiles:\n";
  for (const auto& [index, tile] : llvm::enumerate(tiled_roots_)) {
    ss << index << " root tile: " << tile.ToString(/*print_variables=*/false)
       << "\n";
  }
  if (!constraint_.IsAlwaysSatisfied()) {
    ss << "Constraints:\n" << constraint_.ToString() << "\n";
  }
  if (!divisibility_constraints_.empty()) {
    ss << "Divisibility constraints:\n";
    for (const auto& c : divisibility_constraints_) {
      ss << c.expr.ToString(dimensions_.size()) << " is multiple of "
         << c.tile_size.ToString(dimensions_.size()) << "\n";
    }
  }
  return ss.str();
}

const TilingSpace::DimensionInfo& TilingSpace::GetDimensionInfo(
    const HloInstruction& hlo, int64_t dim_position) const {
  auto it = hlo_to_dimension_.find(std::make_pair(&hlo, dim_position));
  CHECK(it != hlo_to_dimension_.end())
      << "Dimension not found for " << hlo.ToString() << " dimension "
      << dim_position;
  return *it->second;
}

std::optional<const TilingSpace::RTVarInfo*> TilingSpace::GetRTVarInfo(
    const HloInstruction& hlo, int64_t operand_id) const {
  auto it = hlo_to_rt_var_.find(std::make_pair(&hlo, operand_id));
  if (it == hlo_to_rt_var_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status TilingSpace::AssignTileSizes(
    absl::Span<const int64_t> tile_sizes) {
  if (!is_symbolic_) {
    return absl::InternalError(
        "Tile sizes have already been assigned to this tiling space.");
  }
  CHECK_EQ(tile_sizes.size(), dimensions_.size());
  is_symbolic_ = false;

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replacement_map;
  for (const auto& [index, dim] : llvm::enumerate(dimensions_)) {
    dim.tile_size = tile_sizes[index];
    replacement_map[CreateSymbolExpr(dim.id.value(), dimensions_.size(),
                                     mlir_context_)] =
        CreateSymbolicConstant(tile_sizes[index], mlir_context_);

    // If the tile size is greater than or equal to the dimension size, then
    // the dimension is trivial and can be replaced with 0.
    if (dim.dimension_size <= tile_sizes[index]) {
      replacement_map[CreateDimExpr(dim.id.value(), mlir_context_)] =
          CreateSymbolicConstant(0, mlir_context_);
    }
  }

  if (!constraint_.IsSatisfiedBy(tile_sizes)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tile sizes %s do not satisfy constraint %s",
        absl::StrJoin(tile_sizes, ","), constraint_.ToString()));
  }

  for (const auto& c : divisibility_constraints_) {
    SymbolicExpr replaced_size = c.tile_size.Replace(replacement_map);
    if (replaced_size.GetType() != SymbolicExprType::kConstant) {
      return absl::InternalError(absl::StrFormat(
          "Expected tile size symbol to evaluate to a constant after "
          "replacement, but got %s.",
          replaced_size.ToString(dimensions_.size())));
    }
    int64_t concrete_tile_size = replaced_size.GetValue();
    SymbolicExpr replaced_expr = c.expr.Replace(replacement_map);
    if (!replaced_expr.IsMultipleOf(concrete_tile_size)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Divisibility constraint not satisfied: %s is not a clean multiple "
          "of %d.",
          replaced_expr.ToString(dimensions_.size()), concrete_tile_size));
    }
  }

  InitSimplificationIndexing();

  for (auto& tiled_root : tiled_roots_) {
    tiled_root.Replace(replacement_map);
    tiled_root.Simplify();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<TilingSpace>> TilingSpace::Create(
    const HloFusionAdaptor& fusion, mlir::MLIRContext* ctx) {
  RegisterSymbolicExprStorage(ctx);
  auto tiling_space = std::make_unique<TilingSpace>();
  tiling_space->mlir_context_ = ctx;
  auto roots = fusion.GetRoots();
  TF_RET_CHECK(!roots.empty()) << "Fusion has no roots";

  // TODO: b/502910372 - Support multi-output fusions. The option name is
  // misleading as it is not GPU specific.
  if (roots.size() > 1 &&
      !roots.back()
           .instruction()
           .GetModule()
           ->config()
           .debug_options()
           .xla_gpu_unsupported_enable_triton_multi_output_fusion()) {
    return absl::UnimplementedError(
        "TilingSpace does not support fusions with multiple roots");
  }

  // First pass: Append all dimensions. This is necessary because symbols
  // are created using the total number of dimensions, which needs to be known
  // before any symbols are generated.
  for (const HloInstructionAdaptor& root : roots) {
    const Shape& root_shape = root.shape();
    if (!root.shape().IsArray() && root.opcode() != HloOpcode::kReduce &&
        root.opcode() != HloOpcode::kScan) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported root shape ", root_shape.ToString(),
                       " for root ", root.instruction().ToString()));
    }
    // TODO(goncharov): why do we only care about the first shape of a tuple?
    absl::Span<const int64_t> dims =
        GetFirstShape(&root.instruction()).dimensions();
    for (auto [index, dim] : llvm::enumerate(dims)) {
      // Dimensions must be appended first so that the total count is known
      // when creating Symbols.
      tiling_space->AppendDimension(&root.instruction(), index, dim,
                                    DimensionSemantics::kParallel);
    }
  }

  // Iterator in reversed post-order (use-before-def).
  auto post_order = fusion.MakeInstructionPostOrder();
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    tiling_space->ProcessInstruction(it->instruction());
  }

  // Second pass: Create the root tiles now that
  // `tiling_space->num_dimensions()` is known.
  for (const HloInstructionAdaptor& root : roots) {
    const Shape& root_shape = root.shape();
    absl::Span<const int64_t> dims =
        GetFirstShape(&root.instruction()).dimensions();
    llvm::SmallVector<DimTile> dim_tiles;
    dim_tiles.reserve(dims.size());
    for (auto [index, dim] : llvm::enumerate(dims)) {
      int64_t global_dim_id =
          tiling_space->GetDimensionInfo(root.instruction(), index).id.value();
      dim_tiles.push_back(GetDefaultDimTile(
          index,
          CreateSymbolExpr(global_dim_id, tiling_space->num_dimensions(), ctx),
          dim));
    }
    Tile tile{*tiling_space, std::move(dim_tiles)};
    if (root_shape.IsTuple()) {
      for (int64_t i = 0, e = root_shape.tuple_shapes().size(); i < e; ++i) {
        tiling_space->tiled_roots_.push_back(tile);
      }
      continue;
    }
    tiling_space->tiled_roots_.push_back(std::move(tile));
  }

  return tiling_space;
}

int64_t TilingSpace::num_parallel_dimensions() const {
  return absl::c_count_if(dimensions_, [](const DimensionInfo& dim) {
    return dim.type == DimensionSemantics::kParallel;
  });
}

void TilingSpace::InitSimplificationIndexing() {
  CHECK(!is_symbolic_) << "Tile sizes must be assigned before initializing "
                          "cached indexing map variables.";

  dim_vars_indexing_.clear();
  dim_vars_indexing_.reserve(dimensions_.size());
  for (const auto& dim_info : dimensions_) {
    CHECK_GT(dim_info.tile_size.value(), 0);
    int64_t upper_bound =
        llvm::divideCeil(dim_info.dimension_size, dim_info.tile_size.value());
    dim_vars_indexing_.push_back(IndexingMap::Variable{0, upper_bound - 1});
  }

  range_vars_indexing_.assign(dimensions_.size(), IndexingMap::Variable{0, 0});

  rt_vars_indexing_.clear();
  rt_vars_indexing_.reserve(rt_vars_.size());
  for (const auto& rt_var : rt_vars_) {
    rt_vars_indexing_.push_back(IndexingMap::Variable{rt_var.bounds});
  }
}

SymbolicExpr TilingSpace::SimplifyExpression(const SymbolicExpr& expr) const {
  if (is_symbolic_) {
    return expr.Canonicalize();
  }

  SymbolicMap map = SymbolicMap::Get(mlir_context(), dimensions_.size(),
                                     rt_vars_.size(), {expr});

  IndexingMap indexing_map(map, dim_vars_indexing_, range_vars_indexing_,
                           rt_vars_indexing_);
  indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);
  return indexing_map.GetSymbolicMap().GetResults()[0];
}

absl::StatusOr<std::vector<llvm::SmallVector<int64_t, 4>>>
TilingSpace::GetValidTilings() {
  // TODO: b/511080616 - returned tilings should be valid. Right now we return
  // all possible tilings and rely on the downstream to check the validity.
  llvm::SmallVector<int64_t, 4> input_space;

  // Sequential reduce dimensions are not tiled yet. To work around the
  // limitation of `GetFlatTilingsForInputSpace`, we set the tile size to 1 here
  // and later replace with the actual dimension size.
  for (const auto& dim : dimensions_) {
    if (dim.type == DimensionSemantics::kSequential &&
        dim.hlo->opcode() == HloOpcode::kReduce) {
      input_space.push_back(1);
    } else {
      input_space.push_back(dim.dimension_size);
    }
  }

  ASSIGN_OR_RETURN(auto flat_tilings, GetFlatTilingsForInputSpace(input_space));

  for (auto& flat_tiling : flat_tilings) {
    for (const auto& [idx, dim] : llvm::enumerate(dimensions_)) {
      if (dim.type == DimensionSemantics::kSequential &&
          dim.hlo->opcode() == HloOpcode::kReduce) {
        flat_tiling[idx] = dim.dimension_size;
      }
    }
  }

  std::vector<llvm::SmallVector<int64_t, 4>> valid_tilings;
  valid_tilings.reserve(flat_tilings.size());
  for (const auto& flat_tiling : flat_tilings) {
    valid_tilings.push_back({flat_tiling.begin(), flat_tiling.end()});
  }
  return valid_tilings;
}

}  // namespace xla::gpu::experimental
