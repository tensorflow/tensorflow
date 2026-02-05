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

#include "xla/service/gpu/model/experimental/tiling_space.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {
namespace {

std::string HloPtrToString(const HloInstruction* hlo) {
  return hlo == nullptr ? "nullptr" : hlo->ToString();
}

}  // namespace

void TilingSpace::AppendDimension(const HloInstruction* hlo,
                                  int64_t dim_position, int64_t dim_size,
                                  DimensionSemantics dim_type) {
  dimensions_.push_back(DimensionInfo{static_cast<ID>(dimensions_.size()),
                                      dim_size, dim_type, hlo, dim_position});
  hlo_to_dimension_[std::make_pair(hlo, dim_position)] = &dimensions_.back();
}

void TilingSpace::AppendRTVar(const HloInstruction* hlo, int64_t operand_id,
                              const HloInstruction* rt_var,
                              int64_t upper_bound) {
  rt_vars_.push_back(RTVarInfo{
      static_cast<ID>(rt_vars_.size()),
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
    case HloOpcode::kDynamicSlice:
      ProcessDynamicSlice(hlo);
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

std::string TilingSpace::ToString() const {
  std::stringstream ss;
  ss << "Dimensions:\n";
  for (const auto& dim : dimensions_) {
    ss << dim.id << " type: "
       << (dim.type == DimensionSemantics::kParallel ? "parallel"
                                                     : "sequential")
       << " size: " << dim.dimension_size << " dim ID:" << dim.dim_position
       << " hlo: " << HloPtrToString(dim.hlo) << "\n";
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
  if (!constraints_.IsAlwaysSatisfied()) {
    ss << "Constraints:\n" << constraints_.ToString() << "\n";
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

const TilingSpace::RTVarInfo& TilingSpace::GetRTVarInfo(
    const HloInstruction& hlo, int64_t operand_id) const {
  auto it = hlo_to_rt_var_.find(std::make_pair(&hlo, operand_id));
  CHECK(it != hlo_to_rt_var_.end())
      << "Runtime variable not found for " << hlo.ToString() << " operand "
      << operand_id;
  return *it->second;
}

std::unique_ptr<TilingSpace> TilingSpace::Create(const HloFusionAdaptor& fusion,
                                                 mlir::MLIRContext* ctx) {
  auto tiling_space = std::make_unique<TilingSpace>();
  tiling_space->mlir_context_ = ctx;
  auto roots = fusion.GetRoots();
  for (const HloInstructionAdaptor& root : roots) {
    const Shape& root_shape = root.shape();
    if (!root.shape().IsArray() && root.opcode() != HloOpcode::kReduce) {
      LOG(FATAL) << "Unsupported root shape " << root_shape.ToString()
                 << " for root " << root.instruction().ToString();
    }
    // TODO(goncharov): why do we only care about the first shape of a tuple?
    absl::Span<const int64_t> dims =
        GetFirstShape(&root.instruction()).dimensions();
    llvm::SmallVector<DimTile> dim_tiles;
    dim_tiles.reserve(dims.size());
    for (auto [index, dim] : llvm::enumerate(dims)) {
      tiling_space->AppendDimension(&root.instruction(), index, dim,
                                    DimensionSemantics::kParallel);
      dim_tiles.push_back(GetDefaultDimTile(index, dim, ctx));
    }
    SymbolicTile tile{*tiling_space, std::move(dim_tiles)};
    if (root_shape.IsTuple()) {
      for (int64_t i = 0, e = root_shape.tuple_shapes().size(); i < e; ++i) {
        tiling_space->tiled_roots_.push_back(tile);
      }
      continue;
    }
    tiling_space->tiled_roots_.push_back(std::move(tile));
  }
  // Iterator in reversed post-order (use-before-def).
  auto post_order = fusion.MakeInstructionPostOrder();
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    tiling_space->ProcessInstruction(it->instruction());
  }
  return tiling_space;
}

}  // namespace xla::gpu::experimental
