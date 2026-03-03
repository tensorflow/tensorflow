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

#include "xla/backends/gpu/transforms/splitk_rewriter.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct DotDimensions {
  int64_t b;  // batch dimensions
  int64_t m;  // lhs non-contracting dimensions
  int64_t n;  // rhs non-contracting dimensions
  int64_t k;  // contracting dimensions
  // LHS and RHS element sizes, after going up the chain of elementwise
  // operations. That approximates what will be fused.
  int64_t lhs_element_bits;
  int64_t rhs_element_bits;
  int64_t acc_element_bits;
  int64_t out_element_bits;
  int64_t flops_per_element;
};

int64_t GetAlgoritmFlopsPerElement(const HloDotInstruction* dot) {
  switch (dot->precision_config().algorithm()) {
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      return 12;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      return 20;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return 28;
    default:
      return 2;  // Multiplication and addition.
  }
}

DotDimensions GetDotDimensions(const HloInstruction* instr) {
  const HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();
  DotDimensionNumbers dnums = dot->dot_dimension_numbers();

  auto product_dimensions = [](const Shape& shape,
                               absl::Span<const int64_t> dimensions) {
    return absl::c_accumulate(dimensions, static_cast<int64_t>(1),
                              [&](int64_t product, int64_t dimension) {
                                return product * shape.dimensions(dimension);
                              });
  };

  auto get_side_size = [](const HloInstruction* instr) {
    while (instr->IsElementwise() && instr->operand_count() == 1) {
      instr = instr->operand(0);
    }
    return ShapeUtil::ElementSizeInBits(instr->shape());
  };

  return DotDimensions{
      /*.b = */ product_dimensions(lhs_shape, dnums.lhs_batch_dimensions()),
      /*.m = */
      product_dimensions(
          lhs_shape, GetNonContractingDims(lhs_shape.dimensions().size(),
                                           dnums.lhs_contracting_dimensions(),
                                           dnums.lhs_batch_dimensions())),
      /*.n = */
      product_dimensions(
          rhs_shape, GetNonContractingDims(rhs_shape.dimensions().size(),
                                           dnums.rhs_contracting_dimensions(),
                                           dnums.rhs_batch_dimensions())),
      /*.k = */
      product_dimensions(lhs_shape, dnums.lhs_contracting_dimensions()),
      /* .lhs_el_size_in_bits = */ get_side_size(dot->operand(0)),
      /* .rhs_el_size_in_bits = */ get_side_size(dot->operand(1)),
      /* .acc_el_size_in_bits = */
      ShapeUtil::ByteSizeOfPrimitiveType(GetGemmAccumulatorType(dot)) *
          CHAR_BIT,
      /* .result_el_size_in_bits = */
      ShapeUtil::ElementSizeInBits(dot->shape()),
      /* .flops_per_element = */ GetAlgoritmFlopsPerElement(dot),
  };
}

namespace {

constexpr int64_t kMTileSize = 212;
constexpr int64_t kNTileSize = 212;
constexpr int64_t kKLoopStepBytes = 442;
constexpr double kExtraFlopsPerElement = 4.73313;
constexpr double kFlopsPerByteHbm = 2842.3;
constexpr double kFlopsPerByteCached = 1846.31;
constexpr double kCacheThreshold = 2.19872;
constexpr double kReductionLaunchOverheadFlops = 2.74228e9;
constexpr double kReductionFlopsPerByteHbm = 677.076;
constexpr double kReductionFlopsPerByteCached = 0;
constexpr double kReductionCacheThreshold = 0;

}  // namespace

double EstimateGemmCostAfterSplitK(const DotDimensions& gemm, int64_t splitk,
                                   int num_cores, int64_t l2_cache_size) {
  // Effective dimensions after split
  int64_t effective_k = CeilOfRatio(gemm.k, splitk);
  int64_t effective_batch = gemm.b * splitk;

  // Number of tiles in each dimension
  int64_t m_tiles = CeilOfRatio(gemm.m, kMTileSize);
  int64_t n_tiles = CeilOfRatio(gemm.n, kNTileSize);
  int64_t num_waves = CeilOfRatio(effective_batch * m_tiles * n_tiles,
                                  static_cast<int64_t>(num_cores));

  // Compute per tile.
  const int64_t num_k_iterations = CeilOfRatio(
      effective_k * std::max(gemm.lhs_element_bits, gemm.rhs_element_bits) / 8,
      kKLoopStepBytes);
  const int64_t dot_output_element_size =
      splitk > 1 ? gemm.acc_element_bits : gemm.out_element_bits;
  int64_t flops_per_tile = (gemm.flops_per_element + kExtraFlopsPerElement) *
                           kMTileSize * kNTileSize * num_k_iterations *
                           kKLoopStepBytes * gemm.acc_element_bits / 8;
  // Memory per tile.
  int64_t bytes_read_lhs_per_tile =
      kMTileSize * effective_k * gemm.lhs_element_bits / 8;
  int64_t bytes_read_rhs_per_tile =
      kNTileSize * effective_k * gemm.rhs_element_bits / 8;
  int64_t bytes_write_per_tile =
      kMTileSize * kNTileSize * dot_output_element_size / 8;
  int64_t bytes_per_tile =
      bytes_read_lhs_per_tile + bytes_read_rhs_per_tile + bytes_write_per_tile;
  int64_t bytes_per_wave = bytes_per_tile * num_cores;
  int64_t memory_time_per_tile =
      bytes_per_tile * (bytes_per_wave < kCacheThreshold * l2_cache_size
                            ? kFlopsPerByteCached
                            : kFlopsPerByteHbm);

  int64_t wave_cost = std::max(flops_per_tile, memory_time_per_tile);
  int64_t total_dot_cost = wave_cost * num_waves;

  // Reduction cost.
  if (splitk == 1) {
    return total_dot_cost;
  }

  int64_t reduction_read_bytes =
      effective_batch * gemm.m * gemm.n * gemm.acc_element_bits / 8;
  int64_t reduction_write_bytes =
      gemm.b * gemm.m * gemm.n * gemm.out_element_bits / 8;
  int64_t reduction_bytes = reduction_read_bytes + reduction_write_bytes;
  int64_t reduction_time =
      reduction_bytes *
      (reduction_read_bytes < kReductionCacheThreshold * l2_cache_size
           ? kReductionFlopsPerByteCached
           : kReductionFlopsPerByteHbm);

  return total_dot_cost + kReductionLaunchOverheadFlops + reduction_time;
}

size_t ChooseSplitK(const DotDimensions& dims, int num_cores,
                    int64_t l2_cache_size) {
  std::vector<int64_t> candidates = {1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<double> costs;
  costs.reserve(candidates.size());
  absl::c_transform(candidates, std::back_inserter(costs), [&](int64_t splitk) {
    return EstimateGemmCostAfterSplitK(dims, splitk, num_cores, l2_cache_size);
  });
  return candidates[absl::c_min_element(costs) - costs.begin()];
}

// Pads the given instruction with zeros along the given dimension to the given
// size.
HloInstruction* PadInstruction(HloInstruction* instr, int64_t dimension_idx,
                               int64_t new_dimension_size) {
  HloComputation* computation = instr->parent();
  const PrimitiveType element_type = instr->shape().element_type();
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));
  PaddingConfig padding_config =
      MakeNoPaddingConfig(instr->shape().dimensions().size());
  padding_config.mutable_dimensions(dimension_idx)
      ->set_edge_padding_high(new_dimension_size -
                              instr->shape().dimensions(dimension_idx));
  Shape new_shape = instr->shape();
  new_shape.set_dimensions(dimension_idx, new_dimension_size);
  return computation->AddInstruction(
      HloInstruction::CreatePad(new_shape, instr, zero, padding_config));
}

// Returns the padded K dimension so that it is a multiple of split_k and 16B.
int64_t GetPaddedK(HloInstruction& dot, int64_t split_k) {
  DotDimensions dims = GetDotDimensions(&dot);
  const int64_t alignment_in_bits = 16 * 8;
  int64_t min_element_size_in_bits = std::min(
      {alignment_in_bits, dims.lhs_element_bits, dims.rhs_element_bits});
  return RoundUpTo(dims.k,
                   split_k * alignment_in_bits / min_element_size_in_bits);
}

// The contracting dimension index becomes new batch (split) dimension, and all
// dimensions after it are shifted by 1.
HloInstruction* SplitKOperand(HloInstruction* operand,
                              int64_t contracting_dimension_idx,
                              int64_t split_k, int64_t padded_k) {
  // if the K dimension is not divisible by split_k, we need to pad it.
  const int64_t src_k = operand->shape().dimensions(contracting_dimension_idx);
  if (padded_k != src_k) {
    operand = PadInstruction(operand, contracting_dimension_idx, padded_k);
  }
  const Shape& old_shape = operand->shape();

  // Copy the existing shape to keep all the non-dimension/non-layout fields of
  // the shape (element size in bits etc).
  Shape new_shape = old_shape;
  new_shape.clear_dimensions();
  for (int64_t i = 0; i < old_shape.dimensions().size(); ++i) {
    const int64_t old_dim = old_shape.dimensions(i);
    if (i == contracting_dimension_idx) {
      new_shape.add_dimensions(split_k);
      new_shape.add_dimensions(old_dim / split_k);
    } else {
      new_shape.add_dimensions(old_dim);
    }
  }

  // Update the physical layout so the the physical layout is preserved (i.e.
  // the splitK dimension goes right before the contracting dimension, and all
  // remaining dimensions are kept).
  if (new_shape.layout().minor_to_major().size() > 0) {
    new_shape.mutable_layout()->clear_minor_to_major();
    for (int64_t dim_idx : old_shape.layout().minor_to_major()) {
      if (dim_idx >= contracting_dimension_idx) {
        new_shape.mutable_layout()->add_minor_to_major(dim_idx + 1);
      }
      if (dim_idx <= contracting_dimension_idx) {
        new_shape.mutable_layout()->add_minor_to_major(dim_idx);
      }
    }
  }

  // Now reshape into the "new_shape".
  return operand->parent()->AddInstruction(
      HloInstruction::CreateReshape(new_shape, operand));
}

// Sums/reduces the tensor along the given dimension.
absl::StatusOr<HloInstruction*> ReduceDimension(HloInstruction* instr,
                                                int64_t dimension_idx) {
  HloComputation* computation = instr->parent();
  const PrimitiveType element_type = instr->shape().element_type();
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));
  return MakeReduceHlo(instr, zero, {dimension_idx}, HloOpcode::kAdd,
                       &instr->metadata());
}

absl::StatusOr<HloInstruction*> SplitKDimensionOfDot(HloDotInstruction* src_dot,
                                                     size_t split_k) {
  PrimitiveType output_type = src_dot->shape().element_type();
  PrimitiveType accumulator_type = GetGemmAccumulatorType(src_dot);

  // "split_k" is the number on chunks the K dimension is split into.
  const int64_t lhs_k_idx =
      src_dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
  const int64_t rhs_k_idx =
      src_dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
  const int64_t padded_k = GetPaddedK(*src_dot, split_k);
  // The operands' K dimension are split into [split_k, K/split_k] (shifting
  // right all the dimensions after it).
  HloInstruction* lhs =
      SplitKOperand(src_dot->mutable_operand(0), lhs_k_idx, split_k, padded_k);
  HloInstruction* rhs =
      SplitKOperand(src_dot->mutable_operand(1), rhs_k_idx, split_k, padded_k);

  // Update the dot's dimension numbers accordingly (shifting right all the
  // dimensions starting from the K dimension and inserting new batch dims).
  DotDimensionNumbers new_dnums = src_dot->dot_dimension_numbers();
  auto shift_dimension = [](tsl::protobuf::RepeatedField<int64_t>* dims,
                            int64_t idx) {
    absl::c_for_each(*dims, [idx](int64_t& dim) {
      if (dim >= idx) {
        dim++;
      }
    });
  };
  shift_dimension(new_dnums.mutable_lhs_contracting_dimensions(), lhs_k_idx);
  shift_dimension(new_dnums.mutable_rhs_contracting_dimensions(), rhs_k_idx);
  shift_dimension(new_dnums.mutable_lhs_batch_dimensions(), lhs_k_idx);
  shift_dimension(new_dnums.mutable_rhs_batch_dimensions(), rhs_k_idx);
  new_dnums.mutable_lhs_batch_dimensions()->Add(lhs_k_idx);
  new_dnums.mutable_rhs_batch_dimensions()->Add(rhs_k_idx);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_dot,
      MakeDotHlo(lhs, rhs, new_dnums, src_dot->precision_config(),
                 accumulator_type, &src_dot->metadata()));

  // Reduce along the new batch dimension.
  const int64_t splitk_dim_idx = new_dnums.lhs_batch_dimensions_size() - 1;
  TF_ASSIGN_OR_RETURN(HloInstruction * splitk_root,
                      ReduceDimension(new_dot, splitk_dim_idx));
  *splitk_root->mutable_shape()->mutable_layout() = src_dot->shape().layout();
  if (output_type != accumulator_type) {
    splitk_root = MakeConvertToHlo(splitk_root, output_type);
  }
  return splitk_root;
}

class SplitkRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit SplitkRewriterVisitor(se::DeviceDescription device_description)
      : device_description_(device_description) {}

 private:
  absl::Status HandleDot(HloInstruction* instr) override {
    HloDotInstruction* dot = DynCast<HloDotInstruction>(instr);
    if (dot->dot_dimension_numbers().lhs_contracting_dimensions_size() != 1 ||
        dot->dot_dimension_numbers().rhs_contracting_dimensions_size() != 1) {
      // In theory we could support it, but it's rare and adds complexity.
      return absl::OkStatus();
    }
    if (absl::c_any_of(dot->operands(), [](const HloInstruction* operand) {
          return operand->shape().element_type() == S32;
        })) {
      // Neither cuBLAS nor Triton support s32, so we don't benefit from
      // splitting K.
      return absl::OkStatus();
    }
    const size_t split_k =
        ChooseSplitK(GetDotDimensions(dot), device_description_.core_count(),
                     device_description_.l2_cache_size());
    if (split_k == 1) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                        SplitKDimensionOfDot(dot, split_k));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_dot));
    return absl::OkStatus();
  }

  se::DeviceDescription device_description_;
};

}  // namespace

absl::StatusOr<bool> SplitkRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->config()
           .debug_options()
           .xla_gpu_experimental_enable_split_k_rewrite()) {
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    SplitkRewriterVisitor visitor(device_description_);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
