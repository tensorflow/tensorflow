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

#include "xla/service/gpu/transforms/splitk_rewriter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
  int64_t lhs_el_size_in_bits;
  int64_t rhs_el_size_in_bits;
  int64_t result_el_size_in_bits;
};

DotDimensions GetDotDimensions(const HloInstruction* dot) {
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
      /* .result_el_size_in_bits = */
      ShapeUtil::ElementSizeInBits(dot->shape()),
  };
}

size_t ChooseSplitK(const DotDimensions& dims, int num_cores) {
  // Compute the computational intensity in FLOPs per 256 bits of memory I/O
  // (instead of FLOPs per byte to avoid the need for floating point).
  size_t computational_intensity =
      256 * dims.m * dims.n * dims.k /
      (dims.m * dims.k * dims.lhs_el_size_in_bits +
       dims.n * dims.k * dims.rhs_el_size_in_bits +
       dims.m * dims.n * dims.result_el_size_in_bits);

  // The constants below were tuned the following way:
  // 1. Generated random GEMM kernels.
  //    * M, N, K and B dimensions are exponentially distributed between 1 and
  //    200000
  //    * M, N and K are rounded up to the multiple of 16.
  //    * B is set to 1 in the half of samples.
  //    * Use combinations (that make sense)of s32, s8, s4, fp8, bf16, f32 and
  //    f16 as op and result types.
  // 2. Every of these kernels were run on H100 with exhaustive tiling search
  //    enabled.
  // 3. The best values of the constants were picked using brute force search.
  // 4. Two functions were used as a loss function, converging to the same
  //    result (performance of the best splitK was taken as 1.0, and
  //    performance of other splitK value as a fraction of it):
  //    * Geomean.
  //    * Mean square loss.
  constexpr int64_t kIntensityThreshold = 240;
  // The minimum K dimension size for the dot after splitting.
  constexpr int64_t kMemoryBoundMinK = 768;
  constexpr int64_t kComputeBoundMinK = 1220;
  constexpr size_t kMaxSplitK = 128;
  // The target number tiles of num_cores×1.55 was tuned to be the best, but
  // let's keep it more sane-looking 1.5.
  const int64_t kTargetNumTiles = num_cores + num_cores / 2;
  const int64_t kMTileSize = 64;
  const int64_t kNTileSize = 128;

  VLOG(3) << "ChooseSplitK(), b=" << dims.b << " m=" << dims.m
          << " n=" << dims.n << " k=" << dims.k
          << " lhs_sz=" << dims.lhs_el_size_in_bits
          << " rhs_size=" << dims.rhs_el_size_in_bits
          << " result_size=" << dims.result_el_size_in_bits
          << " intensity=" << computational_intensity;

  if (computational_intensity < kIntensityThreshold) {
    // Assume memory throughput bound, choose as high splitK as possible, but
    // keep the resulting K >= kMemoryBoundMinK.
    size_t splitk = std::min(
        kMaxSplitK, size_t{1} << Log2Ceiling(static_cast<uint64_t>(
                        std::max(int64_t{1}, dims.k / kMemoryBoundMinK))));
    VLOG(3) << "Memory throughput bound, splitK=" << splitk;
    return splitk;
  }

  // Assume compute bound, try to fill target number of tiles.
  const int64_t m_tiles = CeilOfRatio(dims.m, kMTileSize);
  const int64_t n_tiles = CeilOfRatio(dims.n, kNTileSize);
  const int64_t num_tiles = dims.b * m_tiles * n_tiles;
  const uint64_t max_splitk = 1 << Log2Floor(static_cast<uint64_t>(std::max(
                                  int64_t{1}, dims.k / kComputeBoundMinK)));
  const uint64_t desired_splitk = CeilOfRatio(kTargetNumTiles, num_tiles);
  const size_t splitk = 1 << Log2Ceiling(std::min(max_splitk, desired_splitk));

  VLOG(3) << "Compute throughput bound, m_tiles=" << m_tiles
          << " n_tiles=" << n_tiles << " num_tiles=" << num_tiles
          << " max_splitk=" << max_splitk
          << " desired_splitk=" << desired_splitk << " splitk=" << splitk;
  return splitk;
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
      ->set_edge_padding_low(new_dimension_size -
                             instr->shape().dimensions(dimension_idx));
  Shape new_shape = instr->shape();
  new_shape.set_dimensions(dimension_idx, new_dimension_size);
  return computation->AddInstruction(
      HloInstruction::CreatePad(new_shape, instr, zero, padding_config));
}

// The contracting dimension index becomes new batch (split) dimension, and all
// dimensions after it are shifted by 1.
HloInstruction* SplitKOperand(HloInstruction* operand,
                              int64_t contracting_dimension_idx,
                              int64_t split_k) {
  // if the K dimension is not divisible by split_k, we need to pad it.
  const int64_t src_k = operand->shape().dimensions(contracting_dimension_idx);
  const bool needs_padding = src_k % split_k != 0;
  if (needs_padding) {
    const int64_t padded_k = RoundUpTo(src_k, split_k);
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
  // The operands' K dimension are split into [split_k, K/split_k] (shifting
  // right all the dimensions after it).
  HloInstruction* lhs =
      SplitKOperand(src_dot->mutable_operand(0), lhs_k_idx, split_k);
  HloInstruction* rhs =
      SplitKOperand(src_dot->mutable_operand(1), rhs_k_idx, split_k);

  // Update the dot's dimension numbers accordingly (shifting right all the
  // dimensions starting from the K dimension and inserting new batch dims).
  DotDimensionNumbers new_dnums = src_dot->dot_dimension_numbers();
  auto shift_dimension = [](tsl::protobuf::RepeatedField<int64_t>* dims,
                            int64_t idx) {
    absl::c_for_each(*dims, [idx](int64_t& dim) {
      if (dim >= idx) dim++;
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
                 accumulator_type, {}, {}, &src_dot->metadata()));

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
    if (dot->sparse_operands()) return absl::OkStatus();
    if (dot->dot_dimension_numbers().lhs_contracting_dimensions_size() != 1 ||
        dot->dot_dimension_numbers().rhs_contracting_dimensions_size() != 1) {
      // In theory we could support it, but it's rare and adds complexity.
      return absl::OkStatus();
    }
    const size_t split_k =
        ChooseSplitK(GetDotDimensions(dot), device_description_.core_count());
    if (split_k == 1) return absl::OkStatus();
    TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                        SplitKDimensionOfDot(dot, split_k));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_dot));
    return absl::OkStatus();
  }

  se::DeviceDescription device_description_;
};

}  // namespace

absl::StatusOr<bool> SplitkRewriter::Run(
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
