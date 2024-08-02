/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/spmd/convolution_handler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/literal_util.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/service/shape_inference.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {

namespace {

// Partition convolution with batch group count.
absl::StatusOr<HloInstruction*> PartitionConvolutionWithBatchGroupCount(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    int64_t num_partitions, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);
  if (original_hlo->batch_group_count() == 1 ||
      original_hlo->batch_group_count() % num_partitions != 0) {
    return nullptr;
  }

  const auto& dnums = original_hlo->convolution_dimension_numbers();
  // Only supports batch_group_size equals input_batch_size case.
  const int64_t input_batch_size =
      lhs.base_shape().dimensions(dnums.input_batch_dimension());
  const int64_t kernel_output_feature_size =
      rhs.base_shape().dimensions(dnums.kernel_output_feature_dimension());
  if (input_batch_size != kernel_output_feature_size ||
      original_hlo->batch_group_count() != input_batch_size) {
    return nullptr;
  }

  // Map RHS indices to LHS indices.
  std::vector<int64_t> rhs_to_lhs_indices(output_base_shape.rank());
  rhs_to_lhs_indices[dnums.kernel_output_feature_dimension()] =
      dnums.input_batch_dimension();
  rhs_to_lhs_indices[dnums.kernel_input_feature_dimension()] =
      dnums.input_feature_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    rhs_to_lhs_indices[dnums.kernel_spatial_dimensions(i)] =
        dnums.input_spatial_dimensions(i);
  }

  // Map LHS indices to RHS indices.
  std::vector<int64_t> lhs_to_rhs_indices(output_base_shape.rank());
  for (int64_t i = 0; i < rhs_to_lhs_indices.size(); ++i) {
    lhs_to_rhs_indices[rhs_to_lhs_indices[i]] = i;
  }

  // Map LHS indices to output indices.
  std::vector<int64_t> lhs_to_output_indices(lhs.base_shape().rank(), -1);
  lhs_to_output_indices[dnums.input_batch_dimension()] =
      dnums.output_feature_dimension();
  lhs_to_output_indices[dnums.input_feature_dimension()] =
      dnums.output_batch_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    lhs_to_output_indices[dnums.input_spatial_dimensions(i)] =
        dnums.output_spatial_dimensions(i);
  }

  // Align LHS or RHS to other operand if input batch dim or kernel output
  // feature dim is partitioned.
  auto aligned_rhs_sharding =
      hlo_sharding_util::TransposeSharding(lhs.sharding(), rhs_to_lhs_indices);
  auto aligned_lhs_sharding =
      hlo_sharding_util::TransposeSharding(rhs.sharding(), lhs_to_rhs_indices);

  bool lhs_batch_dim_is_partitioned =
      (ShardCountAtDim(lhs.sharding(), dnums.input_batch_dimension()) ==
       num_partitions);
  bool rhs_output_feature_dim_is_partitioned =
      (ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) ==
       num_partitions);
  if (!lhs_batch_dim_is_partitioned && !rhs_output_feature_dim_is_partitioned) {
    return nullptr;
  }
  // Reshard LHS or RHS to partition at batch dimension or output feature
  // dimension as the other operand.
  if (lhs_batch_dim_is_partitioned) {
    rhs = rhs.Reshard(aligned_rhs_sharding);
  } else {
    lhs = lhs.Reshard(aligned_lhs_sharding);
  }
  // Align output sharding after LHS and RHS sharding are consistent.
  auto aligned_output_sharding = hlo_sharding_util::TransposeSharding(
      lhs.sharding(), lhs_to_output_indices);

  // Create partitioned convolution.
  TF_ASSIGN_OR_RETURN(
      auto sharded_conv,
      create_sharded_conv(lhs.hlo(), rhs.hlo(), b, conv_window));
  sharded_conv->set_sharding(aligned_output_sharding);
  return PartitionedHlo(sharded_conv, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

// Partition convolution with feature group count.
absl::StatusOr<HloInstruction*> PartitionConvolutionWithFeatureGroupCount(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    int64_t num_partitions, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);
  if (original_hlo->feature_group_count() == 1 ||
      original_hlo->feature_group_count() % num_partitions != 0) {
    return nullptr;
  }

  const auto& dnums = original_hlo->convolution_dimension_numbers();
  const int64_t input_feature_size =
      lhs.base_shape().dimensions(dnums.input_feature_dimension());
  const int64_t kernel_output_feature_size =
      rhs.base_shape().dimensions(dnums.kernel_output_feature_dimension());
  if (kernel_output_feature_size % original_hlo->feature_group_count() != 0 ||
      input_feature_size % original_hlo->feature_group_count() != 0) {
    return nullptr;
  }

  // Align RHS indices to LHS.
  std::vector<int64_t> rhs_to_lhs_indices(output_base_shape.rank());
  rhs_to_lhs_indices[dnums.kernel_output_feature_dimension()] =
      dnums.input_feature_dimension();
  rhs_to_lhs_indices[dnums.kernel_input_feature_dimension()] =
      dnums.input_batch_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    rhs_to_lhs_indices[dnums.kernel_spatial_dimensions(i)] =
        dnums.input_spatial_dimensions(i);
  }

  // Align LHS indices to RHS.
  std::vector<int64_t> lhs_to_rhs_indices(output_base_shape.rank());
  for (int64_t i = 0; i < rhs_to_lhs_indices.size(); ++i) {
    lhs_to_rhs_indices[rhs_to_lhs_indices[i]] = i;
  }

  // Align LHS indices to output.
  std::vector<int64_t> lhs_to_output_indices(output_base_shape.rank());
  lhs_to_output_indices[dnums.input_feature_dimension()] =
      dnums.output_feature_dimension();
  lhs_to_output_indices[dnums.input_batch_dimension()] =
      dnums.output_batch_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    lhs_to_output_indices[dnums.input_spatial_dimensions(i)] =
        dnums.output_spatial_dimensions(i);
  }

  // Align LHS or RHS if input_feature_dim or kernel_output_feature_dim is
  // partitioned.
  auto aligned_rhs_sharding =
      hlo_sharding_util::TransposeSharding(lhs.sharding(), rhs_to_lhs_indices);
  auto aligned_lhs_sharding =
      hlo_sharding_util::TransposeSharding(rhs.sharding(), lhs_to_rhs_indices);

  bool lhs_feature_dim_is_partitioned =
      (ShardCountAtDim(lhs.sharding(), dnums.input_feature_dimension()) ==
       num_partitions);
  bool rhs_output_feature_dim_is_partitioned =
      (ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) ==
       num_partitions);
  if (!lhs_feature_dim_is_partitioned &&
      !rhs_output_feature_dim_is_partitioned) {
    return nullptr;
  }
  // Reshard LHS or RHS to partition at input feature dimension or output
  // feature dimension as the other operand.
  if (lhs_feature_dim_is_partitioned) {
    rhs = rhs.Reshard(aligned_rhs_sharding);
  } else {
    lhs = lhs.Reshard(aligned_lhs_sharding);
  }

  // Align output sharding after LHS and RHS sharding are consistent.
  auto aligned_output_sharding = hlo_sharding_util::TransposeSharding(
      lhs.sharding(), lhs_to_output_indices);

  TF_ASSIGN_OR_RETURN(
      auto sharded_conv,
      create_sharded_conv(lhs.hlo(), rhs.hlo(), b, conv_window));
  sharded_conv->set_sharding(aligned_output_sharding);
  return PartitionedHlo(sharded_conv, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

// Partition convolution when both LHS and RHS are partitioned at spatial
// dimensions. Halo exchange will happen on RHS only.
absl::StatusOr<HloInstruction*>
PartitionConvolutionWithSpatialDimensionHaloExchangeOnRHS(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    HloInstruction* partition_id, HloModule* module, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);
  TF_RET_CHECK(!lhs.sharding().IsTileMaximal() &&
               !rhs.sharding().IsTileMaximal());

  const auto& dnums = original_hlo->convolution_dimension_numbers();
  std::vector<int64_t> rhs_to_lhs_indices(output_base_shape.rank());
  rhs_to_lhs_indices[dnums.kernel_output_feature_dimension()] =
      dnums.input_batch_dimension();
  rhs_to_lhs_indices[dnums.kernel_input_feature_dimension()] =
      dnums.input_feature_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    rhs_to_lhs_indices[dnums.kernel_spatial_dimensions(i)] =
        dnums.input_spatial_dimensions(i);
  }
  std::vector<int64_t> lhs_to_rhs_indices(output_base_shape.rank());
  for (int64_t i = 0; i < rhs_to_lhs_indices.size(); ++i) {
    lhs_to_rhs_indices[rhs_to_lhs_indices[i]] = i;
  }
  auto aligned_rhs_sharding =
      hlo_sharding_util::TransposeSharding(lhs.sharding(), rhs_to_lhs_indices);
  auto aligned_lhs_sharding =
      hlo_sharding_util::TransposeSharding(rhs.sharding(), lhs_to_rhs_indices);

  auto unsupported_sharding = [&](const HloSharding& lhs_sharding,
                                  const HloSharding& rhs_sharding) {
    // We currently don't support partitioning input batch or output feature
    // dimensions.
    return ShardCountAtDim(lhs_sharding, dnums.input_batch_dimension()) != 1 ||
           ShardCountAtDim(rhs_sharding,
                           dnums.kernel_output_feature_dimension()) != 1;
  };

  if (ShapeSizeInBytes(lhs.base_shape()) < ShapeSizeInBytes(rhs.base_shape())) {
    if (unsupported_sharding(aligned_lhs_sharding, rhs.sharding())) {
      return nullptr;
    }
    lhs = lhs.Reshard(aligned_lhs_sharding).PadWithZero();
    rhs = rhs.PadWithZero();
  } else {
    if (unsupported_sharding(lhs.sharding(), aligned_rhs_sharding)) {
      return nullptr;
    }
    lhs = lhs.PadWithZero();
    rhs = rhs.Reshard(aligned_rhs_sharding).PadWithZero();
  }

  if (original_hlo->feature_group_count() > 1 &&
      (ShardCountAtDim(lhs.sharding(), dnums.input_feature_dimension()) > 1 ||
       ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) > 1)) {
    return nullptr;
  }

  if (original_hlo->batch_group_count() > 1 &&
      (ShardCountAtDim(lhs.sharding(), dnums.input_batch_dimension()) > 1 ||
       ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) > 1)) {
    return nullptr;
  }

  // Reshard RHS so that each shard computes the partial sum of the full
  // shape result, and add AllReduce. See HandleConvolutionTiledLhsAndRhs()
  // that reshards LHS.
  //
  // The size of halo on each dimension can be calculated from the
  // projection onto the RHS that shard i needs to read. RHS and LHS below
  // refers to the shard size of RHS and LHS, WC is the number of windows,
  // and D is the window dilation.
  //
  // * offset(i): LHS * i + low_padding - (WC - 1) * stride
  // * limit(i): LHS * (i + 1) + low_padding
  //
  // Since shard i has RHS of range [i * RHS * D, (i + 1) * RHS * D)
  // * left-halo: i * RHS - offset(i)
  //              = i * (RHS * D - LHS) + (WC - 1) * stride - low_padding
  // * right-halo: limit(i) - (i + 1) * RHS
  //              = (i + 1) * (LHS - RHS * D) + low_pading
  const auto& collective_ops_creator = lhs.state().collective_ops_creator;
  std::vector<int64_t> shard_counts(dnums.input_spatial_dimensions_size());
  std::vector<int64_t> lhs_shard_sizes(dnums.input_spatial_dimensions_size());
  std::vector<int64_t> rhs_shard_sizes(dnums.input_spatial_dimensions_size());

  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64_t lhs_dimension = dnums.input_spatial_dimensions(i);
    int64_t rhs_dimension = dnums.kernel_spatial_dimensions(i);
    int64_t shard_count = ShardCountAtDim(rhs.sharding(), rhs_dimension);
    const auto& wd = conv_window.dimensions(i);
    if (wd.base_dilation() != 1 || wd.window_reversal()) {
      return nullptr;
    }

    int64_t lhs_shard_size =
        CeilOfRatio(lhs.base_shape().dimensions(lhs_dimension), shard_count);
    int64_t rhs_shard_size =
        CeilOfRatio(rhs.base_shape().dimensions(rhs_dimension), shard_count);
    shard_counts[i] = shard_count;
    lhs_shard_sizes[i] = lhs_shard_size;
    rhs_shard_sizes[i] = rhs_shard_size;
  }

  std::vector<OffsetCalculation> left_halo_size_functions(
      output_base_shape.rank());
  std::vector<OffsetCalculation> right_halo_size_functions(
      output_base_shape.rank());
  Window new_window = conv_window;

  // Data structures needed for Pad and DynamicSlice on LHS if needed.
  bool need_dynamic_slice_lhs = false;
  auto partition_ordinals =
      MakeTiledPartitionOrdinals(lhs.sharding(), partition_id, b);
  std::vector<int64_t> zero_padding(output_base_shape.rank());
  PaddingConfig pad_config = window_util::MakeSymmetricPadding(zero_padding);
  auto zero_s32 =
      b->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  std::vector<HloInstruction*> dynamic_slice_start_indices(
      output_base_shape.rank(), zero_s32);
  Shape dynamic_slice_shape = lhs.hlo()->shape();
  Shape pad_shape = lhs.hlo()->shape();

  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64_t lhs_dimension = dnums.input_spatial_dimensions(i);
    int64_t rhs_dimension = dnums.kernel_spatial_dimensions(i);
    int64_t lhs_shard_size = lhs_shard_sizes[i];
    int64_t rhs_shard_size = rhs_shard_sizes[i];

    if (shard_counts[i] == 1) {
      continue;
    }

    // Calculate the left and right halo sizes as described in the comments
    // above. It calculcates the halo sizes with dilation, so we apply
    // CeilOfRatio({left,right}_halo_size, window_dilation).
    const auto& wd = conv_window.dimensions(i);
    int64_t padding_low = wd.padding_low();
    int64_t padding_high = wd.padding_high();
    int64_t base = lhs.base_shape().dimensions(lhs_dimension);
    int64_t window_count = 1 + (padding_low + padding_high + base -
                                (1 + (wd.size() - 1) * wd.window_dilation())) /
                                   wd.stride();
    left_halo_size_functions[rhs_dimension] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            rhs_shard_size * wd.window_dilation() - lhs_shard_size,
            (window_count - 1) * wd.stride() - padding_low +
                wd.window_dilation() - 1,
            wd.window_dilation()));
    right_halo_size_functions[rhs_dimension] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            lhs_shard_size - rhs_shard_size * wd.window_dilation(),
            lhs_shard_size - rhs_shard_size * wd.window_dilation() +
                padding_low + wd.window_dilation() - 1,
            wd.window_dilation()));

    // New RHS window size includes the maximum of both left and right
    // halos.
    int64_t halo_size =
        left_halo_size_functions[rhs_dimension].MaxInRange(1, shard_counts[i]) +
        right_halo_size_functions[rhs_dimension].MaxInRange(
            0, shard_counts[i] - 1);
    int64_t new_window_size =
        rhs.hlo()->shape().dimensions(rhs_dimension) + halo_size;

    // The amount of new low padding could be dynamic (e.g., window_dilation
    // != 1), which requires pad (to the maximum) and dynamic slice on LHS.
    //
    // If we consider the first window, the offset of the dilated RHS that
    // aligns with the first valid LHS element for shard i is 'padding_low +
    // LHS * i'. When the left halo is added to RHS, the offset of the first
    // RHS element is (RHS * i - left_halo) * window_dilation. The
    // difference between the two values is the amount of padding_low we
    // need on LHS.
    auto new_padding_low_function =
        OffsetCalculation(HloOpcode::kMultiply,
                          left_halo_size_functions[rhs_dimension],
                          OffsetCalculation(MultiplyAddDivideOffsetCalculation(
                              0, wd.window_dilation(), 1))) -
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            rhs_shard_size * wd.window_dilation() - lhs_shard_size,
            -padding_low, 1));

    int64_t new_padding_low_max =
        new_padding_low_function.MaxInRange(0, shard_counts[i]);
    int64_t new_padding_low = new_padding_low_max;
    int64_t new_padding_high = window_count * wd.stride() +
                               (new_window_size - 1) * wd.window_dilation() -
                               new_padding_low - lhs_shard_size;

    // We do pad/dynamic-slice only when the padding is dynamic.
    if (!new_padding_low_function.IsConstant()) {
      need_dynamic_slice_lhs = true;
      new_padding_low = 0;
      pad_config.mutable_dimensions(lhs_dimension)
          ->set_edge_padding_low(new_padding_low_max);
      pad_config.mutable_dimensions(lhs_dimension)
          ->set_edge_padding_high(new_padding_low_max);
      pad_shape.set_dimensions(lhs_dimension,
                               lhs_shard_size + 2 * new_padding_low_max);
      dynamic_slice_start_indices[lhs_dimension] =
          (OffsetCalculation(
               MultiplyAddDivideOffsetCalculation(0, new_padding_low_max, 1)) -
           new_padding_low_function)
              .Calculate(partition_ordinals[lhs_dimension], b);
      dynamic_slice_shape.set_dimensions(lhs_dimension,
                                         lhs_shard_size + new_padding_low_max);
    }

    // Since the convolution RHS operand size increased with halos, adjust
    // the window config accordingly.
    new_window.mutable_dimensions(i)->set_padding_low(new_padding_low);
    new_window.mutable_dimensions(i)->set_padding_high(new_padding_high);
    new_window.mutable_dimensions(i)->set_size(
        rhs.hlo()->shape().dimensions(rhs_dimension) + halo_size);
  }

  HloInstruction* conv_lhs = lhs.hlo();
  if (need_dynamic_slice_lhs) {
    auto zero = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(lhs.hlo()->shape().element_type())));
    auto pad = b->AddInstruction(
        HloInstruction::CreatePad(pad_shape, lhs.hlo(), zero, pad_config));
    conv_lhs = b->AddInstruction(HloInstruction::CreateDynamicSlice(
        dynamic_slice_shape, pad, dynamic_slice_start_indices,
        dynamic_slice_shape.dimensions()));
  }

  // Exchange halo and concatenate.
  HloInstruction* rhs_with_halo = rhs.hlo();
  for (int i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
    int64_t dim = dnums.kernel_spatial_dimensions(i);
    int64_t explicit_left_padding_on_full_shape =
        left_halo_size_functions[dim].Calculate(0);
    int64_t shard_size_with_halo = new_window.dimensions(i).size();

    // offset_on_padded_shape and padded_full_shape_size are needed only if
    // we want to mask out-of-range values in ExchangeHaloAndGetValidData().
    // Since the default value for both the collective-permute is zero and
    // also we call PadWithValue() on both operands at the beginning, we
    // don't need to mask here.
    //
    // TODO(hyoulkee): Consider removing one of the two PadWithValue() calls
    // if it's always safe.
    auto offset_on_padded_shape =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            rhs_shard_sizes[i], explicit_left_padding_on_full_shape, 1)) -
        left_halo_size_functions[dim];
    int64_t padded_full_shape_size =
        offset_on_padded_shape.Calculate(shard_counts[i] - 1) +
        new_window.dimensions(i).size();
    auto zero = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(rhs.hlo()->shape().element_type())));
    auto concat = ExchangeHaloAndGetValidData(
        rhs_with_halo, rhs.base_shape(), left_halo_size_functions[dim],
        right_halo_size_functions[dim], explicit_left_padding_on_full_shape,
        padded_full_shape_size, shard_size_with_halo, dim, rhs.sharding(),
        offset_on_padded_shape.Calculate(partition_ordinals[dim], b), zero,
        partition_ordinals[dim], collective_ops_creator,
        lhs.state().next_channel_id, b,
        /*mask_invalid_region=*/false);
    if (!concat) {
      return nullptr;
    }
    rhs_with_halo = *concat;
  }

  TF_ASSIGN_OR_RETURN(
      auto conv, create_sharded_conv(conv_lhs, rhs_with_halo, b, new_window));

  auto ar = collective_ops_creator.create_cross_partition_all_reduce(
      b, conv, MakeBinaryAdd(original_hlo->shape().element_type(), module), {},
      (*lhs.state().next_channel_id)++);
  ar->set_sharding(HloSharding::Replicate());
  return PartitionedHlo(ar, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

// Partition convolution when both LHS and RHS are partitioned at spatial
// dimensions. Halo exchange will happen on LHS only.
absl::StatusOr<HloInstruction*>
PartitionConvolutionWithSpatialDimensionHaloExchangeOnLHS(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    HloInstruction* partition_id, HloModule* module, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);
  TF_RET_CHECK(!lhs.sharding().IsTileMaximal() &&
               !rhs.sharding().IsTileMaximal());

  const auto& dnums = original_hlo->convolution_dimension_numbers();

  // Check if the operand shardings are aligned. Also we currently don't
  // support partitioning non-spatial dimensions.
  std::vector<int64_t> rhs_to_lhs_indices(output_base_shape.rank());
  rhs_to_lhs_indices[dnums.kernel_output_feature_dimension()] =
      dnums.input_batch_dimension();
  rhs_to_lhs_indices[dnums.kernel_input_feature_dimension()] =
      dnums.input_feature_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    rhs_to_lhs_indices[dnums.kernel_spatial_dimensions(i)] =
        dnums.input_spatial_dimensions(i);
  }
  std::vector<int64_t> lhs_to_rhs_indices(output_base_shape.rank());
  for (int64_t i = 0; i < rhs_to_lhs_indices.size(); ++i) {
    lhs_to_rhs_indices[rhs_to_lhs_indices[i]] = i;
  }

  const Window& window = conv_window;
  std::vector<int64_t> reversed_rhs_dims;
  for (int64_t i = 0; i < window.dimensions_size(); ++i) {
    if (window.dimensions(i).window_reversal()) {
      reversed_rhs_dims.push_back(dnums.kernel_spatial_dimensions(i));
    }
  }
  if (!reversed_rhs_dims.empty()) {
    // Make the reversed dims left-padded to prepare for window reversal.
    auto left_padded_rhs = HaloExchangeToPadOnLeft(rhs, reversed_rhs_dims);
    if (left_padded_rhs == nullptr) {
      return nullptr;
    }
    left_padded_rhs->set_sharding(rhs.sharding());
    rhs = PartitionedHlo(left_padded_rhs, rhs.base_shape(), rhs.state());
  }
  // Consider window reversal when resharding RHS or LHS. Note: this will not
  // reverse the data in the shard. We use window reversal to do that.
  auto aligned_rhs_sharding = hlo_sharding_util::ReverseSharding(
      hlo_sharding_util::TransposeSharding(lhs.sharding(), rhs_to_lhs_indices),
      reversed_rhs_dims);
  auto aligned_lhs_sharding = hlo_sharding_util::TransposeSharding(
      hlo_sharding_util::ReverseSharding(rhs.sharding(), reversed_rhs_dims),
      lhs_to_rhs_indices);

  auto unsupported_sharding = [&](const HloSharding& lhs_sharding,
                                  const HloSharding& rhs_sharding) {
    return ShardCountAtDim(lhs_sharding, dnums.input_batch_dimension()) != 1 ||
           ShardCountAtDim(rhs_sharding,
                           dnums.kernel_output_feature_dimension()) != 1;
  };

  if (ShapeSizeInBytes(lhs.base_shape()) < ShapeSizeInBytes(rhs.base_shape())) {
    if (unsupported_sharding(aligned_lhs_sharding, rhs.sharding())) {
      return nullptr;
    }
    lhs = lhs.Reshard(aligned_lhs_sharding).PadWithZero();
    rhs = rhs.PadWithZero(reversed_rhs_dims);
  } else {
    if (unsupported_sharding(lhs.sharding(), aligned_rhs_sharding)) {
      return nullptr;
    }
    lhs = lhs.PadWithZero();
    rhs = rhs.Reshard(aligned_rhs_sharding).PadWithZero(reversed_rhs_dims);
  }

  if (original_hlo->feature_group_count() > 1 &&
      (ShardCountAtDim(lhs.sharding(), dnums.input_feature_dimension()) > 1 ||
       ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) > 1)) {
    return nullptr;
  }

  if (original_hlo->batch_group_count() > 1 &&
      (ShardCountAtDim(lhs.sharding(), dnums.input_batch_dimension()) > 1 ||
       ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) > 1)) {
    return nullptr;
  }
  // Reshard LHS by exchanging halo such that each shard computes the partial
  // sum of the full shape result, and add AllReduce.
  //
  // The size of halo on each dimension can be calculated from the projection
  // onto the LHS that each RHS shard i needs to read. RHS and LHS below refers
  // to the shard size of RHS and LHS, WC is the number of windows, and D is the
  // window dilation.
  //
  // * offset(i): RHS * D * i - low_padding
  // * limit(i): {RHS * (i + 1) * D - (D - 1)} + (WC - 1) * stride - low_padding
  //
  // Since shard i has LHS of range [i * LHS, (i + 1) * LHS)
  // * left-halo: i * LHS - offset(i)
  //              = (LHS - RHS * D) * i + low_padding
  // * right-halo: limit(i) - (i + 1) * LHS
  //   = (RHS * D - LHS) * (i + 1) + (1 - D)  + (WC - 1) * stride - low_padding
  //   = (RHS * D - LHS) * i + (RHS * D - LHS) + (1-D)
  //     + (WC - 1) * stride - low_padding
  std::vector<int64_t> shard_counts(dnums.input_spatial_dimensions_size());
  std::vector<int64_t> lhs_shard_sizes(dnums.input_spatial_dimensions_size());
  std::vector<int64_t> rhs_shard_sizes(dnums.input_spatial_dimensions_size());
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64_t lhs_dimension = dnums.input_spatial_dimensions(i);
    int64_t rhs_dimension = dnums.kernel_spatial_dimensions(i);
    int64_t shard_count = ShardCountAtDim(lhs.sharding(), lhs_dimension);
    const auto& wd = window.dimensions(i);
    if (wd.base_dilation() != 1) {
      // TODO(wangtao): support parallel dim if it is replicate here.
      return nullptr;
    }

    int64_t lhs_shard_size =
        CeilOfRatio(lhs.base_shape().dimensions(lhs_dimension), shard_count);
    int64_t rhs_shard_size =
        CeilOfRatio(rhs.base_shape().dimensions(rhs_dimension), shard_count);
    shard_counts[i] = shard_count;
    lhs_shard_sizes[i] = lhs_shard_size;
    rhs_shard_sizes[i] = rhs_shard_size;
  }

  std::vector<OffsetCalculation> left_halo_size_functions(
      output_base_shape.rank());
  std::vector<OffsetCalculation> right_halo_size_functions(
      output_base_shape.rank());
  Window new_window = window;

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(lhs.sharding(), partition_id, b);
  HloInstruction* lhs_with_halo = lhs.hlo();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64_t lhs_dimension = dnums.input_spatial_dimensions(i);
    int64_t lhs_shard_size = lhs_shard_sizes[i];
    int64_t rhs_shard_size = rhs_shard_sizes[i];

    if (shard_counts[i] == 1) {
      continue;
    }

    // Calculate the left and right halo sizes as described in the comments
    // above.
    const auto& wd = window.dimensions(i);
    int64_t padding_low = wd.padding_low();
    int64_t padding_high = wd.padding_high();
    int64_t base = lhs.base_shape().dimensions(lhs_dimension);
    int64_t window_count = 1 + (padding_low + padding_high + base -
                                (1 + (wd.size() - 1) * wd.window_dilation())) /
                                   wd.stride();
    int64_t rhs_shard_size_dilated =
        (rhs_shard_size - 1) * wd.window_dilation() + 1;

    left_halo_size_functions[lhs_dimension] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            lhs_shard_size - rhs_shard_size * wd.window_dilation(), padding_low,
            1));
    right_halo_size_functions[lhs_dimension] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            rhs_shard_size * wd.window_dilation() - lhs_shard_size,
            rhs_shard_size * wd.window_dilation() - lhs_shard_size + 1 -
                wd.window_dilation() + wd.stride() * (window_count - 1) -
                padding_low,
            1));

    // Exchange halo and concatenate.
    int64_t dim = dnums.input_spatial_dimensions(i);
    int64_t explicit_left_padding_on_full_shape = padding_low;
    int64_t shard_size_with_halo =
        wd.stride() * (window_count - 1) + rhs_shard_size_dilated;

    new_window.mutable_dimensions(i)->set_padding_low(0);
    new_window.mutable_dimensions(i)->set_padding_high(0);
    new_window.mutable_dimensions(i)->set_size(rhs_shard_size);

    // offset_on_padded_shape and padded_full_shape_size are needed only if
    // we want to mask out-of-range values in ExchangeHaloAndGetValidData().
    // Since the default value for both the collective-permute is zero and
    // also we call PadWithValue() on both operands at the beginning, we
    // don't need to mask here.
    //
    // TODO(hyoulkee): Consider removing one of the two PadWithValue() calls
    // if it's always safe.
    auto offset_on_padded_shape =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation());
    int64_t padded_full_shape_size = 0;

    auto zero = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(lhs.hlo()->shape().element_type())));
    auto concat = ExchangeHaloAndGetValidData(
        lhs_with_halo, lhs.base_shape(), left_halo_size_functions[dim],
        right_halo_size_functions[dim], explicit_left_padding_on_full_shape,
        padded_full_shape_size, shard_size_with_halo, dim, lhs.sharding(),
        offset_on_padded_shape.Calculate(partition_ordinals[dim], b), zero,
        partition_ordinals[dim], lhs.state().collective_ops_creator,
        lhs.state().next_channel_id, b,
        /*mask_invalid_region=*/false);
    if (!concat) {
      return nullptr;
    }
    lhs_with_halo = *concat;
  }

  TF_ASSIGN_OR_RETURN(
      auto conv, create_sharded_conv(lhs_with_halo, rhs.hlo(), b, new_window));
  auto ar =
      lhs.state().collective_ops_creator.create_cross_partition_all_reduce(
          b, conv, MakeBinaryAdd(output_base_shape.element_type(), module), {},
          (*lhs.state().next_channel_id)++);
  ar->set_sharding(HloSharding::Replicate());
  return PartitionedHlo(ar, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

// Partition convolution when output is sharded. Will shard LHS with replicated
// RHS.
absl::StatusOr<HloInstruction*> PartitionConvolutionTiledOutput(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);
  const auto& dnums = original_hlo->convolution_dimension_numbers();
  TF_RET_CHECK(!output_sharding.IsTileMaximal());
  // We don't currently support sharding on output feature dimension.
  if (ShardCountAtDim(output_sharding, dnums.output_feature_dimension()) > 1) {
    return nullptr;
  }

  if (original_hlo->feature_group_count() > 1 &&
      (ShardCountAtDim(lhs.sharding(), dnums.input_feature_dimension()) > 1 ||
       ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) > 1)) {
    return nullptr;
  }

  if (original_hlo->batch_group_count() > 1 &&
      (ShardCountAtDim(lhs.sharding(), dnums.input_batch_dimension()) > 1 ||
       ShardCountAtDim(rhs.sharding(),
                       dnums.kernel_output_feature_dimension()) > 1)) {
    return nullptr;
  }

  // Check if the operand and the output sharding are aligned.
  std::vector<int64_t> input_to_output_indices(output_base_shape.rank());
  input_to_output_indices[dnums.input_batch_dimension()] =
      dnums.output_batch_dimension();
  input_to_output_indices[dnums.input_feature_dimension()] =
      dnums.output_feature_dimension();
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    input_to_output_indices[dnums.input_spatial_dimensions(i)] =
        dnums.output_spatial_dimensions(i);
  }
  auto target_operand_sharding = hlo_sharding_util::TransposeSharding(
      output_sharding, input_to_output_indices);
  lhs = lhs.Reshard(target_operand_sharding);

  // Replicate the RHS.
  rhs = rhs.Reshard(HloSharding::Replicate());

  // Convolution window config does not include batch and feature dimensions,
  // whereas ReshardAsWindowedInput() expects the same number of window
  // dimensions as the rank of the operand. So add two more trivial
  // dimensions.
  std::vector<int64_t> ones(output_base_shape.rank(), 1);
  auto operand_window = window_util::MakeWindow(ones);
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    *operand_window.mutable_dimensions(dnums.input_spatial_dimensions(i)) =
        conv_window.dimensions(i);
  }

  auto zero = b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(output_base_shape.element_type())));
  auto resharded_operand_and_window =
      lhs.ReshardAsWindowedInput(operand_window, target_operand_sharding, zero);
  if (!resharded_operand_and_window.has_value()) {
    return nullptr;
  }
  Window new_window;
  for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    *new_window.add_dimensions() =
        resharded_operand_and_window->shard_window.dimensions(
            dnums.input_spatial_dimensions(i));
  }

  TF_ASSIGN_OR_RETURN(
      auto sharded_conv,
      create_sharded_conv(resharded_operand_and_window->sharded_input,
                          rhs.hlo(), b, new_window));

  auto shard_shape = MakePartitionedShape(output_base_shape, output_sharding);
  if (!resharded_operand_and_window->dynamic_slice_index_on_output
           .has_value()) {
    CHECK(ShapeUtil::Compatible(shard_shape, sharded_conv->shape()));
    return sharded_conv;
  }
  return b->AddInstruction(HloInstruction::CreateDynamicSlice(
      shard_shape, sharded_conv,
      *resharded_operand_and_window->dynamic_slice_index_on_output,
      shard_shape.dimensions()));
}

// Partition convolution with only one kind of dims partitioned.
absl::StatusOr<HloInstruction*> PartitionConvolutionBaseCase(
    const PartitionedHlo& lhs, const PartitionedHlo& rhs,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    int64_t num_partitions, const SpmdPartitionerOptions& options,
    HloInstruction* partition_id, HloModule* module, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);

  // Case 1: Handle depthwise convolution with batch group count or
  // feature group count.
  if (original_hlo->batch_group_count() > 1) {
    TF_ASSIGN_OR_RETURN(
        auto parallel_partitioned_conv,
        PartitionConvolutionWithBatchGroupCount(
            lhs, rhs, output_base_shape, output_sharding, create_sharded_conv,
            conv_window, original_hlo, num_partitions, b));
    if (parallel_partitioned_conv) {
      return parallel_partitioned_conv;
    }
  }

  if (original_hlo->feature_group_count() > 1) {
    TF_ASSIGN_OR_RETURN(
        auto parallel_partitioned_conv,
        PartitionConvolutionWithFeatureGroupCount(
            lhs, rhs, output_base_shape, output_sharding, create_sharded_conv,
            conv_window, original_hlo, num_partitions, b));
    if (parallel_partitioned_conv) {
      return parallel_partitioned_conv;
    }
  }

  // Case 2: both RHS and LHS are tiled.
  // Handling cases where both operands' shardings are aligned. We check that
  // the LHS batch dimension is not partitioned because it is mapped to the
  // output feature dimension in aligned_rhs_sharding, which are not the same
  // dimension.
  if (!lhs.sharding().IsTileMaximal() && !rhs.sharding().IsTileMaximal()) {
    if (options.conv_halo_exchange_always_on_lhs) {
      TF_ASSIGN_OR_RETURN(
          auto partitioned_conv,
          PartitionConvolutionWithSpatialDimensionHaloExchangeOnLHS(
              lhs, rhs, output_base_shape, output_sharding, create_sharded_conv,
              conv_window, original_hlo, partition_id, module, b));
      if (partitioned_conv) {
        return partitioned_conv;
      }
    } else {
      TF_ASSIGN_OR_RETURN(
          auto partitioned_conv,
          PartitionConvolutionWithSpatialDimensionHaloExchangeOnRHS(
              lhs, rhs, output_base_shape, output_sharding, create_sharded_conv,
              conv_window, original_hlo, partition_id, module, b));
      if (partitioned_conv) {
        return partitioned_conv;
      }
    }
  }

  // Case 3: output is tiled.
  if (!output_sharding.IsTileMaximal()) {
    TF_ASSIGN_OR_RETURN(auto partitioned_conv,
                        PartitionConvolutionTiledOutput(
                            lhs, rhs, output_base_shape, output_sharding,
                            create_sharded_conv, conv_window, original_hlo, b));
    if (partitioned_conv) {
      return partitioned_conv;
    }
  }
  return nullptr;
}

absl::StatusOr<std::unique_ptr<HloInstruction>> CreateShardedConvolution(
    const HloInstruction& conv,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dot_dnums,
    HloInstruction* sharded_lhs_hlo, HloInstruction* sharded_rhs_hlo,
    const Window& conv_window) {
  CHECK_EQ(conv.opcode(), HloOpcode::kConvolution);
  const auto& conv_dnums = conv.convolution_dimension_numbers();
  auto window = conv.window();
  for (const auto& dim : dot_dnums.batch_dims) {
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
    wd->set_stride(std::max<int64_t>(1, wd->size() - 1));
    wd->set_base_dilation(wd->size());
  }
  for (const auto& dim : dot_dnums.contracting_dims) {
    if (dim.spatial_dim < 0) {
      continue;
    }
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_lhs_hlo->shape().dimensions(
        conv_dnums.input_spatial_dimensions(dim.spatial_dim)));
  }
  for (const auto& dim : dot_dnums.rhs_non_contracting_dims) {
    if (dim.spatial_dim < 0) {
      continue;
    }
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    wd->set_size(sharded_rhs_hlo->shape().dimensions(
        conv_dnums.kernel_spatial_dimensions(dim.spatial_dim)));
    wd->set_padding_high(wd->size() - 1);
    wd->set_padding_low(wd->size() - 1);
  }

  for (const auto& dim : dot_dnums.conv_spatial_dims) {
    auto wd = window.mutable_dimensions(dim.spatial_dim);
    const auto& new_window_dimension = conv_window.dimensions(dim.spatial_dim);
    wd->set_size(new_window_dimension.size());
    wd->set_padding_high(new_window_dimension.padding_high());
    wd->set_padding_low(new_window_dimension.padding_low());
  }

  int64_t feature_group_count = conv.feature_group_count();
  if (feature_group_count > 1) {
    feature_group_count = sharded_lhs_hlo->shape().dimensions(
                              conv_dnums.input_feature_dimension()) /
                          sharded_rhs_hlo->shape().dimensions(
                              conv_dnums.kernel_input_feature_dimension());
  }

  // We always have output_batch_size * batch_group_count = input_batch_size.
  const int64_t old_input_batch_size =
      conv.operand(0)->shape().dimensions(conv_dnums.input_batch_dimension());
  const int64_t old_output_batch_size =
      conv.shape().dimensions(conv_dnums.output_batch_dimension());
  const int64_t old_batch_group_count = conv.batch_group_count();
  CHECK_EQ(old_output_batch_size * old_batch_group_count, old_input_batch_size);

  int64_t batch_group_count = old_batch_group_count;
  if (batch_group_count > 1) {
    // For the new convolution instruction, we have the new_input_batch_size
    // from sharded_lhs. We keep the output_batch_size and calculate the new
    // batch_group_count accordingly.
    const int64_t new_input_batch_size =
        sharded_lhs_hlo->shape().dimensions(conv_dnums.input_batch_dimension());
    const int64_t new_output_batch_size = old_output_batch_size;
    CHECK_EQ(new_input_batch_size % new_output_batch_size, 0);
    batch_group_count = new_input_batch_size / new_output_batch_size;
  }

  TF_ASSIGN_OR_RETURN(
      Shape sharded_conv_shape,
      ShapeInference::InferConvolveShape(
          sharded_lhs_hlo->shape(), sharded_rhs_hlo->shape(),
          feature_group_count, batch_group_count, window, conv_dnums,
          /*preferred_element_type=*/conv.shape().element_type()));
  *sharded_conv_shape.mutable_layout() = conv.shape().layout();
  return HloInstruction::CreateConvolve(
      sharded_conv_shape, sharded_lhs_hlo, sharded_rhs_hlo, feature_group_count,
      batch_group_count, window, conv_dnums, conv.precision_config());
}

}  // namespace

// Partition convolution.
absl::StatusOr<HloInstruction*> PartitionConvolution(
    const PartitionedHlo& lhs, const PartitionedHlo& rhs,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dims_mapping,
    absl::FunctionRef<absl::StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>
        create_sharded_conv,
    const Window& conv_window, HloInstruction* original_hlo,
    int64_t num_partitions, const SpmdPartitionerOptions& options,
    HloInstruction* partition_id, HloModule* module, SpmdBuilder* b) {
  TF_RET_CHECK(original_hlo->opcode() == HloOpcode::kConvolution);

  TF_ASSIGN_OR_RETURN(auto try_partitioned_conv,
                      PartitionConvolutionBaseCase(
                          lhs, rhs, output_base_shape, output_sharding,
                          create_sharded_conv, conv_window, original_hlo,
                          num_partitions, options, partition_id, module, b));
  if (try_partitioned_conv) {
    return try_partitioned_conv;
  }

  return nullptr;
}

absl::Status SpmdPartitioningVisitor::HandleConvolution(HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return DefaultAction(hlo);
  }
  const auto dims_info = dot_as_convolution_util::ParseConvolutionDimsInfo(hlo);

  auto create_sharded_conv =
      [&](HloInstruction* lhs_hlo, HloInstruction* rhs_hlo,
          spmd::SpmdBuilder* b,
          const Window& conv_window) -> absl::StatusOr<HloInstruction*> {
    if (dims_info.conv_spatial_dims.empty() &&
        hlo->feature_group_count() == 1 && hlo->batch_group_count() == 1) {
      TF_ASSIGN_OR_RETURN(
          auto sharded_conv,
          dot_as_convolution_util::CreateShardedConvForDotGeneralConvolution(
              *hlo, dims_info, lhs_hlo, rhs_hlo));
      return b->AddInstruction(std::move(sharded_conv));
    } else {
      TF_ASSIGN_OR_RETURN(auto sharded_conv,
                          CreateShardedConvolution(*hlo, dims_info, lhs_hlo,
                                                   rhs_hlo, conv_window));
      return b->AddInstruction(std::move(sharded_conv));
    }
  };

  return HandleDotHelper(hlo, dims_info, create_sharded_conv);
}

}  // namespace spmd
}  // namespace xla
