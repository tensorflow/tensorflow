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

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dot_as_convolution_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/numbers.h"

namespace xla {
namespace spmd {

Status SpmdPartitioningVisitor::HandleConvolutionTiledLhsAndRhs(
    HloInstruction* hlo) {
  TF_RET_CHECK(hlo->opcode() == HloOpcode::kConvolution);

  auto lhs = GetPartitionedHlo(hlo->operand(0));
  auto rhs = GetPartitionedHlo(hlo->operand(1));
  TF_RET_CHECK(!lhs.sharding().IsTileMaximal() &&
               !rhs.sharding().IsTileMaximal());

  const auto& dnums = hlo->convolution_dimension_numbers();

  // Check if the operand shardings are aligned. Also we currently don't
  // support partitioning non-spatial dimensions.
  std::vector<int64> rhs_to_lhs_indices(hlo->shape().rank());
  rhs_to_lhs_indices[dnums.kernel_output_feature_dimension()] =
      dnums.input_batch_dimension();
  rhs_to_lhs_indices[dnums.kernel_input_feature_dimension()] =
      dnums.input_feature_dimension();
  for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    rhs_to_lhs_indices[dnums.kernel_spatial_dimensions(i)] =
        dnums.input_spatial_dimensions(i);
  }
  std::vector<int64> lhs_to_rhs_indices(hlo->shape().rank());
  for (int64 i = 0; i < rhs_to_lhs_indices.size(); ++i) {
    lhs_to_rhs_indices[rhs_to_lhs_indices[i]] = i;
  }

  Window window = hlo->window();
  std::vector<int64> reversed_rhs_dims;
  for (int64 i = 0; i < window.dimensions_size(); ++i) {
    if (window.dimensions(i).window_reversal()) {
      reversed_rhs_dims.push_back(dnums.kernel_spatial_dimensions(i));
    }
  }
  if (!reversed_rhs_dims.empty()) {
    // Make the reversed dims left-padded to prepare for window reversal.
    auto left_padded_rhs = HaloExchangeToPadOnLeft(rhs, reversed_rhs_dims);
    if (left_padded_rhs == nullptr) {
      return DefaultAction(hlo);
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
    return lhs_sharding.tile_assignment().dim(dnums.input_batch_dimension()) !=
               1 ||
           rhs_sharding.tile_assignment().dim(
               dnums.kernel_output_feature_dimension()) != 1;
  };

  auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(hlo->shape().element_type())));
  if (ShapeSizeInBytes(lhs.base_shape()) < ShapeSizeInBytes(rhs.base_shape())) {
    if (unsupported_sharding(aligned_lhs_sharding, rhs.sharding())) {
      return DefaultAction(hlo);
    }
    lhs = lhs.Reshard(aligned_lhs_sharding).PadWithValue(zero);
    rhs = rhs.PadWithValue(zero, reversed_rhs_dims);
  } else {
    if (unsupported_sharding(lhs.sharding(), aligned_rhs_sharding)) {
      return DefaultAction(hlo);
    }
    lhs = lhs.PadWithValue(zero);
    rhs =
        rhs.Reshard(aligned_rhs_sharding).PadWithValue(zero, reversed_rhs_dims);
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
  // * limit(i): {(RHS - 1) * D + 1} * (i + 1) + (WC - 1) * stride - low_padding
  //
  // Since shard i has LHS of range [i * LHS, (i + 1) * LHS)
  // * left-halo: i * LHS - offset(i)
  //              = (LHS - RHS) * i + low_padding
  // * right-halo: limit(i) - (i + 1) * LHS
  //   = [{(RHS - 1) * D + 1} - LHS] * (i + 1) + (WC - 1) * stride - low_padding
  std::vector<int64> shard_counts(dnums.input_spatial_dimensions_size());
  std::vector<int64> lhs_shard_sizes(dnums.input_spatial_dimensions_size());
  std::vector<int64> rhs_shard_sizes(dnums.input_spatial_dimensions_size());
  for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64 lhs_dimension = dnums.input_spatial_dimensions(i);
    int64 rhs_dimension = dnums.kernel_spatial_dimensions(i);
    int64 shard_count = lhs.sharding().tile_assignment().dim(lhs_dimension);
    auto wd = window.dimensions(i);
    if (wd.base_dilation() != 1) {
      return DefaultAction(hlo);
    }

    int64 lhs_shard_size =
        CeilOfRatio(lhs.base_shape().dimensions(lhs_dimension), shard_count);
    int64 rhs_shard_size =
        CeilOfRatio(rhs.base_shape().dimensions(rhs_dimension), shard_count);
    shard_counts[i] = shard_count;
    lhs_shard_sizes[i] = lhs_shard_size;
    rhs_shard_sizes[i] = rhs_shard_size;
  }

  std::vector<OffsetCalculation> left_halo_size_functions(hlo->shape().rank());
  std::vector<OffsetCalculation> right_halo_size_functions(hlo->shape().rank());
  Window new_window = window;

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(lhs.sharding(), partition_id_, &b_);
  HloInstruction* lhs_with_halo = lhs.hlo();
  for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64 lhs_dimension = dnums.input_spatial_dimensions(i);
    int64 lhs_shard_size = lhs_shard_sizes[i];
    int64 rhs_shard_size = rhs_shard_sizes[i];

    if (shard_counts[i] == 1) {
      continue;
    }

    // Calculate the left and right halo sizes as described in the comments
    // above.
    auto wd = window.dimensions(i);
    int64 padding_low = wd.padding_low();
    int64 padding_high = wd.padding_high();
    int64 base = lhs.base_shape().dimensions(lhs_dimension);
    int64 window_count = 1 + (padding_low + padding_high + base -
                              (1 + (wd.size() - 1) * wd.window_dilation())) /
                                 wd.stride();
    int64 rhs_shard_size_dilated =
        (rhs_shard_size - 1) * wd.window_dilation() + 1;

    left_halo_size_functions[lhs_dimension] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            lhs_shard_size - rhs_shard_size * wd.window_dilation(), padding_low,
            1));
    right_halo_size_functions[lhs_dimension] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            rhs_shard_size_dilated - lhs_shard_size,
            rhs_shard_size_dilated - lhs_shard_size +
                wd.stride() * (window_count - 1) - padding_low,
            1));

    // Exchange halo and concatenate.
    int64 dim = dnums.input_spatial_dimensions(i);
    int64 explicit_left_padding_on_full_shape = padding_low;
    int64 shard_size_with_halo =
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
    int64 padded_full_shape_size = 0;
    auto concat = ExchangeHaloAndGetValidData(
        lhs_with_halo, lhs.base_shape(), left_halo_size_functions[dim],
        right_halo_size_functions[dim], explicit_left_padding_on_full_shape,
        padded_full_shape_size, shard_size_with_halo, dim, lhs.sharding(),
        offset_on_padded_shape.Calculate(partition_ordinals[dim], &b_), zero,
        partition_ordinals[dim], collective_ops_creator_, next_channel_id_, &b_,
        /*mask_invalid_region=*/false);
    if (!concat) {
      return DefaultAction(hlo);
    }
    lhs_with_halo = *concat;
  }

  SetPartitionedHlo(hlo, [&]() {
    auto conv = b_.AddInstruction(HloInstruction::CreateConvolve(
        hlo->shape(), lhs_with_halo, rhs.hlo(), hlo->feature_group_count(),
        hlo->batch_group_count(), new_window,
        hlo->convolution_dimension_numbers(), hlo->precision_config()));
    auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
        &b_, conv, MakeBinaryAdd(hlo->shape().element_type(), module_),
        NewChannel());
    ar->set_sharding(HloSharding::Replicate());
    return PartitionedHlo(ar, hlo->shape(), MakePartitioningState())
        .Reshard(hlo->sharding())
        .hlo();
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleConvolution(HloInstruction* hlo) {
  auto dot_dnums = dot_as_convolution_util::ParseDotGeneralFromConvolution(hlo);
  if (dot_dnums) {
    // Use HandleDotHelper() for convs that are actually einsums.
    spmd::DotGeneralDimsMapping mapping;
    for (const auto& dims : dot_dnums->batch_dims) {
      mapping.batch_dims.emplace_back();
      mapping.batch_dims.back().lhs = dims.lhs;
      mapping.batch_dims.back().rhs = dims.rhs;
      mapping.batch_dims.back().output = dims.output;
    }
    for (const auto& dims : dot_dnums->contracting_dims) {
      mapping.contracting_dims.emplace_back();
      mapping.contracting_dims.back().lhs = dims.lhs;
      mapping.contracting_dims.back().rhs = dims.rhs;
      mapping.contracting_dims.back().output = dims.output;
    }
    for (const auto& dims : dot_dnums->lhs_non_contracting_dims) {
      mapping.lhs_non_contracting_dims.emplace_back();
      mapping.lhs_non_contracting_dims.back().lhs = dims.lhs;
      mapping.lhs_non_contracting_dims.back().rhs = dims.rhs;
      mapping.lhs_non_contracting_dims.back().output = dims.output;
    }
    for (const auto& dims : dot_dnums->rhs_non_contracting_dims) {
      mapping.rhs_non_contracting_dims.emplace_back();
      mapping.rhs_non_contracting_dims.back().lhs = dims.lhs;
      mapping.rhs_non_contracting_dims.back().rhs = dims.rhs;
      mapping.rhs_non_contracting_dims.back().output = dims.output;
    }
    auto create_sharded_conv =
        [&](HloInstruction* lhs_hlo, HloInstruction* rhs_hlo,
            spmd::SpmdBuilder* b) -> StatusOr<HloInstruction*> {
      TF_ASSIGN_OR_RETURN(
          auto sharded_conv,
          dot_as_convolution_util::CreateShardedConvForDotGeneralConvolution(
              *hlo, *dot_dnums, lhs_hlo, rhs_hlo));
      return b->AddInstruction(std::move(sharded_conv));
    };
    return HandleDotHelper(hlo, mapping, create_sharded_conv);
  }

  auto lhs = GetPartitionedHlo(hlo->operand(0));
  auto rhs = GetPartitionedHlo(hlo->operand(1));
  const HloSharding& sharding = hlo->sharding();
  const auto& dnums = hlo->convolution_dimension_numbers();
  std::vector<int64> rhs_to_lhs_indices(hlo->shape().rank());
  rhs_to_lhs_indices[dnums.kernel_output_feature_dimension()] =
      dnums.input_batch_dimension();
  rhs_to_lhs_indices[dnums.kernel_input_feature_dimension()] =
      dnums.input_feature_dimension();
  for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    rhs_to_lhs_indices[dnums.kernel_spatial_dimensions(i)] =
        dnums.input_spatial_dimensions(i);
  }
  std::vector<int64> lhs_to_rhs_indices(hlo->shape().rank());
  for (int64 i = 0; i < rhs_to_lhs_indices.size(); ++i) {
    lhs_to_rhs_indices[rhs_to_lhs_indices[i]] = i;
  }
  auto aligned_rhs_sharding =
      hlo_sharding_util::TransposeSharding(lhs.sharding(), rhs_to_lhs_indices);
  auto aligned_lhs_sharding =
      hlo_sharding_util::TransposeSharding(rhs.sharding(), lhs_to_rhs_indices);

  // Handling cases where all the partitioned dimensions are parallel
  // dimensions.
  int64 lhs_parallel_dim_partitions = 1;
  int64 rhs_parallel_dim_partitions = 1;
  std::vector<int64> parallel_spatial_dims;
  for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64 lhs_dim = dnums.input_spatial_dimensions(i);
    int64 lhs_size = lhs.base_shape().dimensions(lhs_dim);
    const auto& wd = hlo->window().dimensions(i);
    int64 rhs_dim = dnums.kernel_spatial_dimensions(i);
    // Only non reversal window is supported right now.
    if (!wd.window_reversal() &&
        dot_as_convolution_util::ConvSpatialDimensionIsParallel(wd, lhs_size)) {
      parallel_spatial_dims.emplace_back(i);
      lhs_parallel_dim_partitions *= ShardCountAtDim(lhs.sharding(), lhs_dim);
      rhs_parallel_dim_partitions *= ShardCountAtDim(rhs.sharding(), rhs_dim);
    }
  }
  bool lhs_partition_dims_are_parallel =
      (lhs_parallel_dim_partitions == num_partitions_);
  bool rhs_partition_dims_are_parallel =
      (rhs_parallel_dim_partitions == num_partitions_);

  // If there is a parallel dim and all the partitioned dimensions are parallel
  // dimensions in either LHS or RHS, simply create partitioned convolutions.
  if (!parallel_spatial_dims.empty() &&
      (lhs_partition_dims_are_parallel || rhs_partition_dims_are_parallel)) {
    // Reshard LHS or RHS to partition at parallel dimensions as the other
    // operand.
    if (lhs_partition_dims_are_parallel) {
      rhs = rhs.Reshard(aligned_rhs_sharding);
    } else {
      lhs = lhs.Reshard(aligned_lhs_sharding);
    }
    auto lhs_shard_shape =
        MakePartitionedShape(lhs.base_shape(), lhs.sharding());
    auto rhs_shard_shape =
        MakePartitionedShape(rhs.base_shape(), rhs.sharding());
    // Update convolution window.
    auto new_window = hlo->window();
    for (const auto& spatial_dim : parallel_spatial_dims) {
      auto wd = new_window.mutable_dimensions(spatial_dim);
      wd->set_size(lhs_shard_shape.dimensions(
          dnums.input_spatial_dimensions(spatial_dim)));
      wd->set_stride(std::max<int64>(1, wd->size() - 1));
      wd->set_base_dilation(wd->size());
    }
    TF_ASSIGN_OR_RETURN(
        Shape sharded_conv_shape,
        ShapeInference::InferConvolveShape(
            lhs_shard_shape, rhs_shard_shape, hlo->feature_group_count(),
            hlo->batch_group_count(), new_window, dnums));
    *sharded_conv_shape.mutable_layout() = hlo->shape().layout();
    SetPartitionedHlo(hlo, [&]() {
      auto sharded_conv = b_.AddInstruction(HloInstruction::CreateConvolve(
          sharded_conv_shape, lhs.hlo(), rhs.hlo(), hlo->feature_group_count(),
          hlo->batch_group_count(), new_window, dnums,
          hlo->precision_config()));
      sharded_conv->set_sharding(hlo->sharding());
      return PartitionedHlo(sharded_conv, hlo->shape(), MakePartitioningState())
          .Reshard(hlo->sharding())
          .hlo();
    });
    return Status::OK();
  }

  // Handling cases where both operands' shardings are aligned. We check that
  // the LHS batch dimension is not partitioned because it is mapped to the
  // output feature dimension in aligned_rhs_sharding, which are not the same
  // dimension.
  if (!lhs.sharding().IsTileMaximal() && !rhs.sharding().IsTileMaximal()) {
    if (options_.conv_halo_exchange_always_on_lhs) {
      return HandleConvolutionTiledLhsAndRhs(hlo);
    } else {
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

      auto unsupported_sharding = [&](const HloSharding& lhs_sharding,
                                      const HloSharding& rhs_sharding) {
        // We currently don't support partitioning input batch or output feature
        // dimensions.
        return lhs_sharding.tile_assignment().dim(
                   dnums.input_batch_dimension()) != 1 ||
               rhs_sharding.tile_assignment().dim(
                   dnums.kernel_output_feature_dimension()) != 1;
      };
      auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(hlo->shape().element_type())));
      if (ShapeSizeInBytes(lhs.base_shape()) <
          ShapeSizeInBytes(rhs.base_shape())) {
        if (unsupported_sharding(aligned_lhs_sharding, rhs.sharding())) {
          return DefaultAction(hlo);
        }
        lhs = lhs.Reshard(aligned_lhs_sharding).PadWithValue(zero);
        rhs = rhs.PadWithValue(zero);
      } else {
        if (unsupported_sharding(lhs.sharding(), aligned_rhs_sharding)) {
          return DefaultAction(hlo);
        }
        lhs = lhs.PadWithValue(zero);
        rhs = rhs.Reshard(aligned_rhs_sharding).PadWithValue(zero);
      }

      Window window = hlo->window();
      std::vector<int64> shard_counts(dnums.input_spatial_dimensions_size());
      std::vector<int64> lhs_shard_sizes(dnums.input_spatial_dimensions_size());
      std::vector<int64> rhs_shard_sizes(dnums.input_spatial_dimensions_size());
      for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
        int64 lhs_dimension = dnums.input_spatial_dimensions(i);
        int64 rhs_dimension = dnums.kernel_spatial_dimensions(i);
        int64 shard_count = rhs.sharding().tile_assignment().dim(rhs_dimension);
        auto wd = window.dimensions(i);
        if (wd.base_dilation() != 1 || wd.window_reversal()) {
          return DefaultAction(hlo);
        }

        int64 lhs_shard_size = CeilOfRatio(
            lhs.base_shape().dimensions(lhs_dimension), shard_count);
        int64 rhs_shard_size = CeilOfRatio(
            rhs.base_shape().dimensions(rhs_dimension), shard_count);
        shard_counts[i] = shard_count;
        lhs_shard_sizes[i] = lhs_shard_size;
        rhs_shard_sizes[i] = rhs_shard_size;
      }

      std::vector<OffsetCalculation> left_halo_size_functions(
          hlo->shape().rank());
      std::vector<OffsetCalculation> right_halo_size_functions(
          hlo->shape().rank());
      Window new_window = window;

      // Data structures needed for Pad and DynamicSlice on LHS if needed.
      bool need_dynamic_slice_lhs = false;
      auto partition_ordinals =
          MakeTiledPartitionOrdinals(lhs.sharding(), partition_id_, &b_);
      std::vector<int64> zero_padding(hlo->shape().rank());
      PaddingConfig pad_config =
          window_util::MakeSymmetricPadding(zero_padding);
      auto zero_s32 = b_.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
      std::vector<HloInstruction*> dynamic_slice_start_indices(
          hlo->shape().rank(), zero_s32);
      Shape dynamic_slice_shape = lhs.hlo()->shape();
      Shape pad_shape = lhs.hlo()->shape();

      for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
        int64 lhs_dimension = dnums.input_spatial_dimensions(i);
        int64 rhs_dimension = dnums.kernel_spatial_dimensions(i);
        int64 lhs_shard_size = lhs_shard_sizes[i];
        int64 rhs_shard_size = rhs_shard_sizes[i];

        if (shard_counts[i] == 1) {
          continue;
        }

        // Calculate the left and right halo sizes as described in the comments
        // above. It calculcates the halo sizes with dilation, so we apply
        // CeilOfRatio({left,right}_halo_size, window_dilation).
        auto wd = window.dimensions(i);
        int64 padding_low = wd.padding_low();
        int64 padding_high = wd.padding_high();
        int64 base = lhs.base_shape().dimensions(lhs_dimension);
        int64 window_count =
            1 + (padding_low + padding_high + base -
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
        int64 halo_size = left_halo_size_functions[rhs_dimension].MaxInRange(
                              1, shard_counts[i]) +
                          right_halo_size_functions[rhs_dimension].MaxInRange(
                              0, shard_counts[i] - 1);
        int64 new_window_size =
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
            OffsetCalculation(
                HloOpcode::kMultiply, left_halo_size_functions[rhs_dimension],
                OffsetCalculation(MultiplyAddDivideOffsetCalculation(
                    0, wd.window_dilation(), 1))) -
            OffsetCalculation(MultiplyAddDivideOffsetCalculation(
                rhs_shard_size * wd.window_dilation() - lhs_shard_size,
                -padding_low, 1));

        int64 new_padding_low_max =
            new_padding_low_function.MaxInRange(0, shard_counts[i]);
        int64 new_padding_low = new_padding_low_max;
        int64 new_padding_high = window_count * wd.stride() +
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
              (OffsetCalculation(MultiplyAddDivideOffsetCalculation(
                   0, new_padding_low_max, 1)) -
               new_padding_low_function)
                  .Calculate(partition_ordinals[lhs_dimension], &b_);
          dynamic_slice_shape.set_dimensions(
              lhs_dimension, lhs_shard_size + new_padding_low_max);
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
        auto pad = b_.AddInstruction(
            HloInstruction::CreatePad(pad_shape, lhs.hlo(), zero, pad_config));
        conv_lhs = b_.AddInstruction(HloInstruction::CreateDynamicSlice(
            dynamic_slice_shape, pad, dynamic_slice_start_indices,
            dynamic_slice_shape.dimensions()));
      }

      // Exchange halo and concatenate.
      HloInstruction* rhs_with_halo = rhs.hlo();
      for (int i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
        int64 dim = dnums.kernel_spatial_dimensions(i);
        int64 explicit_left_padding_on_full_shape =
            left_halo_size_functions[dim].Calculate(0);
        int64 shard_size_with_halo = new_window.dimensions(i).size();

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
        int64 padded_full_shape_size =
            offset_on_padded_shape.Calculate(shard_counts[i] - 1) +
            new_window.dimensions(i).size();
        auto concat = ExchangeHaloAndGetValidData(
            rhs_with_halo, rhs.base_shape(), left_halo_size_functions[dim],
            right_halo_size_functions[dim], explicit_left_padding_on_full_shape,
            padded_full_shape_size, shard_size_with_halo, dim, rhs.sharding(),
            offset_on_padded_shape.Calculate(partition_ordinals[dim], &b_),
            zero, partition_ordinals[dim], collective_ops_creator_,
            next_channel_id_, &b_, /*mask_invalid_region=*/false);
        if (!concat) {
          return DefaultAction(hlo);
        }
        rhs_with_halo = *concat;
      }

      SetPartitionedHlo(hlo, [&]() {
        auto conv = b_.AddInstruction(HloInstruction::CreateConvolve(
            hlo->shape(), conv_lhs, rhs_with_halo, hlo->feature_group_count(),
            hlo->batch_group_count(), new_window, dnums,
            hlo->precision_config()));
        auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
            &b_, conv, MakeBinaryAdd(hlo->shape().element_type(), module_),
            NewChannel());
        ar->set_sharding(HloSharding::Replicate());
        return PartitionedHlo(ar, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  }

  if (!sharding.IsTileMaximal()) {
    // We don't currently support sharding on output feature dimension.
    if (sharding.tile_assignment().dim(dnums.output_feature_dimension()) > 1) {
      return DefaultAction(hlo);
    }

    // Check if the operand and the output sharding are aligned.
    std::vector<int64> input_to_output_indices(hlo->shape().rank());
    input_to_output_indices[dnums.input_batch_dimension()] =
        dnums.output_batch_dimension();
    input_to_output_indices[dnums.input_feature_dimension()] =
        dnums.output_feature_dimension();
    for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      input_to_output_indices[dnums.input_spatial_dimensions(i)] =
          dnums.output_spatial_dimensions(i);
    }
    auto target_operand_sharding =
        hlo_sharding_util::TransposeSharding(sharding, input_to_output_indices);
    lhs = lhs.Reshard(target_operand_sharding);

    // Replicate the RHS.
    rhs = rhs.Reshard(HloSharding::Replicate());

    // Convolution window config does not include batch and feature dimensions,
    // whereas ReshardAsWindowedInput() expects the same number of window
    // dimensions as the rank of the operand. So add two more trivial
    // dimensions.
    std::vector<int64> ones(hlo->shape().rank(), 1);
    auto operand_window = window_util::MakeWindow(ones);
    for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      *operand_window.mutable_dimensions(dnums.input_spatial_dimensions(i)) =
          hlo->window().dimensions(i);
    }

    auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(hlo->shape().element_type())));
    auto resharded_operand_and_window = lhs.ReshardAsWindowedInput(
        operand_window, target_operand_sharding, zero);
    if (!resharded_operand_and_window.has_value()) {
      return DefaultAction(hlo);
    }
    Window new_window;
    for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      *new_window.add_dimensions() =
          resharded_operand_and_window->shard_window.dimensions(
              dnums.input_spatial_dimensions(i));
    }
    TF_ASSIGN_OR_RETURN(
        Shape sharded_conv_shape,
        ShapeInference::InferConvolveShape(
            resharded_operand_and_window->sharded_input->shape(),
            rhs.hlo()->shape(), hlo->feature_group_count(),
            hlo->batch_group_count(), new_window, dnums));
    auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
    *sharded_conv_shape.mutable_layout() = shard_shape.layout();
    SetPartitionedHlo(hlo, [&]() {
      auto sharded_conv = b_.AddInstruction(HloInstruction::CreateConvolve(
          sharded_conv_shape, resharded_operand_and_window->sharded_input,
          rhs.hlo(), hlo->feature_group_count(), hlo->batch_group_count(),
          new_window, dnums, hlo->precision_config()));
      if (!resharded_operand_and_window->dynamic_slice_index_on_output
               .has_value()) {
        CHECK(ShapeUtil::Compatible(shard_shape, sharded_conv->shape()));
        return sharded_conv;
      }
      return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
          shard_shape, sharded_conv,
          *resharded_operand_and_window->dynamic_slice_index_on_output,
          shard_shape.dimensions()));
    });
    return Status::OK();
  }
  return DefaultAction(hlo);
}

}  // namespace spmd
}  // namespace xla
