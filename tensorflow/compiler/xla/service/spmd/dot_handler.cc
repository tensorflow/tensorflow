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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/spmd/convolution_handler.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace xla {
namespace spmd {

namespace {
using hlo_sharding_util::GroupedSharding;
}  // namespace

Status SpmdPartitioningVisitor::HandleDot(HloInstruction* hlo) {
  DotConvDimsMapping mapping;
  const auto& dnums = hlo->dot_dimension_numbers();
  int64_t next_output_dim = 0;
  for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
    mapping.batch_dims.emplace_back();
    mapping.batch_dims.back().lhs = dnums.lhs_batch_dimensions(i);
    mapping.batch_dims.back().rhs = dnums.rhs_batch_dimensions(i);
    mapping.batch_dims.back().output = next_output_dim++;
  }
  for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
    mapping.contracting_dims.emplace_back();
    mapping.contracting_dims.back().lhs = dnums.lhs_contracting_dimensions(i);
    mapping.contracting_dims.back().rhs = dnums.rhs_contracting_dimensions(i);
    mapping.contracting_dims.back().output = -1;
  }
  for (int64_t i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    mapping.lhs_non_contracting_dims.emplace_back();
    mapping.lhs_non_contracting_dims.back().lhs = i;
    mapping.lhs_non_contracting_dims.back().rhs = -1;
    mapping.lhs_non_contracting_dims.back().output = next_output_dim++;
  }
  for (int64_t i = 0; i < hlo->operand(1)->shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    mapping.rhs_non_contracting_dims.emplace_back();
    mapping.rhs_non_contracting_dims.back().lhs = -1;
    mapping.rhs_non_contracting_dims.back().rhs = i;
    mapping.rhs_non_contracting_dims.back().output = next_output_dim++;
  }
  auto create_sharded_dot =
      [&](HloInstruction* l, HloInstruction* r, SpmdBuilder* b,
          const Window& conv_window) -> StatusOr<HloInstruction*> {
    TF_ASSIGN_OR_RETURN(
        auto sharded_dot_shape,
        ShapeInference::InferDotOpShape(
            l->shape(), r->shape(), hlo->dot_dimension_numbers(),
            /*preferred_element_type=*/hlo->shape().element_type()));
    return b->AddInstruction(HloInstruction::CreateDot(
        sharded_dot_shape, l, r, hlo->dot_dimension_numbers(),
        hlo->precision_config()));
  };
  return HandleDotHelper(hlo, mapping, create_sharded_dot);
}

namespace {

enum class WindowedEinsumOperand { LHS, RHS };

struct WindowedEinsumConfig {
  WindowedEinsumOperand windowed_op;
  bool windowed_at_contracting_dims;
  bool windowed_at_batch_dims;
  bool operands_sharded_at_contracting_dims;
};

struct DotDimensionIndexMapping {
  std::vector<int64_t> lhs_to_rhs_indices;
  std::vector<int64_t> lhs_to_output_indices;
  std::vector<int64_t> rhs_to_lhs_indices;
  std::vector<int64_t> rhs_to_output_indices;
  std::vector<int64_t> output_to_lhs_indices;
  std::vector<int64_t> output_to_rhs_indices;
};

void UpdateDDNums(DotDimensionNumbers* new_ddnums, int64_t reshaped_dim,
                  bool lhs) {
  auto update_dims =
      [&reshaped_dim](tensorflow::protobuf::RepeatedField<int64_t>* dims) {
        bool add_reshaped_dim = false;
        if (absl::c_linear_search(*dims, reshaped_dim)) {
          add_reshaped_dim = true;
        }
        for (int64_t i = 0; i < dims->size(); ++i) {
          auto dim = dims->at(i);
          if (reshaped_dim <= dim) {
            dims->Set(i, dim + 1);
          }
        }
        if (add_reshaped_dim) {
          dims->Add(reshaped_dim);
        }
      };

  if (lhs) {
    update_dims(new_ddnums->mutable_lhs_contracting_dimensions());
    update_dims(new_ddnums->mutable_lhs_batch_dimensions());
  } else {  // rhs
    update_dims(new_ddnums->mutable_rhs_contracting_dimensions());
    update_dims(new_ddnums->mutable_rhs_batch_dimensions());
  }
}

Window GenNewWindow(const HloInstruction* original_dot,
                    const HloInstruction* dot_lhs,
                    const HloInstruction* dot_rhs, int64_t lhs_concat_dim,
                    int64_t rhs_concat_dim, bool windowed_at_contracting_dims,
                    bool windowed_at_batch_dims) {
  auto new_window = original_dot->window();
  const ConvolutionDimensionNumbers& conv_dnums =
      original_dot->convolution_dimension_numbers();
  if (lhs_concat_dim != -1) {
    for (int64_t i = 0; i < conv_dnums.input_spatial_dimensions_size(); ++i) {
      if (conv_dnums.input_spatial_dimensions(i) == lhs_concat_dim) {
        auto wd = new_window.mutable_dimensions(i);
        auto lhs_size = dot_lhs->shape().dimensions(lhs_concat_dim + 1);
        if (windowed_at_contracting_dims) {
          wd->set_size(lhs_size);
        }
        if (windowed_at_batch_dims) {
          wd->set_size(lhs_size);
          wd->set_padding_low(0);
          wd->set_padding_high(0);
          wd->set_stride(std::max<int64_t>(1, lhs_size - 1));
          wd->set_window_dilation(1);
          wd->set_base_dilation(lhs_size);
          wd->set_window_reversal(false);
        }
      }
    }
  }
  if (rhs_concat_dim != -1) {
    for (int64_t i = 0; i < conv_dnums.kernel_spatial_dimensions_size(); ++i) {
      if (conv_dnums.kernel_spatial_dimensions(i) == rhs_concat_dim &&
          !windowed_at_contracting_dims && !windowed_at_batch_dims &&
          lhs_concat_dim == -1) {
        auto wd = new_window.mutable_dimensions(i);
        auto rhs_size = dot_rhs->shape().dimensions(rhs_concat_dim + 1);
        wd->set_size(rhs_size);
        wd->set_padding_low(rhs_size - 1);
        wd->set_padding_high(rhs_size - 1);
      }
    }
  }
  // Add the extra dimension to window.
  WindowDimension* new_dim = new_window.add_dimensions();
  if (windowed_at_contracting_dims) {
    new_dim->set_size(2);
    new_dim->set_padding_low(0);
    new_dim->set_padding_high(0);
    new_dim->set_stride(1);
    new_dim->set_window_dilation(1);
    new_dim->set_base_dilation(1);
    new_dim->set_window_reversal(false);
  } else if (windowed_at_batch_dims) {
    new_dim->set_size(2);
    new_dim->set_padding_low(0);
    new_dim->set_padding_high(0);
    new_dim->set_stride(1);  // std::max<int64_t>(1, 2 - 1)
    new_dim->set_window_dilation(1);
    new_dim->set_base_dilation(2);
    new_dim->set_window_reversal(false);
  } else {
    if (lhs_concat_dim != -1) {
      new_dim->set_size(1);
      new_dim->set_padding_low(0);
      new_dim->set_padding_high(0);
      new_dim->set_stride(1);
      new_dim->set_window_dilation(1);
      new_dim->set_base_dilation(1);
      new_dim->set_window_reversal(false);
    }
    if (rhs_concat_dim != -1) {
      new_dim->set_size(2);          // rhs_size
      new_dim->set_padding_low(1);   // rhs_size - 1
      new_dim->set_padding_high(1);  // rhs_size - 1
      new_dim->set_stride(1);
      new_dim->set_window_dilation(1);
      new_dim->set_base_dilation(1);
      new_dim->set_window_reversal(true);
    }
  }

  VLOG(2) << "new_window: " << new_window.ShortDebugString();
  return new_window;
}

ConvolutionDimensionNumbers GenNewConvDNums(
    const HloInstruction* original_dot, const HloInstruction* dot_lhs,
    const HloInstruction* dot_rhs, int64_t lhs_concat_dim,
    int64_t rhs_concat_dim, bool windowed_at_contracting_dims,
    bool windowed_at_batch_dims,
    const std::vector<int64_t>& lhs_to_output_indices,
    const std::vector<int64_t>& rhs_to_output_indices,
    const Shape& new_dot_shape) {
  // Generate the new conv dimension numbers.
  const ConvolutionDimensionNumbers& dnums =
      original_dot->convolution_dimension_numbers();
  // Handle the LHS dimension numbers.
  int64_t input_batch_dimension = dnums.input_batch_dimension();
  int64_t input_feature_dimension = dnums.input_feature_dimension();
  std::vector<int64_t> input_spatial_dimensions(
      dnums.input_spatial_dimensions().begin(),
      dnums.input_spatial_dimensions().end());
  if (lhs_concat_dim != -1) {
    if (lhs_concat_dim <= input_batch_dimension) {
      input_batch_dimension++;
    }
    if (lhs_concat_dim <= input_feature_dimension) {
      input_feature_dimension++;
    }
    for (int64_t i = 0; i < input_spatial_dimensions.size(); ++i) {
      if (lhs_concat_dim <= input_spatial_dimensions[i]) {
        input_spatial_dimensions[i]++;
      }
    }
    input_spatial_dimensions.push_back(lhs_concat_dim);
  }
  if (rhs_concat_dim != -1 && !windowed_at_contracting_dims &&
      !windowed_at_batch_dims) {
    input_spatial_dimensions.push_back(dot_lhs->shape().dimensions_size() - 1);
  }
  // Handle the RHS dimension numbers.
  int64_t kernel_input_feature_dimension =
      dnums.kernel_input_feature_dimension();
  int64_t kernel_output_feature_dimension =
      dnums.kernel_output_feature_dimension();
  std::vector<int64_t> kernel_spatial_dimensions(
      dnums.kernel_spatial_dimensions().begin(),
      dnums.kernel_spatial_dimensions().end());
  if (rhs_concat_dim != -1) {
    if (rhs_concat_dim <= kernel_input_feature_dimension) {
      kernel_input_feature_dimension++;
    }
    if (rhs_concat_dim <= kernel_output_feature_dimension) {
      kernel_output_feature_dimension++;
    }
    for (int64_t i = 0; i < kernel_spatial_dimensions.size(); ++i) {
      if (rhs_concat_dim <= kernel_spatial_dimensions[i]) {
        kernel_spatial_dimensions[i]++;
      }
    }
    kernel_spatial_dimensions.push_back(rhs_concat_dim);
  }
  if (lhs_concat_dim != -1 && !windowed_at_contracting_dims &&
      !windowed_at_batch_dims) {
    kernel_spatial_dimensions.push_back(dot_rhs->shape().dimensions_size() - 1);
  }
  // Handle the Output dimension numbers.
  int64_t output_batch_dimension = dnums.output_batch_dimension();
  int64_t output_feature_dimension = dnums.output_feature_dimension();
  std::vector<int64_t> output_spatial_dimensions(
      dnums.output_spatial_dimensions().begin(),
      dnums.output_spatial_dimensions().end());
  if (!windowed_at_contracting_dims) {
    auto output_slice_dim = lhs_concat_dim != -1
                                ? lhs_to_output_indices[lhs_concat_dim]
                                : rhs_to_output_indices[rhs_concat_dim];
    if (output_slice_dim <= output_batch_dimension) {
      output_batch_dimension++;
    }
    if (output_slice_dim <= output_feature_dimension) {
      output_feature_dimension++;
    }
    for (int64_t i = 0; i < output_spatial_dimensions.size(); ++i) {
      if (output_slice_dim <= output_spatial_dimensions[i]) {
        output_spatial_dimensions[i]++;
      }
    }
    output_spatial_dimensions.push_back(output_slice_dim);
  } else {
    output_spatial_dimensions.push_back(new_dot_shape.dimensions_size() - 1);
  }
  // Construct the new dot dimension numbers.
  ConvolutionDimensionNumbers new_dnums;
  new_dnums.set_input_batch_dimension(input_batch_dimension);
  new_dnums.set_input_feature_dimension(input_feature_dimension);
  for (auto dim : input_spatial_dimensions) {
    new_dnums.add_input_spatial_dimensions(dim);
  }
  new_dnums.set_kernel_input_feature_dimension(kernel_input_feature_dimension);
  new_dnums.set_kernel_output_feature_dimension(
      kernel_output_feature_dimension);
  for (auto dim : kernel_spatial_dimensions) {
    new_dnums.add_kernel_spatial_dimensions(dim);
  }
  new_dnums.set_output_batch_dimension(output_batch_dimension);
  new_dnums.set_output_feature_dimension(output_feature_dimension);
  for (auto dim : output_spatial_dimensions) {
    new_dnums.add_output_spatial_dimensions(dim);
  }

  return new_dnums;
}

DotDimensionIndexMapping ComputeDimensionIndexMapping(
    const DotConvDimsMapping& dims_mapping, int64_t lhs_rank, int64_t rhs_rank,
    int64_t output_rank) {
  std::vector<int64_t> lhs_to_rhs_indices(lhs_rank, -1);
  std::vector<int64_t> lhs_to_output_indices(lhs_rank, -1);
  std::vector<int64_t> rhs_to_lhs_indices(rhs_rank, -1);
  std::vector<int64_t> rhs_to_output_indices(rhs_rank, -1);
  std::vector<int64_t> output_to_lhs_indices(output_rank, -1);
  std::vector<int64_t> output_to_rhs_indices(output_rank, -1);
  auto populate_indices_mapping =
      [&](const DotConvDimsMapping::DimsMapping& mapping) {
        if (mapping.lhs >= 0) {
          lhs_to_rhs_indices[mapping.lhs] = mapping.rhs;
          lhs_to_output_indices[mapping.lhs] = mapping.output;
        }
        if (mapping.rhs >= 0) {
          rhs_to_lhs_indices[mapping.rhs] = mapping.lhs;
          rhs_to_output_indices[mapping.rhs] = mapping.output;
        }
        if (mapping.output >= 0) {
          output_to_lhs_indices[mapping.output] = mapping.lhs;
          output_to_rhs_indices[mapping.output] = mapping.rhs;
        }
      };
  for (const auto& mapping : dims_mapping.batch_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.contracting_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.lhs_non_contracting_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.rhs_non_contracting_dims) {
    populate_indices_mapping(mapping);
  }
  for (const auto& mapping : dims_mapping.conv_spatial_dims) {
    populate_indices_mapping(mapping);
  }
  return DotDimensionIndexMapping{lhs_to_rhs_indices,    lhs_to_output_indices,
                                  rhs_to_lhs_indices,    rhs_to_output_indices,
                                  output_to_lhs_indices, output_to_rhs_indices};
}

std::vector<std::vector<int64_t>> GetPartitionGroupsForReplication(
    const HloSharding& sharding, absl::Span<const int64_t> replication_dims) {
  int64_t group_size = 1;
  for (int64_t i : replication_dims) {
    group_size *= sharding.tile_assignment().dim(i);
  }
  std::vector<std::vector<int64_t>> partition_groups(
      sharding.tile_assignment().num_elements() / group_size);
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t partition) {
        int64_t group_id = 0;
        for (int64_t i = 0; i < indices.size(); ++i) {
          if (!absl::c_linear_search(replication_dims, i)) {
            group_id *= sharding.tile_assignment().dim(i);
            group_id += indices[i];
          }
        }
        partition_groups[group_id].push_back(partition);
      });
  return partition_groups;
}

absl::optional<WindowedEinsumConfig> GetWindowedEinsumConfiguration(
    int64_t num_partitions, int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions,
    int64_t rhs_contracting_partitions, int64_t rhs_non_contracting_partitions,
    int64_t rhs_batch_partitions, int64_t lhs_contracting_partitions,
    int64_t lhs_non_contracting_partitions, int64_t lhs_batch_partitions,
    int64_t rhs_shape_size, int64_t lhs_shape_size, int64_t output_shape_size,
    const SpmdPartitionerOptions& options,
    const absl::optional<HloSharding>& output_sharding_transposed_to_match_lhs,
    const absl::optional<HloSharding>& output_sharding_transposed_to_match_rhs,
    const HloSharding& lhs_sharding, const HloSharding& rhs_sharding,
    const Window& conv_window, const DotConvDimsMapping& dims_mapping,
    int64_t max_iterations = INT64_MAX,
    const HloInstruction* original_hlo = nullptr,
    PartitionedHlo* partitioned_lhs = nullptr,
    PartitionedHlo* partitioned_rhs = nullptr,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot = {},
    SpmdBuilder* b = nullptr, HloModule* module = nullptr,
    SpmdPartitioningVisitor* visitor = nullptr) {
  if (num_partitions > max_iterations) {
    return absl::nullopt;
  }

  const HloInstruction* lhs = nullptr;
  const HloInstruction* rhs = nullptr;
  if (original_hlo) {
    lhs = original_hlo->operand(0);
    rhs = original_hlo->operand(1);
  }

  // Determine if any of the users users have the same shardings that can allow
  // reuse of the resharding for the operand with original_hlo.
  auto check_users_sharding = [original_hlo](
                                  const HloInstruction* to_loop_over) {
    if (to_loop_over->users().size() <= 1) {
      return true;
    }
    constexpr int kAggressiveness = 3;
    absl::optional<HloSharding> original_ideal_sharding =
        ShardingPropagation::GetShardingFromUser(*to_loop_over, *original_hlo,
                                                 kAggressiveness,
                                                 /*is_spmd=*/true);
    // Default to perform collective matmul if GetShardingFromUser() couldn't
    // determine the sharding.
    if (!original_ideal_sharding) {
      return true;
    }
    for (const HloInstruction* user : to_loop_over->users()) {
      if (user == original_hlo) {
        continue;
      }
      absl::optional<HloSharding> from_user =
          ShardingPropagation::GetShardingFromUser(*to_loop_over, *user,
                                                   kAggressiveness,
                                                   /*is_spmd=*/true);
      // Could't determine sharding. Skip to next one and pretend it wouldn't
      // share the resharding.
      if (!from_user) {
        continue;
      }
      // This user doesn't require resharding, so even if has different sharding
      // than original_hlo its ok to do collective matmul.
      if (*from_user == to_loop_over->sharding()) {
        continue;
      }
      // Same sharding needed, so we would share the resharding. Do not do
      // collective matmul.
      if (*original_ideal_sharding == *from_user) {
        return false;
      }
    }
    return true;
  };

  // Disable windowed einsum when the overheads may overweigh the benefits.
  // Specifically, when max(computation time, communication time after
  // decomposition) + extra prologue or epilogue collecitve permute is longer
  // than the sum of computation time and the original communication time
  // which can use more communication links. This is checked with the premise
  // that communication/computation is large enough. For super small
  // communication/computation generated by unit tests, we always allow windowed
  // einsum to have meaningful unit tests.
  auto disable_windowed_einsum = [&](bool lhs_needs_ag, bool rhs_needs_ag) {
    if (visitor == nullptr) {
      return false;
    }

    double computation_time_in_ms = 0.0;
    double communication_time_in_ms = 0.0;
    HloInstruction* dot;
    HloInstruction* collective;
    if (lhs_needs_ag || rhs_needs_ag) {
      CHECK(!lhs_needs_ag || !rhs_needs_ag);
      auto new_lhs = lhs_needs_ag
                         ? PartitionedHlo(partitioned_lhs->hlo(),
                                          partitioned_lhs->base_shape(),
                                          partitioned_lhs->state())
                               .Reshard(HloSharding::Replicate())
                         : *partitioned_lhs;
      auto new_rhs = rhs_needs_ag
                         ? PartitionedHlo(partitioned_rhs->hlo(),
                                          partitioned_rhs->base_shape(),
                                          partitioned_rhs->state())
                               .Reshard(HloSharding::Replicate())
                         : *partitioned_rhs;
      dot = create_sharded_dot(new_lhs.hlo(), new_rhs.hlo(), b, conv_window)
                .ValueOrDie();
      computation_time_in_ms = visitor->GetComputationTimeInMilliSec(dot);

      collective = lhs_needs_ag ? new_lhs.hlo() : new_rhs.hlo();
      while (collective->opcode() != HloOpcode::kAllGather &&
             collective->opcode() != HloOpcode::kAllReduce &&
             collective->operand_count() > 0 &&
             collective != (lhs_needs_ag ? partitioned_lhs->hlo()
                                         : partitioned_rhs->hlo())) {
        collective = collective->mutable_operand(0);
      }
      if (collective->opcode() == HloOpcode::kAllGather ||
          collective->opcode() == HloOpcode::kAllReduce) {
        communication_time_in_ms = visitor->GetCommunicationTimeInMilliSec(
            ShapeUtil::ByteSizeOf(collective->shape()),
            collective->replica_groups());
      }
    } else {
      auto new_lhs =
          PartitionedHlo(partitioned_lhs->hlo(), partitioned_lhs->base_shape(),
                         partitioned_lhs->state())
              .PadWithZero();
      auto new_rhs =
          PartitionedHlo(partitioned_rhs->hlo(), partitioned_rhs->base_shape(),
                         partitioned_rhs->state())
              .PadWithZero();
      dot = create_sharded_dot(new_lhs.hlo(), new_rhs.hlo(), b, conv_window)
                .ValueOrDie();
      computation_time_in_ms = visitor->GetComputationTimeInMilliSec(dot);

      std::vector<int64_t> lhs_contracting_dims;
      lhs_contracting_dims.reserve(new_lhs.base_shape().rank());
      for (const auto& cd : dims_mapping.contracting_dims) {
        lhs_contracting_dims.push_back(cd.lhs);
      }
      collective = new_lhs.state().partitioner->AllReduceAlongShardingDims(
          b, dot, new_lhs.sharding(), new_lhs.state().next_channel_id,
          lhs_contracting_dims, new_lhs.state().collective_ops_creator,
          MakeBinaryAdd(dot->shape().element_type(), module));
      communication_time_in_ms = visitor->GetCommunicationTimeInMilliSec(
          ShapeUtil::ByteSizeOf(dot->shape()), collective->replica_groups());
    }

    VLOG(2) << "collective: " << collective->ToString() << "\n"
            << "dot: " << dot->ToString() << "\n"
            << "num_partitions: " << num_partitions << "\n"
            << "computation_time_in_ms: " << computation_time_in_ms
            << " communication_time_in_ms: " << communication_time_in_ms;
    double extra_collective_permute_time = 0.0;
    if (communication_time_in_ms != 0.0) {
      extra_collective_permute_time =
          communication_time_in_ms *
          visitor->GetCommunicationMultiplier(collective->replica_groups()) *
          2 / num_partitions;
    }
    if (communication_time_in_ms > 1e-5 &&
        (std::max(
             computation_time_in_ms,
             communication_time_in_ms * visitor->GetCommunicationMultiplier(
                                            collective->replica_groups())) +
         extra_collective_permute_time) >=
            (computation_time_in_ms + communication_time_in_ms)) {
      return true;
    } else {
      return false;
    }
  };

  if (output_lhs_non_contracting_partitions == num_partitions &&
      output_sharding_transposed_to_match_lhs == lhs_sharding &&
      rhs_shape_size >=
          options.threshold_for_windowed_einsum_mib * 1024 * 1024 &&
      (!rhs || check_users_sharding(rhs)) &&
      !disable_windowed_einsum(/*lhs_needs_ag=*/false, /*rhs_needs_ag=*/true)) {
    if (rhs_contracting_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::RHS,
          /*windowed_at_contracting_dims*/ true,
          /*windowed_at_batch_dims=*/false,
          /*operands_sharded_at_contracting_dims=*/false};
    }
    if (rhs_non_contracting_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::RHS,
          /*windowed_at_contracting_dims*/ false,
          /*windowed_at_batch_dims=*/false,
          /*operands_sharded_at_contracting_dims=*/false};
    }
    if (rhs_batch_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::RHS,
          /*windowed_at_contracting_dims*/ false,
          /*windowed_at_batch_dims=*/true,
          /*operands_sharded_at_contracting_dims=*/false};
    }
  }
  if (output_rhs_non_contracting_partitions == num_partitions &&
      output_sharding_transposed_to_match_rhs == rhs_sharding &&
      lhs_shape_size >=
          options.threshold_for_windowed_einsum_mib * 1024 * 1024 &&
      (!lhs || check_users_sharding(lhs)) &&
      !disable_windowed_einsum(/*lhs_needs_ag=*/true, /*rhs_needs_ag=*/false)) {
    if (lhs_contracting_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::LHS,
          /*windowed_at_contracting_dims*/ true,
          /*windowed_at_batch_dims=*/false,
          /*operands_sharded_at_contracting_dims=*/false};
    }
    if (lhs_non_contracting_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::LHS,
          /*windowed_at_contracting_dims*/ false,
          /*windowed_at_batch_dims=*/false,
          /*operands_sharded_at_contracting_dims=*/false};
    }
    if (lhs_batch_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::LHS,
          /*windowed_at_contracting_dims*/ false,
          /*windowed_at_batch_dims=*/true,
          /*operands_sharded_at_contracting_dims=*/false};
    }
  }
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions == num_partitions &&
      (output_lhs_non_contracting_partitions == num_partitions ||
       output_rhs_non_contracting_partitions == num_partitions) &&
      output_shape_size >=
          options.threshold_for_windowed_einsum_mib * 1024 * 1024 &&
      !disable_windowed_einsum(/*lhs_needs_ag=*/false,
                               /*rhs_needs_ag=*/false)) {
    if (output_lhs_non_contracting_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::RHS,
          /*windowed_at_contracting_dims*/ false,
          /*windowed_at_batch_dims=*/false,
          /*operands_sharded_at_contracting_dims=*/true};
    }
    if (output_rhs_non_contracting_partitions == num_partitions) {
      return WindowedEinsumConfig{
          /*windowed_op=*/WindowedEinsumOperand::LHS,
          /*windowed_at_contracting_dims*/ false,
          /*windowed_at_batch_dims=*/false,
          /*operands_sharded_at_contracting_dims=*/true};
    }
  }
  return absl::nullopt;
}

std::vector<ReplicaGroup> GetLoopReplicaGroups(HloInstruction* while_loop) {
  std::vector<ReplicaGroup> groups;
  for (auto inst : while_loop->while_body()->instructions()) {
    if (inst->opcode() == HloOpcode::kCollectivePermute) {
      std::vector<std::pair<int64_t, int64_t>> st_pairs =
          inst->source_target_pairs();
      std::vector<int64_t> source_index(st_pairs.size());
      for (int64_t i = 0; i < st_pairs.size(); ++i) {
        source_index[st_pairs[i].first] = i;
      }

      absl::flat_hash_set<int64_t> visited;
      for (int64_t i = 0; i < st_pairs.size(); ++i) {
        if (visited.contains(st_pairs[i].first)) {
          continue;
        }
        std::vector<int64_t> replica_group;
        int64_t source = st_pairs[i].first;
        int64_t target = st_pairs[i].second;
        replica_group.push_back(source);
        replica_group.push_back(target);
        visited.insert(source);
        visited.insert(target);
        while (target != source) {
          target = st_pairs[source_index[target]].second;
          if (target != source) {
            replica_group.push_back(target);
            visited.insert(target);
          }
        }
        absl::c_sort(replica_group);
        groups.emplace_back();
        for (auto id : replica_group) {
          groups.back().add_replica_ids(id);
        }
      }

      VLOG(3) << "while loop: " << while_loop->name()
              << ", replica groups: " << ReplicaGroupsToString(groups);
      break;
    }
  }
  return groups;
}

// We use a recursive approach where sets of matching dimensions are recognized
// one at a time. The base shapes and shardings can be changed during the
// recursion as we group devices together. So refer to the passed in shapes and
// shardings for inputs and output, and do not use shape inference.

StatusOr<HloInstruction*> PartitionBaseCase(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64_t num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    int64_t lhs_batch_partitions, int64_t rhs_batch_partitions,
    int64_t output_batch_partitions, int64_t lhs_contracting_partitions,
    int64_t rhs_contracting_partitions, int64_t lhs_non_contracting_partitions,
    int64_t rhs_non_contracting_partitions,
    int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    bool may_reshard_without_detecting_match,
    SpmdPartitioningVisitor* visitor) {
  const HloSharding& lhs_sharding = lhs.sharding();
  const HloSharding& rhs_sharding = rhs.sharding();
  if (lhs_sharding.ReplicateOnLastTileDim() ||
      rhs_sharding.ReplicateOnLastTileDim() ||
      output_sharding.ReplicateOnLastTileDim()) {
    return nullptr;
  }
  DotDimensionIndexMapping indices_map = ComputeDimensionIndexMapping(
      dims_mapping, lhs.base_shape().rank(), rhs.base_shape().rank(),
      output_base_shape.rank());
  auto lhs_sharding_transposed_to_match_rhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          lhs_sharding, indices_map.lhs_to_rhs_indices,
          indices_map.rhs_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_lhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          rhs_sharding, indices_map.rhs_to_lhs_indices,
          indices_map.lhs_to_rhs_indices);
  auto lhs_sharding_transposed_to_match_output =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          lhs_sharding, indices_map.lhs_to_output_indices,
          indices_map.output_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_output =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          rhs_sharding, indices_map.rhs_to_output_indices,
          indices_map.output_to_rhs_indices);
  auto output_sharding_transposed_to_match_lhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          output_sharding, indices_map.output_to_lhs_indices,
          indices_map.lhs_to_output_indices);
  auto output_sharding_transposed_to_match_rhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          output_sharding, indices_map.output_to_rhs_indices,
          indices_map.rhs_to_output_indices);

  // LHS and RHS are partitioned the same way and only partitioned in batch
  // dimensions.
  if (lhs_batch_partitions == rhs_batch_partitions &&
      rhs_batch_partitions == num_partitions &&
      lhs_sharding_transposed_to_match_rhs == rhs_sharding) {
    TF_ASSIGN_OR_RETURN(
        auto dot, create_sharded_dot(lhs.hlo(), rhs.hlo(), b, conv_window));
    dot->set_sharding(*lhs_sharding_transposed_to_match_output);
    return PartitionedHlo(dot, output_base_shape, lhs.state())
        .Reshard(output_sharding)
        .hlo();
  }

  // Try emit batch-partitioned einsum with one operand resharded. Returns
  // partitioned HLO or nullptr if the attempt fails. If
  // may_reshard_with_allreduce is false, reshard must be done using
  // all-to-all/collective-permute; otherwise this attempt fails.
  auto try_emit_output_batch_partitioned_einsum_with_reshard =
      [&](bool may_reshard_with_allreduce) -> StatusOr<HloInstruction*> {
    // LHS and output are batch partitioned in the same way.
    if (lhs_batch_partitions == num_partitions &&
        output_batch_partitions == num_partitions &&
        lhs_sharding_transposed_to_match_output == output_sharding) {
      if (!may_reshard_with_allreduce &&
          !CanReshardWithCollectivePermute(
              rhs.sharding(), *lhs_sharding_transposed_to_match_rhs) &&
          !GetReshardAllToAllSourceTargetDims(
              rhs.sharding(), *lhs_sharding_transposed_to_match_rhs)) {
        return nullptr;
      }
      auto resharded_rhs = rhs.Reshard(*lhs_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot,
          create_sharded_dot(lhs.hlo(), resharded_rhs.hlo(), b, conv_window));
      return dot;
    }
    // RHS and output are batch partitioned in the same way.
    if (rhs_batch_partitions == num_partitions &&
        output_batch_partitions == num_partitions &&
        rhs_sharding_transposed_to_match_output == output_sharding) {
      if (!may_reshard_with_allreduce &&
          !CanReshardWithCollectivePermute(
              lhs.sharding(), *rhs_sharding_transposed_to_match_lhs) &&
          !GetReshardAllToAllSourceTargetDims(
              lhs.sharding(), *rhs_sharding_transposed_to_match_lhs)) {
        return nullptr;
      }
      auto resharded_lhs = lhs.Reshard(*rhs_sharding_transposed_to_match_lhs);
      TF_ASSIGN_OR_RETURN(
          auto dot,
          create_sharded_dot(resharded_lhs.hlo(), rhs.hlo(), b, conv_window));
      return dot;
    }
    return nullptr;
  };

  {
    // Try batch-parallel by resharding one operand, and not using all-reduce.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * partitioned_dot,
        try_emit_output_batch_partitioned_einsum_with_reshard(false));
    if (partitioned_dot) {
      return partitioned_dot;
    }
  }

  // Try to emit windowed DotGeneral when one operand is partitioned in the same
  // way as the output along non-contracting dimensions, but the other operand
  // is tiled in other dimensions. Or both operands are partitioned in the same
  // way along contracting dimensions, but the output is partitioned along
  // non-contracting dimensions.
  auto emit_windowed_dot_general =
      [&](const WindowedEinsumConfig& einsum_config)
      -> StatusOr<HloInstruction*> {
    CHECK(!einsum_config.windowed_at_batch_dims ||
          !einsum_config.windowed_at_contracting_dims);
    const bool windowed_at_batch_dims = einsum_config.windowed_at_batch_dims;
    const bool windowed_at_contracting_dims =
        einsum_config.windowed_at_contracting_dims;
    const bool operands_sharded_at_contracting_dims =
        einsum_config.operands_sharded_at_contracting_dims;
    auto unpadded_result_buffer_shape =
        MakePartitionedShape(output_base_shape, output_sharding);
    auto padded_result_buffer_shape = unpadded_result_buffer_shape;
    const bool windowed_op_is_lhs =
        einsum_config.windowed_op == WindowedEinsumOperand::LHS;
    // For windowing at batch/non-contracting dims, we produce the result one
    // partition at a time, so we need to pad the shape in case of uneven
    // partitioning in order to make dynamic-update-slice in-bound.
    if (!windowed_at_contracting_dims &&
        !operands_sharded_at_contracting_dims) {
      padded_result_buffer_shape = GetPaddedShapeForUnevenPartitioning(
          padded_result_buffer_shape,
          windowed_op_is_lhs ? *lhs_sharding_transposed_to_match_output
                             : *rhs_sharding_transposed_to_match_output);
    }
    // Mask the padding area of the windowed operand with zero if there is
    // uneven partitioning.
    if (windowed_at_contracting_dims) {
      auto& to_mask = windowed_op_is_lhs ? lhs : rhs;
      to_mask = to_mask.PadWithZero();
    }
    if (operands_sharded_at_contracting_dims) {
      lhs = lhs.PadWithZero();
      rhs = rhs.PadWithZero();
    }

    // Get slice sharding, sharding dim, and lhs/rhs concat dim.
    const HloSharding* slice_sharding;
    if (operands_sharded_at_contracting_dims) {
      slice_sharding = windowed_op_is_lhs
                           ? &*output_sharding_transposed_to_match_rhs
                           : &*output_sharding_transposed_to_match_lhs;
    } else if (windowed_at_contracting_dims || windowed_at_batch_dims) {
      slice_sharding = windowed_op_is_lhs
                           ? &*lhs_sharding_transposed_to_match_rhs
                           : &*rhs_sharding_transposed_to_match_lhs;
    } else {
      slice_sharding = windowed_op_is_lhs
                           ? &*lhs_sharding_transposed_to_match_output
                           : &*rhs_sharding_transposed_to_match_output;
    }
    CHECK_EQ(Product(slice_sharding->tile_assignment().dimensions()),
             num_partitions);
    int64_t slice_sharding_dim = -1;
    for (int64_t i = 0; i < slice_sharding->tile_assignment().num_dimensions();
         ++i) {
      if (slice_sharding->tile_assignment().dim(i) > 1) {
        slice_sharding_dim = i;
        break;
      }
    }
    int64_t lhs_concat_dim = -1;
    int64_t rhs_concat_dim = -1;
    if (operands_sharded_at_contracting_dims) {
      if (windowed_op_is_lhs) {
        rhs_concat_dim = slice_sharding_dim;
      } else {
        lhs_concat_dim = slice_sharding_dim;
      }
    } else if (windowed_at_contracting_dims || windowed_at_batch_dims) {
      lhs_concat_dim = windowed_op_is_lhs
                           ? indices_map.rhs_to_lhs_indices[slice_sharding_dim]
                           : slice_sharding_dim;
      rhs_concat_dim = windowed_op_is_lhs
                           ? slice_sharding_dim
                           : indices_map.lhs_to_rhs_indices[slice_sharding_dim];
    } else {
      if (windowed_op_is_lhs) {
        lhs_concat_dim = indices_map.output_to_lhs_indices[slice_sharding_dim];
      } else {
        rhs_concat_dim = indices_map.output_to_rhs_indices[slice_sharding_dim];
      }
    }

    auto lhs_hlo = lhs.hlo();
    auto rhs_hlo = rhs.hlo();
    // Reshape lhs and rhs before the loop for bidirectional communication case.
    if (options.bidirectional_windowed_einsum && num_partitions % 4 == 0) {
      if (lhs_concat_dim != -1 && windowed_op_is_lhs &&
          !operands_sharded_at_contracting_dims) {
        std::vector<int64_t> reshaped_dims(
            lhs_hlo->shape().dimensions().begin(),
            lhs_hlo->shape().dimensions().end());
        reshaped_dims.insert(reshaped_dims.begin() + lhs_concat_dim, 1);
        lhs_hlo = b->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(lhs_hlo->shape().element_type(),
                                 reshaped_dims),
            lhs_hlo));
      }
      if (rhs_concat_dim != -1 && !windowed_op_is_lhs &&
          !operands_sharded_at_contracting_dims) {
        std::vector<int64_t> reshaped_dims(
            rhs_hlo->shape().dimensions().begin(),
            rhs_hlo->shape().dimensions().end());
        reshaped_dims.insert(reshaped_dims.begin() + rhs_concat_dim, 1);
        rhs_hlo = b->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(rhs_hlo->shape().element_type(),
                                 reshaped_dims),
            rhs_hlo));
      }
    }

    auto result_buffer = CreateZero(padded_result_buffer_shape, b);
    auto extra_buffer =
        (!(options.bidirectional_windowed_einsum && num_partitions % 4 == 0) ||
         operands_sharded_at_contracting_dims)
            ? CreateZero(padded_result_buffer_shape, b)
        : windowed_op_is_lhs ? lhs_hlo
                             : rhs_hlo;

    if (options.bidirectional_windowed_einsum && num_partitions % 4 == 0 &&
        !operands_sharded_at_contracting_dims) {
      std::vector<std::pair<int64_t, int64_t>> pre_sd_pairs(num_partitions);
      for (int64_t source = 0; source < num_partitions; ++source) {
        // 0 -> 1, 1 -> 2, 2 -> 3, ...
        pre_sd_pairs[source] = {source, (source + 1) % num_partitions};
      }
      extra_buffer =
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  b, extra_buffer, pre_sd_pairs,
                  (*lhs.state().next_channel_id)++);
    }

    auto iteration = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(0)));

    // Create a while loop that computes one window per iteration. During each
    // iteration, each partition sends its input window to its neighbor using
    // collective-permute for the next iteration.
    SpmdBuilder body_b("windowed_dot_general_body", original_hlo);

    // Generate partial results used by bidirectional algorithm.
    auto get_partial_bid_results =
        [&](HloInstruction* l, HloInstruction* r, HloInstruction* o,
            HloInstruction* extra_inout, HloInstruction* cw_cp_output,
            HloInstruction* i) -> StatusOr<std::vector<HloInstruction*>> {
      auto partition_id =
          lhs.state().collective_ops_creator.create_partition_id(&body_b);
      auto partition_count =
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32_t>(num_partitions)));
      auto ccw_data_partition_id =
          body_b.AddInstruction(HloInstruction::CreateBinary(
              i->shape(), HloOpcode::kAdd, i, partition_id));
      auto cw_data_partition_id =
          body_b.AddInstruction(HloInstruction::CreateBinary(
              i->shape(), HloOpcode::kAdd, partition_count, partition_id));
      if (operands_sharded_at_contracting_dims) {
        ccw_data_partition_id =
            body_b.AddInstruction(HloInstruction::CreateBinary(
                i->shape(), HloOpcode::kAdd, ccw_data_partition_id,
                body_b.AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<uint32_t>(num_partitions / 2 + 1)))));
        cw_data_partition_id =
            body_b.AddInstruction(HloInstruction::CreateBinary(
                i->shape(), HloOpcode::kSubtract, cw_data_partition_id,
                body_b.AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<uint32_t>(num_partitions / 2)))));
      } else {
        cw_data_partition_id =
            body_b.AddInstruction(HloInstruction::CreateBinary(
                i->shape(), HloOpcode::kSubtract, cw_data_partition_id,
                CreateOne(cw_data_partition_id->shape(), &body_b)));
      }
      ccw_data_partition_id = body_b.AddInstruction(
          HloInstruction::CreateBinary(i->shape(), HloOpcode::kRemainder,
                                       ccw_data_partition_id, partition_count));
      cw_data_partition_id = body_b.AddInstruction(HloInstruction::CreateBinary(
          i->shape(), HloOpcode::kSubtract, cw_data_partition_id, i));
      cw_data_partition_id = body_b.AddInstruction(
          HloInstruction::CreateBinary(i->shape(), HloOpcode::kRemainder,
                                       cw_data_partition_id, partition_count));

      DotDimensionNumbers new_ddnums;
      if (original_hlo->opcode() == HloOpcode::kDot) {
        new_ddnums = original_hlo->dot_dimension_numbers();
      }

      auto dot_lhs = l;
      auto dot_rhs = r;
      auto original_dot_lhs = l;
      auto original_dot_rhs = r;
      // Recover original lhs and rhs, will not be used in real computation.
      if (lhs_concat_dim != -1 && windowed_op_is_lhs) {
        std::vector<int64_t> reshaped_dims(
            original_dot_lhs->shape().dimensions().begin(),
            original_dot_lhs->shape().dimensions().end());
        reshaped_dims.erase(reshaped_dims.begin() + lhs_concat_dim);
        original_dot_lhs = body_b.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(original_dot_lhs->shape().element_type(),
                                 reshaped_dims),
            original_dot_lhs));
      }
      if (rhs_concat_dim != -1 && !windowed_op_is_lhs) {
        std::vector<int64_t> reshaped_dims(
            original_dot_rhs->shape().dimensions().begin(),
            original_dot_rhs->shape().dimensions().end());
        reshaped_dims.erase(reshaped_dims.begin() + rhs_concat_dim);
        original_dot_rhs = body_b.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(original_dot_rhs->shape().element_type(),
                                 reshaped_dims),
            original_dot_rhs));
      }

      if (windowed_at_contracting_dims || windowed_at_batch_dims ||
          operands_sharded_at_contracting_dims) {
        // Slice the matching operand according to the partitioned dimensions
        // on the windowed operand or the output.
        auto slice_operand = !windowed_op_is_lhs ? l : r;

        // Pad the sharding dim first (then the concat dim) for correctness.
        auto sharding_dim_size =
            slice_operand->shape().dimensions(slice_sharding_dim);
        if (sharding_dim_size % num_partitions != 0) {
          slice_operand = PadBaseShapeBeforeUnevenTiledSharding(
              slice_operand, *slice_sharding, &body_b);
        }

        // We do this by treating the matching operand as replicated, and
        // resharding it to match the windowed operand or the output.
        auto gen_slice = [&](HloInstruction* data_partition_id,
                             bool ccw) -> HloInstruction* {
          std::vector<int64_t> new_dims;
          const int64_t dimensions_size =
              slice_operand->shape().dimensions_size();
          new_dims.reserve(dimensions_size + 1);
          for (int64_t i = 0; i < dimensions_size; ++i) {
            if (i == slice_sharding_dim) {
              new_dims.push_back(1);
            }
            new_dims.push_back(slice_operand->shape().dimensions(i));
          }
          auto reshaped_slice_operand =
              body_b.AddInstruction(HloInstruction::CreateReshape(
                  ShapeUtil::MakeShape(slice_operand->shape().element_type(),
                                       new_dims),
                  slice_operand));
          auto min = body_b.AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::MinValue(
                  reshaped_slice_operand->shape().element_type())));
          std::vector<int64_t> min_padding(
              reshaped_slice_operand->shape().rank());
          auto padded_slice_operand = reshaped_slice_operand;
          auto padded_shape = padded_slice_operand->shape();
          int64_t padding_dim = slice_sharding_dim;
          padded_shape.set_dimensions(padding_dim, 2);
          if (ccw) {
            // ccw pad high
            PaddingConfig ccw_pad_config =
                window_util::MakeSymmetricPadding(min_padding);
            ccw_pad_config.mutable_dimensions(padding_dim)
                ->set_edge_padding_low(0);
            ccw_pad_config.mutable_dimensions(padding_dim)
                ->set_edge_padding_high(1);
            padded_slice_operand =
                body_b.AddInstruction(HloInstruction::CreatePad(
                    padded_shape, padded_slice_operand, min, ccw_pad_config));
          } else {
            // cw pad low
            PaddingConfig cw_pad_config =
                window_util::MakeSymmetricPadding(min_padding);
            cw_pad_config.mutable_dimensions(padding_dim)
                ->set_edge_padding_low(1);
            cw_pad_config.mutable_dimensions(padding_dim)
                ->set_edge_padding_high(0);
            padded_slice_operand =
                body_b.AddInstruction(HloInstruction::CreatePad(
                    padded_shape, padded_slice_operand, min, cw_pad_config));
          }

          padded_slice_operand->set_sharding(HloSharding::Replicate());
          auto state = lhs.state();
          state.b = &body_b;
          state.partition_id = data_partition_id;
          state.reshard_cache->per_hlo_cache.erase(padded_slice_operand);
          auto padded_slice_sharding = hlo_sharding_util::ReshapeSharding(
              slice_operand->shape(), reshaped_slice_operand->shape(),
              *slice_sharding);
          auto padded_slice =
              PartitionedHlo(padded_slice_operand,
                             padded_slice_operand->shape(), state)
                  .Reshard(*padded_slice_sharding)
                  .hlo();
          padded_slice_operand->clear_sharding();
          return padded_slice;
        };

        auto ccw_slice = gen_slice(ccw_data_partition_id, true);
        auto cw_slice = gen_slice(cw_data_partition_id, false);
        auto slice = body_b.AddInstruction(HloInstruction::CreateBinary(
            ccw_slice->shape(), HloOpcode::kMaximum, ccw_slice, cw_slice));
        // Reshape. The reshaped slice will not be used to produce the final
        // result, but used as a hint for the shape inference.
        std::vector<int64_t> reshaped_slice_dims;
        const int64_t dim_size = slice->shape().dimensions_size();
        reshaped_slice_dims.reserve(dim_size);
        for (int64_t i = 0; i < dim_size; ++i) {
          auto dim_size = slice->shape().dimensions(i);
          if (i == (slice_sharding_dim + 1)) {
            reshaped_slice_dims.push_back(dim_size * 2);
          } else if (i != slice_sharding_dim) {
            reshaped_slice_dims.push_back(dim_size);
          }
        }
        auto reshaped_slice =
            body_b.AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::MakeShape(slice->shape().element_type(),
                                     reshaped_slice_dims),
                slice));

        if (!windowed_op_is_lhs) {
          dot_lhs = slice;
          original_dot_lhs = reshaped_slice;
          if (original_hlo->opcode() == HloOpcode::kDot) {
            UpdateDDNums(&new_ddnums, slice_sharding_dim, true);
          }
        } else {
          dot_rhs = slice;
          original_dot_rhs = reshaped_slice;
          if (original_hlo->opcode() == HloOpcode::kDot) {
            UpdateDDNums(&new_ddnums, slice_sharding_dim, false);
          }
        }
      }

      auto ccw_dot_lhs = l;
      auto ccw_dot_rhs = r;
      auto cw_dot_lhs = windowed_op_is_lhs ? extra_inout : l;
      auto cw_dot_rhs = windowed_op_is_lhs ? r : extra_inout;
      if (lhs_concat_dim != -1 && windowed_op_is_lhs) {
        // Concat
        auto lhs_concat_shape = ccw_dot_lhs->shape();
        lhs_concat_shape.set_dimensions(lhs_concat_dim, 2);
        dot_lhs = body_b.AddInstruction(HloInstruction::CreateConcatenate(
            lhs_concat_shape, {ccw_dot_lhs, cw_dot_lhs}, lhs_concat_dim));

        std::vector<int64_t> reshaped_dims(
            ccw_dot_lhs->shape().dimensions().begin(),
            ccw_dot_lhs->shape().dimensions().end());
        reshaped_dims.erase(reshaped_dims.begin() + lhs_concat_dim);
        reshaped_dims[lhs_concat_dim] *= 2;
        original_dot_lhs = body_b.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(dot_lhs->shape().element_type(),
                                 reshaped_dims),
            dot_lhs));

        if (original_hlo->opcode() == HloOpcode::kDot) {
          UpdateDDNums(&new_ddnums, lhs_concat_dim, true);
        }
      }
      if (rhs_concat_dim != -1 && !windowed_op_is_lhs) {
        // Concat
        auto rhs_concat_shape = ccw_dot_rhs->shape();
        rhs_concat_shape.set_dimensions(rhs_concat_dim, 2);
        dot_rhs = body_b.AddInstruction(HloInstruction::CreateConcatenate(
            rhs_concat_shape, {ccw_dot_rhs, cw_dot_rhs}, rhs_concat_dim));

        std::vector<int64_t> reshaped_dims(
            ccw_dot_rhs->shape().dimensions().begin(),
            ccw_dot_rhs->shape().dimensions().end());
        reshaped_dims.erase(reshaped_dims.begin() + rhs_concat_dim);
        reshaped_dims[rhs_concat_dim] *= 2;
        original_dot_rhs = body_b.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(dot_rhs->shape().element_type(),
                                 reshaped_dims),
            dot_rhs));

        if (original_hlo->opcode() == HloOpcode::kDot) {
          UpdateDDNums(&new_ddnums, rhs_concat_dim, false);
        }
      }

      // The generated original dot will not be used.
      TF_ASSIGN_OR_RETURN(auto original_dot,
                          create_sharded_dot(original_dot_lhs, original_dot_rhs,
                                             &body_b, conv_window));
      VLOG(2) << original_dot->ToString();

      // Generate the correct shape of the new dot/conv.
      auto original_sharded_dot_shape = original_dot->shape();
      auto new_dot_shape = original_sharded_dot_shape;
      std::vector<int64_t> new_dims(new_dot_shape.dimensions().begin(),
                                    new_dot_shape.dimensions().end());
      if (!windowed_at_contracting_dims) {
        auto slice_dim =
            lhs_concat_dim != -1
                ? indices_map.lhs_to_output_indices[lhs_concat_dim]
                : indices_map.rhs_to_output_indices[rhs_concat_dim];
        new_dims[slice_dim] /= 2;
        new_dims.insert(new_dims.begin() + slice_dim, 2);
      } else if (original_hlo->opcode() != HloOpcode::kDot) {
        new_dims.push_back(1);
      }
      new_dot_shape =
          ShapeUtil::MakeShape(original_hlo->shape().element_type(), new_dims);

      HloInstruction* dot;
      if (original_hlo->opcode() == HloOpcode::kDot) {
        dot = body_b.AddInstruction(HloInstruction::CreateDot(
            new_dot_shape, dot_lhs, dot_rhs, new_ddnums,
            original_hlo->precision_config()));
      } else {
        if (!windowed_at_contracting_dims && !windowed_at_batch_dims) {
          if (lhs_concat_dim != -1) {
            std::vector<int64_t> new_dims(dot_rhs->shape().dimensions().begin(),
                                          dot_rhs->shape().dimensions().end());
            new_dims.push_back(1);
            dot_rhs = body_b.AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::MakeShape(dot_rhs->shape().element_type(), new_dims),
                dot_rhs));
          }
          if (rhs_concat_dim != -1) {
            std::vector<int64_t> new_dims(dot_lhs->shape().dimensions().begin(),
                                          dot_lhs->shape().dimensions().end());
            new_dims.push_back(1);
            dot_lhs = body_b.AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::MakeShape(dot_lhs->shape().element_type(), new_dims),
                dot_lhs));
          }
        }

        dot = body_b.AddInstruction(HloInstruction::CreateConvolve(
            new_dot_shape, dot_lhs, dot_rhs,
            original_dot->feature_group_count(),
            original_dot->batch_group_count(),
            GenNewWindow(original_dot, dot_lhs, dot_rhs, lhs_concat_dim,
                         rhs_concat_dim, windowed_at_contracting_dims,
                         windowed_at_batch_dims),
            GenNewConvDNums(original_dot, dot_lhs, dot_rhs, lhs_concat_dim,
                            rhs_concat_dim, windowed_at_contracting_dims,
                            windowed_at_batch_dims,
                            indices_map.lhs_to_output_indices,
                            indices_map.rhs_to_output_indices, new_dot_shape),
            original_dot->precision_config()));
      }
      VLOG(2) << dot->ToString();

      if (windowed_at_contracting_dims) {
        if (original_hlo->opcode() != HloOpcode::kDot) {
          // Reshape to the original sharded dot shape.
          dot = body_b.AddInstruction(
              HloInstruction::CreateReshape(original_sharded_dot_shape, dot));
        }

        // Accumulate the partial output to the result buffer.
        o = body_b.AddInstruction(
            HloInstruction::CreateBinary(o->shape(), HloOpcode::kAdd, o, dot));
      } else {
        // The windowing operand is partitioned along batch/non-contracting
        // dimensions, so we need a dynamic-update-slice to save the partial
        // output in the result buffer.
        auto slice_shape = dot->shape();
        auto slice_dim =
            lhs_concat_dim != -1
                ? indices_map.lhs_to_output_indices[lhs_concat_dim]
                : indices_map.rhs_to_output_indices[rhs_concat_dim];
        slice_shape.set_dimensions(slice_dim, 1);
        std::vector<int64_t> ccw_start_indices(dot->shape().rank(), 0);
        std::vector<int64_t> cw_start_indices(dot->shape().rank(), 0);
        cw_start_indices[slice_dim] = 1;
        auto ccw_dot = body_b.AddInstruction(HloInstruction::CreateSlice(
            slice_shape, dot, ccw_start_indices, slice_shape.dimensions(),
            std::vector<int64_t>(dot->shape().rank(), 1)));
        auto cw_dot = body_b.AddInstruction(HloInstruction::CreateSlice(
            slice_shape, dot, cw_start_indices, dot->shape().dimensions(),
            std::vector<int64_t>(dot->shape().rank(), 1)));

        std::vector<int64_t> reshaped_dims(
            original_sharded_dot_shape.dimensions().begin(),
            original_sharded_dot_shape.dimensions().end());
        reshaped_dims[slice_dim] /= 2;
        ccw_dot = body_b.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(ccw_dot->shape().element_type(),
                                 reshaped_dims),
            ccw_dot));
        cw_dot = body_b.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(cw_dot->shape().element_type(), reshaped_dims),
            cw_dot));

        if (operands_sharded_at_contracting_dims) {
          // Accumulate the partial output to the result buffer.
          o = body_b.AddInstruction(HloInstruction::CreateBinary(
              o->shape(), HloOpcode::kAdd, o, ccw_dot));
          cw_cp_output = body_b.AddInstruction(HloInstruction::CreateBinary(
              o->shape(), HloOpcode::kAdd, cw_cp_output, cw_dot));
        } else {
          auto ccw_offsets = MakePartitionOffsets(
              o->shape(),
              windowed_op_is_lhs ? *lhs_sharding_transposed_to_match_output
                                 : *rhs_sharding_transposed_to_match_output,
              ccw_data_partition_id, &body_b);
          auto cw_offsets = MakePartitionOffsets(
              o->shape(),
              windowed_op_is_lhs ? *lhs_sharding_transposed_to_match_output
                                 : *rhs_sharding_transposed_to_match_output,
              cw_data_partition_id, &body_b);
          o = body_b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              o->shape(), o, ccw_dot, ccw_offsets));
          o = body_b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              o->shape(), o, cw_dot, cw_offsets));
        }
      }

      std::vector<HloInstruction*> partial_results;
      partial_results.push_back(o);
      partial_results.push_back(cw_cp_output);
      return partial_results;
    };

    // Generate partial result used by unidirectional algorithm.
    auto get_partial_unid_result =
        [&](HloInstruction* l, HloInstruction* r, HloInstruction* o,
            HloInstruction* i) -> StatusOr<HloInstruction*> {
      auto partition_id =
          lhs.state().collective_ops_creator.create_partition_id(&body_b);
      auto data_partition_id =
          body_b.AddInstruction(HloInstruction::CreateBinary(
              i->shape(), HloOpcode::kAdd, i, partition_id));
      auto partition_count =
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32_t>(num_partitions)));
      data_partition_id = body_b.AddInstruction(
          HloInstruction::CreateBinary(i->shape(), HloOpcode::kRemainder,
                                       data_partition_id, partition_count));
      auto dot_lhs = l;
      auto dot_rhs = r;
      if (windowed_at_contracting_dims || windowed_at_batch_dims ||
          operands_sharded_at_contracting_dims) {
        // Slice the matching operand according to the partitioned dimensions on
        // the windowed operand or the output.
        auto slice_operand = !windowed_op_is_lhs ? l : r;
        // We do this by treating the matching operand as replicated, and
        // resharding it to match the windowed operand or the output.
        slice_operand->set_sharding(HloSharding::Replicate());
        auto state = lhs.state();
        state.b = &body_b;
        state.partition_id = data_partition_id;
        state.reshard_cache->per_hlo_cache.erase(slice_operand);
        auto slice =
            PartitionedHlo(slice_operand, slice_operand->shape(), state)
                .Reshard(*slice_sharding)
                .hlo();
        slice_operand->clear_sharding();
        if (!windowed_op_is_lhs) {
          dot_lhs = slice;
        } else {
          dot_rhs = slice;
        }
      }
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(dot_lhs, dot_rhs, &body_b, conv_window));
      if (windowed_at_contracting_dims ||
          operands_sharded_at_contracting_dims) {
        // Accumulate the partial output to the result buffer.
        o = body_b.AddInstruction(
            HloInstruction::CreateBinary(o->shape(), HloOpcode::kAdd, o, dot));
      } else {
        // The windowing operand is partitioned along batch/non-contracting
        // dimensions, so we need a dynamic-update-slice to save the partial
        // output in the result buffer.
        auto offsets = MakePartitionOffsets(
            o->shape(),
            windowed_op_is_lhs ? *lhs_sharding_transposed_to_match_output
                               : *rhs_sharding_transposed_to_match_output,
            data_partition_id, &body_b);
        o = body_b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            o->shape(), o, dot, offsets));
      }
      return o;
    };

    auto param = body_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShapeWithPtrs(
            {&lhs_hlo->shape(), &rhs_hlo->shape(), &result_buffer->shape(),
             &extra_buffer->shape(), &iteration->shape()}),
        "param"));
    auto l = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(lhs_hlo->shape(), param, 0));
    auto r = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(rhs_hlo->shape(), param, 1));
    auto o = body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        result_buffer->shape(), param, 2));
    auto extra_inout = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(extra_buffer->shape(), param, 3));
    auto i = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(iteration->shape(), param, 4));

    // The bidirectional collective permute implementation has loop unrolling
    // of degree 2, so num_partitions is required to be a multiple of 4.
    if (options.bidirectional_windowed_einsum && num_partitions % 4 == 0) {
      std::vector<std::pair<int64_t, int64_t>> ccw_sd_pairs(num_partitions);
      for (int64_t source = 0; source < num_partitions; ++source) {
        // 0 -> n-1, 1 -> 0, 2 -> 1, ...
        ccw_sd_pairs[source] = {source,
                                (source - 1 + num_partitions) % num_partitions};
      }
      std::vector<std::pair<int64_t, int64_t>> cw_sd_pairs(num_partitions);
      for (int64_t source = 0; source < num_partitions; ++source) {
        // 0 -> 1, 1 -> 2, 2 -> 3, ...
        cw_sd_pairs[source] = {source, (source + 1) % num_partitions};
      }

      // Even number iteration.
      auto next_l = l;
      auto next_r = r;
      auto ccw_cp_input = operands_sharded_at_contracting_dims ? o
                          : windowed_op_is_lhs                 ? l
                                                               : r;
      auto ccw_cp_output =
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  &body_b, ccw_cp_input, ccw_sd_pairs,
                  (*lhs.state().next_channel_id)++);
      if (operands_sharded_at_contracting_dims) {
        o = ccw_cp_output;
      } else if (windowed_op_is_lhs) {
        next_l = ccw_cp_output;
      } else {
        next_r = ccw_cp_output;
      }
      auto cw_cp_input = extra_inout;
      auto cw_cp_output =
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  &body_b, cw_cp_input, cw_sd_pairs,
                  (*lhs.state().next_channel_id)++);

      TF_ASSIGN_OR_RETURN(
          auto outputs,
          get_partial_bid_results(l, r, o, extra_inout, cw_cp_output, i));
      o = outputs[0];
      cw_cp_output = outputs[1];

      // ++i
      i = body_b.AddInstruction(HloInstruction::CreateBinary(
          i->shape(), HloOpcode::kAdd, i, CreateOne(i->shape(), &body_b)));

      // Odd number iteration.
      auto second_next_l = next_l;
      auto second_next_r = next_r;
      ccw_cp_input = operands_sharded_at_contracting_dims ? o
                     : windowed_op_is_lhs                 ? next_l
                                                          : next_r;
      ccw_cp_output =
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  &body_b, ccw_cp_input, ccw_sd_pairs,
                  (*lhs.state().next_channel_id)++);
      if (operands_sharded_at_contracting_dims) {
        o = ccw_cp_output;
      } else if (windowed_op_is_lhs) {
        second_next_l = ccw_cp_output;
      } else {
        second_next_r = ccw_cp_output;
      }
      auto next_cw_cp_input = cw_cp_output;
      auto next_cw_cp_output =
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  &body_b, next_cw_cp_input, cw_sd_pairs,
                  (*lhs.state().next_channel_id)++);

      TF_ASSIGN_OR_RETURN(
          outputs, get_partial_bid_results(next_l, next_r, o, cw_cp_output,
                                           next_cw_cp_output, i));
      o = outputs[0];
      next_cw_cp_output = outputs[1];

      // ++i
      i = body_b.AddInstruction(HloInstruction::CreateBinary(
          i->shape(), HloOpcode::kAdd, i, CreateOne(i->shape(), &body_b)));

      body_b.AddInstruction(HloInstruction::CreateTuple(
          {second_next_l, second_next_r, o, next_cw_cp_output, i}));

    } else if (options.unroll_windowed_einsum && num_partitions % 2 == 0) {
      if (operands_sharded_at_contracting_dims) {
        std::vector<std::pair<int64_t, int64_t>> output_sd_pairs(
            num_partitions);
        for (int64_t source = 0; source < num_partitions; ++source) {
          // 0 -> n-2, 1 -> n-1, 2 -> 0, ...
          output_sd_pairs[source] = {
              source, (source - 2 + num_partitions) % num_partitions};
        }

        o = lhs.state()
                .collective_ops_creator
                .create_cross_partition_collective_permute(
                    &body_b, o, output_sd_pairs,
                    (*lhs.state().next_channel_id)++);

        TF_ASSIGN_OR_RETURN(extra_inout,
                            get_partial_unid_result(l, r, extra_inout, i));

        extra_inout = lhs.state()
                          .collective_ops_creator
                          .create_cross_partition_collective_permute(
                              &body_b, extra_inout, output_sd_pairs,
                              (*lhs.state().next_channel_id)++);

        // i+2
        i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32_t>(2)))));
        auto real_i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32_t>(1)))));

        TF_ASSIGN_OR_RETURN(o, get_partial_unid_result(l, r, o, real_i));
        body_b.AddInstruction(
            HloInstruction::CreateTuple({l, r, o, extra_inout, i}));
      } else {
        std::vector<std::pair<int64_t, int64_t>> sd_pairs(num_partitions);
        for (int64_t source = 0; source < num_partitions; ++source) {
          // 0 -> n-1, 1 -> 0, 2 -> 1, ...
          sd_pairs[source] = {source,
                              (source - 1 + num_partitions) % num_partitions};
        }

        // Even number iteration.
        auto next_l = l;
        auto next_r = r;
        auto cp_input = windowed_op_is_lhs ? l : r;
        auto cp_output = lhs.state()
                             .collective_ops_creator
                             .create_cross_partition_collective_permute(
                                 &body_b, cp_input, sd_pairs,
                                 (*lhs.state().next_channel_id)++);
        if (windowed_op_is_lhs) {
          next_l = cp_output;
        } else {
          next_r = cp_output;
        }
        TF_ASSIGN_OR_RETURN(o, get_partial_unid_result(l, r, o, i));

        // ++i
        i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32_t>(1)))));

        // Odd number iteration.
        auto second_next_l = next_l;
        auto second_next_r = next_r;
        cp_input = windowed_op_is_lhs ? next_l : next_r;
        cp_output = lhs.state()
                        .collective_ops_creator
                        .create_cross_partition_collective_permute(
                            &body_b, cp_input, sd_pairs,
                            (*lhs.state().next_channel_id)++);
        if (windowed_op_is_lhs) {
          second_next_l = cp_output;
        } else {
          second_next_r = cp_output;
        }
        TF_ASSIGN_OR_RETURN(o, get_partial_unid_result(next_l, next_r, o, i));

        // ++i
        i = body_b.AddInstruction(HloInstruction::CreateBinary(
            i->shape(), HloOpcode::kAdd, i,
            body_b.AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<uint32_t>(1)))));

        body_b.AddInstruction(HloInstruction::CreateTuple(
            {second_next_l, second_next_r, o, extra_inout, i}));
      }
    } else {
      auto real_i = i;
      if (operands_sharded_at_contracting_dims) {
        // For reduce-scatter case, start from the data_partition_id + 1 to make
        // the data_partition_id of the final data shard in each partition the
        // same as the corresponding partition_id.
        real_i = body_b.AddInstruction(HloInstruction::CreateBinary(
            real_i->shape(), HloOpcode::kAdd, real_i,
            CreateOne(real_i->shape(), &body_b)));
      }
      TF_ASSIGN_OR_RETURN(o, get_partial_unid_result(l, r, o, real_i));

      // ++i
      i = body_b.AddInstruction(HloInstruction::CreateBinary(
          i->shape(), HloOpcode::kAdd, i,
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32_t>(1)))));
      auto has_more = body_b.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), i,
          body_b.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32_t>(num_partitions))),
          ComparisonDirection::kLt));
      // Collective-permute for the next window. We don't need it for the last
      // iteration, so we use a conditional around the collective-permute.
      HloInstruction* conditional;
      {
        SpmdBuilder cp_b("window_collective_permute", original_hlo);
        {
          auto p = cp_b.AddInstruction(HloInstruction::CreateParameter(
              0,
              operands_sharded_at_contracting_dims ? o->shape()
              : windowed_op_is_lhs                 ? l->shape()
                                                   : r->shape(),
              "window"));
          std::vector<std::pair<int64_t, int64_t>> sd_pairs(num_partitions);
          for (int64_t source = 0; source < num_partitions; ++source) {
            // 0 -> n-1, 1 -> 0, 2 -> 1, ...
            sd_pairs[source] = {source,
                                (source - 1 + num_partitions) % num_partitions};
          }
          lhs.state()
              .collective_ops_creator.create_cross_partition_collective_permute(
                  &cp_b, p, sd_pairs, (*lhs.state().next_channel_id)++);
        }
        SpmdBuilder ncp_b("last_iteration_noop", original_hlo);
        {
          ncp_b.AddInstruction(HloInstruction::CreateParameter(
              0,
              operands_sharded_at_contracting_dims ? o->shape()
              : windowed_op_is_lhs                 ? l->shape()
                                                   : r->shape(),
              "window"));
        }
        conditional = body_b.AddInstruction(HloInstruction::CreateConditional(
            operands_sharded_at_contracting_dims ? o->shape()
            : windowed_op_is_lhs                 ? l->shape()
                                                 : r->shape(),
            has_more,
            operands_sharded_at_contracting_dims ? o
            : windowed_op_is_lhs                 ? l
                                                 : r,
            module->AddEmbeddedComputation(cp_b.Build()),
            operands_sharded_at_contracting_dims ? o
            : windowed_op_is_lhs                 ? l
                                                 : r,
            module->AddEmbeddedComputation(ncp_b.Build())));
      }
      if (operands_sharded_at_contracting_dims) {
        o = conditional;
      } else if (windowed_op_is_lhs) {
        l = conditional;
      } else {
        r = conditional;
      }
      body_b.AddInstruction(
          HloInstruction::CreateTuple({l, r, o, extra_inout, i}));
    }

    SpmdBuilder cond_b("windowed_dot_general_cond", original_hlo);
    auto cond_param = cond_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShapeWithPtrs(
            {&lhs_hlo->shape(), &rhs_hlo->shape(), &result_buffer->shape(),
             &extra_buffer->shape(), &iteration->shape()}),
        "param"));
    auto cond_i = cond_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        iteration->shape(), cond_param, 4));
    int64_t adapted_num_partitions =
        (options.bidirectional_windowed_einsum && num_partitions % 4 == 0)
            ? num_partitions / 2
            : num_partitions;
    cond_b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), cond_i,
        cond_b.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32_t>(adapted_num_partitions))),
        ComparisonDirection::kLt));
    auto while_loop = b->AddInstruction(HloInstruction::CreateWhile(
        cond_param->shape(), module->AddEmbeddedComputation(cond_b.Build()),
        module->AddEmbeddedComputation(body_b.Build()),
        b->AddInstruction(HloInstruction::CreateTuple(
            {lhs_hlo, rhs_hlo, result_buffer, extra_buffer, iteration}))));
    windowed_dot_general_loops->push_back(
        {while_loop, windowed_op_is_lhs ? 0 : 1, windowed_at_contracting_dims,
         windowed_at_batch_dims, operands_sharded_at_contracting_dims,
         num_partitions, GetLoopReplicaGroups(while_loop)});
    auto result = b->AddInstruction(HloInstruction::CreateGetTupleElement(
        result_buffer->shape(), while_loop, 2));
    if (((options.bidirectional_windowed_einsum && num_partitions % 4 == 0) ||
         (options.unroll_windowed_einsum && num_partitions % 2 == 0)) &&
        operands_sharded_at_contracting_dims) {
      std::vector<std::pair<int64_t, int64_t>> extra_sd_pairs(num_partitions);
      for (int64_t source = 0; source < num_partitions; ++source) {
        // 0 -> 1, 1 -> 2, 2 -> 3, ...
        extra_sd_pairs[source] = {source, (source + 1) % num_partitions};
      }
      auto extra_result =
          b->AddInstruction(HloInstruction::CreateGetTupleElement(
              extra_buffer->shape(), while_loop, 3));
      if (options.bidirectional_windowed_einsum && num_partitions % 4 == 0) {
        extra_result = lhs.state()
                           .collective_ops_creator
                           .create_cross_partition_collective_permute(
                               b, extra_result, extra_sd_pairs,
                               (*lhs.state().next_channel_id)++);
      }
      if (options.unroll_windowed_einsum && num_partitions % 2 == 0) {
        result = lhs.state()
                     .collective_ops_creator
                     .create_cross_partition_collective_permute(
                         b, result, extra_sd_pairs,
                         (*lhs.state().next_channel_id)++);
      }
      result = b->AddInstruction(HloInstruction::CreateBinary(
          result->shape(), HloOpcode::kAdd, result, extra_result));
    }
    if (!ShapeUtil::Compatible(padded_result_buffer_shape,
                               unpadded_result_buffer_shape)) {
      result = b->AddInstruction(HloInstruction::CreateSlice(
          unpadded_result_buffer_shape, result,
          std::vector<int64_t>(padded_result_buffer_shape.rank(), 0),
          unpadded_result_buffer_shape.dimensions(),
          std::vector<int64_t>(padded_result_buffer_shape.rank(), 1)));
    }
    return result;
  };
  // Hard limit on iteration count based on empirical data (above this amount
  // there's pretty significant overhead).
  constexpr int64_t kMaxIterations = 32;
  absl::optional<WindowedEinsumConfig> e_config =
      GetWindowedEinsumConfiguration(
          num_partitions, output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, rhs_contracting_partitions,
          rhs_non_contracting_partitions, rhs_batch_partitions,
          lhs_contracting_partitions, lhs_non_contracting_partitions,
          lhs_batch_partitions, ShapeSizeInBytes(rhs.base_shape()),
          ShapeSizeInBytes(lhs.base_shape()),
          ShapeSizeInBytes(output_base_shape), options,
          output_sharding_transposed_to_match_lhs,
          output_sharding_transposed_to_match_rhs, lhs_sharding, rhs_sharding,
          conv_window, dims_mapping, kMaxIterations, original_hlo, &lhs, &rhs,
          create_sharded_dot, b, module, visitor);
  if (e_config) {
    VLOG(2) << "Emit windowed dot.";
    return emit_windowed_dot_general(*e_config);
  }

  {
    // Try batch-parallel by resharding one operand, and allowing all-reduce.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * partitioned_dot,
        try_emit_output_batch_partitioned_einsum_with_reshard(true));
    if (partitioned_dot) {
      return partitioned_dot;
    }
  }

  // LHS and RHS have the same partitioned contracting dimensions.
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions == num_partitions) {
    // Pad both sides with zero, since NaN at one side cannot be masked by zero
    // on the other side.
    if (ShapeSizeInBytes(lhs.base_shape()) <
        ShapeSizeInBytes(rhs.base_shape())) {
      lhs = lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithZero();
      rhs = rhs.PadWithZero();
    } else {
      lhs = lhs.PadWithZero();
      rhs = rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithZero();
    }
    TF_ASSIGN_OR_RETURN(
        auto dot, create_sharded_dot(lhs.hlo(), rhs.hlo(), b, conv_window));
    std::vector<int64_t> lhs_contracting_dims;
    lhs_contracting_dims.reserve(lhs.base_shape().rank());
    for (const auto& cd : dims_mapping.contracting_dims) {
      lhs_contracting_dims.push_back(cd.lhs);
    }
    auto ar = lhs.state().partitioner->AllReduceAlongShardingDims(
        b, dot, lhs.sharding(), lhs.state().next_channel_id,
        lhs_contracting_dims, lhs.state().collective_ops_creator,
        MakeBinaryAdd(output_base_shape.element_type(), module));
    ar->set_sharding(HloSharding::Replicate());
    return PartitionedHlo(ar, output_base_shape, lhs.state())
        .Reshard(output_sharding)
        .hlo();
  }

  // LHS and output have the same partitioned non-contracting dimensions.
  if (lhs_non_contracting_partitions == num_partitions &&
      output_lhs_non_contracting_partitions == num_partitions &&
      lhs_sharding_transposed_to_match_output == output_sharding) {
    auto rhs_replicated = rhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(lhs.hlo(), rhs_replicated,
                                                     b, conv_window));
    return dot;
  }

  // RHS and output have the same partitioned non-contracting dimensions.
  if (rhs_non_contracting_partitions == num_partitions &&
      output_rhs_non_contracting_partitions == num_partitions &&
      rhs_sharding_transposed_to_match_output == output_sharding) {
    auto lhs_replicated = lhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(lhs_replicated, rhs.hlo(),
                                                     b, conv_window));
    return dot;
  }

  if (may_reshard_without_detecting_match) {
    // Output is batch partitioned.
    if (output_batch_partitions == num_partitions) {
      auto resharded_lhs =
          lhs.Reshard(*output_sharding_transposed_to_match_lhs);
      auto resharded_rhs =
          rhs.Reshard(*output_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(resharded_lhs.hlo(), resharded_rhs.hlo(),
                                       b, conv_window));
      return dot;
    }
    // Output is partitioned along LHS non-contracting dimensions.
    if (output_lhs_non_contracting_partitions == num_partitions) {
      auto resharded_lhs =
          lhs.Reshard(*output_sharding_transposed_to_match_lhs);
      auto replicated_rhs = rhs.Reshard(HloSharding::Replicate());
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(resharded_lhs.hlo(),
                                       replicated_rhs.hlo(), b, conv_window));
      return dot;
    }
    // Output is partitioned along RHS non-contracting dimensions.
    if (output_rhs_non_contracting_partitions == num_partitions) {
      auto replicated_lhs = lhs.Reshard(HloSharding::Replicate());
      auto resharded_rhs =
          rhs.Reshard(*output_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(replicated_lhs.hlo(),
                                       resharded_rhs.hlo(), b, conv_window));
      return dot;
    }
  }

  // Returns true if it is beneficial to reshard the operand at `operand_idx`
  // across the contracting dimension.
  const auto should_partition_contracting_dim = [&](int64_t operand_idx) {
    if (!output_sharding.IsReplicated()) {
      return false;
    }

    if (operand_idx == 0) {
      // If LHS and output are replicated, we compare the cost of all-gather
      // on RHS vs all-reduce on the output.
      return (rhs_contracting_partitions == num_partitions) &&
             lhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(rhs.base_shape()) >
                 ShapeUtil::ElementsIn(output_base_shape);
    } else {
      return (lhs_contracting_partitions == num_partitions) &&
             rhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(lhs.base_shape()) >
                 ShapeUtil::ElementsIn(output_base_shape);
    }
  };

  // When the output is replicated and one of the operands is partitioned along
  // contracting dimension, align the other operand to be partitioned along
  // the contracting dimensions.
  if (output_sharding.IsReplicated() && (should_partition_contracting_dim(0) ||
                                         should_partition_contracting_dim(1))) {
    if (should_partition_contracting_dim(0)) {
      lhs = lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithZero();
      rhs = rhs.PadWithZero();
    } else {
      lhs = lhs.PadWithZero();
      rhs = rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithZero();
    }
    TF_ASSIGN_OR_RETURN(
        auto dot, create_sharded_dot(lhs.hlo(), rhs.hlo(), b, conv_window));

    std::vector<int64_t> lhs_contracting_dims;
    lhs_contracting_dims.reserve(lhs.base_shape().rank());
    for (const auto& cd : dims_mapping.contracting_dims) {
      lhs_contracting_dims.push_back(cd.lhs);
    }
    return lhs.state().partitioner->AllReduceAlongShardingDims(
        b, dot, lhs.sharding(), lhs.state().next_channel_id,
        lhs_contracting_dims, lhs.state().collective_ops_creator,
        MakeBinaryAdd(output_base_shape.element_type(), module));
  }
  return nullptr;
}

StatusOr<HloInstruction*> PartitionDot(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64_t num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    SpmdPartitioningVisitor* visitor);

StatusOr<HloInstruction*> PartitionDotGroupOnBatch(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64_t num_partitions, int64_t lhs_contracting_partitions,
    int64_t rhs_contracting_partitions, int64_t lhs_non_contracting_partitions,
    int64_t rhs_non_contracting_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    SpmdPartitioningVisitor* visitor) {
  std::vector<std::pair<HloInstruction*, HloSharding>>
      top_level_sharding_to_reset;
  auto cleaner = tensorflow::gtl::MakeCleanup([&] {
    for (auto& to_reset : top_level_sharding_to_reset) {
      to_reset.first->set_sharding(to_reset.second);
    }
  });
  std::vector<int64_t> lhs_dims;
  std::vector<int64_t> rhs_dims;
  std::vector<int64_t> output_dims;
  auto lhs_sharding_dims_adjusted_to_output =
      lhs.sharding().IsReplicated()
          ? std::vector<int64_t>(lhs.base_shape().rank(), 1)
          : lhs.sharding().tile_assignment().dimensions();
  auto rhs_sharding_dims_adjusted_to_output =
      rhs.sharding().IsReplicated()
          ? std::vector<int64_t>(rhs.base_shape().rank(), 1)
          : rhs.sharding().tile_assignment().dimensions();
  auto output_sharding_dims_adjusted_to_lhs =
      output_sharding.tile_assignment().dimensions();
  bool lhs_rhs_dims_matching = true;
  for (const auto& dim : dims_mapping.batch_dims) {
    lhs_dims.push_back(dim.lhs);
    rhs_dims.push_back(dim.rhs);
    output_dims.push_back(dim.output);
    if (lhs_sharding_dims_adjusted_to_output[dim.lhs] !=
        rhs_sharding_dims_adjusted_to_output[dim.rhs]) {
      lhs_rhs_dims_matching = false;
    }
    lhs_sharding_dims_adjusted_to_output[dim.lhs] =
        output_sharding.tile_assignment().dim(dim.output);
    rhs_sharding_dims_adjusted_to_output[dim.rhs] =
        output_sharding.tile_assignment().dim(dim.output);
    output_sharding_dims_adjusted_to_lhs[dim.output] =
        lhs.sharding().tile_assignment().dim(dim.lhs);
  }
  if (require_matching_devices_to_group && lhs_rhs_dims_matching) {
    lhs_rhs_dims_matching =
        rhs.sharding() ==
        UngroupSharding(AlignGroupsWith(
            hlo_sharding_util::GroupShardingOnDims(rhs.sharding(), rhs_dims),
            hlo_sharding_util::GroupShardingOnDims(lhs.sharding(), lhs_dims)));
  }
  auto output_grouped =
      hlo_sharding_util::GroupShardingOnDims(output_sharding, output_dims);
  PartitionedHlo per_group_lhs = lhs;
  PartitionedHlo per_group_rhs = rhs;
  if (lhs_rhs_dims_matching) {
    auto lhs_grouped =
        hlo_sharding_util::GroupShardingOnDims(lhs.sharding(), lhs_dims);
    auto rhs_grouped =
        hlo_sharding_util::GroupShardingOnDims(rhs.sharding(), rhs_dims);
    if (ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) >
        ShapeUtil::ByteSizeOf(rhs.hlo()->shape())) {
      rhs_grouped = AlignGroupsWith(std::move(rhs_grouped), lhs_grouped);
      rhs = rhs.Reshard(UngroupSharding(rhs_grouped));
    } else {
      lhs_grouped = AlignGroupsWith(std::move(lhs_grouped), rhs_grouped);
      lhs = lhs.Reshard(UngroupSharding(lhs_grouped));
    }
    auto reshaped_output_tiling = output_sharding.tile_assignment();
    reshaped_output_tiling.Reshape(output_sharding_dims_adjusted_to_lhs);
    output_grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnDims(
            output_sharding.ReplicateOnLastTileDim()
                ? HloSharding::PartialTile(reshaped_output_tiling)
                : HloSharding::Tile(reshaped_output_tiling),
            output_dims),
        lhs_grouped);
    auto per_group_partitioner_state = CreatePerGroupPartitioningState(
        lhs.state(), lhs_grouped.device_groups, b);
    top_level_sharding_to_reset.emplace_back(lhs.hlo(), lhs.sharding());
    lhs.hlo()->set_sharding(lhs_grouped.sharding);
    top_level_sharding_to_reset.emplace_back(rhs.hlo(), rhs.sharding());
    rhs.hlo()->set_sharding(rhs_grouped.sharding);
    CHECK(lhs.hlo() != rhs.hlo() ||
          lhs_grouped.sharding == rhs_grouped.sharding);
    per_group_lhs = PartitionedHlo(
        lhs.hlo(), GetPerGroupBaseShape(lhs_grouped, lhs.base_shape()),
        per_group_partitioner_state);
    per_group_rhs = PartitionedHlo(
        rhs.hlo(), GetPerGroupBaseShape(rhs_grouped, rhs.base_shape()),
        per_group_partitioner_state);
  } else {
    auto per_group_partitioner_state = CreatePerGroupPartitioningState(
        lhs.state(), output_grouped.device_groups, b);
    auto reshard_to_output_batch =
        [&](PartitionedHlo operand, absl::Span<const int64_t> batch_dims,
            absl::Span<const int64_t> contracting_dims,
            absl::Span<const int64_t> non_contracting_dims,
            int64_t contracting_dim_partitions,
            int64_t non_contracting_dim_partitions,
            int64_t other_contracting_dim_partitions,
            std::vector<int64_t>* sharding_dims_adjusted_to_output)
        -> absl::optional<PartitionedHlo> {
      if (operand.sharding().IsTileMaximal()) {
        auto partially_sharded = PerGroupSliceFromReplicated(
            operand.Replicate().hlo(), operand.state().partition_id,
            output_grouped.device_groups, batch_dims,
            output_grouped.group_dim_sizes, b);
        partially_sharded->set_sharding(HloSharding::Replicate());
        return PartitionedHlo(partially_sharded, partially_sharded->shape(),
                              per_group_partitioner_state);
      }
      auto reshaped_tiling = operand.sharding().tile_assignment();
      // It's possible that the operand is not initially sharded on batch
      // dimensions in the same way as the output, although being tiled. In that
      // case, the current sharding_dims_adjusted_to_output may contain more
      // partitions than available devices. We remove partitioning on other
      // dimensions.
      if (Product(*sharding_dims_adjusted_to_output) >
          reshaped_tiling.num_elements()) {
        if (Product(*sharding_dims_adjusted_to_output) %
                reshaped_tiling.num_elements() !=
            0) {
          return absl::nullopt;
        }
        int64_t ratio = Product(*sharding_dims_adjusted_to_output) /
                        reshaped_tiling.num_elements();
        if (operand.sharding().ReplicateOnLastTileDim() &&
            reshaped_tiling.dimensions().back() % ratio == 0) {
          sharding_dims_adjusted_to_output->back() /= ratio;
          if (sharding_dims_adjusted_to_output->back() == 1) {
            sharding_dims_adjusted_to_output->pop_back();
          }
        } else if (ratio == non_contracting_dim_partitions &&
                   (ratio != contracting_dim_partitions ||
                    contracting_dim_partitions ==
                        other_contracting_dim_partitions)) {
          for (int64_t dim : non_contracting_dims) {
            (*sharding_dims_adjusted_to_output)[dim] = 1;
          }
        } else if (ratio == contracting_dim_partitions) {
          for (int64_t dim : contracting_dims) {
            (*sharding_dims_adjusted_to_output)[dim] = 1;
          }
        } else {
          return absl::nullopt;
        }
      }
      // If the operand is initially sharded more ways than the output in the
      // batch dimensions, sharding_dims_adjusted_to_output currently contains
      // fewer partitions than available devices. We do not handle this case.
      if (Product(*sharding_dims_adjusted_to_output) <
          reshaped_tiling.num_elements()) {
        return absl::nullopt;
      }
      reshaped_tiling.Reshape(*sharding_dims_adjusted_to_output);
      auto grouped =
          AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(
                              operand.base_shape().rank() <
                                      sharding_dims_adjusted_to_output->size()
                                  ? HloSharding::PartialTile(reshaped_tiling)
                                  : HloSharding::Tile(reshaped_tiling),
                              batch_dims),
                          output_grouped);
      if (require_matching_devices_to_group &&
          operand.sharding() != UngroupSharding(grouped)) {
        return absl::nullopt;
      }
      auto resharded = operand.Reshard(UngroupSharding(grouped));
      top_level_sharding_to_reset.emplace_back(resharded.hlo(),
                                               resharded.sharding());
      resharded.hlo()->set_sharding(grouped.sharding);
      return PartitionedHlo(resharded.hlo(),
                            GetPerGroupBaseShape(grouped, operand.base_shape()),
                            per_group_partitioner_state);
    };
    std::vector<int64_t> lhs_contracting_dims;
    std::vector<int64_t> rhs_contracting_dims;
    lhs_contracting_dims.reserve(dims_mapping.contracting_dims.size());
    rhs_contracting_dims.reserve(dims_mapping.contracting_dims.size());
    for (const auto& dim : dims_mapping.contracting_dims) {
      lhs_contracting_dims.push_back(dim.lhs);
      rhs_contracting_dims.push_back(dim.rhs);
    }
    std::vector<int64_t> lhs_non_contracting_dims;
    std::vector<int64_t> rhs_non_contracting_dims;
    lhs_non_contracting_dims.reserve(
        dims_mapping.lhs_non_contracting_dims.size());
    rhs_non_contracting_dims.reserve(
        dims_mapping.rhs_non_contracting_dims.size());
    for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
      lhs_non_contracting_dims.push_back(dim.lhs);
    }
    for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
      rhs_non_contracting_dims.push_back(dim.rhs);
    }
    if (auto resharded = reshard_to_output_batch(
            lhs, lhs_dims, lhs_contracting_dims, lhs_non_contracting_dims,
            lhs_contracting_partitions, lhs_non_contracting_partitions,
            rhs_contracting_partitions,
            &lhs_sharding_dims_adjusted_to_output)) {
      per_group_lhs = *resharded;
    } else {
      return nullptr;
    }
    if (auto resharded = reshard_to_output_batch(
            rhs, rhs_dims, rhs_contracting_dims, rhs_non_contracting_dims,
            rhs_contracting_partitions, rhs_non_contracting_partitions,
            lhs_contracting_partitions,
            &rhs_sharding_dims_adjusted_to_output)) {
      per_group_rhs = *resharded;
    } else {
      return nullptr;
    }
    CHECK(lhs.hlo() != rhs.hlo() ||
          per_group_lhs.sharding() == per_group_rhs.sharding());
  }
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionDot(per_group_lhs, per_group_rhs,
                   GetPerGroupBaseShape(output_grouped, output_base_shape),
                   output_grouped.sharding, dims_mapping,
                   num_partitions / output_grouped.device_groups.size(),
                   create_sharded_dot, conv_window, module, original_hlo,
                   options, b, windowed_dot_general_loops, visitor));
  dot->set_sharding(UngroupSharding(output_grouped));
  return PartitionedHlo(dot, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

GroupedSharding GetNonContractingPartitionGroupedShardingForMatchedOperand(
    bool lhs_matching, const HloSharding& matching_sharding,
    const HloSharding& output_sharding,
    absl::Span<const DotConvDimsMapping::DimsMapping> partitioned_dims) {
  std::vector<int64_t> matching_sharding_dims =
      matching_sharding.tile_assignment().dimensions();
  std::vector<int64_t> matching_dims;
  std::vector<int64_t> output_dims;
  // Make sure the partitioning on matching's non-contracting dimensions
  // defines the same device groups for both matching and output.
  for (const auto& dim : partitioned_dims) {
    int64_t md = lhs_matching ? dim.lhs : dim.rhs;
    matching_sharding_dims[md] =
        output_sharding.tile_assignment().dim(dim.output);
    matching_dims.push_back(md);
    output_dims.push_back(dim.output);
  }
  GroupedSharding output_grouped =
      hlo_sharding_util::GroupShardingOnDims(output_sharding, output_dims);
  Array<int64_t> reshaped_matching_tiling = matching_sharding.tile_assignment();
  reshaped_matching_tiling.Reshape(matching_sharding_dims);
  return AlignGroupsWith(
      hlo_sharding_util::GroupShardingOnDims(
          matching_sharding.ReplicateOnLastTileDim()
              ? HloSharding::PartialTile(reshaped_matching_tiling)
              : HloSharding::Tile(reshaped_matching_tiling),
          matching_dims),
      output_grouped);
}

absl::optional<GroupedSharding>
GetNonContractingPartitionGroupedShardingForOtherOperand(
    bool lhs_matching, const Shape& output_base_shape, const Shape& other_shape,
    int64_t other_contracting_partitions,
    int64_t other_non_contracting_partitions,
    int64_t matching_contracting_partitions,
    int64_t output_other_non_contracting_partitions,
    const HloSharding& other_sharding, const HloSharding& output_sharding,
    absl::Span<const DotConvDimsMapping::DimsMapping> matching_partitioned_dims,
    absl::Span<const DotConvDimsMapping::DimsMapping>
        other_non_contracting_dims,
    absl::Span<const DotConvDimsMapping::DimsMapping> other_contracting_dims) {
  int64_t group_count = 1;
  std::vector<int64_t> output_dims;
  output_dims.reserve(matching_partitioned_dims.size());
  for (const auto& dim : matching_partitioned_dims) {
    output_dims.push_back(dim.output);
    group_count *= output_sharding.tile_assignment().dim(dim.output);
  }
  GroupedSharding output_grouped =
      hlo_sharding_util::GroupShardingOnDims(output_sharding, output_dims);
  std::vector<int64_t> other_group_dims;
  if (other_sharding.ReplicateOnLastTileDim() &&
      other_sharding.tile_assignment().dimensions().back() % group_count == 0) {
    other_group_dims.push_back(
        other_sharding.tile_assignment().num_dimensions() - 1);
  } else {
    const bool may_replicate_other_contracting_dims =
        (other_contracting_partitions == group_count &&
         other_non_contracting_partitions ==
             output_other_non_contracting_partitions);
    const bool may_replicate_other_non_contracting_dims =
        group_count == other_non_contracting_partitions &&
        matching_contracting_partitions == other_contracting_partitions;
    if (auto found_dims = FindMatchingPartitionedDimsForGrouping(
            other_sharding, output_grouped.device_groups)) {
      other_group_dims = std::move(*found_dims);
    } else if (may_replicate_other_contracting_dims &&
               (!may_replicate_other_non_contracting_dims ||
                ShapeUtil::ByteSizeOf(other_shape)) <=
                   ShapeUtil::ByteSizeOf(MakePartitionedShape(
                       output_base_shape, output_sharding))) {
      for (const auto& dim : other_contracting_dims) {
        other_group_dims.push_back(lhs_matching ? dim.rhs : dim.lhs);
      }
    } else if (may_replicate_other_non_contracting_dims) {
      for (const auto& dim : other_non_contracting_dims) {
        other_group_dims.push_back(lhs_matching ? dim.rhs : dim.lhs);
      }
    } else {
      return absl::nullopt;
    }
  }
  if (other_group_dims.size() == 1 &&
      other_group_dims[0] ==
          other_sharding.tile_assignment().num_dimensions() - 1) {
    std::vector<int64_t> group_dim_shards = {
        other_sharding.tile_assignment().dimensions().back() / group_count};
    return AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnDims(
            other_sharding, {other_group_dims[0]}, group_dim_shards),
        output_grouped, /*ignore_group_order=*/true);

  } else if (!other_sharding.IsReplicated()) {
    return AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(
                               other_sharding, other_group_dims),
                           output_grouped,
                           /*ignore_group_order=*/true);
  }
  return absl::nullopt;
}

StatusOr<HloInstruction*> PartitionDotGroupOnNonContracting(
    bool lhs_matching, PartitionedHlo matching, PartitionedHlo other,
    int64_t matching_contracting_partitions,
    int64_t other_contracting_partitions,
    absl::Span<const DotConvDimsMapping::DimsMapping>
        partitioned_non_contracting_dims,
    int64_t other_non_contracting_partitions,
    int64_t output_other_non_contracting_partitions,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    const DotConvDimsMapping& dims_mapping, int64_t num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    SpmdPartitioningVisitor* visitor) {
  std::vector<std::pair<HloInstruction*, HloSharding>>
      top_level_sharding_to_reset;
  auto cleaner = tensorflow::gtl::MakeCleanup([&] {
    for (auto& to_reset : top_level_sharding_to_reset) {
      to_reset.first->set_sharding(to_reset.second);
    }
  });

  std::vector<int64_t> output_dims;
  output_dims.reserve(partitioned_non_contracting_dims.size());
  for (const auto& dim : partitioned_non_contracting_dims) {
    output_dims.push_back(dim.output);
  }
  GroupedSharding output_grouped =
      hlo_sharding_util::GroupShardingOnDims(output_sharding, output_dims);
  GroupedSharding matching_grouped =
      GetNonContractingPartitionGroupedShardingForMatchedOperand(
          lhs_matching, matching.sharding(), output_sharding,
          partitioned_non_contracting_dims);
  if (require_matching_devices_to_group &&
      matching.sharding() != UngroupSharding(matching_grouped)) {
    return nullptr;
  }
  absl::optional<GroupedSharding> other_grouped =
      GetNonContractingPartitionGroupedShardingForOtherOperand(
          lhs_matching, output_base_shape, other.hlo()->shape(),
          other_contracting_partitions, other_non_contracting_partitions,
          matching_contracting_partitions,
          output_other_non_contracting_partitions, other.sharding(),
          output_sharding, partitioned_non_contracting_dims,
          lhs_matching ? dims_mapping.rhs_non_contracting_dims
                       : dims_mapping.lhs_non_contracting_dims,
          dims_mapping.contracting_dims);

  if (!other_grouped) {
    other = other.Replicate();
  }
  matching = matching.Reshard(UngroupSharding(matching_grouped));
  auto per_group_partitioner_state = CreatePerGroupPartitioningState(
      matching.state(), matching_grouped.device_groups, b);
  top_level_sharding_to_reset.emplace_back(matching.hlo(), matching.sharding());
  matching.hlo()->set_sharding(matching_grouped.sharding);
  auto matching_p = PartitionedHlo(
      matching.hlo(),
      GetPerGroupBaseShape(matching_grouped, matching.base_shape()),
      per_group_partitioner_state);

  auto partially_replicated_other = other.hlo();
  if (other_grouped && other_grouped->group_dims.size() == 1 &&
      other_grouped->group_dims[0] == other.base_shape().rank()) {
    // Group on replication dim.
    other = other.Reshard(UngroupSharding(*other_grouped));
    partially_replicated_other = other.hlo();
    top_level_sharding_to_reset.emplace_back(other.hlo(), other.sharding());
    partially_replicated_other->set_sharding(other_grouped->sharding);
  } else if (!other.sharding().IsReplicated()) {
    HloSharding target_sharding = UngroupSharding(*other_grouped);
    GroupedSharding target_group_sharding =
        hlo_sharding_util::GroupShardingOnDims(target_sharding,
                                               other_grouped->group_dims);
    const bool device_group_match = hlo_sharding_util::DeviceGroupsAreMatch(
        target_group_sharding, *other_grouped, /*ignore_group_order=*/false);

    // Do not reshard for partial replicate if device group are matched.
    // There is a reshard to partial replicate right after this reshard. If
    // the device ids within each partial replicate group is the same, no need
    // to reshard here.
    if (!other.sharding().ReplicateOnLastTileDim() || !device_group_match) {
      other = other.Reshard(target_sharding);
    }
    partially_replicated_other =
        other
            .Reshard(hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                other.sharding(), other_grouped->group_dims))
            .hlo();
    top_level_sharding_to_reset.emplace_back(
        partially_replicated_other, partially_replicated_other->sharding());
    partially_replicated_other->set_sharding(other_grouped->sharding);
  }
  auto other_p = PartitionedHlo(partially_replicated_other, other.base_shape(),
                                per_group_partitioner_state);
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionDot(lhs_matching ? matching_p : other_p,
                   lhs_matching ? other_p : matching_p,
                   GetPerGroupBaseShape(output_grouped, output_base_shape),
                   output_grouped.sharding, dims_mapping,
                   num_partitions / matching_grouped.device_groups.size(),
                   create_sharded_dot, conv_window, module, original_hlo,
                   options, b, windowed_dot_general_loops, visitor));
  return dot;
}

std::pair<HloSharding, HloSharding>
GetDotGroupPartitionContractingOutputShardings(
    const DotConvDimsMapping& dims_mapping, const GroupedSharding& lhs_grouped,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    int64_t group_count, int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions,
    int64_t output_batch_partitions,
    std::vector<int64_t>* output_slice_dims_out) {
  HloSharding inner_output_sharding = HloSharding::Replicate();
  HloSharding outer_output_tmp_sharding = HloSharding::Replicate();
  std::vector<int64_t> output_slice_dims;
  if (output_sharding.ReplicateOnLastTileDim() &&
      output_sharding.tile_assignment().dimensions().back() % group_count ==
          0) {
    std::vector<int64_t> group_dim_shards = {
        output_sharding.tile_assignment().dimensions().back() / group_count};
    auto grouped = AlignGroupsWith(
        hlo_sharding_util::GroupShardingOnDims(
            output_sharding,
            {output_sharding.tile_assignment().num_dimensions() - 1},
            group_dim_shards),
        lhs_grouped,
        /*ignore_group_order=*/true);
    outer_output_tmp_sharding = UngroupSharding(grouped);
    inner_output_sharding = std::move(grouped.sharding);
  } else {
    if (auto found_dims = FindMatchingPartitionedDimsForGrouping(
            output_sharding, lhs_grouped.device_groups)) {
      output_slice_dims = std::move(*found_dims);
    } else if (output_lhs_non_contracting_partitions == group_count ||
               output_rhs_non_contracting_partitions == group_count ||
               output_batch_partitions == group_count) {
      if (output_lhs_non_contracting_partitions == group_count) {
        for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
          output_slice_dims.push_back(dim.output);
        }
      } else if (output_rhs_non_contracting_partitions == group_count) {
        for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
          output_slice_dims.push_back(dim.output);
        }
      } else {
        for (const auto& dim : dims_mapping.batch_dims) {
          output_slice_dims.push_back(dim.output);
        }
      }
    }
    if (!output_slice_dims.empty()) {
      auto grouped = AlignGroupsWith(hlo_sharding_util::GroupShardingOnDims(
                                         output_sharding, output_slice_dims),
                                     lhs_grouped);
      inner_output_sharding = grouped.sharding;
      outer_output_tmp_sharding = UngroupSharding(grouped);
    }
  }
  if (output_slice_dims_out) {
    (*output_slice_dims_out) = std::move(output_slice_dims);
  }
  return std::make_pair(inner_output_sharding, outer_output_tmp_sharding);
}

std::pair<HloSharding, HloSharding>
GetDotGroupPartitionContractingLhsRhsShardings(
    const PartitionedHlo& lhs, const PartitionedHlo& rhs,
    absl::Span<const DotConvDimsMapping::DimsMapping>
        partitioned_contracting_dims) {
  HloSharding lhs_sharding = lhs.sharding();
  HloSharding rhs_sharding = rhs.sharding();
  std::vector<int64_t> lhs_tile_shape =
      lhs_sharding.tile_assignment().dimensions();
  std::vector<int64_t> rhs_tile_shape =
      rhs_sharding.tile_assignment().dimensions();
  if (ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) >
      ShapeUtil::ByteSizeOf(rhs.hlo()->shape())) {
    for (const auto& dim : partitioned_contracting_dims) {
      rhs_tile_shape[dim.rhs] = lhs_tile_shape[dim.lhs];
    }
    auto new_tile = rhs.sharding().tile_assignment();
    new_tile.Reshape(rhs_tile_shape);
    rhs_sharding = rhs_sharding.ReplicateOnLastTileDim()
                       ? HloSharding::PartialTile(new_tile)
                       : HloSharding::Tile(new_tile);
  } else {
    for (const auto& dim : partitioned_contracting_dims) {
      lhs_tile_shape[dim.lhs] = rhs_tile_shape[dim.rhs];
    }
    auto new_tile = lhs.sharding().tile_assignment();
    new_tile.Reshape(lhs_tile_shape);
    lhs_sharding = lhs_sharding.ReplicateOnLastTileDim()
                       ? HloSharding::PartialTile(new_tile)
                       : HloSharding::Tile(new_tile);
  }
  return std::make_pair(lhs_sharding, rhs_sharding);
}

StatusOr<HloInstruction*> PartitionDotGroupOnContracting(
    PartitionedHlo lhs, PartitionedHlo rhs,
    absl::Span<const DotConvDimsMapping::DimsMapping>
        partitioned_contracting_dims,
    int64_t output_batch_partitions,
    int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions,
    const Shape& output_base_shape, const HloSharding& output_sharding,
    const DotConvDimsMapping& dims_mapping, int64_t num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    SpmdPartitioningVisitor* visitor) {
  std::vector<std::pair<HloInstruction*, HloSharding>>
      top_level_sharding_to_reset;
  auto cleaner = tensorflow::gtl::MakeCleanup([&] {
    for (auto& to_reset : top_level_sharding_to_reset) {
      to_reset.first->set_sharding(to_reset.second);
    }
  });
  std::vector<int64_t> lhs_dims;
  std::vector<int64_t> rhs_dims;
  int64_t group_count = 1;
  for (const auto& dim : partitioned_contracting_dims) {
    lhs_dims.push_back(dim.lhs);
    rhs_dims.push_back(dim.rhs);
    group_count *= lhs.sharding().tile_assignment().dim(dim.lhs);
  }
  HloSharding lhs_sharding = HloSharding::Replicate();
  HloSharding rhs_sharding = HloSharding::Replicate();
  std::tie(lhs_sharding, rhs_sharding) =
      GetDotGroupPartitionContractingLhsRhsShardings(
          lhs, rhs, partitioned_contracting_dims);
  auto lhs_grouped =
      hlo_sharding_util::GroupShardingOnDims(lhs_sharding, lhs_dims);
  auto rhs_grouped =
      hlo_sharding_util::GroupShardingOnDims(rhs_sharding, rhs_dims);
  if (ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) >
      ShapeUtil::ByteSizeOf(rhs.hlo()->shape())) {
    rhs_grouped = AlignGroupsWith(rhs_grouped, lhs_grouped);
    rhs_sharding = UngroupSharding(rhs_grouped);
    if (require_matching_devices_to_group && rhs.sharding() != rhs_sharding) {
      return nullptr;
    }
    rhs = rhs.Reshard(rhs_sharding);
  } else {
    lhs_grouped = AlignGroupsWith(lhs_grouped, rhs_grouped);
    lhs_sharding = UngroupSharding(lhs_grouped);
    if (require_matching_devices_to_group && lhs.sharding() != lhs_sharding) {
      return nullptr;
    }
    lhs = lhs.Reshard(lhs_sharding);
  }
  // Mask out invalid data.
  std::vector<int64_t> lhs_skipped_dims;
  for (int64_t i = 0; i < lhs.base_shape().rank(); ++i) {
    if (absl::c_linear_search(lhs_dims, i)) {
      continue;
    }
    lhs_skipped_dims.push_back(i);
  }
  lhs = lhs.PadWithZero(
      /*left_padded_dims=*/{}, lhs_skipped_dims);
  std::vector<int64_t> rhs_skipped_dims;
  for (int64_t i = 0; i < rhs.base_shape().rank(); ++i) {
    if (absl::c_linear_search(rhs_dims, i)) {
      continue;
    }
    rhs_skipped_dims.push_back(i);
  }
  rhs = rhs.PadWithZero(
      /*left_padded_dims=*/{}, rhs_skipped_dims);
  top_level_sharding_to_reset.emplace_back(lhs.hlo(), lhs_sharding);
  lhs.hlo()->set_sharding(lhs_grouped.sharding);
  top_level_sharding_to_reset.emplace_back(rhs.hlo(), rhs_sharding);
  rhs.hlo()->set_sharding(rhs_grouped.sharding);

  HloSharding inner_output_sharding = HloSharding::Replicate();
  HloSharding outer_output_tmp_sharding = HloSharding::Replicate();
  std::vector<int64_t> output_slice_dims;
  std::tie(inner_output_sharding, outer_output_tmp_sharding) =
      GetDotGroupPartitionContractingOutputShardings(
          dims_mapping, lhs_grouped, output_base_shape, output_sharding,
          group_count, output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, output_batch_partitions,
          &output_slice_dims);
  Shape inner_output_base_shape = output_base_shape;
  auto get_non_slice_dims = [&] {
    std::vector<int64_t> non_group_dims;
    for (int64_t i = 0; i < output_base_shape.rank(); ++i) {
      if (!absl::c_linear_search(output_slice_dims, i)) {
        non_group_dims.push_back(i);
      }
    }
    return non_group_dims;
  };
  if (!output_slice_dims.empty()) {
    inner_output_base_shape = MakePartitionedShape(
        output_base_shape,
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            output_sharding, get_non_slice_dims()));
  }
  std::function<StatusOr<HloInstruction*>(HloInstruction*, HloInstruction*,
                                          SpmdBuilder*, const Window&)>
      inner_creator =
          [&](HloInstruction* l, HloInstruction* r, SpmdBuilder* b,
              const Window& conv_window) -> StatusOr<HloInstruction*> {
    TF_ASSIGN_OR_RETURN(auto inner_dot,
                        create_sharded_dot(l, r, b, conv_window));
    auto ar = lhs.state().partitioner->AllReduceAlongShardingDims(
        b, inner_dot, lhs_sharding, lhs.state().next_channel_id, lhs_dims,
        lhs.state().collective_ops_creator,
        MakeBinaryAdd(output_base_shape.element_type(), module));
    if (output_slice_dims.empty()) {
      return ar;
    }
    // Use resharding to slice the output. Use a temporary reshard cache since
    // we are faking with replicated sharding.
    PartitionedHlo::PartitioningState new_state = lhs.state();
    new_state.b = b;
    new_state.partition_id =
        lhs.state().collective_ops_creator.create_partition_id(b);
    PartitionedHlo::ReshardCache tmp_cache;
    new_state.reshard_cache = &tmp_cache;
    ar->set_sharding(HloSharding::Replicate());
    return PartitionedHlo(ar, ar->shape(), new_state)
        .Reshard(hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            output_sharding, get_non_slice_dims()))
        .hlo();
  };
  // Disable doing the inner reshard when the "faster windowed einsum" flag is
  // enabled, because the windowed einsum implementation is currently slow with
  // this kind of reshard happening.
  if (options.choose_faster_windowed_einsum_over_mem) {
    inner_output_base_shape = output_base_shape;
    inner_creator = create_sharded_dot;
    outer_output_tmp_sharding =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            outer_output_tmp_sharding, output_slice_dims);
  }
  PartitionedHlo::PartitioningState inner_state =
      CreatePerGroupPartitioningState(lhs.state(), lhs_grouped.device_groups,
                                      b);
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionDot(
          PartitionedHlo(lhs.hlo(),
                         GetPerGroupBaseShape(lhs_grouped, lhs.base_shape()),
                         inner_state),
          PartitionedHlo(rhs.hlo(),
                         GetPerGroupBaseShape(rhs_grouped, rhs.base_shape()),
                         inner_state),
          inner_output_base_shape, inner_output_sharding, dims_mapping,
          num_partitions / group_count, inner_creator, conv_window, module,
          original_hlo, options, b, windowed_dot_general_loops, visitor));
  if (!dot) {
    return nullptr;
  }

  if (options.choose_faster_windowed_einsum_over_mem) {
    HloInstruction* ar = lhs.state().partitioner->AllReduceAlongShardingDims(
        b, dot, lhs_sharding, lhs.state().next_channel_id, lhs_dims,
        lhs.state().collective_ops_creator,
        MakeBinaryAdd(output_base_shape.element_type(), module));
    dot = ar;
  }

  dot->set_sharding(outer_output_tmp_sharding);
  auto d = PartitionedHlo(dot, output_base_shape, lhs.state())
               .Reshard(output_sharding)
               .hlo();
  return d;
}

DotConvDimsMapping ConvertDimsMappingWithFeatureGroupCount(
    const DotConvDimsMapping& dims_mapping, HloInstruction* original_hlo) {
  const auto& dnums = original_hlo->convolution_dimension_numbers();
  DotConvDimsMapping new_dims_mapping;
  new_dims_mapping.batch_dims = dims_mapping.batch_dims;
  new_dims_mapping.conv_spatial_dims = dims_mapping.conv_spatial_dims;
  // Append batch dims.
  new_dims_mapping.batch_dims.emplace_back();
  new_dims_mapping.batch_dims.back().lhs = dnums.input_feature_dimension();
  new_dims_mapping.batch_dims.back().rhs =
      dnums.kernel_output_feature_dimension();
  new_dims_mapping.batch_dims.back().output = dnums.output_feature_dimension();
  new_dims_mapping.batch_dims.back().spatial = -1;
  // Setup non contracting dims.
  new_dims_mapping.lhs_non_contracting_dims.emplace_back();
  new_dims_mapping.lhs_non_contracting_dims.back().lhs =
      dnums.input_batch_dimension();
  new_dims_mapping.rhs_non_contracting_dims.emplace_back();
  new_dims_mapping.rhs_non_contracting_dims.back().rhs =
      dnums.kernel_input_feature_dimension();
  return new_dims_mapping;
}

DotConvDimsMapping ConvertDimsMappingWithBatchGroupCount(
    const DotConvDimsMapping& dims_mapping, HloInstruction* original_hlo) {
  const auto& dnums = original_hlo->convolution_dimension_numbers();
  DotConvDimsMapping new_dims_mapping;
  new_dims_mapping.batch_dims = dims_mapping.batch_dims;
  new_dims_mapping.conv_spatial_dims = dims_mapping.conv_spatial_dims;
  new_dims_mapping.contracting_dims = dims_mapping.contracting_dims;
  // Append batch dims.
  new_dims_mapping.batch_dims.emplace_back();
  new_dims_mapping.batch_dims.back().lhs = dnums.input_batch_dimension();
  new_dims_mapping.batch_dims.back().rhs =
      dnums.kernel_output_feature_dimension();
  new_dims_mapping.batch_dims.back().output = dnums.output_feature_dimension();
  new_dims_mapping.batch_dims.back().spatial = -1;
  return new_dims_mapping;
}

// Estimate the number of iterations of a subsequent windowed einsum
// partitioning if its partitioned in the non-contracting dimensions.
// First value returned is the estimate of the number of iterations if LHS is
// matched while the second is the number of iterations if RHS is matched.
std::pair<absl::optional<int64_t>, absl::optional<int64_t>>
EstimateWindowedEinsumIterationsForNonContractingPartitioning(
    const DotConvDimsMapping& dims_mapping, const PartitionedHlo& lhs,
    const PartitionedHlo& rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const SpmdPartitionerOptions& options,
    int64_t num_partitions, int64_t lhs_non_contracting_partitions,
    int64_t rhs_non_contracting_partitions, int64_t lhs_matching_partitions,
    int64_t rhs_matching_partitions, int64_t lhs_contracting_partitions,
    int64_t rhs_contracting_partitions,
    int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions, int64_t lhs_batch_partitions,
    int64_t rhs_batch_partitions, const Window& conv_window) {
  const DotDimensionIndexMapping indices_map = ComputeDimensionIndexMapping(
      dims_mapping, lhs.base_shape().rank(), rhs.base_shape().rank(),
      output_base_shape.rank());
  auto subsequent_einsum_iterations_estimate =
      [&](bool assume_lhs_match) -> absl::optional<int64_t> {
    const std::vector<DotConvDimsMapping::DimsMapping>&
        matching_non_contracting_dims =
            assume_lhs_match ? dims_mapping.lhs_non_contracting_dims
                             : dims_mapping.rhs_non_contracting_dims;
    const std::vector<DotConvDimsMapping::DimsMapping>&
        other_non_contracting_dims =
            assume_lhs_match ? dims_mapping.rhs_non_contracting_dims
                             : dims_mapping.lhs_non_contracting_dims;
    const std::vector<int64_t>& output_to_matching_indices =
        assume_lhs_match ? indices_map.output_to_lhs_indices
                         : indices_map.output_to_rhs_indices;
    const std::vector<int64_t>& output_to_other_indices =
        assume_lhs_match ? indices_map.output_to_rhs_indices
                         : indices_map.output_to_lhs_indices;
    const std::vector<int64_t>& matching_to_output_indices =
        assume_lhs_match ? indices_map.lhs_to_output_indices
                         : indices_map.rhs_to_output_indices;
    const std::vector<int64_t>& other_to_output_indices =
        assume_lhs_match ? indices_map.rhs_to_output_indices
                         : indices_map.lhs_to_output_indices;
    const HloSharding& matching_sharding =
        assume_lhs_match ? lhs.sharding() : rhs.sharding();
    const HloSharding& other_sharding =
        assume_lhs_match ? rhs.sharding() : lhs.sharding();
    const PartitionedHlo& matching_partitioned = assume_lhs_match ? lhs : rhs;
    const PartitionedHlo& other_partitioned = assume_lhs_match ? rhs : lhs;
    const int64_t matching_non_contracting_partitions =
        assume_lhs_match ? lhs_non_contracting_partitions
                         : rhs_non_contracting_partitions;
    const int64_t other_non_contracting_partitions =
        assume_lhs_match ? rhs_non_contracting_partitions
                         : lhs_non_contracting_partitions;
    const int64_t matching_contracting_partitions =
        assume_lhs_match ? lhs_contracting_partitions
                         : rhs_contracting_partitions;
    const int64_t other_contracting_partitions =
        assume_lhs_match ? rhs_contracting_partitions
                         : lhs_contracting_partitions;
    const int64_t output_matching_non_contracting_partitions =
        assume_lhs_match ? output_lhs_non_contracting_partitions
                         : output_rhs_non_contracting_partitions;
    const int64_t output_other_non_contracting_partitions =
        assume_lhs_match ? output_rhs_non_contracting_partitions
                         : output_lhs_non_contracting_partitions;
    const int64_t matching_batch_partitions =
        assume_lhs_match ? lhs_batch_partitions : rhs_batch_partitions;
    const int64_t other_batch_partitions =
        assume_lhs_match ? rhs_batch_partitions : lhs_batch_partitions;
    const int64_t matching_matched_non_contracting_partitions =
        assume_lhs_match ? lhs_non_contracting_partitions
                         : rhs_non_contracting_partitions;
    std::vector<int64_t> output_dims;
    output_dims.reserve(matching_non_contracting_dims.size());
    for (const DotConvDimsMapping::DimsMapping& dim :
         matching_non_contracting_dims) {
      output_dims.push_back(dim.output);
    }
    GroupedSharding output_grouped =
        hlo_sharding_util::GroupShardingOnDims(output_sharding, output_dims);
    GroupedSharding matching_grouped =
        GetNonContractingPartitionGroupedShardingForMatchedOperand(
            assume_lhs_match, matching_sharding, output_sharding,
            matching_non_contracting_dims);
    absl::optional<GroupedSharding> other_grouped =
        GetNonContractingPartitionGroupedShardingForOtherOperand(
            assume_lhs_match, output_base_shape,
            other_partitioned.hlo()->shape(), other_contracting_partitions,
            other_non_contracting_partitions, matching_contracting_partitions,
            output_other_non_contracting_partitions, other_sharding,
            output_sharding, matching_non_contracting_dims,
            other_non_contracting_dims, dims_mapping.contracting_dims);
    if (!other_grouped) {
      return absl::nullopt;
    }
    absl::optional<HloSharding> output_sharding_transposed_to_match_matching =
        hlo_sharding_util::TransposeShardingWithCollapsedDims(
            output_grouped.sharding, output_to_matching_indices,
            matching_to_output_indices);
    absl::optional<HloSharding> output_sharding_transposed_to_match_other =
        hlo_sharding_util::TransposeShardingWithCollapsedDims(
            output_grouped.sharding, output_to_other_indices,
            other_to_output_indices);
    const int64_t new_num_partitions =
        num_partitions / matching_non_contracting_partitions;
    absl::optional<WindowedEinsumConfig> e_config =
        GetWindowedEinsumConfiguration(
            new_num_partitions, output_matching_non_contracting_partitions,
            output_other_non_contracting_partitions,
            other_contracting_partitions, other_non_contracting_partitions,
            other_batch_partitions, matching_contracting_partitions,
            matching_non_contracting_partitions /
                matching_matched_non_contracting_partitions,
            matching_batch_partitions,
            ShapeSizeInBytes(other_partitioned.base_shape()),
            ShapeSizeInBytes(matching_partitioned.base_shape()) /
                matching_non_contracting_partitions,
            ShapeSizeInBytes(
                GetPerGroupBaseShape(output_grouped, output_base_shape)),
            options, output_sharding_transposed_to_match_matching,
            output_sharding_transposed_to_match_other,
            matching_grouped.sharding, other_grouped->sharding, conv_window,
            dims_mapping);
    return e_config ? new_num_partitions
                    : absl::optional<int64_t>(absl::nullopt);
  };
  absl::optional<int64_t> lhs_matching_iterations;
  if (lhs_matching_partitions != 0) {
    lhs_matching_iterations = subsequent_einsum_iterations_estimate(true);
  }
  absl::optional<int64_t> rhs_matching_iterations;
  if (rhs_matching_partitions != 0) {
    rhs_matching_iterations = subsequent_einsum_iterations_estimate(false);
  }
  return std::make_pair(lhs_matching_iterations, rhs_matching_iterations);
}

// Return if we should prioritize partitioning in the contracting dimensions
// first then non-contracting dimensions if we estimate that would be faster.
// The general idea is similar as the one in
// LhsIsBestMatchForNonContractingPartitioning with one all-gather replaced by
// reduce-scatter.
bool PrioritizeContractingDimensionsPartitioning(
    const DotConvDimsMapping& dims_mapping, const PartitionedHlo& lhs,
    const PartitionedHlo& rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const SpmdPartitionerOptions& options,
    int64_t num_partitions, int64_t lhs_non_contracting_partitions,
    int64_t rhs_non_contracting_partitions, int64_t lhs_contracting_partitions,
    int64_t rhs_contracting_partitions,
    int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions, int64_t lhs_batch_partitions,
    int64_t rhs_batch_partitions, int64_t output_batch_partitions,
    bool require_matching_devices_to_group, SpmdBuilder* b,
    const Window& conv_window,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    SpmdPartitioningVisitor* visitor) {
  const bool may_group_on_lhs_non_contracting =
      lhs_non_contracting_partitions == output_lhs_non_contracting_partitions &&
      lhs_non_contracting_partitions > 1;
  const bool may_group_on_rhs_non_contracting =
      rhs_non_contracting_partitions == output_rhs_non_contracting_partitions &&
      rhs_non_contracting_partitions > 1;
  if (!options.choose_faster_windowed_einsum_over_mem) {
    return false;
  }
  // Check only for perfect dimensions match for now.
  if (!may_group_on_lhs_non_contracting && !may_group_on_rhs_non_contracting) {
    return false;
  }
  absl::optional<int64_t> lhs_matching_iterations;
  absl::optional<int64_t> rhs_matching_iterations;
  const int64_t lhs_matching_non_contracting_partitions =
      may_group_on_lhs_non_contracting ? lhs_non_contracting_partitions : 0;
  const int64_t rhs_matching_non_contracting_partitions =
      may_group_on_rhs_non_contracting ? rhs_non_contracting_partitions : 0;
  std::tie(lhs_matching_iterations, rhs_matching_iterations) =
      EstimateWindowedEinsumIterationsForNonContractingPartitioning(
          dims_mapping, lhs, rhs, output_base_shape, output_sharding, options,
          num_partitions, lhs_non_contracting_partitions,
          rhs_non_contracting_partitions,
          lhs_matching_non_contracting_partitions,
          rhs_matching_non_contracting_partitions, lhs_contracting_partitions,
          rhs_contracting_partitions, output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, lhs_batch_partitions,
          rhs_batch_partitions, conv_window);
  if (!lhs_matching_iterations && !rhs_matching_iterations) {
    return false;
  }
  // Be conservative and handle only case where the two partitions in rhs and
  // lhs match
  if (!(lhs_contracting_partitions == rhs_contracting_partitions &&
        lhs_contracting_partitions > 1)) {
    return false;
  }
  // Estimate the iterations in the case we perform the partitioning on the
  // contracting dimensions instead.
  std::vector<int64_t> lhs_dims;
  std::vector<int64_t> rhs_dims;
  int64_t group_count = 1;
  for (const auto& dim : dims_mapping.contracting_dims) {
    lhs_dims.push_back(dim.lhs);
    rhs_dims.push_back(dim.rhs);
    group_count *= lhs.sharding().tile_assignment().dim(dim.lhs);
  }
  HloSharding lhs_sharding = HloSharding::Replicate();
  HloSharding rhs_sharding = HloSharding::Replicate();
  std::tie(lhs_sharding, rhs_sharding) =
      GetDotGroupPartitionContractingLhsRhsShardings(
          lhs, rhs, dims_mapping.contracting_dims);
  auto lhs_grouped =
      hlo_sharding_util::GroupShardingOnDims(lhs_sharding, lhs_dims);
  auto rhs_grouped =
      hlo_sharding_util::GroupShardingOnDims(rhs_sharding, rhs_dims);
  rhs_grouped = AlignGroupsWith(rhs_grouped, lhs_grouped);
  rhs_sharding = UngroupSharding(rhs_grouped);

  if (require_matching_devices_to_group && rhs.sharding() != rhs_sharding) {
    return false;
  }
  const int64_t new_num_partitions =
      num_partitions / lhs_contracting_partitions;

  HloSharding inner_output_sharding = HloSharding::Replicate();
  HloSharding outer_output_tmp_sharding = HloSharding::Replicate();
  std::vector<int64_t> output_slice_dims;
  std::tie(inner_output_sharding, outer_output_tmp_sharding) =
      GetDotGroupPartitionContractingOutputShardings(
          dims_mapping, lhs_grouped, output_base_shape, output_sharding,
          group_count, output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, output_batch_partitions,
          &output_slice_dims);
  Shape inner_output_base_shape = output_base_shape;
  if (!output_slice_dims.empty()) {
    std::vector<int64_t> non_group_dims;
    for (int64_t i = 0; i < output_base_shape.rank(); ++i) {
      if (!absl::c_linear_search(output_slice_dims, i)) {
        non_group_dims.push_back(i);
      }
    }
    inner_output_base_shape = MakePartitionedShape(
        output_base_shape,
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            output_sharding, non_group_dims));
  }
  int64_t new_output_lhs_non_contracting_partitions = 1;
  int64_t new_output_rhs_non_contracting_partitions = 1;
  if (!inner_output_sharding.IsTileMaximal()) {
    for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
      new_output_lhs_non_contracting_partitions *=
          inner_output_sharding.tile_assignment().dim(dim.output);
    }
    for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
      if (dim.output != -1) {
        new_output_rhs_non_contracting_partitions *=
            inner_output_sharding.tile_assignment().dim(dim.output);
      }
    }
  }

  const DotDimensionIndexMapping indices_map = ComputeDimensionIndexMapping(
      dims_mapping, lhs.base_shape().rank(), rhs.base_shape().rank(),
      inner_output_base_shape.rank());
  absl::optional<HloSharding> output_sharding_transposed_to_match_lhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          inner_output_sharding, indices_map.output_to_lhs_indices,
          indices_map.lhs_to_output_indices);
  absl::optional<HloSharding> output_sharding_transposed_to_match_rhs =
      hlo_sharding_util::TransposeShardingWithCollapsedDims(
          inner_output_sharding, indices_map.output_to_rhs_indices,
          indices_map.rhs_to_output_indices);
  absl::optional<WindowedEinsumConfig> e_config =
      GetWindowedEinsumConfiguration(
          new_num_partitions, new_output_lhs_non_contracting_partitions,
          new_output_rhs_non_contracting_partitions, 1,
          rhs_non_contracting_partitions, rhs_batch_partitions, 1,
          lhs_non_contracting_partitions, lhs_batch_partitions,
          ShapeSizeInBytes(GetPerGroupBaseShape(rhs_grouped, rhs.base_shape())),
          ShapeSizeInBytes(GetPerGroupBaseShape(lhs_grouped, lhs.base_shape())),
          ShapeSizeInBytes(inner_output_base_shape), options,
          output_sharding_transposed_to_match_lhs,
          output_sharding_transposed_to_match_rhs, lhs_grouped.sharding,
          rhs_grouped.sharding, conv_window, dims_mapping);
  if (!e_config) {
    return false;
  }

  int64_t num_iterations = lhs_matching_iterations ? *lhs_matching_iterations
                                                   : *rhs_matching_iterations;
  HloInstruction* other_hlo = lhs_matching_iterations ? rhs.hlo() : lhs.hlo();
  auto other_non_contracting_dims = lhs_matching_iterations
                                        ? dims_mapping.rhs_non_contracting_dims
                                        : dims_mapping.lhs_non_contracting_dims;
  auto other_sharding =
      lhs_matching_iterations ? rhs.sharding() : lhs.sharding();
  auto other_grouped = lhs_matching_iterations ? rhs_grouped : lhs_grouped;
  Shape other_base_shape =
      lhs_matching_iterations ? rhs.base_shape() : lhs.base_shape();

  const int64_t all_gather_bytes =
      ShapeUtil::ByteSizeOf(other_hlo->shape()) * new_num_partitions;
  const int64_t reduce_scatter_bytes =
      ShapeUtil::ByteSizeOf(inner_output_base_shape) / new_num_partitions *
      num_iterations;
  std::vector<int64_t> ag_replication_dims;
  ag_replication_dims.reserve(other_non_contracting_dims.size());
  for (const DotConvDimsMapping::DimsMapping& dim :
       other_non_contracting_dims) {
    ag_replication_dims.push_back(lhs_matching_iterations ? dim.rhs : dim.lhs);
  }
  auto all_gather_subgroups =
      GetPartitionGroupsForReplication(other_sharding, ag_replication_dims);
  auto reduce_scatter_subgroups = GetPartitionGroupsForReplication(
      outer_output_tmp_sharding, output_slice_dims);
  const double all_gather_time_in_ms = visitor->GetCommunicationTimeInMilliSec(
      all_gather_bytes, visitor->CreateReplicaGroups(all_gather_subgroups));
  const double reduce_scatter_time_in_ms =
      visitor->GetCommunicationTimeInMilliSec(
          reduce_scatter_bytes,
          visitor->CreateReplicaGroups(reduce_scatter_subgroups));

  Shape other_original_shape = other_hlo->shape();
  *other_hlo->mutable_shape() =
      GetPerGroupBaseShape(other_grouped, other_base_shape);
  HloInstruction* dot =
      create_sharded_dot(lhs_matching_iterations ? lhs.hlo() : other_hlo,
                         lhs_matching_iterations ? other_hlo : rhs.hlo(), b,
                         conv_window)
          .ValueOrDie();
  const double computation_time_in_ms =
      visitor->GetComputationTimeInMilliSec(dot);
  *other_hlo->mutable_shape() = other_original_shape;

  VLOG(2) << "lhs: " << lhs.hlo()->ToString() << "\n"
          << "rhs: " << rhs.hlo()->ToString() << "\n"
          << "new_num_partitions: " << new_num_partitions
          << " num_iterations: " << num_iterations << "\n"
          << "all_gather_bytes: " << all_gather_bytes
          << " reduce_scatter_bytes: " << reduce_scatter_bytes << "\n"
          << "all_gather_time_in_ms: " << all_gather_time_in_ms
          << " reduce_scatter_time_in_ms: " << reduce_scatter_time_in_ms << "\n"
          << "dot: " << dot->ToString() << "\n"
          << "computation_time_in_ms: " << computation_time_in_ms;
  if (computation_time_in_ms == 0.0 || all_gather_time_in_ms == 0.0 ||
      reduce_scatter_time_in_ms == 0.0) {
    const int64_t min_nc_iterations = std::min(
        lhs_matching_iterations ? *lhs_matching_iterations : INT64_MAX,
        rhs_matching_iterations ? *rhs_matching_iterations : INT64_MAX);
    return min_nc_iterations > new_num_partitions;
  } else if ((computation_time_in_ms <= all_gather_time_in_ms) &&
             (computation_time_in_ms <= reduce_scatter_time_in_ms)) {
    return all_gather_bytes / new_num_partitions <
           reduce_scatter_bytes / num_iterations;
  } else {
    return all_gather_time_in_ms > reduce_scatter_time_in_ms;
  }
}

// Return if it would be better to match the LHS operand or RHS operand
// of a dot for non-contracting partitioning.
bool LhsIsBestMatchForNonContractingPartitioning(
    const DotConvDimsMapping& dims_mapping, const PartitionedHlo& lhs,
    const PartitionedHlo& rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const SpmdPartitionerOptions& options,
    int64_t num_partitions, int64_t lhs_non_contracting_partitions,
    int64_t rhs_non_contracting_partitions, int64_t lhs_matching_partitions,
    int64_t rhs_matching_partitions, int64_t lhs_contracting_partitions,
    int64_t rhs_contracting_partitions,
    int64_t output_lhs_non_contracting_partitions,
    int64_t output_rhs_non_contracting_partitions, int64_t lhs_batch_partitions,
    int64_t rhs_batch_partitions, SpmdBuilder* b, const Window& conv_window,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    SpmdPartitioningVisitor* visitor) {
  const bool may_group_on_lhs_non_contracting =
      lhs_non_contracting_partitions == output_lhs_non_contracting_partitions &&
      lhs_non_contracting_partitions > 1;
  const bool may_group_on_rhs_non_contracting =
      rhs_non_contracting_partitions == output_rhs_non_contracting_partitions &&
      rhs_non_contracting_partitions > 1;
  // If both match output non-contracting dimensions, choose the one which
  // will result in smaller replication of the other operand.
  bool lhs_matching = may_group_on_lhs_non_contracting &&
                      (!may_group_on_rhs_non_contracting ||
                       lhs_non_contracting_partitions *
                               ShapeUtil::ByteSizeOf(rhs.hlo()->shape()) <
                           rhs_non_contracting_partitions *
                               ShapeUtil::ByteSizeOf(lhs.hlo()->shape()));
  // If both grouping are available and the option to choose faster windowed
  // einsums vs saving memory is enabled then try to determine which of the
  // operands will have more overlapping benefits for the windowed einsum
  // when matched (if a windowed einsum is gonna be generated at all).
  // 1) When computation is shorter than both all_gathers, we choose to overlap
  // with the smaller all_gather as it has potentially smaller extra
  // collective-permute overhead outside of the while loop; 2) Otherwise, we
  // choose the all_gather with longer runtime to overlap with.
  if (may_group_on_lhs_non_contracting && may_group_on_rhs_non_contracting &&
      options.choose_faster_windowed_einsum_over_mem) {
    const DotDimensionIndexMapping indices_map = ComputeDimensionIndexMapping(
        dims_mapping, lhs.base_shape().rank(), rhs.base_shape().rank(),
        output_base_shape.rank());
    absl::optional<int64_t> lhs_matching_iterations;
    absl::optional<int64_t> rhs_matching_iterations;
    std::tie(lhs_matching_iterations, rhs_matching_iterations) =
        EstimateWindowedEinsumIterationsForNonContractingPartitioning(
            dims_mapping, lhs, rhs, output_base_shape, output_sharding, options,
            num_partitions, lhs_non_contracting_partitions,
            rhs_non_contracting_partitions, lhs_matching_partitions,
            rhs_matching_partitions, lhs_contracting_partitions,
            rhs_contracting_partitions, output_lhs_non_contracting_partitions,
            output_rhs_non_contracting_partitions, lhs_batch_partitions,
            rhs_batch_partitions, conv_window);
    if (lhs_matching_iterations && rhs_matching_iterations) {
      const int64_t lhs_all_gather_bytes =
          ShapeUtil::ByteSizeOf(lhs.hlo()->shape()) *
          rhs_non_contracting_partitions;
      const int64_t rhs_all_gather_bytes =
          ShapeUtil::ByteSizeOf(rhs.hlo()->shape()) *
          lhs_non_contracting_partitions;
      auto lhs_grouped =
          GetNonContractingPartitionGroupedShardingForMatchedOperand(
              /*lhs_matching=*/true, lhs.sharding(), output_sharding,
              dims_mapping.lhs_non_contracting_dims);
      auto lhs_all_gather_subgroups = lhs_grouped.device_groups;
      auto rhs_grouped =
          GetNonContractingPartitionGroupedShardingForMatchedOperand(
              /*lhs_matching=*/false, rhs.sharding(), output_sharding,
              dims_mapping.rhs_non_contracting_dims);
      auto rhs_all_gather_subgroups = rhs_grouped.device_groups;
      const double lhs_all_gather_time_in_ms =
          visitor->GetCommunicationTimeInMilliSec(
              lhs_all_gather_bytes,
              visitor->CreateReplicaGroups(lhs_all_gather_subgroups));
      const double rhs_all_gather_time_in_ms =
          visitor->GetCommunicationTimeInMilliSec(
              rhs_all_gather_bytes,
              visitor->CreateReplicaGroups(rhs_all_gather_subgroups));

      HloInstruction* compute_lhs = lhs.hlo();
      Shape lhs_original_shape = compute_lhs->shape();
      *compute_lhs->mutable_shape() =
          GetPerGroupBaseShape(lhs_grouped, lhs.base_shape());
      HloInstruction* compute_rhs = rhs.hlo();
      Shape rhs_original_shape = compute_rhs->shape();
      *compute_rhs->mutable_shape() =
          GetPerGroupBaseShape(rhs_grouped, rhs.base_shape());
      HloInstruction* dot =
          create_sharded_dot(compute_lhs, compute_rhs, b, conv_window)
              .ValueOrDie();
      const double computation_time_in_ms =
          visitor->GetComputationTimeInMilliSec(dot);
      *compute_lhs->mutable_shape() = lhs_original_shape;
      *compute_rhs->mutable_shape() = rhs_original_shape;

      VLOG(2) << "lhs: " << lhs.hlo()->ToString() << "\n"
              << "rhs: " << rhs.hlo()->ToString() << "\n"
              << "lhs_non_contracting_partitions: "
              << lhs_non_contracting_partitions
              << " rhs_non_contracting_partitions: "
              << rhs_non_contracting_partitions << "\n"
              << "lhs_matching_iterations: " << *lhs_matching_iterations
              << " rhs_matching_iterations: " << *rhs_matching_iterations
              << "\n"
              << "lhs_all_gather_bytes: " << lhs_all_gather_bytes
              << " rhs_all_gather_bytes: " << rhs_all_gather_bytes << "\n"
              << "lhs_all_gather_time_in_ms: " << lhs_all_gather_time_in_ms
              << " rhs_all_gather_time_in_ms: " << rhs_all_gather_time_in_ms
              << "\n"
              << "dot: " << dot->ToString() << "\n"
              << "computation_time_in_ms: " << computation_time_in_ms;
      if (computation_time_in_ms == 0.0 || lhs_all_gather_time_in_ms == 0.0 ||
          rhs_all_gather_time_in_ms == 0.0) {
        lhs_matching = *lhs_matching_iterations < *rhs_matching_iterations;
      } else if ((computation_time_in_ms <= lhs_all_gather_time_in_ms) &&
                 (computation_time_in_ms <= rhs_all_gather_time_in_ms)) {
        lhs_matching = lhs_all_gather_bytes / rhs_non_contracting_partitions >
                       rhs_all_gather_bytes / lhs_non_contracting_partitions;
      } else {
        lhs_matching = lhs_all_gather_time_in_ms > rhs_all_gather_time_in_ms;
      }
    }
  }
  return lhs_matching;
}

// Recursive partitioning function. If there are partial dimensions matching
// in the operands and output, group the devices and recursively partition
// the in-group dot.
StatusOr<HloInstruction*> PartitionDot(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64_t num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    bool require_matching_devices_to_group,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    SpmdPartitioningVisitor* visitor) {
  // If lhs‘ hlo and rhs' hlo are identical, make a copy for rhs.
  if (lhs.hlo() == rhs.hlo()) {
    auto copy_hlo = b->AddInstruction(HloInstruction::CreateUnary(
        rhs.hlo()->shape(), HloOpcode::kCopy, rhs.hlo()));
    copy_hlo->set_sharding(rhs.sharding());
    rhs = PartitionedHlo(copy_hlo, rhs.base_shape(), rhs.state());
  }

  // lhs_rhs_or_output: 0 lhs, 1 rhs, 2 output.
  auto get_partitions_for_dims =
      [&](const HloSharding& sharding,
          absl::Span<const DotConvDimsMapping::DimsMapping> dims,
          int lhs_rhs_or_output) {
        int64_t partitions = 1;
        if (sharding.IsTileMaximal()) {
          return partitions;
        }
        for (const auto& dim : dims) {
          if (lhs_rhs_or_output == 0) {
            partitions *= sharding.tile_assignment().dim(dim.lhs);
          } else if (lhs_rhs_or_output == 1) {
            partitions *= sharding.tile_assignment().dim(dim.rhs);
          } else {
            CHECK_EQ(lhs_rhs_or_output, 2);
            partitions *= sharding.tile_assignment().dim(dim.output);
          }
        }
        return partitions;
      };
  const int64_t lhs_batch_partitions =
      get_partitions_for_dims(lhs.sharding(), dims_mapping.batch_dims, 0);
  const int64_t rhs_batch_partitions =
      get_partitions_for_dims(rhs.sharding(), dims_mapping.batch_dims, 1);
  const int64_t output_batch_partitions =
      get_partitions_for_dims(output_sharding, dims_mapping.batch_dims, 2);
  const int64_t lhs_contracting_partitions =
      get_partitions_for_dims(lhs.sharding(), dims_mapping.contracting_dims, 0);
  const int64_t rhs_contracting_partitions =
      get_partitions_for_dims(rhs.sharding(), dims_mapping.contracting_dims, 1);
  const int64_t lhs_non_contracting_partitions = get_partitions_for_dims(
      lhs.sharding(), dims_mapping.lhs_non_contracting_dims, 0);
  const int64_t rhs_non_contracting_partitions = get_partitions_for_dims(
      rhs.sharding(), dims_mapping.rhs_non_contracting_dims, 1);
  const int64_t output_lhs_non_contracting_partitions = get_partitions_for_dims(
      output_sharding, dims_mapping.lhs_non_contracting_dims, 2);
  const int64_t output_rhs_non_contracting_partitions = get_partitions_for_dims(
      output_sharding, dims_mapping.rhs_non_contracting_dims, 2);
  const int64_t lhs_conv_spatial_partitions = get_partitions_for_dims(
      lhs.sharding(), dims_mapping.conv_spatial_dims, 0);
  const int64_t rhs_conv_spatial_partitions = get_partitions_for_dims(
      rhs.sharding(), dims_mapping.conv_spatial_dims, 1);
  const int64_t output_conv_spatial_partitions = get_partitions_for_dims(
      output_sharding, dims_mapping.conv_spatial_dims, 2);
  // Before we find partial matches along the dimensions, invoke base case
  // again without may_reshard_without_detecting_match.

  // Try partition the purely spatially-partitioned convolution with
  // convolution spatial dimension partitioned or depthwise parallel
  // dimension partitioned.
  bool is_conv_spatial_dim_partitioned =
      (lhs_conv_spatial_partitions > 1 || rhs_conv_spatial_partitions > 1 ||
       output_conv_spatial_partitions > 1);
  bool is_conv_batch_or_contracting_dim_partitioned =
      (lhs_batch_partitions > 1 || rhs_batch_partitions > 1 ||
       output_batch_partitions > 1 ||
       (lhs_contracting_partitions > 1 && rhs_contracting_partitions > 1));
  if ((!dims_mapping.conv_spatial_dims.empty() &&
       is_conv_spatial_dim_partitioned &&
       !is_conv_batch_or_contracting_dim_partitioned) ||
      (original_hlo->opcode() == HloOpcode::kConvolution &&
       (original_hlo->batch_group_count() > 1 ||
        original_hlo->feature_group_count() > 1))) {
    // Partition with kernel_input_feature_dim > 1 and feature_group_count >
    // 1 is not supported.
    const auto& dnums = original_hlo->convolution_dimension_numbers();
    if (original_hlo->feature_group_count() > 1 &&
        rhs.hlo()->shape().dimensions(dnums.kernel_input_feature_dimension()) >
            1) {
      return nullptr;
    }

    TF_ASSIGN_OR_RETURN(
        auto partitioned_conv,
        PartitionConvolution(lhs, rhs, output_base_shape, output_sharding,
                             dims_mapping, create_sharded_dot, conv_window,
                             original_hlo, num_partitions, options,
                             lhs.state().partition_id, module, b));

    if (partitioned_conv) {
      return partitioned_conv;
    }

    // Recursively partition on different types of dimensions for
    // convolution. Case 0.a: Group partitions by feature group count.
    if (original_hlo->feature_group_count() > 1 ||
        original_hlo->batch_group_count() > 1) {
      absl::optional<DotConvDimsMapping> new_dims_mapping;
      if (original_hlo->feature_group_count() > 1) {
        const int64_t input_feature_dim =
            original_hlo->convolution_dimension_numbers()
                .input_feature_dimension();
        const int64_t kernel_output_feature_dim =
            original_hlo->convolution_dimension_numbers()
                .kernel_output_feature_dimension();
        // If the input and output feature dims are not equal, we require the
        // feature_group_count to be evenly partitioned; otherwise, there will
        // be different padding in the input/output.
        // TODO(xla): Use halo exchange to solve this problem. Can be a
        // preprocessing that uses padding/slicing to make the shape evenly
        // shardable.
        if (lhs.base_shape().dimensions(input_feature_dim) ==
                rhs.base_shape().dimensions(kernel_output_feature_dim) ||
            (lhs.sharding().IsTiled() &&
             original_hlo->feature_group_count() %
                     ShardCountAtDim(lhs.sharding(), input_feature_dim) ==
                 0)) {
          new_dims_mapping = ConvertDimsMappingWithFeatureGroupCount(
              dims_mapping, original_hlo);
        }
      }

      if (original_hlo->batch_group_count() > 1) {
        const int64_t input_batch_dim =
            original_hlo->convolution_dimension_numbers()
                .input_batch_dimension();
        const int64_t kernel_output_feature_dim =
            original_hlo->convolution_dimension_numbers()
                .kernel_output_feature_dimension();
        if (lhs.base_shape().dimensions(input_batch_dim) ==
                rhs.base_shape().dimensions(kernel_output_feature_dim) ||
            (lhs.sharding().IsTiled() &&
             original_hlo->batch_group_count() %
                     ShardCountAtDim(lhs.sharding(), input_batch_dim) ==
                 0)) {
          new_dims_mapping =
              ConvertDimsMappingWithBatchGroupCount(dims_mapping, original_hlo);
        }
      }
      if (!new_dims_mapping.has_value()) {
        return nullptr;
      }

      const int64_t conv_lhs_contracting_partitions = get_partitions_for_dims(
          lhs.sharding(), new_dims_mapping->contracting_dims, 0);
      const int64_t conv_rhs_contracting_partitions = get_partitions_for_dims(
          rhs.sharding(), new_dims_mapping->contracting_dims, 1);
      const int64_t conv_lhs_non_contracting_partitions =
          get_partitions_for_dims(
              lhs.sharding(), new_dims_mapping->lhs_non_contracting_dims, 0);
      const int64_t conv_rhs_non_contracting_partitions =
          get_partitions_for_dims(
              rhs.sharding(), new_dims_mapping->rhs_non_contracting_dims, 1);
      const int64_t conv_lhs_batch_partitions = get_partitions_for_dims(
          lhs.sharding(), new_dims_mapping->batch_dims, 0);
      const int64_t conv_rhs_batch_partitions = get_partitions_for_dims(
          rhs.sharding(), new_dims_mapping->batch_dims, 1);
      const int64_t conv_output_batch_partitions = get_partitions_for_dims(
          output_sharding, new_dims_mapping->batch_dims, 2);
      if ((conv_lhs_batch_partitions == conv_output_batch_partitions ||
           conv_rhs_batch_partitions == conv_output_batch_partitions) &&
          conv_output_batch_partitions > 1) {
        TF_ASSIGN_OR_RETURN(
            auto try_partitioned_conv,
            PartitionDotGroupOnBatch(
                lhs, rhs, output_base_shape, output_sharding, *new_dims_mapping,
                num_partitions, conv_lhs_contracting_partitions,
                conv_rhs_contracting_partitions,
                conv_lhs_non_contracting_partitions,
                conv_rhs_non_contracting_partitions, create_sharded_dot,
                conv_window, module, original_hlo,
                require_matching_devices_to_group, options, b,
                windowed_dot_general_loops, visitor));
        if (try_partitioned_conv) {
          return try_partitioned_conv;
        }
      }
      return nullptr;
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto try_partitioned_dot,
      PartitionBaseCase(
          lhs, rhs, output_base_shape, output_sharding, dims_mapping,
          num_partitions, create_sharded_dot, conv_window, module, original_hlo,
          lhs_batch_partitions, rhs_batch_partitions, output_batch_partitions,
          lhs_contracting_partitions, rhs_contracting_partitions,
          lhs_non_contracting_partitions, rhs_non_contracting_partitions,
          output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, options, b,
          windowed_dot_general_loops,
          /*may_reshard_without_detecting_match=*/false, visitor));
  if (try_partitioned_dot) {
    return try_partitioned_dot;
  }

  // Recursively partition on different types of dimensions.
  //
  // Case 1: Group partitions by batch.
  if ((lhs_batch_partitions == output_batch_partitions ||
       rhs_batch_partitions == output_batch_partitions) &&
      output_batch_partitions > 1) {
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDotGroupOnBatch(
            lhs, rhs, output_base_shape, output_sharding, dims_mapping,
            num_partitions, lhs_contracting_partitions,
            rhs_contracting_partitions, lhs_non_contracting_partitions,
            rhs_non_contracting_partitions, create_sharded_dot, conv_window,
            module, original_hlo, require_matching_devices_to_group, options, b,
            windowed_dot_general_loops, visitor));
    if (dot) {
      return dot;
    }
  }

  // Case 2: Group partitions by non-contracting dimensions.
  const bool may_group_on_lhs_non_contracting =
      lhs_non_contracting_partitions == output_lhs_non_contracting_partitions &&
      lhs_non_contracting_partitions > 1;
  const bool may_group_on_rhs_non_contracting =
      rhs_non_contracting_partitions == output_rhs_non_contracting_partitions &&
      rhs_non_contracting_partitions > 1;
  bool lhs_matching = false;
  std::vector<DotConvDimsMapping::DimsMapping> matching_dims;
  if (may_group_on_lhs_non_contracting || may_group_on_rhs_non_contracting) {
    lhs_matching = LhsIsBestMatchForNonContractingPartitioning(
        dims_mapping, lhs, rhs, output_base_shape, output_sharding, options,
        num_partitions, lhs_non_contracting_partitions,
        rhs_non_contracting_partitions, lhs_non_contracting_partitions,
        rhs_non_contracting_partitions, lhs_contracting_partitions,
        rhs_contracting_partitions, output_lhs_non_contracting_partitions,
        output_rhs_non_contracting_partitions, lhs_batch_partitions,
        rhs_batch_partitions, b, conv_window, create_sharded_dot, visitor);
    matching_dims = lhs_matching ? dims_mapping.lhs_non_contracting_dims
                                 : dims_mapping.rhs_non_contracting_dims;
  } else if (lhs_non_contracting_partitions > 1 &&
             output_lhs_non_contracting_partitions > 1) {
    lhs_matching = true;
    for (const auto& dim : dims_mapping.lhs_non_contracting_dims) {
      int64_t lhs_partitions = lhs.sharding().tile_assignment().dim(dim.lhs);
      if (lhs_partitions > 1 &&
          lhs_partitions == output_sharding.tile_assignment().dim(dim.output)) {
        matching_dims.push_back(dim);
      }
    }
  } else if (rhs_non_contracting_partitions > 1 &&
             output_rhs_non_contracting_partitions > 1) {
    lhs_matching = false;
    for (const auto& dim : dims_mapping.rhs_non_contracting_dims) {
      int64_t rhs_partitions = rhs.sharding().tile_assignment().dim(dim.rhs);
      if (rhs_partitions > 1 &&
          rhs_partitions == output_sharding.tile_assignment().dim(dim.output)) {
        matching_dims.push_back(dim);
      }
    }
  }
  const bool prioritize_contracting_for_faster_windowed_einsum =
      PrioritizeContractingDimensionsPartitioning(
          dims_mapping, lhs, rhs, output_base_shape, output_sharding, options,
          num_partitions, lhs_non_contracting_partitions,
          rhs_non_contracting_partitions, lhs_contracting_partitions,
          rhs_contracting_partitions, output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, lhs_batch_partitions,
          rhs_batch_partitions, output_batch_partitions,
          require_matching_devices_to_group, b, conv_window, create_sharded_dot,
          visitor);
  if (!(matching_dims.empty() ||
        prioritize_contracting_for_faster_windowed_einsum)) {
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDotGroupOnNonContracting(
            lhs_matching, lhs_matching ? lhs : rhs, lhs_matching ? rhs : lhs,
            lhs_matching ? lhs_contracting_partitions
                         : rhs_contracting_partitions,
            lhs_matching ? rhs_contracting_partitions
                         : lhs_contracting_partitions,
            matching_dims,
            lhs_matching ? rhs_non_contracting_partitions
                         : lhs_non_contracting_partitions,
            lhs_matching ? output_rhs_non_contracting_partitions
                         : output_lhs_non_contracting_partitions,
            output_base_shape, output_sharding, dims_mapping, num_partitions,
            create_sharded_dot, conv_window, module, original_hlo,
            require_matching_devices_to_group, options, b,
            windowed_dot_general_loops, visitor));
    if (dot) {
      return dot;
    }
  }

  // Case 3: Group partitions by contracting dimensions.
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions > 1) {
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDotGroupOnContracting(
            lhs, rhs, dims_mapping.contracting_dims, output_batch_partitions,
            output_lhs_non_contracting_partitions,
            output_rhs_non_contracting_partitions, output_base_shape,
            output_sharding, dims_mapping, num_partitions, create_sharded_dot,
            conv_window, module, original_hlo,
            require_matching_devices_to_group, options, b,
            windowed_dot_general_loops, visitor));
    if (dot) {
      return dot;
    }
  }
  if (lhs_contracting_partitions > 1 && rhs_contracting_partitions > 1) {
    // If part of contracting dims match, try them.
    std::vector<DotConvDimsMapping::DimsMapping> matching_dims;
    for (const auto& dim : dims_mapping.contracting_dims) {
      int64_t lhs_partitions = lhs.sharding().tile_assignment().dim(dim.lhs);
      if (lhs_partitions > 1 &&
          lhs_partitions == rhs.sharding().tile_assignment().dim(dim.rhs)) {
        matching_dims.push_back(dim);
      }
    }
    if (!matching_dims.empty()) {
      TF_ASSIGN_OR_RETURN(
          auto dot, PartitionDotGroupOnContracting(
                        lhs, rhs, matching_dims, output_batch_partitions,
                        output_lhs_non_contracting_partitions,
                        output_rhs_non_contracting_partitions,
                        output_base_shape, output_sharding, dims_mapping,
                        num_partitions, create_sharded_dot, conv_window, module,
                        original_hlo, require_matching_devices_to_group,
                        options, b, windowed_dot_general_loops, visitor));
      if (dot) {
        return dot;
      }
    }
  }

  // Case 4: If operands are replicated but output is partially replicated,
  // recursive call with partial replication removed.
  if (lhs.sharding().IsReplicated() && rhs.sharding().IsReplicated() &&
      output_sharding.ReplicateOnLastTileDim()) {
    auto grouped_output = hlo_sharding_util::GroupShardingOnDims(
        output_sharding, {output_base_shape.rank()});
    auto inner_state = CreatePerGroupPartitioningState(
        lhs.state(), grouped_output.device_groups, b);
    TF_ASSIGN_OR_RETURN(
        auto dot,
        PartitionDot(PartitionedHlo(lhs.hlo(), lhs.base_shape(), inner_state),
                     PartitionedHlo(rhs.hlo(), rhs.base_shape(), inner_state),
                     output_base_shape, grouped_output.sharding, dims_mapping,
                     output_sharding.NumTiles(), create_sharded_dot,
                     conv_window, module, original_hlo, options, b,
                     windowed_dot_general_loops, visitor));
    if (dot) {
      return dot;
    }
  }

  // We failed to find partial matches, invoke base case again with
  // may_reshard_without_detecting_match.
  TF_ASSIGN_OR_RETURN(
      auto dot,
      PartitionBaseCase(
          lhs, rhs, output_base_shape, output_sharding, dims_mapping,
          num_partitions, create_sharded_dot, conv_window, module, original_hlo,
          lhs_batch_partitions, rhs_batch_partitions, output_batch_partitions,
          lhs_contracting_partitions, rhs_contracting_partitions,
          lhs_non_contracting_partitions, rhs_non_contracting_partitions,
          output_lhs_non_contracting_partitions,
          output_rhs_non_contracting_partitions, options, b,
          windowed_dot_general_loops,
          /*may_reshard_without_detecting_match=*/true, visitor));
  if (dot) {
    return dot;
  }
  return nullptr;
}

StatusOr<HloInstruction*> PartitionDot(
    PartitionedHlo lhs, PartitionedHlo rhs, const Shape& output_base_shape,
    const HloSharding& output_sharding, const DotConvDimsMapping& dims_mapping,
    int64_t num_partitions,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot,
    const Window& conv_window, HloModule* module, HloInstruction* original_hlo,
    const SpmdPartitionerOptions& options, SpmdBuilder* b,
    std::vector<SpmdPartitioningVisitor::WindowedDotGeneralLoop>*
        windowed_dot_general_loops,
    SpmdPartitioningVisitor* visitor) {
  // First try partitioning without resharding the groups, then try allow
  // resharding the groups.
  for (bool require_matching_devices_to_group : {true, false}) {
    TF_ASSIGN_OR_RETURN(
        auto try_partition,
        PartitionDot(lhs, rhs, output_base_shape, output_sharding, dims_mapping,
                     num_partitions, create_sharded_dot, conv_window, module,
                     original_hlo, require_matching_devices_to_group, options,
                     b, windowed_dot_general_loops, visitor));
    if (try_partition) {
      return try_partition;
    }
  }

  // Default action.
  TF_ASSIGN_OR_RETURN(
      auto dot, create_sharded_dot(lhs.Replicate().hlo(), rhs.Replicate().hlo(),
                                   b, conv_window));
  dot->set_sharding(HloSharding::Replicate());
  return PartitionedHlo(dot, output_base_shape, lhs.state())
      .Reshard(output_sharding)
      .hlo();
}

}  // namespace

Status SpmdPartitioningVisitor::HandleDotHelper(
    HloInstruction* hlo, const DotConvDimsMapping& dims_mapping,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*,
        const Window& conv_window)>& create_sharded_dot) {
  if (hlo->sharding().HasUniqueDevice()) {
    return DefaultAction(hlo);
  }
  auto& lhs = GetPartitionedHlo(hlo->operand(0));
  auto& rhs = GetPartitionedHlo(hlo->operand(1));
  Window conv_window;
  if (hlo->opcode() == HloOpcode::kConvolution) {
    conv_window = hlo->window();
  }

  TF_ASSIGN_OR_RETURN(
      auto partitioned_dot,
      PartitionDot(lhs, rhs, hlo->shape(), hlo->sharding(), dims_mapping,
                   num_partitions_, create_sharded_dot, conv_window, module_,
                   hlo, options_, &b_, &windowed_dot_general_loops_, this));
  SetPartitionedHlo(hlo, [&] { return partitioned_dot; });
  return Status::OK();
}

namespace {

// Finds a cluster of nodes that produce the inputs for `hlo` which only
// depend on small operands, which means the cluster should start with
// broadcasts, constants and iotas. All other internal nodes must be
// non-side-effecting elemntwise ops. Returns the set of nodes, and the small
// operands. E.g., for the following graph,
//
//     a -> broadcast -> multiply
//     iota  ---> add--/
//     constant/
//
// FindInputNodesIfOnlyDependOnSmallOperands(multiply) will return
//    <{broadcast, iota, constant, add, multiply}, [a]>.
std::pair<absl::flat_hash_set<HloInstruction*>, std::vector<HloInstruction*>>
FindInputNodesIfOnlyDependOnSmallOperands(HloInstruction* hlo) {
  absl::flat_hash_set<HloInstruction*> nodes_found;
  std::vector<HloInstruction*> new_operands;
  absl::flat_hash_set<const HloInstruction*> new_operands_set;
  std::vector<HloInstruction*> worklist;
  worklist.push_back(hlo);
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    if (nodes_found.count(inst) > 0) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kBroadcast ||
        inst->opcode() == HloOpcode::kConstant ||
        inst->opcode() == HloOpcode::kIota) {
      nodes_found.insert(inst);
      for (auto o : inst->operands()) {
        auto res = new_operands_set.emplace(o);
        if (res.second) {
          new_operands.push_back(o);
        }
      }
    } else if (inst->IsElementwise() && !inst->HasSideEffectNoRecurse() &&
               absl::c_all_of(inst->operands(),
                              [inst](const HloInstruction* o) {
                                return ShapeUtil::CompatibleIgnoringElementType(
                                    o->shape(), inst->shape());
                              })) {
      nodes_found.insert(inst);
      for (auto o : inst->operands()) {
        worklist.push_back(o);
      }
    } else {
      nodes_found.clear();
      new_operands.clear();
      break;
    }
  }
  return {std::move(nodes_found), std::move(new_operands)};
}

// Moves a cluster of memory-reducing nodes into the windowed dot-general loop
// on contracting dimensions. Such a loop has a dynamic slice on the
// non-windowed operand. If we move the input nodes into the loop, the
// dynamic-slice could be merged with them by later optimization passes, which
// reduces memory.
//
// small_operands             small_operands
//        |                          |
// input_nodes                loop { |
//        |          =>         input_nodes
// loop { |                          |
//    dynamic-slice             dynamic-slice
//    ...                       ...
// }                          }
//
// Later optimization passes (TpuPadSliceMover) will merge the dynamic slice
// with the input nodes.
Status SinkInputNodesIntoWindowedDotGeneralLoopOnContractingDimensions(
    HloInstruction* loop, int64_t non_windowed_operand_index) {
  auto input_tuple = loop->mutable_operand(0);
  auto old_operand = input_tuple->mutable_operand(non_windowed_operand_index);
  auto input_nodes = FindInputNodesIfOnlyDependOnSmallOperands(old_operand);
  auto to_sink = std::move(input_nodes.first);
  auto new_operands = std::move(input_nodes.second);
  if (to_sink.empty()) {
    return Status::OK();
  }
  auto computation = loop->parent();
  // Replace the old operand with a tuple of the found small operands.
  auto new_input_subtuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  TF_RETURN_IF_ERROR(input_tuple->ReplaceOperandWithDifferentShape(
      non_windowed_operand_index, new_input_subtuple));

  auto body = loop->while_body();
  auto body_param = body->parameter_instruction(0);
  auto old_body_param_users = body_param->users();
  // Update all tuple shapes.
  for (auto tuple : std::vector<HloInstruction*>{
           input_tuple, loop, loop->while_condition()->parameter_instruction(0),
           body_param, body->root_instruction()}) {
    *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(),
                                   {non_windowed_operand_index}) =
        new_input_subtuple->shape();
  }
  // Now update the loop body.
  auto new_operand_tuple_inside =
      body->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_input_subtuple->shape(), body_param, non_windowed_operand_index));
  TF_RETURN_IF_ERROR(body->root_instruction()->ReplaceOperandWithDifferentShape(
      non_windowed_operand_index, new_operand_tuple_inside));

  // Create nodes inside the loop body.
  std::vector<HloInstruction*> worklist;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> outside_to_inside;
  auto add_users_if_available = [&](HloInstruction* inst) {
    for (auto u : inst->users()) {
      if (outside_to_inside.count(u) == 0 && to_sink.count(u) > 0 &&
          absl::c_all_of(u->operands(), [&](const HloInstruction* o) {
            return outside_to_inside.count(o) > 0;
          })) {
        worklist.push_back(u);
      }
    }
  };
  for (int64_t i = 0; i < new_operands.size(); ++i) {
    outside_to_inside[new_operands[i]] =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_operands[i]->shape(), new_operand_tuple_inside, i));
    add_users_if_available(new_operands[i]);
  }
  // HLOs to sink without operands.
  std::vector<HloInstruction*> nullaries_to_sink;
  for (auto inst : to_sink) {
    if (inst->operand_count() == 0) {
      nullaries_to_sink.push_back(inst);
    }
  }
  // Sort nullaries_to_sink to make it deterministic.
  absl::c_sort(nullaries_to_sink,
               [](const HloInstruction* a, const HloInstruction* b) {
                 return a->unique_id() < b->unique_id();
               });
  worklist.reserve(nullaries_to_sink.size());
  for (auto inst : nullaries_to_sink) {
    worklist.push_back(inst);
  }
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    std::vector<HloInstruction*> inst_new_operands(inst->operand_count());
    for (int64_t i = 0; i < inst->operand_count(); ++i) {
      inst_new_operands[i] = outside_to_inside[inst->operand(i)];
    }
    outside_to_inside[inst] = body->AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), inst_new_operands));
    add_users_if_available(inst);
  }
  TF_RET_CHECK(outside_to_inside.count(old_operand) > 0);
  for (auto ou : old_body_param_users) {
    if (ou->opcode() == HloOpcode::kGetTupleElement &&
        ou->tuple_index() == non_windowed_operand_index) {
      TF_RETURN_IF_ERROR(
          ou->ReplaceAllUsesWith(outside_to_inside[old_operand]));
      TF_RETURN_IF_ERROR(body->RemoveInstruction(ou));
    }
  }
  return Status::OK();
}

// Moves a cluster of memory-reducing nodes (with reduce nodes at the end)
// into the windowed dot-general loop on non-contracting dimensions. Such a
// loop has a dynamic-update-slice at the output. If we move the user nodes
// into the loop and before the dynamic-update-slice, the user nodes can
// operate on smaller shapes, which reduces memory.
//
// small_operands                   small_operands
//  | |                 =>                  | |
//  | |  loop {                     loop {  | |
//  | |    conv                             | broadcast      conv
//  | |      |                              |     |           /
//  | | dynamic-update-slice                |  dynamic-slice /
//  | |         |                           |     |         /
//  | |  }      |                           |  multiply-----
//  |broadcast  /                           |    /
//  | |        /                            reduce
//  |multiply--                             |
//  \ |                                dynamic-update-slice
//   reduce                         }
//
// Later optimization passes (TpuPadSliceMover) will merge the dynamic slice
// with the input nodes (broadcast).
Status MoveUsersIntoWindowedDotGeneralLoopOnNonContractingDimensions(
    HloInstruction* loop, const SpmdPartitionerOptions& options) {
  CHECK_EQ(loop->user_count(), 1);
  // There should be a single direct user of the while loop, which is the
  // gte for element 2, i.e., the dot output.
  auto* user_gte = loop->users().front();
  CHECK_EQ(user_gte->opcode(), HloOpcode::kGetTupleElement);
  CHECK_EQ(user_gte->tuple_index(), 2);
  auto* computation = loop->parent();

  // Find the reduce outputs and the input nodes they depend on, if input
  // nodes only have small operands.
  absl::flat_hash_set<HloInstruction*> to_move;
  std::vector<HloInstruction*> new_operands;
  absl::flat_hash_set<const HloInstruction*> new_operands_set;
  std::vector<HloInstruction*> reduce_outputs;
  std::vector<HloInstruction*> worklist;
  Shape padded_shape = user_gte->shape();
  Shape unpadded_shape = user_gte->shape();
  auto* original_output = user_gte;

  if (user_gte->user_count() == 1 &&
      user_gte->users().back()->opcode() == HloOpcode::kSlice) {
    original_output = user_gte->users().back();
    unpadded_shape = original_output->shape();
  }
  for (auto* u : original_output->users()) {
    worklist.push_back(u);
  }
  to_move.insert(original_output);
  while (!worklist.empty()) {
    auto* inst = worklist.back();
    worklist.pop_back();
    if (to_move.count(inst) > 0) {
      continue;
    }
    // We only support reduces with simple reduction function, since we may
    // need to accumulate across iterations manually.
    if (inst->opcode() == HloOpcode::kReduce &&
        inst->to_apply()->instruction_count() == 3 &&
        inst->to_apply()->num_parameters() == 2 &&
        inst->to_apply()->root_instruction()->IsElementwise()) {
      to_move.insert(inst);
      auto* other_operand = inst->mutable_operand(1);
      auto res = new_operands_set.emplace(other_operand);
      if (res.second) {
        new_operands.push_back(other_operand);
      }
      reduce_outputs.push_back(inst);
    } else if (inst != computation->root_instruction() &&
               inst->user_count() > 0 && inst->IsElementwise() &&
               !inst->HasSideEffectNoRecurse() &&
               absl::c_all_of(inst->operands(),
                              [inst](const HloInstruction* o) {
                                return ShapeUtil::CompatibleIgnoringElementType(
                                    o->shape(), inst->shape());
                              })) {
      // For an elementwise op, we need to make sure that they depend on only
      // nodes already in to_move and nodes with small operands.
      bool can_include = true;
      for (auto* operand : inst->operands()) {
        if (to_move.count(operand) > 0) {
          continue;
        }
        auto find_result = FindInputNodesIfOnlyDependOnSmallOperands(operand);
        if (find_result.first.empty()) {
          can_include = false;
          break;
        }
        for (auto* n : find_result.first) {
          to_move.insert(n);
        }
        for (auto* new_operand : find_result.second) {
          auto res = new_operands_set.insert(new_operand);
          if (res.second) {
            new_operands.push_back(new_operand);
          }
        }
      }
      if (!can_include) {
        to_move.clear();
        break;
      }
      to_move.insert(inst);
      for (auto* u : inst->users()) {
        worklist.push_back(u);
      }
    } else {
      to_move.clear();
      break;
    }
  }
  // If nothing is found, to_move could contain only original_output, or
  // cleared by the above code.
  if (to_move.size() <= 1) {
    return Status::OK();
  }

  // We will replace the original loop output with reduce-shape outputs.
  // Create the initial buffers before the loop.
  for (auto* out : reduce_outputs) {
    Shape padded_out_shape = out->shape();
    int64_t operand_dim = 0;
    int64_t output_dim = 0;
    while (output_dim < padded_out_shape.rank()) {
      if (absl::c_linear_search(out->dimensions(), operand_dim)) {
        // Dimension colapsed.
        ++operand_dim;
        continue;
      }
      // Kept dimensions have the same size of the padded shape.
      padded_out_shape.set_dimensions(output_dim,
                                      padded_shape.dimensions(operand_dim));
      ++operand_dim;
      ++output_dim;
    }
    auto* broadcast =
        computation->AddInstruction(HloInstruction::CreateBroadcast(
            padded_out_shape,
            computation->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(out->shape().element_type()))),
            {}));
    new_operands.push_back(broadcast);
  }

  auto* input_tuple = loop->mutable_operand(0);
  // Create the new input subtuple that contains the small operands and the
  // reduce-shape result buffers.
  auto* new_input_subtuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  TF_RETURN_IF_ERROR(
      input_tuple->ReplaceOperandWithDifferentShape(2, new_input_subtuple));
  auto* body = loop->while_body();
  auto* body_param = body->parameter_instruction(0);
  auto* body_root = body->root_instruction();
  CHECK_EQ(body_root->opcode(), HloOpcode::kTuple);
  // Update tuple shapes.
  for (auto* tuple : std::vector<HloInstruction*>{
           input_tuple, loop, loop->while_condition()->parameter_instruction(0),
           body_param, body_root}) {
    *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(), {2}) =
        new_input_subtuple->shape();
  }
  auto* new_loop_input =
      body->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_input_subtuple->shape(), body_param, 2));

  // Represents a cluster that's associated to a single dynamic-update-slice op,
  // which should be moved to inside of the windowed dot-general loop. There
  // might be multiple clusters associated with multiple dynamic-update-slice
  // ops which all need moving.
  struct MotionCluster {
    HloInstruction* dus;
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>
        outside_to_inside;
    std::vector<HloInstruction*> slice_offsets;
  };

  std::vector<MotionCluster> motion_clusters;

  // The elementwise nodes will be created with sliced shape. The original
  // loop output corresponds to the dynamic-update-slice's update slice.
  {
    HloInstruction* dus = body_root->mutable_operand(2);
    while (dus->opcode() == HloOpcode::kDynamicUpdateSlice) {
      motion_clusters.emplace_back();
      motion_clusters.back().dus = dus;
      motion_clusters.back().outside_to_inside[original_output] =
          dus->mutable_operand(1);
      motion_clusters.back().slice_offsets.reserve(padded_shape.rank());
      for (int64_t i = 0; i < padded_shape.rank(); ++i) {
        motion_clusters.back().slice_offsets.push_back(
            motion_clusters.back().dus->mutable_operand(i + 2));
      }
      dus = dus->mutable_operand(0);
    }
  }
  // This is at least one cluster that needs moving.
  CHECK_GE(motion_clusters.size(), 1);
  MotionCluster& base_motion_cluster = motion_clusters[0];

  worklist.clear();
  auto add_users_if_available = [&](HloInstruction* inst) {
    for (auto* u : inst->users()) {
      if (base_motion_cluster.outside_to_inside.count(u) == 0 &&
          to_move.count(u) > 0 &&
          absl::c_all_of(u->operands(), [&](const HloInstruction* o) {
            return base_motion_cluster.outside_to_inside.count(o) > 0;
          })) {
        worklist.push_back(u);
      }
    }
  };

  for (int64_t i = 0; i < new_operands.size(); ++i) {
    auto* operand_gte =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_operands[i]->shape(), new_loop_input, i));
    for (MotionCluster& motion_cluster : motion_clusters) {
      motion_cluster.outside_to_inside[new_operands[i]] = operand_gte;
    }
    add_users_if_available(new_operands[i]);
  }
  add_users_if_available(original_output);

  // Now create the moved nodes inside the loop body.
  auto get_slice = [&](HloInstruction* padded,
                       absl::Span<HloInstruction* const> slice_offsets,
                       HloInstruction* dus) {
    return body->AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::ChangeElementType(dus->operand(1)->shape(),
                                     padded->shape().element_type()),
        padded, slice_offsets, dus->operand(1)->shape().dimensions()));
  };
  // Helper functions to create nodes with small operands.
  auto add_broadcast = [&](const HloInstruction* broadcast) {
    Shape padded_operand_shape = broadcast->operand(0)->shape();
    for (int64_t i = 0; i < broadcast->dimensions().size(); ++i) {
      padded_operand_shape.set_dimensions(
          i, padded_shape.dimensions(broadcast->dimensions(i)));
    }
    auto* padded_operand =
        PadToShape(base_motion_cluster.outside_to_inside[broadcast->operand(0)],
                   padded_operand_shape, nullptr, body);
    auto* inside_broadcast =
        body->AddInstruction(broadcast->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(padded_shape,
                                         padded_operand_shape.element_type()),
            {padded_operand}));
    for (MotionCluster& motion_cluster : motion_clusters) {
      motion_cluster.outside_to_inside[broadcast] = get_slice(
          inside_broadcast, motion_cluster.slice_offsets, motion_cluster.dus);
    }
  };
  auto add_iota = [&](const HloInstruction* iota) {
    auto* inside_iota = body->AddInstruction(iota->CloneWithNewOperands(
        ShapeUtil::ChangeElementType(padded_shape,
                                     iota->shape().element_type()),
        {}));
    for (MotionCluster& motion_cluster : motion_clusters) {
      motion_cluster.outside_to_inside[iota] = get_slice(
          inside_iota, motion_cluster.slice_offsets, motion_cluster.dus);
    }
  };
  auto add_constant = [&](const HloInstruction* constant) {
    auto* constant_clone = body->AddInstruction(constant->Clone());
    auto* inside_constant =
        PadToShape(constant_clone,
                   ShapeUtil::ChangeElementType(
                       padded_shape, constant->shape().element_type()),
                   nullptr, body);
    for (MotionCluster& motion_cluster : motion_clusters) {
      motion_cluster.outside_to_inside[constant] = get_slice(
          inside_constant, motion_cluster.slice_offsets, motion_cluster.dus);
    }
  };
  auto add_other_inst = [&](const HloInstruction* inst) {
    std::vector<HloInstruction*> operands_inside(inst->operand_count());
    for (MotionCluster& motion_cluster : motion_clusters) {
      for (int64_t i = 0; i < operands_inside.size(); ++i) {
        operands_inside[i] = motion_cluster.outside_to_inside[inst->operand(i)];
      }
      motion_cluster.outside_to_inside[inst] =
          body->AddInstruction(inst->CloneWithNewOperands(
              ShapeUtil::ChangeElementType(
                  motion_cluster.dus->operand(1)->shape(),
                  inst->shape().element_type()),
              operands_inside));
    }
  };

  while (!worklist.empty()) {
    auto* inst = worklist.back();
    worklist.pop_back();
    if (absl::c_all_of(
            motion_clusters, [inst](const MotionCluster& motion_cluster) {
              return motion_cluster.outside_to_inside.count(inst) > 0;
            })) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kBroadcast) {
      add_broadcast(inst);
    } else if (inst->opcode() == HloOpcode::kIota) {
      add_iota(inst);
    } else if (inst->opcode() == HloOpcode::kConstant) {
      add_constant(inst);
    } else if (inst->opcode() == HloOpcode::kReduce) {
      // This is an output, for which we has special handling later.
    } else if (inst->IsElementwise()) {
      add_other_inst(inst);
    } else {
      // Skip cloning other non-elementwise ops.
    }
    add_users_if_available(inst);
  }
  std::vector<HloInstruction*> new_outputs_inside(new_operands.size());
  for (int64_t i = 0; i < new_outputs_inside.size(); ++i) {
    new_outputs_inside[i] =
        base_motion_cluster.outside_to_inside[new_operands[i]];
  }

  // Now create the reduce outputs inside of the loop.
  for (int64_t i = 0; i < reduce_outputs.size(); ++i) {
    auto* reduce_outside = reduce_outputs[i];
    CHECK_EQ(reduce_outside->opcode(), HloOpcode::kReduce);
    int64_t index_in_operand = new_operands.size() - reduce_outputs.size() + i;
    auto* last_iter_result =
        base_motion_cluster.outside_to_inside[new_operands[index_in_operand]];

    auto create_inside_reduce =
        [&](absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
                outside_to_inside,
            absl::Span<HloInstruction* const> slice_offsets,
            HloInstruction* last_iter_result) -> StatusOr<HloInstruction*> {
      HloInstruction* operand0 = outside_to_inside[reduce_outside->operand(0)];
      HloInstruction* operand1 = outside_to_inside[reduce_outside->operand(1)];
      TF_ASSIGN_OR_RETURN(
          Shape reduce_shape,
          ShapeInference::InferReduceShape(
              {&operand0->shape(), &operand1->shape()},
              reduce_outside->dimensions(),
              reduce_outside->to_apply()->ComputeProgramShape()));
      *reduce_shape.mutable_layout() = reduce_outside->shape().layout();
      std::vector<HloInstruction*> reduce_dus_offsets;
      // If any collapsed dimension is windowed, we need to accumulate with last
      // iteration's result. If such a dimension has padding, we also need to
      // mask off invalid data.
      bool needs_accumulate = false;
      std::vector<int64_t> dims_to_mask;
      for (int64_t i = 0; i < slice_offsets.size(); ++i) {
        if (absl::c_linear_search(reduce_outside->dimensions(), i)) {
          if (reduce_outside->operand(0)->shape().dimensions(i) !=
              operand0->shape().dimensions(i)) {
            needs_accumulate = true;
            if (unpadded_shape.dimensions(i) != padded_shape.dimensions(i)) {
              dims_to_mask.push_back(i);
            }
          }
          continue;
        }
        reduce_dus_offsets.push_back(slice_offsets[i]);
      }
      // Mask off invalid data in collapsed dimensions.
      for (int64_t dim : dims_to_mask) {
        auto* iota = body->AddInstruction(HloInstruction::CreateIota(
            ShapeUtil::ChangeElementType(operand0->shape(), S32), dim));
        auto* add = body->AddInstruction(HloInstruction::CreateBinary(
            iota->shape(), HloOpcode::kAdd, iota,
            body->AddInstruction(HloInstruction::CreateBroadcast(
                iota->shape(), slice_offsets[dim], {}))));
        auto* limit = body->AddInstruction(HloInstruction::CreateBroadcast(
            iota->shape(),
            body->AddInstruction(
                HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                    reduce_outside->operand(0)->shape().dimensions(dim)))),
            {}));
        auto* compare = body->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::ChangeElementType(iota->shape(), PRED), add, limit,
            ComparisonDirection::kLt));
        operand0 = body->AddInstruction(HloInstruction::CreateTernary(
            operand0->shape(), HloOpcode::kSelect, compare, operand0,
            body->AddInstruction(HloInstruction::CreateBroadcast(
                operand0->shape(), operand1, {}))));
      }
      auto* output_inside =
          body->AddInstruction(reduce_outside->CloneWithNewOperands(
              reduce_shape, {operand0, operand1}));
      // Accumulate with previous results if needed.
      if (needs_accumulate) {
        auto* input_slice =
            body->AddInstruction(HloInstruction::CreateDynamicSlice(
                output_inside->shape(), last_iter_result, reduce_dus_offsets,
                output_inside->shape().dimensions()));
        output_inside = body->AddInstruction(HloInstruction::CreateBinary(
            output_inside->shape(),
            reduce_outside->to_apply()->root_instruction()->opcode(),
            output_inside, input_slice));
      }
      // Dynamic-update-slice if needed.
      if (!ShapeUtil::Compatible(output_inside->shape(),
                                 last_iter_result->shape())) {
        output_inside =
            body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
                last_iter_result->shape(), last_iter_result, output_inside,
                reduce_dus_offsets));
      }
      return output_inside;
    };
    for (MotionCluster& motion_cluster : motion_clusters) {
      TF_ASSIGN_OR_RETURN(
          last_iter_result,
          create_inside_reduce(motion_cluster.outside_to_inside,
                               motion_cluster.slice_offsets, last_iter_result));
    }
    new_outputs_inside[index_in_operand] = last_iter_result;
  }

  // Body output.
  auto* new_output_inside =
      body->AddInstruction(HloInstruction::CreateTuple(new_outputs_inside));
  TF_RETURN_IF_ERROR(
      body_root->ReplaceOperandWithDifferentShape(2, new_output_inside));
  TF_RETURN_IF_ERROR(
      body->RemoveInstructionAndUnusedOperands(base_motion_cluster.dus));
  // Replace uses of the reduces outside the loop.
  auto* new_output_gte =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_output_inside->shape(), loop, 2));
  for (int64_t i = 0; i < reduce_outputs.size(); ++i) {
    int64_t index_in_operand = new_operands.size() - reduce_outputs.size() + i;
    auto* new_output =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_outputs_inside[index_in_operand]->shape(), new_output_gte,
            index_in_operand));
    if (!ShapeUtil::Compatible(new_output->shape(),
                               reduce_outputs[i]->shape())) {
      new_output = computation->AddInstruction(HloInstruction::CreateSlice(
          reduce_outputs[i]->shape(), new_output,
          std::vector<int64_t>(new_output->shape().rank(), 0),
          reduce_outputs[i]->shape().dimensions(),
          std::vector<int64_t>(new_output->shape().rank(), 1)));
    }
    TF_RETURN_IF_ERROR(reduce_outputs[i]->ReplaceAllUsesWith(new_output));
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(reduce_outputs[i]));
  }
  return Status::OK();
}

}  // namespace

Status SpmdPartitioningVisitor::DoCodeMotionForWindowedDotGeneralLoops(
    HloComputation* computation, const SpmdPartitionerOptions& options) {
  for (auto& loop : windowed_dot_general_loops_) {
    if (loop.windowed_in_contracting_dims || loop.windowed_in_batch_dims ||
        loop.operands_sharded_at_contracting_dims) {
      // We have a dynamic-slice for the non-windowed operand in
      // batch/contracting-dim/noncontracting-dim windowed dot-general. So
      // moving the broadcast/iota/elementwise ops into the loop could help
      // reduce memory via fusion.
      TF_RETURN_IF_ERROR(
          SinkInputNodesIntoWindowedDotGeneralLoopOnContractingDimensions(
              loop.while_loop, 1 - loop.windowed_operand));
    }
    // Currently unrolled loop does not support this optimization.
    if (!loop.windowed_in_contracting_dims &&
        !loop.operands_sharded_at_contracting_dims) {
      // We have a dynamic-update-slice for the output in
      // batch/non-contracting-dim windowed dot-general. So moving reduce ops
      // into the loop could help reduce memory.
      TF_RETURN_IF_ERROR(
          MoveUsersIntoWindowedDotGeneralLoopOnNonContractingDimensions(
              loop.while_loop, options));
    }
  }
  return Status::OK();
}

}  // namespace spmd
}  // namespace xla
