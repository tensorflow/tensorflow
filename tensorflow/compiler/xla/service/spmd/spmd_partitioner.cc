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

#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"

#include <float.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dot_as_convolution_util.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/numbers.h"

namespace xla {
namespace spmd {

string SpmdLogger::MakeReport() {
  string report;
  absl::StrAppend(&report,
                  "\n\n***** SPMD memory during transformation *****\n");

  std::sort(entries_.begin(), entries_.end(),
            [](auto const& entry0, auto const& entry1) {
              return entry0.first > entry1.first;
            });
  for (int64 i = 0;
       i < std::min<int64>(report_instruction_count_, entries_.size()); ++i) {
    absl::StrAppend(
        &report, "\n  ",
        tensorflow::strings::HumanReadableNumBytes(entries_[i].first), " : ",
        entries_[i].second, "\n");
  }

  return report;
}

void SpmdLogger::RegisterLogEntry(HloInstruction* hlo,
                                  const std::vector<HloInstruction*>& group) {
  string report = hlo->ToString();
  int64 max_value = -1;
  for (HloInstruction* inst : group) {
    if (!inst->shape().IsArray()) {
      continue;
    }
    max_value = std::max<int64>(max_value, ShapeSizeInBytes(inst->shape()));
    absl::StrAppend(&report, "     * ", inst->ToString(), "\n");
  }
  entries_.push_back(std::make_pair(max_value, report));
}

/* static */ string SpmdLogger::ReportBeforePartition(
    const HloModule& module, int64 report_instruction_count) {
  string report;
  absl::StrAppend(&report,
                  "\n\n***** SPMD memory usage before partition *****\n");
  absl::StrAppend(&report, "\n  ** Replicated instructions\n");
  absl::StrAppend(&report, ReportMemoryUsage(
                               module,
                               [](const HloInstruction* hlo) {
                                 return !hlo->has_sharding() ||
                                        hlo->sharding().IsReplicated();
                               },
                               report_instruction_count));
  absl::StrAppend(&report, "\n  ** All instructions\n");
  absl::StrAppend(&report,
                  ReportMemoryUsage(
                      module, [](const HloInstruction* hlo) { return true; },
                      report_instruction_count));
  return report;
}

/* static */ string SpmdLogger::ReportAfterPartition(
    const HloModule& module, int64 report_instruction_count) {
  string report;
  absl::StrAppend(&report,
                  "\n\n***** SPMD memory usage after partition *****\n");
  absl::StrAppend(&report,
                  ReportMemoryUsage(
                      module, [](const HloInstruction* hlo) { return true; },
                      report_instruction_count));
  return report;
}

template <typename F>
/* static */ string SpmdLogger::ReportMemoryUsage(
    const HloModule& module, const F& filter, int64 report_instruction_count) {
  string report;
  std::vector<HloInstruction*> instructions;
  instructions.reserve(module.instruction_count());

  for (auto computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (auto hlo : computation->instructions()) {
      if (hlo->shape().IsTuple() ||
          ShapeUtil::IsEffectiveScalar(hlo->shape())) {
        continue;
      }
      if (filter(hlo)) {
        instructions.push_back(hlo);
      }
    }
  }

  const auto add_report = [&](std::vector<HloInstruction*>* insts) {
    std::sort(insts->begin(), insts->end(),
              [](const HloInstruction* inst0, const HloInstruction* inst1) {
                return ShapeSizeInBytes(inst0->shape()) >
                       ShapeSizeInBytes(inst1->shape());
              });
    for (int64 i = 0;
         i < std::min<int64>(report_instruction_count, insts->size()); ++i) {
      absl::StrAppend(&report, "  ",
                      tensorflow::strings::HumanReadableNumBytes(
                          ShapeSizeInBytes((*insts)[i]->shape())),
                      " : ", (*insts)[i]->ToString(), "\n");
    }
  };

  add_report(&instructions);
  return report;
}

namespace {

// Returns the replica group configuration where each replica belongs to its own
// group.
std::vector<ReplicaGroup> CreateReplicaGroups(int64 num_replicas) {
  std::vector<ReplicaGroup> groups(num_replicas);
  for (int64 i = 0; i < num_replicas; ++i) {
    groups[i].add_replica_ids(i);
  }
  return groups;
}

bool CanReshardWithAllToAll(const HloSharding& source,
                            const HloSharding& target) {
  return UniqueTiledDim(source) && UniqueTiledDim(target) &&
         UniqueTiledDim(source) != UniqueTiledDim(target);
}

bool CanReshardWithCollectivePermute(const HloSharding& source,
                                     const HloSharding& target) {
  return UniqueTiledDim(source) && UniqueTiledDim(target) &&
         UniqueTiledDim(source) == UniqueTiledDim(target) && source != target;
}

// Clears all sharding attributes from instructions in the module. This must be
// called only after all SPMD transformation is complete.
Status ClearShardingAttributes(HloModule* module) {
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      // Keep sharding annotation on Infeed and entry parameters since they're
      // used by HloReplicationAnalysis later (for ArCrsCombiner).
      if (hlo->opcode() == HloOpcode::kInfeed) {
        continue;
      }
      if (hlo->opcode() == HloOpcode::kParameter &&
          computation == module->entry_computation()) {
        continue;
      }
      hlo->clear_sharding();
    }
  }
  return Status::OK();
}

}  // namespace

HloInstruction* SpmdBuilder::AddInstruction(
    std::unique_ptr<HloInstruction> instruction) {
  HloInstruction* hlo =
      HloComputation::Builder::AddInstruction(std::move(instruction));
  if (visiting_hlo_) {
    instructions_[visiting_hlo_].push_back(hlo);
  }
  return hlo;
}

PartitionedHlo PartitionedHlo::Reshard(const HloSharding& target) {
  auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
  for (auto& entry : cache) {
    if (entry.first == target) {
      return entry.second;
    }
  }
  cache.emplace_back(target, ReshardNoCache(target));
  state_.reshard_cache->per_hlo_cache[cache.back().second.hlo()]
      .reshard_cache.emplace_back(sharding(), *this);
  return cache.back().second;
}

PartitionedHlo PartitionedHlo::ReshardNoCache(const HloSharding& target) {
  VLOG(2) << "Resharding " << hlo_->ToString() << " from "
          << hlo_->sharding().ToString() << " to " << target.ToString();
  const Shape& shape = hlo_->shape();
  CHECK(shape.IsTuple() || !target.IsTuple());

  // Tuple shape instructions may have non-tuple sharding, which means that the
  // same sharding applies to all the leaves.
  if (shape.IsTuple() && !target.IsTuple()) {
    return Reshard(target.GetTupleSharding(shape).ValueOrDie());
  }

  // For a tuple shape, recursively apply Reshard to all the leaves and return
  // a tuple instruction.
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      auto subshape = ShapeUtil::GetTupleElementShape(shape, i);
      auto element = state_.b->AddInstruction(
          HloInstruction::CreateGetTupleElement(subshape, hlo(), i));
      element->set_sharding(sharding().GetSubSharding(shape, {i}));
      elements.push_back(
          PartitionedHlo(
              element, ShapeUtil::GetTupleElementShape(base_shape_, i), state_)
              .Reshard(target.GetSubSharding(shape, {i}))
              .hlo());
    }
    auto tuple =
        state_.b->AddInstruction(HloInstruction::CreateTuple(elements));
    tuple->set_sharding(target);
    return PartitionedHlo(tuple, base_shape_, state_);
  }

  if (sharding() == target) {
    return *this;
  }

  if (shape.element_type() == TOKEN) {
    return *this;
  }

  if (CanReshardWithCollectivePermute(sharding(), target)) {
    return ReshardWithCollectivePermute(target);
  }

  if (CanReshardWithAllToAll(sharding(), target)) {
    return ReshardWithAllToAll(target);
  }

  // If not replicated yet, first replicate and then reshard to use one of the
  // two implementations below.
  if (!sharding().IsReplicated()) {
    return Replicate().Reshard(target);
  }

  // 'Replicated' to 'SingleDevice'.
  if (target.IsTileMaximal()) {
    auto copy = state_.b->AddInstruction(
        HloInstruction::CreateUnary(hlo_->shape(), HloOpcode::kCopy, hlo_));
    copy->set_sharding(target);
    return PartitionedHlo(copy, base_shape_, state_);
  }

  // 'Replicated' to 'Tiled'.
  auto padded_hlo =
      PadBaseShapeBeforeUnevenTiledSharding(hlo_, target, state_.b);
  auto shard_shape = MakePartitionedShape(shape, target);
  auto slice = state_.b->AddInstruction(HloInstruction::CreateDynamicSlice(
      shard_shape, padded_hlo,
      MakePartitionOffsets(shape, target, state_.partition_id, state_.b),
      shard_shape.dimensions()));
  slice->set_sharding(target);
  return PartitionedHlo(slice, base_shape_, state_);
}

PartitionedHlo PartitionedHlo::PadWithValue(HloInstruction* pad_value) const {
  const HloSharding& sharding = hlo_->sharding();
  const Shape& shape = hlo_->shape();
  CHECK(!shape.IsTuple() && shape.element_type() != TOKEN);
  if (sharding.IsReplicated() || EvenlyPartitions(base_shape_, sharding)) {
    return *this;
  }
  CHECK(!sharding.IsTileMaximal());
  auto index_shape = ShapeUtil::ChangeElementType(shape, S32);
  auto mask_shape = ShapeUtil::ChangeElementType(index_shape, PRED);
  auto get_mask_for_dim = [&](int64 dim, HloInstruction* start_index) {
    // Comparison: iota + start_index < valid_size
    auto iota =
        state_.b->AddInstruction(HloInstruction::CreateIota(index_shape, dim));
    auto broadcast_start_index = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(index_shape, start_index, {}));
    auto index_in_full_shape =
        state_.b->AddInstruction(HloInstruction::CreateBinary(
            index_shape, HloOpcode::kAdd, iota, broadcast_start_index));
    auto valid_size = state_.b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32>(base_shape_.dimensions(dim))));
    auto broadcast_valid_size = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(index_shape, valid_size, {}));
    return state_.b->AddInstruction(HloInstruction::CreateCompare(
        mask_shape, index_in_full_shape, broadcast_valid_size,
        ComparisonDirection::kLt));
  };

  HloInstruction* mask = nullptr;
  auto offsets = MakePartitionOffsets(base_shape_, sharding,
                                      state_.partition_id, state_.b);
  for (int64 i = 0; i < shape.rank(); ++i) {
    if (base_shape_.dimensions(i) % sharding.tile_assignment().dim(i) == 0) {
      continue;
    }
    if (mask == nullptr) {
      mask = get_mask_for_dim(i, offsets[i]);
    } else {
      mask = state_.b->AddInstruction(
          HloInstruction::CreateBinary(mask->shape(), HloOpcode::kAnd, mask,
                                       get_mask_for_dim(i, offsets[i])));
    }
  }

  if (mask == nullptr) {
    return *this;
  }

  auto broadcast_pad_value = state_.b->AddInstruction(
      HloInstruction::CreateBroadcast(shape, pad_value, {}));
  auto result = state_.b->AddInstruction(HloInstruction::CreateTernary(
      shape, HloOpcode::kSelect, mask, hlo_, broadcast_pad_value));
  result->set_sharding(sharding);
  return PartitionedHlo(result, base_shape_, state_);
}

absl::optional<PartitionedHlo::WindowedInputShardReturnValue>
PartitionedHlo::ReshardAsWindowedInput(const Window& window,
                                       const HloSharding& target,
                                       HloInstruction* pad_value,
                                       bool mask_invalid_region) {
  auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].window_reshard_cache;
  for (auto& entry : cache) {
    if (std::get<0>(entry) == target &&
        protobuf_util::ProtobufEquals(std::get<1>(entry), window)) {
      return std::get<2>(entry);
    }
  }
  auto update_cache = [&](WindowedInputShardReturnValue result) {
    cache.emplace_back(target, window, std::move(result));
    return std::get<2>(cache.back());
  };
  VLOG(2) << "ReshardAsWindowedInput()\n"
          << "\twindow:" << window_util::ToString(window)
          << "\ttarget sharding:" << target.ToString();

  CHECK(!target.IsTileMaximal());
  auto partition_ordinals =
      MakeTiledPartitionOrdinals(target, state_.partition_id, state_.b);
  auto shard_shape = base_shape_;

  std::vector<MultiplyAddDivideOffsetCalculation> start_on_padded_calculations(
      base_shape_.rank());
  std::vector<MultiplyAddDivideOffsetCalculation> limit_on_padded_calculations(
      base_shape_.rank());
  std::vector<HloInstruction*> dynamic_slice_offset_on_output(
      base_shape_.rank(), nullptr);

  Window shard_window = window;
  auto padded_shape = base_shape_;
  std::vector<HloInstruction*> offsets_on_padded_shape(base_shape_.rank());
  std::vector<int64> per_shard_window_counts(base_shape_.rank());
  std::vector<int64> explicit_left_padding(base_shape_.rank());
  for (int64 i = 0; i < base_shape_.rank(); ++i) {
    // Do not pad non-partitioned dimensions.
    int64 shard_count = target.tile_assignment().dim(i);
    if (shard_count == 1) {
      offsets_on_padded_shape[i] = state_.b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
      continue;
    }
    const auto& wd = window.dimensions(i);
    if (wd.window_dilation() != 1) {
      // TODO(yuanzx): Support window dilation.
      VLOG(2) << "Failed to reshard window operand due to window dilation";
      return absl::nullopt;
    }
    int64 full_size =
        base_shape_.dimensions(i) +
        (wd.base_dilation() - 1) * (base_shape_.dimensions(i) - 1) +
        wd.padding_high() + wd.padding_low();
    if (full_size < wd.size()) {
      VLOG(2) << "Failed to reshard window operand because the window size is "
                 "larger than padded base size";
      return absl::nullopt;
    }
    int64 window_count = (full_size - wd.size()) / wd.stride() + 1;
    per_shard_window_counts[i] = CeilOfRatio(window_count, shard_count);
    if (wd.stride() != 1 &&
        (wd.stride() * per_shard_window_counts[i]) % wd.base_dilation() != 0) {
      // TODO(yuanzx): Support this case.
      VLOG(2) << "Failed to reshard window operand due to non-trivial dilation";
      return absl::nullopt;
    }

    // We use explicit padding for full dilations, then use padding_low and
    // padding_high on the sharded op for the remaining. padding_low and
    // padding_high are now given initial values, which will be later updated if
    // dilation is not 1.
    auto swd = shard_window.mutable_dimensions(i);
    explicit_left_padding[i] = wd.padding_low() / wd.base_dilation();
    swd->set_padding_low(wd.padding_low() % wd.base_dilation());
    swd->set_padding_high(0);

    // Calculation for the first element needed on the 'padded-but-not-dilated'
    // shape. The start on the dilated shape could be a hole, so we add
    // wd.base_dilation() - 1 to the constant term to skip the leading holes.
    start_on_padded_calculations[i] = MultiplyAddDivideOffsetCalculation(
        wd.stride() * per_shard_window_counts[i],
        wd.base_dilation() - 1 - swd->padding_low(), wd.base_dilation());
    int64 dilated_shard_size =
        wd.stride() * (per_shard_window_counts[i] - 1) + wd.size();
    limit_on_padded_calculations[i] = MultiplyAddDivideOffsetCalculation(
        wd.stride() * per_shard_window_counts[i],
        dilated_shard_size + wd.base_dilation() - 1 - swd->padding_low(),
        wd.base_dilation());

    offsets_on_padded_shape[i] = start_on_padded_calculations[i].Calculate(
        partition_ordinals[i], state_.b);

    auto shard_size_function =
        limit_on_padded_calculations[i] - start_on_padded_calculations[i];
    int64 max_shard_size = shard_size_function.MaxInRange(0, shard_count);
    shard_shape.set_dimensions(i, max_shard_size);
    padded_shape.set_dimensions(
        i, limit_on_padded_calculations[i].Calculate(shard_count - 1));

    // For base dilation, calculate the needed padding_low and padding_high, as
    // well as the offset for the output if a dynamic slice is needed after the
    // sharded op.
    if (wd.base_dilation() != 1) {
      // Returns the offset of a shard's first valid element in the dilated
      // shard.
      auto get_first_valid_element_offset_on_dilated_shard =
          [&](int64 shard_ordinal) {
            return start_on_padded_calculations[i].Calculate(shard_ordinal) *
                       wd.base_dilation() +
                   swd->padding_low() -
                   wd.stride() * per_shard_window_counts[i] * shard_ordinal;
          };
      CHECK_EQ(get_first_valid_element_offset_on_dilated_shard(0),
               swd->padding_low());

      // Determine swd->padding_high.
      for (int64 shard_ordinal = 0; shard_ordinal < shard_count;
           ++shard_ordinal) {
        int64 wanted_limit_on_dilated_shard =
            wd.stride() * (per_shard_window_counts[i] - 1) + wd.size();
        int64 actual_limit_on_dilated_shard_without_pad_high =
            get_first_valid_element_offset_on_dilated_shard(shard_ordinal) +
            (max_shard_size - 1) * wd.base_dilation() + 1;
        swd->set_padding_high(std::max<int64>(
            swd->padding_high(),
            wanted_limit_on_dilated_shard -
                actual_limit_on_dilated_shard_without_pad_high));
      }

      // Determine swd->padding_low and output dynamic slice index.
      if (wd.stride() == 1) {
        int64 max_pad_low = get_first_valid_element_offset_on_dilated_shard(0);
        bool all_same = true;
        for (int64 shard_ordinal = 1; shard_ordinal < shard_count;
             ++shard_ordinal) {
          int64 start =
              get_first_valid_element_offset_on_dilated_shard(shard_ordinal);
          if (start != swd->padding_low()) {
            all_same = false;
          }
          max_pad_low = std::max(max_pad_low, start);
        }
        if (!all_same) {
          auto start_on_padded_input =
              start_on_padded_calculations[i].Calculate(partition_ordinals[i],
                                                        state_.b);
          // We will calculate
          //   max_pad_low - (first_window - required_first_window)
          // which equals
          //   required_first_window - (first_window - max_pad_low)
          auto first_window_minus_max_pad_low =
              MultiplyAddDivideOffsetCalculation(
                  wd.base_dilation(), swd->padding_low() - max_pad_low, 1)
                  .Calculate(start_on_padded_input, state_.b);
          auto required_first_window =
              MultiplyAddDivideOffsetCalculation(per_shard_window_counts[i], 0,
                                                 1)
                  .Calculate(partition_ordinals[i], state_.b);
          dynamic_slice_offset_on_output[i] =
              state_.b->AddInstruction(HloInstruction::CreateBinary(
                  required_first_window->shape(), HloOpcode::kSubtract,
                  required_first_window, first_window_minus_max_pad_low));
        }
        swd->set_padding_low(max_pad_low);
      } else {
        CHECK_EQ(
            (wd.stride() * per_shard_window_counts[i]) % wd.base_dilation(), 0)
            << "General base dilation not yet implemented.";
        // padding_low on all shards should equal the initially assigned
        // swd->padding_low(), i.e., the padding_low() on the original window.
      }
    }
  }

  // Returns the output dynamic slice offset when needed, and absl::nullopt
  // otherwise.
  auto get_dynamic_slice_offset_on_output_if_needed =
      [&]() -> absl::optional<std::vector<HloInstruction*>> {
    if (absl::c_all_of(
            dynamic_slice_offset_on_output,
            [](HloInstruction* offset) { return offset == nullptr; })) {
      return absl::nullopt;
    }
    auto zero = state_.b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
    for (int64 i = 0; i < dynamic_slice_offset_on_output.size(); ++i) {
      if (dynamic_slice_offset_on_output[i] == nullptr) {
        dynamic_slice_offset_on_output[i] = zero;
      }
    }
    return dynamic_slice_offset_on_output;
  };

  // If the currrent HLO is replicated, pad then slice.
  if (sharding().IsReplicated()) {
    PaddingConfig padding_config;
    for (int64 i = 0; i < base_shape_.rank(); ++i) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_interior_padding(0);
      // Do not pad non-partitioned dimensions.
      if (target.tile_assignment().dim(i) == 1) {
        padding_config_dim->set_edge_padding_low(0);
        padding_config_dim->set_edge_padding_high(0);
        continue;
      }
      padding_config_dim->set_edge_padding_low(explicit_left_padding[i]);
      padding_config_dim->set_edge_padding_high(padded_shape.dimensions(i) -
                                                explicit_left_padding[i] -
                                                base_shape_.dimensions(i));
    }
    auto padded_hlo = ShapeUtil::Compatible(padded_shape, base_shape_)
                          ? hlo_
                          : state_.b->AddInstruction(HloInstruction::CreatePad(
                                padded_shape, hlo_, pad_value, padding_config));
    auto sharded_input =
        state_.b->AddInstruction(HloInstruction::CreateDynamicSlice(
            shard_shape, padded_hlo, offsets_on_padded_shape,
            shard_shape.dimensions()));
    return update_cache(WindowedInputShardReturnValue{
        sharded_input, shard_window,
        get_dynamic_slice_offset_on_output_if_needed()});
  }

  if (target != sharding()) {
    return Replicate().ReshardAsWindowedInput(window, target, pad_value);
  }

  // Halo exchange.
  HloInstruction* visiting_hlo = hlo_;
  auto original_shard_shape = MakePartitionedShape(base_shape_, target);

  std::vector<OffsetCalculation> left_halo_size_functions(base_shape_.rank());
  std::vector<OffsetCalculation> right_halo_size_functions(base_shape_.rank());
  // TODO(yuanzx): We are concatenating on each sharded dimension one at time,
  // and in the second dimension (and beyond) we create halos by slicing the
  // concat in the previous dimension, which is not optimal. We should generate
  // halos only concating slices, instead of slicing concats.
  for (int dim = 0; dim < base_shape_.rank(); ++dim) {
    int64 shard_count = target.tile_assignment().dim(dim);
    if (shard_count == 1) {
      continue;
    }
    int64 input_shard_size =
        CeilOfRatio(base_shape_.dimensions(dim), shard_count);

    // Left halo. The size of the halo is derived by subtracting the first read
    // element offset of the i'th partition from the limit of the (i-1)'th
    // partition.
    MultiplyAddDivideOffsetCalculation shard_limit_of_previous_on_padded(
        input_shard_size, explicit_left_padding[dim], 1);
    left_halo_size_functions[dim] =
        shard_limit_of_previous_on_padded - start_on_padded_calculations[dim];

    // Right halo.
    MultiplyAddDivideOffsetCalculation shard_start_of_next_on_padded(
        input_shard_size, input_shard_size + explicit_left_padding[dim], 1);
    right_halo_size_functions[dim] =
        limit_on_padded_calculations[dim] - shard_start_of_next_on_padded;

    auto resharded = ExchangeHaloAndGetValidData(
        visiting_hlo, base_shape_, left_halo_size_functions[dim],
        right_halo_size_functions[dim], explicit_left_padding[dim],
        padded_shape.dimensions(dim), shard_shape.dimensions(dim), dim, target,
        offsets_on_padded_shape[dim], pad_value, partition_ordinals[dim],
        state_.collective_ops_creator, state_.next_channel_id, state_.b,
        mask_invalid_region);
    if (!resharded) {
      VLOG(1) << "ReshardAsWindowedInput failed without replicate first: halo "
                 "is beyond the neighbor.";
      return Replicate().ReshardAsWindowedInput(window, target, pad_value);
    }
    visiting_hlo = *resharded;
  }
  return update_cache(WindowedInputShardReturnValue{
      visiting_hlo, shard_window,
      get_dynamic_slice_offset_on_output_if_needed()});
}

PartitionedHlo PartitionedHlo::Replicate() {
  const HloSharding& sharding = hlo_->sharding();
  const Shape& shape = hlo_->shape();
  CHECK(!shape.IsTuple() && shape.element_type() != TOKEN);

  if (sharding.IsReplicated()) {
    return *this;
  }
  auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
  for (auto& entry : cache) {
    if (entry.first.IsReplicated()) {
      return entry.second;
    }
  }
  auto update_cache = [&](PartitionedHlo resharded) {
    state_.reshard_cache->per_hlo_cache[resharded.hlo()]
        .reshard_cache.emplace_back(sharding, *this);
    cache.emplace_back(HloSharding::Replicate(), std::move(resharded));
    return cache.back().second;
  };
  // 'Single Device' to 'Repliated'.
  if (sharding.IsTileMaximal()) {
    return update_cache(Broadcast());
  }

  // 'Tiled' to 'Replicated'.
  HloInstruction* result = nullptr;
  if (state_.collective_ops_creator.create_cross_partition_all_gather) {
    result = state_.partitioner->AllGatherShards(state_.b, hlo_, sharding,
                                                 NewChannel());
  }
  Shape padded_base_shape = shape;
  for (int64 i = 0; i < padded_base_shape.rank(); ++i) {
    padded_base_shape.set_dimensions(
        i, shape.dimensions(i) * sharding.tile_assignment().dim(i));
  }
  if (result == nullptr) {
    auto zero = state_.b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(shape.element_type())));
    auto zero_bcast = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(padded_base_shape, zero, {}));
    auto dus =
        state_.b->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            padded_base_shape, zero_bcast, hlo_,
            MakePartitionOffsets(padded_base_shape, sharding,
                                 state_.partition_id, state_.b)));
    HloComputation* reduction =
        MakeBinaryAdd(shape.element_type(), state_.module);

    auto all_reduce =
        state_.collective_ops_creator.create_cross_partition_all_reduce(
            state_.b, dus, reduction, NewChannel());
    result = all_reduce;
  }
  if (!ShapeUtil::Compatible(base_shape_, padded_base_shape)) {
    std::vector<int64> start_indices(shape.rank(), 0);
    std::vector<int64> strides(shape.rank(), 1);
    result = state_.b->AddInstruction(HloInstruction::CreateSlice(
        base_shape_, result, start_indices, base_shape_.dimensions(), strides));
  }
  result->set_sharding(HloSharding::Replicate());
  return update_cache(PartitionedHlo(result, base_shape_, state_));
}

PartitionedHlo PartitionedHlo::Broadcast() const {
  const Shape& shape = hlo_->shape();
  const HloSharding& sharding = hlo_->sharding();
  CHECK(sharding.HasUniqueDevice());
  CHECK(!shape.IsTuple() && shape.element_type() != TOKEN);

  auto src_core_id = state_.b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0<uint32>(sharding.GetUniqueDevice())));
  Shape bcast_shape = ShapeUtil::ChangeElementType(shape, PRED);
  auto is_src_core = state_.b->AddInstruction(HloInstruction::CreateBroadcast(
      bcast_shape,
      state_.b->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), state_.partition_id, src_core_id,
          ComparisonDirection::kEq)),
      {}));

  auto zero = state_.b->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(shape.element_type())));
  auto zero_bcast = state_.b->AddInstruction(
      HloInstruction::CreateBroadcast(shape, zero, {}));
  auto operand = state_.b->AddInstruction(HloInstruction::CreateTernary(
      shape, HloOpcode::kSelect, is_src_core, hlo(), zero_bcast));
  HloComputation* reduction =
      MakeBinaryAdd(shape.element_type(), state_.module);

  auto result = state_.collective_ops_creator.create_cross_partition_all_reduce(
      state_.b, operand, reduction, NewChannel());
  result->set_sharding(HloSharding::Replicate());
  return PartitionedHlo(result, base_shape_, state_);
}

PartitionedHlo PartitionedHlo::ReshardWithAllToAll(
    const HloSharding& target) const {
  int64 partition_count = sharding().tile_assignment().num_elements();
  absl::optional<int64> input_partition_dim = UniqueTiledDim(sharding());
  absl::optional<int64> output_partition_dim = UniqueTiledDim(target);
  CHECK(input_partition_dim.has_value());
  CHECK(output_partition_dim.has_value());

  // If the device order is different in the target, fix the order with
  // ReshardWithCollectivePermute.
  auto input_tile_fixed_device_order = target.tile_assignment();
  input_tile_fixed_device_order.Reshape(
      sharding().tile_assignment().dimensions());
  auto input_sharding_fixed_device_order =
      HloSharding::Tile(input_tile_fixed_device_order);
  if (input_sharding_fixed_device_order != sharding()) {
    auto fixed_order =
        ReshardWithCollectivePermute(input_sharding_fixed_device_order);
    return fixed_order.ReshardWithAllToAll(target);
  }

  auto padded_hlo =
      PadBaseShapeBeforeUnevenTiledSharding(hlo_, target, state_.b);

  // The order of ids in the group must follow the target sharding.
  std::vector<ReplicaGroup> groups(1);
  for (int64 device : target.tile_assignment()) {
    groups[0].add_replica_ids(device);
  }

  HloInstruction* result = nullptr;

  // Split along the split dimension (output_partition_dim) of the all-to-all
  // output.
  std::vector<int64> dimensions;
  for (int64 i = 0; i < base_shape_.rank(); ++i) {
    if (i == *output_partition_dim) {
      dimensions.push_back(partition_count);
      dimensions.push_back(padded_hlo->shape().dimensions(i) / partition_count);
    } else {
      dimensions.push_back(padded_hlo->shape().dimensions(i));
    }
  }
  auto reshape = state_.b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(base_shape_.element_type(), dimensions),
      padded_hlo));
  // After the reshape, it is guaranteed to have at least 3 dimensions.
  auto all_to_all =
      state_.collective_ops_creator.create_cross_partition_all_to_all(
          state_.b, {reshape}, groups, (*state_.next_channel_id)++,
          output_partition_dim);

  // Reorder the split dimension of the reshape to be located in front of the
  // input partition dimension, so the two dimensions can be combined.
  int64 new_input_partition_dim = (*output_partition_dim < *input_partition_dim)
                                      ? *input_partition_dim + 1
                                      : *input_partition_dim;
  std::vector<int64> permutation;
  for (int64 i = 0; i < all_to_all->shape().rank(); ++i) {
    if (i == *output_partition_dim) {
      continue;
    }
    if (i == new_input_partition_dim) {
      permutation.push_back(*output_partition_dim);
    }
    permutation.push_back(i);
  }
  auto transpose = state_.b->AddInstruction(HloInstruction::CreateTranspose(
      ShapeInference::InferTransposeShape(all_to_all->shape(), permutation)
          .ValueOrDie(),
      all_to_all, permutation));

  // Combine the split dimension and the input partition dimension.
  auto new_shape = ShapeInference::InferAllToAllShape(
                       padded_hlo->shape(), *output_partition_dim,
                       *input_partition_dim, partition_count)
                       .ValueOrDie();
  result = state_.b->AddInstruction(
      HloInstruction::CreateReshape(new_shape, transpose));

  const Shape result_shape = MakePartitionedShape(base_shape_, target);
  if (result_shape != result->shape()) {
    result = state_.b->AddInstruction(HloInstruction::CreateSlice(
        result_shape, result, std::vector<int64>(result_shape.rank(), 0),
        result_shape.dimensions(), std::vector<int64>(result_shape.rank(), 1)));
  }
  result->set_sharding(target);
  return PartitionedHlo(result, base_shape_, state_);
}

PartitionedHlo PartitionedHlo::ReshardWithCollectivePermute(
    const HloSharding& target) const {
  CHECK(CanReshardWithCollectivePermute(sharding(), target));
  std::vector<std::pair<int64, int64>> src_dst_pairs;
  sharding().tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64 src_device) {
        int64 dst_device = target.tile_assignment()(indices);
        if (dst_device != src_device) {
          src_dst_pairs.emplace_back(src_device, dst_device);
        }
      });
  auto cp =
      state_.collective_ops_creator.create_cross_partition_collective_permute(
          state_.b, hlo(), src_dst_pairs, (*state_.next_channel_id)++);
  cp->set_sharding(target);
  return PartitionedHlo(cp, base_shape_, state_);
}

SpmdPartitioningVisitor::SpmdPartitioningVisitor(
    HloComputation* computation, int64 num_partitions, int64 num_replicas,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdLogger* logger, SpmdPartitionerOptions options,
    SpmdPartitioner* partitioner)
    : changed_(false),
      module_(computation->parent()),
      num_partitions_(num_partitions),
      num_replicas_(num_replicas),
      collective_ops_creator_(collective_ops_creator),
      next_channel_id_(next_channel_id),
      b_(SpmdBuilder(computation->name() + "_spmd", /*hlo=*/nullptr)),
      partition_id_(collective_ops_creator_.create_partition_id(&b_)),
      logger_(logger),
      options_(std::move(options)),
      partitioner_(partitioner) {}

Status SpmdPartitioningVisitor::DefaultAction(HloInstruction* hlo) {
  if (hlo->HasSideEffect()) {
    return Unimplemented("Side-effect ops cannot be replicated: %s",
                         hlo->ToString());
  }

  if (hlo->IsElementwise() && hlo->operand_count() > 0) {
    return HandleElementwise(hlo);
  }

  if (!hlo->sharding().IsTileMaximal()) {
    VLOG(1) << "Not partitioned in SPMD mode (DefaultAction):"
            << hlo->ToString();
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      VLOG(1) << "  operand " << i
              << " sharding:" << hlo->operand(i)->sharding().ToString();
    }
  }

  // If the instruction cannot be partitioned, replicate the instruction unless
  // the instruction has side-effect.
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : hlo->operands()) {
    new_operands.push_back(
        GetPartitionedHlo(operand).Reshard(HloSharding::Replicate()).hlo());
  }
  auto clone =
      b_.AddInstruction(hlo->CloneWithNewOperands(hlo->shape(), new_operands));
  clone->set_sharding(HloSharding::Replicate());
  clone->set_metadata(hlo->metadata());
  SetPartitionedHlo(hlo,
                    PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
                        .Reshard(hlo->sharding()));
  return Status::OK();
}

Status SpmdPartitioningVisitor::Preprocess(HloInstruction* hlo) {
  visiting_hlo_ = hlo;
  b_.set_visiting_hlo(hlo);
  return Status::OK();
}

Status SpmdPartitioningVisitor::Postprocess(HloInstruction* hlo) {
  logger_->RegisterLogEntry(GetPartitionedHlo(hlo).hlo(),
                            b_.derived_instructions(hlo));
  visiting_hlo_ = nullptr;
  b_.set_visiting_hlo(nullptr);
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleElementwise(HloInstruction* hlo) {
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : hlo->operands()) {
    new_operands.push_back(
        GetPartitionedHlo(operand).Reshard(hlo->sharding()).hlo());
  }
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(hlo->CloneWithNewOperands(
        MakePartitionedShape(hlo->shape(), hlo->sharding()), new_operands));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleConcatenate(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  const Shape shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
  const int64 dimension = hlo->concatenate_dimension();
  if (sharding.tile_assignment().dim(dimension) == 1) {
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : hlo->operands()) {
      new_operands.push_back(
          GetPartitionedHlo(operand).Reshard(sharding).hlo());
    }
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(
          hlo->CloneWithNewOperands(shard_shape, new_operands));
    });
    return Status::OK();
  }

  // If the concatenate dimension is along one of the partitioned dimensions,
  // allocate the full output shape, each partition updates its owned region,
  // all-reduce across partitions, and then slice its output region.

  // We currently don't support subgroup all-reduce along partitions, so more
  // than 1 partitioned dimensions is not supported.
  if (sharding.tile_assignment().dim(dimension) != num_partitions_) {
    return DefaultAction(hlo);
  }

  // temp_output_shape is the output shape where the concatenate dimension
  // is changed to the full (and padded to shard count) dimension size.
  auto temp_output_shape = MakePartitionedShape(hlo->shape(), sharding);
  temp_output_shape.set_dimensions(
      dimension, temp_output_shape.dimensions(dimension) *
                     sharding.tile_assignment().dim(dimension));
  auto temp_output = CreateZero(temp_output_shape, &b_);

  // Offset of each operand along the concatenate dimension.
  int64 offset = 0;
  for (HloInstruction* operand : hlo->operands()) {
    auto spmd_operand = GetPartitionedHlo(operand).Reshard(sharding).hlo();
    std::vector<HloInstruction*> start_indices(
        hlo->shape().rank(), b_.AddInstruction(HloInstruction::CreateConstant(
                                 LiteralUtil::Zero(S32))));
    start_indices[dimension] =
        MultiplyAddDivideOffsetCalculation(
            spmd_operand->shape().dimensions(dimension), offset, 1)
            .Calculate(MakeTiledPartitionOrdinals(sharding, partition_id_,
                                                  &b_)[dimension],
                       &b_);
    temp_output = b_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        temp_output_shape, temp_output, spmd_operand, start_indices));
    offset += operand->shape().dimensions(dimension);
  }
  auto all_reduce = collective_ops_creator_.create_cross_partition_all_reduce(
      &b_, temp_output, MakeBinaryAdd(hlo->shape().element_type(), module_),
      NewChannel());
  SetPartitionedHlo(hlo, [&] {
    auto start_indices =
        MakeTiledPartitionOrdinals(hlo->sharding(), partition_id_, &b_);
    start_indices[dimension] = MultiplyAddDivideOffsetCalculation(
                                   shard_shape.dimensions(dimension), 0, 1)
                                   .Calculate(start_indices[dimension], &b_);
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, all_reduce, start_indices, shard_shape.dimensions()));
  });

  return Status::OK();
}

// If partitioning in the operand only happens in dimensions in passthrough
// dimensions (offset dimensions in the gather output (or scatter update) that
// have the same size as the operand), returns the corresponding output (or
// update) sharding by passing through the input sharding.
absl::optional<HloSharding> PassthroughOperandToGatherOutputOrScatterUpdate(
    const PartitionedHlo& operand, const Shape& update_or_gather_shape,
    absl::Span<const int64> collapsed_or_inserted_dims,
    absl::Span<const int64> index_map,
    absl::Span<const int64> offset_or_window_dims,
    absl::Span<const int64> slice_size) {
  if (operand.sharding().IsTileMaximal()) {
    return operand.sharding();
  }
  std::vector<int64> passthrough_tile(update_or_gather_shape.rank(), 1);
  int64 collapsed = 0;
  for (int64 i = 0; i < operand.base_shape().rank(); ++i) {
    int64 dim_partitions = operand.sharding().tile_assignment().dim(i);
    if (absl::c_linear_search(collapsed_or_inserted_dims, i) ||
        absl::c_linear_search(index_map, i)) {
      if (dim_partitions > 1) {
        return absl::nullopt;
      }
      collapsed++;
      continue;
    }
    if (slice_size[i] != operand.base_shape().dimensions(i) &&
        dim_partitions > 1) {
      return absl::nullopt;
    }
    int64 offset_dim = offset_or_window_dims[i - collapsed];
    if (i - collapsed > 0 &&
        offset_dim < offset_or_window_dims[i - collapsed - 1]) {
      // Output offsets are transposed, we do not support this case.
      return absl::nullopt;
    }
    passthrough_tile[offset_dim] = dim_partitions;
  }
  Array<int64> tile_assignment = operand.sharding().tile_assignment();
  tile_assignment.Reshape(passthrough_tile);
  return HloSharding::Tile(tile_assignment);
}

// Returns whether partitioning in the operand only happens in dimensions with
// gather/scatter slice size 1.
bool GatherScatterOperandPartitionedOnlyOnTrivialSliceDims(
    const PartitionedHlo& operand, absl::Span<const int64> index_map,
    absl::Span<const int64> slice_size, int64 num_partitions) {
  if (operand.sharding().IsTileMaximal()) {
    return false;
  }
  int64 trivial_slice_dims_partitions = 1;
  for (int64 dim : index_map) {
    if (slice_size[dim] == 1) {
      trivial_slice_dims_partitions *=
          operand.sharding().tile_assignment().dim(dim);
    }
  }
  return trivial_slice_dims_partitions == num_partitions;
}

// Returns the min and max for the indices (replicated) in a scatter/gather
// which has the operand partitioned on trivial slice dimensions (slice size 1).
std::pair<HloInstruction*, HloInstruction*>
IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
    const PartitionedHlo& operand, const PartitionedHlo& replicated_indices,
    HloInstruction* partition_id, absl::Span<const int64> index_map,
    int64 index_vector_dim, SpmdBuilder* b) {
  auto operand_offsets = MakePartitionOffsets(
      operand.base_shape(), operand.sharding(), partition_id, b);
  // Find the per-dimension index bounds.
  std::vector<HloInstruction*> min_indices;
  std::vector<HloInstruction*> max_indices;
  for (int64 i = 0; i < index_map.size(); ++i) {
    int64 dim = index_map[i];
    int64 partitions = operand.sharding().tile_assignment().dim(dim);
    if (partitions == 1) {
      min_indices.push_back(CreateR0WithType<int32>(
          replicated_indices.base_shape().element_type(), 0, b));
      max_indices.push_back(CreateR0WithType<int32>(
          replicated_indices.base_shape().element_type(),
          operand.base_shape().dimensions(dim), b));
      continue;
    }
    auto offset = operand_offsets[dim];
    if (offset->shape().element_type() !=
        replicated_indices.base_shape().element_type()) {
      offset = b->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeShape(replicated_indices.base_shape().element_type(),
                               {}),
          offset));
    }
    min_indices.push_back(offset);
    auto partition_size_minus_1 =
        CreateR0WithType<int32>(replicated_indices.base_shape().element_type(),
                                operand.hlo()->shape().dimensions(dim) - 1, b);
    max_indices.push_back(b->AddInstruction(HloInstruction::CreateBinary(
        offset->shape(), HloOpcode::kAdd, offset, partition_size_minus_1)));
  }
  // Broadcast the index bounds to the same shape as the indices.
  HloInstruction* broadcast_min;
  HloInstruction* broadcast_max;
  if (index_vector_dim < replicated_indices.base_shape().rank()) {
    // The index vector is an R1, we need to reshape individual bounds to
    // [1], and concat them if there are more than one.
    for (int64 i = 0; i < min_indices.size(); ++i) {
      min_indices[i] = b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(min_indices[i]->shape().element_type(), {1}),
          min_indices[i]));
      max_indices[i] = b->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(max_indices[i]->shape().element_type(), {1}),
          max_indices[i]));
    }
    int64 slice_dims = max_indices.size();
    if (slice_dims > 1) {
      min_indices[0] = b->AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeShape(min_indices[0]->shape().element_type(),
                               {slice_dims}),
          min_indices, 0));
      max_indices[0] = b->AddInstruction(HloInstruction::CreateConcatenate(
          min_indices[0]->shape(), max_indices, 0));
    }
    broadcast_min = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), min_indices[0], {index_vector_dim}));
    broadcast_max = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), max_indices[0], {index_vector_dim}));
  } else {
    CHECK_EQ(max_indices.size(), 1);
    broadcast_min = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), min_indices[0], {}));
    broadcast_max = b->AddInstruction(HloInstruction::CreateBroadcast(
        replicated_indices.base_shape(), max_indices[0], {}));
  }
  return {broadcast_min, broadcast_max};
}

Status SpmdPartitioningVisitor::HandleScatter(HloInstruction* hlo) {
  auto scatter = Cast<HloScatterInstruction>(hlo);
  auto dnums = scatter->scatter_dimension_numbers();
  auto operand = GetPartitionedHlo(scatter->operand(0));
  auto indices = GetPartitionedHlo(scatter->operand(1));
  auto updates = GetPartitionedHlo(scatter->operand(2));
  std::vector<int64> slice_size(operand.base_shape().rank(), 1);
  int64 num_update_window_dims = 0;
  for (int64 i = 0; i < operand.base_shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.inserted_window_dims(), i)) {
      continue;
    }
    slice_size[i] = updates.base_shape().dimensions(
        dnums.update_window_dims(num_update_window_dims++));
  }
  std::vector<int64> inserted_window_dims(dnums.inserted_window_dims().begin(),
                                          dnums.inserted_window_dims().end());
  std::vector<int64> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64> update_window_dims(dnums.update_window_dims().begin(),
                                        dnums.update_window_dims().end());
  if (!operand.sharding().IsTileMaximal()) {
    auto maybe_passthrough = PassthroughOperandToGatherOutputOrScatterUpdate(
        operand, updates.base_shape(), inserted_window_dims,
        scatter_dims_to_operand_dims, update_window_dims, slice_size);
    // Handle pass through cases if we can use compatible sharding for update.
    if (maybe_passthrough.has_value()) {
      indices = indices.Reshard(HloSharding::Replicate());
      updates = updates.Reshard(*maybe_passthrough);
      auto pscatter = b_.AddInstruction(HloInstruction::CreateScatter(
          operand.hlo()->shape(), operand.hlo(), indices.hlo(), updates.hlo(),
          scatter->to_apply(), dnums, scatter->indices_are_sorted(),
          scatter->unique_indices()));
      pscatter->set_sharding(*maybe_passthrough);
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pscatter, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
    if (GatherScatterOperandPartitionedOnlyOnTrivialSliceDims(
            operand, scatter_dims_to_operand_dims, slice_size,
            num_partitions_) &&
        ShapeSizeInBytes(updates.base_shape()) <
            ShapeSizeInBytes(scatter->shape())) {
      // Operand is sharded on trivial slice dims (update slice size 1). We can
      // adjust the indices on each partition by subtracting the offsets. Then
      // we execute a scatter on full updated indices, and out-of-bound accesses
      // will have no effect on the result as guaranteed by the scatter
      // semantics.
      indices = indices.Reshard(HloSharding::Replicate());
      updates = updates.Reshard(HloSharding::Replicate());
      HloInstruction* indices_min;
      HloInstruction* indices_max_unused;
      std::tie(indices_min, indices_max_unused) =
          IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, indices, partition_id_, scatter_dims_to_operand_dims,
              dnums.index_vector_dim(), &b_);
      auto adjusted_indices = b_.AddInstruction(HloInstruction::CreateBinary(
          indices.hlo()->shape(), HloOpcode::kSubtract, indices.hlo(),
          indices_min));
      auto pscatter = b_.AddInstruction(HloInstruction::CreateScatter(
          operand.hlo()->shape(), operand.hlo(), adjusted_indices,
          updates.hlo(), scatter->to_apply(), dnums,
          scatter->indices_are_sorted(), scatter->unique_indices()));
      pscatter->set_sharding(operand.sharding());
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pscatter, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleSlice(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  auto operand = GetPartitionedHlo(hlo->operand(0)).Reshard(sharding);

  // Create a window config to represent the slice.
  Window window;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_stride(hlo->slice_strides(i));
    dim->set_window_dilation(1);
    dim->set_window_reversal(false);
    dim->set_padding_low(-hlo->slice_starts(i));
    dim->set_padding_high(hlo->slice_limits(i) -
                          hlo->operand(0)->shape().dimensions(i));
    dim->set_base_dilation(1);
  }

  auto reshard_operand = operand.ReshardAsWindowedInput(
      window, sharding,
      CreateZero(ShapeUtil::MakeShape(hlo->shape().element_type(), {}), &b_),
      /*mask_invalid_region=*/false);
  if (!reshard_operand.has_value()) {
    return DefaultAction(hlo);
  }
  TF_RET_CHECK(!reshard_operand->dynamic_slice_index_on_output.has_value());
  const Shape& operand_shape = reshard_operand->sharded_input->shape();

  std::vector<int64> start_indices = hlo->slice_starts();
  std::vector<int64> limit_indices = hlo->slice_limits();
  std::vector<int64> strides = hlo->slice_strides();
  bool need_slice = false;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    auto dim = reshard_operand->shard_window.dimensions(i);
    start_indices[i] = -dim.padding_low();
    limit_indices[i] = operand_shape.dimensions(i) + dim.padding_high();
    if (start_indices[i] != 0 || strides[i] != 1 ||
        limit_indices[i] != operand_shape.dimensions(i)) {
      need_slice = true;
    }
  }

  SetPartitionedHlo(hlo, [&] {
    if (need_slice) {
      auto shard_shape = MakePartitionedShape(hlo->shape(), sharding);
      return b_.AddInstruction(HloInstruction::CreateSlice(
          shard_shape, reshard_operand->sharded_input, start_indices,
          limit_indices, strides));
    }
    return reshard_operand->sharded_input;
  });

  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleSort(HloInstruction* hlo) {
  HloSharding sharding = hlo->sharding();
  if (hlo->shape().IsTuple()) {
    // Check that all elements are sharded in the same way.
    if (hlo->shape().tuple_shapes_size() == 0) {
      return DefaultAction(hlo);
    }
    sharding = hlo->sharding().GetSubSharding(hlo->shape(), {0});
    for (int64 i = 1; i < hlo->operand_count(); ++i) {
      if (sharding != hlo->sharding().GetSubSharding(hlo->shape(), {i})) {
        return DefaultAction(hlo);
      }
    }
  }
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  for (int64 dim : hlo->dimensions()) {
    if (sharding.tile_assignment().dim(dim) > 1) {
      return DefaultAction(hlo);
    }
  }
  // Reshard operands to the same as the output.
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : hlo->operands()) {
    new_operands.push_back(GetPartitionedHlo(operand).Reshard(sharding).hlo());
  }
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(hlo->CloneWithNewOperands(
        MakePartitionedShape(hlo->shape(), hlo->sharding()), new_operands));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (hlo->custom_call_target() == "SPMDFullToShardShape") {
    // This op switches from auto partitioning to manual partitioning.
    auto input_partitioned = GetPartitionedHlo(hlo->operand(0));
    if (!EvenlyPartitions(hlo->shape(), input_partitioned.sharding())) {
      input_partitioned = input_partitioned.PadWithValue(
          CreateR0WithType(hlo->shape().element_type(), 0, &b_));
    }
    auto input = input_partitioned.hlo();
    CHECK(hlo->sharding().IsReplicated());
    CHECK(ShapeUtil::Compatible(input->shape(), hlo->shape()));
    auto copy = b_.AddInstruction(
        HloInstruction::CreateUnary(input->shape(), HloOpcode::kCopy, input));
    SetPartitionedHlo(hlo, [&] { return copy; });
    return Status::OK();
  }
  if (hlo->custom_call_target() == "SPMDShardToFullShape") {
    // This op switches from manual partitioning to auto partitioning.
    auto input = GetPartitionedHlo(hlo->operand(0)).hlo();
    CHECK(input->sharding().IsReplicated());
    auto copy = b_.AddInstruction(
        HloInstruction::CreateUnary(input->shape(), HloOpcode::kCopy, input));
    CHECK(ShapeUtil::Compatible(
        copy->shape(), MakePartitionedShape(hlo->shape(), hlo->sharding())));
    SetPartitionedHlo(hlo, [&] { return copy; });
    return Status::OK();
  }
  if (hlo->custom_call_target() != "TopK") {
    return DefaultAction(hlo);
  }

  if (!hlo->operand(0)->has_sharding()) {
    return DefaultAction(hlo);
  }

  const HloSharding& sharding = hlo->operand(0)->sharding();
  if (sharding.IsTileMaximal() || sharding.IsReplicated()) {
    return DefaultAction(hlo);
  }

  const int64 sort_dim = 1;
  const int64 shard_count = sharding.tile_assignment().dim(sort_dim);

  if (shard_count <= 1) {
    return DefaultAction(hlo);
  }

  const int64 input_size = hlo->operand(0)->shape().dimensions(sort_dim);
  const int64 batch_size = hlo->shape().tuple_shapes(0).dimensions(0);
  const int64 k = hlo->shape().tuple_shapes(0).dimensions(sort_dim);
  const int64 per_partition_size = CeilOfRatio(input_size, shard_count);

  if (k >= per_partition_size) {
    return DefaultAction(hlo);
  }

  auto input = hlo->operand(0);
  const auto element_type = input->shape().element_type();

  // Pad input with minimal value.
  auto min_value = b_.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MinValue(element_type)));
  // TODO(wangtao): add test to see if -NaN < -Inf in BF16.
  if (element_type == F32) {
    auto float_pad_value = std::numeric_limits<float>::quiet_NaN();
    min_value = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<float>(-float_pad_value)));
  }
  auto partitioned_input = GetPartitionedHlo(input).PadWithValue(min_value);

  // Each partition needs to do TopK separately, thus the base shape
  // becomes [batch_size, k * shard_count].
  const Shape replicated_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(hlo->operand(0)->shape().element_type(),
                            {batch_size, k * shard_count}),
       ShapeUtil::MakeShape(S32, {batch_size, k * shard_count})});
  auto custom_call_sharding =
      sharding.GetTupleSharding(replicated_shape).ValueOrDie();
  auto shard_shape =
      MakePartitionedShape(replicated_shape, custom_call_sharding);
  auto topk = b_.AddInstruction(
      hlo->CloneWithNewOperands(shard_shape, {partitioned_input.hlo()}));
  topk->set_sharding(custom_call_sharding);
  // Partition customcall.
  PartitionedHlo partitioned_topk(topk, replicated_shape,
                                  MakePartitioningState());
  topk = partitioned_topk.hlo();

  // Get value from TopK.
  HloInstruction* value_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          topk->shape().tuple_shapes(0), topk, 0));
  value_gte->set_sharding(sharding);
  // Partition GetTupleElement of value.
  PartitionedHlo value_partitioned_gte(
      value_gte, partitioned_topk.base_shape().tuple_shapes(0),
      MakePartitioningState());
  // Reshard value to be replicated.
  auto replicated_value_gte =
      value_partitioned_gte.Reshard(HloSharding::Replicate()).hlo();

  // Get index from TopK.
  HloInstruction* index_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          topk->shape().tuple_shapes(1), topk, 1));
  auto partition_id_s32 = b_.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::MakeShape(S32, partition_id_->shape().dimensions()),
      partition_id_));
  // Add per partition offset to index, index returned from CustomCall always
  // starts from 0.
  auto index_offset = b_.AddInstruction(HloInstruction::CreateBroadcast(
      index_gte->shape(),
      b_.AddInstruction(HloInstruction::CreateBinary(
          partition_id_s32->shape(), HloOpcode::kMultiply, partition_id_s32,
          b_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32>(per_partition_size))))),
      {}));
  index_gte = b_.AddInstruction(HloInstruction::CreateBinary(
      index_offset->shape(), HloOpcode::kAdd, index_gte, index_offset));
  index_gte->set_sharding(sharding);
  // Parttion GetTupleElement of index.
  PartitionedHlo index_partitioned_gte(
      index_gte, partitioned_topk.base_shape().tuple_shapes(1),
      MakePartitioningState());
  // Reshard index to be replicated.
  auto replicated_index_gte =
      index_partitioned_gte.Reshard(HloSharding::Replicate()).hlo();

  // Creates replicated sort to do TopK, the input is value and index pairs
  // from all the partitions. The reason to use Sort instead of CustomCall TopK
  // is CustomCall only takes value as input. There will be an extra Gather
  // to get the correct index if CustomCall is used here.

  // Create comparator for the sort.
  XlaBuilder b("Sort.Compare");
  XlaComputation comparator = CreateScalarComparisonComputation(
      "compare-value-and-index", {input->shape().element_type(), S32}, {Gt, Lt},
      &b);
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, comparator.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(comparator.proto(), config));
  HloCloneContext context(module_);
  auto compare_computation =
      module_->DeepCloneComputation(new_module->entry_computation(), &context);
  auto sort = b_.AddInstruction(HloInstruction::CreateSort(
      replicated_shape, sort_dim, {replicated_value_gte, replicated_index_gte},
      compare_computation, true));
  sort->set_sharding(
      HloSharding::Replicate().GetTupleSharding(sort->shape()).ValueOrDie());
  PartitionedHlo replicated_sort(sort, replicated_shape,
                                 MakePartitioningState());

  // Slice value and index from top-k for output.
  HloInstruction* sort_value_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          replicated_sort.hlo()->shape().tuple_shapes(0), replicated_sort.hlo(),
          0));
  HloInstruction* sort_index_gte =
      b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          replicated_sort.hlo()->shape().tuple_shapes(1), replicated_sort.hlo(),
          1));
  const Shape& hlo_shape = sort_value_gte->shape();
  auto hlo_dims = hlo_shape.dimensions();
  std::vector<int64> start_indices(hlo_shape.dimensions_size(), 0);
  std::vector<int64> limit_indices(hlo_dims.begin(), hlo_dims.end());
  std::vector<int64> strides(hlo_shape.dimensions_size(), sort_dim);
  limit_indices[sort_dim] = k;
  auto output_shape = hlo_shape;
  output_shape.set_dimensions(sort_dim, k);
  // Slice value from final sort.
  HloInstruction* slice_sort_value =
      b_.AddInstruction(HloInstruction::CreateSlice(
          output_shape, sort_value_gte, start_indices, limit_indices, strides));
  // Slice index from final sort.
  auto index_output_shape = sort_index_gte->shape();
  index_output_shape.set_dimensions(sort_dim, k);
  HloInstruction* slice_index_value = b_.AddInstruction(
      HloInstruction::CreateSlice(index_output_shape, sort_index_gte,
                                  start_indices, limit_indices, strides));
  auto create_tuple = b_.AddInstruction(
      HloInstruction::CreateTuple({slice_sort_value, slice_index_value}));
  create_tuple->set_sharding(HloSharding::Replicate());

  SetPartitionedHlo(hlo, PartitionedHlo(create_tuple, create_tuple->shape(),
                                        MakePartitioningState())
                             .Reshard(hlo->sharding()));

  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleTranspose(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  std::vector<int64> inverse_dimensions(hlo->shape().rank());
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    inverse_dimensions[hlo->dimensions(i)] = i;
  }
  auto desired_operand_sharding =
      hlo_sharding_util::TransposeSharding(sharding, inverse_dimensions);

  auto operand = GetPartitionedHlo(hlo->operand(0))
                     .Reshard(desired_operand_sharding)
                     .hlo();
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(hlo->CloneWithNewOperands(
        MakePartitionedShape(hlo->shape(), hlo->sharding()), {operand}));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleReshape(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  auto operand = GetPartitionedHlo(hlo->operand(0));
  // The output shape is the source and the operand shape is the target to get
  // the aligned sharding for the operand.
  auto desired_operand_sharding = hlo_sharding_util::ReshapeSharding(
      hlo->shape(), hlo->operand(0)->shape(), hlo->sharding());
  if (desired_operand_sharding.has_value()) {
    auto operand_hlo = operand.Reshard(*desired_operand_sharding).hlo();
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(hlo->CloneWithNewOperands(
          MakePartitionedShape(hlo->shape(), hlo->sharding()), {operand_hlo}));
    });
    return Status::OK();
  }

  // Try use halo exchange for certain split-dim/merge-dims cases.
  // ReshapeSharding failed in these cases probably due to uneven partitioning,
  // where halo exchange could help. Specifically we check the following
  // conditions to detect supported cases:
  // 1) Both input and output are partitioned on one dimension.
  // 2) The combined size of dimensions before the partitioned dimension are the
  // same on input and output. This means we don't need to consider the major
  // dimensions.
  // 3) Let A = the input size on the partitioned dimension, and
  //        B = the output size on the partitioned dimension; then
  //    either A % B == 0 (split dim) or B % A == 0 (merge dims).
  auto maybe_input_sharded_dim = UniqueTiledDim(operand.sharding());
  auto maybe_output_sharded_dim = UniqueTiledDim(sharding);
  if (!maybe_input_sharded_dim || !maybe_output_sharded_dim) {
    return DefaultAction(hlo);
  }
  int64 input_sharded_dim = *maybe_input_sharded_dim;
  int64 output_sharded_dim = *maybe_output_sharded_dim;
  // Check that the major dims before the sharded dim have the same total size
  // for input and output.
  int64 input_major_dims_size = 1;
  for (int64 i = 0; i < input_sharded_dim; ++i) {
    input_major_dims_size *= operand.base_shape().dimensions(i);
  }
  int64 output_major_dims_size = 1;
  for (int64 i = 0; i < output_sharded_dim; ++i) {
    output_major_dims_size *= hlo->shape().dimensions(i);
  }
  if (input_major_dims_size != output_major_dims_size) {
    return DefaultAction(hlo);
  }
  // Fix potential device ordering mismatch in tile assignment.
  Array<int64> new_input_tile_assignment = sharding.tile_assignment();
  new_input_tile_assignment.Reshape(
      operand.sharding().tile_assignment().dimensions());
  operand = operand.Reshard(HloSharding::Tile(new_input_tile_assignment));

  int64 input_dim_size = operand.base_shape().dimensions(input_sharded_dim);
  int64 output_dim_size = hlo->shape().dimensions(output_sharded_dim);
  auto input_shard_shape =
      MakePartitionedShape(operand.base_shape(), operand.sharding());
  auto output_shard_shape = MakePartitionedShape(hlo->shape(), sharding);
  if (input_dim_size % output_dim_size == 0) {
    // Split dim.
    int64 split_factor = input_dim_size / output_dim_size;
    int64 output_shard_size = output_shard_shape.dimensions(output_sharded_dim);
    // Use halo exchange to fix misaligned data.
    Window window;
    for (int64 i = 0; i < hlo->shape().rank(); ++i) {
      WindowDimension* dim = window.add_dimensions();
      dim->set_size(1);
      dim->set_stride(1);
      dim->set_window_dilation(1);
      dim->set_window_reversal(false);
      dim->set_base_dilation(1);
      dim->set_padding_low(0);
      if (i == input_sharded_dim) {
        dim->set_padding_high(output_shard_size * split_factor *
                                  num_partitions_ -
                              input_dim_size);
      } else {
        dim->set_padding_high(0);
      }
    }

    auto reshard_operand = operand.ReshardAsWindowedInput(
        window, operand.sharding(),
        CreateZero(ShapeUtil::MakeShape(hlo->shape().element_type(), {}), &b_),
        /*mask_invalid_region=*/false);
    if (!reshard_operand.has_value()) {
      return DefaultAction(hlo);
    }
    TF_RET_CHECK(!reshard_operand->dynamic_slice_index_on_output.has_value());
    CHECK_EQ(
        reshard_operand->sharded_input->shape().dimensions(input_sharded_dim),
        output_shard_size * split_factor);
    SetPartitionedHlo(hlo, [&] {
      // Do a local reshape.
      return b_.AddInstruction(HloInstruction::CreateReshape(
          output_shard_shape, reshard_operand->sharded_input));
    });
    return Status::OK();
  } else if (output_dim_size % input_dim_size == 0) {
    // Merge dims.
    int64 merge_factor = output_dim_size / input_dim_size;
    // First reshape locally. (The sharded dimension could include padded data.)
    auto tmp_shard_shape = output_shard_shape;
    tmp_shard_shape.set_dimensions(
        output_sharded_dim,
        input_shard_shape.dimensions(input_sharded_dim) * merge_factor);
    auto tmp_reshape = b_.AddInstruction(
        HloInstruction::CreateReshape(tmp_shard_shape, operand.hlo()));
    tmp_reshape->set_metadata(hlo->metadata());
    tmp_reshape->set_sharding(hlo->sharding());
    auto tmp_full_shape = tmp_shard_shape;
    tmp_full_shape.set_dimensions(
        output_sharded_dim,
        tmp_shard_shape.dimensions(output_sharded_dim) * num_partitions_);
    auto tmp_output =
        PartitionedHlo(tmp_reshape, tmp_full_shape, MakePartitioningState());

    // Use halo exchange to fix misaligned data.
    Window window;
    for (int64 i = 0; i < tmp_shard_shape.rank(); ++i) {
      WindowDimension* dim = window.add_dimensions();
      dim->set_size(1);
      dim->set_stride(1);
      dim->set_window_dilation(1);
      dim->set_window_reversal(false);
      dim->set_base_dilation(1);
      dim->set_padding_low(0);
      if (i == output_sharded_dim) {
        dim->set_padding_high(output_dim_size -
                              tmp_shard_shape.dimensions(output_sharded_dim) *
                                  num_partitions_);
      } else {
        dim->set_padding_high(0);
      }
    }

    auto reshard_output = tmp_output.ReshardAsWindowedInput(
        window, sharding,
        CreateZero(ShapeUtil::MakeShape(hlo->shape().element_type(), {}), &b_),
        /*mask_invalid_region=*/false);
    if (!reshard_output.has_value()) {
      return DefaultAction(hlo);
    }
    TF_RET_CHECK(!reshard_output->dynamic_slice_index_on_output.has_value());
    CHECK_EQ(
        reshard_output->sharded_input->shape().dimensions(output_sharded_dim),
        output_shard_shape.dimensions(output_sharded_dim));
    SetPartitionedHlo(hlo, [&] { return reshard_output->sharded_input; });
    return Status::OK();
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleIota(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  SetPartitionedHlo(hlo, [&] {
    int64 dimension = Cast<HloIotaInstruction>(hlo)->iota_dimension();
    auto iota = b_.AddInstruction(HloInstruction::CreateIota(
        MakePartitionedShape(hlo->shape(), sharding), dimension));

    if (sharding.tile_assignment().dim(dimension) > 1) {
      auto partition_ordinals =
          MakeTiledPartitionOrdinals(sharding, partition_id_, &b_);
      auto multiplier = b_.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(iota->shape().dimensions(dimension))));
      auto offset = b_.AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeShape(S32, {}), HloOpcode::kMultiply,
          partition_ordinals[dimension], multiplier));
      if (iota->shape().element_type() != S32) {
        offset = b_.AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::MakeShape(iota->shape().element_type(), {}), offset));
      }
      auto broadcast = b_.AddInstruction(
          HloInstruction::CreateBroadcast(iota->shape(), offset, {}));
      return b_.AddInstruction(HloInstruction::CreateBinary(
          iota->shape(), HloOpcode::kAdd, iota, broadcast));
    }

    return iota;
  });

  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleSingleDevice(const HloInstruction* hlo) {
  TF_RET_CHECK(hlo->sharding().HasUniqueDevice());
  int64 device = hlo->sharding().GetUniqueDevice();
  const HloSharding sharding = HloSharding::AssignDevice(device);

  std::vector<HloInstruction*> operands;
  std::vector<Shape> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operands.push_back(GetPartitionedHlo(operand).Reshard(sharding).hlo());
    operand_shapes.push_back(operand->shape());
  }
  auto operand = b_.AddInstruction(HloInstruction::CreateTuple(operands));
  auto operand_shape = ShapeUtil::MakeTupleShape(operand_shapes);

  auto on_device = b_.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(device)));
  auto pred = b_.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), partition_id_, on_device,
      ComparisonDirection::kEq));

  SpmdBuilder true_b("true_computation", visiting_hlo_);
  HloComputation* true_computation;
  {
    auto param = true_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, operand_shape, "true_branch_param"));
    std::vector<HloInstruction*> new_operands;
    for (int64 i = 0; i < operands.size(); ++i) {
      new_operands.push_back(true_b.AddInstruction(
          HloInstruction::CreateGetTupleElement(operand_shapes[i], param, i)));
    }
    auto root = true_b.AddInstruction(
        hlo->CloneWithNewOperands(hlo->shape(), new_operands));
    true_computation = module_->AddEmbeddedComputation(true_b.Build(root));
  }

  SpmdBuilder false_b("false_computation", visiting_hlo_);
  HloComputation* false_computation;
  {
    false_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, operand_shape, "false_branch_param"));
    auto root = CreateZero(hlo->shape(), &false_b);
    false_computation = module_->AddEmbeddedComputation(false_b.Build(root));
  }

  SetPartitionedHlo(hlo, [&]() {
    return b_.AddInstruction(HloInstruction::CreateConditional(
        hlo->shape(), pred, operand, true_computation, operand,
        false_computation));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleAllReduce(HloInstruction* hlo) {
  if (hlo->IsCrossReplicaAllReduce() && hlo->operand_count() == 1) {
    return HandleElementwise(hlo);
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleBroadcast(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  auto& operand = GetPartitionedHlo(hlo->operand(0));

  // Tiled output.
  std::vector<int64> wanted_input_tile_size(operand.base_shape().rank());
  std::vector<int64> sharded_new_dims;
  for (int64 i = 0; i < operand.base_shape().rank(); ++i) {
    wanted_input_tile_size[i] =
        hlo->sharding().tile_assignment().dim(hlo->dimensions(i));
  }
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (!absl::c_linear_search(hlo->dimensions(), i) &&
        hlo->sharding().tile_assignment().dim(i) > 1) {
      sharded_new_dims.push_back(i);
    }
  }
  if (sharded_new_dims.empty()) {
    // The new dimensions are replicated, so that we can do the adjustment on
    // the input.
    Array<int64> wanted_input_tile_assignment(wanted_input_tile_size);
    wanted_input_tile_assignment.Each(
        [&](absl::Span<const int64> indices, int64* val) {
          std::vector<int64> indices_in_broadcast(hlo->shape().rank(), 0);
          for (int64 i = 0; i < operand.base_shape().rank(); ++i) {
            indices_in_broadcast[hlo->dimensions(i)] = indices[i];
          }
          *val = hlo->sharding().tile_assignment()(indices_in_broadcast);
        });
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(hlo->CloneWithNewOperands(
          MakePartitionedShape(hlo->shape(), hlo->sharding()),
          {operand.Reshard(HloSharding::Tile(wanted_input_tile_assignment))
               .hlo()}));
    });
  } else {
    auto input = operand.Reshard(HloSharding::Replicate()).hlo();
    // We pad and shard the input first, then broadcast to the final shard
    // shape.
    auto output_offsets =
        MakePartitionOffsets(hlo->shape(), hlo->sharding(), partition_id_, &b_);
    std::vector<HloInstruction*> input_offsets(operand.base_shape().rank());
    auto output_shard_shape =
        MakePartitionedShape(hlo->shape(), hlo->sharding());
    auto input_shard_shape = input->shape();
    auto padded_input_shape = input->shape();
    for (int64 i = 0; i < input_offsets.size(); ++i) {
      input_offsets[i] = output_offsets[hlo->dimensions(i)];
      input_shard_shape.set_dimensions(
          i, output_shard_shape.dimensions(hlo->dimensions(i)));
      padded_input_shape.set_dimensions(
          i, hlo->sharding().tile_assignment().dim(hlo->dimensions(i)) *
                 input_shard_shape.dimensions(i));
    }
    auto padded_input = PadToShape(input, padded_input_shape, &b_);
    auto input_shard =
        ShapeUtil::Compatible(input_shard_shape, padded_input->shape())
            ? padded_input
            : b_.AddInstruction(HloInstruction::CreateDynamicSlice(
                  input_shard_shape, padded_input, input_offsets,
                  input_shard_shape.dimensions()));
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(
          hlo->CloneWithNewOperands(output_shard_shape, {input_shard}));
    });
  }
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleConstant(HloInstruction* hlo) {
  const Literal& literal = hlo->literal();
  if (literal.shape().IsTuple() ||
      (!hlo->sharding().IsTileMaximal() &&
       (!EvenlyPartitions(hlo->shape(), hlo->sharding()) ||
        !literal.IsAllFirst()))) {
    return DefaultAction(hlo);
  }

  SetPartitionedHlo(hlo, [&]() {
    auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
    std::vector<int64> start_indices(hlo->shape().rank(), 0);
    auto constant = b_.AddInstruction(HloInstruction::CreateConstant(
        literal.Slice(start_indices, shard_shape.dimensions())));
    *constant->mutable_shape() = shard_shape;
    return constant;
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleDynamicSlice(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->sharding().tile_assignment().dim(i) != 1 &&
        (hlo->dynamic_slice_sizes()[i] != hlo->shape().dimensions(i) ||
         !hlo->operand(i + 1)->IsConstant() ||
         !hlo->operand(i + 1)->literal().IsZero({}))) {
      // We currently do not partition the sliced dimensions.
      return DefaultAction(hlo);
    }
  }
  std::vector<HloInstruction*> new_indices(hlo->shape().rank());
  auto new_input =
      GetPartitionedHlo(hlo->operand(0)).Reshard(hlo->sharding()).hlo();
  for (int64 i = 0; i < new_indices.size(); ++i) {
    // Replicate the indices.
    new_indices[i] = GetPartitionedHlo(hlo->operand(i + 1))
                         .Reshard(HloSharding::Replicate())
                         .hlo();
  }
  SetPartitionedHlo(hlo, [&]() {
    auto partitioned_shape =
        MakePartitionedShape(hlo->shape(), hlo->sharding());
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        partitioned_shape, new_input, new_indices,
        partitioned_shape.dimensions()));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleDynamicUpdateSlice(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->sharding().tile_assignment().dim(i) != 1 &&
        (hlo->operand(1)->shape().dimensions(i) != hlo->shape().dimensions(i) ||
         !hlo->operand(i + 2)->IsConstant() ||
         !hlo->operand(i + 2)->literal().IsZero({}))) {
      // We currently do not partition the sliced dimensions.
      return DefaultAction(hlo);
    }
  }
  std::vector<HloInstruction*> new_indices(hlo->shape().rank());
  auto new_input =
      GetPartitionedHlo(hlo->operand(0)).Reshard(hlo->sharding()).hlo();
  auto new_update =
      GetPartitionedHlo(hlo->operand(1)).Reshard(hlo->sharding()).hlo();
  for (int64 i = 0; i < new_indices.size(); ++i) {
    // Replicate the indices.
    new_indices[i] = GetPartitionedHlo(hlo->operand(i + 2))
                         .Reshard(HloSharding::Replicate())
                         .hlo();
  }
  SetPartitionedHlo(hlo, [&]() {
    auto partitioned_shape =
        MakePartitionedShape(hlo->shape(), hlo->sharding());
    return b_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        partitioned_shape, new_input, new_update, new_indices));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleGather(HloInstruction* hlo) {
  auto gather = Cast<HloGatherInstruction>(hlo);
  const auto& dnums = gather->gather_dimension_numbers();
  auto operand = GetPartitionedHlo(gather->operand(0));
  auto indices = GetPartitionedHlo(gather->operand(1));
  std::vector<int64> collapsed_slice_dims(dnums.collapsed_slice_dims().begin(),
                                          dnums.collapsed_slice_dims().end());
  std::vector<int64> start_index_map(dnums.start_index_map().begin(),
                                     dnums.start_index_map().end());
  std::vector<int64> offset_dims(dnums.offset_dims().begin(),
                                 dnums.offset_dims().end());
  if (!operand.sharding().IsTileMaximal()) {
    auto maybe_passthrough = PassthroughOperandToGatherOutputOrScatterUpdate(
        operand, gather->shape(), collapsed_slice_dims, start_index_map,
        offset_dims, gather->gather_slice_sizes());
    if (maybe_passthrough.has_value()) {
      indices = indices.Reshard(HloSharding::Replicate());
      auto pshape = MakePartitionedShape(gather->shape(), *maybe_passthrough);
      std::vector<int64> pslice_sizes(gather->gather_slice_sizes().begin(),
                                      gather->gather_slice_sizes().end());
      for (int64 i = 0; i < pslice_sizes.size(); ++i) {
        if (operand.sharding().tile_assignment().dim(i) > 1) {
          pslice_sizes[i] = operand.hlo()->shape().dimensions(i);
        }
      }
      auto pgather = b_.AddInstruction(HloInstruction::CreateGather(
          pshape, operand.hlo(), indices.hlo(), dnums, pslice_sizes,
          gather->indices_are_sorted()));
      pgather->set_sharding(*maybe_passthrough);
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(pgather, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
    if (GatherScatterOperandPartitionedOnlyOnTrivialSliceDims(
            operand, start_index_map, gather->gather_slice_sizes(),
            num_partitions_) &&
        ShapeSizeInBytes(gather->shape()) <
            ShapeSizeInBytes(gather->operand(0)->shape())) {
      indices = indices.Reshard(HloSharding::Replicate());
      // Now the operand is partitioned in trivial slice dimensions, and the
      // indices are replicated. We execute a gather on partitioned operand,
      // with full number of indices, where out-of-bounds indices are clamped,
      // and masked out with 0 in the result; then we use all-reduce to combine
      // results. Although gather will not get faster, we avoided the need to
      // replicate the operand.
      HloInstruction* indices_min;
      HloInstruction* indices_max;
      std::tie(indices_min, indices_max) =
          IndexBoundsForGatherScatterOperandPartitionedOnTrivialSliceDims(
              operand, indices, partition_id_, start_index_map,
              dnums.index_vector_dim(), &b_);
      // Clamp the indices.
      auto adjusted_indices = b_.AddInstruction(HloInstruction::CreateTernary(
          indices.base_shape(), HloOpcode::kClamp, indices_min, indices.hlo(),
          indices_max));
      // Adjust the indices by subtracting the offset.
      adjusted_indices = b_.AddInstruction(HloInstruction::CreateBinary(
          indices.base_shape(), HloOpcode::kSubtract, adjusted_indices,
          indices_min));
      // Gather on adjusted indices.
      auto pgather = b_.AddInstruction(HloInstruction::CreateGather(
          gather->shape(), operand.hlo(), adjusted_indices, dnums,
          gather->gather_slice_sizes(), gather->indices_are_sorted()));
      // Mask out invalid results.
      auto filter = b_.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(indices.base_shape(), PRED),
          indices.hlo(), indices_min, ComparisonDirection::kLt));
      filter = b_.AddInstruction(HloInstruction::CreateBinary(
          filter->shape(), HloOpcode::kOr, filter,
          b_.AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::ChangeElementType(indices.base_shape(), PRED),
              indices.hlo(), indices_max, ComparisonDirection::kGt))));
      if (dnums.index_vector_dim() < indices.base_shape().rank()) {
        std::vector<int64> reduced_filter_dims;
        for (int64 i = 0; i < filter->shape().rank(); ++i) {
          if (i != dnums.index_vector_dim()) {
            reduced_filter_dims.push_back(filter->shape().dimensions(i));
          }
        }
        filter = b_.AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeShape(PRED, reduced_filter_dims), filter,
            CreateR0WithType(PRED, false, &b_), {dnums.index_vector_dim()},
            MakeBinaryAdd(PRED, module_)));
      }
      std::vector<int64> batch_dims;
      for (int64 i = 0; i < pgather->shape().rank(); ++i) {
        if (!absl::c_linear_search(dnums.offset_dims(), i)) {
          batch_dims.push_back(i);
        }
      }
      auto broadcast_filter = b_.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::ChangeElementType(pgather->shape(), PRED), filter,
          batch_dims));
      auto filtered = b_.AddInstruction(HloInstruction::CreateTernary(
          pgather->shape(), HloOpcode::kSelect, broadcast_filter,
          CreateZero(pgather->shape(), &b_), pgather));
      // Combine from different partitions.
      auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
          &b_, filtered,
          MakeBinaryAdd(filtered->shape().element_type(), module_),
          NewChannel());
      ar->set_sharding(HloSharding::Replicate());
      SetPartitionedHlo(hlo, [&]() {
        return PartitionedHlo(ar, hlo->shape(), MakePartitioningState())
            .Reshard(hlo->sharding())
            .hlo();
      });
      return Status::OK();
    }
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleGetTupleElement(HloInstruction* hlo) {
  const auto& tuple = GetPartitionedHlo(hlo->operand(0));
  auto gte = b_.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetTupleElementShape(tuple.hlo()->shape(), hlo->tuple_index()),
      tuple.hlo(), hlo->tuple_index()));
  SetPartitionedHlo(hlo, [&]() {
    const auto source_sharding = tuple.sharding().GetSubSharding(
        tuple.base_shape(), {hlo->tuple_index()});
    gte->set_sharding(source_sharding);
    PartitionedHlo source_partitioned_gte(gte, hlo->shape(),
                                          MakePartitioningState());
    return source_partitioned_gte.Reshard(hlo->sharding()).hlo();
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleInfeed(HloInstruction* hlo) {
  const Shape& shape = ShapeUtil::GetTupleElementShape(hlo->shape(), 0);
  auto token = GetPartitionedHlo(hlo->operand(0)).hlo();
  if (ShapeUtil::GetLeafCount(shape) == 0) {
    // TODO(b/155819021): HloSharding has issues with tuple-shaped sharding: it
    // requires one element for an empty tuple, but leaf-count number of
    // elements for non-empty tuple. So if it has a nested empty tuple, we
    // cannot invoke GetSubSharding() since it expects a sharding for the empty
    // tuple. This is a workaround for that case.
    SetPartitionedHlo(hlo, [&]() {
      return b_.AddInstruction(
          HloInstruction::CreateInfeed(shape, token, hlo->infeed_config()));
    });
    return Status::OK();
  }
  auto sharding = hlo->sharding().GetSubSharding(hlo->shape(), {0});
  auto shard_shape = MakePartitionedShape(shape, sharding);
  if (EvenlyPartitions(shape, sharding)) {
    SetPartitionedHlo(hlo, [&]() {
      return b_.AddInstruction(HloInstruction::CreateInfeed(
          shard_shape, token, hlo->infeed_config()));
    });
    return Status::OK();
  }

  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }

  // Create a branch for each unique partitioned shape.
  std::vector<Shape> per_branch_partitioned_shapes;
  std::vector<int32> conditional_branch_indices(num_partitions_);
  for (int64 i = 0; i < num_partitions_; ++i) {
    auto partitioned_shape =
        MakeNonPaddedShapeForGivenPartition(shape, sharding, i);
    int64 matching_existing_index = 0;
    for (; matching_existing_index < per_branch_partitioned_shapes.size();
         ++matching_existing_index) {
      if (ShapeUtil::Compatible(
              partitioned_shape,
              per_branch_partitioned_shapes[matching_existing_index])) {
        break;
      }
    }
    if (matching_existing_index < per_branch_partitioned_shapes.size()) {
      conditional_branch_indices[i] = matching_existing_index;
    } else {
      conditional_branch_indices[i] = per_branch_partitioned_shapes.size();
      per_branch_partitioned_shapes.push_back(std::move(partitioned_shape));
    }
  }

  HloInstruction* branch_index;
  if (per_branch_partitioned_shapes.size() == num_partitions_) {
    // Use partition ID as the branch index if each partition has its own
    // branch.
    branch_index = partition_id_;
    // PartitionId's output is U32 but conditional requires S32.
    if (branch_index->shape().element_type() != S32) {
      branch_index = b_.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(branch_index->shape(), S32),
          branch_index));
    }
  } else {
    // Otherwise, use a constant table to look up the branch index.
    auto branch_index_table = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<int32>(conditional_branch_indices)));
    branch_index = b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::MakeShape(S32, {1}), branch_index_table, {partition_id_},
        {1}));
    branch_index = b_.AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(S32, {}), branch_index));
  }

  std::vector<HloComputation*> branches(per_branch_partitioned_shapes.size());
  for (int64 i = 0; i < branches.size(); ++i) {
    SpmdBuilder branch_b(absl::StrCat("infeed_branch_", i), visiting_hlo_);
    auto param = branch_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, token->shape(), "infeed_token_param"));
    auto infeed = branch_b.AddInstruction(HloInstruction::CreateInfeed(
        per_branch_partitioned_shapes[i], param, hlo->infeed_config()));
    branches[i] = module_->AddEmbeddedComputation(branch_b.Build(infeed));
    if (!ShapeUtil::Compatible(per_branch_partitioned_shapes[i], shard_shape)) {
      TF_ASSIGN_OR_RETURN(
          auto padded,
          branches[i]->DeepCopyInstructionWithCustomCopier(
              infeed, [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
                          HloComputation* comp) {
                // Index {1} corresponds to the token.
                if (leaf_index.empty() || leaf_index[0] != 0) {
                  return leaf;
                }
                ShapeIndexView subindex(leaf_index, 1);
                if (ShapeUtil::Compatible(
                        ShapeUtil::GetSubshape(per_branch_partitioned_shapes[i],
                                               subindex),
                        ShapeUtil::GetSubshape(shard_shape, subindex))) {
                  return leaf;
                }
                return PadToShape(leaf,
                                  ShapeUtil::GetSubshape(shard_shape, subindex),
                                  nullptr, comp);
              }));
      branches[i]->set_root_instruction(padded,
                                        /*accept_different_shape=*/true);
    }
  }
  SetPartitionedHlo(hlo, [&]() {
    return b_.AddInstruction(HloInstruction::CreateConditional(
        ShapeUtil::MakeTupleShape({shard_shape, token->shape()}), branch_index,
        branches, std::vector<HloInstruction*>(branches.size(), token)));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandlePad(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    const auto& pd = hlo->padding_config().dimensions(i);
    // Right now we only support non-padded dimensions to be partitioned.
    if (hlo->sharding().tile_assignment().dim(i) > 1 &&
        (pd.edge_padding_high() != 0 || pd.edge_padding_low() != 0 ||
         pd.interior_padding() != 0)) {
      return DefaultAction(hlo);
    }
  }
  auto resharded_lhs =
      GetPartitionedHlo(hlo->operand(0)).Reshard(hlo->sharding()).hlo();
  auto replicated_rhs = GetPartitionedHlo(hlo->operand(1))
                            .Reshard(HloSharding::Replicate())
                            .hlo();
  SetPartitionedHlo(hlo, [&]() {
    auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
    return b_.AddInstruction(hlo->CloneWithNewOperands(
        shard_shape, {resharded_lhs, replicated_rhs}));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleParameter(HloInstruction* hlo) {
  SetPartitionedHlo(hlo, [&]() {
    auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
    auto new_param = b_.AddInstruction(HloInstruction::CreateParameter(
        hlo->parameter_number(), shard_shape, "param"));
    if (hlo->parameter_replicated_at_leaf_buffers()) {
      new_param->set_parameter_replicated_at_leaf_buffers(
          *hlo->parameter_replicated_at_leaf_buffers());
    }
    return new_param;
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleReduce(HloInstruction* hlo) {
  int64 input_count = 1;
  auto per_input_sharding = hlo->sharding();
  if (hlo->shape().IsTuple()) {
    input_count = hlo->shape().tuple_shapes_size();
    CHECK_GT(input_count, 0);
    per_input_sharding = hlo->sharding().GetSubSharding(hlo->shape(), {0});
  }

  std::vector<PartitionedHlo> inputs;
  std::vector<HloInstruction*> inits;
  for (int64 operand_id = 0; operand_id < input_count; ++operand_id) {
    inits.push_back(GetPartitionedHlo(hlo->operand(operand_id + input_count))
                        .Reshard(HloSharding::Replicate())
                        .hlo());
    inputs.push_back(GetPartitionedHlo(hlo->operand(operand_id)));
    if (operand_id > 0) {
      // Make sure all operands are sharded in the same way.
      inputs.back() = inputs.back().Reshard(inputs[0].sharding());
    }
    if (!inputs[0].sharding().IsTileMaximal()) {
      inputs.back() = inputs.back().PadWithValue(inits[operand_id]);
    }
  }
  bool reduce_sharded_dimension = false;
  if (!inputs[0].sharding().IsTileMaximal()) {
    reduce_sharded_dimension = absl::c_any_of(hlo->dimensions(), [&](int64 i) {
      return inputs[0].sharding().tile_assignment().dim(i) > 1;
    });

    // reduce_sharded_dimension is not supported for tuple-shaped reduces.
    if (reduce_sharded_dimension && input_count > 1) {
      return DefaultAction(hlo);
    }

    // Currently we only support reducing all or none of the sharded
    // dimensions.
    if (reduce_sharded_dimension) {
      for (int64 i = 0; i < inputs[0].base_shape().rank(); ++i) {
        if (inputs[0].sharding().tile_assignment().dim(i) > 1 &&
            absl::c_count(hlo->dimensions(), i) == 0) {
          return DefaultAction(hlo);
        }
      }
    }
  }

  std::vector<Shape*> new_operand_shapes(input_count * 2);
  for (int64 i = 0; i < input_count; ++i) {
    new_operand_shapes[i] = inputs[i].hlo()->mutable_shape();
    new_operand_shapes[i + input_count] = inits[i]->mutable_shape();
  }
  // Create the shard shape of the reduce result.
  TF_ASSIGN_OR_RETURN(
      auto reduce_shape,
      ShapeInference::InferReduceShape(new_operand_shapes, hlo->dimensions(),
                                       hlo->to_apply()->ComputeProgramShape()));
  *reduce_shape.mutable_layout() = hlo->shape().layout();

  std::vector<HloInstruction*> input_hlos(input_count);
  for (int64 i = 0; i < input_count; ++i) {
    input_hlos[i] = inputs[i].hlo();
  }
  auto local_reduce = b_.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, input_hlos, inits, hlo->dimensions(), hlo->to_apply()));
  local_reduce->set_metadata(hlo->metadata());

  SetPartitionedHlo(hlo, [&]() {
    HloInstruction* reduce;
    if (reduce_sharded_dimension) {
      CHECK(local_reduce->shape().IsArray());
      reduce = collective_ops_creator_.create_cross_partition_all_reduce(
          &b_, local_reduce, hlo->to_apply(), NewChannel());
      reduce->set_sharding(HloSharding::Replicate());
    } else {
      reduce = local_reduce;
      if (inputs[0].sharding().IsTileMaximal()) {
        reduce->set_sharding(inputs[0].sharding());
      } else {
        // Remove tile assignment dimensions that are reduced.
        std::vector<int64> tile_dimensions;
        for (int64 i = 0; i < input_hlos[0]->shape().rank(); ++i) {
          if (absl::c_count(hlo->dimensions(), i) == 0) {
            tile_dimensions.push_back(
                inputs[0].sharding().tile_assignment().dim(i));
          }
        }
        Array<int64> new_tile = inputs[0].sharding().tile_assignment();
        new_tile.Reshape(tile_dimensions);
        auto sharding = HloSharding::Tile(new_tile);
        if (input_count > 1) {
          std::vector<HloSharding> tuple(input_count, sharding);
          sharding = HloSharding::Tuple(hlo->shape(), tuple);
        }
        reduce->set_sharding(sharding);
      }
    }

    return PartitionedHlo(reduce, hlo->shape(), MakePartitioningState())
        .Reshard(hlo->sharding())
        .hlo();
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleReverse(HloInstruction* hlo) {
  auto reverse = Cast<HloReverseInstruction>(hlo);
  if (reverse->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  if (absl::c_all_of(reverse->dimensions(), [&](int64 d) {
        return reverse->sharding().tile_assignment().dim(d) == 1;
      })) {
    auto operand =
        GetPartitionedHlo(reverse->operand(0)).Reshard(reverse->sharding());
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(
          hlo->CloneWithNewOperands(operand.hlo()->shape(), {operand.hlo()}));
    });
    return Status::OK();
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleWhile(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();

  // Shardings for the body parameter, body root, and cond parameter must be
  // the same, and the condition root must be replicated so that all partitions
  // follow the same control flow.
  hlo->while_condition()->parameter_instruction(0)->set_sharding(sharding);
  hlo->while_body()->parameter_instruction(0)->set_sharding(sharding);
  TF_RETURN_IF_ERROR(partitioner_
                         ->PartitionComputation(hlo->while_condition(),
                                                HloSharding::Replicate(),
                                                next_channel_id_, logger_)
                         .status());
  TF_RETURN_IF_ERROR(partitioner_
                         ->PartitionComputation(hlo->while_body(), sharding,
                                                next_channel_id_, logger_)
                         .status());
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(HloInstruction::CreateWhile(
        MakePartitionedShape(hlo->shape(), sharding), hlo->while_condition(),
        hlo->while_body(),
        GetPartitionedHlo(hlo->operand(0)).Reshard(sharding).hlo()));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleConditional(HloInstruction* hlo) {
  std::vector<HloInstruction*> branch_args;
  for (int64 i = 0; i < hlo->branch_count(); ++i) {
    HloComputation* computation = hlo->branch_computation(i);

    // Shardings of the branch computation parameter and its argument must be
    // the same.
    computation->parameter_instruction(0)->set_sharding(
        hlo->operand(i + 1)->sharding());
    branch_args.push_back(GetPartitionedHlo(hlo->operand(i + 1)).hlo());
  }

  // The root of the branch computations must follow the sharding of the
  // conditional instruction.
  for (int64 i = 0; i < hlo->branch_count(); ++i) {
    HloComputation* computation = hlo->branch_computation(i);
    TF_RETURN_IF_ERROR(partitioner_
                           ->PartitionComputation(computation, hlo->sharding(),
                                                  next_channel_id_, logger_)
                           .status());
  }

  // We replicate the predicate of the conditional (the first operand) so that
  // all partitions follow the same control flow.
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(HloInstruction::CreateConditional(
        MakePartitionedShape(hlo->shape(), hlo->sharding()),
        GetPartitionedHlo(hlo->operand(0))
            .Reshard(HloSharding::Replicate())
            .hlo(),
        hlo->called_computations(), branch_args));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleOutfeed(HloInstruction* hlo) {
  TF_RET_CHECK(hlo->sharding().HasUniqueDevice());
  return HandleSingleDevice(hlo);
}

Status SpmdPartitioningVisitor::HandleRng(HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }

  if (hlo->sharding().IsReplicated()) {
    SetPartitionedHlo(hlo, [&] {
      // Run on a single device (0) and distribute the data to all other cores.
      std::vector<HloInstruction*> new_operands;
      for (int64 i = 0; i < hlo->operand_count(); ++i) {
        new_operands.push_back(GetPartitionedHlo(hlo->operand(i))
                                   .Reshard(HloSharding::AssignDevice(0))
                                   .hlo());
      }
      auto clone = b_.AddInstruction(
          hlo->CloneWithNewOperands(hlo->shape(), new_operands));
      clone->set_sharding(HloSharding::AssignDevice(0));
      return PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
          .Reshard(HloSharding::Replicate())
          .hlo();
    });
    return Status::OK();
  }

  TF_RET_CHECK(!hlo->sharding().IsTileMaximal());
  SetPartitionedHlo(hlo, [&] {
    // Replicate the operands and run partitioned Rng on all devices.
    std::vector<HloInstruction*> new_operands;
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      new_operands.push_back(GetPartitionedHlo(hlo->operand(i))
                                 .Reshard(HloSharding::Replicate())
                                 .hlo());
    }
    return b_.AddInstruction(HloInstruction::CreateRng(
        MakePartitionedShape(hlo->shape(), hlo->sharding()),
        hlo->random_distribution(), new_operands));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleReduceWindow(HloInstruction* hlo) {
  auto& operand = GetPartitionedHlo(hlo->operand(0));
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  // Replicate init
  auto replicated_init = GetPartitionedHlo(hlo->mutable_operand(1))
                             .Reshard(HloSharding::Replicate());
  auto resharded_operand_and_window = operand.ReshardAsWindowedInput(
      hlo->window(), hlo->sharding(), replicated_init.hlo());
  if (!resharded_operand_and_window.has_value()) {
    return DefaultAction(hlo);
  }

  TF_ASSIGN_OR_RETURN(Shape sharded_rw_shape,
                      ShapeInference::InferReduceWindowShape(
                          resharded_operand_and_window->sharded_input->shape(),
                          replicated_init.hlo()->shape(),
                          resharded_operand_and_window->shard_window,
                          hlo->to_apply()->ComputeProgramShape()));
  auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
  *sharded_rw_shape.mutable_layout() = shard_shape.layout();
  SetPartitionedHlo(hlo, [&]() {
    auto sharded_rw = b_.AddInstruction(HloInstruction::CreateReduceWindow(
        sharded_rw_shape, resharded_operand_and_window->sharded_input,
        replicated_init.hlo(), resharded_operand_and_window->shard_window,
        hlo->to_apply()));
    if (!resharded_operand_and_window->dynamic_slice_index_on_output
             .has_value()) {
      CHECK(ShapeUtil::Compatible(shard_shape, sharded_rw->shape()));
      return sharded_rw;
    }
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, sharded_rw,
        *resharded_operand_and_window->dynamic_slice_index_on_output,
        shard_shape.dimensions()));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleSelectAndScatter(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  auto operand = GetPartitionedHlo(hlo->operand(0));
  auto source = GetPartitionedHlo(hlo->mutable_operand(1));
  if (hlo->sharding() != operand.sharding()) {
    operand = operand.Reshard(hlo->sharding());
  }
  if (hlo->sharding() != source.sharding()) {
    source = source.Reshard(hlo->sharding());
  }

  // For F32 and BF16 types, we can use NaN padding to workaround the issue with
  // low/high padding, since comparison will return false with NaN input.
  if (hlo->shape().element_type() != F32 &&
      hlo->shape().element_type() != BF16) {
    return DefaultAction(hlo);
  }

  auto select = hlo->called_computations()[0];
  auto select_root = select->root_instruction();
  if (select_root->opcode() != HloOpcode::kCompare ||
      select_root->operand(0)->opcode() != HloOpcode::kParameter ||
      select_root->operand(1)->opcode() != HloOpcode::kParameter ||
      select_root->operand(0)->parameter_number() +
              select_root->operand(1)->parameter_number() !=
          1) {
    return DefaultAction(hlo);
  }

  float float_pad_value;
  if (select_root->comparison_direction() == ComparisonDirection::kGe ||
      select_root->comparison_direction() == ComparisonDirection::kGt) {
    if (select_root->operand(0)->parameter_number() == 0) {
      float_pad_value = -std::numeric_limits<float>::infinity();
    } else {
      float_pad_value = std::numeric_limits<float>::infinity();
    }
  } else if (select_root->comparison_direction() == ComparisonDirection::kLe ||
             select_root->comparison_direction() == ComparisonDirection::kLt) {
    if (select_root->operand(0)->parameter_number() == 0) {
      float_pad_value = std::numeric_limits<float>::infinity();
    } else {
      float_pad_value = -std::numeric_limits<float>::infinity();
    }
  } else {
    return DefaultAction(hlo);
  }

  auto pad_value = b_.AddInstruction(HloInstruction::CreateConstant(
      hlo->shape().element_type() == BF16
          ? LiteralUtil::CreateR0<bfloat16>(
                static_cast<bfloat16>(float_pad_value))
          : LiteralUtil::CreateR0<float>(float_pad_value)));

  // Replicate init
  auto replicated_init = GetPartitionedHlo(hlo->mutable_operand(2))
                             .Reshard(HloSharding::Replicate());

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(hlo->sharding(), partition_id_, &b_);

  // The first window for each dimension that overlaps with the shard area.
  std::vector<MultiplyAddDivideOffsetCalculation> first_window(
      hlo->shape().rank());
  // The first window for each dimension that goes beyond with the shard area.
  std::vector<MultiplyAddDivideOffsetCalculation> limit_window(
      hlo->shape().rank());
  std::vector<OffsetCalculation> data_left_halo_sizes(hlo->shape().rank());
  std::vector<OffsetCalculation> data_right_halo_sizes(hlo->shape().rank());
  std::vector<OffsetCalculation> source_left_halo_sizes(hlo->shape().rank());
  std::vector<OffsetCalculation> source_right_halo_sizes(hlo->shape().rank());
  auto unpadded_data_shard_shape =
      MakePartitionedShape(hlo->shape(), hlo->sharding());
  auto unpadded_source_shard_shape =
      MakePartitionedShape(hlo->operand(1)->shape(), hlo->sharding());
  auto source_shard_hlo = source.hlo();
  auto data_shard_hlo = operand.hlo();
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    int64 shard_count = hlo->sharding().tile_assignment().dim(i);
    if (shard_count == 1) {
      continue;
    }
    // If stride > window_size, there will be gaps between windows. These gaps
    // will also exist in the output, so we keep them during halo exchange.
    //
    // TODO(yuanzx): This could introduce overhead if partitions start at
    // different offsets in a gap.
    auto wd = hlo->window().dimensions(i);
    if (wd.stride() > wd.size()) {
      wd.set_size(wd.stride());
    }
    // shard_size * i < stride * k - pad_low + window_size  =>
    //   k > (shard_size * i + pad_low - window_size) / stride  =>
    //   first_k == (shard_size * i + pad_low - window_size + stride) / stride
    first_window[i] = MultiplyAddDivideOffsetCalculation(
        unpadded_data_shard_shape.dimensions(i),
        wd.padding_low() - wd.size() + wd.stride(), wd.stride());
    // shard_size * (i + 1) <= stride * k - pad_low  =>
    //   k >= (shard_size * i + shard_size + pad_low) / stride  =>
    //   limit_k == (shard_size * i + shard_size + pad_low + stride - 1) /
    //     stride
    limit_window[i] = MultiplyAddDivideOffsetCalculation(
        unpadded_data_shard_shape.dimensions(i),
        unpadded_data_shard_shape.dimensions(i) + wd.padding_low() +
            wd.stride() - 1,
        wd.stride());
    source_left_halo_sizes[i] =
        MultiplyAddDivideOffsetCalculation(
            unpadded_source_shard_shape.dimensions(i), 0, 1) -
        first_window[i];
    source_right_halo_sizes[i] =
        limit_window[i] - MultiplyAddDivideOffsetCalculation(
                              unpadded_source_shard_shape.dimensions(i),
                              unpadded_source_shard_shape.dimensions(i), 1);
    data_left_halo_sizes[i] =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            unpadded_data_shard_shape.dimensions(i), wd.padding_low(), 1)) -
        OffsetCalculation(
            HloOpcode::kMultiply, first_window[i],
            MultiplyAddDivideOffsetCalculation(0, wd.stride(), 1));
    data_right_halo_sizes[i] =
        OffsetCalculation(
            HloOpcode::kMultiply, limit_window[i],
            MultiplyAddDivideOffsetCalculation(0, wd.stride(), 1)) -
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            unpadded_data_shard_shape.dimensions(i),
            unpadded_data_shard_shape.dimensions(i) + wd.stride() +
                wd.padding_low() - wd.size(),
            1));

    int64 max_windows =
        (limit_window[i] - first_window[i]).MaxInRange(0, shard_count);
    auto first_window_hlo =
        first_window[i].Calculate(partition_ordinals[i], &b_);
    // Padding on the source is filled with the init value so they do not change
    // the data on overlapping windows.
    auto resharded_source = ExchangeHaloAndGetValidData(
        source_shard_hlo, source.base_shape(), source_left_halo_sizes[i],
        source_right_halo_sizes[i], 0,
        limit_window[i].Calculate(shard_count - 1), max_windows, i,
        hlo->sharding(), first_window_hlo, replicated_init.hlo(),
        partition_ordinals[i], collective_ops_creator_, next_channel_id_, &b_);
    if (!resharded_source) {
      return DefaultAction(hlo);
    }
    source_shard_hlo = *resharded_source;

    auto offset_start_in_data =
        MultiplyAddDivideOffsetCalculation(wd.stride(), 0, 1)
            .Calculate(first_window_hlo, &b_);
    int64 padded_data_size =
        (limit_window[i].Calculate(shard_count - 1) - 1) * wd.stride() +
        wd.size();
    int64 data_shard_size = (max_windows - 1) * wd.stride() + wd.size();
    auto resharded_data = ExchangeHaloAndGetValidData(
        data_shard_hlo, operand.base_shape(), data_left_halo_sizes[i],
        data_right_halo_sizes[i], wd.padding_low(), padded_data_size,
        data_shard_size, i, hlo->sharding(), offset_start_in_data, pad_value,
        partition_ordinals[i], collective_ops_creator_, next_channel_id_, &b_);
    if (!resharded_data) {
      return DefaultAction(hlo);
    }
    data_shard_hlo = *resharded_data;
  }

  Window window_on_shard = hlo->window();
  for (int64 i = 0; i < window_on_shard.dimensions_size(); ++i) {
    int64 shard_count = hlo->sharding().tile_assignment().dim(i);
    if (shard_count == 1) {
      continue;
    }
    auto reshard_wd = window_on_shard.mutable_dimensions(i);
    // The shards are already explicitly padded.
    reshard_wd->set_padding_low(0);
    reshard_wd->set_padding_high(0);
  }

  auto sharded_select_and_scatter =
      b_.AddInstruction(HloInstruction::CreateSelectAndScatter(
          data_shard_hlo->shape(), data_shard_hlo, select, window_on_shard,
          source_shard_hlo, replicated_init.hlo(),
          hlo->called_computations()[1]));
  SetPartitionedHlo(hlo, [&]() {
    auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
    if (ShapeUtil::Compatible(sharded_select_and_scatter->shape(),
                              shard_shape)) {
      return sharded_select_and_scatter;
    }
    auto zero = b_.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
    std::vector<HloInstruction*> slice_offsets(shard_shape.rank(), zero);
    for (int64 i = 0; i < window_on_shard.dimensions_size(); ++i) {
      if (hlo->sharding().tile_assignment().dim(i) == 1) {
        continue;
      }
      int64 pad_low = hlo->window().dimensions(i).padding_low();
      auto left_halo_size =
          data_left_halo_sizes[i].Calculate(partition_ordinals[i], &b_);
      if (data_left_halo_sizes[i].Calculate(0) == pad_low) {
        slice_offsets[i] = left_halo_size;
      } else {
        auto is_shard0 = b_.AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), zero, partition_ordinals[i],
            ComparisonDirection::kEq));
        auto pad_low_hlo = b_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(pad_low)));
        slice_offsets[i] = b_.AddInstruction(HloInstruction::CreateTernary(
            zero->shape(), HloOpcode::kSelect, is_shard0, pad_low_hlo,
            left_halo_size));
      }
    }
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, sharded_select_and_scatter, slice_offsets,
        shard_shape.dimensions()));
  });
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleTuple(HloInstruction* hlo) {
  std::vector<HloInstruction*> new_operands;
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    new_operands.push_back(
        GetPartitionedHlo(hlo->operand(i))
            .Reshard(hlo->sharding().GetSubSharding(hlo->shape(), {i}))
            .hlo());
  }
  SetPartitionedHlo(hlo, [&]() {
    return b_.AddInstruction(HloInstruction::CreateTuple(new_operands));
  });
  return Status::OK();
}

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
  auto aligned_rhs_sharding =
      hlo_sharding_util::TransposeSharding(lhs.sharding(), rhs_to_lhs_indices);
  auto aligned_lhs_sharding =
      hlo_sharding_util::TransposeSharding(rhs.sharding(), lhs_to_rhs_indices);

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
    rhs = rhs.PadWithValue(zero);
  } else {
    if (unsupported_sharding(lhs.sharding(), aligned_rhs_sharding)) {
      return DefaultAction(hlo);
    }
    lhs = lhs.PadWithValue(zero);
    rhs = rhs.Reshard(aligned_rhs_sharding).PadWithValue(zero);
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

  Window window = hlo->window();
  std::vector<int64> shard_counts(dnums.input_spatial_dimensions_size());
  std::vector<int64> lhs_shard_sizes(dnums.input_spatial_dimensions_size());
  std::vector<int64> rhs_shard_sizes(dnums.input_spatial_dimensions_size());
  for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
    int64 lhs_dimension = dnums.input_spatial_dimensions(i);
    int64 rhs_dimension = dnums.kernel_spatial_dimensions(i);
    int64 shard_count = lhs.sharding().tile_assignment().dim(lhs_dimension);
    auto wd = window.dimensions(i);
    if (wd.base_dilation() != 1 || wd.window_reversal()) {
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

Status SpmdPartitioningVisitor::HandleDot(HloInstruction* hlo) {
  DotGeneralDimsMapping mapping;
  const auto& dnums = hlo->dot_dimension_numbers();
  int64 next_output_dim = 0;
  for (int64 i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
    mapping.batch_dims.emplace_back();
    mapping.batch_dims.back().lhs = dnums.lhs_batch_dimensions(i);
    mapping.batch_dims.back().rhs = dnums.rhs_batch_dimensions(i);
    mapping.batch_dims.back().output = next_output_dim++;
  }
  for (int64 i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
    mapping.contracting_dims.emplace_back();
    mapping.contracting_dims.back().lhs = dnums.lhs_contracting_dimensions(i);
    mapping.contracting_dims.back().rhs = dnums.rhs_contracting_dimensions(i);
    mapping.contracting_dims.back().output = -1;
  }
  for (int64 i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    mapping.lhs_non_contracting_dims.emplace_back();
    mapping.lhs_non_contracting_dims.back().lhs = i;
    mapping.lhs_non_contracting_dims.back().rhs = -1;
    mapping.lhs_non_contracting_dims.back().output = next_output_dim++;
  }
  for (int64 i = 0; i < hlo->operand(1)->shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    mapping.rhs_non_contracting_dims.emplace_back();
    mapping.rhs_non_contracting_dims.back().lhs = -1;
    mapping.rhs_non_contracting_dims.back().rhs = i;
    mapping.rhs_non_contracting_dims.back().output = next_output_dim++;
  }
  auto create_sharded_dot = [&](HloInstruction* l, HloInstruction* r,
                                SpmdBuilder* b) -> StatusOr<HloInstruction*> {
    TF_ASSIGN_OR_RETURN(
        auto sharded_dot_shape,
        ShapeInference::InferDotOpShape(l->shape(), r->shape(),
                                        hlo->dot_dimension_numbers()));
    return b->AddInstruction(HloInstruction::CreateDot(
        sharded_dot_shape, l, r, hlo->dot_dimension_numbers(),
        hlo->precision_config()));
  };
  return HandleDotHelper(hlo, mapping, create_sharded_dot);
}

Status SpmdPartitioningVisitor::HandleDotHelper(
    HloInstruction* hlo, const DotGeneralDimsMapping& dims_mapping,
    const std::function<StatusOr<HloInstruction*>(
        HloInstruction*, HloInstruction*, SpmdBuilder*)>& create_sharded_dot) {
  const HloSharding& lhs_sharding = hlo->operand(0)->sharding();
  const HloSharding& rhs_sharding = hlo->operand(1)->sharding();

  // Similar to hlo_sharding_util::TransposeSharding(), but allows
  // removing/adding non-partitioned dimensions.
  auto transpose_sharding =
      [&](const HloSharding& source, absl::Span<int64 const> src_to_tgt,
          absl::Span<int64 const> tgt_to_src) -> absl::optional<HloSharding> {
    if (source.IsTileMaximal()) {
      return source;
    }
    std::vector<int64> tgt_dims_skipping_new(tgt_to_src.size(), -1);
    int64 skipped_tgt_dims = 0;
    for (int64 i = 0; i < tgt_to_src.size(); ++i) {
      if (tgt_to_src[i] < 0) {
        skipped_tgt_dims++;
      } else {
        tgt_dims_skipping_new[i] = i - skipped_tgt_dims;
      }
    }
    int64 skipped_src_dims = absl::c_count(src_to_tgt, -1);
    std::vector<int64> perm(src_to_tgt.size());
    for (int64 i = 0; i < src_to_tgt.size(); ++i) {
      if (src_to_tgt[i] < 0) {
        if (source.tile_assignment().dim(i) > 1) {
          return absl::nullopt;
        }
        perm[src_to_tgt.size() - skipped_src_dims] = i;
        skipped_src_dims--;
      } else {
        perm[tgt_dims_skipping_new[src_to_tgt[i]]] = i;
      }
    }
    auto tgt_sharding = hlo_sharding_util::TransposeSharding(source, perm);
    if (skipped_tgt_dims == 0) {
      return tgt_sharding;
    }
    auto reshape_tiles = tgt_sharding.tile_assignment();
    std::vector<int64> tgt_tiles(tgt_to_src.size(), 1);
    for (int64 i = 0; i < tgt_tiles.size(); ++i) {
      if (tgt_to_src[i] >= 0) {
        tgt_tiles[i] = reshape_tiles.dim(tgt_dims_skipping_new[i]);
      }
    }
    reshape_tiles.Reshape(tgt_tiles);
    return HloSharding::Tile(reshape_tiles);
  };

  std::vector<int64> lhs_to_rhs_indices(hlo->operand(0)->shape().rank(), -1);
  std::vector<int64> lhs_to_output_indices(hlo->operand(0)->shape().rank(), -1);
  std::vector<int64> rhs_to_lhs_indices(hlo->operand(1)->shape().rank(), -1);
  std::vector<int64> rhs_to_output_indices(hlo->operand(1)->shape().rank(), -1);
  std::vector<int64> output_to_lhs_indices(hlo->shape().rank(), -1);
  std::vector<int64> output_to_rhs_indices(hlo->shape().rank(), -1);
  auto populate_indices_mapping =
      [&](const DotGeneralDimsMapping::DimsMapping& mapping) {
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
  auto lhs_sharding_transposed_to_match_rhs =
      transpose_sharding(lhs_sharding, lhs_to_rhs_indices, rhs_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_lhs =
      transpose_sharding(rhs_sharding, rhs_to_lhs_indices, lhs_to_rhs_indices);
  auto lhs_sharding_transposed_to_match_output = transpose_sharding(
      lhs_sharding, lhs_to_output_indices, output_to_lhs_indices);
  auto rhs_sharding_transposed_to_match_output = transpose_sharding(
      rhs_sharding, rhs_to_output_indices, output_to_rhs_indices);
  auto output_sharding_transposed_to_match_lhs = transpose_sharding(
      hlo->sharding(), output_to_lhs_indices, lhs_to_output_indices);
  auto output_sharding_transposed_to_match_rhs = transpose_sharding(
      hlo->sharding(), output_to_rhs_indices, rhs_to_output_indices);

  // lhs_rhs_or_output: 0 lhs, 1 rhs, 2 output.
  auto get_partitions_for_dims =
      [&](const HloSharding& sharding,
          absl::Span<const DotGeneralDimsMapping::DimsMapping> dims,
          int lhs_rhs_or_output) {
        int64 partitions = 1;
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
  const int64 lhs_batch_partitions =
      get_partitions_for_dims(lhs_sharding, dims_mapping.batch_dims, 0);
  const int64 rhs_batch_partitions =
      get_partitions_for_dims(rhs_sharding, dims_mapping.batch_dims, 1);
  const int64 output_batch_partitions =
      get_partitions_for_dims(hlo->sharding(), dims_mapping.batch_dims, 2);
  const int64 lhs_contracting_partitions =
      get_partitions_for_dims(lhs_sharding, dims_mapping.contracting_dims, 0);
  const int64 rhs_contracting_partitions =
      get_partitions_for_dims(rhs_sharding, dims_mapping.contracting_dims, 1);
  const int64 lhs_non_contracting_partitions = get_partitions_for_dims(
      lhs_sharding, dims_mapping.lhs_non_contracting_dims, 0);
  const int64 rhs_non_contracting_partitions = get_partitions_for_dims(
      rhs_sharding, dims_mapping.rhs_non_contracting_dims, 1);
  const int64 output_lhs_non_contracting_partitions = get_partitions_for_dims(
      hlo->sharding(), dims_mapping.lhs_non_contracting_dims, 2);
  const int64 output_rhs_non_contracting_partitions = get_partitions_for_dims(
      hlo->sharding(), dims_mapping.rhs_non_contracting_dims, 2);

  auto& lhs = GetPartitionedHlo(hlo->operand(0));
  auto& rhs = GetPartitionedHlo(hlo->operand(1));
  // LHS and RHS are partitioned the same way and only partitioned in batch
  // dimensions.
  if (lhs_batch_partitions == rhs_batch_partitions &&
      rhs_batch_partitions == num_partitions_ &&
      lhs_sharding_transposed_to_match_rhs == rhs_sharding) {
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] {
      dot->set_sharding(*lhs_sharding_transposed_to_match_output);
      return PartitionedHlo(dot, hlo->shape(), MakePartitioningState())
          .Reshard(hlo->sharding())
          .hlo();
    });
    return Status::OK();
  }

  // Try emit batch-partitioned einsum with one operand resharded. Returns
  // whether the attempt succeeds. If may_reshard_with_allreduce is false,
  // reshard must be done using all-to-all; otherwise this attempt fails.
  auto try_emit_output_batch_partitioned_einsum_with_reshard =
      [&](bool may_reshard_with_allreduce) -> StatusOr<bool> {
    // LHS and output are batch partitioned in the same way.
    if (lhs_batch_partitions == num_partitions_ &&
        output_batch_partitions == num_partitions_ &&
        lhs_sharding_transposed_to_match_output == hlo->sharding()) {
      if (!may_reshard_with_allreduce &&
          !CanReshardWithAllToAll(rhs.sharding(),
                                  *lhs_sharding_transposed_to_match_rhs)) {
        return false;
      }
      auto resharded_rhs = rhs.Reshard(*lhs_sharding_transposed_to_match_rhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(lhs.hlo(), resharded_rhs.hlo(), &b_));
      SetPartitionedHlo(hlo, [&] { return dot; });
      return true;
    }
    // RHS and output are batch partitioned in the same way.
    if (rhs_batch_partitions == num_partitions_ &&
        output_batch_partitions == num_partitions_ &&
        rhs_sharding_transposed_to_match_output == hlo->sharding()) {
      if (!may_reshard_with_allreduce &&
          !CanReshardWithAllToAll(lhs.sharding(),
                                  *rhs_sharding_transposed_to_match_lhs)) {
        return false;
      }
      auto resharded_lhs = lhs.Reshard(*rhs_sharding_transposed_to_match_lhs);
      TF_ASSIGN_OR_RETURN(
          auto dot, create_sharded_dot(resharded_lhs.hlo(), rhs.hlo(), &b_));
      SetPartitionedHlo(hlo, [&] { return dot; });
      return true;
    }
    return false;
  };

  {
    // Try batch-parallel by resharding one operand, and not using all-reduce.
    TF_ASSIGN_OR_RETURN(
        bool emitted,
        try_emit_output_batch_partitioned_einsum_with_reshard(false));
    if (emitted) {
      return Status::OK();
    }
  }

  // Try to emit windowed DotGeneral when one operand is partitioned in the same
  // way as the output along non-contracting dimensions, but the other operand
  // is tiled in other dimensions.
  auto emit_windowed_dot_general = [&](int64 matching_operand,
                                       int64 windowing_operand,
                                       bool windowed_at_contracting_dims,
                                       bool windowed_at_batch_dims) {
    CHECK_EQ(matching_operand + windowing_operand, 1);
    CHECK(!windowed_at_batch_dims || !windowed_at_contracting_dims);
    auto unpadded_result_buffer_shape =
        MakePartitionedShape(hlo->shape(), hlo->sharding());
    auto padded_result_buffer_shape = unpadded_result_buffer_shape;
    // For windowing at batch/non-contracting dims, we produce the result one
    // partition at a time, so we need to pad the shape in case of uneven
    // partitioning in order to make dynamic-update-slice in-bound.
    if (!windowed_at_contracting_dims) {
      padded_result_buffer_shape = GetPaddedShapeForUnevenPartitioning(
          padded_result_buffer_shape,
          windowing_operand == 0 ? *lhs_sharding_transposed_to_match_output
                                 : *rhs_sharding_transposed_to_match_output);
    }
    // Mask the padding area of the windowed operand with zero if there is
    // uneven partitioning.
    if (windowed_at_contracting_dims) {
      auto& to_mask = windowing_operand == 0 ? lhs : rhs;
      to_mask =
          to_mask.PadWithValue(b_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(hlo->shape().element_type()))));
    }
    auto result_buffer = CreateZero(padded_result_buffer_shape, &b_);
    auto iteration = b_.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(0)));

    // Create a while loop that computes one window per iteration. During each
    // iteration, each partition sends its input window to its neighbor using
    // collective-permute for the next iteration.
    SpmdBuilder body_b("windowed_dot_general_body", visiting_hlo_);
    auto param = body_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape({lhs.hlo()->shape(), rhs.hlo()->shape(),
                                   result_buffer->shape(), iteration->shape()}),
        "param"));
    auto l = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(lhs.hlo()->shape(), param, 0));
    auto r = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(rhs.hlo()->shape(), param, 1));
    auto o = body_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        result_buffer->shape(), param, 2));
    auto i = body_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(iteration->shape(), param, 3));

    auto partition_id = collective_ops_creator_.create_partition_id(&body_b);
    auto data_partition_id = body_b.AddInstruction(HloInstruction::CreateBinary(
        i->shape(), HloOpcode::kAdd, i, partition_id));
    auto partition_count = body_b.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<uint32>(num_partitions_)));
    data_partition_id = body_b.AddInstruction(HloInstruction::CreateBinary(
        i->shape(), HloOpcode::kRemainder, data_partition_id, partition_count));
    auto dot_lhs = l;
    auto dot_rhs = r;
    if (windowed_at_contracting_dims || windowed_at_batch_dims) {
      // Slice the matching operand according to the partitioned contracting
      // dimensions on the windowed operand. We do this by treating the matching
      // operand as replicated, and resharding it to match the windowed operand.
      auto slice_operand = matching_operand == 0 ? l : r;
      slice_operand->set_sharding(HloSharding::Replicate());
      auto state = MakePartitioningState();
      state.b = &body_b;
      state.partition_id = data_partition_id;
      auto slice = PartitionedHlo(slice_operand, slice_operand->shape(), state)
                       .Reshard(windowing_operand == 0
                                    ? *lhs_sharding_transposed_to_match_rhs
                                    : *rhs_sharding_transposed_to_match_lhs)
                       .hlo();
      slice_operand->clear_sharding();
      if (matching_operand == 0) {
        dot_lhs = slice;
      } else {
        dot_rhs = slice;
      }
    }
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(dot_lhs, dot_rhs, &body_b));
    if (windowed_at_contracting_dims) {
      // Accumulate the partial output to the result buffer.
      o = body_b.AddInstruction(
          HloInstruction::CreateBinary(o->shape(), HloOpcode::kAdd, o, dot));
    } else {
      // The windowing operand is partitioned along batch/non-contracting
      // dimensions, so we need a dynamic-update-slice to save the partial
      // output in the result buffer.
      auto offsets = MakePartitionOffsets(
          o->shape(),
          windowing_operand == 0 ? *lhs_sharding_transposed_to_match_output
                                 : *rhs_sharding_transposed_to_match_output,
          data_partition_id, &body_b);
      o = body_b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          o->shape(), o, dot, offsets));
    }

    // ++i
    i = body_b.AddInstruction(HloInstruction::CreateBinary(
        i->shape(), HloOpcode::kAdd, i,
        body_b.AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32>(1)))));
    auto has_more = body_b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), i,
        body_b.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32>(num_partitions_))),
        ComparisonDirection::kLt));
    // Collective-permute for the next window. We don't need it for the last
    // iteration, so we use a conditional around the collective-permute.
    HloInstruction* conditional;
    {
      SpmdBuilder cp_b("window_collective_permute", visiting_hlo_);
      {
        auto p = cp_b.AddInstruction(HloInstruction::CreateParameter(
            0, windowing_operand == 0 ? l->shape() : r->shape(), "window"));
        std::vector<std::pair<int64, int64>> sd_pairs(num_partitions_);
        for (int64 source = 0; source < num_partitions_; ++source) {
          // 0 -> n-1, 1 -> 0, 2 -> 1, ...
          sd_pairs[source] = {source,
                              (source - 1 + num_partitions_) % num_partitions_};
        }
        collective_ops_creator_.create_cross_partition_collective_permute(
            &cp_b, p, sd_pairs, (*next_channel_id_)++);
      }
      SpmdBuilder ncp_b("last_iteration_noop", visiting_hlo_);
      {
        ncp_b.AddInstruction(HloInstruction::CreateParameter(
            0, windowing_operand == 0 ? l->shape() : r->shape(), "window"));
      }
      conditional = body_b.AddInstruction(HloInstruction::CreateConditional(
          windowing_operand == 0 ? l->shape() : r->shape(), has_more,
          windowing_operand == 0 ? l : r,
          module_->AddEmbeddedComputation(cp_b.Build()),
          windowing_operand == 0 ? l : r,
          module_->AddEmbeddedComputation(ncp_b.Build())));
    }
    if (windowing_operand == 0) {
      l = conditional;
    } else {
      r = conditional;
    }
    body_b.AddInstruction(HloInstruction::CreateTuple({l, r, o, i}));

    SpmdBuilder cond_b("windowed_dot_general_cond", visiting_hlo_);
    auto cond_param = cond_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape({lhs.hlo()->shape(), rhs.hlo()->shape(),
                                   result_buffer->shape(), iteration->shape()}),
        "param"));
    auto cond_i = cond_b.AddInstruction(HloInstruction::CreateGetTupleElement(
        iteration->shape(), cond_param, 3));
    cond_b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), cond_i,
        cond_b.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32>(num_partitions_))),
        ComparisonDirection::kLt));
    auto while_loop = b_.AddInstruction(HloInstruction::CreateWhile(
        cond_param->shape(), module_->AddEmbeddedComputation(cond_b.Build()),
        module_->AddEmbeddedComputation(body_b.Build()),
        b_.AddInstruction(HloInstruction::CreateTuple(
            {lhs.hlo(), rhs.hlo(), result_buffer, iteration}))));
    windowed_dot_general_loops_.push_back({while_loop, windowing_operand,
                                           windowed_at_contracting_dims,
                                           windowed_at_batch_dims});
    SetPartitionedHlo(hlo, [&] {
      auto result = b_.AddInstruction(HloInstruction::CreateGetTupleElement(
          result_buffer->shape(), while_loop, 2));
      if (!ShapeUtil::Compatible(padded_result_buffer_shape,
                                 unpadded_result_buffer_shape)) {
        result = b_.AddInstruction(HloInstruction::CreateSlice(
            unpadded_result_buffer_shape, result,
            std::vector<int64>(padded_result_buffer_shape.rank(), 0),
            unpadded_result_buffer_shape.dimensions(),
            std::vector<int64>(padded_result_buffer_shape.rank(), 1)));
      }
      return result;
    });
    return Status::OK();
  };
  if (output_lhs_non_contracting_partitions == num_partitions_ &&
      output_sharding_transposed_to_match_lhs == lhs_sharding &&
      ShapeSizeInBytes(hlo->operand(1)->shape()) >=
          options_.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (rhs_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(0, 1, true, false);
    }
    if (rhs_non_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(0, 1, false, false);
    }
    if (rhs_batch_partitions == num_partitions_) {
      return emit_windowed_dot_general(0, 1, false, true);
    }
  }
  if (output_rhs_non_contracting_partitions == num_partitions_ &&
      output_sharding_transposed_to_match_rhs == rhs_sharding &&
      ShapeSizeInBytes(hlo->operand(0)->shape()) >=
          options_.threshold_for_windowed_einsum_mib * 1024 * 1024) {
    if (lhs_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(1, 0, true, false);
    }
    if (lhs_non_contracting_partitions == num_partitions_) {
      return emit_windowed_dot_general(1, 0, false, false);
    }
    if (lhs_batch_partitions == num_partitions_) {
      return emit_windowed_dot_general(1, 0, false, true);
    }
  }

  {
    // Try batch-parallel by resharding one operand, and allowing all-reduce.
    TF_ASSIGN_OR_RETURN(
        bool emitted,
        try_emit_output_batch_partitioned_einsum_with_reshard(true));
    if (emitted) {
      return Status::OK();
    }
  }

  // LHS and RHS have the same partitioned contracting dimensions.
  if (lhs_contracting_partitions == rhs_contracting_partitions &&
      lhs_contracting_partitions == num_partitions_) {
    auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(hlo->shape().element_type())));
    // Pad both sides with zero, since NaN at one side cannot be masked by zero
    // on the other side.
    if (ShapeSizeInBytes(lhs.base_shape()) <
        ShapeSizeInBytes(rhs.base_shape())) {
      lhs =
          lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithValue(zero);
      rhs = rhs.PadWithValue(zero);
    } else {
      lhs = lhs.PadWithValue(zero);
      rhs =
          rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithValue(zero);
    }
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] {
      auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
          &b_, dot, MakeBinaryAdd(hlo->shape().element_type(), module_),
          NewChannel());
      ar->set_sharding(HloSharding::Replicate());
      return PartitionedHlo(ar, hlo->shape(), MakePartitioningState())
          .Reshard(hlo->sharding())
          .hlo();
    });
    return Status::OK();
  }

  // LHS and output have the same partitioned non-contracting dimensions.
  if (lhs_non_contracting_partitions == num_partitions_ &&
      output_lhs_non_contracting_partitions == num_partitions_ &&
      lhs_sharding == hlo->sharding()) {
    auto rhs_replicated = rhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs_replicated, &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }

  // RHS and output have the same partitioned non-contracting dimensions.
  if (rhs_non_contracting_partitions == num_partitions_ &&
      output_rhs_non_contracting_partitions == num_partitions_ &&
      rhs_sharding_transposed_to_match_output == hlo->sharding()) {
    auto lhs_replicated = lhs.Reshard(HloSharding::Replicate()).hlo();
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs_replicated, rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }

  // Output is batch partitioned.
  if (output_batch_partitions == num_partitions_) {
    auto resharded_lhs = lhs.Reshard(*output_sharding_transposed_to_match_lhs);
    auto resharded_rhs = rhs.Reshard(*output_sharding_transposed_to_match_rhs);
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(resharded_lhs.hlo(),
                                                     resharded_rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }
  // Output is partitioned along LHS non-contracting dimensions.
  if (output_lhs_non_contracting_partitions == num_partitions_) {
    auto resharded_lhs = lhs.Reshard(*output_sharding_transposed_to_match_lhs);
    auto replicated_rhs = rhs.Reshard(HloSharding::Replicate());
    TF_ASSIGN_OR_RETURN(
        auto dot,
        create_sharded_dot(resharded_lhs.hlo(), replicated_rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }
  // Output is partitioned along RHS non-contracting dimensions.
  if (output_rhs_non_contracting_partitions == num_partitions_) {
    auto replicated_lhs = lhs.Reshard(HloSharding::Replicate());
    auto resharded_rhs = rhs.Reshard(*output_sharding_transposed_to_match_rhs);
    TF_ASSIGN_OR_RETURN(auto dot, create_sharded_dot(replicated_lhs.hlo(),
                                                     resharded_rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] { return dot; });
    return Status::OK();
  }

  // Returns true if it is beneficial to reshard the operand at `operand_idx`
  // across the contracting dimension.
  const auto should_partition_contracting_dim = [&](int64 operand_idx) {
    if (!hlo->sharding().IsReplicated()) {
      return false;
    }

    if (operand_idx == 0) {
      // If LHS and output are replicated, we compare the cost of all-gather
      // on RHS vs all-reduce on the output.
      return (rhs_contracting_partitions == num_partitions_) &&
             lhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(hlo->operand(1)->shape()) >
                 ShapeUtil::ElementsIn(hlo->shape());
    } else {
      return (lhs_contracting_partitions == num_partitions_) &&
             rhs.sharding().IsReplicated() &&
             ShapeUtil::ElementsIn(hlo->operand(0)->shape()) >
                 ShapeUtil::ElementsIn(hlo->shape());
    }
  };

  // When the output is replicated and one of the operands is partitioned along
  // contracting dimension, align the other operand to be partitioned along
  // the contracting dimensions.
  if (hlo->sharding().IsReplicated() && (should_partition_contracting_dim(0) ||
                                         should_partition_contracting_dim(1))) {
    auto zero = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(hlo->shape().element_type())));
    if (should_partition_contracting_dim(0)) {
      lhs =
          lhs.Reshard(*rhs_sharding_transposed_to_match_lhs).PadWithValue(zero);
      rhs = rhs.PadWithValue(zero);
    } else {
      lhs = lhs.PadWithValue(zero);
      rhs =
          rhs.Reshard(*lhs_sharding_transposed_to_match_rhs).PadWithValue(zero);
    }
    TF_ASSIGN_OR_RETURN(auto dot,
                        create_sharded_dot(lhs.hlo(), rhs.hlo(), &b_));
    SetPartitionedHlo(hlo, [&] {
      auto ar = collective_ops_creator_.create_cross_partition_all_reduce(
          &b_, dot, MakeBinaryAdd(hlo->shape().element_type(), module_),
          NewChannel());
      ar->set_sharding(HloSharding::Replicate());
      return PartitionedHlo(ar, hlo->shape(), MakePartitioningState()).hlo();
    });
    return Status::OK();
  }

  return DefaultAction(hlo);
}

namespace {

// Finds a cluster of nodes that produce the inputs for `hlo` which only depend
// on small operands, which means the cluster should start with broadcasts,
// constants and iotas. All other internal nodes must be non-side-effecting
// elemntwise ops. Returns the set of nodes, and the small operands. E.g., for
// the following graph,
//
//     a -> broadcast -> multiply
//     iota  ---> add--/
//     constant/
//
// FindInputNodesIfOnlyDependOnSmallOperands(multiply) will return
//    <{broadcast, iota, constant, add, multiply}, [a]>.
std::pair<std::unordered_set<HloInstruction*>, std::vector<HloInstruction*>>
FindInputNodesIfOnlyDependOnSmallOperands(HloInstruction* hlo) {
  std::unordered_set<HloInstruction*> nodes_found;
  std::vector<HloInstruction*> new_operands;
  std::unordered_set<const HloInstruction*> new_operands_set;
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
               inst->opcode() != HloOpcode::kAllReduce &&
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
    HloInstruction* loop, int64 non_windowed_operand_index) {
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
  std::unordered_map<const HloInstruction*, HloInstruction*> outside_to_inside;
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
  for (int64 i = 0; i < new_operands.size(); ++i) {
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
  for (auto inst : nullaries_to_sink) {
    worklist.push_back(inst);
  }
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    std::vector<HloInstruction*> inst_new_operands(inst->operand_count());
    for (int64 i = 0; i < inst->operand_count(); ++i) {
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

// Moves a cluster of memory-reducing nodes (with reduce nodes at the end) into
// the windowed dot-general loop on non-contracting dimensions. Such a loop has
// a dynamic-update-slice at the output. If we move the user nodes into the loop
// and before the dynamic-update-slice, the user nodes can operate on smaller
// shapes, which reduces memory.
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
    HloInstruction* loop) {
  CHECK_EQ(loop->user_count(), 1);
  // There should be a single direct user of the while loop, which is the
  // gte for element 2, i.e., the dot output.
  auto user_gte = loop->users().front();
  CHECK_EQ(user_gte->opcode(), HloOpcode::kGetTupleElement);
  CHECK_EQ(user_gte->tuple_index(), 2);
  auto computation = loop->parent();

  // Find the reduce outputs and the input nodes they depend on, if input nodes
  // only have small operands.
  std::unordered_set<HloInstruction*> to_move;
  std::vector<HloInstruction*> new_operands;
  std::unordered_set<const HloInstruction*> new_operands_set;
  std::vector<HloInstruction*> reduce_outputs;
  std::vector<HloInstruction*> worklist;
  Shape padded_shape = user_gte->shape();
  Shape unpadded_shape = user_gte->shape();
  auto original_output = user_gte;

  if (user_gte->user_count() == 1 &&
      user_gte->users().back()->opcode() == HloOpcode::kSlice) {
    original_output = user_gte->users().back();
    unpadded_shape = original_output->shape();
  }
  for (auto u : original_output->users()) {
    worklist.push_back(u);
  }
  to_move.insert(original_output);
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    if (to_move.count(inst) > 0) {
      continue;
    }
    // We only support reduces with simple reduction function, since we may need
    // to accumulate across iterations manually.
    if (inst->opcode() == HloOpcode::kReduce &&
        inst->to_apply()->instruction_count() == 3 &&
        inst->to_apply()->num_parameters() == 2 &&
        inst->to_apply()->root_instruction()->IsElementwise()) {
      to_move.insert(inst);
      auto other_operand = inst->mutable_operand(1);
      auto res = new_operands_set.emplace(other_operand);
      if (res.second) {
        new_operands.push_back(other_operand);
      }
      reduce_outputs.push_back(inst);
    } else if (inst != computation->root_instruction() &&
               inst->user_count() > 0 && inst->IsElementwise() &&
               !inst->HasSideEffectNoRecurse() &&
               inst->opcode() != HloOpcode::kAllReduce &&
               absl::c_all_of(inst->operands(),
                              [inst](const HloInstruction* o) {
                                return ShapeUtil::CompatibleIgnoringElementType(
                                    o->shape(), inst->shape());
                              })) {
      // For an elementwise op, we need to make sure that they depend on only
      // nodes already in to_move and nodes with small operands.
      bool can_include = true;
      for (auto operand : inst->operands()) {
        if (to_move.count(operand) > 0) {
          continue;
        }
        auto find_result = FindInputNodesIfOnlyDependOnSmallOperands(operand);
        if (find_result.first.empty()) {
          can_include = false;
          break;
        }
        for (auto n : find_result.first) {
          to_move.insert(n);
        }
        for (auto new_operand : find_result.second) {
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
      for (auto u : inst->users()) {
        worklist.push_back(u);
      }
    } else {
      to_move.clear();
      break;
    }
  }
  // If nothing is found, to_move could contain only original_output, or cleared
  // by the above code.
  if (to_move.size() <= 1) {
    return Status::OK();
  }

  // We will replace the original loop output with reduce-shape outputs. Create
  // the initial buffers before the loop.
  for (auto out : reduce_outputs) {
    auto padded_out_shape = out->shape();
    int64 operand_dim = 0;
    int64 output_dim = 0;
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
    auto broadcast =
        computation->AddInstruction(HloInstruction::CreateBroadcast(
            padded_out_shape,
            computation->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(out->shape().element_type()))),
            {}));
    new_operands.push_back(broadcast);
  }

  auto input_tuple = loop->mutable_operand(0);
  // Create the new input subtuple that contains the small operands and the
  // reduce-shape result buffers.
  auto new_input_subtuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  TF_RETURN_IF_ERROR(
      input_tuple->ReplaceOperandWithDifferentShape(2, new_input_subtuple));
  auto body = loop->while_body();
  auto body_param = body->parameter_instruction(0);
  auto body_root = body->root_instruction();
  CHECK_EQ(body_root->opcode(), HloOpcode::kTuple);
  // Update tuple shapes.
  for (auto tuple : std::vector<HloInstruction*>{
           input_tuple, loop, loop->while_condition()->parameter_instruction(0),
           body_param, body_root}) {
    *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(), {2}) =
        new_input_subtuple->shape();
  }
  auto new_loop_input =
      body->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_input_subtuple->shape(), body_param, 2));

  // Now create the moved nodes inside the loop body.
  std::unordered_map<const HloInstruction*, HloInstruction*> outside_to_inside;
  worklist.clear();
  auto add_users_if_available = [&](HloInstruction* inst) {
    for (auto u : inst->users()) {
      if (outside_to_inside.count(u) == 0 && to_move.count(u) > 0 &&
          absl::c_all_of(u->operands(), [&](const HloInstruction* o) {
            return outside_to_inside.count(o) > 0;
          })) {
        worklist.push_back(u);
      }
    }
  };
  for (int64 i = 0; i < new_operands.size(); ++i) {
    outside_to_inside[new_operands[i]] =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_operands[i]->shape(), new_loop_input, i));
    add_users_if_available(new_operands[i]);
  }
  // The elementwise nodes will be created with sliced shape. The original loop
  // output corresponds to the dynamic-update-slice's update slice.
  auto dus = body_root->mutable_operand(2);
  CHECK_EQ(dus->opcode(), HloOpcode::kDynamicUpdateSlice);
  outside_to_inside[original_output] = dus->mutable_operand(1);
  add_users_if_available(original_output);
  std::vector<HloInstruction*> slice_offsets(padded_shape.rank());
  for (int64 i = 0; i < slice_offsets.size(); ++i) {
    slice_offsets[i] = dus->mutable_operand(i + 2);
  }
  auto get_slice = [&](HloInstruction* padded) {
    return body->AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::ChangeElementType(dus->operand(1)->shape(),
                                     padded->shape().element_type()),
        padded, slice_offsets, dus->operand(1)->shape().dimensions()));
  };
  // Helper functions to create nodes with small operands.
  auto add_broadcast = [&](const HloInstruction* broadcast) {
    auto padded_operand_shape = broadcast->operand(0)->shape();
    for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
      padded_operand_shape.set_dimensions(
          i, padded_shape.dimensions(broadcast->dimensions(i)));
    }
    auto padded_operand = PadToShape(outside_to_inside[broadcast->operand(0)],
                                     padded_operand_shape, nullptr, body);
    outside_to_inside[broadcast] =
        get_slice(body->AddInstruction(broadcast->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(padded_shape,
                                         padded_operand_shape.element_type()),
            {padded_operand})));
  };
  auto add_iota = [&](const HloInstruction* iota) {
    outside_to_inside[iota] =
        get_slice(body->AddInstruction(iota->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(padded_shape,
                                         iota->shape().element_type()),
            {})));
  };
  auto add_constant = [&](const HloInstruction* constant) {
    outside_to_inside[constant] = body->AddInstruction(constant->Clone());
    outside_to_inside[constant] = get_slice(
        PadToShape(outside_to_inside[constant],
                   ShapeUtil::ChangeElementType(
                       padded_shape, constant->shape().element_type()),
                   nullptr, body));
  };
  while (!worklist.empty()) {
    auto inst = worklist.back();
    worklist.pop_back();
    if (outside_to_inside.count(inst) > 0) {
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
    } else {
      std::vector<HloInstruction*> operands_inside(inst->operand_count());
      for (int64 i = 0; i < operands_inside.size(); ++i) {
        operands_inside[i] = outside_to_inside[inst->operand(i)];
      }
      outside_to_inside[inst] = body->AddInstruction(inst->CloneWithNewOperands(
          ShapeUtil::ChangeElementType(dus->operand(1)->shape(),
                                       inst->shape().element_type()),
          operands_inside));
    }
    add_users_if_available(inst);
  }
  std::vector<HloInstruction*> new_outputs_inside(new_operands.size());
  for (int64 i = 0; i < new_outputs_inside.size(); ++i) {
    new_outputs_inside[i] = outside_to_inside[new_operands[i]];
  }
  // Now create the reduce outpus inside of the loop.
  for (int64 i = 0; i < reduce_outputs.size(); ++i) {
    auto reduce_outside = reduce_outputs[i];
    CHECK_EQ(reduce_outside->opcode(), HloOpcode::kReduce);
    int64 index_in_operand = new_operands.size() - reduce_outputs.size() + i;
    auto last_iter_result = outside_to_inside[new_operands[index_in_operand]];
    auto operand0 = outside_to_inside[reduce_outside->operand(0)];
    auto operand1 = outside_to_inside[reduce_outside->operand(1)];
    TF_ASSIGN_OR_RETURN(auto reduce_shape,
                        ShapeInference::InferReduceShape(
                            {&operand0->shape(), &operand1->shape()},
                            reduce_outside->dimensions(),
                            reduce_outside->to_apply()->ComputeProgramShape()));
    *reduce_shape.mutable_layout() = reduce_outside->shape().layout();
    std::vector<HloInstruction*> reduce_dus_offsets;
    // If any collapsed dimension is windowed, we need to accumulate with last
    // iteration's result. If such a dimension has padding, we also need to mask
    // off invalid data.
    bool needs_accumulate = false;
    std::vector<int64> dims_to_mask;
    for (int64 i = 0; i < slice_offsets.size(); ++i) {
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
    for (int64 dim : dims_to_mask) {
      auto iota = body->AddInstruction(HloInstruction::CreateIota(
          ShapeUtil::ChangeElementType(operand0->shape(), S32), dim));
      auto add = body->AddInstruction(HloInstruction::CreateBinary(
          iota->shape(), HloOpcode::kAdd, iota,
          body->AddInstruction(HloInstruction::CreateBroadcast(
              iota->shape(), slice_offsets[dim], {}))));
      auto limit = body->AddInstruction(HloInstruction::CreateBroadcast(
          iota->shape(),
          body->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
                  reduce_outside->operand(0)->shape().dimensions(dim)))),
          {}));
      auto compare = body->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(iota->shape(), PRED), add, limit,
          ComparisonDirection::kLt));
      operand0 = body->AddInstruction(HloInstruction::CreateTernary(
          operand0->shape(), HloOpcode::kSelect, compare, operand0,
          body->AddInstruction(HloInstruction::CreateBroadcast(
              operand0->shape(), operand1, {}))));
    }
    auto output_inside =
        body->AddInstruction(reduce_outside->CloneWithNewOperands(
            reduce_shape, {operand0, operand1}));
    // Accumulate with previous results if needed.
    if (needs_accumulate) {
      auto input_slice =
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
    new_outputs_inside[index_in_operand] = output_inside;
  }
  // Body output.
  auto new_output_inside =
      body->AddInstruction(HloInstruction::CreateTuple(new_outputs_inside));
  TF_RETURN_IF_ERROR(
      body_root->ReplaceOperandWithDifferentShape(2, new_output_inside));
  TF_RETURN_IF_ERROR(body->RemoveInstructionAndUnusedOperands(dus));
  // Replace uses of the reduces outside the loop.
  auto new_output_gte =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_output_inside->shape(), loop, 2));
  for (int64 i = 0; i < reduce_outputs.size(); ++i) {
    int64 index_in_operand = new_operands.size() - reduce_outputs.size() + i;
    auto new_output =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_outputs_inside[index_in_operand]->shape(), new_output_gte,
            index_in_operand));
    if (!ShapeUtil::Compatible(new_output->shape(),
                               reduce_outputs[i]->shape())) {
      new_output = computation->AddInstruction(HloInstruction::CreateSlice(
          reduce_outputs[i]->shape(), new_output,
          std::vector<int64>(new_output->shape().rank(), 0),
          reduce_outputs[i]->shape().dimensions(),
          std::vector<int64>(new_output->shape().rank(), 1)));
    }
    TF_RETURN_IF_ERROR(reduce_outputs[i]->ReplaceAllUsesWith(new_output));
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(reduce_outputs[i]));
  }
  return Status::OK();
}

}  // namespace

Status SpmdPartitioningVisitor::DoCodeMotionForWindowedDotGeneralLoops(
    HloComputation* computation) {
  for (auto& loop : windowed_dot_general_loops_) {
    if (loop.windowed_in_contracting_dims || loop.windowed_in_batch_dims) {
      // We have a dynamic-slice for the non-windowed operand in
      // batch/contracting-dim windowed dot-general. So moving the
      // broadcast/iota/elementwise ops into the loop could help reduce memory
      // via fusion.
      TF_RETURN_IF_ERROR(
          SinkInputNodesIntoWindowedDotGeneralLoopOnContractingDimensions(
              loop.while_loop, 1 - loop.windowed_operand));
    }
    if (!loop.windowed_in_contracting_dims) {
      // We have a dynamic-update-slice for the output in
      // batch/non-contracting-dim windowed dot-general. So moving reduce ops
      // into the loop could help reduce memory.
      TF_RETURN_IF_ERROR(
          MoveUsersIntoWindowedDotGeneralLoopOnNonContractingDimensions(
              loop.while_loop));
    }
  }
  return Status::OK();
}

StatusOr<bool> SpmdPartitioningVisitor::DoPartition(
    HloComputation* computation, const HloSharding& root_sharding) {
  VLOG(2) << "Partitioning computation " << computation->name() << " for "
          << num_replicas_ << " replicas and " << num_partitions_
          << " partitions";
  TF_RETURN_IF_ERROR(computation->Accept(this));

  HloModule* module = computation->parent();
  auto new_root =
      GetPartitionedHlo(computation->root_instruction()).Reshard(root_sharding);
  auto new_computation =
      module->AddEmbeddedComputation(b_.Build(new_root.hlo()));
  TF_RETURN_IF_ERROR(DoCodeMotionForWindowedDotGeneralLoops(new_computation));

  // Replace the original computation with the new SPMD computation.
  std::unordered_map<HloComputation*, HloComputation*> replacement;
  replacement[computation] = new_computation;
  module->ReplaceComputations(replacement);
  return changed_;
}

Status SpmdPartitioningVisitor::HandlePartitionId(HloInstruction* hlo) {
  return Unimplemented(
      "PartitionId instruction is not supported for SPMD partitioning since "
      "the meaning is ambiguous -- whether the instruction is replicated or "
      "the data is replicated, and if the latter which data is replicated.");
}

SPMDCollectiveOpsCreator GetDefaultCollectiveOpsCreator(int64 num_partitions,
                                                        int64 num_replicas) {
  return {
      [](SpmdBuilder* b) {
        return b->AddInstruction(HloInstruction::CreatePartitionId());
      },
      [num_replicas](SpmdBuilder* b, HloInstruction* operand,
                     HloComputation* reduction, int64 channel_id) {
        return b->AddInstruction(HloInstruction::CreateAllReduce(
            operand->shape(), {operand}, reduction,
            CreateReplicaGroups(num_replicas),
            /*constrain_layout=*/false, channel_id,
            /*use_global_device_ids=*/false));
      },
      [](SpmdBuilder* b, HloInstruction* operand,
         std::vector<std::pair<int64, int64>>& src_dst_pairs,
         int64 channel_id) {
        return b->AddInstruction(HloInstruction::CreateCollectivePermute(
            operand->shape(), operand, src_dst_pairs, channel_id));
      },
      [](SpmdBuilder* b, absl::Span<HloInstruction* const> operands,
         const std::vector<ReplicaGroup>& replica_groups, int64 channel_id,
         absl::optional<int64> split_dimension) {
        std::vector<Shape> shapes(operands.size(), operands[0]->shape());
        const Shape output_shape = (shapes.size() == 1)
                                       ? shapes[0]
                                       : ShapeUtil::MakeTupleShape(shapes);
        return b->AddInstruction(HloInstruction::CreateAllToAll(
            output_shape, operands, replica_groups,
            /*constrain_layout=*/false, channel_id, split_dimension));
      },
      [num_replicas, num_partitions](
          SpmdBuilder* b, HloInstruction* operand, const Shape& ag_shape,
          const std::vector<std::vector<int64>>& partition_subgroups,
          int64 channel_id, int64 all_gather_dimension) {
        std::vector<ReplicaGroup> device_groups;
        device_groups.reserve(partition_subgroups.size() * num_replicas);
        for (int64 i = 0; i < num_replicas; ++i) {
          for (const auto& pgroup : partition_subgroups) {
            device_groups.emplace_back();
            for (int64 pid : pgroup) {
              device_groups.back().add_replica_ids(i * num_partitions + pid);
            }
          }
        }
        return b->AddInstruction(HloInstruction::CreateAllGather(
            ag_shape, operand, all_gather_dimension, device_groups,
            /*constrain_layout=*/false, channel_id,
            /*use_global_device_ids=*/true));
      },
  };
}

SpmdPartitioner::SpmdPartitioner(int64 num_partitions, int64 num_replicas,
                                 SpmdPartitionerOptions options)
    : SpmdPartitioner(
          num_partitions, num_replicas, std::move(options),
          GetDefaultCollectiveOpsCreator(num_partitions, num_replicas)) {}

HloInstruction* SpmdPartitioner::AllGatherShards(SpmdBuilder* b,
                                                 HloInstruction* operand,
                                                 const HloSharding& sharding,
                                                 int64 channel_id) {
  CHECK(!sharding.IsTileMaximal());
  // Add one leading dimension to gather all partitions.
  std::vector<int64> shape;
  shape.push_back(1);
  for (int64 dim : operand->shape().dimensions()) {
    shape.push_back(dim);
  }
  auto reshape = b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(operand->shape().element_type(), shape), operand));
  std::vector<std::vector<int64>> partition_subgroups(1);
  for (int64 pid : sharding.tile_assignment()) {
    partition_subgroups[0].push_back(pid);
  }
  shape[0] = sharding.tile_assignment().num_elements();
  auto result = collective_ops_creator_.create_cross_partition_all_gather(
      b, reshape, ShapeUtil::MakeShape(operand->shape().element_type(), shape),
      partition_subgroups, channel_id, /*all_gather_dimension=*/0);
  // If n > 1 dimensions are partitioned, split the leading dimension to n.
  std::vector<int64> tiled_dims;
  for (int64 i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (sharding.tile_assignment().dim(i) > 1) {
      tiled_dims.push_back(i);
    }
  }
  if (tiled_dims.size() > 1) {
    std::vector<int64> split_dim_shape;
    split_dim_shape.reserve(tiled_dims.size() + operand->shape().rank());
    for (int64 i : tiled_dims) {
      split_dim_shape.push_back(sharding.tile_assignment().dim(i));
    }
    for (int64 dim : operand->shape().dimensions()) {
      split_dim_shape.push_back(dim);
    }
    result = b->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(operand->shape().element_type(), split_dim_shape),
        result));
  }
  // Transpose the gathered dimensions to next to their corresponding
  // partitioned dimensions.
  std::vector<int64> xpose_permutation(result->shape().rank());
  int64 split_dims_added = 0;
  for (int64 i = 0; i < xpose_permutation.size(); ++i) {
    if (sharding.tile_assignment().dim(i - split_dims_added) == 1) {
      xpose_permutation[i] = i + tiled_dims.size() - split_dims_added;
    } else {
      xpose_permutation[i] = split_dims_added;
      split_dims_added++;
      xpose_permutation[i + 1] = i + tiled_dims.size();
      i++;
    }
  }
  result = b->AddInstruction(HloInstruction::CreateTranspose(
      ShapeInference::InferTransposeShape(result->shape(), xpose_permutation)
          .ValueOrDie(),
      result, xpose_permutation));
  // Reshape to the desired shape.
  auto ag_shape = operand->shape();
  for (int64 i : tiled_dims) {
    ag_shape.set_dimensions(
        i, ag_shape.dimensions(i) * sharding.tile_assignment().dim(i));
  }
  result = b->AddInstruction(HloInstruction::CreateReshape(ag_shape, result));
  return result;
}

StatusOr<bool> SpmdPartitioner::PartitionComputation(
    HloComputation* computation, const HloSharding& root_sharding,
    int64* next_channel_id, SpmdLogger* logger) {
  auto visitor =
      CreateVisitor(computation, num_partitions_, num_replicas_,
                    collective_ops_creator_, next_channel_id, logger, options_);
  return visitor->DoPartition(computation, root_sharding);
}

std::unique_ptr<SpmdPartitioningVisitor> SpmdPartitioner::CreateVisitor(
    HloComputation* computation, int64 num_partitions, int64 num_replicas,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, SpmdLogger* logger,
    SpmdPartitionerOptions options) {
  return absl::make_unique<SpmdPartitioningVisitor>(
      computation, num_partitions, num_replicas, collective_ops_creator,
      next_channel_id, logger, std::move(options), this);
}

StatusOr<bool> SpmdPartitioner::Run(HloModule* module) {
  TF_RETURN_IF_ERROR(PreprocessSharding(module));

  XLA_VLOG_LINES(1, SpmdLogger::ReportBeforePartition(
                        *module, options_.report_instruction_count));

  // Add the parameters' and output's shardings to the module.
  std::vector<HloSharding> entry_params_shardings;
  for (int64 i = 0; i < module->entry_computation()->num_parameters(); ++i) {
    auto param = module->entry_computation()->parameter_instruction(i);
    CHECK(param->has_sharding()) << "Missing sharding in entry parameter " << i;
    entry_params_shardings.push_back(param->sharding());
  }
  module->set_spmd_parameters_shardings(entry_params_shardings);
  auto entry_root = module->entry_computation()->root_instruction();
  CHECK(entry_root->has_sharding()) << "Missing sharding in entry root.";
  module->set_spmd_output_sharding(entry_root->sharding());

  FlattenCallGraph flatten;
  TF_ASSIGN_OR_RETURN(auto changed, flatten.Run(module));

  SpmdLogger logger(options_.report_instruction_count);
  auto program_shape = module->entry_computation()->ComputeProgramShape();
  int64 next_channel_id = hlo_query::NextChannelId(*module);
  TF_ASSIGN_OR_RETURN(
      bool partition_changed,
      PartitionComputation(
          module->entry_computation(),
          module->entry_computation()->root_instruction()->sharding(),
          &next_channel_id, &logger));
  changed |= partition_changed;

  // For the entry computation, make sure that the root instruction and the
  // parameters preserve their signatures.
  auto new_program_shape = module->entry_computation()->ComputeProgramShape();
  if (!options_.allow_module_signature_change) {
    TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
        program_shape.result(), new_program_shape.result()))
        << "Result shape changed for the entry computation";
    TF_RET_CHECK(program_shape.parameters_size() ==
                 new_program_shape.parameters_size())
        << "Parameter count changed for the entry computation";
    for (int64 i = 0; i < program_shape.parameters_size(); ++i) {
      TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
          program_shape.parameters(i), new_program_shape.parameters(i)))
          << "Parameter shape changed for the entry computation";
    }
  } else {
    const auto& old_entry_layout = module->entry_computation_layout();
    // Shapes can change but the layout should still remain the same.
    for (int64 i = 0; i < new_program_shape.parameters_size(); ++i) {
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          old_entry_layout.parameter_shape(i),
          new_program_shape.mutable_parameters(i)));
    }
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        old_entry_layout.result_shape(), new_program_shape.mutable_result()));

    HloModuleConfig config = module->config();
    *config.mutable_entry_computation_layout() =
        ComputationLayout(new_program_shape, /*ignore_layouts=*/false);
    module->set_config(config);
  }

  XLA_VLOG_LINES(1, SpmdLogger::ReportAfterPartition(
                        *module, options_.report_instruction_count));
  XLA_VLOG_LINES(1, logger.MakeReport());

  if (changed) {
    HloPassPipeline pass("spmd-cleanup");
    pass.AddPass<TupleSimplifier>();
    pass.AddPass<HloDCE>();
    pass.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
    pass.AddPass<FlattenCallGraph>();
    TF_RETURN_IF_ERROR(pass.Run(module).status());
  }

  TF_RETURN_IF_ERROR(ClearShardingAttributes(module));
  return changed;
}

Status SpmdPartitioner::PreprocessSharding(HloModule* module) {
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->HasSideEffectNoRecurse() && hlo->opcode() != HloOpcode::kRng) {
        TF_RET_CHECK(hlo->has_sharding())
            << "Side-effect HLO must have sharding: " << hlo->ToString();
        TF_RET_CHECK(!HasReplicatedSharding(hlo->sharding()) ||
                     hlo->opcode() == HloOpcode::kInfeed)
            << "Non-infeed side-effect HLO cannot have a replicated sharding:"
            << hlo->ToString();
      }

      // For unassigned HLOs, annotate with replicated sharding.
      //
      // Among side-effecting ops, only Rng is allowed to omit the annotation.
      // In that case, we currently force it to run on core 0, since we don't
      // support partitioning or replicating the Rng op (the values depend on
      // the seed provided to each device).
      //
      // TODO(hyouklee): Should we also convert single-device shardings (without
      // side-effects) into replicated?
      if (!hlo->has_sharding()) {
        if (hlo->opcode() == HloOpcode::kRng) {
          hlo->set_sharding(HloSharding::AssignDevice(0));
        } else {
          hlo->set_sharding(
              HloSharding::Single(hlo->shape(), HloSharding::Replicate()));
        }
      } else if (!hlo->sharding().IsTileMaximal()) {
        std::vector<int64> available(num_partitions_);
        std::iota(available.begin(), available.end(), 0);
        TF_RET_CHECK(num_partitions_ == hlo_sharding_util::DevicesForSharding(
                                            hlo->sharding(), available)
                                            .size())
            << "num_partitions:" << num_partitions_ << "\n"
            << "SPMD partitioner only supports tile sharding that includes all "
               "partitions. If you didn't add this sharding annotation in the "
               "model, please file a bug to XLA team.\n"
            << hlo->ToString();
      }
    }
  }

  // Entry computation's parameter and root sharding must be either all
  // replicated or all on a single device.
  if (!options_.allow_module_signature_change) {
    const HloComputation* entry = module->entry_computation();
    TF_RET_CHECK(entry->root_instruction()->has_sharding());
    const HloSharding& root_sharding = entry->root_instruction()->sharding();
    TF_RET_CHECK(root_sharding.IsReplicated() ||
                 root_sharding.UniqueDevice().has_value())
        << "Unsupported entry root sharding: " << root_sharding.ToString();

    for (const HloInstruction* param : entry->parameter_instructions()) {
      TF_RET_CHECK(param->has_sharding());
      TF_RET_CHECK(param->sharding().IsReplicated() ||
                   param->sharding().UniqueDevice().has_value())
          << "Unsupported entry parameter sharding:"
          << param->sharding().ToString();
    }
  }

  return Status::OK();
}

}  // namespace spmd
}  // namespace xla
