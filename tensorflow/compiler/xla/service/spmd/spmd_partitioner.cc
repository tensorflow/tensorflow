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
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
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
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
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

std::vector<std::vector<int64>> GetPartitionGroupsForReplication(
    const HloSharding& sharding, absl::Span<const int64> replication_dims) {
  int64 group_size = 1;
  for (int64 i : replication_dims) {
    group_size *= sharding.tile_assignment().dim(i);
  }
  std::vector<std::vector<int64>> partition_groups(
      sharding.tile_assignment().num_elements() / group_size);
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64 partition) {
        int64 group_id = 0;
        for (int64 i = 0; i < indices.size(); ++i) {
          if (!absl::c_linear_search(replication_dims, i)) {
            group_id *= sharding.tile_assignment().dim(i);
            group_id += indices[i];
          }
        }
        partition_groups[group_id].push_back(partition);
      });
  return partition_groups;
}

}  // namespace

HloInstruction* SpmdBuilder::AddInstruction(
    std::unique_ptr<HloInstruction> instruction) {
  HloInstruction* hlo =
      HloComputation::Builder::AddInstruction(std::move(instruction));
  if (visiting_hlo_) {
    instructions_[visiting_hlo_].push_back(hlo);
  }
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    for (int64 i = 0; i < hlo->shape().rank(); ++i) {
      if (!absl::c_linear_search(hlo->dimensions(), i)) {
        broadcast_dims_[hlo].insert(i);
      }
    }
  }
  if (hlo->IsElementwise() && hlo->operand_count() > 0) {
    absl::flat_hash_set<int64> broadcast_dims;
    for (int64 i = 0; i < hlo->shape().rank(); ++i) {
      broadcast_dims.insert(i);
    }
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      auto it = broadcast_dims_.find(hlo->operand(i));
      if (it == broadcast_dims_.end()) {
        broadcast_dims.clear();
        break;
      }
      for (int64 i = 0; i < hlo->shape().rank(); ++i) {
        if (!it->second.contains(i)) {
          broadcast_dims.erase(i);
        }
      }
    }
    if (!broadcast_dims.empty()) {
      broadcast_dims_[hlo] = std::move(broadcast_dims);
    }
  }
  if (hlo->opcode() == HloOpcode::kTranspose) {
    auto it = broadcast_dims_.find(hlo->operand(0));
    if (it != broadcast_dims_.end()) {
      absl::flat_hash_set<int64> xpose_broadcast_dims;
      std::vector<int64> reverse_map(hlo->shape().rank());
      for (int64 i = 0; i < reverse_map.size(); ++i) {
        reverse_map[hlo->dimensions(i)] = i;
      }
      for (int64 dim : it->second) {
        xpose_broadcast_dims.insert(reverse_map[dim]);
      }
      broadcast_dims_[hlo] = std::move(xpose_broadcast_dims);
    }
  }
  if (hlo->opcode() == HloOpcode::kReshape &&
      Product(hlo->shape().dimensions()) > 0) {
    auto it = broadcast_dims_.find(hlo->operand(0));
    if (it != broadcast_dims_.end()) {
      absl::flat_hash_set<int64> reshape_broadcast_dims;
      for (int64 i = 0; i < hlo->shape().rank(); ++i) {
        reshape_broadcast_dims.insert(i);
      }
      std::vector<int64> before_dim_size_stack;
      std::vector<int64> after_dim_size_stack;
      for (int64 i = hlo->operand(0)->shape().rank() - 1; i >= 0; --i) {
        before_dim_size_stack.push_back(hlo->operand(0)->shape().dimensions(i));
      }
      for (int64 i = hlo->shape().rank() - 1; i >= 0; --i) {
        after_dim_size_stack.push_back(hlo->shape().dimensions(i));
      }
      while (!before_dim_size_stack.empty() && !after_dim_size_stack.empty()) {
        int64 before_size = before_dim_size_stack.back();
        int64 after_size = after_dim_size_stack.back();
        int64 current_before_dim =
            hlo->operand(0)->shape().rank() - before_dim_size_stack.size();
        int64 current_after_dim =
            hlo->shape().rank() - after_dim_size_stack.size();
        before_dim_size_stack.pop_back();
        after_dim_size_stack.pop_back();
        if (!it->second.contains(current_before_dim)) {
          reshape_broadcast_dims.erase(current_after_dim);
        }
        if (before_size == after_size) {
          continue;
        }
        if (before_size % after_size == 0) {
          // Split dim.
          before_dim_size_stack.push_back(before_size / after_size);
        } else if (after_size % before_size == 0) {
          // Merge dim.
          after_dim_size_stack.push_back(after_size / before_size);
        } else {
          // Other cases, mark all remaining dims as non-broadcast.
          for (int64 i = current_after_dim; i < hlo->shape().rank(); ++i) {
            reshape_broadcast_dims.erase(i);
          }
          break;
        }
      }
      if (!before_dim_size_stack.empty() || !after_dim_size_stack.empty()) {
        reshape_broadcast_dims.clear();
      }
      if (!reshape_broadcast_dims.empty()) {
        broadcast_dims_[hlo] = std::move(reshape_broadcast_dims);
      }
    }
  }
  if (hlo->opcode() == HloOpcode::kSlice ||
      hlo->opcode() == HloOpcode::kDynamicSlice) {
    auto it = broadcast_dims_.find(hlo->operand(0));
    if (it != broadcast_dims_.end()) {
      auto dims = it->second;
      broadcast_dims_[hlo] = std::move(dims);
    }
  }
  if (hlo->opcode() == HloOpcode::kPad) {
    auto it = broadcast_dims_.find(hlo->operand(0));
    if (it != broadcast_dims_.end()) {
      absl::flat_hash_set<int64> pad_broadcast_dims;
      for (int64 i = 0; i < hlo->shape().rank(); ++i) {
        const auto& dim = hlo->padding_config().dimensions(i);
        if (dim.edge_padding_low() == 0 && dim.edge_padding_high() == 0 &&
            dim.interior_padding() == 0 && it->second.contains(i)) {
          pad_broadcast_dims.insert(i);
        }
      }
      if (!pad_broadcast_dims.empty()) {
        broadcast_dims_[hlo] = std::move(pad_broadcast_dims);
      }
    }
  }
  return hlo;
}

PartitionedHlo PartitionedHlo::Reshard(const HloSharding& target) {
  if (sharding() == target) {
    return *this;
  }
  auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
  const bool is_to_replicate =
      hlo_->shape().IsArray() && target.NumTiles() < sharding().NumTiles();
  if (!is_to_replicate || state_.partitioner->options().cache_all_gather) {
    for (auto& entry : cache) {
      if (entry.first == target) {
        return entry.second;
      }
    }
  }
  auto resharded = ReshardNoCache(target);
  state_.reshard_cache->per_hlo_cache[resharded.hlo()]
      .reshard_cache.emplace_back(sharding(), *this);
  if (!is_to_replicate || state_.partitioner->options().cache_all_gather) {
    cache.emplace_back(target, std::move(resharded));
    return cache.back().second;
  }
  return resharded;
}

PartitionedHlo PartitionedHlo::ReshardNoCache(const HloSharding& target) {
  VLOG(2) << "Resharding " << hlo_->ToString() << " from "
          << hlo_->sharding().ToString() << " to " << target.ToString();
  const Shape& shape = hlo_->shape();
  if (shape.element_type() == TOKEN) {
    return *this;
  }
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

  if (CanReshardWithCollectivePermute(sharding(), target)) {
    return ReshardWithCollectivePermute(target);
  }

  if (auto src_tgt_dims =
          GetReshardAllToAllSourceTargetDims(sharding(), target)) {
    return ReshardWithAllToAll(target, *src_tgt_dims);
  }

  if (!target.IsTileMaximal() && sharding().ReplicateOnLastTileDim()) {
    auto try_reshard = ReshardFromPartialReplicateWithDynamicSlice(target);
    if (try_reshard.has_value()) {
      return try_reshard.value();
    }
    try_reshard = ReshardPartialReplicateWithAllToAll(target);
    if (try_reshard.has_value()) {
      return try_reshard.value();
    }
  }

  if (!sharding().IsTileMaximal() && target.ReplicateOnLastTileDim()) {
    auto try_reshard = ReshardToPartialReplicateWithAllGather(target);
    if (try_reshard.has_value()) {
      return try_reshard.value();
    }
    try_reshard = ReshardPartialReplicateWithAllToAll(target);
    if (try_reshard.has_value()) {
      return try_reshard.value();
    }
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

  // 'Replicated' to partial replicated.
  if (target.ReplicateOnLastTileDim()) {
    std::vector<int64> group_dims(target.tile_assignment().num_dimensions() -
                                  1);
    std::iota(group_dims.begin(), group_dims.end(), 0);
    auto target_grouped = GroupShardingOnDims(target, group_dims);
    auto partially_sharded = PerGroupSliceFromReplicated(
        hlo_, state_.partition_id, target_grouped.device_groups, group_dims,
        target_grouped.group_dim_sizes, state_.b);
    partially_sharded->set_sharding(target);
    return PartitionedHlo(partially_sharded, base_shape(), state_);
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

PartitionedHlo PartitionedHlo::PadWithValue(
    HloInstruction* pad_value, absl::Span<const int64> left_padded_dims,
    absl::Span<const int64> skipped_dims) const {
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
    ComparisonDirection direction = ComparisonDirection::kLt;
    int64 index_limit = base_shape_.dimensions(dim);
    if (absl::c_linear_search(left_padded_dims, dim)) {
      direction = ComparisonDirection::kGe;
      index_limit =
          index_shape.dimensions(dim) * sharding.tile_assignment().dim(dim) -
          index_limit;
    }
    auto limit = state_.b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32>(index_limit)));
    auto broadcast_limit = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(index_shape, limit, {}));
    return state_.b->AddInstruction(HloInstruction::CreateCompare(
        mask_shape, index_in_full_shape, broadcast_limit, direction));
  };

  HloInstruction* mask = nullptr;
  auto offsets = MakePartitionOffsets(base_shape_, sharding,
                                      state_.partition_id, state_.b);
  for (int64 i = 0; i < shape.rank(); ++i) {
    if (base_shape_.dimensions(i) % sharding.tile_assignment().dim(i) == 0 ||
        absl::c_linear_search(skipped_dims, i)) {
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
    const auto dilated_size = 1 + (wd.size() - 1) * wd.window_dilation();
    int64 full_size =
        base_shape_.dimensions(i) +
        (wd.base_dilation() - 1) * (base_shape_.dimensions(i) - 1) +
        wd.padding_high() + wd.padding_low();
    if (full_size < dilated_size) {
      VLOG(2) << "Failed to reshard window operand because the window size is "
                 "larger than padded base size";
      return absl::nullopt;
    }
    int64 window_count = (full_size - dilated_size) / wd.stride() + 1;
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
        wd.stride() * (per_shard_window_counts[i] - 1) + dilated_size;
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
            wd.stride() * (per_shard_window_counts[i] - 1) + dilated_size;
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
        if ((wd.stride() * per_shard_window_counts[i]) % wd.base_dilation() !=
            0) {
          // General base dilation not yet implemented.
          return absl::nullopt;
        }
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
    return Reshard(target).ReshardAsWindowedInput(window, target, pad_value);
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
  auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
  if (state_.partitioner->options().cache_all_gather) {
    for (auto& entry : cache) {
      if (entry.first.IsReplicated()) {
        return entry.second;
      }
    }
  }
  const HloSharding& sharding = hlo_->sharding();
  const Shape& shape = hlo_->shape();
  CHECK(!shape.IsTuple() && shape.element_type() != TOKEN);

  if (sharding.IsReplicated()) {
    return *this;
  }
  for (auto& entry : cache) {
    if (entry.first.IsReplicated()) {
      return entry.second;
    }
  }
  auto update_cache = [&](PartitionedHlo resharded) {
    state_.reshard_cache->per_hlo_cache[resharded.hlo()]
        .reshard_cache.emplace_back(sharding, *this);
    if (state_.partitioner->options().cache_all_gather) {
      cache.emplace_back(HloSharding::Replicate(), std::move(resharded));
      return cache.back().second;
    }
    return resharded;
  };
  // 'Single Device' to 'Repliated'.
  if (sharding.IsTileMaximal()) {
    return update_cache(Broadcast());
  }

  // 'Tiled' to 'Replicated'.
  std::vector<int64> all_dims(shape.rank());
  std::iota(all_dims.begin(), all_dims.end(), 0);
  HloInstruction* result = ReplicatePartial(all_dims);
  result->set_sharding(HloSharding::Replicate());
  return update_cache(PartitionedHlo(result, base_shape_, state_));
}

HloInstruction* PartitionedHlo::ReplicatePartial(absl::Span<const int64> dims) {
  CHECK(!sharding().IsTileMaximal());
  const Shape& shard_shape = hlo()->shape();
  Shape target_shape = shard_shape;
  Shape padded_target_shape = shard_shape;
  for (int64 i : dims) {
    padded_target_shape.set_dimensions(
        i, shard_shape.dimensions(i) * sharding().tile_assignment().dim(i));
    target_shape.set_dimensions(i, base_shape().dimensions(i));
  }

  HloInstruction* result = nullptr;
  if (state_.collective_ops_creator.create_cross_partition_all_gather) {
    result = state_.partitioner->AllGatherShards(state_.b, hlo_, sharding(),
                                                 state_.next_channel_id, dims,
                                                 state_.collective_ops_creator);
  }
  if (result == nullptr) {
    auto zero = state_.b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(shard_shape.element_type())));
    auto zero_bcast = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(padded_target_shape, zero, {}));
    auto offsets = MakePartitionOffsets(padded_target_shape, sharding(),
                                        state_.partition_id, state_.b, dims);
    auto dus =
        state_.b->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            padded_target_shape, zero_bcast, hlo_, offsets));
    HloComputation* reduction =
        MakeBinaryAdd(shard_shape.element_type(), state_.module);
    result = state_.partitioner->AllReduceAlongShardingDims(
        state_.b, dus, sharding(), state_.next_channel_id, dims,
        state_.collective_ops_creator, reduction);
  }
  if (!ShapeUtil::Compatible(target_shape, padded_target_shape)) {
    std::vector<int64> start_indices(target_shape.rank(), 0);
    std::vector<int64> strides(target_shape.rank(), 1);
    result = state_.b->AddInstruction(
        HloInstruction::CreateSlice(target_shape, result, start_indices,
                                    target_shape.dimensions(), strides));
  }
  return result;
}

absl::optional<PartitionedHlo>
PartitionedHlo::ReshardToPartialReplicateWithAllGather(
    const HloSharding& target) {
  if (!target.ReplicateOnLastTileDim()) {
    return absl::nullopt;
  }
  // Tiled/partial replicate to partial replicate
  // Get the comptible sharding to target with resharding by all reduce.
  auto compatible_sharding =
      PartialReplicateReshardCompatibleSharding(target, sharding());
  if (!compatible_sharding.has_value()) {
    return absl::nullopt;
  }

  const auto& temp_sharding = compatible_sharding.value();
  auto partitioned_hlo = *this;
  // Use collective permute to adjust device assignment if needed.
  if (CanReshardWithCollectivePermute(sharding(), temp_sharding)) {
    partitioned_hlo =
        partitioned_hlo.ReshardWithCollectivePermute(temp_sharding);
  }

  // Get replicate dims and replicate factor of each dimensions.
  int64 rank = hlo_->shape().rank();
  std::vector<int64> replicate_dims;
  std::vector<int64> replicate_factors;
  for (int64 dim = 0; dim < rank; dim++) {
    int64 replicate_factor = temp_sharding.tile_assignment().dim(dim) /
                             target.tile_assignment().dim(dim);
    if (replicate_factor > 1) {
      replicate_dims.emplace_back(dim);
      replicate_factors.emplace_back(replicate_factor);
    }
  }

  // Do left halo exchange if all-reduce directly will remove useful data
  // from the source.
  auto halo_exchange = TileToPartialReplicateHaloExchange(
      partitioned_hlo.hlo_, base_shape_, temp_sharding, target, replicate_dims,
      partitioned_hlo.state().collective_ops_creator,
      partitioned_hlo.state().next_channel_id,
      partitioned_hlo.state().partition_id, partitioned_hlo.state().b);
  if (!halo_exchange.has_value()) {
    return absl::nullopt;
  }
  auto halo_exchange_hlo = halo_exchange.value();
  // Grouped on replicate dimensions.
  auto sharding_grouped =
      GroupShardingOnDims(temp_sharding, replicate_dims, replicate_factors);
  auto per_group_partitioner_state = CreatePerGroupPartitioningState(
      partitioned_hlo.state(), sharding_grouped.device_groups,
      partitioned_hlo.state().b);
  auto base_shape = MakePartitionedShape(base_shape_, target);
  // It's possible that halo_exchange_hlo == hlo.hlo().
  // Record the sharding of hlo here, and reset it before return.
  auto original_sharding = partitioned_hlo.sharding();
  halo_exchange_hlo->set_sharding(sharding_grouped.sharding);
  auto partial_replicate_hlo = PartitionedHlo(halo_exchange_hlo, base_shape,
                                              per_group_partitioner_state);
  HloInstruction* result =
      partial_replicate_hlo.ReplicatePartial(replicate_dims);
  partitioned_hlo.hlo()->set_sharding(original_sharding);
  result->set_sharding(target);
  return PartitionedHlo(result, base_shape_, partitioned_hlo.state());
}

absl::optional<PartitionedHlo>
PartitionedHlo::ReshardFromPartialReplicateWithDynamicSlice(
    const HloSharding& target) {
  if (!sharding().ReplicateOnLastTileDim()) {
    return absl::nullopt;
  }

  // Get the temp sharding target from partial replicate to target tile dims.
  // target_compatible_sharding has the same tile_assignment dimensions
  // as the target and can reshard to target by collective permute.
  // target_compatible_sharding could have different device assignment as
  // targe. sharding() can reshard to target_compatible_sharding by
  // dynamic slice.
  auto target_compatible_sharding =
      PartialReplicateReshardCompatibleSharding(sharding(), target);
  // Reshard to target_compatible_sharding by dynamic slice.
  if (!target_compatible_sharding.has_value()) {
    return absl::nullopt;
  }
  std::vector<int64> expand_tile_dims;
  std::vector<int64> tiling_dim_factors;
  int64 rank = hlo_->shape().rank();
  tiling_dim_factors.reserve(target.tile_assignment().num_dimensions());
  const auto& temp_target_sharding = target_compatible_sharding.value();
  for (int64 dim = 0; dim < rank; dim++) {
    if (temp_target_sharding.tile_assignment().dim(dim) >
        sharding().tile_assignment().dim(dim)) {
      expand_tile_dims.push_back(dim);
    }
    tiling_dim_factors.emplace_back(
        temp_target_sharding.tile_assignment().dim(dim) /
        sharding().tile_assignment().dim(dim));
  }

  // Add another dimension in tiling_dim_factors if target is partial replicate.
  if (target.ReplicateOnLastTileDim()) {
    tiling_dim_factors.emplace_back(
        target.tile_assignment().dimensions().back());
  }

  // 2. Get the padded_hlo, do right halo exchange if needed.
  auto padded_hlo = PadFromPartialReplicateShape(
      hlo_, base_shape_, sharding(), temp_target_sharding, expand_tile_dims,
      state_.collective_ops_creator, state_.next_channel_id,
      state_.partition_id, state_.b);
  if (!padded_hlo.has_value()) {
    return absl::nullopt;
  }
  // 3. Slice out the tile from replicate ones.
  auto shard_shape = MakePartitionedShape(base_shape_, temp_target_sharding);
  // Since we are just slicing, we can just use the differences between the new
  // and old offsets in the full shape as the dynamic-slice offsets.
  auto padded_base_shape = shard_shape;
  for (int64 i = 0; i < padded_base_shape.rank(); ++i) {
    padded_base_shape.set_dimensions(
        i, padded_base_shape.dimensions(i) *
               temp_target_sharding.tile_assignment().dim(i));
  }
  auto offsets = MakePartitionOffsets(padded_base_shape, temp_target_sharding,
                                      state_.partition_id, state_.b);
  auto old_offsets = MakePartitionOffsets(padded_base_shape, sharding(),
                                          state_.partition_id, state_.b);
  for (int64 i = 0; i < offsets.size(); ++i) {
    offsets[i] = state_.b->AddInstruction(HloInstruction::CreateBinary(
        offsets[i]->shape(), HloOpcode::kSubtract, offsets[i], old_offsets[i]));
  }
  auto slice = state_.b->AddInstruction(HloInstruction::CreateDynamicSlice(
      shard_shape, padded_hlo.value(), offsets, shard_shape.dimensions()));
  slice->set_sharding(temp_target_sharding);
  auto result = PartitionedHlo(slice, base_shape_, state_);
  // If temp_target_sharding's device assignment is different from target,
  // use collective permute to reshard.
  if (CanReshardWithCollectivePermute(temp_target_sharding, target)) {
    return result.ReshardWithCollectivePermute(target);
  }
  // If device assignment in temp_target_sharding and target are the same,
  // return result directly.
  return result;
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
      state_.b, operand, reduction, {}, NewChannel());
  result->set_sharding(HloSharding::Replicate());
  return PartitionedHlo(result, base_shape_, state_);
}

PartitionedHlo PartitionedHlo::ReshardWithAllToAll(
    const HloSharding& target,
    absl::Span<const std::pair<int64, int64>> source_target_dims) const {
  if (source_target_dims.empty()) {
    if (target == sharding()) {
      return *this;
    }
    // If the device order is different in the target, fix the order with
    // ReshardWithCollectivePermute.
    return ReshardWithCollectivePermute(target);
  }

  // Swap one pair of dimensions.
  int64 source_dim = source_target_dims[0].first;
  int64 target_dim = source_target_dims[0].second;
  const int64 group_size = sharding().tile_assignment().dim(source_dim) /
                           sharding().tile_assignment().dim(target_dim);

  auto temp_target_tile = sharding().tile_assignment();
  {
    std::vector<int64> reshape_tile_dims(temp_target_tile.num_dimensions() + 2);
    int64 i = 0;
    int64 added_source_dim = -1;
    int64 added_target_dim = -1;
    for (int64 j = 0; j < temp_target_tile.num_dimensions(); ++j) {
      if (source_dim == j) {
        reshape_tile_dims[i] = temp_target_tile.dim(j) / group_size;
        reshape_tile_dims[++i] = group_size;
        added_source_dim = i;
      } else if (target_dim == j) {
        reshape_tile_dims[i] = temp_target_tile.dim(j);
        reshape_tile_dims[++i] = 1;
        added_target_dim = i;
      } else {
        reshape_tile_dims[i] = temp_target_tile.dim(j);
      }
      ++i;
    }
    temp_target_tile.Reshape(reshape_tile_dims);
    std::vector<int64> xpose_dims(temp_target_tile.num_dimensions());
    std::iota(xpose_dims.begin(), xpose_dims.end(), 0);
    xpose_dims[added_source_dim] = added_target_dim;
    xpose_dims[added_target_dim] = added_source_dim;
    temp_target_tile = hlo_sharding_util::TransposeSharding(
                           HloSharding::Tile(temp_target_tile), xpose_dims)
                           .tile_assignment();
    auto temp_target_tile_dims = sharding().tile_assignment().dimensions();
    temp_target_tile_dims[source_dim] =
        sharding().tile_assignment().dim(target_dim);
    temp_target_tile_dims[target_dim] =
        sharding().tile_assignment().dim(source_dim);
    temp_target_tile.Reshape(temp_target_tile_dims);
  }
  auto temp_target = target.ReplicateOnLastTileDim()
                         ? HloSharding::PartialTile(temp_target_tile)
                         : HloSharding::Tile(temp_target_tile);
  auto padded_shape = hlo_->shape();
  padded_shape.set_dimensions(
      target_dim,
      RoundUpToNearest(padded_shape.dimensions(target_dim),
                       temp_target.tile_assignment().dim(target_dim)));
  auto padded_hlo = PadToShape(hlo_, padded_shape, state_.b);

  // The order of ids in the group must follow the temp_target sharding.
  std::vector<std::vector<int64>> groups(
      temp_target.tile_assignment().num_elements() / group_size);
  temp_target.tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64 device) {
        int64 group_id = 0;
        for (int64 dim = 0; dim < indices.size(); ++dim) {
          if (dim == target_dim) {
            group_id *= temp_target.tile_assignment().dim(dim) / group_size;
            group_id += indices[dim] / group_size;
          } else {
            group_id *= temp_target.tile_assignment().dim(dim);
            group_id += indices[dim];
          }
        }
        groups[group_id].push_back(device);
      });

  HloInstruction* result = nullptr;

  // Split along the split dimension (target_dim) of the all-to-all
  // output.
  std::vector<int64> dimensions;
  for (int64 i = 0; i < base_shape_.rank(); ++i) {
    if (i == target_dim) {
      dimensions.push_back(group_size);
      dimensions.push_back(padded_hlo->shape().dimensions(i) / group_size);
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
          state_.b, {reshape}, groups, (*state_.next_channel_id)++, target_dim);

  // Reorder the split dimension of the reshape to be located in front of the
  // input partition dimension, so the two dimensions can be combined.
  int64 new_source_dim =
      (target_dim < source_dim) ? source_dim + 1 : source_dim;
  std::vector<int64> permutation;
  for (int64 i = 0; i < all_to_all->shape().rank(); ++i) {
    if (i == target_dim) {
      continue;
    }
    if (i == new_source_dim) {
      permutation.push_back(target_dim);
    }
    permutation.push_back(i);
  }
  auto transpose = state_.b->AddInstruction(HloInstruction::CreateTranspose(
      ShapeInference::InferTransposeShape(all_to_all->shape(), permutation)
          .ValueOrDie(),
      all_to_all, permutation));

  // Combine the split dimension and the input partition dimension.
  auto new_shape = ShapeInference::InferAllToAllShape(
                       padded_hlo->shape(), target_dim, source_dim, group_size)
                       .ValueOrDie();
  result = state_.b->AddInstruction(
      HloInstruction::CreateReshape(new_shape, transpose));

  const Shape result_shape = MakePartitionedShape(base_shape_, temp_target);
  if (result_shape != result->shape()) {
    result = state_.b->AddInstruction(HloInstruction::CreateSlice(
        result_shape, result, std::vector<int64>(result_shape.rank(), 0),
        result_shape.dimensions(), std::vector<int64>(result_shape.rank(), 1)));
  }
  result->set_sharding(temp_target);
  auto remaining_source_target_dims = source_target_dims;
  remaining_source_target_dims.remove_prefix(1);
  return PartitionedHlo(result, base_shape_, state_)
      .ReshardWithAllToAll(target, remaining_source_target_dims);
}

absl::optional<PartitionedHlo>
PartitionedHlo::ReshardPartialReplicateWithAllToAll(const HloSharding& target) {
  bool source_is_partial_replicate = sharding().ReplicateOnLastTileDim();
  const auto& partial_replicate_sharding =
      source_is_partial_replicate ? sharding() : target;
  // If neither the source nor the target is partial replicate, return null.
  if (!partial_replicate_sharding.ReplicateOnLastTileDim()) {
    return absl::nullopt;
  }
  const auto& tile_sharding = source_is_partial_replicate ? target : sharding();
  // If both source and target are partial replicate, should be supported in
  // Reshard with AllToAll already.
  if (tile_sharding.ReplicateOnLastTileDim() || tile_sharding.IsTileMaximal()) {
    return absl::nullopt;
  }

  // Only support resharding from sharding={devices=[2,3]0,1,2,3,4,5}
  // to sharding={devices=[1,2,3]0,1,2,3,4,5 last_tile_dim_replicate}, where
  // the last tile dim will be replicate first before all-to-all.
  // Or resharding from
  // sharding={devices=[1,2,3]0,1,2,3,4,5 last_tile_dim_replicate}
  // to sharding={devices=[2,3]0,1,2,3,4,5}, where
  // the last tile dim will be sharded after all-to-all.
  const int num_replicas =
      partial_replicate_sharding.tile_assignment().dimensions().back();
  if (((tile_sharding.tile_assignment().num_dimensions() + 1) !=
       partial_replicate_sharding.tile_assignment().num_dimensions()) ||
      (partial_replicate_sharding.tile_assignment().dim(0) != 1)) {
    return absl::nullopt;
  }
  int to_replicate_dim = -1;
  for (int i = tile_sharding.tile_assignment().num_dimensions() - 1; i >= 0;
       --i) {
    if (tile_sharding.tile_assignment().dim(i) > 1 &&
        (to_replicate_dim == -1)) {
      if (tile_sharding.tile_assignment().dim(i) != num_replicas) {
        return absl::nullopt;
      }
      to_replicate_dim = i;
    }

    if (tile_sharding.tile_assignment().dim(i) !=
        partial_replicate_sharding.tile_assignment().dim(i + 1)) {
      return absl::nullopt;
    }
  }

  if (to_replicate_dim == -1) {
    return absl::nullopt;
  }

  // Check if core assignments for source and the target are the same.
  auto reshape_tile_assignment = partial_replicate_sharding.tile_assignment();
  reshape_tile_assignment.Reshape(tile_sharding.tile_assignment().dimensions());
  if (reshape_tile_assignment != tile_sharding.tile_assignment()) {
    return absl::nullopt;
  }

  auto tmp_tile_assignment = tile_sharding.tile_assignment();
  auto tmp_tile_assignment_dimensions =
      tile_sharding.tile_assignment().dimensions();
  tmp_tile_assignment_dimensions[to_replicate_dim] = 1;
  tmp_tile_assignment_dimensions.push_back(num_replicas);
  tmp_tile_assignment.Reshape(tmp_tile_assignment_dimensions);
  auto tmp_partial_replicate_sharding =
      HloSharding::PartialTile(tmp_tile_assignment);

  if (source_is_partial_replicate) {
    if (auto src_tgt_dims = GetReshardAllToAllSourceTargetDims(
            sharding(), tmp_partial_replicate_sharding)) {
      auto partitioned_hlo =
          ReshardWithAllToAll(tmp_partial_replicate_sharding, *src_tgt_dims);
      return partitioned_hlo.Reshard(target);
    }
  } else {
    auto partitioned_hlo = Reshard(tmp_partial_replicate_sharding);

    if (auto src_tgt_dims = GetReshardAllToAllSourceTargetDims(
            partitioned_hlo.sharding(), target)) {
      return partitioned_hlo.ReshardWithAllToAll(target, *src_tgt_dims);
    }
  }

  return absl::nullopt;
}

PartitionedHlo PartitionedHlo::ReshardWithCollectivePermute(
    const HloSharding& target) const {
  CHECK(CanReshardWithCollectivePermute(sharding(), target))
      << sharding().ToString() << " to " << target.ToString();
  if (auto broadcast_dims = state_.b->BroadcastDimsForCreatedHlo(hlo())) {
    if (!(*broadcast_dims)->empty()) {
      // If hlo() has broadcast dims, check if data is already the same between
      // source/destination pairs.
      std::vector<int64> broadcast_dims_vector;
      for (int64 i = 0; i < hlo()->shape().rank(); ++i) {
        if ((*broadcast_dims)->contains(i)) {
          broadcast_dims_vector.push_back(i);
        }
      }
      if (hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
              sharding(), broadcast_dims_vector) ==
          hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
              target, broadcast_dims_vector)) {
        auto copy = state_.b->AddInstruction(HloInstruction::CreateUnary(
            hlo()->shape(), HloOpcode::kCopy, hlo()));
        copy->set_sharding(target);
        return PartitionedHlo(copy, base_shape_, state_);
      }
    }
  }
  std::vector<std::pair<int64, int64>> src_dst_pairs;
  sharding().tile_assignment().Each(
      [&](absl::Span<const int64> indices, int64 src_device) {
        int64 dst_device = target.tile_assignment()(indices);
        src_dst_pairs.emplace_back(src_device, dst_device);
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

  HloSharding sharding = hlo->sharding().HasUniqueDevice()
                             ? hlo->sharding()
                             : HloSharding::Replicate();

  // If the instruction cannot be partitioned, replicate the instruction unless
  // the instruction has side-effect.
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : hlo->operands()) {
    new_operands.push_back(GetPartitionedHlo(operand).Reshard(sharding).hlo());
  }
  auto clone =
      b_.AddInstruction(hlo->CloneWithNewOperands(hlo->shape(), new_operands));
  clone->set_sharding(sharding);
  clone->set_metadata(hlo->metadata());
  SetPartitionedHlo(hlo,
                    PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
                        .Reshard(hlo->sharding()));
  return Status::OK();
}

Status SpmdPartitioningVisitor::Preprocess(HloInstruction* hlo) {
  visiting_hlo_ = hlo;
  b_.set_visiting_hlo(hlo);
  // Temporarily replace manual sharding to one-device sharding so that the
  // partitioner will not change the HLOs.
  auto manual_to_onedevice = [&](const Shape& shape,
                                 const HloSharding& sharding) {
    // If a tuple's elements are all manual, then sharding.IsManual() == True,
    // so we test whether it is tuple first.
    if (sharding.IsTuple()) {
      std::vector<HloSharding> subshardings = sharding.tuple_elements();
      for (HloSharding& subsharding : subshardings) {
        if (subsharding.IsManual()) {
          subsharding = HloSharding::AssignDevice(0);
        }
      }
      return HloSharding::Tuple(shape, subshardings);
    }
    if (sharding.IsManual()) {
      return HloSharding::AssignDevice(0);
    }
    return sharding;
  };
  const bool has_manual_sharding =
      hlo->sharding().IsManual() ||
      (hlo->sharding().IsTuple() &&
       absl::c_any_of(
           hlo->sharding().tuple_elements(),
           [](const HloSharding& sharding) { return sharding.IsManual(); }));
  if (has_manual_sharding && !hlo->IsCustomCall("SPMDFullToShardShape")) {
    visiting_hlo_sharding_ = hlo->sharding();
    hlo->set_sharding(
        manual_to_onedevice(hlo->shape(), *visiting_hlo_sharding_));

    visiting_hlo_operand_shardings_.reserve(hlo->operand_count());
    for (auto operand : hlo->operands()) {
      visiting_hlo_operand_shardings_.push_back(operand->sharding());
      operand->set_sharding(
          manual_to_onedevice(operand->shape(), operand->sharding()));
      GetPartitionedHlo(operand).hlo()->set_sharding(operand->sharding());
    }
  }
  return Status::OK();
}

Status SpmdPartitioningVisitor::Postprocess(HloInstruction* hlo) {
  logger_->RegisterLogEntry(GetPartitionedHlo(hlo).hlo(),
                            b_.derived_instructions(hlo));
  visiting_hlo_ = nullptr;
  b_.set_visiting_hlo(nullptr);
  // Revert fake one-device shardings for manually partitioned ops.
  if (visiting_hlo_sharding_) {
    hlo->set_sharding(*visiting_hlo_sharding_);
    GetPartitionedHlo(hlo).hlo()->set_sharding(*visiting_hlo_sharding_);
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      auto operand = hlo->mutable_operand(i);
      operand->set_sharding(visiting_hlo_operand_shardings_[i]);
      GetPartitionedHlo(operand).hlo()->set_sharding(operand->sharding());
    }
    visiting_hlo_sharding_.reset();
    visiting_hlo_operand_shardings_.clear();
  }
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

  // temp_output_shape is the output shape where the concatenate dimension
  // is changed to the full (and padded to shard count) dimension size.
  auto temp_output_shape = MakePartitionedShape(hlo->shape(), sharding);
  auto last_operand_padded_shape =
      MakePartitionedShape(hlo->operands().back()->shape(), sharding);
  // If the last operand has more padding than the temp_output padding, needs to
  // add extra padding to avoid dynamic update slice out of bound.
  int last_operand_padding =
      last_operand_padded_shape.dimensions(dimension) *
          sharding.tile_assignment().dim(dimension) -
      hlo->operands().back()->shape().dimensions(dimension);
  int temp_output_padding = temp_output_shape.dimensions(dimension) *
                                sharding.tile_assignment().dim(dimension) -
                            hlo->shape().dimensions(dimension);
  int padding_for_last_operand =
      last_operand_padding < temp_output_padding
          ? 0
          : last_operand_padding - temp_output_padding;
  temp_output_shape.set_dimensions(
      dimension, temp_output_shape.dimensions(dimension) *
                         sharding.tile_assignment().dim(dimension) +
                     padding_for_last_operand);
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
  std::vector<int64> non_concat_dims;
  non_concat_dims.reserve(hlo->shape().rank() - 1);
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (i != dimension) {
      non_concat_dims.push_back(i);
    }
  }
  auto grouped = GroupShardingOnDims(sharding, non_concat_dims);
  auto per_group_partitioner_state = CreatePerGroupPartitioningState(
      MakePartitioningState(), grouped.device_groups, &b_);
  auto all_reduce = per_group_partitioner_state.collective_ops_creator
                        .create_cross_partition_all_reduce(
                            &b_, temp_output,
                            MakeBinaryAdd(hlo->shape().element_type(), module_),
                            {}, NewChannel());
  SetPartitionedHlo(hlo, [&] {
    auto start_indices = MakeTiledPartitionOrdinals(
        grouped.sharding, per_group_partitioner_state.partition_id, &b_);
    start_indices[dimension] = MultiplyAddDivideOffsetCalculation(
                                   shard_shape.dimensions(dimension), 0, 1)
                                   .Calculate(start_indices[dimension], &b_);
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, all_reduce, start_indices, shard_shape.dimensions()));
  });

  return Status::OK();
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
                          operand.base_shape().dimensions(i));
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
    auto data = reshard_operand->sharded_input;
    // Create a copy so that it will not share the resharding cache.
    return b_.AddInstruction(
        HloInstruction::CreateUnary(data->shape(), HloOpcode::kCopy, data));
  });

  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleSort(HloInstruction* hlo) {
  HloSharding sharding = hlo->sharding();
  // Special handling for sort in TopK when first operand partitioined at
  // sort dimension.
  auto k = GetKValueInTopKWhenPartitionSortDim(hlo);
  if (k.has_value()) {
    // When the first operand partitioned at sort dimension:
    //   1. Partition sort computation to different partitions;
    //   2. Slice TopK value and index from different partitions;
    //   3. Gather and replicate value and index from different partitions,
    //      the shape of replicated value and index will be
    //      [batch_size, ..., partition_count * k, ...];
    //   4. Final sort uses replicated value and index from different partitions
    //      as input.
    // GetTupleElement and Slice after the non-partitoned sort won't change
    // at this point, as HandleGetTupleElement and HandleSlice will update them.
    HloSortInstruction* sort = DynCast<HloSortInstruction>(hlo);
    const int64 sort_dim = sort->sort_dimension();
    auto input = hlo->operand(0);
    auto index = hlo->operand(1);
    const HloSharding& input_sharding = input->sharding();
    const int64 partition_count =
        input_sharding.tile_assignment().dim(sort_dim);
    const int64 input_size = input->shape().dimensions(sort_dim);
    const int64 per_partition_size = CeilOfRatio(input_size, partition_count);
    const auto element_type = input->shape().element_type();
    const auto index_type = index->shape().element_type();

    // Partition and pad input and index.
    // Pad input with minimal value.
    auto partitioned_input = GetPartitionedHlo(input).PadWithValue(
        CreateFirstWithType(element_type, &b_));
    // Pad index with max value.
    auto partitioned_index =
        GetPartitionedHlo(index)
            .Reshard(input_sharding)
            .PadWithValue(CreateLastWithType(index_type, &b_));

    // Each partition needs to do TopK separately, thus the base shape
    // becomes the padded shape.
    std::vector<int64> replicated_dimensions(
        input->shape().dimensions().begin(), input->shape().dimensions().end());
    replicated_dimensions[sort_dim] = per_partition_size * partition_count;
    const Shape replicated_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(element_type, replicated_dimensions),
         ShapeUtil::MakeShape(index_type, replicated_dimensions)});

    // Partition original topk to different shards.
    auto topk_sharding =
        input_sharding.GetTupleSharding(replicated_shape).ValueOrDie();
    auto shard_shape = MakePartitionedShape(replicated_shape, topk_sharding);
    auto topk = b_.AddInstruction(hlo->CloneWithNewOperands(
        shard_shape, {partitioned_input.hlo(), partitioned_index.hlo()}));

    // Get value from first sort.
    HloInstruction* value_gte =
        b_.AddInstruction(HloInstruction::CreateGetTupleElement(
            topk->shape().tuple_shapes(0), topk, 0));
    HloInstruction* index_gte =
        b_.AddInstruction(HloInstruction::CreateGetTupleElement(
            topk->shape().tuple_shapes(1), topk, 1));

    // Slice top K value from the first partitioned sort.
    replicated_dimensions[sort_dim] = k.value() * partition_count;
    auto slice_input = SliceFirstK(value_gte, &b_, sort_dim, k.value());
    slice_input->set_sharding(input_sharding);
    PartitionedHlo partitioned_slice_input(
        slice_input, ShapeUtil::MakeShape(element_type, replicated_dimensions),
        MakePartitioningState());
    // Reshard value to be replicated.
    auto replicated_slice_input =
        partitioned_slice_input.Reshard(HloSharding::Replicate()).hlo();

    // Slice top K index from the first parttioned sort.
    auto slice_index = SliceFirstK(index_gte, &b_, sort_dim, k.value());
    slice_index->set_sharding(input_sharding);
    PartitionedHlo partitioned_slice_index(
        slice_index, ShapeUtil::MakeShape(index_type, replicated_dimensions),
        MakePartitioningState());
    // Reshard value to be replicated.
    auto replicated_slice_index =
        partitioned_slice_index.Reshard(HloSharding::Replicate()).hlo();

    // Creates replicated sort to do TopK, the input is value and index pairs
    // from all the partitions.
    const Shape final_topk_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(element_type, replicated_dimensions),
         ShapeUtil::MakeShape(index_type, replicated_dimensions)});
    HloInstruction* final_sort = b_.AddInstruction(HloInstruction::CreateSort(
        final_topk_shape, sort_dim,
        {replicated_slice_input, replicated_slice_index}, sort->to_apply(),
        sort->is_stable()));
    final_sort->set_sharding(HloSharding::Replicate()
                                 .GetTupleSharding(final_sort->shape())
                                 .ValueOrDie());
    PartitionedHlo replicated_sort(final_sort, final_sort->shape(),
                                   MakePartitioningState());
    SetPartitionedHlo(hlo, replicated_sort.Reshard(hlo->sharding()));

    return Status::OK();
  }

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
    CHECK(hlo->sharding().IsManual());
    CHECK(ShapeUtil::Compatible(input->shape(), hlo->shape()));
    auto copy = b_.AddInstruction(
        HloInstruction::CreateUnary(input->shape(), HloOpcode::kCopy, input));
    SetPartitionedHlo(hlo, [&] { return copy; });
    return Status::OK();
  }
  if (hlo->custom_call_target() == "SPMDShardToFullShape") {
    // This op switches from manual partitioning to auto partitioning.
    auto input = GetPartitionedHlo(hlo->operand(0)).hlo();
    CHECK(input->sharding().IsManual());
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

  auto partitioned_input = GetPartitionedHlo(input).PadWithValue(
      CreateFirstWithType(element_type, &b_));

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
  // Slice value from final sort.
  HloInstruction* slice_sort_value =
      SliceFirstK(sort_value_gte, &b_, sort_dim, k);
  // Slice index from final sort.
  HloInstruction* slice_index_value =
      SliceFirstK(sort_index_gte, &b_, sort_dim, k);
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
  absl::optional<HloSharding> desired_operand_sharding =
      hlo_sharding_util::ReshapeSharding(hlo->shape(), hlo->operand(0)->shape(),
                                         hlo->sharding());
  if (desired_operand_sharding.has_value()) {
    auto operand_hlo = operand.Reshard(*desired_operand_sharding).hlo();
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(hlo->CloneWithNewOperands(
          MakePartitionedShape(hlo->shape(), hlo->sharding()), {operand_hlo}));
    });
    return Status::OK();
  }
  absl::optional<HloSharding> desired_output_sharding =
      hlo_sharding_util::ReshapeSharding(hlo->operand(0)->shape(), hlo->shape(),
                                         operand.sharding());
  if (desired_output_sharding.has_value()) {
    auto reshape = b_.AddInstruction(hlo->CloneWithNewOperands(
        MakePartitionedShape(hlo->shape(), *desired_output_sharding),
        {operand.hlo()}));
    reshape->set_sharding(*desired_output_sharding);
    SetPartitionedHlo(hlo, [&] {
      return PartitionedHlo(reshape, hlo->shape(), MakePartitioningState())
          .Reshard(sharding)
          .hlo();
    });
    return Status::OK();
  }

  // Check if operand sharding and sharding are both tiled or partial replicate.
  // If both of them are partial replicate, check num_replications are the same.
  if (operand.sharding().ReplicateOnLastTileDim() !=
          sharding.ReplicateOnLastTileDim() ||
      (sharding.ReplicateOnLastTileDim() &&
       (operand.sharding().tile_assignment().dimensions().back() !=
        sharding.tile_assignment().dimensions().back()))) {
    return DefaultAction(hlo);
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
  auto aligned_sharding =
      sharding.ReplicateOnLastTileDim()
          ? HloSharding::PartialTile(new_input_tile_assignment)
          : HloSharding::Tile(new_input_tile_assignment);
  operand = operand.Reshard(aligned_sharding);
  auto replication_count = sharding.ReplicateOnLastTileDim()
                               ? sharding.tile_assignment().dimensions().back()
                               : 1;

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
                                  num_partitions_ / replication_count -
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
        output_sharded_dim, tmp_shard_shape.dimensions(output_sharded_dim) *
                                num_partitions_ / replication_count);
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
                                  num_partitions_ / replication_count);
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
  std::vector<int64> new_dims;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (!absl::c_linear_search(hlo->dimensions(), i)) {
      new_dims.push_back(i);
    }
  }
  auto desired_input_sharding = hlo_sharding_util::RemoveShapeDimensions(
      hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(hlo->sharding(),
                                                               new_dims),
      new_dims);
  auto input = operand.Reshard(desired_input_sharding).hlo();
  auto output_shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(
        hlo->CloneWithNewOperands(output_shard_shape, {input}));
  });
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

  std::vector<int64> partitioned_slice_dims;
  std::vector<int64> slice_dims;
  std::vector<int64> partitioned_non_slice_dims;
  std::vector<int64> partitioned_slice_offsets;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->operand(1)->shape().dimensions(i) != hlo->shape().dimensions(i)) {
      slice_dims.push_back(i);
      if (hlo->sharding().tile_assignment().dim(i) != 1) {
        if (!hlo->operand(i + 2)->IsConstant()) {
          return DefaultAction(hlo);
        }
        partitioned_slice_dims.push_back(i);
        partitioned_slice_offsets.push_back(
            hlo->operand(i + 2)->literal().Get<int>({}));
      }
    } else if (hlo->sharding().tile_assignment().dim(i) != 1) {
      if (!hlo->operand(i + 2)->IsConstant() ||
          !hlo->operand(i + 2)->literal().IsZero({})) {
        return DefaultAction(hlo);
      }
      partitioned_non_slice_dims.push_back(i);
    }
  }

  // Handle when there is slice dim partitioned.
  if (!partitioned_slice_dims.empty()) {
    auto add_hlo = [&](std::unique_ptr<HloInstruction> to_add) {
      return b_.AddInstruction(std::move(to_add));
    };
    std::vector<HloInstruction*> new_indices(hlo->shape().rank());
    for (int64 i = 0; i < new_indices.size(); ++i) {
      // Replicate the indices.
      new_indices[i] = GetPartitionedHlo(hlo->operand(i + 2))
                           .Reshard(HloSharding::Replicate())
                           .hlo();
    }

    // Get partitioned input.
    const auto& dus_sharding = hlo->sharding();
    const auto& partitioned_input =
        GetPartitionedHlo(hlo->operand(0)).Reshard(dus_sharding).hlo();

    // Get replicate update.
    auto update_sharding = HloSharding::Replicate();
    if (!partitioned_non_slice_dims.empty()) {
      // Do partial replicate for update if non slice dims are partitioned.
      update_sharding =
          hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(dus_sharding,
                                                                   slice_dims);
    }
    HloInstruction* replicate_update =
        GetPartitionedHlo(hlo->operand(1)).Reshard(update_sharding).hlo();

    const auto& update_shape = replicate_update->shape();
    const auto& partitioned_shape = partitioned_input->shape();
    auto partition_ordinals =
        MakeTiledPartitionOrdinals(hlo->sharding(), partition_id_, &b_);
    HloInstruction* all_dims_within_partition = add_hlo(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));

    for (int i = 0; i < partitioned_slice_dims.size(); ++i) {
      int dim = partitioned_slice_dims[i];
      // Calculate per partition size.
      const int64 per_partition_size = partitioned_shape.dimensions(dim);

      // Only update within a single partition is supported.
      if ((partitioned_slice_offsets[i] / per_partition_size) !=
          ((partitioned_slice_offsets[i] + update_shape.dimensions(dim) - 1) /
           per_partition_size)) {
        return DefaultAction(hlo);
      }

      // within_partition = (offset >= partition_id * per_partition_size) &&
      //                    (offset < (partition_id + 1) * per_partition_size)
      const Shape& compare_shape =
          ShapeUtil::ChangeElementType(partition_id_->shape(), PRED);
      auto per_partition_size_hlo = add_hlo(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int>(per_partition_size)));
      const Shape& offset_shape = per_partition_size_hlo->shape();
      auto partition_offset = add_hlo(HloInstruction::CreateBinary(
          offset_shape, HloOpcode::kMultiply, partition_ordinals[dim],
          per_partition_size_hlo));
      // offset >= partition_id * per_partition_size
      auto offset_ge = add_hlo(HloInstruction::CreateCompare(
          compare_shape, new_indices[dim], partition_offset,
          ComparisonDirection::kGe));
      // offset < (partition_id + 1) * per_partition_size
      auto offset_lt = add_hlo(HloInstruction::CreateCompare(
          compare_shape, new_indices[dim],
          add_hlo(HloInstruction::CreateBinary(
              offset_shape, HloOpcode::kMultiply,
              add_hlo(HloInstruction::CreateBinary(
                  offset_shape, HloOpcode::kAdd, partition_ordinals[dim],
                  add_hlo(HloInstruction::CreateConstant(
                      LiteralUtil::CreateR0<int>(1))))),
              per_partition_size_hlo)),
          ComparisonDirection::kLt));
      auto update_within_partition = add_hlo(HloInstruction::CreateBinary(
          compare_shape, HloOpcode::kAnd, offset_ge, offset_lt));

      all_dims_within_partition = add_hlo(HloInstruction::CreateBinary(
          compare_shape, HloOpcode::kAnd, all_dims_within_partition,
          update_within_partition));

      // Calculate offset.
      // slice dim offset =
      //  within_partition ?
      //  offset - partition_id * per_partition_size : 0
      new_indices[dim] = add_hlo(HloInstruction::CreateTernary(
          new_indices[dim]->shape(), HloOpcode::kSelect,
          update_within_partition,
          add_hlo(HloInstruction::CreateBinary(
              new_indices[dim]->shape(), HloOpcode::kSubtract, new_indices[dim],
              partition_offset)),
          add_hlo(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(0)))));
    }

    // Create dynamic update slice.
    auto dus = add_hlo(HloInstruction::CreateDynamicUpdateSlice(
        partitioned_shape, partitioned_input, replicate_update, new_indices));
    SetPartitionedHlo(hlo, [&]() {
      // Select if update is needed.
      return add_hlo(HloInstruction::CreateTernary(
          dus->shape(), HloOpcode::kSelect,
          add_hlo(HloInstruction::CreateBroadcast(
              ShapeUtil::ChangeElementType(dus->shape(), PRED),
              all_dims_within_partition, {})),
          dus, partitioned_input));
    });
    return Status::OK();
  }

  // Partition non slice dims only.
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

Status SpmdPartitioningVisitor::HandleGetTupleElement(HloInstruction* hlo) {
  const auto& tuple = GetPartitionedHlo(hlo->operand(0));
  auto gte = b_.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetTupleElementShape(tuple.hlo()->shape(), hlo->tuple_index()),
      tuple.hlo(), hlo->tuple_index()));
  const auto source_sharding =
      tuple.sharding().GetSubSharding(tuple.base_shape(), {hlo->tuple_index()});
  gte->set_sharding(source_sharding);
  PartitionedHlo source_partitioned_gte(
      gte, tuple.base_shape().tuple_shapes(hlo->tuple_index()),
      MakePartitioningState());
  source_partitioned_gte = source_partitioned_gte.Reshard(hlo->sharding());
  SetPartitionedHlo(hlo, source_partitioned_gte);
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
    if (!ShapeUtil::Compatible(per_branch_partitioned_shapes[i], shard_shape)) {
      std::function<HloInstruction*(const ShapeIndex&, HloInstruction*)>
          pad_infeed = [&](const ShapeIndex& index,
                           HloInstruction* infeed_element) -> HloInstruction* {
        if (index == ShapeIndex({1})) {
          // Token.
          return infeed_element;
        }
        const Shape& element_shape =
            ShapeUtil::GetSubshape(infeed->shape(), index);
        if (element_shape.IsTuple() && element_shape.tuple_shapes_size() > 0) {
          std::vector<HloInstruction*> padded_elements(
              element_shape.tuple_shapes_size());
          for (int64 i = 0; i < padded_elements.size(); ++i) {
            auto sub_index = index;
            sub_index.push_back(i);
            padded_elements[i] = pad_infeed(
                sub_index,
                branch_b.AddInstruction(HloInstruction::CreateGetTupleElement(
                    ShapeUtil::GetSubshape(element_shape, {i}), infeed_element,
                    i)));
          }
          return branch_b.AddInstruction(
              HloInstruction::CreateTuple(padded_elements));
        }
        const Shape& pad_shape =
            ShapeUtil::GetSubshape(shard_shape, ShapeIndexView(index, 1));
        if (ShapeUtil::Compatible(element_shape, pad_shape)) {
          return infeed_element;
        }
        if (element_shape.IsArray()) {
          CHECK(pad_shape.IsArray());
          return PadToShape(infeed_element, pad_shape, &branch_b);
        }
        CHECK(element_shape.IsTuple());
        CHECK(element_shape.tuple_shapes().empty());
        return CreateZero(pad_shape, &branch_b);
      };
      pad_infeed({}, infeed);
    }
    branches[i] = module_->AddEmbeddedComputation(branch_b.Build());
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
  auto lhs = GetPartitionedHlo(hlo->operand(0));
  // Create a window config to represent the pad.
  Window window;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    const auto& pd = hlo->padding_config().dimensions(i);
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_window_reversal(false);
    dim->set_padding_low(pd.edge_padding_low());
    dim->set_padding_high(pd.edge_padding_high());
    dim->set_base_dilation(pd.interior_padding() + 1);
  }

  auto replicated_rhs = GetPartitionedHlo(hlo->operand(1))
                            .Reshard(HloSharding::Replicate())
                            .hlo();
  auto reshard_operand =
      lhs.ReshardAsWindowedInput(window, hlo->sharding(), replicated_rhs,
                                 /*mask_invalid_region=*/false);
  if (!reshard_operand.has_value()) {
    return DefaultAction(hlo);
  }
  PaddingConfig sharded_padding_config;
  bool need_pad = false;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    auto dim = sharded_padding_config.add_dimensions();
    const auto& wd = reshard_operand->shard_window.dimensions(i);
    dim->set_edge_padding_low(wd.padding_low());
    dim->set_edge_padding_high(wd.padding_high());
    dim->set_interior_padding(wd.base_dilation() - 1);
    if (wd.padding_low() != 0 || wd.padding_high() != 0 ||
        wd.base_dilation() != 1) {
      need_pad = true;
    }
  }
  auto sharded_pad = reshard_operand->sharded_input;
  if (need_pad) {
    TF_ASSIGN_OR_RETURN(auto sharded_pad_shape,
                        ShapeInference::InferPadShape(sharded_pad->shape(),
                                                      replicated_rhs->shape(),
                                                      sharded_padding_config));
    sharded_pad = b_.AddInstruction(hlo->CreatePad(sharded_pad_shape,
                                                   sharded_pad, replicated_rhs,
                                                   sharded_padding_config));
  }

  SetPartitionedHlo(hlo, [&]() {
    if (!reshard_operand->dynamic_slice_index_on_output) {
      return sharded_pad;
    }
    auto shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, sharded_pad,
        *reshard_operand->dynamic_slice_index_on_output,
        shard_shape.dimensions()));
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
  std::vector<int64> preserved_dims;
  for (int64 i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
    if (!absl::c_linear_search(hlo->dimensions(), i)) {
      preserved_dims.push_back(i);
    }
  }

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
      inputs.back() =
          inputs.back().PadWithValue(inits[operand_id], /*left_padded_dims=*/{},
                                     /*skipped_dims=*/preserved_dims);
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

  std::vector<HloInstruction*> input_hlos(input_count);
  for (int64 i = 0; i < input_count; ++i) {
    input_hlos[i] = inputs[i].hlo();
  }
  auto local_reduce = b_.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, input_hlos, inits, hlo->dimensions(), hlo->to_apply()));
  local_reduce->set_metadata(hlo->metadata());

  SetPartitionedHlo(hlo, [&]() {
    HloInstruction* reduce = local_reduce;
    const bool reduce_sharded_dimension =
        !inputs[0].sharding().IsTileMaximal() &&
        absl::c_any_of(hlo->dimensions(), [&](int64 i) {
          return inputs[0].sharding().tile_assignment().dim(i) > 1;
        });
    if (reduce_sharded_dimension) {
      if (inputs[0].sharding().ReplicateOnLastTileDim()) {
        preserved_dims.push_back(inputs[0].base_shape().rank());
      }
      if (local_reduce->shape().IsArray()) {
        reduce = partitioner_->AllReduceAlongShardingDims(
            &b_, local_reduce, inputs[0].sharding(), next_channel_id_,
            hlo->dimensions(), collective_ops_creator_, hlo->to_apply());
      } else {
        auto grouped =
            GroupShardingOnDims(inputs[0].sharding(), preserved_dims);
        auto grouped_state = CreatePerGroupPartitioningState(
            inputs[0].state(), grouped.device_groups, &b_);
        std::vector<HloInstruction*> all_gathered_partial_results(input_count);
        for (int64 i = 0; i < input_count; ++i) {
          auto gte = b_.AddInstruction(HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetTupleElementShape(reduce_shape, i), local_reduce,
              i));
          auto expanded_shape = input_hlos[i]->shape();
          auto all_gather_shape = input_hlos[i]->shape();
          for (int64 dim : hlo->dimensions()) {
            expanded_shape.set_dimensions(dim, 1);
            all_gather_shape.set_dimensions(
                dim, inputs[0].sharding().tile_assignment().dim(dim));
          }
          auto reshape = b_.AddInstruction(
              HloInstruction::CreateReshape(expanded_shape, gte));
          // Replicate per group.
          reshape->set_sharding(grouped.sharding);
          all_gathered_partial_results[i] =
              PartitionedHlo(reshape, all_gather_shape, grouped_state)
                  .Replicate()
                  .hlo();
        }
        reduce = b_.AddInstruction(HloInstruction::CreateReduce(
            reduce_shape, all_gathered_partial_results, inits,
            hlo->dimensions(), hlo->to_apply()));
      }
    }
    auto sharding = hlo_sharding_util::RemoveShapeDimensions(
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            inputs[0].sharding(), hlo->dimensions()),
        hlo->dimensions());
    if (local_reduce->shape().IsArray()) {
      reduce->set_sharding(sharding);
    } else {
      reduce->set_sharding(HloSharding::Tuple(
          reduce->shape(), std::vector<HloSharding>(input_count, sharding)));
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
  auto operand = GetPartitionedHlo(reverse->operand(0))
                     .Reshard(hlo_sharding_util::ReverseSharding(
                         reverse->sharding(), reverse->dimensions()));
  auto left_padded_operand =
      HaloExchangeToPadOnLeft(operand, reverse->dimensions());
  if (!left_padded_operand) {
    return DefaultAction(hlo);
  }
  SetPartitionedHlo(hlo, [&] {
    return b_.AddInstruction(hlo->CloneWithNewOperands(
        left_padded_operand->shape(), {left_padded_operand}));
  });
  return Status::OK();
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
  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }

  const auto& sharding = hlo->sharding();
  const Shape& shape = hlo->operand(0)->shape();
  auto partitioned_operand =
      GetPartitionedHlo(hlo->operand(0)).Reshard(sharding);
  const auto& shard_shape = partitioned_operand.hlo()->shape();
  const auto& operand = partitioned_operand.hlo();
  auto token = GetPartitionedHlo(hlo->operand(1)).hlo();

  if (EvenlyPartitions(shape, sharding)) {
    Shape outfeed_shape = operand->shape();
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(hlo->outfeed_shape(),
                                                           &outfeed_shape));
    SetPartitionedHlo(hlo, [&]() {
      return b_.AddInstruction(HloInstruction::CreateOutfeed(
          outfeed_shape, operand, token, hlo->outfeed_config()));
    });
    return Status::OK();
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

  // Get branch index for this partition.
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

  // Create conditional for the outfeed.
  std::vector<HloComputation*> branches(per_branch_partitioned_shapes.size());
  for (int64 i = 0; i < branches.size(); ++i) {
    SpmdBuilder branch_b(absl::StrCat("outfeed_branch_", i), visiting_hlo_);
    // Create tuple param within the branch.
    auto param = branch_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0,
        ShapeUtil::MakeTupleShape({operand->shape(), token->shape()}),
        "outfeed_token_param"));
    auto outfeed_data = branch_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(operand->shape(), param, 0));
    auto outfeed_token = branch_b.AddInstruction(
        HloInstruction::CreateGetTupleElement(token->shape(), param, 1));
    if (!ShapeUtil::Compatible(per_branch_partitioned_shapes[i], shard_shape)) {
      std::function<HloInstruction*(const ShapeIndex&, HloInstruction*)>
          slice_outfeed =
              [&](const ShapeIndex& index,
                  HloInstruction* outfeed_operand) -> HloInstruction* {
        // Get outfeed element shape.
        const Shape& element_shape =
            ShapeUtil::GetSubshape(outfeed_data->shape(), index);
        // Recursively call slice_outfeed for tuple shapes.
        if (element_shape.IsTuple() && element_shape.tuple_shapes_size() > 0) {
          std::vector<HloInstruction*> slice_elements(
              element_shape.tuple_shapes_size());
          for (int64 i = 0; i < slice_elements.size(); ++i) {
            auto sub_index = index;
            sub_index.push_back(i);
            slice_elements[i] = slice_outfeed(
                sub_index,
                branch_b.AddInstruction(HloInstruction::CreateGetTupleElement(
                    ShapeUtil::GetSubshape(element_shape, {i}), outfeed_operand,
                    i)));
          }
          return branch_b.AddInstruction(
              HloInstruction::CreateTuple(slice_elements));
        }
        // Get the slice shape.
        const Shape& slice_shape = ShapeUtil::GetSubshape(
            per_branch_partitioned_shapes[i], ShapeIndexView(index));
        if (ShapeUtil::Compatible(element_shape, slice_shape)) {
          return outfeed_operand;
        }
        // Slice out useful data.
        if (element_shape.IsArray()) {
          CHECK(slice_shape.IsArray());
          std::vector<int64> start_indices(slice_shape.rank(), 0);
          std::vector<int64> slice_strides(slice_shape.rank(), 1);
          return branch_b.AddInstruction(HloInstruction::CreateSlice(
              slice_shape, outfeed_operand, start_indices,
              slice_shape.dimensions(), slice_strides));
        }
        CHECK(element_shape.IsTuple());
        CHECK(element_shape.tuple_shapes().empty());
        return outfeed_operand;
      };
      outfeed_data = slice_outfeed({}, outfeed_data);
    }
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        hlo->outfeed_shape(), &per_branch_partitioned_shapes[i]));
    branch_b.AddInstruction(HloInstruction::CreateOutfeed(
        per_branch_partitioned_shapes[i], outfeed_data, outfeed_token,
        hlo->outfeed_config()));
    branches[i] = module_->AddEmbeddedComputation(branch_b.Build());
  }
  SetPartitionedHlo(hlo, [&]() {
    return b_.AddInstruction(HloInstruction::CreateConditional(
        token->shape(), branch_index, branches,
        std::vector<HloInstruction*>(
            branches.size(),
            b_.AddInstruction(HloInstruction::CreateTuple({operand, token})))));
  });
  return Status::OK();
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
  // Replicate the operands and run partitioned Rng on all devices.
  std::vector<HloInstruction*> new_operands;
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    new_operands.push_back(GetPartitionedHlo(hlo->operand(i))
                               .Reshard(HloSharding::Replicate())
                               .hlo());
  }

  if (!hlo->sharding().ReplicateOnLastTileDim()) {
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(HloInstruction::CreateRng(
          MakePartitionedShape(hlo->shape(), hlo->sharding()),
          hlo->random_distribution(), new_operands));
    });
  } else {
    std::vector<int64> group_dims(
        hlo->sharding().tile_assignment().num_dimensions() - 1);
    std::iota(group_dims.begin(), group_dims.end(), 0);
    auto sharding_grouped = GroupShardingOnDims(hlo->sharding(), group_dims);
    auto per_group_state = CreatePerGroupPartitioningState(
        MakePartitioningState(), sharding_grouped.device_groups, &b_);
    auto rng = b_.AddInstruction(HloInstruction::CreateRng(
        MakePartitionedShape(hlo->shape(), hlo->sharding()),
        hlo->random_distribution(), new_operands));
    rng->set_sharding(HloSharding::AssignDevice(0));
    SetPartitionedHlo(hlo, [&]() {
      return PartitionedHlo(rng, rng->shape(), per_group_state)
          .Replicate()
          .hlo();
    });
  }
  return Status::OK();
}

Status SpmdPartitioningVisitor::HandleReduceWindow(HloInstruction* hlo) {
  // TODO(b/73062247) Variadic reduce window not yet supported in partitioner.
  if (hlo->shape().IsTuple()) {
    return DefaultAction(hlo);
  }
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

StatusOr<bool> SpmdPartitioningVisitor::DoPartition(
    HloComputation* computation, const HloSharding& root_sharding,
    const SpmdPartitionerOptions& options) {
  VLOG(2) << "Partitioning computation " << computation->name() << " for "
          << num_replicas_ << " replicas and " << num_partitions_
          << " partitions";
  TF_RETURN_IF_ERROR(computation->Accept(this));

  HloModule* module = computation->parent();
  auto new_root =
      GetPartitionedHlo(computation->root_instruction()).Reshard(root_sharding);
  auto new_computation =
      module->AddEmbeddedComputation(b_.Build(new_root.hlo()));
  TF_RETURN_IF_ERROR(
      DoCodeMotionForWindowedDotGeneralLoops(new_computation, options));

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
      [num_replicas, num_partitions](
          SpmdBuilder* b, HloInstruction* operand, HloComputation* reduction,
          const std::vector<std::vector<int64>>& partition_subgroups,
          int64 channel_id) {
        if (partition_subgroups.size() <= 1) {
          std::vector<ReplicaGroup> groups(num_replicas);
          // TODO(yuanzx): Unify subgroup definition with AllToAll.
          for (int64 i = 0; i < num_replicas; ++i) {
            groups[i].add_replica_ids(i);
          }
          return b->AddInstruction(HloInstruction::CreateAllReduce(
              operand->shape(), {operand}, reduction, groups,
              /*constrain_layout=*/false, channel_id,
              /*use_global_device_ids=*/false));
        }

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
        return b->AddInstruction(HloInstruction::CreateAllReduce(
            operand->shape(), {operand}, reduction, device_groups,
            /*constrain_layout=*/false, channel_id,
            /*use_global_device_ids=*/true));
      },
      [](SpmdBuilder* b, HloInstruction* operand,
         std::vector<std::pair<int64, int64>>& src_dst_pairs,
         int64 channel_id) {
        return b->AddInstruction(HloInstruction::CreateCollectivePermute(
            operand->shape(), operand, src_dst_pairs, channel_id));
      },
      [](SpmdBuilder* b, absl::Span<HloInstruction* const> operands,
         const std::vector<std::vector<int64>>& partition_subgroups,
         int64 channel_id, absl::optional<int64> split_dimension) {
        std::vector<Shape> shapes(operands.size(), operands[0]->shape());
        const Shape output_shape = (shapes.size() == 1)
                                       ? shapes[0]
                                       : ShapeUtil::MakeTupleShape(shapes);
        std::vector<ReplicaGroup> groups(partition_subgroups.size());
        for (int64 i = 0; i < groups.size(); ++i) {
          for (int64 id : partition_subgroups[i]) {
            groups[i].add_replica_ids(id);
          }
        }
        return b->AddInstruction(HloInstruction::CreateAllToAll(
            output_shape, operands, groups,
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

HloInstruction* SpmdPartitioner::AllGatherShards(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64* next_channel_id, absl::Span<const int64> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator) {
  return AllGatherShardsInternal(b, operand, sharding, next_channel_id,
                                 selected_dims, collectives_creator,
                                 /*per_dim_ag=*/true);
}

HloInstruction* SpmdPartitioner::AllGatherShardsInternal(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64* next_channel_id, absl::Span<const int64> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator, bool per_dim_ag) {
  if (selected_dims.empty()) {
    return operand;
  }
  CHECK(!sharding.IsTileMaximal());
  // Add one leading dimension to gather all partitions.
  std::vector<int64> shape;
  shape.push_back(1);
  for (int64 dim : operand->shape().dimensions()) {
    shape.push_back(dim);
  }
  auto reshape = b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(operand->shape().element_type(), shape), operand));
  HloInstruction* result = reshape;
  if (per_dim_ag) {
    for (auto it = selected_dims.rbegin(); it != selected_dims.rend(); ++it) {
      if (sharding.tile_assignment().dim(*it) == 1) {
        continue;
      }
      auto partition_subgroups =
          GetPartitionGroupsForReplication(sharding, {*it});
      shape[0] *= partition_subgroups[0].size();
      result = collectives_creator.create_cross_partition_all_gather(
          b, result,
          ShapeUtil::MakeShape(operand->shape().element_type(), shape),
          partition_subgroups, (*next_channel_id)++,
          /*all_gather_dimension=*/0);
    }
  } else {
    auto partition_subgroups =
        GetPartitionGroupsForReplication(sharding, selected_dims);
    shape[0] *= partition_subgroups[0].size();
    result = collectives_creator.create_cross_partition_all_gather(
        b, result, ShapeUtil::MakeShape(operand->shape().element_type(), shape),
        partition_subgroups, (*next_channel_id)++,
        /*all_gather_dimension=*/0);
  }
  // If n > 1 dimensions are partitioned, split the leading dimension to n.
  std::vector<int64> tiled_dims;
  for (int64 i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (sharding.tile_assignment().dim(i) > 1 &&
        absl::c_linear_search(selected_dims, i)) {
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
    if (sharding.tile_assignment().dim(i - split_dims_added) == 1 ||
        !absl::c_linear_search(selected_dims, i - split_dims_added)) {
      xpose_permutation[i] = i + tiled_dims.size() - split_dims_added;
    } else {
      xpose_permutation[i] = split_dims_added;
      xpose_permutation[i + 1] = i + tiled_dims.size() - split_dims_added;
      split_dims_added++;
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

HloInstruction* SpmdPartitioner::AllReduceAlongShardingDims(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64* next_channel_id, absl::Span<const int64> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator,
    HloComputation* reduction) {
  return AllReduceAlongShardingDimsInternal(
      b, operand, sharding, next_channel_id, selected_dims, collectives_creator,
      reduction, /*per_dim_ar=*/true);
}

HloInstruction* SpmdPartitioner::AllReduceAlongShardingDimsInternal(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64* next_channel_id, absl::Span<const int64> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator,
    HloComputation* reduction, bool per_dim_ar) {
  if (!per_dim_ar) {
    auto partition_subgroups =
        GetPartitionGroupsForReplication(sharding, selected_dims);
    return collectives_creator.create_cross_partition_all_reduce(
        b, operand, reduction, partition_subgroups, (*next_channel_id)++);
  }
  auto result = operand;
  for (auto it = selected_dims.rbegin(); it != selected_dims.rend(); ++it) {
    if (sharding.tile_assignment().dim(*it) == 1) {
      continue;
    }
    auto partition_subgroups =
        GetPartitionGroupsForReplication(sharding, {*it});
    result = collectives_creator.create_cross_partition_all_reduce(
        b, result, reduction, partition_subgroups, (*next_channel_id)++);
  }
  return result;
}

StatusOr<bool> SpmdPartitioner::PartitionComputation(
    HloComputation* computation, const HloSharding& root_sharding,
    int64* next_channel_id, SpmdLogger* logger) {
  auto visitor =
      CreateVisitor(computation, num_partitions_, num_replicas_,
                    collective_ops_creator_, next_channel_id, logger, options_);
  return visitor->DoPartition(computation, root_sharding, options_);
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
  // Copy the root sharding since the partitioner visitor may temporarily change
  // the sharding to work around manual sharding.
  HloSharding root_sharding = entry_root->sharding();
  TF_ASSIGN_OR_RETURN(
      bool partition_changed,
      PartitionComputation(module->entry_computation(), root_sharding,
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
    pass.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
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
                     hlo->opcode() == HloOpcode::kInfeed ||
                     hlo->opcode() == HloOpcode::kOutfeed)
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
      } else if (!hlo->sharding().IsTileMaximal() &&
                 !hlo->sharding().IsManual()) {
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
