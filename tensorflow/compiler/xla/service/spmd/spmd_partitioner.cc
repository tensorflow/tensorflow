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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
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
#include "tensorflow/compiler/xla/service/spmd/custom_call_handler.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace spmd {

namespace {
using hlo_sharding_util::GroupedSharding;
}  // namespace

std::string SpmdLogger::MakeReport() {
  std::string report;
  absl::StrAppend(&report,
                  "\n\n***** SPMD memory during transformation *****\n");

  std::sort(entries_.begin(), entries_.end(),
            [](auto const& entry0, auto const& entry1) {
              return entry0.first > entry1.first;
            });
  for (int64_t i = 0;
       i < std::min<int64_t>(report_instruction_count_, entries_.size()); ++i) {
    absl::StrAppend(
        &report, "\n  ",
        tensorflow::strings::HumanReadableNumBytes(entries_[i].first), " : ",
        entries_[i].second, "\n");
  }

  return report;
}

void SpmdLogger::RegisterLogEntry(HloInstruction* hlo,
                                  const std::vector<HloInstruction*>& group) {
  if (disabled_) {
    return;
  }
  std::string report = hlo->ToString();
  int64_t max_value = -1;
  for (HloInstruction* inst : group) {
    if (!inst->shape().IsArray()) {
      continue;
    }
    max_value = std::max<int64_t>(max_value, ShapeSizeInBytes(inst->shape()));
    absl::StrAppend(&report, "     * ", inst->ToString(), "\n");
  }
  entries_.push_back(std::make_pair(max_value, report));
}

/* static */ std::string SpmdLogger::ReportBeforePartition(
    const HloModule& module, int64_t report_instruction_count) {
  std::string report;
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

/* static */ std::string SpmdLogger::ReportAfterPartition(
    const HloModule& module, int64_t report_instruction_count) {
  std::string report;
  absl::StrAppend(&report,
                  "\n\n***** SPMD memory usage after partition *****\n");
  absl::StrAppend(&report,
                  ReportMemoryUsage(
                      module, [](const HloInstruction* hlo) { return true; },
                      report_instruction_count));
  return report;
}

template <typename F>
/* static */ std::string SpmdLogger::ReportMemoryUsage(
    const HloModule& module, const F& filter,
    int64_t report_instruction_count) {
  std::string report;
  std::vector<HloInstruction*> instructions;
  instructions.reserve(module.instruction_count());

  for (auto computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (auto hlo : computation->instructions()) {
      if (!hlo->shape().IsArray() ||
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
    for (int64_t i = 0;
         i < std::min<int64_t>(report_instruction_count, insts->size()); ++i) {
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
Status ClearShardingAttributes(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* hlo : computation->instructions()) {
      // Keep sharding annotation on Infeed and entry parameters since they're
      // used by HloReplicationAnalysis later (for ArCrsCombiner).
      if (hlo->HasSideEffect()) {
        continue;
      }
      if (hlo->opcode() == HloOpcode::kParameter &&
          computation == module->entry_computation()) {
        continue;
      }
      hlo->clear_sharding();
    }
  }
  return OkStatus();
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

// Returns a sharding that is replicated on all the dimensions where the given
// window is not unary.
HloSharding GetShardingReplicatedOnWindowedDimension(
    const HloSharding& sharding, const Window& window) {
  std::vector<int64_t> dimensions_to_replicate;
  for (int i = 0; i < window.dimensions_size(); ++i) {
    const WindowDimension& wd = window.dimensions(i);
    if (window_util::IsTrivialWindowDimension(wd)) {
      continue;
    }
    dimensions_to_replicate.push_back(i);
  }
  return hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
      sharding, dimensions_to_replicate);
}

}  // namespace

HloInstruction* SpmdBuilder::AddInstruction(
    std::unique_ptr<HloInstruction> instruction) {
  HloInstruction* hlo =
      HloComputation::Builder::AddInstruction(std::move(instruction));
  if (visiting_hlo_) {
    hlo->set_metadata(visiting_hlo_->metadata());
    instructions_[visiting_hlo_].push_back(hlo);
  }
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
      if (!absl::c_linear_search(hlo->dimensions(), i)) {
        broadcast_dims_[hlo].insert(i);
      }
    }
  }
  if (hlo->IsElementwise() && hlo->operand_count() > 0 &&
      // Copy can have a tuple result.
      hlo->shape().IsArray()) {
    absl::flat_hash_set<int64_t> broadcast_dims;
    for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
      broadcast_dims.insert(i);
    }
    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      auto it = broadcast_dims_.find(hlo->operand(i));
      if (it == broadcast_dims_.end()) {
        broadcast_dims.clear();
        break;
      }
      for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
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
      absl::flat_hash_set<int64_t> xpose_broadcast_dims;
      std::vector<int64_t> reverse_map(hlo->shape().rank());
      for (int64_t i = 0; i < reverse_map.size(); ++i) {
        reverse_map[hlo->dimensions(i)] = i;
      }
      for (int64_t dim : it->second) {
        xpose_broadcast_dims.insert(reverse_map[dim]);
      }
      broadcast_dims_[hlo] = std::move(xpose_broadcast_dims);
    }
  }
  if (hlo->opcode() == HloOpcode::kReshape &&
      Product(hlo->shape().dimensions()) > 0) {
    auto it = broadcast_dims_.find(hlo->operand(0));
    if (it != broadcast_dims_.end()) {
      absl::flat_hash_set<int64_t> reshape_broadcast_dims;
      for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
        reshape_broadcast_dims.insert(i);
      }
      std::vector<int64_t> before_dim_size_stack;
      std::vector<int64_t> after_dim_size_stack;
      const int64_t operand0_rank = hlo->operand(0)->shape().rank();
      const int64_t hlo_shape_rank = hlo->shape().rank();
      before_dim_size_stack.reserve(operand0_rank);
      after_dim_size_stack.reserve(hlo_shape_rank);
      for (int64_t i = operand0_rank - 1; i >= 0; --i) {
        before_dim_size_stack.push_back(hlo->operand(0)->shape().dimensions(i));
      }
      for (int64_t i = hlo_shape_rank - 1; i >= 0; --i) {
        after_dim_size_stack.push_back(hlo->shape().dimensions(i));
      }
      while (!before_dim_size_stack.empty() && !after_dim_size_stack.empty()) {
        int64_t before_size = before_dim_size_stack.back();
        int64_t after_size = after_dim_size_stack.back();
        int64_t current_before_dim =
            hlo->operand(0)->shape().rank() - before_dim_size_stack.size();
        int64_t current_after_dim =
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
          for (int64_t i = current_after_dim; i < hlo->shape().rank(); ++i) {
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
      absl::flat_hash_set<int64_t> pad_broadcast_dims;
      for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
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

PartitionedHlo PartitionedHlo::Reshard(const HloSharding& target,
                                       std::optional<Literal> pad_value) {
  if (sharding() == target) {
    return *this;
  }
  auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
  // Replace existing reshard cache for target if we are sharding with new
  // padding value.
  const bool replace_cache = pad_value.has_value();
  const bool is_to_replicate =
      hlo_->shape().IsArray() && target.NumTiles() < sharding().NumTiles();
  const bool use_cache =
      !is_to_replicate || state_.partitioner->options().cache_all_gather;
  if (!replace_cache && use_cache) {
    auto it = cache.find(target);
    if (it != cache.end()) {
      return it->second;
    }
  }

  auto resharded = ReshardNoCache(target, std::move(pad_value));
  // Update cache for resharded hlo.
  {
    auto& cache =
        state_.reshard_cache->per_hlo_cache[resharded.hlo()].reshard_cache;
    cache.insert_or_assign(sharding(), *this);
  }
  // Update cache for to-reshard hlo.
  if (use_cache) {
    // Get the cache again as it might be invalidated by the insertion above.
    auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
    auto [it, _] = cache.insert_or_assign(target, std::move(resharded));
    return it->second;
  }
  return resharded;
}

PartitionedHlo PartitionedHlo::ReshardNoCache(const HloSharding& target,
                                              std::optional<Literal> pad_value,
                                              bool allow_full_replication) {
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
    return Reshard(target.GetTupleSharding(shape).value());
  }

  // For a tuple shape, recursively apply Reshard to all the leaves and return
  // a tuple instruction.
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> elements;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
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

  CHECK_EQ(target.IsManualSubgroup(), sharding().IsManualSubgroup());
  if (sharding().IsManualSubgroup()) {
    auto grouped = hlo_sharding_util::GetManualSubgroupSharding(sharding());
    auto target_grouped = AlignGroupsWithIfCompatible(
        hlo_sharding_util::GetManualSubgroupSharding(target), grouped);
    CHECK(target_grouped.has_value())
        << "Resharding target has incompatible sharding subgroups. From "
        << sharding().ToString() << " to " << target.ToString();
    HloSharding original_sharding = sharding();
    hlo_->set_sharding(grouped.sharding);
    HloInstruction* partitioned =
        PartitionedHlo(hlo_, base_shape_,
                       CreatePerGroupPartitioningState(
                           state(), grouped.device_groups, state_.b))
            .ReshardNoCache(target_grouped->sharding)
            .hlo();
    hlo_->set_sharding(original_sharding);
    partitioned->set_sharding(target);
    return PartitionedHlo(partitioned, base_shape_, state_);
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
    if (!target.IsReplicated()) {
      auto reshard = TryComplexReshardHandling(target);
      if (reshard.has_value()) {
        return reshard.value();
      }
      if (!allow_full_replication) {
        return *this;
      }
      LOG(ERROR)
          << "[spmd] Involuntary full rematerialization. The compiled was "
             "not able to go from sharding "
          << sharding().ToString(/*include_metadata=*/true) << " to "
          << target.ToString(/*include_metadata=*/true)
          << " without doing a full rematerialization of the tensor. You "
             "probably want to enrich the sharding annotations to prevent "
             "this from happening.";
    }
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
    std::vector<int64_t> group_dims(target.tile_assignment().num_dimensions() -
                                    1);
    std::iota(group_dims.begin(), group_dims.end(), 0);
    auto target_grouped =
        hlo_sharding_util::GroupShardingOnDims(target, group_dims);
    auto partially_sharded = PerGroupSliceFromReplicated(
        hlo_, state_.partition_id, target_grouped.device_groups, group_dims,
        target_grouped.group_dim_sizes, state_.b);
    partially_sharded->set_sharding(target);
    return PartitionedHlo(partially_sharded, base_shape(), state_);
  }

  // 'Replicated' to 'Tiled'.
  auto padded_hlo = PadBaseShapeBeforeUnevenTiledSharding(
      hlo_, target, state_.b, std::move(pad_value));
  auto shard_shape = MakePartitionedShape(shape, target);
  auto slice = state_.b->AddInstruction(HloInstruction::CreateDynamicSlice(
      shard_shape, padded_hlo,
      MakePartitionOffsets(shape, target, state_.partition_id, state_.b),
      shard_shape.dimensions()));
  slice->set_sharding(target);
  return PartitionedHlo(slice, base_shape_, state_);
}

PartitionedHlo PartitionedHlo::PadWithValue(
    HloInstruction* pad_value, absl::Span<const int64_t> left_padded_dims,
    absl::Span<const int64_t> skipped_dims) const {
  HloInstruction* result =
      PadWithValueHlo(pad_value, left_padded_dims, skipped_dims);
  if (hlo_ != result) {
    result->set_sharding(hlo_->sharding());
  }
  return PartitionedHlo(result, base_shape_, state_);
}

HloInstruction* PartitionedHlo::PadWithValueHlo(
    HloInstruction* pad_value, absl::Span<const int64_t> left_padded_dims,
    absl::Span<const int64_t> skipped_dims) const {
  const HloSharding& sharding = hlo_->sharding();
  const Shape& shape = hlo_->shape();
  CHECK(!shape.IsTuple() && shape.element_type() != TOKEN);
  if (sharding.IsReplicated() || EvenlyPartitions(base_shape_, sharding)) {
    return hlo_;
  }
  CHECK(!sharding.IsTileMaximal());
  auto index_shape = ShapeUtil::ChangeElementType(shape, S32);
  auto mask_shape = ShapeUtil::ChangeElementType(index_shape, PRED);
  auto get_mask_for_dim = [&](int64_t dim, HloInstruction* start_index) {
    // Comparison: iota + start_index < valid_size
    auto iota =
        state_.b->AddInstruction(HloInstruction::CreateIota(index_shape, dim));
    auto broadcast_start_index = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(index_shape, start_index, {}));
    auto index_in_full_shape =
        state_.b->AddInstruction(HloInstruction::CreateBinary(
            index_shape, HloOpcode::kAdd, iota, broadcast_start_index));
    ComparisonDirection direction = ComparisonDirection::kLt;
    int64_t index_limit = base_shape_.dimensions(dim);
    if (absl::c_linear_search(left_padded_dims, dim)) {
      direction = ComparisonDirection::kGe;
      index_limit =
          index_shape.dimensions(dim) * sharding.tile_assignment().dim(dim) -
          index_limit;
    }
    auto limit = state_.b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(index_limit)));
    auto broadcast_limit = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(index_shape, limit, {}));
    return state_.b->AddInstruction(HloInstruction::CreateCompare(
        mask_shape, index_in_full_shape, broadcast_limit, direction));
  };

  HloInstruction* mask = nullptr;
  auto offsets = MakePartitionOffsets(base_shape_, sharding,
                                      state_.partition_id, state_.b);
  for (int64_t i = 0; i < shape.rank(); ++i) {
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
    return hlo_;
  }

  auto broadcast_pad_value = state_.b->AddInstruction(
      HloInstruction::CreateBroadcast(shape, pad_value, {}));
  return state_.b->AddInstruction(HloInstruction::CreateTernary(
      shape, HloOpcode::kSelect, mask, hlo_, broadcast_pad_value));
}

PartitionedHlo PartitionedHlo::PadWithZero(
    absl::Span<const int64_t> left_padded_dims,
    absl::Span<const int64_t> skipped_dims) const {
  auto zero = state_.b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(hlo_->shape().element_type())));
  return PadWithValue(zero, left_padded_dims, skipped_dims);
}

std::optional<PartitionedHlo::WindowedInputShardReturnValue>
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
  Shape padded_shape = base_shape_;
  std::vector<HloInstruction*> offsets_on_padded_shape(base_shape_.rank());
  std::vector<int64_t> per_shard_window_counts(base_shape_.rank());
  std::vector<int64_t> explicit_left_padding(base_shape_.rank());
  // Track if any shards can be skipped.
  std::vector<int64_t> trimmed_target_sharding_tile_shape(base_shape_.rank());
  // There can be at most 2 ranges of skipped shards on a dimension: 1) on the
  // right side, 2) in the middle. The following vector tracks the middle range
  // (i.e., <start, size>). The leftmost shard must not be skipped because
  // outputs are left-aligned.
  std::vector<std::pair<int64_t, int64_t>> trimmed_target_sharding_middle_range(
      base_shape_.rank(), std::pair<int64_t, int64_t>(-1, -1));
  bool trimmed_shards = false;
  std::vector<int64_t> dims_needs_pre_masking;
  Shape halo_exchange_base_shape = base_shape_;
  // If all useful input data are in a single shard, we can skip in-shard data
  // (e.g., those that belong to negative padding) via a local slice.
  bool trimmed_in_shard = false;
  std::vector<int64_t> pre_halo_exchange_slice_starts(base_shape_.rank(), 0);
  std::vector<int64_t> pre_halo_exchange_slice_limits(
      hlo_->shape().dimensions().begin(), hlo_->shape().dimensions().end());
  std::vector<bool> can_leave_dimension_partitioned(base_shape_.rank(), false);
  for (int64_t i = 0; i < base_shape_.rank(); ++i) {
    can_leave_dimension_partitioned[i] =
        window_util::IsTrivialWindowDimension(window.dimensions(i));
  }
  for (int64_t i = 0; i < base_shape_.rank(); ++i) {
    // Do not pad non-partitioned dimensions.
    int64_t shard_count = target.tile_assignment().dim(i);
    trimmed_target_sharding_tile_shape[i] = shard_count;
    if (shard_count == 1 || can_leave_dimension_partitioned[i]) {
      offsets_on_padded_shape[i] = state_.b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
      shard_shape.set_dimensions(
          i, CeilOfRatio(base_shape_.dimensions(i), shard_count));
      continue;
    }
    const WindowDimension& wd = window.dimensions(i);
    WindowDimension* swd = shard_window.mutable_dimensions(i);
    const int64_t dilated_size = 1 + (wd.size() - 1) * wd.window_dilation();
    const int64_t full_size =
        1 + (base_shape_.dimensions(i) - 1) * wd.base_dilation() +
        wd.padding_high() + wd.padding_low();
    int64_t window_count = (full_size - dilated_size) / wd.stride() + 1;
    per_shard_window_counts[i] = CeilOfRatio(window_count, shard_count);
    // Find skippable shards on the right side. This could only happen when
    // window_count < shard_count so that the right-most shard does not have any
    // output.
    int64_t input_shard_size = hlo_->shape().dimensions(i);
    if (window_count < shard_count && wd.window_dilation() == 1 &&
        wd.base_dilation() == 1) {
      // Test if some shards do not have any useful input (all uneven padding or
      // window negative padding).
      int64_t useful_input_shards = CeilOfRatio(
          base_shape_.dimensions(i) + wd.padding_high(), input_shard_size);
      if (useful_input_shards < shard_count) {
        shard_count = std::max<int64_t>(useful_input_shards, window_count);
        trimmed_shards = true;
        trimmed_target_sharding_tile_shape[i] = shard_count;
        if (shard_count == 1) {
          offsets_on_padded_shape[i] = state_.b->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
          swd->set_padding_high(base_shape_.dimensions(i) + wd.padding_high() -
                                hlo_->shape().dimensions(i));
          continue;
        }
        // Make sure the halo-exchange base shape is evenly sharded on the new
        // shard count.
        halo_exchange_base_shape.set_dimensions(i,
                                                input_shard_size * shard_count);
        if (input_shard_size * shard_count > base_shape_.dimensions(i) &&
            wd.padding_high() > 0) {
          // The new shape has paddings, make sure it's masked.
          dims_needs_pre_masking.push_back(i);
        } else if (wd.padding_high() < 0 &&
                   full_size - wd.padding_low() < input_shard_size) {
          // If the useful input is smaller than a shard, we treat the shard
          // size as the useful input size and slice later.
          input_shard_size = full_size - wd.padding_low();
          halo_exchange_base_shape.set_dimensions(
              i, input_shard_size * shard_count);
          pre_halo_exchange_slice_limits[i] = input_shard_size;
          trimmed_in_shard = true;
        }
      }
    }

    // We use explicit padding for full dilations, then use padding_low and
    // padding_high on the sharded op for the remaining. padding_low and
    // padding_high are now given initial values, which will be later updated if
    // dilation is not 1.
    explicit_left_padding[i] = wd.padding_low() / wd.base_dilation();
    swd->set_padding_low(wd.padding_low() % wd.base_dilation());
    swd->set_padding_high(0);

    // Find potential skippable range in the middle. This could happen when only
    // a few shards have outputs (window_count < shard_count), but there is a
    // large negative left padding such that the start shard that has useful
    // input does not have any outputs.
    if (window_count < shard_count && wd.window_dilation() == 1 &&
        wd.base_dilation() == 1) {
      int64_t middle_empty_shards =
          (-explicit_left_padding[i]) / input_shard_size - window_count;
      if (middle_empty_shards > 0) {
        shard_count -= middle_empty_shards;
        CHECK_GT(shard_count, 1);
        trimmed_target_sharding_middle_range[i].first = window_count;
        trimmed_target_sharding_middle_range[i].second = middle_empty_shards;
        trimmed_shards = true;
        trimmed_target_sharding_tile_shape[i] = shard_count;
        // Reduce negative padding.
        explicit_left_padding[i] += middle_empty_shards * input_shard_size;
        halo_exchange_base_shape.set_dimensions(i,
                                                input_shard_size * shard_count);
        HloInstruction* ordinal = partition_ordinals[i];
        HloInstruction* left_count = CreateR0WithType<int32_t>(
            ordinal->shape().element_type(), window_count, state_.b);
        HloInstruction* on_left =
            state_.b->AddInstruction(HloInstruction::CreateCompare(
                ShapeUtil::ChangeElementType(ordinal->shape(), PRED), ordinal,
                left_count, ComparisonDirection::kLt));
        HloInstruction* right_ordinal =
            state_.b->AddInstruction(HloInstruction::CreateBinary(
                ordinal->shape(), HloOpcode::kSubtract, ordinal, left_count));
        partition_ordinals[i] =
            state_.b->AddInstruction(HloInstruction::CreateTernary(
                partition_ordinals[i]->shape(), HloOpcode::kSelect, on_left,
                partition_ordinals[i], right_ordinal));
        if (-explicit_left_padding[i] > input_shard_size * (shard_count - 1)) {
          // If all useful data is on the last shard, we can skip extra negative
          // left padding.
          int64_t skip_amount =
              -explicit_left_padding[i] - input_shard_size * (shard_count - 1);
          input_shard_size -= skip_amount;
          explicit_left_padding[i] += skip_amount * shard_count;
          pre_halo_exchange_slice_starts[i] = skip_amount;
          trimmed_in_shard = true;
          // We may have enabled a new skipping opportunity on the right side
          // within the only shard that has useful input, because we excluded
          // negative left padding regions this time.
          if (full_size < input_shard_size) {
            skip_amount = input_shard_size - full_size;
            pre_halo_exchange_slice_limits[i] -= skip_amount;
            explicit_left_padding[i] += skip_amount * (shard_count - 1);
            input_shard_size = full_size;
          }
          halo_exchange_base_shape.set_dimensions(
              i, input_shard_size * shard_count);
        }
      }
    }
    if (full_size < dilated_size) {
      VLOG(2) << "Failed to reshard window operand because the window size is "
                 "larger than padded base size";
      return std::nullopt;
    }
    if (wd.stride() != 1 &&
        (wd.stride() * per_shard_window_counts[i]) % wd.base_dilation() != 0) {
      // TODO(yuanzx): Support this case.
      VLOG(2) << "Failed to reshard window operand due to non-trivial dilation";
      return std::nullopt;
    }

    // Calculation for the first element needed on the 'padded-but-not-dilated'
    // shape. The start on the dilated shape could be a hole, so we add
    // wd.base_dilation() - 1 to the constant term to skip the leading holes.
    start_on_padded_calculations[i] = MultiplyAddDivideOffsetCalculation(
        wd.stride() * per_shard_window_counts[i],
        wd.base_dilation() - 1 - swd->padding_low(), wd.base_dilation());
    int64_t dilated_shard_size =
        wd.stride() * (per_shard_window_counts[i] - 1) + dilated_size;
    limit_on_padded_calculations[i] = MultiplyAddDivideOffsetCalculation(
        wd.stride() * per_shard_window_counts[i],
        dilated_shard_size + wd.base_dilation() - 1 - swd->padding_low(),
        wd.base_dilation());

    offsets_on_padded_shape[i] = start_on_padded_calculations[i].Calculate(
        partition_ordinals[i], state_.b);

    auto shard_size_function =
        limit_on_padded_calculations[i] - start_on_padded_calculations[i];
    int64_t max_shard_size = shard_size_function.MaxInRange(0, shard_count);
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
          [&](int64_t shard_ordinal) {
            return start_on_padded_calculations[i].Calculate(shard_ordinal) *
                       wd.base_dilation() +
                   swd->padding_low() -
                   wd.stride() * per_shard_window_counts[i] * shard_ordinal;
          };
      CHECK_EQ(get_first_valid_element_offset_on_dilated_shard(0),
               swd->padding_low());

      // Determine swd->padding_high.
      for (int64_t shard_ordinal = 0; shard_ordinal < shard_count;
           ++shard_ordinal) {
        int64_t wanted_limit_on_dilated_shard =
            wd.stride() * (per_shard_window_counts[i] - 1) + dilated_size;
        int64_t actual_limit_on_dilated_shard_without_pad_high =
            get_first_valid_element_offset_on_dilated_shard(shard_ordinal) +
            (max_shard_size - 1) * wd.base_dilation() + 1;
        swd->set_padding_high(std::max<int64_t>(
            swd->padding_high(),
            wanted_limit_on_dilated_shard -
                actual_limit_on_dilated_shard_without_pad_high));
      }

      // Determine swd->padding_low and output dynamic slice index.
      if (wd.stride() == 1) {
        int64_t max_pad_low =
            get_first_valid_element_offset_on_dilated_shard(0);
        bool all_same = true;
        for (int64_t shard_ordinal = 1; shard_ordinal < shard_count;
             ++shard_ordinal) {
          int64_t start =
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
          return std::nullopt;
        }
        // padding_low on all shards should equal the initially assigned
        // swd->padding_low(), i.e., the padding_low() on the original window.
      }
    }
  }

  // Returns the output dynamic slice offset when needed, and std::nullopt
  // otherwise.
  auto get_dynamic_slice_offset_on_output_if_needed =
      [&]() -> std::optional<std::vector<HloInstruction*>> {
    if (absl::c_all_of(
            dynamic_slice_offset_on_output,
            [](HloInstruction* offset) { return offset == nullptr; })) {
      return std::nullopt;
    }
    auto zero = state_.b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
    for (int64_t i = 0; i < dynamic_slice_offset_on_output.size(); ++i) {
      if (dynamic_slice_offset_on_output[i] == nullptr) {
        dynamic_slice_offset_on_output[i] = zero;
      }
    }
    return dynamic_slice_offset_on_output;
  };

  auto handle_all_windowed_dimensions_are_replicated = [&]() {
    PaddingConfig padding_config;
    auto pad_hlo_shape = padded_shape;
    for (int64_t i = 0; i < base_shape_.rank(); ++i) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_interior_padding(0);
      // Do not pad non-partitioned dimensions or partitioned dimensions that
      // are already sharded in a way that where the windowed sharding matches
      // the sharding we want.
      if (target.tile_assignment().dim(i) == 1 ||
          can_leave_dimension_partitioned[i]) {
        padding_config_dim->set_edge_padding_low(0);
        padding_config_dim->set_edge_padding_high(0);
        pad_hlo_shape.set_dimensions(i, hlo_->shape().dimensions(i));
        continue;
      }
      padding_config_dim->set_edge_padding_low(explicit_left_padding[i]);
      padding_config_dim->set_edge_padding_high(padded_shape.dimensions(i) -
                                                explicit_left_padding[i] -
                                                base_shape_.dimensions(i));
    }
    auto padded_hlo =
        ShapeUtil::Compatible(pad_hlo_shape, base_shape_)
            ? hlo_
            : state_.b->AddInstruction(HloInstruction::CreatePad(
                  pad_hlo_shape, hlo_, pad_value, padding_config));
    auto sharded_input =
        state_.b->AddInstruction(HloInstruction::CreateDynamicSlice(
            shard_shape, padded_hlo, offsets_on_padded_shape,
            shard_shape.dimensions()));
    return update_cache(WindowedInputShardReturnValue{
        sharded_input, shard_window,
        get_dynamic_slice_offset_on_output_if_needed()});
  };

  auto sharding_with_non_windowed_dims_replicated =
      GetShardingReplicatedOnWindowedDimension(target, window);
  // If the currrent HLO is replicated or all windows dimensions are replicated,
  // pad then slice. If the target sharding and current sharding are not the
  // same then give the halo exchange system a chance to run as it can skip
  // generating a dynamic slice.
  if (sharding().IsReplicated() ||
      (target != sharding() &&
       sharding_with_non_windowed_dims_replicated == sharding())) {
    return handle_all_windowed_dimensions_are_replicated();
  }
  if (target != sharding() &&
      sharding_with_non_windowed_dims_replicated != sharding()) {
    return Reshard(target).ReshardAsWindowedInput(window, target, pad_value);
  }
  if (Product(trimmed_target_sharding_tile_shape) == 1) {
    // The trimmed sharding may have just one shard left. We can simply return
    // hlo_ in this case.
    return update_cache(WindowedInputShardReturnValue{
        hlo_, shard_window, get_dynamic_slice_offset_on_output_if_needed()});
  }
  if (target.ReplicateOnLastTileDim()) {
    trimmed_target_sharding_tile_shape.push_back(
        target.tile_assignment().dimensions().back());
  }
  std::optional<HloSharding> trimmed_target;
  const HloSharding* halo_exchange_target = &target;
  if (trimmed_shards) {
    // Remove devices on the right side.
    Array<int64_t> trimmed_devices(trimmed_target_sharding_tile_shape);
    trimmed_devices.Each([&](absl::Span<const int64_t> indices, int64_t* d) {
      std::vector<int64_t> target_indices(indices.begin(), indices.end());
      for (int64_t i = 0; i < base_shape_.rank(); ++i) {
        const auto& range = trimmed_target_sharding_middle_range[i];
        if (range.first >= 0 && indices[i] >= range.first) {
          target_indices[i] += range.second;
        }
      }
      *d = target.tile_assignment()(target_indices);
    });
    trimmed_target = target.ReplicateOnLastTileDim()
                         ? HloSharding::PartialTile(trimmed_devices)
                         : HloSharding::Tile(trimmed_devices);
    halo_exchange_target = &*trimmed_target;
  }

  // Halo exchange.
  HloInstruction* visiting_hlo = hlo_;

  if (!dims_needs_pre_masking.empty()) {
    std::vector<int64_t> skipped_dims;
    for (int dim = 0; dim < base_shape_.rank(); ++dim) {
      if (!absl::c_linear_search(dims_needs_pre_masking, dim)) {
        skipped_dims.push_back(dim);
      }
    }
    visiting_hlo = PadWithValueHlo(pad_value, /*left_padded_dims=*/{},
                                   /*skipped_dims=*/skipped_dims);
  }

  // If we skipped unused data within a shard, we need to slice the input shard.
  if (trimmed_in_shard) {
    std::vector<int64_t> slice_sizes(halo_exchange_base_shape.rank());
    for (int64_t i = 0; i < slice_sizes.size(); ++i) {
      slice_sizes[i] =
          pre_halo_exchange_slice_limits[i] - pre_halo_exchange_slice_starts[i];
    }
    visiting_hlo = state_.b->AddInstruction(HloInstruction::CreateSlice(
        ShapeUtil::MakeShape(halo_exchange_base_shape.element_type(),
                             slice_sizes),
        visiting_hlo,
        /*start_indices=*/pre_halo_exchange_slice_starts,
        /*limit_indices=*/pre_halo_exchange_slice_limits,
        /*strides=*/
        std::vector<int64_t>(halo_exchange_base_shape.rank(), 1)));
  }

  std::vector<OffsetCalculation> left_halo_size_functions(base_shape_.rank());
  std::vector<OffsetCalculation> right_halo_size_functions(base_shape_.rank());
  // TODO(yuanzx): We are concatenating on each sharded dimension one at time,
  // and in the second dimension (and beyond) we create halos by slicing the
  // concat in the previous dimension, which is not optimal. We should generate
  // halos only concating slices, instead of slicing concats.
  for (int dim = 0; dim < base_shape_.rank(); ++dim) {
    int64_t shard_count = halo_exchange_target->tile_assignment().dim(dim);
    if (shard_count == 1 || can_leave_dimension_partitioned[dim]) {
      continue;
    }
    int64_t input_shard_size =
        CeilOfRatio(halo_exchange_base_shape.dimensions(dim), shard_count);

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
        visiting_hlo, halo_exchange_base_shape, left_halo_size_functions[dim],
        right_halo_size_functions[dim], explicit_left_padding[dim],
        padded_shape.dimensions(dim), shard_shape.dimensions(dim), dim,
        *halo_exchange_target, offsets_on_padded_shape[dim], pad_value,
        partition_ordinals[dim], state_.collective_ops_creator,
        state_.next_channel_id, state_.b, mask_invalid_region);
    if (!resharded) {
      VLOG(1) << "ReshardAsWindowedInput failed without replicate first: halo "
                 "is beyond the neighbor.";
      // If we are already sharded in such a way that all windowed dimensions
      // are replicated then just handle it with pad + slice.
      if (sharding_with_non_windowed_dims_replicated == sharding()) {
        return handle_all_windowed_dimensions_are_replicated();
      }
      return Reshard(sharding_with_non_windowed_dims_replicated)
          .ReshardAsWindowedInput(window, target, pad_value);
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
  // Do not use a reference as the HLO's sharding can be temporarily replaced.
  const HloSharding sharding = hlo_->sharding();
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
        .reshard_cache.insert_or_assign(sharding, *this);
    // Get the cache again as it might be invalidated by the insertion above.
    auto& cache = state_.reshard_cache->per_hlo_cache[hlo()].reshard_cache;
    if (state_.partitioner->options().cache_all_gather) {
      auto [it, _] = cache.insert_or_assign(HloSharding::Replicate(),
                                            std::move(resharded));
      return it->second;
    }
    return resharded;
  };
  // 'Single Device' to 'Repliated'.
  if (sharding.IsTileMaximal()) {
    return update_cache(Broadcast());
  }

  // 'Tiled' to 'Replicated'.
  std::vector<int64_t> all_dims(shape.rank());
  std::iota(all_dims.begin(), all_dims.end(), 0);
  HloInstruction* result = ReplicatePartial(all_dims);
  result->set_sharding(HloSharding::Replicate());
  return update_cache(PartitionedHlo(result, base_shape_, state_));
}

HloInstruction* PartitionedHlo::ReplicatePartial(
    absl::Span<const int64_t> dims) {
  CHECK(!sharding().IsTileMaximal());
  const Shape& shard_shape = hlo()->shape();
  Shape target_shape = shard_shape;
  Shape padded_target_shape = shard_shape;
  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> dus_ar_dims;
  std::vector<int64_t> ag_dims;
  // Find dimensions that can be replicated with Broadcast() (shard size 1) and
  // others that need all-gather. dus_ar_dims is a generalization of
  // broadcast_dims where the full size is less than half of allgather size, and
  // we will use dus->allreduce on them.
  for (int64_t i : dims) {
    int64_t partitions = sharding().tile_assignment().dim(i);
    if (partitions == 1) {
      continue;
    }
    target_shape.set_dimensions(i, base_shape().dimensions(i));
    if (target_shape.dimensions(i) == shard_shape.dimensions(i)) {
      broadcast_dims.push_back(i);
    } else if (target_shape.dimensions(i) <= partitions / 2) {
      dus_ar_dims.push_back(i);
    } else {
      padded_target_shape.set_dimensions(
          i, shard_shape.dimensions(i) * partitions);
      ag_dims.push_back(i);
    }
  }

  HloInstruction* broadcast = hlo_;
  if (!broadcast_dims.empty()) {
    std::vector<int64_t> other_dims;
    for (int64_t i = 0; i < sharding().tile_assignment().num_dimensions();
         ++i) {
      if (!absl::c_linear_search(broadcast_dims, i)) {
        other_dims.push_back(i);
      }
    }
    HloSharding original_sharding = sharding();
    auto grouped =
        hlo_sharding_util::GroupShardingOnDims(original_sharding, other_dims);
    std::vector<int64_t> dev_indices(
        grouped.sharding.tile_assignment().num_dimensions(), 0);
    hlo_->set_sharding(HloSharding::AssignDevice(
        grouped.sharding.tile_assignment()(dev_indices)));
    auto per_group_partitioner_state = CreatePerGroupPartitioningState(
        state(), grouped.device_groups, state().b);
    auto partial_replicate_hlo =
        PartitionedHlo(hlo_, shard_shape, per_group_partitioner_state)
            .Broadcast();
    hlo_->set_sharding(original_sharding);
    partial_replicate_hlo.hlo()->clear_sharding();
    broadcast = partial_replicate_hlo.hlo();
  }

  if (ag_dims.empty() && dus_ar_dims.empty()) {
    return broadcast;
  }

  HloInstruction* result = nullptr;
  if (state_.collective_ops_creator.create_cross_partition_all_gather) {
    result = state_.partitioner->AllGatherShards(
        state_.b, broadcast, sharding(), state_.next_channel_id, ag_dims,
        state_.collective_ops_creator);
  }
  // May also contain failed allgather dims.
  if (result == nullptr) {
    for (int64_t dim : ag_dims) {
      dus_ar_dims.push_back(dim);
    }
    result = broadcast;
  }
  if (!dus_ar_dims.empty()) {
    auto zero = state_.b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(shard_shape.element_type())));
    std::vector<int64_t> masking_dims;
    for (int64_t dim : dus_ar_dims) {
      int64_t partitions = sharding().tile_assignment().dim(dim);
      if (base_shape().dimensions(dim) < partitions) {
        // DUS will be out-of-bound and offset will be clamped, so we need to
        // mask this dim with 0.
        masking_dims.push_back(dim);
        // Adjust the padded dim size. This can be also failed allgather dims.
        padded_target_shape.set_dimensions(dim, base_shape().dimensions(dim));
      }
    }
    if (!masking_dims.empty()) {
      std::vector<int64_t> skipped_dims;
      for (int64_t i = 0; i < base_shape().rank(); ++i) {
        if (!absl::c_linear_search(masking_dims, i)) {
          skipped_dims.push_back(i);
        }
      }
      result->set_sharding(sharding());
      result = PartitionedHlo(result, padded_target_shape, state_)
                   .PadWithValue(zero,
                                 /*left_padded_dims=*/{},
                                 /*skipped_dims=*/skipped_dims)
                   .hlo();
    }
    auto zero_bcast = state_.b->AddInstruction(
        HloInstruction::CreateBroadcast(padded_target_shape, zero, {}));
    auto offsets = MakePartitionOffsets(
        padded_target_shape,
        hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
            sharding(), dus_ar_dims),
        state_.partition_id, state_.b, dus_ar_dims);
    auto dus =
        state_.b->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            padded_target_shape, zero_bcast, result, offsets));
    HloComputation* reduction =
        MakeBinaryAdd(shard_shape.element_type(), state_.module);
    result = state_.partitioner->AllReduceAlongShardingDims(
        state_.b, dus, sharding(), state_.next_channel_id, dus_ar_dims,
        state_.collective_ops_creator, reduction);
  }
  if (!ShapeUtil::Compatible(target_shape, padded_target_shape)) {
    std::vector<int64_t> start_indices(target_shape.rank(), 0);
    std::vector<int64_t> strides(target_shape.rank(), 1);
    result = state_.b->AddInstruction(
        HloInstruction::CreateSlice(target_shape, result, start_indices,
                                    target_shape.dimensions(), strides));
  }
  return result;
}

std::optional<PartitionedHlo>
PartitionedHlo::ReshardToPartialReplicateWithAllGather(
    const HloSharding& target) {
  if (!target.ReplicateOnLastTileDim()) {
    return std::nullopt;
  }
  // Tiled/partial replicate to partial replicate
  // Get the comptible sharding to target with resharding by all reduce.
  auto compatible_sharding =
      PartialReplicateReshardCompatibleSharding(target, sharding());
  if (!compatible_sharding.has_value()) {
    return std::nullopt;
  }

  const auto& temp_sharding = compatible_sharding.value();
  auto partitioned_hlo = *this;
  // Use collective permute to adjust device assignment if needed.
  if (CanReshardWithCollectivePermute(sharding(), temp_sharding)) {
    partitioned_hlo =
        partitioned_hlo.ReshardWithCollectivePermute(temp_sharding);
  }

  // Get replicate dims and replicate factor of each dimensions.
  int64_t rank = hlo_->shape().rank();
  std::vector<int64_t> replicate_dims;
  std::vector<int64_t> replicate_factors;
  for (int64_t dim = 0; dim < rank; dim++) {
    int64_t replicate_factor = temp_sharding.tile_assignment().dim(dim) /
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
    return std::nullopt;
  }
  auto halo_exchange_hlo = halo_exchange.value();
  // Grouped on replicate dimensions.
  auto sharding_grouped = hlo_sharding_util::GroupShardingOnDims(
      temp_sharding, replicate_dims, replicate_factors);
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

std::optional<PartitionedHlo>
PartitionedHlo::ReshardFromPartialReplicateWithDynamicSlice(
    const HloSharding& target) {
  if (!sharding().ReplicateOnLastTileDim()) {
    return std::nullopt;
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
    return std::nullopt;
  }
  std::vector<int64_t> expand_tile_dims;
  std::vector<int64_t> tiling_dim_factors;
  int64_t rank = hlo_->shape().rank();
  tiling_dim_factors.reserve(target.tile_assignment().num_dimensions());
  const auto& temp_target_sharding = target_compatible_sharding.value();
  for (int64_t dim = 0; dim < rank; dim++) {
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
    return std::nullopt;
  }
  // 3. Slice out the tile from replicate ones.
  auto shard_shape = MakePartitionedShape(base_shape_, temp_target_sharding);
  // Since we are just slicing, we can just use the differences between the new
  // and old offsets in the full shape as the dynamic-slice offsets.
  auto padded_base_shape = shard_shape;
  for (int64_t i = 0; i < padded_base_shape.rank(); ++i) {
    padded_base_shape.set_dimensions(
        i, padded_base_shape.dimensions(i) *
               temp_target_sharding.tile_assignment().dim(i));
  }
  auto offsets = MakePartitionOffsets(padded_base_shape, temp_target_sharding,
                                      state_.partition_id, state_.b);
  auto old_offsets = MakePartitionOffsets(padded_base_shape, sharding(),
                                          state_.partition_id, state_.b);
  for (int64_t i = 0; i < offsets.size(); ++i) {
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
      LiteralUtil::CreateR0<uint32_t>(sharding.GetUniqueDevice())));
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
    absl::Span<const std::pair<int64_t, int64_t>> source_target_dims) const {
  if (source_target_dims.empty()) {
    if (target == sharding()) {
      return *this;
    }
    // If the device order is different in the target, fix the order with
    // ReshardWithCollectivePermute.
    return ReshardWithCollectivePermute(target);
  }

  VLOG(5) << "Source: " << sharding().ToString();
  VLOG(5) << "Target: " << target.ToString();
  // Swap one pair of dimensions.
  int64_t source_dim = source_target_dims[0].first;
  int64_t target_dim = source_target_dims[0].second;
  const int64_t group_size = sharding().tile_assignment().dim(source_dim) /
                             sharding().tile_assignment().dim(target_dim);

  VLOG(5) << "Group size: " << group_size;
  auto temp_target_tile = sharding().tile_assignment();
  {
    std::vector<int64_t> reshape_tile_dims(temp_target_tile.num_dimensions() +
                                           2);
    int64_t i = 0;
    int64_t added_source_dim = -1;
    int64_t added_target_dim = -1;
    for (int64_t j = 0; j < temp_target_tile.num_dimensions(); ++j) {
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
    VLOG(5) << "Added target: " << added_target_dim;
    VLOG(5) << "Added source: " << added_source_dim;
    temp_target_tile.Reshape(reshape_tile_dims);
    std::vector<int64_t> xpose_dims(temp_target_tile.num_dimensions());
    std::iota(xpose_dims.begin(), xpose_dims.end(), 0);
    xpose_dims[added_source_dim] = added_target_dim;
    xpose_dims[added_target_dim] = added_source_dim;
    temp_target_tile = hlo_sharding_util::TransposeSharding(
                           HloSharding::Tile(temp_target_tile), xpose_dims)
                           .tile_assignment();
    VLOG(5) << "Transposed target: " << temp_target_tile.ToString();
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
  VLOG(5) << "Temp target sharding: " << temp_target.ToString();
  auto padded_shape = hlo_->shape();
  auto padded_base_shape = base_shape_;
  auto current_base_padded_shape = base_shape_;
  padded_base_shape.set_dimensions(
      target_dim, RoundUpTo(base_shape_.dimensions(target_dim),
                            temp_target.tile_assignment().dim(target_dim)));
  current_base_padded_shape.set_dimensions(
      target_dim, hlo_->shape().dimensions(target_dim) *
                      sharding().tile_assignment().dim(target_dim));

  auto padded_source_base_shape = base_shape_;
  auto current_source_base_padded_shape = base_shape_;
  padded_source_base_shape.set_dimensions(
      source_dim, RoundUpTo(base_shape_.dimensions(source_dim),
                            temp_target.tile_assignment().dim(source_dim)));
  current_source_base_padded_shape.set_dimensions(
      source_dim, hlo_->shape().dimensions(source_dim) *
                      sharding().tile_assignment().dim(source_dim));

  VLOG(5) << "Target dim: " << target_dim;
  VLOG(5) << "Source dim: " << source_dim;
  VLOG(5) << "Original sharded shape: " << hlo_->shape();
  VLOG(5) << "Base shape: " << base_shape_.ToString();
  VLOG(5) << "Padded base shape: " << padded_base_shape.ToString();
  VLOG(5) << "Current padded shape: " << current_base_padded_shape.ToString();
  VLOG(5) << "Padded source base shape: "
          << padded_source_base_shape.ToString();
  VLOG(5) << "Current source padded shape: "
          << current_source_base_padded_shape.ToString();
  VLOG(5) << "Dimension padded target_dim: "
          << hlo_->shape().dimensions(target_dim) *
                 sharding().tile_assignment().dim(target_dim);
  CHECK_GE(padded_base_shape.rank(), current_base_padded_shape.rank());
  CHECK_LE(padded_source_base_shape.rank(),
           current_source_base_padded_shape.rank());

  PaddingConfig pc;
  for (int64_t i = 0; i < hlo_->shape().rank(); ++i) {
    auto* pd = pc.add_dimensions();
    pd->set_edge_padding_low(0);
    pd->set_edge_padding_high(padded_base_shape.dimensions(i) -
                              current_base_padded_shape.dimensions(i));
    pd->set_interior_padding(0);
  }
  PartitionedHlo p_hlo = *this;
  VLOG(5) << "Before reshard: " << p_hlo.hlo_->ToString();
  HloInstruction* zero = CreateZero(
      ShapeUtil::MakeShape(hlo_->shape().element_type(), {}), state_.b);
  auto padded_phlo = ReshardDataForPad(zero, pc, p_hlo, padded_base_shape,
                                       sharding(), state_.b);
  CHECK(padded_phlo.has_value());
  VLOG(5) << "Resharded: " << padded_phlo->sharded_input->ToString();
  VLOG(5) << "Padded Window: " << padded_phlo->shard_window.DebugString();
  HloInstruction* padded_hlo =
      PadDataFromWindowReshard(*padded_phlo, zero, state_.b);
  VLOG(5) << "Padded data: " << padded_hlo->ToString();

  // The order of ids in the group must follow the temp_target sharding.
  std::vector<std::vector<int64_t>> groups(
      temp_target.tile_assignment().num_elements() / group_size);
  temp_target.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t device) {
        int64_t group_id = 0;
        for (int64_t dim = 0; dim < indices.size(); ++dim) {
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
  std::vector<int64_t> dimensions;
  const int64_t rank = base_shape_.rank();
  dimensions.reserve(rank + 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == target_dim) {
      dimensions.push_back(group_size);
      dimensions.push_back(padded_hlo->shape().dimensions(i) / group_size);
    } else {
      dimensions.push_back(padded_hlo->shape().dimensions(i));
    }
  }
  VLOG(5) << "Target ata shape: "
          << ShapeUtil::MakeShape(base_shape_.element_type(), dimensions)
                 .ToString();
  auto reshape = state_.b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(base_shape_.element_type(), dimensions),
      padded_hlo));
  // After the reshape, it is guaranteed to have at least 3 dimensions.
  auto all_to_all =
      state_.collective_ops_creator.create_cross_partition_all_to_all(
          state_.b, {reshape}, groups, (*state_.next_channel_id)++, target_dim);

  // Reorder the split dimension of the reshape to be located in front of the
  // input partition dimension, so the two dimensions can be combined.
  int64_t new_source_dim =
      (target_dim < source_dim) ? source_dim + 1 : source_dim;
  std::vector<int64_t> permutation;
  for (int64_t i = 0; i < all_to_all->shape().rank(); ++i) {
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
          .value(),
      all_to_all, permutation));

  // Combine the split dimension and the input partition dimension.
  auto new_shape = ShapeInference::InferAllToAllShape(
                       padded_hlo->shape(), target_dim, source_dim, group_size)
                       .value();
  result = state_.b->AddInstruction(
      HloInstruction::CreateReshape(new_shape, transpose));
  result->set_sharding(temp_target);
  std::vector<int64_t> strides(result->shape().rank(), 1);
  std::vector<int64_t> starts(result->shape().rank(), 0);
  std::vector<int64_t> limits(result->shape().rank());
  for (int64_t i = 0; i < result->shape().rank(); ++i) {
    limits[i] = padded_source_base_shape.dimensions(i);
  }
  auto sliced_phlo = ReshardDataForSlicing(
      strides, starts, limits,
      PartitionedHlo(result, current_source_base_padded_shape, state_),
      temp_target, state_.b);
  CHECK(sliced_phlo.has_value());
  result = SliceDataFromWindowReshard(*sliced_phlo, strides, base_shape_,
                                      temp_target, state_.b);
  result->set_sharding(temp_target);
  auto remaining_source_target_dims = source_target_dims;
  remaining_source_target_dims.remove_prefix(1);
  return PartitionedHlo(result, base_shape_, state_)
      .ReshardWithAllToAll(target, remaining_source_target_dims);
}

namespace {

// Matching a pattern like [..,X,..,Y] -> [..,X*Y,..,1] or [..,X,..,Y] ->
// [..,1,..,X*Y].
std::optional<std::pair<HloSharding, int>> PatternMatchReshape(
    const Shape& shape, const HloSharding& source, const HloSharding& target) {
  if (!source.IsTiled() || !target.IsTiled()) {
    return std::nullopt;
  }
  if (source.TiledDataRank() != target.TiledDataRank()) {
    return std::nullopt;
  }
  if ((source.HasPartialReplication() ^ target.HasPartialReplication()) ||
      (source.HasPartialReplication() &&
       source.tile_assignment().dimensions()[source.TiledDataRank()] !=
           target.tile_assignment().dimensions()[target.TiledDataRank()])) {
    return std::nullopt;
  }
  for (int i = 0; i < target.TiledDataRank(); ++i) {
    if (source.tile_assignment().dim(i) > target.tile_assignment().dim(i) &&
        target.tile_assignment().dim(i) == 1 &&
        (source.tile_assignment().dim(i) % target.tile_assignment().dim(i)) ==
            0) {
      const int64_t dimension_size =
          source.tile_assignment().dim(i) / target.tile_assignment().dim(i);
      for (int j = i - 1; j >= 0; --j) {
        if (target.tile_assignment().dim(j) == 1) {
          continue;
        }
        if (target.tile_assignment().dim(j) !=
            dimension_size * source.tile_assignment().dim(j)) {
          continue;
        }
        // Do not consider if it requires additional padding.
        if (shape.dimensions(j) % dimension_size != 0) {
          continue;
        }
        auto reshaped_sharding = hlo_sharding_util::SplitShardingDimension(
            source, j, source.tile_assignment().dim(j));
        std::vector<int64_t> permutation(reshaped_sharding.TiledDataRank(), 0);
        absl::c_iota(permutation, 0);
        std::swap(permutation[i + 1], permutation[j]);
        reshaped_sharding = hlo_sharding_util::TransposeSharding(
            reshaped_sharding, permutation);
        return std::make_pair(reshaped_sharding, j);
      }
      for (int j = i + 1; j < target.TiledDataRank(); ++j) {
        if (target.tile_assignment().dim(j) == 1) {
          continue;
        }
        if (target.tile_assignment().dim(j) !=
            dimension_size * source.tile_assignment().dim(j)) {
          continue;
        }
        // Do not consider if it requires additional padding.
        if (shape.dimensions(j) % dimension_size != 0) {
          continue;
        }

        auto reshaped_sharding = hlo_sharding_util::SplitShardingDimension(
            source, j, source.tile_assignment().dim(j));
        std::vector<int64_t> permutation(reshaped_sharding.TiledDataRank(), 0);
        absl::c_iota(permutation, 0);
        VLOG(5) << "Reshaped sharding before: " << reshaped_sharding.ToString();
        std::swap(permutation[i], permutation[j]);
        reshaped_sharding = hlo_sharding_util::TransposeSharding(
            reshaped_sharding, permutation);
        VLOG(5) << "Reshaped sharding: " << reshaped_sharding.ToString();
        return std::make_pair(reshaped_sharding, j);
      }
    }
  }
  return std::nullopt;
}
// Match patterns like [..,X,..,Z,..,Y,..] -> [..,Y,..,Z,..,X]
// last_tile_dim_replicate, where X gets replicated and Y also changes
// position. We try to perform the replication, so we can match some other
// targets instead.
std::optional<HloSharding> PatternMatchPartiallyReplicateDim(
    const HloSharding& source, const HloSharding& target) {
  if (!(!source.ReplicateOnLastTileDim() && target.ReplicateOnLastTileDim())) {
    return std::nullopt;
  }
  const int64_t target_replicated_dim = target.SubgroupReplicationDim();
  CHECK_NE(target_replicated_dim, -1) << "Expected replicated dim";
  for (int i = 0; i < source.tile_assignment().num_dimensions(); ++i) {
    if (source.tile_assignment().dim(i) !=
        target.tile_assignment().dim(target_replicated_dim)) {
      continue;
    }
    auto replicated_sharding =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(source, {i});
    return replicated_sharding;
  }
  return std::nullopt;
}

// Helper to split a PartitionedHlo over a specific dimension.
PartitionedHlo SplitReshapeHelper(PartitionedHlo to_reshape,
                                  int64_t dim_to_split, int64_t dim_size,
                                  const HloSharding& target_sharding) {
  Shape original_shape = to_reshape.hlo()->shape();
  std::vector<int64_t> shape_dim(original_shape.dimensions().begin(),
                                 original_shape.dimensions().end());
  shape_dim.insert(shape_dim.begin() + dim_to_split + 1, 1);
  std::vector<int64_t> base_shape_dim(
      to_reshape.base_shape().dimensions().begin(),
      to_reshape.base_shape().dimensions().end());
  base_shape_dim.insert(base_shape_dim.begin() + dim_to_split + 1, dim_size);
  base_shape_dim[dim_to_split] /= dim_size;
  Shape shape = ShapeUtil::MakeShape(original_shape.element_type(), shape_dim);
  HloInstruction* reshaped_instr = to_reshape.state().b->AddInstruction(
      HloInstruction::CreateReshape(shape, to_reshape.hlo()));
  reshaped_instr->set_sharding(target_sharding);
  return PartitionedHlo{
      reshaped_instr,
      ShapeUtil::MakeShape(to_reshape.base_shape().element_type(),
                           base_shape_dim),
      to_reshape.state()};
}
// Merge a PartitionedHlo over a specific dimension.
PartitionedHlo MergeReshapeHelper(PartitionedHlo to_reshape,
                                  int64_t dim_to_merge,
                                  const HloSharding& target_sharding) {
  Shape original_shape = to_reshape.hlo()->shape();
  std::vector<int64_t> shape_dim(original_shape.dimensions().begin(),
                                 original_shape.dimensions().end());
  shape_dim[dim_to_merge] *= shape_dim[dim_to_merge + 1];
  shape_dim.erase(shape_dim.begin() + dim_to_merge + 1);
  std::vector<int64_t> base_shape_dim(
      to_reshape.base_shape().dimensions().begin(),
      to_reshape.base_shape().dimensions().end());
  base_shape_dim[dim_to_merge] *= base_shape_dim[dim_to_merge + 1];
  base_shape_dim.erase(base_shape_dim.begin() + dim_to_merge + 1);
  Shape shape = ShapeUtil::MakeShape(original_shape.element_type(), shape_dim);
  HloInstruction* reshaped_instr = to_reshape.state().b->AddInstruction(
      HloInstruction::CreateReshape(shape, to_reshape.hlo()));
  reshaped_instr->set_sharding(target_sharding);
  return PartitionedHlo(
      reshaped_instr,
      ShapeUtil::MakeShape(original_shape.element_type(), base_shape_dim),
      to_reshape.state());
}

}  // namespace

std::optional<PartitionedHlo> PartitionedHlo::TryComplexReshardHandling(
    const HloSharding& target) {
  VLOG(5) << "Trying to split complicated reshard: " << sharding().ToString()
          << " to " << target.ToString();
  const bool is_source_partially_replicated =
      sharding().ReplicateOnLastTileDim();
  const bool is_target_partially_replicated = target.ReplicateOnLastTileDim();
  if (auto reshape =
          PatternMatchReshape(this->hlo()->shape(), sharding(), target)) {
    VLOG(5) << "Matched \"pattern_match_reshape()\": "
            << reshape->first.ToString();
    VLOG(5) << "Original shape: " << hlo()->shape().ToString();
    auto before_sharding = hlo_sharding_util::SplitShardingDimension(
        sharding(), reshape->second,
        sharding().tile_assignment().dim(reshape->second));
    PartitionedHlo reshaped = SplitReshapeHelper(
        *this, reshape->second,
        sharding().tile_assignment().dim(reshape->second), before_sharding);
    VLOG(5) << "Reshaped shape: " << reshaped.hlo()->shape().ToString();
    auto reshard = reshaped.ReshardNoCache(reshape->first,
                                           /*pad_value=*/std::nullopt,
                                           /*allow_full_replication=*/false);
    if (reshard.sharding() != reshape->first) {
      return std::nullopt;
    }
    auto reshaped_sharding = hlo_sharding_util::MergeShardingDimension(
        reshard.sharding(), reshape->second);
    reshaped = MergeReshapeHelper(reshard, reshape->second, reshaped_sharding);
    if (reshaped.sharding() != target) {
      reshaped = reshaped.ReshardNoCache(target, /*pad_value=*/std::nullopt,
                                         /*allow_full_replication=*/false);
      if (reshaped.sharding() != target) {
        return std::nullopt;
      }
    }
    return reshaped;
  }
  if (auto intermediate_target =
          PatternMatchPartiallyReplicateDim(sharding(), target)) {
    VLOG(5) << "Matched \"pattern_match_partially_replicate_dim()\": "
            << intermediate_target->ToString();
    auto intermediate_reshard = Reshard(*intermediate_target);
    auto final_reshard = intermediate_reshard.ReshardNoCache(
        target, /*pad_value=*/std::nullopt, /*allow_full_replication=*/false);
    if (final_reshard.sharding() != target) {
      return std::nullopt;
    }
    return final_reshard;
  }
  if (is_source_partially_replicated && !is_target_partially_replicated) {
    const int64_t partial_repl_amount =
        sharding().tile_assignment().dimensions().back();
    int64_t first_different_dimension = -1;
    // Trying to match conditions like [..,X,..,Z,..,Y] last_tile_dim_replicate
    // to [..,Y,..,Z,..,X,..], where Y in the source is partially replicated,
    // but in the target it is not and some other dimension got moved or
    // modified. Try to remove the partial replication to simplify the step from
    // source to target sharding.
    for (int64_t i = 0; i < target.tile_assignment().num_dimensions(); ++i) {
      if (target.tile_assignment().dim(i) !=
              sharding().tile_assignment().dim(i) &&
          sharding().tile_assignment().dim(i) == 1 &&
          target.tile_assignment().dim(i) % partial_repl_amount == 0) {
        first_different_dimension = i;
        break;
      }
    }
    if (first_different_dimension == -1) {
      return std::nullopt;
    }
    VLOG(5) << "Matched partially replicated to non partially replicated: "
            << sharding().ToString();
    std::vector<int64_t> transpose_dims(
        sharding().tile_assignment().num_dimensions(), 0);
    std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
    std::swap(transpose_dims[first_different_dimension], transpose_dims.back());
    auto intermediate_sharding =
        hlo_sharding_util::TransposeSharding(sharding(), transpose_dims);
    auto intermediate_reshard = Reshard(intermediate_sharding);
    auto reshard = intermediate_reshard.ReshardNoCache(
        target, /*pad_value=*/std::nullopt, /*allow_full_replication=*/false);
    if (reshard.sharding() != target) {
      return std::nullopt;
    }
    return reshard;
  }
  return std::nullopt;
}

std::optional<PartitionedHlo>
PartitionedHlo::ReshardPartialReplicateWithAllToAll(const HloSharding& target) {
  bool source_is_partial_replicate = sharding().ReplicateOnLastTileDim();
  const auto& partial_replicate_sharding =
      source_is_partial_replicate ? sharding() : target;
  // If neither the source nor the target is partial replicate, return null.
  if (!partial_replicate_sharding.ReplicateOnLastTileDim()) {
    return std::nullopt;
  }
  const auto& tile_sharding = source_is_partial_replicate ? target : sharding();
  // If both source and target are partial replicate, should be supported in
  // Reshard with AllToAll already.
  if (tile_sharding.ReplicateOnLastTileDim() || tile_sharding.IsTileMaximal()) {
    return std::nullopt;
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
    return std::nullopt;
  }
  int to_replicate_dim = -1;
  for (int i = tile_sharding.tile_assignment().num_dimensions() - 1; i >= 0;
       --i) {
    if (tile_sharding.tile_assignment().dim(i) > 1 &&
        (to_replicate_dim == -1)) {
      if (tile_sharding.tile_assignment().dim(i) != num_replicas) {
        return std::nullopt;
      }
      to_replicate_dim = i;
    }

    if (tile_sharding.tile_assignment().dim(i) !=
        partial_replicate_sharding.tile_assignment().dim(i + 1)) {
      return std::nullopt;
    }
  }

  if (to_replicate_dim == -1) {
    return std::nullopt;
  }

  // Check if core assignments for source and the target are the same.
  auto reshape_tile_assignment = partial_replicate_sharding.tile_assignment();
  reshape_tile_assignment.Reshape(tile_sharding.tile_assignment().dimensions());
  if (reshape_tile_assignment != tile_sharding.tile_assignment()) {
    return std::nullopt;
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

  return std::nullopt;
}

PartitionedHlo PartitionedHlo::ReshardWithCollectivePermute(
    const HloSharding& target) const {
  CHECK(CanReshardWithCollectivePermute(sharding(), target))
      << sharding().ToString() << " to " << target.ToString();
  if (auto broadcast_dims = state_.b->BroadcastDimsForCreatedHlo(hlo())) {
    if (!(*broadcast_dims)->empty()) {
      // If hlo() has broadcast dims, check if data is already the same between
      // source/destination pairs.
      std::vector<int64_t> broadcast_dims_vector;
      for (int64_t i = 0; i < hlo()->shape().rank(); ++i) {
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
  std::vector<std::pair<int64_t, int64_t>> src_dst_pairs;
  sharding().tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t src_device) {
        int64_t dst_device = target.tile_assignment()(indices);
        src_dst_pairs.emplace_back(src_device, dst_device);
      });
  auto cp =
      state_.collective_ops_creator.create_cross_partition_collective_permute(
          state_.b, hlo(), src_dst_pairs, (*state_.next_channel_id)++);
  cp->set_sharding(target);
  return PartitionedHlo(cp, base_shape_, state_);
}

SpmdPartitioningVisitor::SpmdPartitioningVisitor(
    HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdLogger* logger,
    SpmdPartitionerOptions options, SpmdPartitioner* partitioner)
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

PartitionedHlo::PartitioningState
SpmdPartitioningVisitor::MakePartitioningState() {
  PartitionedHlo::PartitioningState state;
  state.b = &b_;
  state.module = module_;
  state.num_replicas = num_replicas_;
  state.next_channel_id = next_channel_id_;
  state.reshard_cache = &reshard_cache_;
  state.partitioner = partitioner_;
  if (!device_groups_.empty()) {
    // Use the original collective creator and partition_id to call
    // CreatePerGroupPartitioningState(). Current collective_ops_creator_ and
    // partition_id_ have been rewritten to be subgrouped.
    state.collective_ops_creator = *visiting_collective_ops_creator_;
    state.partition_id = *visiting_partition_id_;
    return CreatePerGroupPartitioningState(state, device_groups_, &b_);
  } else {
    state.collective_ops_creator = collective_ops_creator_;
    state.partition_id = partition_id_;
  }
  return state;
}

std::vector<ReplicaGroup> SpmdPartitioningVisitor::CreateReplicaGroups(
    std::vector<std::vector<int64_t>>& groups) {
  std::vector<ReplicaGroup> device_groups;
  device_groups.reserve(groups.size() * num_replicas_);
  for (int64_t i = 0; i < num_replicas_; ++i) {
    for (const auto& group : groups) {
      device_groups.emplace_back();
      for (int64_t id : group) {
        device_groups.back().add_replica_ids(i * num_partitions_ + id);
      }
    }
  }
  return device_groups;
}

Status SpmdPartitioningVisitor::DefaultAction(HloInstruction* hlo) {
  if (hlo->HasSideEffect() && !hlo->sharding().HasUniqueDevice()) {
    return Unimplemented("Side-effect ops cannot be replicated: %s",
                         hlo->ToString());
  }

  if (hlo->IsElementwise() && hlo->operand_count() > 0) {
    return HandleElementwise(hlo);
  }

  if (!hlo->sharding().IsTileMaximal()) {
    VLOG(1) << "Not partitioned in SPMD mode (DefaultAction):"
            << hlo->ToString();
    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      VLOG(1) << "  operand " << i
              << " sharding:" << hlo->operand(i)->sharding().ToString();
    }
  }

  HloSharding sharding = hlo->sharding().HasUniqueDevice()
                             ? hlo->sharding()
                             : HloSharding::Replicate();
  if (hlo->opcode() == HloOpcode::kSend || hlo->opcode() == HloOpcode::kRecv ||
      hlo->opcode() == HloOpcode::kRecvDone) {
    sharding = sharding.GetSubSharding(hlo->shape(), {0});
  }

  // If the instruction cannot be partitioned, replicate the instruction unless
  // the instruction has side-effect.
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : hlo->operands()) {
    new_operands.push_back(GetPartitionedHlo(operand).Reshard(sharding).hlo());
  }
  auto clone =
      b_.AddInstruction(hlo->CloneWithNewOperands(hlo->shape(), new_operands));
  clone->set_sharding(sharding);
  SetPartitionedHlo(hlo,
                    PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
                        .Reshard(hlo->sharding()));
  return OkStatus();
}

Status SpmdPartitioningVisitor::Preprocess(HloInstruction* hlo) {
  visiting_hlo_ = hlo;
  b_.set_visiting_hlo(hlo);
  // Temporarily replace manual sharding to one-device sharding so that the
  // partitioner will not change the HLOs.
  auto manual_to_onedevice = [&](HloOpcode opcode, const Shape& shape,
                                 const HloSharding& sharding) {
    // If a tuple's elements are all manual, then sharding.IsManual() == True,
    // so we test whether it is tuple first.
    if (sharding.IsTuple()) {
      std::vector<HloSharding> subshardings = sharding.tuple_elements();
      for (HloSharding& subsharding : subshardings) {
        // Delay manual sharding substitution for CustomCalls.
        if (subsharding.IsManual() && opcode != HloOpcode::kCustomCall) {
          subsharding = HloSharding::AssignDevice(0);
        }
      }
      return HloSharding::Tuple(shape, subshardings);
    }
    // Delay manual sharding substitution for CustomCalls.
    if (sharding.IsManual() && opcode != HloOpcode::kCustomCall) {
      return HloSharding::AssignDevice(0);
    }
    return sharding;
  };

  if (hlo->opcode() != HloOpcode::kConditional &&
      hlo->opcode() != HloOpcode::kTuple &&
      hlo->opcode() != HloOpcode::kGetTupleElement &&
      hlo->opcode() != HloOpcode::kParameter &&
      hlo->opcode() != HloOpcode::kWhile && hlo->opcode() != HloOpcode::kRng &&
      hlo->opcode() != HloOpcode::kAllReduce) {
    const bool has_manual_sharding =
        hlo->sharding().IsManual() ||
        (hlo->sharding().IsTuple() &&
         absl::c_any_of(
             hlo->sharding().tuple_elements(),
             [](const HloSharding& sharding) { return sharding.IsManual(); }));
    if (has_manual_sharding && !hlo->IsCustomCall("SPMDFullToShardShape")) {
      visiting_hlo_sharding_ = hlo->sharding();
      hlo->set_sharding(manual_to_onedevice(hlo->opcode(), hlo->shape(),
                                            *visiting_hlo_sharding_));

      visiting_hlo_operand_shardings_.reserve(hlo->operand_count());
      for (HloInstruction* operand : hlo->unique_operands()) {
        visiting_hlo_operand_shardings_.push_back(operand->sharding());
        operand->set_sharding(manual_to_onedevice(
            hlo->opcode(), operand->shape(), operand->sharding()));
        GetPartitionedHlo(operand).hlo()->set_sharding(operand->sharding());
      }
    } else {
      const bool has_manual_subgroup =
          hlo->sharding().IsManualSubgroup() ||
          (hlo->sharding().IsTuple() &&
           absl::c_any_of(hlo->sharding().tuple_elements(),
                          [](const HloSharding& sharding) {
                            return sharding.IsManualSubgroup();
                          }));
      if (has_manual_subgroup && !hlo->IsCustomCall("SPMDFullToShardShape")) {
        auto get_grouped_sharding =
            [&](const HloSharding& sharding, const Shape& shape,
                const GroupedSharding* ref =
                    nullptr) -> StatusOr<GroupedSharding> {
          if (!sharding.IsTuple()) {
            GroupedSharding grouped =
                hlo_sharding_util::GetManualSubgroupSharding(sharding);
            if (ref != nullptr) {
              auto aligned =
                  AlignGroupsWithIfCompatible(std::move(grouped), *ref);
              TF_RET_CHECK(aligned.has_value())
                  << "Incompatible manual sharding at " << hlo->ToString();
              return *aligned;
            }
            return grouped;
          }
          std::vector<HloSharding> elements;
          elements.reserve(sharding.tuple_elements().size());
          CHECK(!sharding.tuple_elements().empty());
          GroupedSharding grouped0 =
              hlo_sharding_util::GetManualSubgroupSharding(
                  sharding.tuple_elements()[0]);
          if (ref != nullptr) {
            auto aligned =
                AlignGroupsWithIfCompatible(std::move(grouped0), *ref);
            TF_RET_CHECK(aligned.has_value())
                << "Incompatible manual sharding at " << hlo->ToString();
            grouped0 = std::move(*aligned);
          }
          elements.push_back(std::move(grouped0.sharding));
          for (int64_t i = 1; i < sharding.tuple_elements().size(); ++i) {
            auto grouped_i = AlignGroupsWithIfCompatible(
                hlo_sharding_util::GetManualSubgroupSharding(
                    sharding.tuple_elements()[i]),
                grouped0);
            TF_RET_CHECK(grouped_i.has_value())
                << "Incompatible manual sharding between tuple elements: "
                << hlo->ToString();
            elements.push_back(std::move(grouped_i->sharding));
          }
          grouped0.sharding = HloSharding::Tuple(shape, elements);
          return grouped0;
        };
        TF_ASSIGN_OR_RETURN(
            auto group_sharding,
            get_grouped_sharding(hlo->sharding(), hlo->shape()));
        // Update sharding.
        visiting_hlo_sharding_ = hlo->sharding();
        hlo->set_sharding(group_sharding.sharding);
        // Update device_groups and num_partitions.
        // Set device_groups_, visiting_partition_id_ and
        // visiting_collective_ops_creator_ before MakePartitioningState() which
        // uses them.
        device_groups_ = group_sharding.device_groups;
        visiting_num_partitions_ = num_partitions_;
        num_partitions_ = num_partitions_ / group_sharding.device_groups.size();
        visiting_partition_id_ = partition_id_;
        visiting_collective_ops_creator_ = std::move(collective_ops_creator_);
        auto grouped_state = MakePartitioningState();
        collective_ops_creator_ =
            std::move(grouped_state.collective_ops_creator);
        partition_id_ = grouped_state.partition_id;

        // Update sharding for the operands.
        visiting_hlo_operand_shardings_.reserve(hlo->operand_count());
        visiting_state_.reserve(hlo->operand_count());
        for (HloInstruction* operand : hlo->unique_operands()) {
          visiting_hlo_operand_shardings_.push_back(operand->sharding());
          auto old_state = GetPartitionedHlo(operand).state();
          visiting_state_.push_back(old_state);
          if (operand->shape().IsArray() && operand->IsConstant() &&
              operand->shape().rank() == 0 &&
              !operand->sharding().IsManualSubgroup()) {
            // We allowed scalar constants to be CSE'ed between manual/auto
            // subgraphs. It's possible that it doesn't have a manual subgroup.
            continue;
          }
          TF_ASSIGN_OR_RETURN(
              auto op_group_sharding,
              get_grouped_sharding(operand->sharding(), operand->shape(),
                                   &group_sharding));
          operand->set_sharding(op_group_sharding.sharding);
          GetPartitionedHlo(operand).hlo()->set_sharding(operand->sharding());
          auto group_state = CreatePerGroupPartitioningState(
              old_state, op_group_sharding.device_groups, &b_);
          GetPartitionedHlo(operand).set_state(group_state);
        }
      }
    }
  }
  return OkStatus();
}

Status SpmdPartitioningVisitor::Postprocess(HloInstruction* hlo) {
  logger_->RegisterLogEntry(hlo, b_.derived_instructions(hlo));
  visiting_hlo_ = nullptr;
  b_.set_visiting_hlo(nullptr);
  // Revert fake one-device shardings for manually partitioned ops.
  if (visiting_hlo_sharding_) {
    hlo->set_sharding(*visiting_hlo_sharding_);
    GetPartitionedHlo(hlo).hlo()->set_sharding(*visiting_hlo_sharding_);
    int64_t i = 0;
    for (HloInstruction* operand : hlo->unique_operands()) {
      operand->set_sharding(visiting_hlo_operand_shardings_[i++]);
      GetPartitionedHlo(operand).hlo()->set_sharding(operand->sharding());
    }
    visiting_hlo_sharding_.reset();
    visiting_hlo_operand_shardings_.clear();
  }

  if (!device_groups_.empty()) {
    device_groups_.clear();
    num_partitions_ = *visiting_num_partitions_;
    visiting_num_partitions_.reset();
    collective_ops_creator_ = *visiting_collective_ops_creator_;
    visiting_collective_ops_creator_.reset();
    partition_id_ = *visiting_partition_id_;
    visiting_partition_id_.reset();
    GetPartitionedHlo(hlo).set_state(MakePartitioningState());
  }

  if (!visiting_state_.empty()) {
    int64_t i = 0;
    for (const HloInstruction* operand : hlo->unique_operands()) {
      GetPartitionedHlo(operand).set_state(visiting_state_[i++]);
    }
    visiting_state_.clear();
  }

  return OkStatus();
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleConcatenate(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  const Shape shard_shape = MakePartitionedShape(hlo->shape(), hlo->sharding());
  const int64_t dimension = hlo->concatenate_dimension();
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
    return OkStatus();
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
  int64_t offset = 0;
  auto state = MakePartitioningState();
  for (HloInstruction* operand : hlo->operands()) {
    auto spmd_operand = GetPartitionedHlo(operand).Reshard(sharding).hlo();
    std::vector<HloInstruction*> start_indices(
        hlo->shape().rank(), b_.AddInstruction(HloInstruction::CreateConstant(
                                 LiteralUtil::Zero(S32))));
    start_indices[dimension] =
        MultiplyAddDivideOffsetCalculation(
            spmd_operand->shape().dimensions(dimension), offset, 1)
            .Calculate(MakeTiledPartitionOrdinals(sharding, state.partition_id,
                                                  &b_)[dimension],
                       &b_);
    temp_output = b_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        temp_output_shape, temp_output, spmd_operand, start_indices));
    offset += operand->shape().dimensions(dimension);
  }
  std::vector<int64_t> non_concat_dims;
  non_concat_dims.reserve(hlo->shape().rank() - 1);
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
    if (i != dimension) {
      non_concat_dims.push_back(i);
    }
  }
  auto grouped =
      hlo_sharding_util::GroupShardingOnDims(sharding, non_concat_dims);
  auto per_group_partitioner_state =
      CreatePerGroupPartitioningState(state, grouped.device_groups, &b_);
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

  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleSlice(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  auto operand = GetPartitionedHlo(hlo->operand(0)).Reshard(sharding);
  auto reshard_operand =
      ReshardDataForSlicing(hlo->slice_strides(), hlo->slice_starts(),
                            hlo->slice_limits(), operand, sharding, &b_);
  if (!reshard_operand.has_value()) {
    return DefaultAction(hlo);
  }
  TF_RET_CHECK(!reshard_operand->dynamic_slice_index_on_output.has_value());
  HloInstruction* final_operand = SliceDataFromWindowReshard(
      *reshard_operand, hlo->slice_strides(), hlo->shape(), sharding, &b_);

  SetPartitionedHlo(hlo, [&] {
    if (final_operand != reshard_operand->sharded_input) {
      return final_operand;
    }
    // Create a copy so that it will not share the resharding cache.
    return b_.AddInstruction(HloInstruction::CreateUnary(
        final_operand->shape(), HloOpcode::kCopy, final_operand));
  });

  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleSort(HloInstruction* hlo) {
  HloSharding sharding = hlo->sharding();
  if (sharding.HasUniqueDevice()) {
    return DefaultAction(hlo);
  }
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
    const int64_t sort_dim = sort->sort_dimension();
    auto input = hlo->operand(0);
    auto index = hlo->operand(1);
    const HloSharding& input_sharding = input->sharding();
    const int64_t partition_count =
        input_sharding.tile_assignment().dim(sort_dim);
    const int64_t input_size = input->shape().dimensions(sort_dim);
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
    std::vector<int64_t> replicated_dimensions(
        input->shape().dimensions().begin(), input->shape().dimensions().end());
    replicated_dimensions[sort_dim] = RoundUpTo(input_size, partition_count);
    const Shape replicated_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(element_type, replicated_dimensions),
         ShapeUtil::MakeShape(index_type, replicated_dimensions)});

    // Partition original topk to different shards.
    auto topk_sharding =
        input_sharding.GetTupleSharding(replicated_shape).value();
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
    final_sort->set_sharding(
        HloSharding::Replicate().GetTupleSharding(final_sort->shape()).value());
    PartitionedHlo replicated_sort(final_sort, final_sort->shape(),
                                   MakePartitioningState());
    SetPartitionedHlo(hlo, replicated_sort.Reshard(hlo->sharding()));

    return OkStatus();
  }

  if (hlo->shape().IsTuple()) {
    // Check that all elements are sharded in the same way.
    if (hlo->shape().tuple_shapes_size() == 0) {
      return DefaultAction(hlo);
    }
    sharding = hlo->sharding().GetSubSharding(hlo->shape(), {0});
    for (int64_t i = 1; i < hlo->operand_count(); ++i) {
      if (sharding != hlo->sharding().GetSubSharding(hlo->shape(), {i})) {
        return DefaultAction(hlo);
      }
    }
  }
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  for (int64_t dim : hlo->dimensions()) {
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleTranspose(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  std::vector<int64_t> inverse_dimensions(hlo->shape().rank());
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleReshape(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  auto operand = GetPartitionedHlo(hlo->operand(0));
  // The output shape is the source and the operand shape is the target to get
  // the aligned sharding for the operand.
  std::optional<HloSharding> desired_operand_sharding =
      hlo_sharding_util::ReshapeSharding(hlo->shape(), hlo->operand(0)->shape(),
                                         hlo->sharding());
  // Use the desired operand sharding only if the number of tiles returned
  // matches the number of tiles in the output.
  if (desired_operand_sharding.has_value() &&
      hlo->sharding().NumTiles() == desired_operand_sharding->NumTiles()) {
    auto operand_hlo = operand.Reshard(*desired_operand_sharding).hlo();
    SetPartitionedHlo(hlo, [&] {
      return b_.AddInstruction(hlo->CloneWithNewOperands(
          MakePartitionedShape(hlo->shape(), hlo->sharding()), {operand_hlo}));
    });
    return OkStatus();
  }
  std::optional<HloSharding> desired_output_sharding =
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
    return OkStatus();
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
  int64_t input_sharded_dim = *maybe_input_sharded_dim;
  int64_t output_sharded_dim = *maybe_output_sharded_dim;
  // Check that the major dims before the sharded dim have the same total size
  // for input and output.
  int64_t input_major_dims_size = 1;
  for (int64_t i = 0; i < input_sharded_dim; ++i) {
    input_major_dims_size *= operand.base_shape().dimensions(i);
  }
  int64_t output_major_dims_size = 1;
  for (int64_t i = 0; i < output_sharded_dim; ++i) {
    output_major_dims_size *= hlo->shape().dimensions(i);
  }
  if (input_major_dims_size != output_major_dims_size) {
    return DefaultAction(hlo);
  }
  // Fix potential device ordering mismatch in tile assignment.
  Array<int64_t> new_input_tile_assignment = sharding.tile_assignment();
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

  int64_t input_dim_size = operand.base_shape().dimensions(input_sharded_dim);
  int64_t output_dim_size = hlo->shape().dimensions(output_sharded_dim);
  auto input_shard_shape =
      MakePartitionedShape(operand.base_shape(), operand.sharding());
  auto output_shard_shape = MakePartitionedShape(hlo->shape(), sharding);
  if (input_dim_size % output_dim_size == 0) {
    // Split dim.
    int64_t split_factor = input_dim_size / output_dim_size;
    int64_t output_shard_size =
        output_shard_shape.dimensions(output_sharded_dim);
    // Use halo exchange to fix misaligned data.
    Window window;
    for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
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
    return OkStatus();
  } else if (output_dim_size % input_dim_size == 0) {
    // Merge dims.
    int64_t merge_factor = output_dim_size / input_dim_size;
    // First reshape locally. (The sharded dimension could include padded data.)
    auto tmp_shard_shape = output_shard_shape;
    tmp_shard_shape.set_dimensions(
        output_sharded_dim,
        input_shard_shape.dimensions(input_sharded_dim) * merge_factor);
    auto tmp_reshape = b_.AddInstruction(
        HloInstruction::CreateReshape(tmp_shard_shape, operand.hlo()));
    tmp_reshape->set_sharding(hlo->sharding());
    auto tmp_full_shape = tmp_shard_shape;
    tmp_full_shape.set_dimensions(
        output_sharded_dim, tmp_shard_shape.dimensions(output_sharded_dim) *
                                num_partitions_ / replication_count);
    auto tmp_output =
        PartitionedHlo(tmp_reshape, tmp_full_shape, MakePartitioningState());

    // Use halo exchange to fix misaligned data.
    Window window;
    for (int64_t i = 0; i < tmp_shard_shape.rank(); ++i) {
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
    return OkStatus();
  }
  return DefaultAction(hlo);
}

Status SpmdPartitioningVisitor::HandleIota(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();
  if (sharding.IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  SetPartitionedHlo(hlo, [&] {
    int64_t dimension = Cast<HloIotaInstruction>(hlo)->iota_dimension();
    auto iota = b_.AddInstruction(HloInstruction::CreateIota(
        MakePartitionedShape(hlo->shape(), sharding), dimension));

    if (sharding.tile_assignment().dim(dimension) > 1) {
      auto partition_ordinals = MakeTiledPartitionOrdinals(
          sharding, MakePartitioningState().partition_id, &b_);
      auto multiplier = b_.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(iota->shape().dimensions(dimension))));
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

  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleSingleDevice(const HloInstruction* hlo) {
  TF_RET_CHECK(hlo->sharding().HasUniqueDevice());
  int64_t device = hlo->sharding().GetUniqueDevice();
  const HloSharding sharding = HloSharding::AssignDevice(device);

  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> operand_shapes;
  const auto& old_operands = hlo->operands();
  const auto old_operands_size = old_operands.size();
  operands.reserve(old_operands_size);
  operand_shapes.reserve(old_operands_size);
  for (const HloInstruction* operand : old_operands) {
    operands.push_back(GetPartitionedHlo(operand).Reshard(sharding).hlo());
    operand_shapes.push_back(&operand->shape());
  }
  auto operand = b_.AddInstruction(HloInstruction::CreateTuple(operands));
  auto operand_shape = ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);

  auto on_device = b_.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(device)));
  auto pred = b_.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), MakePartitioningState().partition_id,
      on_device, ComparisonDirection::kEq));

  SpmdBuilder true_b("true_computation", visiting_hlo_);
  HloComputation* true_computation;
  {
    auto param = true_b.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, operand_shape, "true_branch_param"));
    std::vector<HloInstruction*> new_operands;
    for (int64_t i = 0; i < operands.size(); ++i) {
      new_operands.push_back(true_b.AddInstruction(
          HloInstruction::CreateGetTupleElement(*operand_shapes[i], param, i)));
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleAllReduce(HloInstruction* hlo) {
  if (hlo->IsCrossReplicaAllReduce() && hlo->operand_count() == 1) {
    return HandleElementwise(hlo);
  }
  if (hlo->channel_id()) {
    TF_RET_CHECK(hlo->operand_count() == 1)
        << "SPMD partitioner supports only single-operand allreduce in manual "
           "partitioning mode.";
    if (hlo->sharding().IsManual()) {
      return HandleElementwise(hlo);
    }
    TF_RET_CHECK(hlo->sharding().IsManualSubgroup())
        << "Cross-partition allreduce must be in (partial) manual partitioning "
           "mode.";
    auto* ar = Cast<HloAllReduceInstruction>(hlo);
    TF_RET_CHECK(ar->use_global_device_ids())
        << "Cross-partition allreduce in partial manual partitioning mode must "
           "use global device IDs.";
    absl::flat_hash_map<int64_t, int64_t> partition_to_group_id;
    hlo->sharding().tile_assignment().Each(
        [&](absl::Span<const int64_t> indices, int64_t partition) {
          int64_t group_id = 0;
          for (int64_t i = 0; i < indices.size(); ++i) {
            if (i == hlo->sharding().SubgroupManualDim()) {
              continue;
            }
            group_id *= hlo->sharding().tile_assignment().dim(i);
            group_id += indices[i];
          }
          partition_to_group_id[partition] = group_id;
        });
    for (const auto& group : ar->replica_groups()) {
      int64_t first_partition = group.replica_ids(0) % num_partitions_;
      for (int64_t device : group.replica_ids()) {
        int64_t partition = device % num_partitions_;
        if (partition_to_group_id[partition] !=
            partition_to_group_id[first_partition]) {
          return InvalidArgumentStrCat(
              "Manual all-reduce across devices that belong to different "
              "manual subgroups: ",
              ar->ToString());
        }
      }
    }
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
  std::vector<int64_t> new_dims;
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
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
  return OkStatus();
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
    std::vector<int64_t> start_indices(hlo->shape().rank(), 0);
    auto constant = b_.AddInstruction(HloInstruction::CreateConstant(
        literal.Slice(start_indices, shard_shape.dimensions())));
    *constant->mutable_shape() = shard_shape;
    return constant;
  });
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleDynamicSlice(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->sharding().tile_assignment().dim(i) != 1 &&
        hlo->dynamic_slice_sizes()[i] !=
            hlo->operand(0)->shape().dimensions(i)) {
      // We currently do not partition the sliced dimensions.
      return DefaultAction(hlo);
    }
  }
  std::vector<HloInstruction*> new_indices(hlo->shape().rank());
  auto new_input =
      GetPartitionedHlo(hlo->operand(0)).Reshard(hlo->sharding()).hlo();
  for (int64_t i = 0; i < new_indices.size(); ++i) {
    if (hlo->dynamic_slice_sizes()[i] ==
        hlo->operand(0)->shape().dimensions(i)) {
      // Trivial slice dim: index must be clampped to 0.
      new_indices[i] = CreateZero(hlo->operand(i + 1)->shape(), &b_);
      continue;
    }
    // Replicate the indices.;
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleDynamicUpdateSlice(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }

  std::vector<int64_t> partitioned_slice_dims;
  std::vector<int64_t> slice_dims;
  std::vector<int64_t> partitioned_non_slice_dims;
  std::vector<int64_t> partitioned_slice_offsets;
  bool any_non_constant_sliced_dim = false;
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->operand(1)->shape().dimensions(i) != hlo->shape().dimensions(i)) {
      slice_dims.push_back(i);
      int64_t slice_size = hlo->operand(1)->shape().dimensions(i);
      if (hlo->sharding().tile_assignment().dim(i) != 1) {
        if (!hlo->operand(i + 2)->IsConstant() && slice_size != 1) {
          any_non_constant_sliced_dim = true;
          continue;
        }
        partitioned_slice_dims.push_back(i);
        // Set partitioned_slice_offsets to -1 when slice_size is 1.
        if (slice_size == 1) {
          partitioned_slice_offsets.push_back(-1);
        } else {
          partitioned_slice_offsets.push_back(
              hlo->operand(i + 2)->literal().Get<int>({}));
        }
      }
    } else if (hlo->sharding().tile_assignment().dim(i) != 1) {
      partitioned_non_slice_dims.push_back(i);
    }
  }
  auto handle_with_replicate_slice_dims = [&]() {
    HloSharding replicated_sharding =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
            hlo->operand(0)->sharding(), partitioned_non_slice_dims);
    auto base = GetPartitionedHlo(hlo->operand(0)).Reshard(replicated_sharding);
    auto operand =
        GetPartitionedHlo(hlo->operand(1)).Reshard(replicated_sharding);
    std::vector<HloInstruction*> new_indices(hlo->shape().rank());
    for (int64_t i = 0; i < new_indices.size(); ++i) {
      // Replicate the indices.
      new_indices[i] = GetPartitionedHlo(hlo->operand(i + 2))
                           .Reshard(HloSharding::Replicate())
                           .hlo();
    }
    auto dus = b_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        base.hlo()->shape(), base.hlo(), operand.hlo(), new_indices));
    dus->set_sharding(replicated_sharding);
    SetPartitionedHlo(hlo, PartitionedHlo(dus, base.base_shape(), base.state())
                               .Reshard(hlo->sharding()));
  };
  if (any_non_constant_sliced_dim) {
    if (partitioned_non_slice_dims.empty()) {
      return DefaultAction(hlo);
    }
    handle_with_replicate_slice_dims();
    return OkStatus();
  }

  // Handle when there is slice dim partitioned.
  if (!partitioned_slice_dims.empty()) {
    auto add_hlo = [&](std::unique_ptr<HloInstruction> to_add) {
      return b_.AddInstruction(std::move(to_add));
    };
    std::vector<HloInstruction*> new_indices(hlo->shape().rank());
    for (int64_t i = 0; i < new_indices.size(); ++i) {
      if (hlo->operand(1)->shape().dimensions(i) ==
          hlo->shape().dimensions(i)) {
        new_indices[i] = CreateZero(hlo->operand(i + 2)->shape(), &b_);
        continue;
      }
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

    // TODO(wangtao): use collective permute for sharded update.
    HloInstruction* replicate_update =
        GetPartitionedHlo(hlo->operand(1)).Reshard(update_sharding).hlo();

    const auto& update_shape = replicate_update->shape();
    const auto& partitioned_shape = partitioned_input->shape();
    auto partition_ordinals = MakeTiledPartitionOrdinals(
        hlo->sharding(), MakePartitioningState().partition_id, &b_);
    HloInstruction* all_dims_within_partition = add_hlo(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));

    for (int i = 0; i < partitioned_slice_dims.size(); ++i) {
      int dim = partitioned_slice_dims[i];
      // Calculate per partition size.
      const int64_t per_partition_size = partitioned_shape.dimensions(dim);

      // Only update within a single partition is supported.
      // Will ignore this check when slice size is 1 where
      // partitioned_slice_offsets[i] is -1.
      if ((partitioned_slice_offsets[i] != -1) &&
          (partitioned_slice_offsets[i] / per_partition_size) !=
              ((partitioned_slice_offsets[i] + update_shape.dimensions(dim) -
                1) /
               per_partition_size)) {
        handle_with_replicate_slice_dims();
        return Status::OK();
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
    return OkStatus();
  }

  // Partition non slice dims only.
  std::vector<HloInstruction*> new_indices(hlo->shape().rank());
  auto new_input =
      GetPartitionedHlo(hlo->operand(0)).Reshard(hlo->sharding()).hlo();
  auto new_update =
      GetPartitionedHlo(hlo->operand(1)).Reshard(hlo->sharding()).hlo();
  for (int64_t i = 0; i < new_indices.size(); ++i) {
    if (hlo->operand(1)->shape().dimensions(i) == hlo->shape().dimensions(i)) {
      new_indices[i] = CreateZero(hlo->operand(i + 2)->shape(), &b_);
      continue;
    }
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
  return OkStatus();
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
  return OkStatus();
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
    return OkStatus();
  }
  auto sharding = hlo->sharding().GetSubSharding(hlo->shape(), {0});
  auto shard_shape = MakePartitionedShape(shape, sharding);
  if (EvenlyPartitions(shape, sharding)) {
    SetPartitionedHlo(hlo, [&]() {
      return b_.AddInstruction(HloInstruction::CreateInfeed(
          shard_shape, token, hlo->infeed_config()));
    });
    return OkStatus();
  }

  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }

  // Create a branch for each unique partitioned shape.
  std::vector<Shape> per_branch_partitioned_shapes;
  std::vector<int32_t> conditional_branch_indices(num_partitions_);
  for (int64_t i = 0; i < num_partitions_; ++i) {
    auto partitioned_shape =
        MakeNonPaddedShapeForGivenPartition(shape, sharding, i);
    int64_t matching_existing_index = 0;
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
  auto state = MakePartitioningState();
  if (per_branch_partitioned_shapes.size() == num_partitions_) {
    // Use partition ID as the branch index if each partition has its own
    // branch.
    branch_index = state.partition_id;
    // PartitionId's output is U32 but conditional requires S32.
    if (branch_index->shape().element_type() != S32) {
      branch_index = b_.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(branch_index->shape(), S32),
          branch_index));
    }
  } else {
    // Otherwise, use a constant table to look up the branch index.
    auto branch_index_table = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<int32_t>(conditional_branch_indices)));
    branch_index = b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::MakeShape(S32, {1}), branch_index_table,
        {state.partition_id}, {1}));
    branch_index = b_.AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(S32, {}), branch_index));
  }

  std::vector<HloComputation*> branches(per_branch_partitioned_shapes.size());
  for (int64_t i = 0; i < branches.size(); ++i) {
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
          for (int64_t i = 0; i < padded_elements.size(); ++i) {
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
        const Shape& pad_shape = ShapeUtil::GetSubshape(
            shard_shape, ShapeIndexView(index).subspan(1));
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandlePad(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  auto lhs = GetPartitionedHlo(hlo->operand(0));
  auto replicated_rhs = GetPartitionedHlo(hlo->operand(1))
                            .Reshard(HloSharding::Replicate())
                            .hlo();
  auto reshard_operand =
      ReshardDataForPad(replicated_rhs, hlo->padding_config(), lhs,
                        hlo->shape(), hlo->sharding(), &b_);
  if (!reshard_operand.has_value()) {
    return DefaultAction(hlo);
  }
  auto* sharded_pad =
      PadDataFromWindowReshard(*reshard_operand, replicated_rhs, &b_);

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
  return OkStatus();
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleReduce(HloInstruction* hlo) {
  int64_t input_count = 1;
  if (hlo->shape().IsTuple()) {
    input_count = hlo->shape().tuple_shapes_size();
    CHECK_GT(input_count, 0);
  }
  if (hlo->sharding().HasUniqueDevice()) {
    std::vector<HloInstruction*> new_operands(input_count * 2, nullptr);
    for (auto i = 0; i != input_count; ++i) {
      // Handle variadic reduce sharding.
      HloSharding subsharding =
          hlo->sharding().IsTuple()
              ? hlo->sharding().GetSubSharding(hlo->shape(), {i})
              : hlo->sharding();
      CHECK(!subsharding.IsTuple() && subsharding.HasUniqueDevice());
      // Partition reduce operands and init values.
      new_operands[i] =
          GetPartitionedHlo(hlo->operand(i)).Reshard(subsharding).hlo();
      new_operands[input_count + i] =
          GetPartitionedHlo(hlo->operand(input_count + i))
              .Reshard(subsharding)
              .hlo();
    }
    auto clone = b_.AddInstruction(
        hlo->CloneWithNewOperands(hlo->shape(), new_operands));
    clone->set_sharding(hlo->sharding());
    SetPartitionedHlo(
        hlo, PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
                 .Reshard(hlo->sharding()));
    return OkStatus();
  }

  std::vector<PartitionedHlo> inputs;
  std::vector<HloInstruction*> inits;
  std::vector<int64_t> preserved_dims;
  for (int64_t i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
    if (!absl::c_linear_search(hlo->dimensions(), i)) {
      preserved_dims.push_back(i);
    }
  }

  for (int64_t operand_id = 0; operand_id < input_count; ++operand_id) {
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

  std::vector<const Shape*> new_operand_shapes(input_count * 2);
  for (int64_t i = 0; i < input_count; ++i) {
    new_operand_shapes[i] = &inputs[i].hlo()->shape();
    new_operand_shapes[i + input_count] = &inits[i]->shape();
  }
  // Create the shard shape of the reduce result.
  TF_ASSIGN_OR_RETURN(
      auto reduce_shape,
      ShapeInference::InferReduceShape(new_operand_shapes, hlo->dimensions(),
                                       hlo->to_apply()->ComputeProgramShape()));

  std::vector<HloInstruction*> input_hlos(input_count);
  for (int64_t i = 0; i < input_count; ++i) {
    input_hlos[i] = inputs[i].hlo();
  }
  auto local_reduce = b_.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, input_hlos, inits, hlo->dimensions(), hlo->to_apply()));

  SetPartitionedHlo(hlo, [&]() {
    HloInstruction* reduce = local_reduce;
    const bool reduce_sharded_dimension =
        !inputs[0].sharding().IsTileMaximal() &&
        absl::c_any_of(hlo->dimensions(), [&](int64_t i) {
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
        auto grouped = hlo_sharding_util::GroupShardingOnDims(
            inputs[0].sharding(), preserved_dims);
        auto grouped_state = CreatePerGroupPartitioningState(
            inputs[0].state(), grouped.device_groups, &b_);
        std::vector<HloInstruction*> all_gathered_partial_results(input_count);
        for (int64_t i = 0; i < input_count; ++i) {
          auto gte = b_.AddInstruction(HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetTupleElementShape(reduce_shape, i), local_reduce,
              i));
          auto expanded_shape = input_hlos[i]->shape();
          auto all_gather_shape = input_hlos[i]->shape();
          for (int64_t dim : hlo->dimensions()) {
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
  return OkStatus();
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleWhile(HloInstruction* hlo) {
  const HloSharding& sharding = hlo->sharding();

  // Shardings for the body parameter, body root, and cond parameter must be
  // the same, and the condition root must be replicated so that all partitions
  // follow the same control flow.
  hlo->while_condition()->parameter_instruction(0)->set_sharding(sharding);
  hlo->while_body()->parameter_instruction(0)->set_sharding(sharding);
  const HloSharding& cond_root_sharding =
      hlo->while_condition()->root_instruction()->sharding();
  TF_RETURN_IF_ERROR(partitioner_
                         ->PartitionComputation(hlo->while_condition(),
                                                cond_root_sharding.IsManual()
                                                    ? cond_root_sharding
                                                    : HloSharding::Replicate(),
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleConditional(HloInstruction* hlo) {
  std::vector<HloInstruction*> branch_args;
  for (int64_t i = 0; i < hlo->branch_count(); ++i) {
    HloComputation* computation = hlo->branch_computation(i);

    // Shardings of the branch computation parameter and its argument must be
    // the same.
    computation->parameter_instruction(0)->set_sharding(
        hlo->operand(i + 1)->sharding());
    branch_args.push_back(GetPartitionedHlo(hlo->operand(i + 1)).hlo());
  }

  // The root of the branch computations must follow the sharding of the
  // conditional instruction.
  for (int64_t i = 0; i < hlo->branch_count(); ++i) {
    HloComputation* computation = hlo->branch_computation(i);
    TF_RETURN_IF_ERROR(partitioner_
                           ->PartitionComputation(computation, hlo->sharding(),
                                                  next_channel_id_, logger_)
                           .status());
  }
  SetPartitionedHlo(hlo, [&] {
    HloInstruction* cond = GetPartitionedHlo(hlo->operand(0)).hlo();
    if (!hlo->operand(0)->sharding().IsManual()) {
      // We replicate the predicate of the conditional (the first operand) so
      // that all partitions follow the same control flow.
      cond = GetPartitionedHlo(hlo->operand(0))
                 .Reshard(HloSharding::Replicate())
                 .hlo();
    }
    return b_.AddInstruction(HloInstruction::CreateConditional(
        MakePartitionedShape(hlo->shape(), hlo->sharding()), cond,
        hlo->called_computations(), branch_args));
  });
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleOptimizationBarrier(HloInstruction* hlo) {
  return HandleElementwise(hlo);
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
    return OkStatus();
  }

  // Create a branch for each unique partitioned shape.
  std::vector<Shape> per_branch_partitioned_shapes;
  std::vector<int32_t> conditional_branch_indices(num_partitions_);
  for (int64_t i = 0; i < num_partitions_; ++i) {
    auto partitioned_shape =
        MakeNonPaddedShapeForGivenPartition(shape, sharding, i);
    int64_t matching_existing_index = 0;
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
  auto state = MakePartitioningState();
  if (per_branch_partitioned_shapes.size() == num_partitions_) {
    // Use partition ID as the branch index if each partition has its own
    // branch.
    branch_index = state.partition_id;
    // PartitionId's output is U32 but conditional requires S32.
    if (branch_index->shape().element_type() != S32) {
      branch_index = b_.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(branch_index->shape(), S32),
          branch_index));
    }
  } else {
    // Otherwise, use a constant table to look up the branch index.
    auto branch_index_table = b_.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<int32_t>(conditional_branch_indices)));
    branch_index = b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::MakeShape(S32, {1}), branch_index_table, {partition_id_},
        {1}));
    branch_index = b_.AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(S32, {}), branch_index));
  }

  // Create conditional for the outfeed.
  std::vector<HloComputation*> branches(per_branch_partitioned_shapes.size());
  for (int64_t i = 0; i < branches.size(); ++i) {
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
          for (int64_t i = 0; i < slice_elements.size(); ++i) {
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
          std::vector<int64_t> start_indices(slice_shape.rank(), 0);
          std::vector<int64_t> slice_strides(slice_shape.rank(), 1);
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleRng(HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }
  auto clone_from_original = [&](const HloSharding& shared_sharding) {
    std::vector<HloInstruction*> new_operands;
    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      new_operands.push_back(
          GetPartitionedHlo(hlo->operand(i)).Reshard(shared_sharding).hlo());
    }
    auto clone = b_.AddInstruction(
        hlo->CloneWithNewOperands(hlo->shape(), new_operands));
    clone->set_sharding(shared_sharding);
    return clone;
  };

  if (hlo->sharding().IsManual()) {
    SetPartitionedHlo(hlo,
                      [&] { return clone_from_original(hlo->sharding()); });
    return OkStatus();
  }

  if (hlo->sharding().IsReplicated()) {
    SetPartitionedHlo(hlo, [&] {
      // Run on a single device (0) and distribute the data to all other cores.
      auto clone = clone_from_original(HloSharding::AssignDevice(0));
      return PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
          .Reshard(HloSharding::Replicate())
          .hlo();
    });
    return OkStatus();
  }

  TF_RET_CHECK(!hlo->sharding().IsTileMaximal());
  // Replicate the operands and run partitioned Rng on all devices.
  std::vector<HloInstruction*> new_operands;
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
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
    std::vector<int64_t> group_dims(
        hlo->sharding().tile_assignment().num_dimensions() - 1);
    std::iota(group_dims.begin(), group_dims.end(), 0);
    auto sharding_grouped =
        hlo_sharding_util::GroupShardingOnDims(hlo->sharding(), group_dims);
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
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleReduceWindow(HloInstruction* hlo) {
  if (hlo->sharding().IsTileMaximal()) {
    return DefaultAction(hlo);
  }
  HloReduceWindowInstruction* reduce_window =
      Cast<HloReduceWindowInstruction>(hlo);
  absl::Span<HloInstruction* const> input_arrays = reduce_window->inputs();
  absl::Span<HloInstruction* const> init_values = reduce_window->init_values();
  int64_t input_idx = 0;
  absl::InlinedVector<PartitionedHlo::WindowedInputShardReturnValue, 2>
      sharded_results;
  absl::InlinedVector<const Shape*, 2> sharded_input_shapes,
      replicated_init_shapes;
  absl::InlinedVector<HloInstruction*, 2> sharded_inputs, replicated_inits;
  for (const HloInstruction* input_array : input_arrays) {
    PartitionedHlo& operand = GetPartitionedHlo(input_array);
    // Replicate init
    PartitionedHlo replicated_init = GetPartitionedHlo(init_values[input_idx++])
                                         .Reshard(HloSharding::Replicate());
    auto resharded_operand_and_window = operand.ReshardAsWindowedInput(
        hlo->window(), hlo->sharding(), replicated_init.hlo());
    if (!resharded_operand_and_window.has_value()) {
      return DefaultAction(hlo);
    }
    sharded_results.push_back(resharded_operand_and_window.value());
    sharded_inputs.push_back(resharded_operand_and_window->sharded_input);
    sharded_input_shapes.push_back(&sharded_inputs.back()->shape());
    replicated_inits.push_back(replicated_init.hlo());
    replicated_init_shapes.push_back(&replicated_inits.back()->shape());
  }
  TF_ASSIGN_OR_RETURN(Shape sharded_rw_shape,
                      ShapeInference::InferReduceWindowShape(
                          sharded_input_shapes, replicated_init_shapes,
                          sharded_results[0].shard_window,
                          hlo->to_apply()->ComputeProgramShape()));
  HloSharding result_sharding =
      (hlo->shape().IsTuple())
          ? hlo->sharding().GetTupleSharding(hlo->shape()).value()
          : hlo->sharding();
  Shape shard_shape = MakePartitionedShape(hlo->shape(), result_sharding);
  if (shard_shape.has_layout()) {
    *sharded_rw_shape.mutable_layout() = shard_shape.layout();
  }
  SetPartitionedHlo(hlo, [&]() {
    HloInstruction* sharded_rw =
        b_.AddInstruction(HloInstruction::CreateReduceWindow(
            sharded_rw_shape, sharded_inputs, replicated_inits,
            sharded_results[0].shard_window, hlo->to_apply()));
    if (!sharded_results[0].dynamic_slice_index_on_output.has_value()) {
      CHECK(ShapeUtil::Compatible(shard_shape, sharded_rw->shape()))
          << shard_shape << " vs " << sharded_rw->shape() << "\n";
      return sharded_rw;
    }
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, sharded_rw,
        *sharded_results[0].dynamic_slice_index_on_output,
        shard_shape.dimensions()));
  });
  return OkStatus();
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

  auto state = MakePartitioningState();
  auto partition_ordinals =
      MakeTiledPartitionOrdinals(hlo->sharding(), state.partition_id, &b_);

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
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
    int64_t shard_count = hlo->sharding().tile_assignment().dim(i);
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

    int64_t max_windows =
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
    int64_t padded_data_size =
        (limit_window[i].Calculate(shard_count - 1) - 1) * wd.stride() +
        wd.size();
    int64_t data_shard_size = (max_windows - 1) * wd.stride() + wd.size();
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
  for (int64_t i = 0; i < window_on_shard.dimensions_size(); ++i) {
    int64_t shard_count = hlo->sharding().tile_assignment().dim(i);
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
    for (int64_t i = 0; i < window_on_shard.dimensions_size(); ++i) {
      if (hlo->sharding().tile_assignment().dim(i) == 1) {
        continue;
      }
      int64_t pad_low = hlo->window().dimensions(i).padding_low();
      auto left_halo_size =
          data_left_halo_sizes[i].Calculate(partition_ordinals[i], &b_);
      if (data_left_halo_sizes[i].Calculate(0) == pad_low) {
        slice_offsets[i] = left_halo_size;
      } else {
        auto is_shard0 = b_.AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), zero, partition_ordinals[i],
            ComparisonDirection::kEq));
        auto pad_low_hlo = b_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(pad_low)));
        slice_offsets[i] = b_.AddInstruction(HloInstruction::CreateTernary(
            zero->shape(), HloOpcode::kSelect, is_shard0, pad_low_hlo,
            left_halo_size));
      }
    }
    return b_.AddInstruction(HloInstruction::CreateDynamicSlice(
        shard_shape, sharded_select_and_scatter, slice_offsets,
        shard_shape.dimensions()));
  });
  return OkStatus();
}

Status SpmdPartitioningVisitor::HandleTuple(HloInstruction* hlo) {
  std::vector<HloInstruction*> new_operands;
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    new_operands.push_back(
        GetPartitionedHlo(hlo->operand(i))
            .Reshard(hlo->sharding().GetSubSharding(hlo->shape(), {i}))
            .hlo());
  }
  SetPartitionedHlo(hlo, [&]() {
    return b_.AddInstruction(HloInstruction::CreateTuple(new_operands));
  });
  return OkStatus();
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
  absl::flat_hash_map<HloComputation*, HloComputation*> replacement;
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

SPMDCollectiveOpsCreator GetDefaultCollectiveOpsCreator(int64_t num_partitions,
                                                        int64_t num_replicas) {
  return {
      [](SpmdBuilder* b) {
        return b->AddInstruction(HloInstruction::CreatePartitionId());
      },
      [num_replicas, num_partitions](
          SpmdBuilder* b, HloInstruction* operand, HloComputation* reduction,
          const std::vector<std::vector<int64_t>>& partition_subgroups,
          int64_t channel_id) {
        if (partition_subgroups.size() <= 1) {
          std::vector<ReplicaGroup> groups(num_replicas);
          // TODO(yuanzx): Unify subgroup definition with AllToAll.
          for (int64_t i = 0; i < num_replicas; ++i) {
            groups[i].add_replica_ids(i);
          }
          return b->AddInstruction(HloInstruction::CreateAllReduce(
              operand->shape(), {operand}, reduction, groups,
              /*constrain_layout=*/false, channel_id,
              /*use_global_device_ids=*/false));
        }

        std::vector<ReplicaGroup> device_groups;
        device_groups.reserve(partition_subgroups.size() * num_replicas);
        for (int64_t i = 0; i < num_replicas; ++i) {
          for (const auto& pgroup : partition_subgroups) {
            device_groups.emplace_back();
            for (int64_t pid : pgroup) {
              device_groups.back().add_replica_ids(i * num_partitions + pid);
            }
          }
        }
        return b->AddInstruction(HloInstruction::CreateAllReduce(
            operand->shape(), {operand}, reduction, device_groups,
            /*constrain_layout=*/false, channel_id,
            /*use_global_device_ids=*/true));
      },
      [num_partitions](SpmdBuilder* b, HloInstruction* operand,
                       std::vector<std::pair<int64_t, int64_t>>& src_dst_pairs,
                       int64_t channel_id) {
        /* optimize trivial collective permute */
        if (src_dst_pairs.empty()) {
          // If the src/dst pairs are empty, then the collective permute just
          // initializes the output to zero.
          return CreateZero(operand->shape(), b);
        } else {
          // A collective-permute is a copy if all pairs are "identity" and
          // all partitions are listed.
          bool is_copy =
              src_dst_pairs.size() == num_partitions &&
              absl::c_all_of(src_dst_pairs,
                             [](const std::pair<int64_t, int64_t>& pair) {
                               return pair.first == pair.second;
                             });
          if (is_copy) {
            return operand;
          } else {
            return b->AddInstruction(HloInstruction::CreateCollectivePermute(
                operand->shape(), operand, src_dst_pairs, channel_id));
          }
        }
      },
      [](SpmdBuilder* b, absl::Span<HloInstruction* const> operands,
         const std::vector<std::vector<int64_t>>& partition_subgroups,
         int64_t channel_id, std::optional<int64_t> split_dimension) {
        std::vector<Shape> shapes(operands.size(), operands[0]->shape());
        const Shape output_shape = (shapes.size() == 1)
                                       ? shapes[0]
                                       : ShapeUtil::MakeTupleShape(shapes);
        std::vector<ReplicaGroup> groups(partition_subgroups.size());
        for (int64_t i = 0; i < groups.size(); ++i) {
          for (int64_t id : partition_subgroups[i]) {
            groups[i].add_replica_ids(id);
          }
        }
        return b->AddInstruction(HloInstruction::CreateAllToAll(
            output_shape, operands, groups,
            /*constrain_layout=*/false, channel_id, split_dimension));
      },
      [num_replicas, num_partitions](
          SpmdBuilder* b, HloInstruction* operand, const Shape& ag_shape,
          const std::vector<std::vector<int64_t>>& partition_subgroups,
          int64_t channel_id, int64_t all_gather_dimension) {
        std::vector<ReplicaGroup> device_groups;
        device_groups.reserve(partition_subgroups.size() * num_replicas);
        for (int64_t i = 0; i < num_replicas; ++i) {
          for (const auto& pgroup : partition_subgroups) {
            device_groups.emplace_back();
            for (int64_t pid : pgroup) {
              device_groups.back().add_replica_ids(i * num_partitions + pid);
            }
          }
        }
        return b->AddInstruction(HloInstruction::CreateAllGather(
            ag_shape, {operand}, all_gather_dimension, device_groups,
            /*constrain_layout=*/false, channel_id,
            /*use_global_device_ids=*/true));
      },
  };
}

SpmdPartitioner::SpmdPartitioner(int64_t num_partitions, int64_t num_replicas,
                                 SpmdPartitionerOptions options)
    : SpmdPartitioner(
          num_partitions, num_replicas, std::move(options),
          GetDefaultCollectiveOpsCreator(num_partitions, num_replicas)) {}

HloInstruction* SpmdPartitioner::AllGatherShards(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator) {
  return AllGatherShardsInternal(b, operand, sharding, next_channel_id,
                                 selected_dims, collectives_creator,
                                 /*per_dim_ag=*/true);
}

HloInstruction* SpmdPartitioner::AllGatherShardsInternal(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator, bool per_dim_ag) {
  if (selected_dims.empty()) {
    return operand;
  }
  CHECK(!sharding.IsTileMaximal());
  if (per_dim_ag || selected_dims.size() == 1) {
    HloInstruction* result = operand;
    Shape result_shape = operand->shape();
    for (auto it = selected_dims.rbegin(); it != selected_dims.rend(); ++it) {
      if (sharding.tile_assignment().dim(*it) == 1) {
        continue;
      }
      auto partition_subgroups =
          GetPartitionGroupsForReplication(sharding, {*it});
      result_shape.set_dimensions(
          *it, result_shape.dimensions(*it) * partition_subgroups[0].size());
      result = collectives_creator.create_cross_partition_all_gather(
          b, result, result_shape, partition_subgroups, (*next_channel_id)++,
          /*all_gather_dimension=*/*it);
    }
    return result;
  }
  std::vector<int64_t> shape;
  shape.push_back(1);
  for (int64_t dim : operand->shape().dimensions()) {
    shape.push_back(dim);
  }
  // Add one leading dimension to gather all partitions.
  auto reshape = b->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(operand->shape().element_type(), shape), operand));
  HloInstruction* result = reshape;
  auto partition_subgroups =
      GetPartitionGroupsForReplication(sharding, selected_dims);
  shape[0] *= partition_subgroups[0].size();
  result = collectives_creator.create_cross_partition_all_gather(
      b, result, ShapeUtil::MakeShape(operand->shape().element_type(), shape),
      partition_subgroups, (*next_channel_id)++,
      /*all_gather_dimension=*/0);
  // If n > 1 dimensions are partitioned, split the leading dimension to n.
  std::vector<int64_t> tiled_dims;
  for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (sharding.tile_assignment().dim(i) > 1 &&
        absl::c_linear_search(selected_dims, i)) {
      tiled_dims.push_back(i);
    }
  }
  if (tiled_dims.size() > 1) {
    std::vector<int64_t> split_dim_shape;
    split_dim_shape.reserve(tiled_dims.size() + operand->shape().rank());
    for (int64_t i : tiled_dims) {
      split_dim_shape.push_back(sharding.tile_assignment().dim(i));
    }
    for (int64_t dim : operand->shape().dimensions()) {
      split_dim_shape.push_back(dim);
    }
    result = b->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(operand->shape().element_type(), split_dim_shape),
        result));
  }
  // Transpose the gathered dimensions to next to their corresponding
  // partitioned dimensions.
  std::vector<int64_t> xpose_permutation(result->shape().rank());
  int64_t split_dims_added = 0;
  for (int64_t i = 0; i < xpose_permutation.size(); ++i) {
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
          .value(),
      result, xpose_permutation));
  // Reshape to the desired shape.
  auto ag_shape = operand->shape();
  for (int64_t i : tiled_dims) {
    ag_shape.set_dimensions(
        i, ag_shape.dimensions(i) * sharding.tile_assignment().dim(i));
  }
  result = b->AddInstruction(HloInstruction::CreateReshape(ag_shape, result));
  return result;
}

HloInstruction* SpmdPartitioner::AllReduceAlongShardingDims(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
    const SPMDCollectiveOpsCreator& collectives_creator,
    HloComputation* reduction) {
  return AllReduceAlongShardingDimsInternal(
      b, operand, sharding, next_channel_id, selected_dims, collectives_creator,
      reduction, /*per_dim_ar=*/true);
}

HloInstruction* SpmdPartitioner::AllReduceAlongShardingDimsInternal(
    SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
    int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
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
    int64_t* next_channel_id, SpmdLogger* logger) {
  auto visitor =
      CreateVisitor(computation, num_partitions_, num_replicas_,
                    collective_ops_creator_, next_channel_id, logger, options_);
  return visitor->DoPartition(computation, root_sharding, options_);
}

std::unique_ptr<SpmdPartitioningVisitor> SpmdPartitioner::CreateVisitor(
    HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdLogger* logger,
    SpmdPartitionerOptions options) {
  return std::make_unique<SpmdPartitioningVisitor>(
      computation, num_partitions, num_replicas, collective_ops_creator,
      next_channel_id, logger, std::move(options), this);
}

StatusOr<bool> SpmdPartitioner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(PreprocessSharding(module, execution_threads));
  TF_RETURN_IF_ERROR(PreprocessHlos(module, execution_threads));

  XLA_VLOG_LINES(1, SpmdLogger::ReportBeforePartition(
                        *module, options_.report_instruction_count));

  // Add the parameters' and output's shardings to the module.
  std::vector<HloSharding> entry_params_shardings;
  const auto num_parameters = module->entry_computation()->num_parameters();
  entry_params_shardings.reserve(num_parameters);
  for (int64_t i = 0; i < num_parameters; ++i) {
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

  SpmdLogger logger(options_.report_instruction_count,
                    /*disabled=*/!VLOG_IS_ON(1));
  auto program_shape = module->entry_computation()->ComputeProgramShape();
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
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
    for (int64_t i = 0; i < program_shape.parameters_size(); ++i) {
      TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
          program_shape.parameters(i), new_program_shape.parameters(i)))
          << "Parameter shape changed for the entry computation";
    }
  } else {
    // Fix up some bad tiling in entry computation layout.
    auto update_shape = [this](Shape* subshape, const xla::ShapeIndex& index) {
      if (subshape->IsArray()) {
        UpdateLayout(subshape);
      }
    };
    const auto& old_entry_layout = module->entry_computation_layout();
    // Shapes can change but the layout should still remain the same.
    for (int64_t i = 0; i < new_program_shape.parameters_size(); ++i) {
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          old_entry_layout.parameter_shape(i),
          new_program_shape.mutable_parameters(i)));
      ShapeUtil::ForEachMutableSubshape(new_program_shape.mutable_parameters(i),
                                        update_shape);
    }
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        old_entry_layout.result_shape(), new_program_shape.mutable_result()));
    ShapeUtil::ForEachMutableSubshape(new_program_shape.mutable_result(),
                                      update_shape);

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
    pass.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
    pass.AddPass<TupleSimplifier>();
    pass.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
    pass.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pass.AddPass<FlattenCallGraph>();
    TF_RETURN_IF_ERROR(pass.Run(module, execution_threads).status());
  }

  TF_RETURN_IF_ERROR(ClearShardingAttributes(module, execution_threads));
  return changed;
}

Status SpmdPartitioner::PreprocessSharding(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->HasSideEffectNoRecurse() && hlo->opcode() != HloOpcode::kRng) {
        TF_RET_CHECK(hlo->has_sharding())
            << "Side-effect HLO must have sharding: " << hlo->ToString();
        TF_RET_CHECK(!HasReplicatedSharding(hlo->sharding()) ||
                     CanSideEffectingHaveReplicatedSharding(hlo))
            << "side-effect HLO cannot have a replicated sharding: "
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
        std::vector<int64_t> available(num_partitions_);
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
    if (!root_sharding.UniqueDevice().has_value()) {
      if (root_sharding.IsTuple()) {
        TF_RET_CHECK(absl::c_all_of(root_sharding.tuple_elements(),
                                    [](const HloSharding& s) {
                                      return s.IsReplicated() || s.IsManual();
                                    }))
            << "Unsupported entry root sharding: " << root_sharding.ToString();

      } else {
        TF_RET_CHECK(root_sharding.IsReplicated() || root_sharding.IsManual())
            << "Unsupported entry root sharding: " << root_sharding.ToString();
      }
    }

    for (const HloInstruction* param : entry->parameter_instructions()) {
      TF_RET_CHECK(param->has_sharding());
      TF_RET_CHECK(param->sharding().IsReplicated() ||
                   param->sharding().UniqueDevice().has_value())
          << "Unsupported entry parameter sharding:"
          << param->sharding().ToString();
    }
  }

  return OkStatus();
}

Status SpmdPartitioner::PreprocessHlos(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto skip_copy_operands = [](HloInstruction* operand,
                               bool check_single_use =
                                   true) -> HloInstruction* {
    while (operand->user_count() == 1 &&
           operand->opcode() == HloOpcode::kCopy) {
      operand = operand->mutable_operand(0);
    }
    if (check_single_use && operand->user_count() != 1) {
      return nullptr;
    }
    return operand;
  };

  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->sharding().IsTileMaximal() || hlo->sharding().IsManual()) {
        // No need to optimize for tile-maximal or manual sharding.
        continue;
      }

      if (hlo->opcode() == HloOpcode::kSlice) {
        HloInstruction* operand = skip_copy_operands(hlo->mutable_operand(0));
        if (operand == nullptr || operand->sharding() != hlo->sharding()) {
          continue;
        }

        // Merge pad->slice to avoid multiple halo exchanges.
        if (operand->opcode() == HloOpcode::kPad) {
          std::optional<PaddingConfig> merged_padding =
              operand->padding_config();
          bool may_have_multi_halo_exchanges = false;
          for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
            const auto& dim = operand->padding_config().dimensions(i);
            if (dim.interior_padding() != 0 || hlo->slice_strides(i) != 1) {
              merged_padding = std::nullopt;
              break;
            }
            if (hlo->sharding().tile_assignment().dim(i) != 1 &&
                (dim.edge_padding_low() != 0 || dim.edge_padding_high() != 0) &&
                hlo->shape().dimensions(i) != operand->shape().dimensions(i)) {
              // There are padding, slicing, and sharding on this dim.
              may_have_multi_halo_exchanges = true;
            }

            auto* merged_dim = merged_padding->mutable_dimensions(i);
            merged_dim->set_edge_padding_low(dim.edge_padding_low() -
                                             hlo->slice_starts(i));
            merged_dim->set_edge_padding_high(hlo->slice_limits(i) -
                                              operand->shape().dimensions(i));
          }
          if (merged_padding.has_value() && may_have_multi_halo_exchanges) {
            // Rewrite to a single Pad.
            HloInstruction* new_pad =
                computation->AddInstruction(HloInstruction::CreatePad(
                    hlo->shape(), operand->mutable_operand(0),
                    operand->mutable_operand(1), *merged_padding));
            new_pad->set_metadata(operand->metadata());
            new_pad->set_sharding(hlo->sharding());
            TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_pad));
            TF_RETURN_IF_ERROR(
                computation->RemoveInstructionAndUnusedOperands(hlo));
          }
        }
      }
      if (hlo->opcode() == HloOpcode::kConcatenate) {
        const int64_t dim = hlo->concatenate_dimension();
        if (hlo->sharding().tile_assignment().dim(dim) == 1) {
          continue;
        }
        if (hlo->operand_count() == 2) {
          // Find a pattern of "rotate right on one dimension":
          // concat(slice(input), slice(input)).
          HloInstruction* lhs = skip_copy_operands(hlo->mutable_operand(0));
          HloInstruction* rhs = skip_copy_operands(hlo->mutable_operand(1));
          if (lhs == nullptr || rhs == nullptr) {
            continue;
          }
          const int64_t amount = FindRotateRightPattern(hlo, lhs, rhs);
          if (amount < 0) {
            continue;
          }
          HloInstruction* to_rotate = lhs->mutable_operand(0);
          HloInstruction* rotate = computation->AddInstruction(
              CreateCustomCallSPMDInternal_RotateRight(to_rotate, dim, amount));
          rotate->set_metadata(hlo->metadata());
          rotate->set_sharding(hlo->sharding());
          TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(rotate));
          TF_RETURN_IF_ERROR(
              computation->RemoveInstructionAndUnusedOperands(hlo));
        } else if (hlo->operand_count() == 3) {
          // Find the pattern for "pad with wrap": concat(slice(x), x, slice(x))
          // All involved values with same sharding.
          HloInstruction* lhs = skip_copy_operands(hlo->mutable_operand(0));
          HloInstruction* mid = skip_copy_operands(hlo->mutable_operand(1),
                                                   /*check_single_use=*/false);
          HloInstruction* rhs = skip_copy_operands(hlo->mutable_operand(2));
          std::optional<PadWithWrapPattern> pad_pattern =
              FindPadWithWrapPattern(hlo, lhs, mid, rhs);
          if (!pad_pattern) {
            continue;
          }

          // Since the concat requires that the size of all operands along the
          // non-concat dimension is the same, it implies that the lhs/rhs slice
          // is slicing along the concat dims.

          // Step 1: Pad the mid operand to the final size. The low padding is
          // the size of the lhs shape, and high padding is size of rhs shape.
          PaddingConfig padding_config =
              MakeNoPaddingConfig(hlo->shape().rank());
          auto* padding_config_dim = padding_config.mutable_dimensions(dim);
          const int64_t low_pad = lhs->shape().dimensions(dim);
          const int64_t high_pad = rhs->shape().dimensions(dim);
          padding_config_dim->set_edge_padding_low(low_pad);
          padding_config_dim->set_edge_padding_high(high_pad);
          HloInstruction* zero =
              computation->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::Zero(hlo->shape().element_type())));
          zero->set_sharding(HloSharding::Replicate());
          HloInstruction* pad =
              computation->AddInstruction(HloInstruction::CreatePad(
                  hlo->shape(), mid, zero, padding_config));
          pad->set_metadata(hlo->metadata());
          pad->set_sharding(hlo->sharding());

          // Step 2: rotate the padded value so that the lhs slice aligns to the
          // low of the padded size.
          //  padded_operand = low_pad | mid | high_pad.
          //  slice_start in padded_operand = lhs->slice_start + low_pad.
          //  Rotate left by (lhs->slice_start + low_pad)
          //  i.e., rotate right = padded_size - (lhs_slice_start + low_pad).
          const int64_t padded_size = hlo->shape().dimensions(dim);
          const int rotate_lhs_amount =
              padded_size - (pad_pattern->lhs_slice_start + low_pad);
          HloInstruction* rotate_lhs = computation->AddInstruction(
              CreateCustomCallSPMDInternal_RotateRight(pad, dim,
                                                       rotate_lhs_amount));
          rotate_lhs->set_metadata(hlo->metadata());
          rotate_lhs->set_sharding(hlo->sharding());

          auto apply_modifiers =
              [&](HloInstruction* inst,
                  const std::vector<const HloInstruction*>& modifiers) {
                // Apply the modifiers in the reverse order.
                for (auto it = modifiers.crbegin(), end = modifiers.crend();
                     it != end; ++it) {
                  const HloInstruction* modifier = *it;
                  // New shape has same element type as the modifier, but dims
                  // as inst.
                  Shape new_shape = ShapeUtil::ChangeElementType(
                      inst->shape(), modifier->shape().element_type());
                  inst = computation->AddInstruction(
                      modifier->CloneWithNewOperands(new_shape, {inst}));
                }
                return inst;
              };
          rotate_lhs = apply_modifiers(rotate_lhs, pad_pattern->lhs_modifiers);

          // Step 3: rotate the padded value so that the rhs slice aligns to
          // high of the padded size.
          //  padded_operand = low_pad | mid | high_pad.
          //  slice_start in padded_operand = rhs->slice_start + low_pad.
          //  slice_end in padded_operand = rhs->slice_start + low_pad +
          //  high_pad; Rotate right by padded_size - (rhs->slice_start +
          //  low_pad + high_pad)
          const int64_t rotate_rhs_amount =
              padded_size - (pad_pattern->rhs_slice_start + low_pad + high_pad);
          HloInstruction* rotate_rhs = computation->AddInstruction(
              CreateCustomCallSPMDInternal_RotateRight(pad, dim,
                                                       rotate_rhs_amount));
          rotate_rhs->set_metadata(hlo->metadata());
          rotate_rhs->set_sharding(hlo->sharding());
          rotate_rhs = apply_modifiers(rotate_rhs, pad_pattern->rhs_modifiers);

          // Now merge the 3 results using appropriate selects.
          const Shape iota_shape =
              ShapeUtil::ChangeElementType(hlo->shape(), U32);
          HloInstruction* iota = computation->AddInstruction(
              HloInstruction::CreateIota(iota_shape, dim));
          iota->set_metadata(hlo->metadata());
          iota->set_sharding(hlo->sharding());

          struct SelectSpec {
            int64_t limit;
            HloInstruction* hlo;
            Comparison::Direction cmp;
          };
          const std::array<SelectSpec, 2> selects = {
              {// All elements < low_pad come from rotate_lhs.
               {low_pad, rotate_lhs, Comparison::Direction::kLt},
               // All elements >= padded_size - high_pad come from rotate_rhs
               {padded_size - high_pad, rotate_rhs,
                Comparison::Direction::kGe}}};

          Shape pred_shape = ShapeUtil::ChangeElementType(hlo->shape(), PRED);

          HloInstruction* merged = pad;
          for (const SelectSpec& select_spec : selects) {
            HloInstruction* limit =
                computation->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<uint32_t>(select_spec.limit)));
            limit->set_sharding(HloSharding::Replicate());
            HloInstruction* limit_bcast = computation->AddInstruction(
                HloInstruction::CreateBroadcast(iota_shape, limit, {}));
            limit_bcast->set_metadata(hlo->metadata());
            limit_bcast->set_sharding(hlo->sharding());
            HloInstruction* compare =
                computation->AddInstruction(HloInstruction::CreateCompare(
                    pred_shape, iota, limit_bcast, select_spec.cmp));
            compare->set_metadata(hlo->metadata());
            compare->set_sharding(hlo->sharding());
            merged = computation->AddInstruction(HloInstruction::CreateTernary(
                hlo->shape(), HloOpcode::kSelect, compare, select_spec.hlo,
                merged));
            merged->set_metadata(hlo->metadata());
            merged->set_sharding(hlo->sharding());
          }

          TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(merged));
          TF_RETURN_IF_ERROR(
              computation->RemoveInstructionAndUnusedOperands(hlo));
        }
      }
    }
  }
  return OkStatus();
}

}  // namespace spmd
}  // namespace xla
