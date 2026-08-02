/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/async_collective_custom_call_rewriter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla {

namespace {

std::string GetConfigString(HloInstruction* instr) {
  if (instr->has_frontend_attributes()) {
    const auto& map = instr->frontend_attributes().map();
    auto it = map.find("async_collective_config");
    if (it != map.end()) {
      return it->second;
    }
  }
  return instr->raw_backend_config_string();
}

bool IsCollectiveCustomCall(HloInstruction* instr, absl::string_view suffix) {
  if (instr->opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  absl::string_view target = instr->custom_call_target();
  if (!absl::EndsWith(target, suffix)) {
    return false;
  }

  absl::string_view base = target;
  base.remove_suffix(suffix.size());

  return base == "all-gather" || base == "all-reduce" ||
         base == "reduce-scatter" || base == "all-to-all" ||
         base == "collective-permute";
}

absl::Status CleanupAndPropagate(HloComputation* computation,
                                 HloInstruction* start_call,
                                 HloInstruction* done_call,
                                 HloInstruction* async_start,
                                 HloInstruction* async_done,
                                 HloInstruction* final_result) {
  async_start->set_metadata(start_call->metadata());
  async_done->set_metadata(done_call->metadata());
  if (final_result != async_done) {
    final_result->set_metadata(done_call->metadata());
  }

  for (HloInstruction* pred : start_call->control_predecessors()) {
    RETURN_IF_ERROR(pred->AddControlDependencyTo(async_start));
  }
  for (HloInstruction* succ : done_call->control_successors()) {
    RETURN_IF_ERROR(final_result->AddControlDependencyTo(succ));
  }

  RETURN_IF_ERROR(done_call->ReplaceAllUsesWith(final_result));
  RETURN_IF_ERROR(computation->RemoveInstruction(done_call));
  RETURN_IF_ERROR(computation->RemoveInstruction(start_call));
  return absl::OkStatus();
}

absl::StatusOr<bool> ProcessAllGather(HloComputation* computation,
                                      HloInstruction* start_call,
                                      HloInstruction* done_call,
                                      bool use_legacy_collectives) {
  std::string config_str = GetConfigString(start_call);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  TF_RET_CHECK(config.all_gather_dimension.has_value());
  int64_t all_gather_dim = *config.all_gather_dimension;

  Shape shape = done_call->shape();

  auto device_list =
      std::make_shared<CollectiveDeviceList>(config.replica_groups);

  HloInstruction* async_start = nullptr;
  HloInstruction* async_done = nullptr;

  if (use_legacy_collectives) {
    std::vector<const Shape*> operand_shapes;
    operand_shapes.reserve(start_call->operand_count());
    for (const HloInstruction* op : start_call->operands()) {
      operand_shapes.push_back(&op->shape());
    }
    Shape start_shape = ShapeUtil::MakeTupleShape(
        {start_call->operand_count() > 1
             ? ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes)
             : *operand_shapes[0],
         shape});
    async_start =
        computation->AddInstruction(HloInstruction::CreateAllGatherStart(
            start_shape, start_call->operands(), all_gather_dim, device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids));
    async_done = computation->AddInstruction(HloInstruction::CreateUnary(
        shape, HloOpcode::kAllGatherDone, async_start));
  } else {
    std::unique_ptr<HloInstruction> sync_all_gather =
        HloInstruction::CreateAllGather(
            shape, start_call->operands(), all_gather_dim, device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids);

    ASSIGN_OR_RETURN(async_done,
                     computation->CreateAsyncInstructions(
                         sync_all_gather.get(), /*context_shapes=*/{},
                         computation->execution_thread(), /*replace=*/false));

    async_start = async_done->mutable_operand(0);
  }

  RETURN_IF_ERROR(CleanupAndPropagate(computation, start_call, done_call,
                                      async_start, async_done, async_done));

  return true;
}

absl::StatusOr<bool> ProcessAllReduce(HloComputation* computation,
                                      HloInstruction* start_call,
                                      HloInstruction* done_call,
                                      bool use_legacy_collectives) {
  std::string config_str = GetConfigString(start_call);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  TF_RET_CHECK(start_call->called_computations().size() == 1)
      << "Expected 1 called computation for AllReduce, got "
      << start_call->called_computations().size();
  HloComputation* reduce_computation = start_call->called_computations()[0];

  Shape shape = done_call->shape();
  auto device_list =
      std::make_shared<CollectiveDeviceList>(config.replica_groups);

  HloInstruction* async_start = nullptr;
  HloInstruction* async_done = nullptr;

  if (use_legacy_collectives) {
    async_start =
        computation->AddInstruction(HloInstruction::CreateAllReduceStart(
            shape, start_call->operands(), reduce_computation, device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids));
    async_done = computation->AddInstruction(HloInstruction::CreateUnary(
        shape, HloOpcode::kAllReduceDone, async_start));
  } else {
    std::unique_ptr<HloInstruction> sync_all_reduce =
        HloInstruction::CreateAllReduce(
            shape, start_call->operands(), reduce_computation, device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids);

    ASSIGN_OR_RETURN(async_done,
                     computation->CreateAsyncInstructions(
                         sync_all_reduce.get(), /*context_shapes=*/{},
                         computation->execution_thread(), /*replace=*/false));

    async_start = async_done->mutable_operand(0);
  }

  RETURN_IF_ERROR(CleanupAndPropagate(computation, start_call, done_call,
                                      async_start, async_done, async_done));

  return true;
}

absl::StatusOr<bool> ProcessReduceScatter(HloComputation* computation,
                                          HloInstruction* start_call,
                                          HloInstruction* done_call) {
  std::string config_str = GetConfigString(start_call);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  TF_RET_CHECK(config.scatter_dimension.has_value());
  int64_t scatter_dimension = *config.scatter_dimension;
  TF_RET_CHECK(config.tiled.has_value());
  bool tiled = *config.tiled;

  TF_RET_CHECK(start_call->called_computations().size() == 1)
      << "Expected 1 called computation for ReduceScatter, got "
      << start_call->called_computations().size();
  HloComputation* reduce_computation = start_call->called_computations()[0];

  Shape rs_shape = done_call->shape();
  int64_t axis_size = config.replica_groups.empty()
                          ? 1
                          : config.replica_groups[0].replica_ids_size();
  TF_RET_CHECK(axis_size > 0)
      << "ReduceScatter axis size must be positive, got " << axis_size;

  Shape input_shape = start_call->operand(0)->shape();
  if (!tiled) {
    TF_RET_CHECK(input_shape.dimensions(scatter_dimension) % axis_size == 0)
        << "ReduceScatter input shape dimension " << scatter_dimension << " ("
        << input_shape.dimensions(scatter_dimension)
        << ") must be divisible by axis size " << axis_size;
    std::vector<int64_t> rs_dims(input_shape.dimensions().begin(),
                                 input_shape.dimensions().end());
    rs_dims[scatter_dimension] /= axis_size;
    rs_shape = ShapeUtil::MakeShape(input_shape.element_type(), rs_dims);
  }

  auto device_list =
      std::make_shared<CollectiveDeviceList>(config.replica_groups);

  std::unique_ptr<HloInstruction> sync_reduce_scatter =
      HloInstruction::CreateReduceScatter(
          rs_shape, start_call->operands(), reduce_computation, device_list,
          /*constrain_layout=*/false, config.channel_id,
          config.use_global_device_ids, scatter_dimension);

  ASSIGN_OR_RETURN(HloInstruction * async_done,
                   computation->CreateAsyncInstructions(
                       sync_reduce_scatter.get(), /*context_shapes=*/{},
                       computation->execution_thread(), /*replace=*/false));

  HloInstruction* async_start = async_done->mutable_operand(0);

  HloInstruction* final_result = async_done;
  if (!tiled) {
    final_result = computation->AddInstruction(
        HloInstruction::CreateReshape(done_call->shape(), async_done));
  }

  RETURN_IF_ERROR(CleanupAndPropagate(computation, start_call, done_call,
                                      async_start, async_done, final_result));

  return true;
}

absl::StatusOr<bool> ProcessAllToAll(HloComputation* computation,
                                     HloInstruction* start_call,
                                     HloInstruction* done_call) {
  std::string config_str = GetConfigString(start_call);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  TF_RET_CHECK(config.split_dimension.has_value());
  int64_t split_dimension = *config.split_dimension;
  TF_RET_CHECK(config.concat_dimension.has_value());
  int64_t concat_dimension = *config.concat_dimension;
  TF_RET_CHECK(config.split_count.has_value());
  int64_t split_count = *config.split_count;
  TF_RET_CHECK(split_count > 0)
      << "AllToAll split count must be positive, got " << split_count;

  Shape input_shape = start_call->operand(0)->shape();
  TF_RET_CHECK(!input_shape.is_dynamic())
      << "Dynamic shape AllToAll decomposition not supported yet";

  auto device_list =
      std::make_shared<CollectiveDeviceList>(config.replica_groups);
  HloInstruction* final_result = nullptr;
  HloInstruction* async_done = nullptr;

  if (split_dimension == concat_dimension) {
    Shape shape = done_call->shape();
    std::unique_ptr<HloInstruction> sync_all_to_all =
        HloInstruction::CreateAllToAll(
            shape, start_call->operands(), device_list,
            /*constrain_layout=*/false, config.channel_id, split_dimension);

    ASSIGN_OR_RETURN(async_done,
                     computation->CreateAsyncInstructions(
                         sync_all_to_all.get(), /*context_shapes=*/{},
                         computation->execution_thread(), /*replace=*/false));
    final_result = async_done;
  } else {
    std::unique_ptr<HloInstruction> sync_all_to_all =
        HloInstruction::CreateAllToAll(
            input_shape, start_call->operands(), device_list,
            /*constrain_layout=*/false, config.channel_id, split_dimension);

    ASSIGN_OR_RETURN(async_done,
                     computation->CreateAsyncInstructions(
                         sync_all_to_all.get(), /*context_shapes=*/{},
                         computation->execution_thread(), /*replace=*/false));

    std::vector<int64_t> reshape_sizes;
    for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
      if (i != split_dimension) {
        reshape_sizes.push_back(input_shape.dimensions(i));
      } else {
        TF_RET_CHECK(input_shape.dimensions(i) % split_count == 0)
            << "AllToAll input dimension " << i << " ("
            << input_shape.dimensions(i)
            << ") must be divisible by split count " << split_count;
        reshape_sizes.push_back(split_count);
        reshape_sizes.push_back(input_shape.dimensions(i) / split_count);
      }
    }
    Shape reshape_shape =
        ShapeUtil::MakeShape(input_shape.element_type(), reshape_sizes);
    HloInstruction* reshape1 = computation->AddInstruction(
        HloInstruction::CreateReshape(reshape_shape, async_done));

    std::vector<int64_t> permutation;
    const auto rank = input_shape.dimensions().size();
    permutation.reserve(rank + 1);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t dim_after_reshape = i >= split_dimension ? i + 1 : i;
      if (i == concat_dimension) {
        permutation.push_back(split_dimension);
      }
      permutation.push_back(dim_after_reshape);
    }

    std::vector<int64_t> transpose_sizes;
    for (int64_t axis : permutation) {
      transpose_sizes.push_back(reshape_sizes[axis]);
    }
    Shape transpose_shape =
        ShapeUtil::MakeShape(input_shape.element_type(), transpose_sizes);
    HloInstruction* transpose =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            transpose_shape, reshape1, permutation));

    final_result = computation->AddInstruction(
        HloInstruction::CreateReshape(done_call->shape(), transpose));
  }

  HloInstruction* async_start = async_done->mutable_operand(0);

  RETURN_IF_ERROR(CleanupAndPropagate(computation, start_call, done_call,
                                      async_start, async_done, final_result));

  return true;
}

absl::StatusOr<bool> ProcessCollectivePermute(HloComputation* computation,
                                              HloInstruction* start_call,
                                              HloInstruction* done_call,
                                              bool use_legacy_collectives) {
  std::string config_str = GetConfigString(start_call);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  HloInstruction* async_start = nullptr;
  HloInstruction* async_done = nullptr;

  if (use_legacy_collectives) {
    std::vector<const Shape*> operand_shapes;
    for (HloInstruction* operand : start_call->operands()) {
      operand_shapes.push_back(&operand->shape());
    }

    ASSIGN_OR_RETURN(
        Shape start_shape,
        ShapeInference::InferCollectivePermuteStartShape(
            operand_shapes, /*context_shapes=*/{}, /*inplace=*/false));

    async_start = computation->AddInstruction(
        HloInstruction::CreateCollectivePermuteStart(
            start_shape, start_call->operands(), config.permutation,
            config.channel_id));

    async_done = computation->AddInstruction(HloInstruction::CreateUnary(
        done_call->shape(), HloOpcode::kCollectivePermuteDone, async_start));
  } else {
    std::unique_ptr<HloInstruction> sync_collective_permute =
        HloInstruction::CreateCollectivePermute(
            done_call->shape(), start_call->operands(), config.permutation,
            config.channel_id);

    ASSIGN_OR_RETURN(async_done,
                     computation->CreateAsyncInstructions(
                         sync_collective_permute.get(), /*context_shapes=*/{},
                         computation->execution_thread(), /*replace=*/false));

    async_start = async_done->mutable_operand(0);
  }

  RETURN_IF_ERROR(CleanupAndPropagate(computation, start_call, done_call,
                                      async_start, async_done, async_done));

  return true;
}

absl::StatusOr<bool> ProcessPair(HloComputation* computation,
                                 HloInstruction* start_call,
                                 HloInstruction* done_call,
                                 bool use_legacy_collectives) {
  absl::string_view target = start_call->custom_call_target();

  if (absl::StartsWith(target, "all-gather")) {
    return ProcessAllGather(computation, start_call, done_call,
                            use_legacy_collectives);
  }
  if (absl::StartsWith(target, "all-reduce")) {
    return ProcessAllReduce(computation, start_call, done_call,
                            use_legacy_collectives);
  }
  if (absl::StartsWith(target, "reduce-scatter")) {
    return ProcessReduceScatter(computation, start_call, done_call);
  }
  if (absl::StartsWith(target, "all-to-all")) {
    return ProcessAllToAll(computation, start_call, done_call);
  }
  if (absl::StartsWith(target, "collective-permute")) {
    return ProcessCollectivePermute(computation, start_call, done_call,
                                    use_legacy_collectives);
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> AsyncCollectiveCustomCallRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> computations(module->computations().begin(),
                                            module->computations().end());
  for (HloComputation* computation : computations) {
    // First pass to find all pairs to avoid modifying while iterating.
    std::vector<std::pair<HloInstruction*, HloInstruction*>> pairs_to_rewrite;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (IsCollectiveCustomCall(instr, "-done")) {
        HloInstruction* done_call = instr;
        if (done_call->operand_count() == 0) {
          continue;
        }
        HloInstruction* start_call = done_call->mutable_operand(0);
        if (IsCollectiveCustomCall(start_call, "-start")) {
          pairs_to_rewrite.push_back({start_call, done_call});
        }
      }
    }

    // Second pass to process the collected pairs.
    for (const auto& pair : pairs_to_rewrite) {
      ASSIGN_OR_RETURN(bool pair_changed,
                       ProcessPair(computation, pair.first, pair.second,
                                   use_legacy_collectives_));
      if (pair_changed) {
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
