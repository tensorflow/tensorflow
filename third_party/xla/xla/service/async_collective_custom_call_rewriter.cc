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
#include "xla/frontend_attributes.h"
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

bool IsCollectiveStart(HloInstruction* instr) {
  if (IsCollectiveCustomCall(instr, "-start")) {
    return true;
  }
  return instr->opcode() == HloOpcode::kAsyncStart ||
         instr->opcode() == HloOpcode::kAllGatherStart ||
         instr->opcode() == HloOpcode::kAllReduceStart ||
         instr->opcode() == HloOpcode::kCollectivePermuteStart;
}

struct TraceTarget {
  HloInstruction* inst;
  int64_t index;
};

bool IsTrivialFormattingOp(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kCustomCall) {
    absl::string_view target = instr->custom_call_target();
    return target == "xla.sdy.GlobalToLocalShape" ||
           target == "xla.sdy.LocalToGlobalShape" || target == "Sharding" ||
           target == "xla.sdy.FuncResultSharding";
  }
  return instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape ||
         instr->opcode() == HloOpcode::kTranspose ||
         instr->opcode() == HloOpcode::kSlice;
}

struct ParameterIndex {
  HloInstruction* parameter;
  int64_t index;
};

std::optional<ParameterIndex> ResolveParameterIndex(HloInstruction* instr) {
  int64_t index = -1;
  HloInstruction* curr = instr;
  while (true) {
    if (curr->opcode() == HloOpcode::kParameter) {
      return ParameterIndex{curr, index};
    } else if (curr->opcode() == HloOpcode::kGetTupleElement) {
      index = curr->tuple_index();
      curr = curr->mutable_operand(0);
    } else if (curr->opcode() == HloOpcode::kTuple) {
      if (index == -1) return std::nullopt;
      curr = curr->mutable_operand(index);
      index = -1;
    } else if (curr->opcode() == HloOpcode::kOptimizationBarrier ||
               curr->opcode() == HloOpcode::kCopy) {
      curr = curr->mutable_operand(0);
    } else if (IsTrivialFormattingOp(curr)) {
      curr = curr->mutable_operand(0);
    } else {
      return std::nullopt;
    }
  }
}

std::vector<HloInstruction*> FindPathBetween(HloInstruction* start,
                                             HloInstruction* end) {
  std::vector<HloInstruction*> path;
  HloInstruction* current = end;
  int64_t index = -1;
  while (current != start) {
    path.push_back(current);
    if (current->opcode() == HloOpcode::kTuple) {
      if (index == -1) return {};
      current = current->mutable_operand(index);
      index = -1;
    } else if (current->opcode() == HloOpcode::kGetTupleElement) {
      index = current->tuple_index();
      current = current->mutable_operand(0);
    } else if (current->opcode() == HloOpcode::kOptimizationBarrier ||
               current->opcode() == HloOpcode::kCopy) {
      current = current->mutable_operand(0);
    } else if (IsTrivialFormattingOp(current)) {
      current = current->mutable_operand(0);
    } else {
      LOG(INFO) << "FindPathBetween: unsupported opcode "
                << HloOpcodeString(current->opcode())
                << " for instruction: " << current->ToString();
      return {};
    }
  }
  std::reverse(path.begin(), path.end());
  return path;
}

HloInstruction* TraceForwardToDone(HloInstruction* inst) {
  if (inst == nullptr) return nullptr;
  if (IsCollectiveCustomCall(inst, "-done")) {
    return inst;
  }
  if (inst->opcode() == HloOpcode::kGetTupleElement ||
      inst->opcode() == HloOpcode::kCopy ||
      inst->opcode() == HloOpcode::kOptimizationBarrier) {
    for (HloInstruction* user : inst->users()) {
      HloInstruction* done = TraceForwardToDone(user);
      if (done != nullptr) return done;
    }
  }
  return nullptr;
}

std::vector<HloInstruction*> FindPathFromStartToDone(HloInstruction* start,
                                                     HloInstruction* done) {
  return FindPathBetween(start, done->mutable_operand(0));
}

absl::StatusOr<HloInstruction*> PropagateShape(
    HloInstruction* old_start, HloInstruction* new_start,
    const std::vector<HloInstruction*>& path) {
  HloInstruction* last_inst = new_start;
  HloInstruction* prev_in_path = old_start;

  for (size_t i = 0; i < path.size(); ++i) {
    HloInstruction* inst = path[i];
    int64_t idx = -1;
    if (inst->opcode() == HloOpcode::kTuple) {
      idx = inst->operand_index(prev_in_path);
      if (idx == -1) {
        return absl::InternalError("Operand not found in tuple");
      }
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      idx = 0;
    } else if (inst->opcode() == HloOpcode::kOptimizationBarrier ||
               inst->opcode() == HloOpcode::kCopy) {
      idx = 0;
    } else if (IsTrivialFormattingOp(inst)) {
      idx = 0;
    } else {
      return absl::InternalError("Unsupported instruction in path");
    }

    Shape old_shape = inst->shape();
    RETURN_IF_ERROR(inst->ReplaceOperandWithDifferentShape(idx, last_inst));

    Shape new_shape = inst->shape();
    if (inst->opcode() == HloOpcode::kTuple) {
      *new_shape.mutable_tuple_shapes(idx) = last_inst->shape();
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      new_shape =
          ShapeUtil::GetSubshape(last_inst->shape(), {inst->tuple_index()});
    } else if (inst->opcode() == HloOpcode::kOptimizationBarrier ||
               inst->opcode() == HloOpcode::kCopy) {
      new_shape = last_inst->shape();
    } else if (IsTrivialFormattingOp(inst)) {
      return absl::InternalError(absl::StrCat(
          "Formatting op in propagation path: ", inst->ToString()));
    }
    *inst->mutable_shape() = new_shape;
    LOG(INFO) << "  PropagateShape updated " << inst->name() << " from "
              << old_shape.ToString() << " to " << new_shape.ToString();

    prev_in_path = inst;
    last_inst = inst;
  }

  return last_inst;
}

absl::Status PropagateShapeHelper(HloInstruction* start, HloInstruction* done,
                                  const std::vector<HloInstruction*>& path,
                                  HloInstruction* async_start,
                                  HloInstruction* async_done) {
  ASSIGN_OR_RETURN(HloInstruction * last_inst,
                   PropagateShape(start, async_start, path));
  HloInstruction* operand = last_inst;
  while (operand->opcode() == HloOpcode::kCopy ||
         operand->opcode() == HloOpcode::kOptimizationBarrier ||
         IsTrivialFormattingOp(operand)) {
    operand = operand->mutable_operand(0);
  }
  RETURN_IF_ERROR(async_done->ReplaceOperandWithDifferentShape(0, operand));
  return absl::OkStatus();
}

bool PathRequiresPropagation(const std::vector<HloInstruction*>& path) {
  for (HloInstruction* inst : path) {
    if (inst->opcode() == HloOpcode::kTuple ||
        inst->opcode() == HloOpcode::kGetTupleElement) {
      return true;
    }
    if (inst->opcode() == HloOpcode::kOptimizationBarrier &&
        inst->shape().IsTuple()) {
      return true;
    }
  }
  return false;
}

void FindMatchingStartsHelper(TraceTarget target,
                              absl::flat_hash_set<HloInstruction*>& visited,
                              std::vector<HloInstruction*>& starts) {
  HloInstruction* inst = target.inst;
  int64_t index = target.index;

  if (inst == nullptr) return;
  if (!visited.insert(inst).second) return;

  if (index == -1) {
    if (IsCollectiveStart(inst)) {
      starts.push_back(inst);
      return;
    }
    if (IsTrivialFormattingOp(inst)) {
      FindMatchingStartsHelper({inst->mutable_operand(0), -1}, visited, starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kOptimizationBarrier ||
        inst->opcode() == HloOpcode::kCopy) {
      if (inst->operand_count() == 1) {
        FindMatchingStartsHelper({inst->mutable_operand(0), -1}, visited,
                                 starts);
      }
      return;
    }
    if (inst->opcode() == HloOpcode::kGetTupleElement) {
      FindMatchingStartsHelper({inst->mutable_operand(0), inst->tuple_index()},
                               visited, starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kCall) {
      HloComputation* called = inst->to_apply();
      FindMatchingStartsHelper({called->root_instruction(), -1}, visited,
                               starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kParameter) {
      HloInstruction* param = inst;
      HloComputation* comp = param->parent();
      for (HloInstruction* caller : comp->caller_instructions()) {
        if (caller->opcode() == HloOpcode::kCall) {
          FindMatchingStartsHelper(
              {caller->mutable_operand(param->parameter_number()), -1}, visited,
              starts);
        }
      }
      return;
    }
  } else {
    if (inst->opcode() == HloOpcode::kTuple) {
      FindMatchingStartsHelper({inst->mutable_operand(index), -1}, visited,
                               starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kOptimizationBarrier ||
        inst->opcode() == HloOpcode::kCopy) {
      FindMatchingStartsHelper({inst->mutable_operand(0), index}, visited,
                               starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kCall) {
      HloComputation* called = inst->to_apply();
      FindMatchingStartsHelper({called->root_instruction(), index}, visited,
                               starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kWhile) {
      HloInstruction* while_op = inst;
      HloInstruction* init = while_op->mutable_operand(0);
      FindMatchingStartsHelper({init, index}, visited, starts);

      HloComputation* body = while_op->while_body();
      HloInstruction* body_root = body->root_instruction();
      FindMatchingStartsHelper({body_root, index}, visited, starts);
      return;
    }
    if (inst->opcode() == HloOpcode::kParameter) {
      HloInstruction* param = inst;
      HloComputation* comp = param->parent();
      for (HloInstruction* caller : comp->caller_instructions()) {
        if (caller->opcode() == HloOpcode::kCall) {
          FindMatchingStartsHelper(
              {caller->mutable_operand(param->parameter_number()), index},
              visited, starts);
        } else if (caller->opcode() == HloOpcode::kWhile) {
          HloInstruction* while_op = caller;
          HloInstruction* init = while_op->mutable_operand(0);
          FindMatchingStartsHelper({init, index}, visited, starts);

          HloInstruction* body_root = comp->root_instruction();
          FindMatchingStartsHelper({body_root, index}, visited, starts);
        }
      }
      return;
    }
  }
}

std::vector<HloInstruction*> FindMatchingStarts(HloInstruction* done_call) {
  std::vector<HloInstruction*> starts;
  absl::flat_hash_set<HloInstruction*> visited;
  FindMatchingStartsHelper({done_call->mutable_operand(0), -1}, visited,
                           starts);
  return starts;
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

absl::StatusOr<bool> ProcessAllGatherSync(HloInstruction* start_call,
                                          HloInstruction* done_call) {
  LOG(INFO) << "Falling back to sync AllGather for sibling pair: "
            << start_call->name() << " and " << done_call->name();
  HloComputation* comp_start = start_call->parent();
  HloComputation* comp_done = done_call->parent();

  std::string config_str = GetConfigString(start_call);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  TF_RET_CHECK(config.all_gather_dimension.has_value());
  int64_t all_gather_dim = *config.all_gather_dimension;
  auto device_list =
      std::make_shared<CollectiveDeviceList>(config.replica_groups);

  HloInstruction* sync_ag =
      comp_start->AddInstruction(HloInstruction::CreateAllGather(
          start_call->shape(), start_call->operands(), all_gather_dim,
          device_list, /*constrain_layout=*/false, config.channel_id,
          config.use_global_device_ids));

  sync_ag->set_metadata(start_call->metadata());

  RETURN_IF_ERROR(start_call->ReplaceAllUsesWith(sync_ag));
  RETURN_IF_ERROR(comp_start->RemoveInstruction(start_call));

  RETURN_IF_ERROR(done_call->ReplaceAllUsesWith(done_call->mutable_operand(0)));
  RETURN_IF_ERROR(comp_done->RemoveInstruction(done_call));

  return true;
}

absl::StatusOr<bool> ProcessAllGather(HloComputation* computation,
                                      HloInstruction* start_call,
                                      HloInstruction* done_call,
                                      bool use_legacy_collectives) {
  if (start_call->parent() != done_call->parent()) {
    return ProcessAllGatherSync(start_call, done_call);
  }
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

  std::vector<HloInstruction*> path =
      FindPathFromStartToDone(start_call, done_call);
  if (!path.empty()) {
    RETURN_IF_ERROR(PropagateShapeHelper(start_call, done_call, path,
                                         async_start, async_done));
    async_start->set_metadata(start_call->metadata());
    async_done->set_metadata(done_call->metadata());
    for (HloInstruction* pred : start_call->control_predecessors()) {
      RETURN_IF_ERROR(pred->AddControlDependencyTo(async_start));
    }
    for (HloInstruction* succ : done_call->control_successors()) {
      RETURN_IF_ERROR(async_done->AddControlDependencyTo(succ));
    }
    RETURN_IF_ERROR(done_call->ReplaceAllUsesWith(async_done));
    RETURN_IF_ERROR(computation->RemoveInstruction(done_call));
    RETURN_IF_ERROR(computation->RemoveInstruction(start_call));
  } else {
    RETURN_IF_ERROR(CleanupAndPropagate(computation, start_call, done_call,
                                        async_start, async_done, async_done));
  }

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

absl::StatusOr<HloInstruction*> SinkFormattingOps(HloInstruction* done_call) {
  HloInstruction* current = done_call->mutable_operand(0);
  HloComputation* comp = done_call->parent();

  while (IsTrivialFormattingOp(current) ||
         current->opcode() == HloOpcode::kOptimizationBarrier) {
    if (current->opcode() == HloOpcode::kOptimizationBarrier &&
        current->operand_count() != 1) {
      break;
    }

    HloInstruction* format_op = current;
    HloInstruction* format_input = format_op->mutable_operand(0);

    Shape done_new_shape = format_input->shape();
    std::vector<HloInstruction*> new_operands = {format_input};
    for (int i = 1; i < done_call->operand_count(); ++i) {
      new_operands.push_back(done_call->mutable_operand(i));
    }
    HloInstruction* done_new = comp->AddInstruction(
        done_call->CloneWithNewOperands(done_new_shape, new_operands));

    HloInstruction* format_new = comp->AddInstruction(
        format_op->CloneWithNewOperands(done_call->shape(), {done_new}));

    RETURN_IF_ERROR(done_call->ReplaceAllUsesWith(format_new));
    RETURN_IF_ERROR(comp->RemoveInstruction(done_call));
    if (format_op->IsDead()) {
      RETURN_IF_ERROR(comp->RemoveInstruction(format_op));
    }

    done_call = done_new;
    current = format_input;
  }
  return done_call;
}

absl::Status PropagateLoopCarryType(HloInstruction* while_op, int64_t carry_idx,
                                    HloInstruction* start_outside,
                                    HloInstruction* async_start_outside,
                                    HloInstruction* async_start_inside,
                                    HloInstruction* body_root_operand) {
  HloInstruction* init = while_op->mutable_operand(0);
  TF_RET_CHECK(init->opcode() == HloOpcode::kTuple);
  LOG(INFO) << "Jetski: PropagateLoopCarryType entered. while_op: "
            << while_op->name() << ", carry_idx: " << carry_idx;
  LOG(INFO) << "Jetski: while_op shape before: "
            << while_op->shape().ToString();
  LOG(INFO) << "Jetski: init shape before: " << init->shape().ToString();

  while_op->set_frontend_attribute(kXlaPreserveTupleIndices, "true");

  HloInstruction* init_operand = init->mutable_operand(carry_idx);
  std::vector<HloInstruction*> path_outside =
      FindPathBetween(start_outside, init_operand);
  HloInstruction* propagated_start_outside = async_start_outside;
  if (!path_outside.empty()) {
    ASSIGN_OR_RETURN(
        propagated_start_outside,
        PropagateShape(start_outside, async_start_outside, path_outside));
  }

  // Update init
  RETURN_IF_ERROR(init->ReplaceOperandWithDifferentShape(
      carry_idx, propagated_start_outside));
  *init->mutable_shape()->mutable_tuple_shapes(carry_idx) =
      propagated_start_outside->shape();

  HloComputation* body = while_op->while_body();
  HloInstruction* body_root = body->root_instruction();
  TF_RET_CHECK(body_root->opcode() == HloOpcode::kTuple);

  // Update body root
  RETURN_IF_ERROR(body_root->ReplaceOperandWithDifferentShape(
      carry_idx, body_root_operand));
  *body_root->mutable_shape()->mutable_tuple_shapes(carry_idx) =
      body_root_operand->shape();

  // Update while shape
  const Shape& new_shape = propagated_start_outside->shape();
  *while_op->mutable_shape()->mutable_tuple_shapes(carry_idx) = new_shape;

  // Update GTEs outside that use this while_op and index
  for (HloInstruction* user : while_op->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == carry_idx) {
      *user->mutable_shape() = new_shape;
    }
  }

  // Update body param
  HloInstruction* body_param = body->parameter_instruction(0);
  *body_param->mutable_shape()->mutable_tuple_shapes(carry_idx) = new_shape;

  // Update GTEs in body that use this param and index
  for (HloInstruction* user : body_param->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == carry_idx) {
      *user->mutable_shape() = new_shape;
    }
  }

  // Update cond param
  HloComputation* cond = while_op->while_condition();
  HloInstruction* cond_param = cond->parameter_instruction(0);
  *cond_param->mutable_shape()->mutable_tuple_shapes(carry_idx) = new_shape;

  for (HloInstruction* user : cond_param->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == carry_idx) {
      *user->mutable_shape() = new_shape;
    }
  }

  HloComputation* parent = while_op->parent();
  HloModule* module = parent->parent();
  if (parent == module->entry_computation()) {
    *module->mutable_entry_computation_layout() =
        ComputationLayout(module->entry_computation()->ComputeProgramShape());
  }

  LOG(INFO) << "Jetski: while_op shape after: " << while_op->shape().ToString();
  LOG(INFO) << "Jetski: init shape after: " << init->shape().ToString();
  return absl::OkStatus();
}

absl::StatusOr<bool> ProcessLoopCarriedAllGather(HloInstruction* while_op,
                                                 int64_t carry_idx,
                                                 HloInstruction* start_outside,
                                                 HloInstruction* start_inside,
                                                 HloInstruction* done_inside,
                                                 bool use_legacy_collectives) {
  LOG(INFO) << "ProcessLoopCarriedAllGather entered for while_op: "
            << while_op->name()

            << ", carry_idx: " << carry_idx
            << ", start_outside: " << start_outside->name()
            << ", start_inside: " << start_inside->name()
            << ", done_inside: " << done_inside->name();

  HloComputation* body = while_op->while_body();
  HloComputation* parent = while_op->parent();

  std::string config_str = GetConfigString(start_outside);
  ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                   ParseAsyncCollectiveConfig(config_str));

  TF_RET_CHECK(config.all_gather_dimension.has_value());
  int64_t all_gather_dim = *config.all_gather_dimension;
  Shape shape = done_inside->shape();
  auto device_list =
      std::make_shared<CollectiveDeviceList>(config.replica_groups);

  HloInstruction* async_start_outside = nullptr;
  HloInstruction* async_done_outside = nullptr;
  HloInstruction* async_start_inside = nullptr;
  HloInstruction* async_done_inside = nullptr;

  if (use_legacy_collectives) {
    std::vector<const Shape*> operand_shapes_outside;
    for (const HloInstruction* op : start_outside->operands()) {
      operand_shapes_outside.push_back(&op->shape());
    }
    Shape start_shape_outside = ShapeUtil::MakeTupleShape(
        {start_outside->operand_count() > 1
             ? ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes_outside)
             : *operand_shapes_outside[0],
         shape});
    async_start_outside =
        parent->AddInstruction(HloInstruction::CreateAllGatherStart(
            start_shape_outside, start_outside->operands(), all_gather_dim,
            device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids));
    async_done_outside = parent->AddInstruction(HloInstruction::CreateUnary(
        shape, HloOpcode::kAllGatherDone, async_start_outside));

    std::vector<const Shape*> operand_shapes_inside;
    for (const HloInstruction* op : start_inside->operands()) {
      operand_shapes_inside.push_back(&op->shape());
    }
    Shape start_shape_inside = ShapeUtil::MakeTupleShape(
        {start_inside->operand_count() > 1
             ? ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes_inside)
             : *operand_shapes_inside[0],
         shape});
    async_start_inside =
        body->AddInstruction(HloInstruction::CreateAllGatherStart(
            start_shape_inside, start_inside->operands(), all_gather_dim,
            device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids));

    async_done_inside = body->AddInstruction(HloInstruction::CreateUnary(
        shape, HloOpcode::kAllGatherDone, async_start_inside));
  } else {
    std::unique_ptr<HloInstruction> sync_op_outside =
        HloInstruction::CreateAllGather(
            shape, start_outside->operands(), all_gather_dim, device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids);
    ASSIGN_OR_RETURN(async_done_outside,
                     parent->CreateAsyncInstructions(
                         sync_op_outside.get(), /*context_shapes=*/{},
                         parent->execution_thread(), /*replace=*/false));
    async_start_outside = async_done_outside->mutable_operand(0);

    std::unique_ptr<HloInstruction> sync_op_inside =
        HloInstruction::CreateAllGather(
            shape, start_inside->operands(), all_gather_dim, device_list,
            /*constrain_layout=*/false, config.channel_id,
            config.use_global_device_ids);
    ASSIGN_OR_RETURN(async_done_inside,
                     body->CreateAsyncInstructions(
                         sync_op_inside.get(), /*context_shapes=*/{},
                         body->execution_thread(), /*replace=*/false));
    async_start_inside = async_done_inside->mutable_operand(0);
  }

  async_start_outside->set_metadata(start_outside->metadata());
  async_start_inside->set_metadata(start_inside->metadata());
  async_done_inside->set_metadata(done_inside->metadata());

  HloInstruction* body_root = body->root_instruction();
  HloInstruction* carry_out_gte = body_root->mutable_operand(carry_idx);
  std::vector<HloInstruction*> path2 =
      FindPathBetween(start_inside, carry_out_gte);
  LOG(INFO) << "path2 (start_inside to carry_out_gte) size: " << path2.size();
  for (HloInstruction* inst : path2) {
    LOG(INFO) << "  path2 element: " << inst->ToString();
  }
  HloInstruction* body_root_operand = async_start_inside;
  if (!path2.empty()) {
    ASSIGN_OR_RETURN(body_root_operand,
                     PropagateShape(start_inside, async_start_inside, path2));
  }

  RETURN_IF_ERROR(PropagateLoopCarryType(
      while_op, carry_idx, start_outside, async_start_outside,
      async_start_inside, body_root_operand));

  HloInstruction* body_param = body->parameter_instruction(0);
  HloInstruction* param_gte = nullptr;
  for (HloInstruction* user : body_param->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == carry_idx) {
      param_gte = user;
      break;
    }
  }
  TF_RET_CHECK(param_gte != nullptr);

  std::vector<HloInstruction*> path1 =
      FindPathFromStartToDone(param_gte, done_inside);
  LOG(INFO) << "path1 (param_gte to done_inside) size: " << path1.size();
  for (HloInstruction* inst : path1) {
    LOG(INFO) << "  path1 element: " << inst->ToString();
  }
  if (!path1.empty()) {
    RETURN_IF_ERROR(PropagateShapeHelper(param_gte, done_inside, path1,
                                         param_gte, async_done_inside));
  } else {
    RETURN_IF_ERROR(async_done_inside->ReplaceOperandWith(0, param_gte));
  }
  RETURN_IF_ERROR(done_inside->ReplaceAllUsesWith(async_done_inside));

  RETURN_IF_ERROR(body->RemoveInstruction(done_inside));
  TF_RET_CHECK(start_inside->IsDead());
  RETURN_IF_ERROR(body->RemoveInstruction(start_inside));

  // Find while_gte
  HloInstruction* while_gte = nullptr;
  for (HloInstruction* user : while_op->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == carry_idx) {
      while_gte = user;
      break;
    }
  }
  TF_RET_CHECK(while_gte != nullptr);

  // Find done_outside
  HloInstruction* done_outside = TraceForwardToDone(while_gte);
  if (done_outside != nullptr) {
    LOG(INFO) << "Found done_outside: " << done_outside->ToString();
    async_done_outside->set_metadata(done_outside->metadata());
    std::vector<HloInstruction*> path_outside_done =
        FindPathFromStartToDone(while_gte, done_outside);
    LOG(INFO) << "path_outside_done size: " << path_outside_done.size();
    for (HloInstruction* inst : path_outside_done) {
      LOG(INFO) << "  path_outside_done element: " << inst->ToString();
    }
    if (!path_outside_done.empty()) {
      RETURN_IF_ERROR(PropagateShapeHelper(while_gte, done_outside,
                                           path_outside_done, while_gte,
                                           async_done_outside));
    } else {
      RETURN_IF_ERROR(async_done_outside->ReplaceOperandWith(0, while_gte));
    }
    RETURN_IF_ERROR(done_outside->ReplaceAllUsesWith(async_done_outside));
    RETURN_IF_ERROR(parent->RemoveInstruction(done_outside));
  } else {
    LOG(INFO) << "done_outside NOT found for while_gte: "
              << while_gte->ToString();
    RETURN_IF_ERROR(parent->RemoveInstruction(async_done_outside));
  }

  TF_RET_CHECK(start_outside->IsDead());
  RETURN_IF_ERROR(parent->RemoveInstruction(start_outside));

  return true;
}

absl::StatusOr<bool> ProcessLoopCarriedGroup(HloInstruction* while_op,
                                             int64_t carry_idx,
                                             HloInstruction* start_outside,
                                             HloInstruction* start_inside,
                                             HloInstruction* done_inside,
                                             bool use_legacy_collectives) {
  absl::string_view target = done_inside->custom_call_target();
  if (absl::StartsWith(target, "all-gather")) {
    return ProcessLoopCarriedAllGather(while_op, carry_idx, start_outside,
                                       start_inside, done_inside,
                                       use_legacy_collectives);
  }
  return false;
}

absl::StatusOr<bool> RewriteLoneDone(HloComputation* computation,
                                     HloInstruction* done_call,
                                     bool use_legacy_collectives) {
  HloInstruction* operand = done_call->mutable_operand(0);
  absl::string_view target = done_call->custom_call_target();
  HloOpcode done_opcode = HloOpcode::kAsyncDone;

  if (use_legacy_collectives) {
    if (absl::StartsWith(target, "all-gather")) {
      done_opcode = HloOpcode::kAllGatherDone;
    } else if (absl::StartsWith(target, "all-reduce")) {
      done_opcode = HloOpcode::kAllReduceDone;
    } else if (absl::StartsWith(target, "collective-permute")) {
      done_opcode = HloOpcode::kCollectivePermuteDone;
    } else {
      return false;
    }
  }

  HloInstruction* async_done;
  if (use_legacy_collectives) {
    async_done = computation->AddInstruction(
        HloInstruction::CreateUnary(done_call->shape(), done_opcode, operand));
  } else {
    async_done = computation->AddInstruction(
        HloInstruction::CreateAsyncDone(done_call->shape(), operand));
  }

  async_done->set_metadata(done_call->metadata());
  RETURN_IF_ERROR(done_call->ReplaceAllUsesWith(async_done));
  RETURN_IF_ERROR(computation->RemoveInstruction(done_call));
  return true;
}

}  // namespace

absl::StatusOr<bool> AsyncCollectiveCustomCallRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->name() == "jit_g") {
    LOG(INFO) << "=== HLO Module jit_g at rewriter entry ===";
    LOG(INFO) << module->ToString();
    LOG(INFO) << "=== End HLO Module jit_g at rewriter entry ===";
  }
  bool has_ag_start = false;

  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kCustomCall) {
        LOG(INFO) << "Found custom call in " << comp->name() << ": "
                  << instr->custom_call_target() << " -> " << instr->ToString();
        if (absl::EndsWith(instr->custom_call_target(), "all-gather-start")) {
          has_ag_start = true;
        }
      }
    }
  }
  bool changed = false;
  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder();
  for (HloComputation* computation : computations) {
    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();
    for (HloInstruction* instr : instructions) {
      if (IsCollectiveCustomCall(instr, "-done")) {
        HloInstruction* done_call = instr;
        if (done_call->operand_count() > 0) {
          ASSIGN_OR_RETURN(HloInstruction * new_done,
                           SinkFormattingOps(done_call));
          if (new_done != done_call) {
            changed = true;
          }
        }
      }
    }

    std::vector<std::pair<HloInstruction*, HloInstruction*>> pairs_to_rewrite;
    std::vector<HloInstruction*> lone_dones_to_rewrite;
    struct LoopCarriedGroup {
      HloInstruction* while_op;
      int64_t carry_idx;
      HloInstruction* start_outside;
      HloInstruction* start_inside;
      HloInstruction* done_inside;
    };
    std::vector<LoopCarriedGroup> groups_to_rewrite;

    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (IsCollectiveCustomCall(instr, "-done")) {
        HloInstruction* done_call = instr;
        LOG(INFO) << "Found done call: " << done_call->ToString()
                  << " in computation " << computation->name();
        if (done_call->operand_count() == 0) {
          LOG(INFO) << "Done call has no operands";
          continue;
        }
        std::vector<HloInstruction*> starts = FindMatchingStarts(done_call);
        LOG(INFO) << "Found " << starts.size() << " matching starts for "
                  << done_call->name();
        for (HloInstruction* start : starts) {
          LOG(INFO) << "  Start: " << start->ToString();
        }

        if (starts.size() == 1) {
          if (starts[0]->opcode() == HloOpcode::kCustomCall) {
            pairs_to_rewrite.push_back({starts[0], done_call});
          } else {
            lone_dones_to_rewrite.push_back(done_call);
          }
        } else if (starts.size() == 2) {
          HloInstruction* start_outside = nullptr;
          HloInstruction* start_inside = nullptr;
          for (HloInstruction* s : starts) {
            if (s->parent() == computation) {
              start_inside = s;
            } else {
              start_outside = s;
            }
          }
          if (start_outside && start_inside) {
            if (start_outside->opcode() == HloOpcode::kCustomCall &&
                start_inside->opcode() == HloOpcode::kCustomCall) {
              const auto& callers = computation->caller_instructions();
              if (callers.size() == 1 &&
                  callers[0]->opcode() == HloOpcode::kWhile) {
                HloInstruction* while_op = callers[0];
                auto param_idx =
                    ResolveParameterIndex(done_call->mutable_operand(0));
                if (param_idx.has_value() &&
                    param_idx->parameter->parameter_number() == 0 &&
                    param_idx->parameter->parent() == computation) {
                  groups_to_rewrite.push_back({while_op, param_idx->index,
                                               start_outside, start_inside,
                                               done_call});
                }
              }
            } else if (start_outside->opcode() != HloOpcode::kCustomCall &&
                       start_inside->opcode() != HloOpcode::kCustomCall) {
              lone_dones_to_rewrite.push_back(done_call);
            }
          }
        }
      }
    }

    for (const auto& pair : pairs_to_rewrite) {
      ASSIGN_OR_RETURN(bool pair_changed,
                       ProcessPair(computation, pair.first, pair.second,
                                   use_legacy_collectives_));
      if (pair_changed) {
        changed = true;
      }
    }

    for (const auto& group : groups_to_rewrite) {
      ASSIGN_OR_RETURN(
          bool group_changed,
          ProcessLoopCarriedGroup(group.while_op, group.carry_idx,
                                  group.start_outside, group.start_inside,
                                  group.done_inside, use_legacy_collectives_));
      if (group_changed) {
        changed = true;
      }
    }

    for (HloInstruction* done : lone_dones_to_rewrite) {
      ASSIGN_OR_RETURN(
          bool done_changed,
          RewriteLoneDone(computation, done, use_legacy_collectives_));
      if (done_changed) {
        changed = true;
      }
    }
  }
  // Fallback for any remaining collective custom calls.
  std::vector<HloInstruction*> remaining_starts;
  std::vector<HloInstruction*> remaining_dones;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* inst : comp->instructions()) {
      if (IsCollectiveCustomCall(inst, "-start")) {
        remaining_starts.push_back(inst);
      } else if (IsCollectiveCustomCall(inst, "-done")) {
        remaining_dones.push_back(inst);
      }
    }
  }

  for (HloInstruction* done : remaining_dones) {
    LOG(INFO) << "Falling back to sync: removing done: " << done->ToString();
    RETURN_IF_ERROR(done->ReplaceAllUsesWith(done->mutable_operand(0)));
    RETURN_IF_ERROR(done->parent()->RemoveInstruction(done));
    changed = true;
  }

  for (HloInstruction* start : remaining_starts) {
    LOG(INFO) << "Falling back to sync: converting start to sync: "
              << start->ToString();
    absl::string_view target = start->custom_call_target();
    if (absl::StartsWith(target, "all-gather")) {
      std::string config_str = GetConfigString(start);
      ASSIGN_OR_RETURN(AsyncCollectiveConfig config,
                       ParseAsyncCollectiveConfig(config_str));
      TF_RET_CHECK(config.all_gather_dimension.has_value());
      int64_t all_gather_dim = *config.all_gather_dimension;
      auto device_list =
          std::make_shared<CollectiveDeviceList>(config.replica_groups);

      HloInstruction* sync_ag =
          start->parent()->AddInstruction(HloInstruction::CreateAllGather(
              start->shape(), start->operands(), all_gather_dim, device_list,
              /*constrain_layout=*/false, config.channel_id,
              config.use_global_device_ids));
      sync_ag->set_metadata(start->metadata());
      RETURN_IF_ERROR(start->ReplaceAllUsesWith(sync_ag));
      RETURN_IF_ERROR(start->parent()->RemoveInstruction(start));
      changed = true;
    } else {
      return Unimplemented(
          "Fallback to sync not implemented for collective: %s",
          std::string(target));
    }
  }

  if (has_ag_start && !changed) {
    LOG(WARNING) << "Module " << module->name()
                 << " has all-gather-start but was NOT changed by rewriter!";
    LOG(INFO) << "HLO module: " << module->ToString();
  }
  if (module->name() == "jit_g" && changed) {
    for (HloComputation* comp : module->computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kWhile) {
          LOG(INFO) << "Jetski: RunImpl final while_op shape: "
                    << instr->shape().ToString();
          LOG(INFO) << "Jetski: RunImpl final init shape: "
                    << instr->operand(0)->shape().ToString();
        }
      }
    }
    LOG(INFO) << "=== HLO Module jit_g after rewriter ===";
    LOG(INFO) << module->ToString();
    LOG(INFO) << "=== End HLO Module jit_g after rewriter ===";
  }
  return changed;
}

}  // namespace xla
