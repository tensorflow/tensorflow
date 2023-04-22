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

#include "tensorflow/compiler/xla/service/while_loop_all_reduce_code_motion.h"

#include <iterator>
#include <stack>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/status.h"

namespace xla {

namespace {

struct AccumulationContext {
  HloInstruction* accumulation_instruction;
  HloInstruction* accumulation_buffer;
  std::vector<int> param_tuple_indices;
};

// Describes whether an all-reduce instruction can be sinked from a while body
// computation and all the accumulation uses of the all-reduce's result in the
// while body if movable.
struct MovableAllReduceContext {
  bool is_movable;
  // If movable, `accumulation_contexts` contains one accumulation
  // context for each accumulation in the while body that uses the all-reduce's
  // result. Otherwise, this field is undefined.
  std::vector<AccumulationContext> accumulation_contexts;
};

// Checks if an all-reduce instruction is eligible for sinking and finds all of
// the all-reduce's accumulation uses inside the while body if eligible.
MovableAllReduceContext IsAllReduceMovable(HloInstruction* all_reduce,
                                           HloComputation* while_body) {
  auto all_reduce_is_summation = [](HloInstruction* all_reduce) -> bool {
    HloInstruction* to_apply_root = all_reduce->to_apply()->root_instruction();
    if (all_reduce->to_apply()->num_parameters() != 2) {
      return false;
    }
    return Match(to_apply_root,
                 match::AddAnyOrder(match::Parameter(0), match::Parameter(1)));
  };

  // We only support numerical types.
  const absl::InlinedVector<PrimitiveType, 12> kSupportedTypes{
      BF16, F16, F32, F64, S8, S16, S32, S64, U8, U16, U32, U64};

  if (!absl::c_linear_search(kSupportedTypes,
                             all_reduce->shape().element_type()) ||
      !all_reduce_is_summation(all_reduce)) {
    return MovableAllReduceContext{/*is_movable=*/false,
                                   /*accumulation_contexts=*/{}};
  }

  struct BufferTupleIndex {
    bool unsupported_operation{false};
    std::vector<int> tuple_index;
    bool returned_from_computation{false};
  };

  // If the instruction is a buffer forwarded from a tuple element of the
  // computation's parameter, returns the indices of the buffer in the parameter
  // tuple. The returned_from_computation field in the result is unused.
  auto get_origin_tuple_index =
      [](HloInstruction* instruction) -> BufferTupleIndex {
    // The returned_from_computation is never touched in this function.
    BufferTupleIndex result;
    while (!result.unsupported_operation) {
      switch (instruction->opcode()) {
        default:
          result.unsupported_operation = true;
          break;
        case HloOpcode::kBitcast:
        case HloOpcode::kConvert:
        case HloOpcode::kReshape:
        case HloOpcode::kTranspose:
        case HloOpcode::kDynamicReshape:
          instruction = instruction->mutable_operand(0);
          break;
        case HloOpcode::kGetTupleElement: {
          if (!result.tuple_index.empty()) {
            // Note that we don't support nested tuples as of now.
            result.unsupported_operation = true;
          } else {
            result.tuple_index.push_back(
                Cast<HloGetTupleElementInstruction>(instruction)
                    ->tuple_index());
            instruction = instruction->mutable_operand(0);
          }
          break;
        }
        case HloOpcode::kParameter: {
          int parameter_number =
              Cast<HloParameterInstruction>(instruction)->parameter_number();
          CHECK_EQ(parameter_number, 0);
          break;
        }
      }
      if (instruction->opcode() == HloOpcode::kParameter) {
        break;
      }
    }
    return result;
  };

  // If the instruction's result is returned from its parent computation with
  // only forwarding operations, returns the index of the result buffer in the
  // output parameter tuple.
  auto get_output_tuple_index =
      [](HloInstruction* instruction,
         HloComputation* while_body) -> BufferTupleIndex {
    BufferTupleIndex result;
    std::stack<HloInstruction*> to_visit;
    to_visit.push(instruction);
    while (!to_visit.empty() && !result.unsupported_operation) {
      HloInstruction* instruction = to_visit.top();
      to_visit.pop();
      for (HloInstruction* user : instruction->users()) {
        switch (user->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kConvert:
          case HloOpcode::kReshape:
          case HloOpcode::kGetTupleElement:
          case HloOpcode::kTranspose:
          case HloOpcode::kSlice: {
            to_visit.push(user);
            break;
          }
          case HloOpcode::kDynamicSlice: {
            if (user->operand_index(instruction) == 0) {
              to_visit.push(user);
            } else {
              result.unsupported_operation = true;
            }
            break;
          }
          case HloOpcode::kTuple: {
            if (!result.tuple_index.empty()) {
              // Note that we don't support nested tuples as of now.
              result.unsupported_operation = true;
            } else {
              result.tuple_index.push_back(user->operand_index(instruction));
              if (while_body->root_instruction() == user) {
                if (result.returned_from_computation) {
                  result.unsupported_operation = true;
                }
                result.returned_from_computation = true;
              } else {
                to_visit.push(user);
              }
            }
            break;
          }
          default:
            result.unsupported_operation = true;
        }
        if (result.unsupported_operation) {
          break;
        }
      }
    }
    return result;
  };

  // Checks whether any buffer in the list of accumulation contexts is used in
  // the parent computation except for forwarding uses.
  auto is_buffer_used =
      [](absl::Span<const AccumulationContext> accumulation_contexts,
         HloComputation* while_body_computation) -> bool {
    std::vector<HloInstruction*> parameter_instructions;
    absl::c_copy_if(while_body_computation->instructions(),
                    std::back_inserter(parameter_instructions),
                    [](HloInstruction* instruction) -> bool {
                      return instruction->opcode() == HloOpcode::kParameter;
                    });
    for (const auto& accumulation : accumulation_contexts) {
      HloInstruction* accumulation_instruction =
          accumulation.accumulation_instruction;
      int tuple_index = accumulation.param_tuple_indices[0];
      std::stack<HloInstruction*> to_visit;
      // TODO(b/176437845): simplify the logic below by using
      // TuplePointsToAnalysis.
      for (HloInstruction* parameter_instruction : parameter_instructions) {
        // Iterate over all users of the while body parameter and find all
        // instructions that use the accumulation buffer, as specified by
        // tuple_index.
        // This logic could be simplied by using TuplePointsToAnalysis, which
        // we leave to a future CL (see TODO above).
        for (HloInstruction* user : parameter_instruction->users()) {
          if (auto* gte = DynCast<HloGetTupleElementInstruction>(user)) {
            if (gte->tuple_index() == tuple_index) {
              to_visit.push(user);
            }
          } else {
            return true;
          }
        }
      }

      while (!to_visit.empty()) {
        HloInstruction* instruction = to_visit.top();
        to_visit.pop();
        for (HloInstruction* user : instruction->users()) {
          switch (user->opcode()) {
            case HloOpcode::kBitcast:
            case HloOpcode::kConvert:
            case HloOpcode::kReshape:
            case HloOpcode::kTranspose:
              to_visit.push(user);
              break;
            case HloOpcode::kDynamicReshape: {
              if (instruction == user->operand(0)) {
                to_visit.push(user);
              } else {
                return true;
              }
              break;
            }
            case HloOpcode::kAdd: {
              if (user != accumulation_instruction) {
                return true;
              }
              break;
            }
            default:
              return true;
          }
        }
      }
    }
    return false;
  };

  // Finds all accumulation contexts of the given all-reduce instruction if it
  // is movable.
  auto get_accumulation_contexts =
      [&get_origin_tuple_index, &get_output_tuple_index, &is_buffer_used](
          HloInstruction* all_reduce,
          HloComputation* while_body) -> MovableAllReduceContext {
    std::vector<AccumulationContext> accumulation_contexts;
    // DFS starting from the all-reduce instruction and stops at the first
    // non-triival uses of the all-reduce result or finds all accmululations
    // of the all-reduce result.
    std::stack<HloInstruction*> to_visit;
    // By default movable unless we find that it's not.
    bool is_all_reduce_movable = true;
    to_visit.push(all_reduce);

    while (!to_visit.empty() && is_all_reduce_movable) {
      HloInstruction* instruction = to_visit.top();
      to_visit.pop();
      for (HloInstruction* user : instruction->users()) {
        switch (user->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kConvert:
          case HloOpcode::kReshape:
          case HloOpcode::kGetTupleElement:
          case HloOpcode::kTranspose:
          case HloOpcode::kSlice: {
            to_visit.push(user);
            break;
          }
          case HloOpcode::kDynamicSlice: {
            if (user->operand_index(instruction) == 0) {
              to_visit.push(user);
            } else {
              is_all_reduce_movable = false;
              break;
            }
            break;
          }
          case HloOpcode::kAdd: {
            int64 buffer_index = 1 - user->operand_index(instruction);
            HloInstruction* accumulation_buffer =
                user->mutable_operand(buffer_index);

            auto origin_buffer_tuple_index =
                get_origin_tuple_index(accumulation_buffer);
            if (origin_buffer_tuple_index.unsupported_operation) {
              is_all_reduce_movable = false;
              break;
            }

            auto output_buffer_tuple_index =
                get_output_tuple_index(user, while_body);
            if (!output_buffer_tuple_index.unsupported_operation &&
                output_buffer_tuple_index.returned_from_computation &&
                !origin_buffer_tuple_index.tuple_index.empty() &&
                ContainersEqual(origin_buffer_tuple_index.tuple_index,
                                output_buffer_tuple_index.tuple_index)) {
              accumulation_contexts.push_back(AccumulationContext{
                  user, accumulation_buffer,
                  std::move(output_buffer_tuple_index.tuple_index)});
            } else {
              is_all_reduce_movable = false;
            }
            break;
          }
          default:
            is_all_reduce_movable = false;
        }
      }
    }
    if (is_buffer_used(accumulation_contexts, while_body)) {
      is_all_reduce_movable = false;
    }
    return MovableAllReduceContext{is_all_reduce_movable,
                                   accumulation_contexts};
  };
  return get_accumulation_contexts(all_reduce, while_body);
}

struct WhileInitContext {
  HloInstruction* while_init{nullptr};
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_old_buffer;
};

// Creates a new while init instruction, which replaces each accumulation buffer
// in the given accumulation contexts with a zero-initialized buffer. In other
// words, we are accumulating all the deltas in the while loop with a zero
// initial value.
WhileInitContext CreateNewWhileInit(
    HloInstruction* old_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  HloInstruction* old_while_init = old_while_instruction->mutable_operand(0);
  HloComputation* while_parent = old_while_instruction->parent();
  std::vector<HloInstruction*> new_while_init_elements(
      old_while_init->operand_count(), nullptr);
  for (const auto& all_reduce_and_accumulations_pair :
       all_reduce_to_accumulations) {
    const std::vector<AccumulationContext>& accumulations =
        all_reduce_and_accumulations_pair.second;
    for (auto& accumulation_context : accumulations) {
      CHECK_EQ(accumulation_context.param_tuple_indices.size(), 1);
      int tuple_index = accumulation_context.param_tuple_indices[0];
      HloInstruction* old_buffer = old_while_init->mutable_operand(tuple_index);
      HloInstruction* new_buffer = while_parent->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateFromDimensions(
              old_buffer->shape().element_type(),
              old_buffer->shape().dimensions())));
      new_while_init_elements[tuple_index] = new_buffer;
    }
  }
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_old_buffer;
  for (int i = 0; i < old_while_init->operand_count(); i++) {
    if (!new_while_init_elements[i]) {
      new_while_init_elements[i] = old_while_init->mutable_operand(i);
    } else {
      tuple_index_to_old_buffer[i] = old_while_init->mutable_operand(i);
    }
  }
  HloInstruction* new_while_init = while_parent->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_elements));
  return WhileInitContext{new_while_init, tuple_index_to_old_buffer};
}

// Creates all the sinked all-reduce instructions in the while instruction's
// parent computation. Returns a map that maps a tuple index of an accumulation
// buffer to it's corresponding all-reduce.
absl::flat_hash_map<int, HloInstruction*> CreateSinkedAllReduces(
    HloInstruction* new_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations,
    const absl::flat_hash_map<int, HloInstruction*>&
        tuple_index_to_old_buffer) {
  HloComputation* while_parent = new_while_instruction->parent();
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_new_buffer;
  for (const auto& all_reduce_and_accumulations_pair :
       all_reduce_to_accumulations) {
    HloInstruction* loop_all_reduce = all_reduce_and_accumulations_pair.first;
    const std::vector<AccumulationContext>& accumulations =
        all_reduce_and_accumulations_pair.second;
    for (const auto& accumulation_context : accumulations) {
      CHECK_EQ(accumulation_context.param_tuple_indices.size(), 1);
      int tuple_index = accumulation_context.param_tuple_indices[0];
      const Shape& accumulation_buffer_shape =
          new_while_instruction->shape().tuple_shapes(tuple_index);
      HloInstruction* accumulation_buffer =
          while_parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              accumulation_buffer_shape, new_while_instruction, tuple_index));
      HloAllReduceInstruction* old_all_reduce =
          Cast<HloAllReduceInstruction>(loop_all_reduce);
      HloInstruction* all_reduce_operand = accumulation_buffer;
      if (!ShapeUtil::SameElementType(old_all_reduce->shape(),
                                      accumulation_buffer_shape)) {
        Shape all_reduce_shape =
            ShapeUtil::MakeShape(old_all_reduce->shape().element_type(),
                                 accumulation_buffer_shape.dimensions());
        all_reduce_operand =
            while_parent->AddInstruction(HloInstruction::CreateConvert(
                all_reduce_shape, accumulation_buffer));
      }
      HloInstruction* new_all_reduce =
          while_parent->AddInstruction(HloInstruction::CreateAllReduce(
              all_reduce_operand->shape(), {all_reduce_operand},
              old_all_reduce->called_computations()[0],
              old_all_reduce->replica_groups(),
              old_all_reduce->constrain_layout(),
              hlo_query::NextChannelId(*(while_parent->parent())),
              old_all_reduce->use_global_device_ids()));
      HloInstruction* all_reduced_delta = new_all_reduce;
      if (!ShapeUtil::SameElementType(all_reduced_delta->shape(),
                                      accumulation_buffer_shape)) {
        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateConvert(
                accumulation_buffer_shape, all_reduced_delta));
      }
      CHECK(ContainsKey(tuple_index_to_old_buffer, tuple_index));
      HloInstruction* old_buffer = tuple_index_to_old_buffer.at(tuple_index);
      CHECK(Shape::Equal().IgnoreLayout()(old_buffer->shape(),
                                          all_reduced_delta->shape()));
      HloInstruction* add_to_old_buffer =
          while_parent->AddInstruction(HloInstruction::CreateBinary(
              all_reduced_delta->shape(), HloOpcode::kAdd, old_buffer,
              all_reduced_delta));
      tuple_index_to_new_buffer[tuple_index] = add_to_old_buffer;
    }
  }
  return tuple_index_to_new_buffer;
}

// Creates a tuple which is equivalent to the original while instruction's
// output.
HloInstruction* CreateNewWhileResult(
    HloInstruction* new_while_instruction,
    const absl::flat_hash_map<int, HloInstruction*>&
        tuple_index_to_new_buffer) {
  HloComputation* while_parent = new_while_instruction->parent();
  CHECK(new_while_instruction->shape().IsTuple());
  std::vector<HloInstruction*> new_while_result_elements(
      new_while_instruction->shape().tuple_shapes_size(), nullptr);
  for (int i = 0; i < new_while_result_elements.size(); i++) {
    if (ContainsKey(tuple_index_to_new_buffer, i)) {
      new_while_result_elements[i] = tuple_index_to_new_buffer.at(i);
    } else {
      HloInstruction* gte =
          while_parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_while_instruction->shape().tuple_shapes(i),
              new_while_instruction, i));
      new_while_result_elements[i] = gte;
    }
  }
  HloInstruction* new_while_result = while_parent->AddInstruction(
      HloInstruction::CreateTuple(new_while_result_elements));
  return new_while_result;
}

// Creates the sinked all-reduce instructions for all accumulation buffers. The
// all-reduce outputs are then added to the original accumulation buffers.
// Creates a tuple that groups the while loop output and the accumulated
// buffers and replaces all uses of the old while with this new tuple.
Status AddSinkedAllReducesAndReplaceWhile(
    HloInstruction* while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  // Note that we create all instructions before replacing and removing any old
  // instruction. This ensures that we do not accidentally access any deleted
  // instruction when creating new instructions.

  // Step 1) create the new while init instruction, which uses zero-initialized
  // tensors as the accumulation buffers for the all-reduce.
  auto new_while_init_context =
      CreateNewWhileInit(while_instruction, all_reduce_to_accumulations);
  // Step 2) create the new while instruction.
  HloInstruction* new_while_instruction =
      while_instruction->parent()->AddInstruction(HloInstruction::CreateWhile(
          new_while_init_context.while_init->shape(),
          while_instruction->while_condition(), while_instruction->while_body(),
          new_while_init_context.while_init));
  // Step 3) create the new all-reduce instructions after the while loop.
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_new_buffer =
      CreateSinkedAllReduces(new_while_instruction, all_reduce_to_accumulations,
                             new_while_init_context.tuple_index_to_old_buffer);
  // Step 4) create the tuple and replace the old while instruction for all of
  // its uses.
  HloInstruction* new_while_result =
      CreateNewWhileResult(new_while_instruction, tuple_index_to_new_buffer);
  TF_RETURN_IF_ERROR(while_instruction->parent()->ReplaceInstruction(
      while_instruction, new_while_result));
  return Status::OK();
}

}  // namespace

StatusOr<bool> WhileLoopAllReduceCodeMotion::Run(HloModule* module) {
  bool is_changed = false;
  bool run_next_pass = true;
  // In case of MPMD, all-reduces might be cross-module and should preserve
  // their channel ID. Do not move all-reduces in this case since the channel
  // ID might be changed.
  if (module->config().num_partitions() > 1 &&
      !module->config().use_spmd_partitioning()) {
    return false;
  }
  // The while instruction's parent could be a while body for another while
  // loop. We recursively sink the all-reduce through nested while loops if
  // applicable by repeating this process.
  while (run_next_pass) {
    run_next_pass = false;
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
    // A computation could be the while body of multiple while instructions,
    // so we start from the computation and find all of its callers that is a
    // kWhile if there is any.
    for (HloComputation* computation : module->computations()) {
      std::vector<HloInstruction*> computation_callers =
          call_graph->GetComputationCallers(computation);
      std::vector<HloInstruction*> while_caller_instructions;
      for (HloInstruction* caller_instruction : computation_callers) {
        // For simplicity, we only support while instructions whose shape is
        // tuple.
        if (caller_instruction->opcode() == HloOpcode::kWhile &&
            caller_instruction->shape().IsTuple() &&
            caller_instruction->while_body() == computation) {
          while_caller_instructions.push_back(caller_instruction);
        }
      }
      // Skip to next computation if this computation is not the while body of
      // any while instruction.
      if (while_caller_instructions.empty()) {
        continue;
      }
      std::vector<HloInstruction*> while_body_all_reduces;
      for (HloInstruction* while_body_instruction :
           computation->MakeInstructionPostOrder()) {
        if (auto* all_reduce_instruction =
                DynCast<HloAllReduceInstruction>(while_body_instruction)) {
          if (all_reduce_instruction->constrain_layout()) {
            return false;
          } else {
            while_body_all_reduces.push_back(all_reduce_instruction);
          }
        }
      }
      HloInstructionMap<std::vector<AccumulationContext>>
          all_reduce_to_accumulations;
      for (HloInstruction* all_reduce : while_body_all_reduces) {
        auto movable_all_reduce_context =
            IsAllReduceMovable(all_reduce, computation);
        if (movable_all_reduce_context.is_movable) {
          all_reduce_to_accumulations[all_reduce] =
              std::move(movable_all_reduce_context.accumulation_contexts);
        }
      }
      if (all_reduce_to_accumulations.empty()) {
        continue;
      }
      // For each while instruction calling this computation, create the
      // corresponding all-reduces after the while loop.
      for (HloInstruction* while_instruction : while_caller_instructions) {
        TF_RETURN_IF_ERROR(AddSinkedAllReducesAndReplaceWhile(
            while_instruction, all_reduce_to_accumulations));
        is_changed = true;
        run_next_pass = true;
      }
      // At last, remove the old all-reduce instructions in the while body.
      for (const auto& all_reduce_accumulations_pair :
           all_reduce_to_accumulations) {
        HloInstruction* all_reduce = all_reduce_accumulations_pair.first;
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
            all_reduce, all_reduce->mutable_operand(0)));
      }
    }
  }
  return is_changed;
}

}  // namespace xla
