/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/windowed_einsum_handler.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace m = match;

// Enables the creation of FP8 GEMM Custom Calls for all-gather and
// reduce-scatter windowed einsums in gemm_rewriter.cc by moving the scalings
// and type conversions of FP8 operands into the bodies of their while loops,
// i.e. rewrites
//
//   inputs --> dequant --> (unary) --> while loop {collective-permute/dot/etc}
//
// into
//
//   inputs --> (unary) --> while loop {dequant --> collective-permute/dot/etc}.
//
// Unary bitcast, broadcast, copy, reshape and transpose ops are allowed between
// dequantization and while loop. Returns the new while loop if the computation
// was changed.
absl::StatusOr<HloInstruction*> ShiftDequantizationF8(
    HloComputation* while_body) {
  auto maybe_while = while_body->GetUniqueCaller(HloOpcode::kWhile);
  // The input of the while loop will be modified and must have no other users.
  if (!maybe_while || (*maybe_while)->operand(0)->user_count() != 1) {
    return absl::InvalidArgumentError("Expected body to be a loop.");
  }
  HloInstruction* while_instr = *maybe_while;

  // Identify the scalings and type conversions applied to the inputs of the
  // while loop.
  HloInstruction* param_tuple = while_instr->mutable_operand(0);
  std::array<HloInstruction*, 2> binaries, operands, scales;
  std::array<std::vector<HloInstruction*>, 2> unaries;
  for (int k = 0; k < 2; ++k) {
    HloInstruction* operand = param_tuple->mutable_operand(k);
    // Capture bitcast, broadcast, copy, reshape and transpose ops between
    // dequantization and the loop.
    while (HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kBroadcast,
                            HloOpcode::kCopy, HloOpcode::kReshape,
                            HloOpcode::kTranspose>(operand)) {
      unaries[k].push_back(operand);
      operand = operand->mutable_operand(0);
    }
    std::reverse(unaries[k].begin(), unaries[k].end());
    if (!Match(operand,
               m::AnyOf<HloInstruction>(
                   m::Divide(&binaries[k], m::Convert(m::Op(&operands[k])),
                             m::Broadcast(m::Op(&scales[k]))),
                   m::MultiplyAnyOrder(&binaries[k],
                                       m::Convert(m::Op(&operands[k])),
                                       m::Broadcast(m::Op(&scales[k])))))) {
      VLOG(5) << "Unable to identify FP8 dequantization pattern.";
      return nullptr;
    }
  }

  // For the dot to be rewritten by gemm_rewriter.cc into an FP8 GEMM, at most
  // one of the inputs can be F8E5M2.
  std::array<PrimitiveType, 2> operand_types{
      operands[0]->shape().element_type(), operands[1]->shape().element_type()};
  if (!((operand_types[0] == F8E4M3FN && operand_types[1] == F8E4M3FN) ||
        (operand_types[0] == F8E4M3FN && operand_types[1] == F8E5M2) ||
        (operand_types[0] == F8E5M2 && operand_types[1] == F8E4M3FN))) {
    VLOG(5) << "Unsupported types.";
    return nullptr;
  }

  // The dequantized types must be BF16, FP16 or FP32.
  for (int k = 0; k < 2; ++k) {
    if (binaries[k]->shape().element_type() != BF16 &&
        binaries[k]->shape().element_type() != F16 &&
        binaries[k]->shape().element_type() != F32) {
      VLOG(5) << "Unsupported types.";
      return nullptr;
    }
  }

  // The FP8 scaling operands must be scalars.
  if (!ShapeUtil::IsScalar(scales[0]->shape()) ||
      !ShapeUtil::IsScalar(scales[1]->shape())) {
    VLOG(5) << "Scaling factors must be scalars.";
    return nullptr;
  }

  // Identify the dot, get-tuple-element and collective-permute or dynamic-slice
  // instructions in the all-gather or reduce-scatter patterns in while's body.
  HloComputation* while_condition = while_instr->while_condition();
  HloInstruction* while_root = while_body->root_instruction();
  std::array<HloInstruction*, 2> dots, gtes, dyn_slices{nullptr, nullptr},
      coll_perms{nullptr, nullptr};
  if (Match(while_root,
            m::Tuple(m::CollectivePermute(
                         &coll_perms[1],
                         m::CollectivePermute(
                             &coll_perms[0],
                             m::GetTupleElement(&gtes[0], m::Parameter(), 0))),
                     m::GetTupleElement(&gtes[1], m::Parameter(), 1),
                     m::DynamicUpdateSlice(
                         m::DynamicUpdateSlice().WithOperand(
                             1, m::Dot(&dots[0], m::Op(), m::Op())),
                         m::Dot(&dots[1], m::Op(), m::Op()), m::Op(), m::Op(),
                         m::Op()),
                     m::Op(), m::Op())) &&
      dots[0]->operand(0) == gtes[0] && dots[0]->operand(1) == gtes[1] &&
      dots[1]->operand(1) == gtes[1]) {
    VLOG(5) << "Identified all-gather windowed einsum pattern.";
  } else if (Match(
                 while_root,
                 m::Tuple(m::GetTupleElement(&gtes[0], m::Parameter(), 0),
                          m::GetTupleElement(&gtes[1], m::Parameter(), 1),
                          m::AddAnyOrder(
                              m::Dot(&dots[0], m::DynamicSlice(&dyn_slices[0]),
                                     m::Op()),
                              m::Op()),
                          m::CollectivePermute(m::AddAnyOrder(
                              m::Dot(&dots[1], m::DynamicSlice(&dyn_slices[1]),
                                     m::Op()),
                              m::Op())),
                          m::Op())) &&
             dots[0]->operand(1) == gtes[1] && dots[1]->operand(1) == gtes[1]) {
    VLOG(5) << "Identified reduce-scatter windowed einsum pattern.";
  } else {
    VLOG(5) << "Unable to identify valid windowed einsum pattern.";
    return nullptr;
  }

  // Replace any dequantized bitcast, broadcast, copy, reshape and transpose ops
  // before the while loop with FP8 unary ops.
  for (int k = 0; k < 2; ++k) {
    for (HloInstruction* unary : unaries[k]) {
      Shape new_shape = ShapeUtil::MakeShapeWithDenseLayout(
          operands[k]->shape().element_type(), unary->shape().dimensions(),
          unary->shape().layout().minor_to_major());

      operands[k] = unary->AddInstruction(unary->CloneWithNewOperands(
          ShapeUtil::MakeShapeWithDenseLayout(
              operands[k]->shape().element_type(), unary->shape().dimensions(),
              unary->shape().layout().minor_to_major()),
          {operands[k]}));
    }
  }

  // Replace the dequantized dot operands in the parameter tuple used by while
  // with FP8 operands.
  for (int k = 0; k < 2; ++k) {
    TF_RETURN_IF_ERROR(
        param_tuple->ReplaceOperandWithDifferentShape(k, operands[k]));
    ShapeUtil::UpdateTupleShape(operands[k]->shape(), k,
                                param_tuple->mutable_shape());
    param_tuple->AppendOperand(scales[k]);
    ShapeUtil::AppendShapeToTuple(scales[k]->shape(),
                                  param_tuple->mutable_shape());
  }

  // Update the parameter tuples of while's body and condition computations.
  for (HloComputation* while_comp : {while_body, while_condition}) {
    while_comp->ReplaceParameter(
        0, HloInstruction::CreateParameter(
               0, param_tuple->shape(),
               while_comp->parameter_instruction(0)->name()));
  }

  // In the while body, replace the existing get-tuple-element instructions
  // retrieving BF16/FP16/FP32 dot operands with get-tuple-element
  // instructions retrieving FP8 dot operands from the input tuple.
  HloInstruction* body_param = while_body->parameter_instruction(0);
  for (int k = 0; k < 2; ++k) {
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_f8,
                        MakeGetTupleElementHlo(body_param, k));

    if (while_root->operand(k) == gtes[k]) {
      TF_RETURN_IF_ERROR(
          while_root->ReplaceOperandWithDifferentShape(k, operand_f8));
      ShapeUtil::UpdateTupleShape(operand_f8->shape(), k,
                                  while_root->mutable_shape());
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * operand_scale,
        MakeGetTupleElementHlo(
            body_param, body_param->shape().tuple_shapes().size() - 2 + k));

    // Also add the scaling factor to the output tuple of the while body.
    while_root->AppendOperand(operand_scale);
    ShapeUtil::AppendShapeToTuple(operand_scale->shape(),
                                  while_root->mutable_shape());

    // Dequantize the operands of the dots and dynamic-slices.
    HloInstruction* operand_f32 =
        MakeConvertToHlo(operand_f8, gtes[k]->shape().element_type());
    HloInstruction* broadcast_scale =
        MakeBroadcastHlo(operand_scale, {}, operand_f32->shape());
    TF_ASSIGN_OR_RETURN(
        HloInstruction * operand_scaled,
        MakeBinaryHlo(binaries[k]->opcode(), operand_f32, broadcast_scale));

    // Replace the original get-tuple-element instructions accessing the
    // operands of the dots and dynamic-slices with the dequantized FP8
    // operands. The order of dequantization and dynamic-slices will be
    // exchanged in gemm_rewriter.cc.
    for (int l = 0; l < 2; ++l) {
      if (dots[l]->operand(k) == gtes[k]) {
        TF_RETURN_IF_ERROR(dots[l]->ReplaceOperandWith(k, operand_scaled));
      }
      if (dyn_slices[l] && dyn_slices[l]->operand(0) == gtes[k]) {
        TF_RETURN_IF_ERROR(
            dyn_slices[l]->ReplaceOperandWith(0, operand_scaled));
      }
    }

    // In the all-gather case, coll_perms[0] has two users, coll_perms[1] and
    // dots[1], which prevents it from being exchanged with dequantization in
    // gemm_rewriter.cc. Instead, directly insert the dequantization before
    // dots[1] here.
    if (coll_perms[0] && coll_perms[0]->operand(0) == gtes[k]) {
      std::array<HloInstruction*, 2> coll_perms_f8{nullptr, nullptr};
      // Change the type of both collective-permutes to FP8.
      coll_perms_f8[0] =
          while_body->AddInstruction(coll_perms[0]->CloneWithNewOperands(
              operand_f8->shape(), {operand_f8}));
      coll_perms_f8[1] =
          while_body->AddInstruction(coll_perms[1]->CloneWithNewOperands(
              coll_perms_f8[0]->shape(), {coll_perms_f8[0]}));

      // Insert the dequantization between coll_perms[0] and dots[1].
      HloInstruction* coll_perm0_f32 =
          MakeConvertToHlo(coll_perms_f8[0], gtes[k]->shape().element_type());
      TF_ASSIGN_OR_RETURN(HloInstruction * x_scaled,
                          MakeBinaryHlo(binaries[k]->opcode(), coll_perm0_f32,
                                        broadcast_scale));
      TF_RETURN_IF_ERROR(dots[1]->ReplaceOperandWith(0, x_scaled));

      // Update the output tuple.
      TF_RETURN_IF_ERROR(
          while_root->ReplaceOperandWithDifferentShape(0, coll_perms_f8[1]));
      ShapeUtil::UpdateTupleShape(coll_perms_f8[1]->shape(), 0,
                                  while_root->mutable_shape());
    }
  }

  // Update the shape of the while call in the parent computation.
  HloInstruction* new_while_instr = while_instr->AddInstruction(
      while_instr->CloneWithNewShape(while_root->shape()));
  TF_RETURN_IF_ERROR(
      while_instr->ReplaceAllUsesWithDifferentShape(new_while_instr));
  TF_RETURN_IF_ERROR(while_instr->parent()->RemoveInstruction(while_instr));

  if (coll_perms[0]) {
    TF_RETURN_IF_ERROR(while_body->RemoveInstruction(coll_perms[1]));
    TF_RETURN_IF_ERROR(while_body->RemoveInstruction(coll_perms[0]));
  }
  TF_RETURN_IF_ERROR(while_body->RemoveInstruction(gtes[0]));
  TF_RETURN_IF_ERROR(while_body->RemoveInstruction(gtes[1]));

  VLOG(5) << "FP8 dequantization moved into while loop.";
  return new_while_instr;
}

int64_t NumberOfInstructionsInComp(const HloComputation* comp, HloOpcode op) {
  int64_t total_count = 0;
  for (const HloInstruction* inst : comp->instructions()) {
    if (inst->opcode() == op) {
      ++total_count;
    }
  }
  return total_count;
}

absl::Status UpdateDotAndConsumerConfig(HloInstruction* dot,
                                        int64_t stream_id) {
  auto dot_gpu_config = dot->backend_config<gpu::GpuBackendConfig>();
  HloInstruction* updater = dot->users()[0];
  auto updater_gpu_config = updater->backend_config<gpu::GpuBackendConfig>();
  dot_gpu_config->set_operation_queue_id(stream_id);
  if (!absl::c_linear_search(updater_gpu_config->wait_on_operation_queues(),
                             stream_id)) {
    updater_gpu_config->mutable_wait_on_operation_queues()->Add(stream_id);
  }

  TF_RETURN_IF_ERROR(dot->set_backend_config(dot_gpu_config.value()));
  TF_RETURN_IF_ERROR(updater->set_backend_config(updater_gpu_config.value()));
  return absl::OkStatus();
}

absl::Status SetForceDelayForInstruction(HloInstruction* instr,
                                         bool force_delay) {
  auto gpu_config = instr->backend_config<gpu::GpuBackendConfig>();

  gpu_config->set_force_earliest_schedule(force_delay);

  TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config.value()));
  return absl::OkStatus();
}

static int64_t GetAgActivationCacheIndex(const HloInstruction* while_loop) {
  const HloInstruction* loop_tuple = while_loop->operand(0);
  const Shape& tuple_shape = loop_tuple->shape();
  CHECK(tuple_shape.IsTuple());
  return tuple_shape.tuple_shapes().size() - 1;
}

bool FindDusSliceForCachedActivation(HloInstruction* inst,
                                     HloInstruction** dus_boundary_constant,
                                     HloInstruction** slice_indices,
                                     bool is_first_slice) {
  // We are only interested in DUS in the loop body.
  if (HloPredicateIsNotOp<HloOpcode::kDynamicUpdateSlice>(inst)) {
    return false;
  }
  // Check that the first operand of DUS is a:
  // 1. GTE of loop input param in case of the first slice of data
  // 2. DUS in case of the second slice of data from unrolled loop.
  HloInstruction* dus_destination = inst->mutable_operand(0);
  if (is_first_slice &&
      !Match(dus_destination, m::GetTupleElement(m::Parameter()))) {
    return false;
  }
  if (!is_first_slice && !Match(dus_destination, m::DynamicUpdateSlice())) {
    return false;
  }
  HloInstruction* dus_constant = nullptr;
  HloInstruction* dus_slice_index = nullptr;
  // Now we loop through all the index operands to find boundary and slice
  // index.
  for (int64_t i = 2; i < inst->operand_count(); i++) {
    if (!Match(inst->mutable_operand(i), m::Constant(&dus_constant)) &&
        !Match(
            inst->mutable_operand(i),
            m::Reshape(m::DynamicSlice(&dus_slice_index, m::Op(), m::Op())))) {
      return false;
    }
  }
  if (!dus_constant || !dus_slice_index) {
    return false;
  }
  *dus_boundary_constant = dus_constant;
  *slice_indices = dus_slice_index;
  return true;
}

absl::Status ProcessWindowedEinsumLoopForActivationCaching(
    WindowedEinsumHandler::WindowedEinsumAgLoops& ag_loop) {
  HloInstruction* loop = ag_loop.loop;
  // Transform the while body to cache the allgathered result in the
  // output buffer to be consumed by the dot
  HloComputation* while_body = loop->while_body();
  HloInstruction* input_gte;
  for (HloInstruction* gte : while_body->parameter_instruction(0)->users()) {
    if (gte->tuple_index() == 0) {
      input_gte = gte;
    }
  }
  // Get the output operand of the full buffer.
  HloInstruction* root = while_body->root_instruction();
  // Change loop body to include the new input and output element.
  HloInstruction* input_tuple = while_body->parameter_instruction(0);
  const Shape& input_shape = input_tuple->shape();
  // The full buffer that we will use to cache the accumulated activation
  // is the last operand in the output tuple.
  int64_t full_cache_buffer_index = GetAgActivationCacheIndex(loop);
  HloInstruction* full_buffer_output_gte =
      while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetTupleElementShape(input_shape, full_cache_buffer_index),
          input_tuple, full_cache_buffer_index));

  HloInstruction* new_full_buffer_output = nullptr;
  // Find the DUS in the loop body and re-use the slice indices
  // This should just be a constant(0)
  HloInstruction* dus_boundary_constant;
  // The slice we need this time is the output of the first
  // collective-permute
  HloInstruction* first_cp_output;
  for (HloInstruction* gte_user : input_gte->users()) {
    if (HloPredicateIsOp<HloOpcode::kCollectivePermute>(gte_user)) {
      first_cp_output = gte_user;
      break;
    }
  }

  for (HloInstruction* inst : while_body->MakeInstructionPostOrder()) {
    HloInstruction* slice_indices;
    // If we have a DUS(PARAM,DS) pattern, we need to update the output
    // buffer with the first slice.
    if (FindDusSliceForCachedActivation(inst, &dus_boundary_constant,
                                        &slice_indices,
                                        /*is_first_slice=*/true)) {
      slice_indices = while_body->AddInstruction(HloInstruction::CreateReshape(
          dus_boundary_constant->shape(), slice_indices));
      VLOG(5) << "Created slice op for first slice: "
              << slice_indices->ToString();
      full_buffer_output_gte =
          while_body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              full_buffer_output_gte->shape(), full_buffer_output_gte,
              input_gte,
              {dus_boundary_constant, slice_indices, dus_boundary_constant}));
    }
    // If we have a DUS(DUS,DS) pattern, then the einsum loop is
    // unrolled, we need to update the output buffer again with the
    // second slice. Since the second slice will have different indices,
    // we need to re-capture slice_indices.
    if (FindDusSliceForCachedActivation(inst, &dus_boundary_constant,
                                        &slice_indices,
                                        /*is_first_slice=*/false)) {
      slice_indices = while_body->AddInstruction(HloInstruction::CreateReshape(
          dus_boundary_constant->shape(), slice_indices));
      VLOG(5) << "Created slice op for second slice: "
              << slice_indices->ToString();
      new_full_buffer_output =
          while_body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              full_buffer_output_gte->shape(), full_buffer_output_gte,
              first_cp_output,
              {dus_boundary_constant, slice_indices, dus_boundary_constant}));
    }

    // If we have a Dot(DS(parameter_index1)), then operands are sharded along
    // the contracting dim. Slice indices will be the contracting dim's slices.
    HloInstruction* slice_index;
    HloInstruction* ds_index_constant;
    HloInstruction* remainder;
    HloInstruction* ds_param;
    // There will be 2 dynamic-slices for unrolled loops, match for each one to
    // get the slice index which will be used to write the corresponding
    // received shard into cached activation buffer. For unrolled loops, we need
    // to write to the final buffer twice per iteration, so we need to match for
    // the correct slice index based on each DS.
    if (Match(inst, m::Dot(m::Op(), m::DynamicSlice(&ds_param))) &&
        Match(ds_param->operand(0), m::GetTupleElement(m::Parameter(), 1))) {
      for (int64_t ds_op_i = 1; ds_op_i < ds_param->operands().size();
           ds_op_i++) {
        if (!Match(
                ds_param->mutable_operand(ds_op_i),
                m::Reshape(&slice_index, m::DynamicSlice(m::Constant(),
                                                         m::Op(&remainder)))) &&
            !Match(ds_param->mutable_operand(ds_op_i),
                   m::Constant(&ds_index_constant))) {
          return absl::OkStatus();
        }
      }
      // First DS has slice index calculated based on loop iterator
      // Remainder(add(gte, partition_id))
      if (Match(remainder,
                m::Remainder(m::Add(m::GetTupleElement(), m::Op()), m::Op()))) {
        full_buffer_output_gte =
            while_body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
                full_buffer_output_gte->shape(), full_buffer_output_gte,
                input_gte,
                {ds_index_constant, ds_index_constant, slice_index}));
      }
      // Second DS has slice index calculated based on loop iterator+1 hence
      // Remainder(add(add(gte, 1), partition_id))
      if (Match(remainder,
                m::Remainder(
                    m::Add(m::Add(m::GetTupleElement(), m::Op()), m::Op()),
                    m::Op()))) {
        new_full_buffer_output =
            while_body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
                full_buffer_output_gte->shape(), full_buffer_output_gte,
                first_cp_output,
                {ds_index_constant, ds_index_constant, slice_index}));
      }
    }
  }
  std::vector<HloInstruction*> original_operands(root->operands().begin(),
                                                 root->operands().end());
  original_operands.push_back(new_full_buffer_output);
  HloInstruction* new_output_tuple = while_body->AddInstruction(
      HloInstruction::CreateTuple(original_operands));
  TF_RETURN_IF_ERROR(
      while_body->ReplaceInstructionWithDifferentShape(root, new_output_tuple));

  return absl::OkStatus();
}

bool HasReplicaGroups(const HloInstruction* inst) {
  return inst->replica_groups().size() > 0;
}

bool ShouldAddToChain(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kTranspose:
    case HloOpcode::kReshape:
    case HloOpcode::kCopy:
      return inst->user_count() == 1;
    default:
      return false;
  }
}

HloComputation* MakeSumComputation(PrimitiveType type, HloModule* module) {
  HloComputation::Builder sum_b("add");
  auto x = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(type, {}), "x"));
  auto y = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(type, {}), "y"));
  sum_b.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(type, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction = module->AddEmbeddedComputation(sum_b.Build());
  return reduction;
}

// Transform partial accumulations into a reduction on a contiguous buffer.
// Partial accumulations will impact the overlap between dots because the
// dot+add pattern will be fused into a single gemm later in gemm rewriter
// which adds data dependencies between gemms. Instead we write all
// intermediate results into a larger buffer and perform a one-shot reduction.
// The high-level transformation is:
//
// 'prev_res' is previously partially accumulated result.
//
// shape(x,y) prev_res   shape(x,y) dot0
//          \            /
//           \         /
//     shape(x,y) add0    shape(x,y) dot1
//             \                /
//              \              /
//             shape(x,y) add1
//                     |
//        shape(x,y) loop output
//
// transformed into:
// shape(x,y) prev_res         shape(x,y) dot0    shape(x,y) dot1
//    \                        /                   /
//     \                      /                   /
//     shape(n,x,y) concatenate on first axis, n is the number of partitions
//                        |
//             shape(n,x,y) loop output
//                        |
//             shape(x,y) reduction on first axis
//
// The final reduction is pulled outside of the loop to overlap with other
// collectives.
absl::Status MoveAccumulationOutsideLoop(
    std::vector<HloInstruction*>& partial_accumulations,
    HloComputation* while_body, HloInstruction* loop) {
  // The input of the while loop will be modified and must have no other users.
  if (!loop || loop->operand(0)->user_count() != 1) {
    return absl::OkStatus();
  }

  std::vector<HloInstruction*> partials_to_concat;

  // We reshape it to a N+1 dimensioned tensor with left-most dim being 1.
  Shape shape = partial_accumulations[0]->shape();
  shape = ShapeUtil::PrependMajorDimension(1, shape);

  for (auto& inst : partial_accumulations) {
    HloInstruction* reshaped_partial =
        while_body->AddInstruction(HloInstruction::CreateReshape(shape, inst));
    partials_to_concat.push_back(reshaped_partial);
  }
  Shape concat_shape = partial_accumulations[0]->shape();
  concat_shape = ShapeUtil::PrependMajorDimension(partial_accumulations.size(),
                                                  concat_shape);

  HloInstruction* concat = while_body->AddInstruction(
      HloInstruction::CreateConcatenate(concat_shape, partials_to_concat, 0));

  HloComputation* comp = loop->parent();
  HloInstruction* windowed_lhs = loop->mutable_operand(0)->mutable_operand(0);
  // Add a broadcasted zero of the same type as windowed_lhs. This holds all
  // the partial accumulations and will be fed to a global reduction after
  // this windowed einsum loop. We move the reduction outside of the loop so
  // it can be fused or overlap with other instructions in the main
  // computation.
  Literal zero_literal =
      LiteralUtil::Zero(windowed_lhs->shape().element_type());
  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(std::move(zero_literal)));
  Shape zero_bcast_shape = ShapeUtil::ChangeElementType(
      concat_shape, windowed_lhs->shape().element_type());
  HloInstruction* zero_bcast = MakeBroadcastHlo(zero, {}, zero_bcast_shape);
  loop->mutable_operand(0)->AppendOperand(zero_bcast);
  ShapeUtil::AppendShapeToTuple(zero_bcast->shape(),
                                loop->mutable_operand(0)->mutable_shape());

  // Update the parameter tuples of while's body and condition
  // computations.
  for (HloComputation* while_comp : {while_body, loop->while_condition()}) {
    while_comp->ReplaceParameter(
        0, HloInstruction::CreateParameter(
               0, loop->mutable_operand(0)->shape(),
               while_comp->parameter_instruction(0)->name()));
  }
  HloInstruction* root = while_body->root_instruction();
  std::vector<HloInstruction*> original_operands(root->operands().begin(),
                                                 root->operands().end());
  original_operands.push_back(concat);
  HloInstruction* new_output_tuple = while_body->AddInstruction(
      HloInstruction::CreateTuple(original_operands));
  TF_RETURN_IF_ERROR(
      while_body->ReplaceInstructionWithDifferentShape(root, new_output_tuple));

  // Update the shape of the while loop instruction.
  *loop->mutable_shape() = loop->operand(0)->shape();

  // The final reduction
  HloInstruction* concat_result_gte =
      comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          loop, (loop->operand(0)->shape().tuple_shapes().size() - 1)));
  HloInstruction* reduced_result =
      comp->AddInstruction(HloInstruction::CreateReduce(
          partial_accumulations[0]->shape(), concat_result_gte, zero, {0},
          MakeSumComputation(shape.element_type(), loop->GetModule())));

  // Replace the original output if present.
  HloInstruction* original_output_gte;
  auto it = absl::c_find_if(loop->users(), [&](HloInstruction* instr) {
    // Index of the original output. It's fixed to be the third element in the
    // tuple.
    return instr->tuple_index() == 2;
  });
  if (it != loop->users().end()) {
    original_output_gte = *it;
    TF_RETURN_IF_ERROR(original_output_gte->ReplaceAllUsesWith(reduced_result));
  }
  return absl::OkStatus();
}
absl::Status PostProcessUnrolledLoop(HloInstruction* loop, int64_t stream_id) {
  HloComputation* while_body = loop->while_body();
  // This is to set force delay for the first collective permute so it can
  // be scheduled always at the top of computation. The GTE index it's consuming
  // is 2 for RS loop; 0 for AG loop.
  int64_t force_delay_cp_gte_index =
      while_body->name().find(
          WindowedEinsumHandler::kWindowedEinsumRsLoopName) == 0
          ? 2
          : 0;
  std::vector<HloInstruction*> partial_accumulations;
  for (HloInstruction* inst : while_body->MakeInstructionPostOrder()) {
    HloInstruction* matched_cp;
    if (Match(inst,
              m::CollectivePermute(
                  &matched_cp, m::GetTupleElement(m::Parameter(),
                                                  force_delay_cp_gte_index)))) {
      TF_RETURN_IF_ERROR(
          SetForceDelayForInstruction(matched_cp, /*force_delay=*/true));
    }

    if (HloPredicateIsOp<HloOpcode::kDot>(inst)) {
      // Dispatch the dot to additional compute stream.
      TF_RETURN_IF_ERROR(UpdateDotAndConsumerConfig(inst, stream_id));
      ++stream_id;
    }
    // If dot's result is accumulated, this means we found a loop with
    // contracting dim sharded.
    HloInstruction* partial_dot;
    if (Match(inst, m::AddAnyOrder(m::Op(),
                                   m::Dot(&partial_dot, m::Op(), m::Op())))) {
      partial_accumulations.push_back(partial_dot);
    }
  }
  if (partial_accumulations.size() > 0 &&
      while_body->name().find(
          WindowedEinsumHandler::kWindowedEinsumAgLoopName) !=
          std::string::npos) {
    TF_RETURN_IF_ERROR(
        MoveAccumulationOutsideLoop(partial_accumulations, while_body, loop));
  }
  return absl::OkStatus();
}

struct MatchedGemmA2aResult {
  HloInstruction* producer_gemm;
  HloInstruction* lhs;
  HloInstruction* rhs;
  HloInstruction* a2a_replacement = nullptr;
  bool matched = false;
};

class WindowedEinsumVisitor : public DfsHloRewriteVisitor {
 public:
  explicit WindowedEinsumVisitor(
      std::vector<WindowedEinsumHandler::WindowedEinsumAgLoops>& all_ag_loops)
      : all_ag_loops_(all_ag_loops) {}
  absl::StatusOr<bool> MatchA2aGemmWithIntermediateReshapes(
      HloInstruction* dot, HloInstruction** lhs, HloInstruction** rhs) {
    if (Match(dot, m::Dot(m::AllToAll(lhs).WithOneUse().WithPredicate(
                              HasReplicaGroups),
                          m::Op(rhs))) &&
        !DynCast<HloAllToAllInstruction>((*lhs))->constrain_layout() &&
        !(*lhs)->shape().IsTuple()) {
      return true;
    }
    std::vector<HloInstruction*> allowed_intermediate_ops(
        {dot->mutable_operand(0)});

    HloAllToAllInstruction* matched_a2a = nullptr;
    // We keep pushing until an unmet condition or we have found the a2a.
    while (true) {
      HloInstruction* curr = allowed_intermediate_ops.back();
      if (ShouldAddToChain(curr)) {
        allowed_intermediate_ops.insert(allowed_intermediate_ops.end(),
                                        std::begin(curr->operands()),
                                        std::end(curr->operands()));
      } else if (HloPredicateIsOp<HloOpcode::kAllToAll>(curr) &&
                 curr->user_count() == 1) {
        matched_a2a = DynCast<HloAllToAllInstruction>(curr);
        allowed_intermediate_ops.pop_back();
        break;
      } else {
        return false;
      }
    }
    CHECK(matched_a2a != nullptr);
    if (matched_a2a->constrain_layout() || matched_a2a->shape().IsTuple() ||
        !HasReplicaGroups(matched_a2a) || !matched_a2a->split_dimension()) {
      return false;
    }
    // We need to create a new a2a that's a direct producer of the dot and
    // replace it with the original a2a. A new reshape will be added to the
    // orginal a2a's input. We first need to determine the new split dimension
    // after all the reshape ops.
    int64_t split_dimension = *matched_a2a->split_dimension();
    for (int64_t i = allowed_intermediate_ops.size() - 1; i >= 0; i--) {
      HloInstruction* current_op = allowed_intermediate_ops[i];
      if (HloPredicateIsOp<HloOpcode::kReshape>(current_op)) {
        std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
            ShapeUtil::DimensionsUnmodifiedByReshape(
                current_op->operand(0)->shape(), current_op->shape());
        auto it = absl::c_find_if(
            unmodified_dims,
            [&split_dimension](std::pair<int64_t, int64_t>& dim_pair) {
              return dim_pair.first == split_dimension;
            });
        // Split dimension of a2a has been modified, we cannot deduce the new
        // split dim easily, so skip decomposition.
        if (it == unmodified_dims.end()) {
          VLOG(5) << "Split dimension of: " << matched_a2a->ToShortString()
                  << " has been modified by reshapes. Skip process it for "
                     "decomposition.";
          return false;
        }
        // Assign the new split dim.
        split_dimension = it->second;
      } else if (HloPredicateIsOp<HloOpcode::kTranspose>(current_op)) {
        const auto& transpose_dims = current_op->dimensions();
        for (int64_t j = 0; j < transpose_dims.size(); j++) {
          if ((int64_t)transpose_dims[j] == split_dimension) {
            split_dimension = j;
            break;
          }
        }
      }
    }
    TF_RETURN_IF_ERROR(allowed_intermediate_ops.back()->ReplaceOperandWith(
        0, matched_a2a->mutable_operand(0)));
    HloInstruction* new_a2a =
        matched_a2a->parent()->AddInstruction(HloInstruction::CreateAllToAll(
            allowed_intermediate_ops.front()->shape(),
            {allowed_intermediate_ops.front()}, matched_a2a->replica_groups(),
            false, hlo_query::NextChannelId(*matched_a2a->GetModule()),
            split_dimension));

    TF_RETURN_IF_ERROR(dot->ReplaceOperandWith(0, new_a2a));
    TF_RETURN_IF_ERROR(
        matched_a2a->parent()->RemoveInstructionAndUnusedOperands(matched_a2a));
    MarkAsChanged();
    *lhs = new_a2a;
    *rhs = dot->mutable_operand(1);
    return true;
  }

  absl::Status HandleDot(HloInstruction* dot) override {
    CHECK_EQ(dot->opcode(), HloOpcode::kDot);
    HloComputation* comp = dot->parent();
    // Rewrites an allgather-dot pattern that shares the same operand with a
    // windowed einsum loop to consume the output of the loop and remove the
    // all-gather. Now that we have processed all loops, we can check if there
    // are any allgather-dot pattern that we can optimize. We'd want to
    // transform:
    //                       input
    //                       /    |
    //               dequantize   |
    //               (optional)   |
    //                   /        |
    //                 AG     windowed loop
    //                 /
    //                /
    //              dot
    // to:
    //                     input
    //                       |
    //                       |
    //                  windowed loop
    //                       |
    //                   dequantize
    //                     (FP8)
    //                       |
    //                      dot
    // The windowed einsum loop will also be rewritten to output the full input
    // to be consumed by the dot. This is advantageous since the chained dot can
    // fully utilize all the resources on the GPU while comm is hidden by the
    // first collective matmul loop. When the data type is FP8, input is
    // dequantized, i.e. type converted and scaled, ahead of the all-gather. The
    // dequantization is moved in WindowedEinsumVisitor between the windowed
    // loop and the dot.
    for (WindowedEinsumHandler::WindowedEinsumAgLoops& ag_loop :
         all_ag_loops_) {
      HloComputation* comp = dot->parent();
      HloInstruction* loop = ag_loop.loop;

      HloInstruction* windowed_lhs =
          loop->mutable_operand(0)->mutable_operand(0);

      // In the FP8 case, the all-gather operates on the dequantized
      // windowed_lhs. The dequantization is shifted to the output of the while
      // loop below.
      HloInstruction *all_gather, *binary, *scale = nullptr;
      auto all_gather_optionally_dequantized = m::AnyOf<HloInstruction>(
          m::AllGather(&all_gather,
                       m::Divide(&binary, m::Convert(m::Op().Is(windowed_lhs)),
                                 m::Broadcast(m::Op(&scale)))),
          m::AllGather(
              &all_gather,
              m::MultiplyAnyOrder(&binary, m::Convert(m::Op().Is(windowed_lhs)),
                                  m::Broadcast(m::Op(&scale)))),
          m::AllGather(&all_gather, m::Op().Is(windowed_lhs)));

      if (!Match(dot, m::Dot(all_gather_optionally_dequantized, m::Op())) &&
          !Match(dot, m::Dot(m::Op(), all_gather_optionally_dequantized))) {
        continue;
      }

      if (scale) {
        // When the loop contains an FP8 GEMM, a scalar scaling factor must be
        // captured.
        if (!ShapeUtil::IsScalar(scale->shape())) {
          continue;
        }

        // The element type of windowed_lhs must be a supported FP8 type.
        if (windowed_lhs->shape().element_type() != F8E4M3FN &&
            windowed_lhs->shape().element_type() != F8E5M2) {
          continue;
        }

        // The scaling multiplication or division must be in BF16, FP16 or FP32.
        if (binary->shape().element_type() != BF16 &&
            binary->shape().element_type() != F16 &&
            binary->shape().element_type() != F32) {
          continue;
        }
      }

      if (!ag_loop.consumed) {
        // Add a broadcasted zero of the same type as windowed_lhs. This caches
        // the accumulated activation inside the loop.
        Literal zero_literal =
            LiteralUtil::Zero(windowed_lhs->shape().element_type());
        HloInstruction* zero = comp->AddInstruction(
            HloInstruction::CreateConstant(std::move(zero_literal)));
        Shape zero_bcast_shape = ShapeUtil::ChangeElementType(
            all_gather->shape(), windowed_lhs->shape().element_type());
        HloInstruction* zero_bcast =
            MakeBroadcastHlo(zero, {}, zero_bcast_shape);
        loop->mutable_operand(0)->AppendOperand(zero_bcast);
        ShapeUtil::AppendShapeToTuple(
            zero_bcast->shape(), loop->mutable_operand(0)->mutable_shape());

        // Update the parameter tuples of while's body and condition
        // computations.
        for (HloComputation* while_comp :
             {loop->while_body(), loop->while_condition()}) {
          while_comp->ReplaceParameter(
              0, HloInstruction::CreateParameter(
                     0, loop->mutable_operand(0)->shape(),
                     while_comp->parameter_instruction(0)->name()));
        }

        // Update the shape of the while loop in the parent computation.
        *loop->mutable_shape() = loop->operand(0)->shape();

        VLOG(5) << "Found all-gather that shares the same operand with a "
                   "windowed einsum loop : "
                << loop->ToString();

        TF_RETURN_IF_ERROR(
            ProcessWindowedEinsumLoopForActivationCaching(ag_loop));
        ag_loop.consumed = true;
      }

      int64_t cache_output_index = dot->operand_index(all_gather);
      HloInstruction* new_gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              loop, GetAgActivationCacheIndex(loop)));

      HloInstruction* new_gte_scaled;

      if (scale) {
        // In the FP8 case, insert the dequantization of windowed_lhs between
        // the while loop and the dot.
        HloInstruction* new_convert =
            MakeConvertToHlo(new_gte, binary->shape().element_type());
        HloInstruction* bcast_scale =
            MakeBroadcastHlo(scale, {}, new_convert->shape());
        TF_ASSIGN_OR_RETURN(
            new_gte_scaled,
            MakeBinaryHlo(binary->opcode(), new_convert, bcast_scale));
      }

      TF_RETURN_IF_ERROR(dot->ReplaceOperandWith(
          cache_output_index, scale ? new_gte_scaled : new_gte));
      if (all_gather->user_count() == 0) {
        TF_RETURN_IF_ERROR(comp->RemoveInstruction(all_gather));
      }
    }
    // Rewrites an all-to-all+gemm into multiple independent partial a2a+gemms
    // to minimize communication overhead. To do this, the original input will
    // be sliced into replica_group size and perform all-to-all+gemm.
    if (!dot->GetModule()
             ->config()
             .debug_options()
             .xla_gpu_experimental_enable_alltoall_windowed_einsum()) {
      return absl::OkStatus();
    }
    HloInstruction* lhs;
    HloInstruction* rhs;
    std::vector<xla::ReplicaGroup> replica_groups;
    TF_ASSIGN_OR_RETURN(bool matched,
                        MatchA2aGemmWithIntermediateReshapes(dot, &lhs, &rhs));
    if (matched) {
      replica_groups = lhs->replica_groups();
      // We split the a2a+gemm along the contracting dimension into multiple
      // a2a+gemms and perform partial dots, partial results are added to the
      // final output buffer.
      int64_t group_size = replica_groups[0].replica_ids_size();
      if (absl::c_find_if(replica_groups, [&](ReplicaGroup& group) {
            return group.replica_ids_size() != group_size;
          }) != replica_groups.end()) {
        VLOG(5) << "All-to-all split groups don't have the same number of "
                   "replicas.";
        return absl::OkStatus();
      }

      // Get the dimension to slice for lhs and rhs, we slice on the contracting
      // dimensions to calculate partial results
      const DotDimensionNumbers& original_dot_dnums =
          dot->dot_dimension_numbers();
      const PrecisionConfig& original_precision = dot->precision_config();
      const auto& lhs_contracting_dims =
          dot->dot_dimension_numbers().lhs_contracting_dimensions();
      const auto& rhs_contracting_dims =
          dot->dot_dimension_numbers().rhs_contracting_dimensions();

      if (lhs_contracting_dims.size() != 1 ||
          rhs_contracting_dims.size() != 1) {
        VLOG(5) << "Contracting dimensions have multiple elements, all-to-all "
                   "sharding will be skipped.";
        return absl::OkStatus();
      }
      int64_t lhs_contracting_dim = lhs_contracting_dims[0];
      int64_t rhs_contracting_dim = rhs_contracting_dims[0];
      HloAllToAllInstruction* a2a = DynCast<HloAllToAllInstruction>(lhs);
      int64_t contracting_dim_value =
          rhs->shape().dimensions()[rhs_contracting_dim];

      // Each split is sliced out of the input buffer, we need to determine the
      // slice sizes and increments.
      std::vector<int64_t> lhs_slice_sizes(a2a->shape().dimensions().size(), 0);
      std::vector<int64_t> lhs_slice_increments(
          a2a->shape().dimensions().size(), 1);
      std::vector<int64_t> lhs_slice_max_range(
          a2a->shape().dimensions().begin(), a2a->shape().dimensions().end());

      std::vector<int64_t> rhs_slice_sizes(rhs->shape().dimensions().size(), 0);
      std::vector<int64_t> rhs_slice_increments(
          rhs->shape().dimensions().size(), 1);
      std::vector<int64_t> rhs_slice_max_range(
          rhs->shape().dimensions().begin(), rhs->shape().dimensions().end());

      // Create a zero-valued buffer to hold output.
      HloInstruction* output_buffer =
          comp->AddInstruction(HloInstruction::CreateBroadcast(
              dot->shape(),
              comp->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::Zero(dot->shape().element_type()))),
              {}));
      HloInstruction* a2a_operand = a2a->mutable_operand(0);
      if (contracting_dim_value % group_size) {
        VLOG(5) << absl::StrFormat(
            "Contracting dimension %d needs to be divisible by group_size %d",
            contracting_dim_value, group_size);
        return absl::OkStatus();
      }
      int64_t size_per_split = contracting_dim_value / group_size;

      // Each split is sliced out of the input buffer, we need to determine the
      // slice sizes and increments.
      lhs_slice_max_range[lhs_contracting_dim] = size_per_split;
      rhs_slice_max_range[rhs_contracting_dim] = size_per_split;

      Shape lhs_slice_shape = a2a->shape();
      Shape rhs_slice_shape = rhs->shape();

      lhs_slice_shape.set_dimensions(lhs_contracting_dim, size_per_split);
      rhs_slice_shape.set_dimensions(rhs_contracting_dim, size_per_split);

      HloInstruction* lhs_slice;
      HloInstruction* rhs_slice;

      HloInstruction* partial_result = output_buffer;

      Shape partial_all_to_all_shape = lhs_slice_shape;

      TF_ASSIGN_OR_RETURN(
          Shape partial_dot_shape,
          ShapeInference::InferDotOpShape(
              partial_all_to_all_shape, rhs_slice_shape, original_dot_dnums,
              /*preferred_element_type=*/std::nullopt));
      int64_t stream_id = hlo_query::NextChannelId(*a2a->GetModule());
      for (int64_t i = 0; i < group_size; ++i) {
        lhs_slice = comp->AddInstruction(HloInstruction::CreateSlice(
            lhs_slice_shape, a2a_operand, lhs_slice_sizes, lhs_slice_max_range,
            lhs_slice_increments));
        a2a->SetupDerivedInstruction(lhs_slice);
        lhs_slice_sizes[lhs_contracting_dim] =
            lhs_slice_max_range[lhs_contracting_dim];
        lhs_slice_max_range[lhs_contracting_dim] += size_per_split;

        rhs_slice = comp->AddInstruction(HloInstruction::CreateSlice(
            rhs_slice_shape, rhs, rhs_slice_sizes, rhs_slice_max_range,
            rhs_slice_increments));
        a2a->SetupDerivedInstruction(rhs_slice);
        rhs_slice_sizes[rhs_contracting_dim] =
            rhs_slice_max_range[rhs_contracting_dim];
        rhs_slice_max_range[rhs_contracting_dim] += size_per_split;

        HloInstruction* partial_all_to_all =
            comp->AddInstruction(HloInstruction::CreateAllToAll(
                partial_all_to_all_shape, {lhs_slice}, a2a->device_list(),
                false, hlo_query::NextChannelId(*a2a->GetModule()),
                a2a->split_dimension()));
        a2a->SetupDerivedInstruction(partial_all_to_all);

        HloInstruction* partial_dot =
            comp->AddInstruction(HloInstruction::CreateDot(
                partial_dot_shape, partial_all_to_all, rhs_slice,
                original_dot_dnums, original_precision));
        partial_result = comp->AddInstruction(
            HloInstruction::CreateBinary(partial_dot->shape(), HloOpcode::kAdd,
                                         partial_dot, partial_result));
        a2a->SetupDerivedInstruction(partial_result);
        TF_RETURN_IF_ERROR(
            UpdateDotAndConsumerConfig(partial_dot, stream_id++));
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(dot, partial_result));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<MatchedGemmA2aResult> MatchGemmA2aWithIntermediateReshapes(
      HloInstruction* inst) {
    MatchedGemmA2aResult result;
    HloAllToAllInstruction* a2a = DynCast<HloAllToAllInstruction>(inst);
    if (!HasReplicaGroups(a2a) || a2a->constrain_layout() ||
        a2a->shape().IsTuple()) {
      return result;
    }
    if (Match(a2a, m::AllToAll(m::Dot(&result.producer_gemm, m::Op(&result.lhs),
                                      m::Op(&result.rhs))
                                   .WithOneUse()))) {
      result.matched = true;
      return result;
    }
    std::vector<HloInstruction*> allowed_intermediate_ops(
        {a2a->mutable_operand(0)});

    HloInstruction* matched_dot = nullptr;
    // We keep pushing until an unmet condition or we have found the producer
    // dot.
    while (true) {
      HloInstruction* curr = allowed_intermediate_ops.back();
      if (ShouldAddToChain(curr)) {
        allowed_intermediate_ops.insert(allowed_intermediate_ops.end(),
                                        std::begin(curr->operands()),
                                        std::end(curr->operands()));
      } else if (HloPredicateIsOp<HloOpcode::kDot>(curr) &&
                 curr->user_count() == 1) {
        matched_dot = curr;
        allowed_intermediate_ops.pop_back();
        break;
      } else {
        return result;
      }
    }
    CHECK(matched_dot != nullptr);
    // We need to create a new a2a that's a direct consumer of the dot and
    // replace it with the original a2a. A new reshape will be added to the
    // orginal a2a's output. We first need to determine the new split dimension
    // after all the reshape ops.
    int64_t split_dimension = *a2a->split_dimension();
    for (int64_t i = 0; i < allowed_intermediate_ops.size(); i++) {
      HloInstruction* current_op = allowed_intermediate_ops[i];
      if (HloPredicateIsOp<HloOpcode::kReshape>(current_op)) {
        std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
            ShapeUtil::DimensionsUnmodifiedByReshape(
                current_op->operand(0)->shape(), current_op->shape());
        auto it = absl::c_find_if(
            unmodified_dims,
            [&split_dimension](std::pair<int64_t, int64_t>& dim_pair) {
              return dim_pair.second == split_dimension;
            });
        // Split dimension of a2a has been modified, we cannot deduce the new
        // split dim easily, so skip decomposition.
        if (it == unmodified_dims.end()) {
          VLOG(5) << "Split dimension of: " << a2a->ToShortString()
                  << " has been modified by reshapes. Skip process it for "
                     "decomposition.";
          return result;
        }
        // Assign the new split dim.
        split_dimension = it->first;
      } else if (HloPredicateIsOp<HloOpcode::kTranspose>(current_op)) {
        const auto& transpose_dims = current_op->dimensions();
        split_dimension = transpose_dims[split_dimension];
      }
    }
    result.a2a_replacement =
        matched_dot->parent()->AddInstruction(HloInstruction::CreateAllToAll(
            matched_dot->shape(), {matched_dot}, a2a->replica_groups(), false,
            hlo_query::NextChannelId(*matched_dot->GetModule()),
            split_dimension));
    TF_RETURN_IF_ERROR(allowed_intermediate_ops.back()->ReplaceOperandWith(
        0, result.a2a_replacement));
    inst->SetupDerivedInstruction(result.a2a_replacement);

    TF_RETURN_IF_ERROR(
        ReplaceInstruction(inst, allowed_intermediate_ops.front()));
    result.lhs = matched_dot->mutable_operand(0);
    result.rhs = matched_dot->mutable_operand(1);
    result.producer_gemm = matched_dot;
    result.matched = true;
    return result;
  }

  // Rewrites an gemm+all-to-all into multiple independent partial gemm+a2a's
  // to minimize communication overhead. To do this, the original input will be
  // sliced into replica_group size and perform gemm+all-to-all.
  absl::Status HandleAllToAll(HloInstruction* inst) override {
    CHECK_EQ(inst->opcode(), HloOpcode::kAllToAll);
    HloComputation* comp = inst->parent();
    if (!inst->GetModule()
             ->config()
             .debug_options()
             .xla_gpu_experimental_enable_alltoall_windowed_einsum()) {
      return absl::OkStatus();
    }
    // Rewrites a gemm+alltoall into multiple independent partial gemm+a2as
    // to minimize communication overhead.
    std::vector<xla::ReplicaGroup> replica_groups;
    TF_ASSIGN_OR_RETURN(MatchedGemmA2aResult matched_result,
                        MatchGemmA2aWithIntermediateReshapes(inst));
    if (matched_result.matched) {
      HloInstruction* a2a = inst;
      if (matched_result.a2a_replacement) {
        a2a = matched_result.a2a_replacement;
      }
      replica_groups = a2a->replica_groups();
      // Similar to a2a+gemm, we split along contracting dimensions
      // and aggregate result at each step.
      int64_t group_size = replica_groups[0].replica_ids_size();

      if (absl::c_find_if(replica_groups, [&](ReplicaGroup& group) {
            return group.replica_ids_size() != group_size;
          }) != replica_groups.end()) {
        VLOG(5) << "All-to-all split groups don't have the same number of "
                   "replicas.";
        return absl::OkStatus();
      }

      // Get the dimension to slice for lhs and rhs, we slice on the contracting
      // dimensions to calculate partial results
      const DotDimensionNumbers& original_dot_dnums =
          matched_result.producer_gemm->dot_dimension_numbers();
      const PrecisionConfig& original_precision =
          matched_result.producer_gemm->precision_config();
      const auto& lhs_contracting_dims =
          matched_result.producer_gemm->dot_dimension_numbers()
              .lhs_contracting_dimensions();
      const auto& rhs_contracting_dims =
          matched_result.producer_gemm->dot_dimension_numbers()
              .rhs_contracting_dimensions();

      if (lhs_contracting_dims.size() != 1 ||
          rhs_contracting_dims.size() != 1) {
        VLOG(5) << "Contracting dimensions have multiple elements, all-to-all "
                   "sharding will be skipped.";
        return absl::OkStatus();
      }
      int64_t lhs_contracting_dim = lhs_contracting_dims[0];
      int64_t rhs_contracting_dim = rhs_contracting_dims[0];
      HloAllToAllInstruction* all_to_all = DynCast<HloAllToAllInstruction>(a2a);
      int64_t contracting_dim_value =
          matched_result.rhs->shape().dimensions()[rhs_contracting_dim];
      // Each split is sliced out of the input buffer, we need to determine the
      // slice sizes and increments.
      std::vector<int64_t> lhs_slice_sizes(
          matched_result.lhs->shape().dimensions().size(), 0);
      std::vector<int64_t> lhs_slice_increments(
          matched_result.lhs->shape().dimensions().size(), 1);
      std::vector<int64_t> lhs_slice_max_range(
          matched_result.lhs->shape().dimensions().begin(),
          matched_result.lhs->shape().dimensions().end());

      std::vector<int64_t> rhs_slice_sizes(
          matched_result.rhs->shape().dimensions().size(), 0);
      std::vector<int64_t> rhs_slice_increments(
          matched_result.rhs->shape().dimensions().size(), 1);
      std::vector<int64_t> rhs_slice_max_range(
          matched_result.rhs->shape().dimensions().begin(),
          matched_result.rhs->shape().dimensions().end());

      // Create a zero-valued buffer to hold output.
      HloInstruction* output_buffer =
          comp->AddInstruction(HloInstruction::CreateBroadcast(
              all_to_all->shape(),
              comp->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::Zero(all_to_all->shape().element_type()))),
              {}));
      if (contracting_dim_value % group_size) {
        VLOG(5) << absl::StrFormat(
            "Contracting dimension %d needs to be divisible by group_size %d",
            contracting_dim_value, group_size);
        return absl::OkStatus();
      }

      int64_t size_per_split = contracting_dim_value / group_size;
      // Each split is sliced out of the input buffer, we need to determine the
      // slice sizes and increments.
      lhs_slice_max_range[lhs_contracting_dim] = size_per_split;
      rhs_slice_max_range[rhs_contracting_dim] = size_per_split;

      Shape lhs_slice_shape = matched_result.lhs->shape();
      Shape rhs_slice_shape = matched_result.rhs->shape();

      lhs_slice_shape.set_dimensions(lhs_contracting_dim, size_per_split);
      rhs_slice_shape.set_dimensions(rhs_contracting_dim, size_per_split);

      HloInstruction* lhs_slice;
      HloInstruction* rhs_slice;

      HloInstruction* partial_result = output_buffer;
      Shape partial_all_to_all_shape = all_to_all->shape();

      TF_ASSIGN_OR_RETURN(
          Shape partial_dot_shape,
          ShapeInference::InferDotOpShape(
              lhs_slice_shape, rhs_slice_shape, original_dot_dnums,
              /*preferred_element_type=*/std::nullopt));
      int64_t stream_id = hlo_query::NextChannelId(*all_to_all->GetModule());
      for (int64_t i = 0; i < group_size; ++i) {
        lhs_slice = comp->AddInstruction(HloInstruction::CreateSlice(
            lhs_slice_shape, matched_result.lhs, lhs_slice_sizes,
            lhs_slice_max_range, lhs_slice_increments));
        all_to_all->SetupDerivedInstruction(lhs_slice);
        lhs_slice_sizes[lhs_contracting_dim] =
            lhs_slice_max_range[lhs_contracting_dim];
        lhs_slice_max_range[lhs_contracting_dim] += size_per_split;

        rhs_slice = comp->AddInstruction(HloInstruction::CreateSlice(
            rhs_slice_shape, matched_result.rhs, rhs_slice_sizes,
            rhs_slice_max_range, rhs_slice_increments));

        all_to_all->SetupDerivedInstruction(rhs_slice);
        rhs_slice_sizes[rhs_contracting_dim] =
            rhs_slice_max_range[rhs_contracting_dim];
        rhs_slice_max_range[rhs_contracting_dim] += size_per_split;

        HloInstruction* partial_dot = comp->AddInstruction(
            HloInstruction::CreateDot(partial_dot_shape, lhs_slice, rhs_slice,
                                      original_dot_dnums, original_precision));

        HloInstruction* partial_all_to_all =
            comp->AddInstruction(HloInstruction::CreateAllToAll(
                partial_all_to_all_shape, {partial_dot},
                all_to_all->device_list(), false,
                hlo_query::NextChannelId(*all_to_all->GetModule()),
                all_to_all->split_dimension()));
        all_to_all->SetupDerivedInstruction(partial_all_to_all);
        partial_result = comp->AddInstruction(HloInstruction::CreateBinary(
            partial_all_to_all_shape, HloOpcode::kAdd, partial_all_to_all,
            partial_result));
        all_to_all->SetupDerivedInstruction(partial_result);
        TF_RETURN_IF_ERROR(
            UpdateDotAndConsumerConfig(partial_dot, stream_id++));
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(all_to_all, partial_result));
    }

    return absl::OkStatus();
  }

 private:
  std::vector<WindowedEinsumHandler::WindowedEinsumAgLoops>& all_ag_loops_;
};

}  // namespace

absl::StatusOr<bool> WindowedEinsumHandler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      5, "WindowedEinsumHandler::Run(), before:\n" + module->ToString());
  bool changed = false;
  int64_t stream_id = hlo_query::NextChannelId(*module);
  std::vector<HloInstruction*> all_windowed_einsum_loops;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    // If we have a einsum loop with less than 1 dot, it's not
    // a loop of interest.
    if (NumberOfInstructionsInComp(comp, HloOpcode::kDot) <= 1) {
      continue;
    }
    if (comp->name().find(kWindowedEinsumRsLoopName) == 0 ||
        comp->name().find(kWindowedEinsumAgLoopName) == 0) {
      VLOG(5) << "Processing computation: " << comp->name();
      // If present, move the dequantization of FP8 operands of the dot into the
      // while loop to allow e.g. gemm_rewriter.cc to fuse the dequantization
      // and dot into an FP8 GEMM.
      auto maybe_while_op = comp->GetUniqueCaller(HloOpcode::kWhile);
      if (!maybe_while_op.has_value()) {
        return absl::InvalidArgumentError(
            "Expected computation to be a loop body.");
      }

      auto* while_op = *maybe_while_op;
      TF_ASSIGN_OR_RETURN(auto maybe_new_op, ShiftDequantizationF8(comp));
      if (maybe_new_op) {
        changed = true;
        while_op = maybe_new_op;
      }

      if (comp->name().find(kWindowedEinsumAgLoopName) == 0) {
        all_ag_loops_.push_back(WindowedEinsumAgLoops(while_op));
      }
      all_windowed_einsum_loops.push_back(while_op);
    }
  }

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    WindowedEinsumVisitor visitor(all_ag_loops_);
    TF_RETURN_IF_ERROR(comp->Accept(&visitor));
    changed |= visitor.changed();
  }

  if (!all_windowed_einsum_loops.empty()) {
    // This is to prepare the module for unrolling. WhileLoopUnroller
    // looks for the induction variable by matching specific patterns
    // which expect AlgebraicSimplifier and HloConstantFolding to be applied.
    // Since we get the loop directly from SPMD patitioner,
    // the induction variable pattern doesn't conform to what unroller
    // expects until the passes are applied.
    TF_ASSIGN_OR_RETURN(bool applied_algsimp,
                        AlgebraicSimplifier(AlgebraicSimplifierOptions())
                            .Run(module, execution_threads));
    changed |= applied_algsimp;
    TF_ASSIGN_OR_RETURN(bool applied_cf,
                        HloConstantFolding().Run(module, execution_threads));
    changed |= applied_cf;
  }
  for (HloInstruction* loop : all_windowed_einsum_loops) {
    VLOG(5) << "Processing " << loop->ToString() << " for unrolling.";
    std::string original_body_name = std::string(loop->while_body()->name());
    std::string original_cond_name =
        std::string(loop->while_condition()->name());

    // We fully unroll the loop here to maximize overlap.
    // Without unrolling, each iteration will end with a DUS and gemm.
    // The gemm will not overlap with anything which means wave quantization
    // overhead is fully exposed.
    // After unrolling, all gemms, DUSes, and collectives can overlap
    // with each other.
    // We also need to keep the unrolled instructions in an isolated computation
    // unit such as a trivial loop so instructions here won't be fused with
    // other instructions later to disrupt the gemm-gemm overlap.
    TF_ASSIGN_OR_RETURN(
        UnrollResult result,
        WhileLoopUnroller::UnrollAndReturnReplacement(
            loop, /*unroll_factor=*/-1, /*wrap_in_trivial_loop=*/true,
            /*force_unroll=*/false));

    if (result.unrolled) {
      result.new_while_op->while_body()->SetAndSanitizeName(
          absl::StrCat("unrolled_", original_body_name));
      result.new_while_op->while_condition()->SetAndSanitizeName(
          absl::StrCat("unrolled_", original_cond_name));
      // The loop is fully unrolled but has a trip count of 1
      // To prevent it from being inlined by while loop simplifier,
      // we add this attribute to it.
      result.new_while_op->set_frontend_attribute(
          "skip-simplify-while-loops_trip-count-one", "true");
      TF_RETURN_IF_ERROR(
          PostProcessUnrolledLoop(result.new_while_op, stream_id));
    }
    changed |= result.unrolled;
  }
  XLA_VLOG_LINES(5,
                 "WindowedEinsumHandler::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
