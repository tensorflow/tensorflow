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

#include "xla/service/gpu/gpu_windowed_einsum_handler.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
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
//   inputs --> dequant --> while loop {dynamic-slice/collective-permute/dot}
//
// into
//
//   inputs --> while loop {dequant --> dynamic-slice/collective-permute/dot}.
absl::Status ShiftDequantizationF8(const HloComputation* comp,
                                   const std::array<HloInstruction*, 2>& gte) {
  HloInstruction* while_instr = comp->WhileCallInstruction();
  if (!while_instr) {
    return absl::OkStatus();
  }

  // Identify the scalings and type conversions applied to the inputs of the
  // while loop.
  HloInstruction* param_tuple = while_instr->mutable_operand(0);
  std::array<HloInstruction*, 2> binaries, operands, scales;
  for (int k = 0; k < 2; ++k) {
    if (!Match(param_tuple->mutable_operand(k),
               m::AnyOf<HloInstruction>(
                   m::Divide(&binaries[k], m::Convert(m::Op(&operands[k])),
                             m::Broadcast(m::Op(&scales[k]))),
                   m::MultiplyAnyOrder(&binaries[k],
                                       m::Convert(m::Op(&operands[k])),
                                       m::Broadcast(m::Op(&scales[k])))))) {
      VLOG(5) << "Unable to identify FP8 dequantization pattern.";
      return absl::OkStatus();
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
    return absl::OkStatus();
  }

  // The dequantized types must be BF16, FP16 or FP32.
  for (int k = 0; k < 2; ++k) {
    if (binaries[k]->shape().element_type() != BF16 &&
        binaries[k]->shape().element_type() != F16 &&
        binaries[k]->shape().element_type() != F32) {
      VLOG(5) << "Unsupported types.";
      return absl::OkStatus();
    }
  }

  // The FP8 scaling operands must be scalars.
  if (!ShapeUtil::IsScalar(scales[0]->shape()) ||
      !ShapeUtil::IsScalar(scales[1]->shape())) {
    VLOG(5) << "Scaling factors must be scalars.";
    return absl::OkStatus();
  }

  // Identify the dot and collective-permute or dynamic-slice instructions in
  // the all-gather or reduce-scatter patterns in while's body.
  HloComputation* while_body = while_instr->while_body();
  HloComputation* while_condition = while_instr->while_condition();
  HloInstruction* while_root = while_body->root_instruction();
  std::array<HloInstruction*, 2> dots, dyn_slices{nullptr, nullptr},
      coll_perms{nullptr, nullptr};
  if (Match(
          while_root,
          m::Tuple(m::CollectivePermute(
                       &coll_perms[1], m::CollectivePermute(
                                           &coll_perms[0], m::Op().Is(gte[0]))),
                   m::Op().Is(gte[1]),
                   m::DynamicUpdateSlice(
                       m::DynamicUpdateSlice().WithOperand(
                           1, m::Dot(&dots[0], m::Op().Is(gte[0]),
                                     m::Op().Is(gte[1]))),
                       m::Dot(&dots[1], m::Op(), m::Op().Is(gte[1])), m::Op(),
                       m::Op(), m::Op()),
                   m::Op(), m::Op()))) {
    VLOG(5) << "Identified all-gather windowed einsum pattern.";
  } else if (Match(
                 while_root,
                 m::Tuple(m::Op().Is(gte[0]), m::Op().Is(gte[1]),
                          m::AddAnyOrder(
                              m::Dot(&dots[0], m::DynamicSlice(&dyn_slices[0]),
                                     m::Op().Is(gte[1])),
                              m::Op()),
                          m::CollectivePermute(m::AddAnyOrder(
                              m::Dot(&dots[1], m::DynamicSlice(&dyn_slices[1]),
                                     m::Op().Is(gte[1])),
                              m::Op())),
                          m::Op()))) {
    VLOG(5) << "Identified reduce-scatter windowed einsum pattern.";
  } else {
    VLOG(5) << "Unable to identify valid windowed einsum pattern.";
    return absl::OkStatus();
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
  // retrieving BF16/FP16/FP32 dot operands with dequantized get-tuple-element
  // instructions retrieving FP8 dot operands from the input tuple.
  HloInstruction* body_param = while_body->parameter_instruction(0);
  for (int k = 0; k < 2; ++k) {
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_f8,
                        MakeGetTupleElementHlo(body_param, k));

    if (while_root->operand(k) == gte[k]) {
      TF_RETURN_IF_ERROR(
          while_root->ReplaceOperandWithDifferentShape(k, operand_f8));
      ShapeUtil::UpdateTupleShape(operand_f8->shape(), k,
                                  while_root->mutable_shape());
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * operand_scale,
        MakeGetTupleElementHlo(
            body_param, body_param->shape().tuple_shapes_size() - 2 + k));

    // Also add the scaling factor to the output tuple of the while body.
    while_root->AppendOperand(operand_scale);
    ShapeUtil::AppendShapeToTuple(operand_scale->shape(),
                                  while_root->mutable_shape());

    // Dequantize the operands of the dots and dynamic-slices.
    HloInstruction* operand_f32 =
        MakeConvertToHlo(operand_f8, gte[k]->shape().element_type());
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
      if (dots[l]->operand(k) == gte[k]) {
        TF_RETURN_IF_ERROR(dots[l]->ReplaceOperandWith(k, operand_scaled));
      }
      if (dyn_slices[l] && dyn_slices[l]->operand(0) == gte[k]) {
        TF_RETURN_IF_ERROR(
            dyn_slices[l]->ReplaceOperandWith(0, operand_scaled));
      }
    }

    // In the all-gather case, coll_perms[0] has two users, coll_perms[1] and
    // dots[1], which prevents it from being exchanged with dequantization in
    // gemm_rewriter.cc. Instead, directly insert the dequantization before
    // dots[1] here.
    if (coll_perms[0] && coll_perms[0]->operand(0) == gte[k]) {
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
          MakeConvertToHlo(coll_perms_f8[0], gte[k]->shape().element_type());
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
  TF_RETURN_IF_ERROR(
      while_instr->ReplaceAllUsesWithDifferentShape(while_instr->AddInstruction(
          while_instr->CloneWithNewShape(while_root->shape()))));
  TF_RETURN_IF_ERROR(while_instr->parent()->RemoveInstruction(while_instr));

  if (coll_perms[0]) {
    TF_RETURN_IF_ERROR(while_body->RemoveInstruction(coll_perms[1]));
    TF_RETURN_IF_ERROR(while_body->RemoveInstruction(coll_perms[0]));
  }
  TF_RETURN_IF_ERROR(while_body->RemoveInstruction(gte[0]));
  TF_RETURN_IF_ERROR(while_body->RemoveInstruction(gte[1]));

  VLOG(5) << "FP8 dequantization moved into while loop.";
  return absl::OkStatus();
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
  updater_gpu_config->mutable_wait_on_operation_queues()->Add(stream_id);

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

absl::StatusOr<bool> HandleRsWindowedEinsumLoop(HloComputation* comp,
                                                int64_t stream_id) {
  bool changed = false;
  // If we have a einsum loop with only 1 dot, this means either
  // the loop is not unrolled or only 1 partition is available.
  // It's a no-op in either case.
  if (NumberOfInstructionsInComp(comp, HloOpcode::kDot) <= 1) {
    return changed;
  }
  for (auto inst : comp->MakeInstructionPostOrder()) {
    HloInstruction* matched_dot;
    std::array<HloInstruction*, 2> gte;
    // The dot we'd like to parallelize is consuming the second loop input
    // as RHS.
    if (Match(inst,
              m::Dot(&matched_dot,
                     m::DynamicSlice().WithOperand(
                         0, m::GetTupleElement(&gte[0], m::Parameter(), 0)),
                     m::GetTupleElement(&gte[1], m::Parameter(), 1)))) {
      // If present, move the dequantization of FP8 operands of the dot into the
      // while loop to allow gemm_rewriter.cc to rewrite into an FP8 Custom
      // Call.
      TF_RETURN_IF_ERROR(ShiftDequantizationF8(comp, gte));

      // Dispatch the dot to additional compute stream.
      TF_RETURN_IF_ERROR(UpdateDotAndConsumerConfig(matched_dot, stream_id));
      ++stream_id;
      changed = true;
    }

    // We need to enforce the first collective-permute to be always scheduled
    // at the beginning of the loop.
    HloInstruction* matched_cp;
    if (Match(inst, m::CollectivePermute(
                        &matched_cp, m::GetTupleElement(m::Parameter(), 2)))) {
      TF_RETURN_IF_ERROR(
          SetForceDelayForInstruction(matched_cp, /*force_delay=*/true));
      changed = true;
    }
  }
  return changed;
}

absl::StatusOr<bool> HandleAgWindowedEinsumLoop(HloComputation* comp,
                                                int64_t stream_id) {
  bool changed = false;
  // If we have a einsum loop with only 1 dot, this means either
  // the loop is not unrolled or only 1 partition is available.
  // It's a no-op in either case.
  if (NumberOfInstructionsInComp(comp, HloOpcode::kDot) <= 1) {
    return changed;
  }
  for (auto inst : comp->MakeInstructionPostOrder()) {
    HloInstruction* matched_dot;
    std::array<HloInstruction*, 2> gte;
    // The dot we'd like to parallelize is consuming the second loop input
    // as RHS and first loop input as LHS.
    if (Match(inst, m::Dot(&matched_dot,
                           m::GetTupleElement(&gte[0], m::Parameter(), 0),
                           m::GetTupleElement(&gte[1], m::Parameter(), 1)))) {
      // If present, move the dequantization of FP8 operands of the dot into the
      // while loop to allow gemm_rewriter.cc to rewrite into an FP8 Custom
      // Call.
      TF_RETURN_IF_ERROR(ShiftDequantizationF8(comp, gte));

      // Dispatch the dot to additional compute stream.
      TF_RETURN_IF_ERROR(UpdateDotAndConsumerConfig(matched_dot, stream_id));
      ++stream_id;
      TF_RETURN_IF_ERROR(
          SetForceDelayForInstruction(matched_dot, /*force_delay=*/true));
      changed = true;
    }

    // We need to enforce the first collective-permute to be always scheduled
    // at the beginning of the loop.
    HloInstruction* matched_cp;
    if (Match(inst, m::CollectivePermute(
                        &matched_cp, m::GetTupleElement(m::Parameter(), 0)))) {
      TF_RETURN_IF_ERROR(
          SetForceDelayForInstruction(matched_cp, /*force_delay=*/true));
      changed = true;
    }
  }
  return changed;
}

absl::Status ProcessWindowedEinsumLoopForActivationCaching(
    GpuWindowedEinsumHandler::WindowedEinsumAgLoops& ag_loop) {
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
  // The full buffer that we will use to cache the accumulated activation
  // is the 4th operand in the output tuple.
  int64_t full_cache_buffer_index = 3;
  HloInstruction* full_buffer_output_gte =
      root->mutable_operand(full_cache_buffer_index);
  HloInstruction* new_full_buffer_output;
  // Find the DUS in the loop body and re-use the slice indices
  // This should just be a constant(0)
  HloInstruction* dus_boundary_constant;
  for (HloInstruction* inst : while_body->MakeInstructionPostOrder()) {
    HloInstruction* slice_indices;
    // If we have a DUS(PARAM,DS) pattern, we need to update the output
    // buffer with the first slice.
    if (Match(inst,
              m::DynamicUpdateSlice(
                  m::GetTupleElement(m::Parameter()), m::Op(),
                  m::Constant(&dus_boundary_constant),
                  m::Reshape(m::DynamicSlice(&slice_indices, m::Op(), m::Op())),
                  m::Op()))) {
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
    if (Match(inst,
              m::DynamicUpdateSlice(
                  m::DynamicUpdateSlice(), m::Op(), m::Constant(),
                  m::Reshape(m::DynamicSlice(&slice_indices, m::Op(), m::Op())),
                  m::Op()))) {
      slice_indices = while_body->AddInstruction(HloInstruction::CreateReshape(
          dus_boundary_constant->shape(), slice_indices));
      VLOG(5) << "Created slice op for second slice: "
              << slice_indices->ToString();
      // The slice we need this time is the output of the first
      // collective-permute
      HloInstruction* cp_output;
      for (HloInstruction* gte_user : input_gte->users()) {
        if (gte_user->opcode() == HloOpcode::kCollectivePermute) {
          cp_output = gte_user;
          break;
        }
      }
      new_full_buffer_output =
          while_body->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              full_buffer_output_gte->shape(), full_buffer_output_gte,
              cp_output,
              {dus_boundary_constant, slice_indices, dus_boundary_constant}));
    }
  }
  TF_RETURN_IF_ERROR(root->ReplaceOperandWith(full_cache_buffer_index,
                                              new_full_buffer_output));
  return absl::OkStatus();
}

class WindowedEinsumVisitor : public DfsHloRewriteVisitor {
 public:
  explicit WindowedEinsumVisitor(
      std::vector<GpuWindowedEinsumHandler::WindowedEinsumAgLoops>&
          all_ag_loops)
      : all_ag_loops_(all_ag_loops) {}
  // Rewrites a allgather-dot pattern that shares the same operand
  // with a windowed einsum loop to consume the output of the loop
  // and remove the all-gather.
  absl::Status HandleDot(HloInstruction* dot) override {
    CHECK_EQ(dot->opcode(), HloOpcode::kDot);
    for (GpuWindowedEinsumHandler::WindowedEinsumAgLoops ag_loop :
         all_ag_loops_) {
      HloInstruction* loop = ag_loop.loop;
      HloInstruction* ag_operand = nullptr;

      if (Match(dot, m::Dot(m::AllGather(&ag_operand), m::Op())) ||
          Match(dot, m::Dot(m::Op(), m::AllGather(&ag_operand)))) {
        HloInstruction* windowed_lhs =
            loop->mutable_operand(0)->mutable_operand(0);
        HloInstruction* ag_with_shared_operand = nullptr;
        if (ag_operand && ag_operand->mutable_operand(0) == windowed_lhs) {
          ag_with_shared_operand = ag_operand;
        }

        if (!ag_with_shared_operand) {
          continue;
        }

        VLOG(5) << "Found all-gather that shares the same operand with a "
                   "windowed einsum loop : "
                << loop->ToString();
        int64_t cache_output_index = dot->operand_index(ag_with_shared_operand);
        HloComputation* comp = dot->parent();
        HloInstruction* new_gte = comp->AddInstruction(
            HloInstruction::CreateGetTupleElement(loop, 3));
        TF_RETURN_IF_ERROR(
            dot->ReplaceOperandWith(cache_output_index, new_gte));
        TF_RETURN_IF_ERROR(comp->RemoveInstruction(ag_with_shared_operand));
        if (!ag_loop.consumed) {
          TF_RETURN_IF_ERROR(
              ProcessWindowedEinsumLoopForActivationCaching(ag_loop));
          ag_loop.consumed = true;
        }
      }
    }
    return absl::OkStatus();
  }

 private:
  std::vector<GpuWindowedEinsumHandler::WindowedEinsumAgLoops>& all_ag_loops_;
};

}  // namespace

absl::StatusOr<bool> GpuWindowedEinsumHandler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      5, "GpuWindowedEinsumHandler::Run(), before:\n" + module->ToString());
  bool changed = false;
  int64_t stream_id = hlo_query::NextChannelId(*module);

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->name().find(kWindowedEinsumRsLoopName) == 0) {
      VLOG(5) << "Processing computation: " << comp->name();
      TF_ASSIGN_OR_RETURN(bool comp_result,
                          HandleRsWindowedEinsumLoop(comp, stream_id));
      changed = comp_result;
    } else if (comp->name().find(kWindowedEinsumAgLoopName) == 0) {
      VLOG(5) << "Processing computation: " << comp->name();
      TF_ASSIGN_OR_RETURN(bool comp_result,
                          HandleAgWindowedEinsumLoop(comp, stream_id));
      all_ag_loops_.push_back(
          WindowedEinsumAgLoops(comp->WhileCallInstruction()));
      changed = comp_result;
    }
  }
  // Now that we have processed all loops, we can check if there are any
  // allgather-dot pattern that we can optimize. We'd want to transform:
  //                       input
  //                       /    |
  //                      /     |
  //                     AG    windowed loop
  //                     /
  //                    /
  //                   dot
  // to:
  //                       input
  //                       |
  //                       |
  //                     windowed loop
  //                       |
  //                       |
  //                      dot
  // The windowed einsum loop will also be rewritten to output the full input to
  // be consumed by the dot.
  // This is advantageous since the chained dot can fully utilize all the
  // resources on the GPU while comm is hidden by the first collective matmul
  // loop.
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    WindowedEinsumVisitor visitor(all_ag_loops_);
    TF_RETURN_IF_ERROR(comp->Accept(&visitor));
    changed |= visitor.changed();
  }

  XLA_VLOG_LINES(
      5, "GpuWindowedEinsumHandler::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
