/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/thunk_emitter.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/runtime/all_reduce_thunk.h"
#include "xla/service/cpu/runtime/call_thunk.h"
#include "xla/service/cpu/runtime/conditional_thunk.h"
#include "xla/service/cpu/runtime/copy_thunk.h"
#include "xla/service/cpu/runtime/dot_thunk.h"
#include "xla/service/cpu/runtime/fft_thunk.h"
#include "xla/service/cpu/runtime/infeed_thunk.h"
#include "xla/service/cpu/runtime/kernel_thunk.h"
#include "xla/service/cpu/runtime/outfeed_thunk.h"
#include "xla/service/cpu/runtime/rng_state_thunk.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/while_thunk.h"
#include "xla/service/cpu/target_machine_features.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

ThunkEmitter::ThunkEmitter(IrEmitter2& ir_emitter,
                           const BufferAssignment& buffer_assignment,
                           const TargetMachineFeatures& target_machine_features,
                           const HloModuleConfig& hlo_module_config)
    : ir_emitter_(ir_emitter),
      buffer_assignment_(buffer_assignment),
      target_machine_features_(target_machine_features),
      hlo_module_config_(hlo_module_config) {}

static Thunk::Info ThunkInfo(const HloInstruction* instruction) {
  const HloModule* module = instruction->GetModule();
  return Thunk::Info{std::string(instruction->name()),
                     std::string(module->name()), module->unique_id()};
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitEntryComputation(
    const HloModule& module) {
  if (!module.has_schedule()) {
    return absl::InternalError("HLO module must be scheduled to emit thunks");
  }
  return EmitHloComputation(module.entry_computation());
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSlice(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return buffer_assignment_.GetUniqueSlice(instruction, index);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  ThunkSequence thunks;

  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation)) {
    return absl::InternalError(
        absl::StrCat("Computation ", computation->name(),
                     " must be scheduled to emit thunks"));
  }

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_ASSIGN_OR_RETURN(ThunkSequence instr_thunks, EmitHloInstruction(instr));
    thunks.Append(std::move(instr_thunks));
  }

  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    // Instructions that do not have a thunk implementation and instead fully
    // defined by the corresponding buffer assignment.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return ThunkSequence::Empty();

    // No-op operations that are used to provide more metadata about the HLO
    // dataflow graph.
    case HloOpcode::kAfterAll:             // Defines an execution order.
    case HloOpcode::kDomain:               // Defines an HLO domain.
    case HloOpcode::kOptimizationBarrier:  // Prevents moving ops past barrier.
      return ThunkSequence::Empty();

    // Allocations for constants owned by the executable, and resolved at run
    // time according to the buffer assignment (using allocation index). We do
    // not need to emit any thunks for constant instructions.
    case HloOpcode::kConstant:
      return ThunkSequence::Empty();

    // Call operations are simply converted to a ThunkSequence emitted from the
    // called computation and embedded into the "main" one.
    case HloOpcode::kCall:
      return EmitCallThunk(instruction);

    // Control flow thunks check predicates on the host and launch nested thunk
    // sequences for branches and loops.
    case HloOpcode::kConditional:
      return EmitConditionThunk(instruction);
    case HloOpcode::kWhile:
      return EmitWhileThunk(instruction);

    // Simple HLO instructions lowered to elemental host kernels (plain loops
    // behind the HostKernel API).
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBroadcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog1p:
    case HloOpcode::kLog:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kRemainder:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kXor:
      return EmitElementalKernelThunk(instruction);

    case HloOpcode::kAllReduce:
      return EmitAllReduceThunk(instruction);

    // TODO(ezhulenev): Port pad optimizations from IrEmitter.
    case HloOpcode::kPad:
      return EmitElementalKernelThunk(instruction);

    // TODO(ezhulenev): Implement slice operations as separate Thunks because
    // it's much easier to get peak performance from hand written code.
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    // TODO(ezhulenev): Port dynamic update slice optimizations from IrEmitter.
    case HloOpcode::kDynamicUpdateSlice:
      return EmitElementalKernelThunk(instruction);

    case HloOpcode::kConcatenate:
      return EmitConcatenateThunk(instruction);

    case HloOpcode::kFusion:
      return EmitFusionKernelThunk(instruction);

    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return EmitReductionKernelThunk(instruction);

    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateStateThunk(instruction);

    case HloOpcode::kInfeed:
      return EmitInfeedThunk(instruction);

    case HloOpcode::kOutfeed:
      return EmitOutfeedThunk(instruction);

    case HloOpcode::kCopy:
      return EmitCopyThunk(instruction);

    case HloOpcode::kDot:
      return EmitDotThunk(instruction);

    case HloOpcode::kFft:
      return EmitFftThunk(instruction);

    default:
      return absl::UnimplementedError(
          absl::StrCat("HLO opcode `", HloOpcodeString(instruction->opcode()),
                       "` is not supported by XLA:CPU ThunkEmitter"));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllReduceThunk(
    const HloInstruction* instruction) {
  auto* all_reduce = Cast<HloAllReduceInstruction>(instruction);

  // Check that we recognize the reduction computation attached to a collective.
  auto reduction_kind = MatchReductionComputation(all_reduce->to_apply());
  if (!reduction_kind.has_value()) {
    return Unimplemented("AllReduce for computation '%s' is not supported",
                         all_reduce->to_apply()->ToString());
  }

  // Collect buffer slices for all operands.
  std::vector<BufferAllocation::Slice> source_buffers;
  std::vector<Shape> source_shapes;

  for (const HloInstruction* operand : all_reduce->operands()) {
    TF_ASSIGN_OR_RETURN(source_buffers.emplace_back(),
                        GetAllocationSlice(operand));
    source_shapes.push_back(operand->shape());
  }

  // Collect buffer slices for all results.
  std::vector<BufferAllocation::Slice> destination_buffers;
  std::vector<Shape> destination_shapes;

  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(destination_buffers.emplace_back(),
                        GetAllocationSlice(instruction, indexed.index));
    destination_shapes.push_back(indexed.shape);
  }

  AllReduceThunk::OpParams op_params = {
      /*op_id=*/all_reduce->channel_id().has_value()
          ? all_reduce->channel_id().value()
          : all_reduce->GetModule()->unique_id(),
      /*has_channel_id=*/all_reduce->channel_id().has_value(),
      /*use_global_device_ids=*/all_reduce->use_global_device_ids(),
      /*replica_groups=*/all_reduce->replica_groups(),
  };

  bool single_replica = hlo_module_config_.replica_count() == 1 &&
                        hlo_module_config_.num_partitions() == 1;

  return ThunkSequence::Of<AllReduceThunk>(
      ThunkInfo(all_reduce), *reduction_kind, std::move(op_params),
      source_buffers, source_shapes, destination_buffers, destination_shapes,
      single_replica);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCallThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(
      ThunkSequence called_sequence,
      EmitHloComputation(instruction->called_computations().front()));
  return ThunkSequence::Of<CallThunk>(ThunkInfo(instruction),
                                      std::move(called_sequence));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConcatenateThunk(
    const HloInstruction* instruction) {
  // TODO(ezhulenev): Port optimized concat implementation from IrEmitter.
  return EmitElementalKernelThunk(instruction);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyThunk(
    const HloInstruction* instruction) {
  const HloInstruction* source = instruction->operand(0);
  TF_ASSIGN_OR_RETURN(auto source_buffer, GetAllocationSlice(source));
  TF_ASSIGN_OR_RETURN(auto destination_buffer, GetAllocationSlice(instruction));
  return ThunkSequence::Of<CopyThunk>(ThunkInfo(instruction), source_buffer,
                                      source->shape(), destination_buffer,
                                      instruction->shape());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitElementalKernelThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto kernel,
                      ir_emitter_.EmitElementalHostKernel(instruction));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(instruction));

  return ThunkSequence::Of<KernelThunk>(
      ThunkInfo(instruction), buffers.arguments, buffers.results, kernel.name,
      kernel.thread_dims, /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFusionKernelThunk(
    const HloInstruction* instruction) {
  auto* fusion = Cast<HloFusionInstruction>(instruction);
  TF_ASSIGN_OR_RETURN(auto kernel, ir_emitter_.EmitFusionHostKernel(fusion));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(instruction));

  return ThunkSequence::Of<KernelThunk>(
      ThunkInfo(instruction), buffers.arguments, buffers.results, kernel.name,
      kernel.thread_dims, /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReductionKernelThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto kernel,
                      ir_emitter_.EmitReductionHostKernel(instruction));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(instruction));

  return ThunkSequence::Of<KernelThunk>(
      ThunkInfo(instruction), buffers.arguments, buffers.results, kernel.name,
      kernel.thread_dims, /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngGetAndUpdateStateThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto state_buffer, GetAllocationSlice(instruction));
  auto* rng_state = Cast<HloRngGetAndUpdateStateInstruction>(instruction);
  return ThunkSequence::Of<RngGetAndUpdateStateThunk>(
      ThunkInfo(instruction), state_buffer, rng_state->delta());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitInfeedThunk(
    const HloInstruction* instruction) {
  auto* infeed = Cast<HloInfeedInstruction>(instruction);
  const Shape& infeed_shape = infeed->infeed_shape();

  // Collect buffer allocation slices corresponding to data buffers produced by
  // the infeed instruction;
  std::vector<InfeedThunk::InfeedBuffer> infeed_buffers;
  for (auto& infeed_leaf : ShapeUtil::GetLeafShapes(infeed_shape)) {
    infeed_leaf.index.push_front(0);  // prepend infeed tuple index

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice infeed_slice,
                        GetAllocationSlice(infeed, infeed_leaf.index));

    infeed_buffers.push_back(InfeedThunk::InfeedBuffer{
        infeed_slice,
        infeed_leaf.shape,
    });
  }

  return ThunkSequence::Of<InfeedThunk>(ThunkInfo(instruction), infeed_buffers);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOutfeedThunk(
    const HloInstruction* instruction) {
  auto* outfeed = Cast<HloOutfeedInstruction>(instruction);
  const Shape& outfeed_shape = outfeed->outfeed_shape();

  // Collect buffer allocation slices corresponding to data buffers fed into the
  // outfeed instruction as first operand.
  std::vector<OutfeedThunk::OutfeedBuffer> outfeed_buffers;
  for (auto& outfeed_leaf : ShapeUtil::GetLeafShapes(outfeed_shape)) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice outfeed_slice,
        GetAllocationSlice(outfeed->operand(0), outfeed_leaf.index));

    outfeed_buffers.push_back(OutfeedThunk::OutfeedBuffer{
        outfeed_slice,
        outfeed_leaf.shape,
    });
  }

  return ThunkSequence::Of<OutfeedThunk>(ThunkInfo(instruction),
                                         outfeed_buffers);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConditionThunk(
    const HloInstruction* instruction) {
  std::vector<ThunkSequence> branches;
  TF_ASSIGN_OR_RETURN(auto branch_index_buffer,
                      GetAllocationSlice(instruction->operand(0)));

  for (HloComputation* branch : instruction->branch_computations()) {
    TF_ASSIGN_OR_RETURN(branches.emplace_back(), EmitHloComputation(branch));
  }

  return ThunkSequence::Of<ConditionalThunk>(
      ThunkInfo(instruction), branch_index_buffer, std::move(branches));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitWhileThunk(
    const HloInstruction* instruction) {
  HloInstruction* cond = instruction->while_condition()->root_instruction();
  TF_ASSIGN_OR_RETURN(auto cond_buffer, GetAllocationSlice(cond));

  TF_ASSIGN_OR_RETURN(ThunkSequence cond_thunk,
                      EmitHloComputation(instruction->while_condition()));
  TF_ASSIGN_OR_RETURN(ThunkSequence body_thunk,
                      EmitHloComputation(instruction->while_body()));

  return ThunkSequence::Of<WhileThunk>(ThunkInfo(instruction), cond_buffer,
                                       std::move(cond_thunk),
                                       std::move(body_thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDotThunk(
    const HloInstruction* instruction) {
  const HloInstruction* lhs = instruction->operand(0);
  const HloInstruction* rhs = instruction->operand(1);

  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      *instruction, /*operands=*/{lhs, rhs},
      /*supported_types=*/
      {PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32, F64, C64, C128}));

  const DotDimensionNumbers& dnums = instruction->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1) {
    return Unimplemented(
        "Dot with multiple contracting dimensions is not implemented.");
  }

  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_config_, *instruction, target_machine_features_);

  switch (strategy) {
    // Emit host kernel implementing dot instruction.
    case DotImplementationStrategy::kNaiveLlvmIr:
    case DotImplementationStrategy::kTiledLlvmIrGemm:
    case DotImplementationStrategy::kTiledLlvmIrGemv: {
      TF_ASSIGN_OR_RETURN(auto kernel,
                          ir_emitter_.EmitDotHostKernel(instruction));
      TF_ASSIGN_OR_RETURN(auto buffers,
                          GetHostKernelAllocationSlices(instruction));

      return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction),
                                            buffers.arguments, buffers.results,
                                            kernel.name, kernel.thread_dims);
    }

    // Emit DotThunk implementing dot instruction as a library call.
    case DotImplementationStrategy::kEigen: {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_slice,
                          GetAllocationSlice(lhs));
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_slice,
                          GetAllocationSlice(rhs));
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out_slice,
                          GetAllocationSlice(instruction));

      return ThunkSequence::Of<DotThunk>(
          ThunkInfo(instruction), dnums, lhs_slice, lhs->shape(), rhs_slice,
          rhs->shape(), out_slice, instruction->shape());
    }
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFftThunk(
    const HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*instruction, /*operands=*/{instruction->operands()},
      /*supported_types=*/{F32, F64, C64, C128}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                      GetAllocationSlice(instruction->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                      GetAllocationSlice(instruction));
  return ThunkSequence::Of<FftThunk>(
      /*info=*/ThunkInfo(instruction),
      /*is_multi_thread_eigen=*/
      hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen(),
      /*fft_type=*/instruction->fft_type(),
      /*fft_length=*/instruction->fft_length(),
      /*input_buffer=*/arg_slice,
      /*input_shape=*/instruction->operand(0)->shape(),
      /*output_buffer=*/dest_slice,
      /*output_shape=*/instruction->shape());
}

absl::StatusOr<ThunkEmitter::HostKernelAllocationSlices>
ThunkEmitter::GetHostKernelAllocationSlices(const HloInstruction* instruction) {
  HostKernelAllocationSlices slices;

  auto add_buffers = [&](std::vector<BufferAllocation::Slice>& buffers,
                         const HloInstruction* instr) -> absl::Status {
    for (const auto& indexed : ShapeUtil::GetLeafShapes(instr->shape())) {
      TF_ASSIGN_OR_RETURN(buffers.emplace_back(),
                          GetAllocationSlice(instr, indexed.index));
    }
    return absl::OkStatus();
  };

  for (HloInstruction* operand : instruction->operands()) {
    TF_RETURN_IF_ERROR(add_buffers(slices.arguments, operand));
  }

  TF_RETURN_IF_ERROR(add_buffers(slices.results, instruction));

  return slices;
}

absl::Status ThunkEmitter::ElementTypesSameAndSupported(
    const HloInstruction& instruction,
    absl::Span<const HloInstruction* const> operands,
    absl::Span<const PrimitiveType> supported_types) {
  for (auto operand : operands) {
    TF_RET_CHECK(
        ShapeUtil::SameElementType(operands[0]->shape(), operand->shape()));
  }

  TF_RET_CHECK(!operands.empty());
  PrimitiveType primitive_type = operands[0]->shape().element_type();
  if (!absl::c_linear_search(supported_types, primitive_type)) {
    return Unimplemented("unsupported operand type %s in op %s",
                         PrimitiveType_Name(primitive_type),
                         HloOpcodeString(instruction.opcode()));
  }
  return absl::OkStatus();
}

}  // namespace xla::cpu
