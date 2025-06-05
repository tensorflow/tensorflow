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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "xla/backends/cpu/codegen/computation_kernel_emitter.h"
#include "xla/backends/cpu/codegen/dot/dot_kernel_emitter.h"
#include "xla/backends/cpu/codegen/elemental/concatenate_kernel_emitter.h"
#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/all_gather_thunk.h"
#include "xla/backends/cpu/runtime/all_reduce_thunk.h"
#include "xla/backends/cpu/runtime/all_to_all_thunk.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/collective_permute_thunk.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/convolution_thunk.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/dot_thunk.h"
#include "xla/backends/cpu/runtime/fft_thunk.h"
#include "xla/backends/cpu/runtime/infeed_thunk.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/logical_id_thunk.h"
#include "xla/backends/cpu/runtime/outfeed_thunk.h"
#include "xla/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "xla/backends/cpu/runtime/rng_state_thunk.h"
#include "xla/backends/cpu/runtime/sort_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/topk_thunk.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"
#include "xla/backends/cpu/xnn_emitter.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/llvm_kernel_definition.h"
#include "xla/codegen/mlir_kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/comparison_util.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout_util.h"
#include "xla/runtime/resource_use.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/dump.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

#if XLA_ONEDNN_USE_GRAPH_API
#include "xla/backends/cpu/onednn_emitter.h"
#include "xla/backends/cpu/onednn_fusion.h"
#include "xla/backends/cpu/runtime/onednn/onednn_fusion_thunk.h"
#endif  // XLA_ONEDNN_USE_GRAPH_API

namespace xla::cpu {

static FusionCompiler FusionCompilerFactory(const HloModule& hlo_module) {
  FusionCompiler::Options options{
      hlo_module.config().debug_options().xla_cpu_prefer_vector_width()};

  FusionCompiler::CompilationHooks hooks;
  if (DumpingEnabledForHloModule(hlo_module)) {
    auto callback_factory = [&hlo_module](std::string stage_name) {
      return [&hlo_module, stage_name](mlir::ModuleOp module) {
        std::optional<llvm::StringRef> name = module.getName();
        if (!name.has_value()) {
          return;
        }

        DumpToFileInDirOrStdout(
            hlo_module, "",
            absl::StrCat(absl::string_view(*name), "-", stage_name, ".mlir"),
            mlir::debugString(module));
      };
    };

    hooks.pre_optimization = callback_factory("pre-optimization");
    hooks.post_optimization = callback_factory("post-optimization");
    hooks.post_lowering = callback_factory("post-lowering");
  }

  return FusionCompiler(std::move(options), std::move(hooks));
}

ThunkEmitter::ThunkEmitter(IrEmitter2& ir_emitter,
                           const BufferAssignment& buffer_assignment,
                           const TargetMachineFeatures& target_machine_features,
                           const HloModule& hlo_module, const Options& options)
    : ir_emitter_(ir_emitter),
      buffer_assignment_(buffer_assignment),
      target_machine_features_(target_machine_features),
      hlo_module_config_(hlo_module.config()),
      options_(options),
      communicator_resource_(
          Resource::Create(Resource::kCollectiveCommunicator)),
      fusion_compiler_(FusionCompilerFactory(hlo_module)) {}

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
  tsl::profiler::TraceMe trace("ThunkEmitter::EmitEntryComputation");
  return EmitHloComputation(module.entry_computation());
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSlice(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return buffer_assignment_.GetUniqueSlice(instruction, index);
}

absl::StatusOr<std::shared_ptr<Resource>> ThunkEmitter::GetTokenResource(
    const HloInstruction* instruction, const ShapeIndex& index) {
  DCHECK(ShapeUtil::GetSubshape(instruction->shape(), index).IsToken());
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSlice(instruction, index));
  if (auto it = token_resources_.find(slice); it != token_resources_.end()) {
    return it->second;
  }
  return token_resources_[slice] = Resource::Create(Resource::kToken);
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
    case HloOpcode::kAddDependency:        // Defines an execution order.
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

    // Dimension size operations.
    case HloOpcode::kGetDimensionSize:
      return EmitGetDimensionSizeThunk(instruction);
    case HloOpcode::kSetDimensionSize:
      return EmitSetDimensionSizeThunk(instruction);

    case HloOpcode::kBatchNormGrad:
      return EmitBatchNormGradThunk(instruction);
    case HloOpcode::kBatchNormTraining:
      return EmitBatchNormTrainingThunk(instruction);

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
    case HloOpcode::kGather:
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
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
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
    case HloOpcode::kTranspose:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kXor:
      return EmitElementalKernelThunk(instruction);

    // ReplicaId and PartitionId identify the location of the current device in
    // a logical grid of communicating devices.
    case HloOpcode::kReplicaId:
      return EmitReplicaIdThunk(instruction);
    case HloOpcode::kPartitionId:
      return EmitPartitionIdThunk(instruction);

    case HloOpcode::kAllGather:
      return EmitAllGatherThunk(instruction);
    case HloOpcode::kAllReduce:
      return EmitAllReduceThunk(instruction);
    case HloOpcode::kReduceScatter:
      return EmitReduceScatterThunk(instruction);
    case HloOpcode::kAllToAll:
      return EmitAllToAllThunk(instruction);
    case HloOpcode::kCollectivePermute:
      return EmitCollectivePermuteThunk(instruction);

    case HloOpcode::kPad:
      return EmitPadKernelThunk(instruction);

    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
      return EmitSliceThunk(instruction);

    case HloOpcode::kDynamicUpdateSlice:
      return EmitDynamicUpdateSliceThunk(instruction);

    case HloOpcode::kConcatenate:
      return EmitConcatenateKernelThunk(instruction);

    case HloOpcode::kFusion:
      if (instruction->fusion_kind() == HloInstruction::FusionKind::kCustom) {
        // Fusion must have backend config with custom fusion config.
        TF_RET_CHECK(instruction->has_backend_config())
            << "Fusion must have backend config";
        TF_ASSIGN_OR_RETURN(auto backend_config,
                            instruction->backend_config<BackendConfig>());
        TF_RET_CHECK(backend_config.has_fusion_config())
            << "Backend config must have fusion config";

#if XLA_ONEDNN_USE_GRAPH_API
        if (backend_config.fusion_config().kind() == kOneDnnFusionKind) {
          return EmitOneDnnFusionThunk(instruction);
        }
#endif  // XLA_ONEDNN_USE_GRAPH_API

        if (backend_config.fusion_config().kind() == kXnnFusionKind) {
          return EmitXnnFusionThunk(instruction);
        }

        return Internal("Unsupported custom fusion kind: %s",
                        backend_config.DebugString());
      }
      return EmitFusionKernelThunk(instruction);

    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return EmitReductionKernelThunk(instruction);

    case HloOpcode::kRng:
      return EmitRngThunk(instruction);

    case HloOpcode::kRngBitGenerator:
      return EmitRngBitGeneratorThunk(instruction);

    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateStateThunk(instruction);

    case HloOpcode::kStochasticConvert:
      return EmitStochasticConvertThunk(instruction);

    case HloOpcode::kInfeed:
      return EmitInfeedThunk(instruction);

    case HloOpcode::kOutfeed:
      return EmitOutfeedThunk(instruction);

    case HloOpcode::kConvolution:
      return EmitConvolutionThunk(instruction);

    case HloOpcode::kCopy: {
      if (options_.compile_copy_as_llvm_kernel) {
        return EmitElementalKernelThunk(instruction);
      }
      return EmitCopyThunk(instruction);
    }

    case HloOpcode::kDot:
      return EmitDotThunk(instruction);

    case HloOpcode::kFft:
      return EmitFftThunk(instruction);

    case HloOpcode::kTopK:
      return Unimplemented("TopK is not yet supported by XLA:CPU ThunkEmitter");

    case HloOpcode::kCustomCall:
      return EmitCustomCallThunk(instruction);

    case HloOpcode::kSort:
      return EmitSortThunk(instruction);

    default:
      return absl::UnimplementedError(
          absl::StrCat("HLO opcode `", HloOpcodeString(instruction->opcode()),
                       "` is not supported by XLA:CPU ThunkEmitter"));
  }
}

static absl::StatusOr<ReductionKind> MatchReductionKind(
    const HloComputation* computation) {
  if (auto reduction_kind = MatchReductionComputation(computation)) {
    return reduction_kind.value();
  }
  return Unimplemented("Unsupported reduction computation: %s",
                       computation->ToString());
}

template <typename CollectiveInstruction>
static absl::StatusOr<CollectiveThunk::OpParams> GetCollectiveOpParams(
    const CollectiveInstruction* instruction) {
  return CollectiveThunk::OpParams{
      /*op_id=*/instruction->channel_id().has_value()
          ? instruction->channel_id().value()
          : instruction->GetModule()->unique_id(),
      /*has_channel_id=*/instruction->channel_id().has_value(),
      /*use_global_device_ids=*/instruction->use_global_device_ids(),
      /*replica_groups=*/instruction->replica_groups(),
  };
}

// TODO(ezhulenev): Figure out why AllToAll instruction does not have
// `use_global_device_ids` field and how to unify it with every other collective
// operation.
static absl::StatusOr<CollectiveThunk::OpParams> GetCollectiveOpParams(
    const HloAllToAllInstruction* instruction) {
  return CollectiveThunk::OpParams{
      /*op_id=*/instruction->channel_id().has_value()
          ? instruction->channel_id().value()
          : instruction->GetModule()->unique_id(),
      /*has_channel_id=*/instruction->channel_id().has_value(),
      /*use_global_device_ids=*/std::nullopt,
      /*replica_groups=*/instruction->replica_groups(),
  };
}

// TODO(ezhulenev): Figure out why CollectivePermute instruction does not have
// `use_global_device_ids` field and how to unify it with every other collective
// operation.
static absl::StatusOr<CollectiveThunk::OpParams> GetCollectiveOpParams(
    const HloCollectivePermuteInstruction* instruction) {
  return CollectiveThunk::OpParams{
      /*op_id=*/instruction->channel_id().has_value()
          ? instruction->channel_id().value()
          : instruction->GetModule()->unique_id(),
      /*has_channel_id=*/instruction->channel_id().has_value(),
      /*use_global_device_ids=*/std::nullopt,
      /*replica_groups=*/{},  // CollectivePermute does not have replica groups
  };
}

static absl::StatusOr<CollectiveThunk::OpBuffers> GetCollectiveOpBuffers(
    const HloInstruction* instruction,
    const BufferAssignment& buffer_assignment) {
  // Collect buffer slices for all operands.
  std::vector<BufferAllocation::Slice> source_buffers;
  std::vector<Shape> source_shapes;

  for (const HloInstruction* operand : instruction->operands()) {
    TF_ASSIGN_OR_RETURN(source_buffers.emplace_back(),
                        buffer_assignment.GetUniqueSlice(operand, {}));
    source_shapes.push_back(operand->shape());
  }

  // Collect buffer slices for all results.
  std::vector<BufferAllocation::Slice> destination_buffers;
  std::vector<Shape> destination_shapes;

  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(
        destination_buffers.emplace_back(),
        buffer_assignment.GetUniqueSlice(instruction, indexed.index));
    destination_shapes.push_back(indexed.shape);
  }

  return CollectiveThunk::OpBuffers{
      /*source_buffers=*/std::move(source_buffers),
      /*source_shapes=*/std::move(source_shapes),
      /*destination_buffers=*/std::move(destination_buffers),
      /*destination_shapes=*/std::move(destination_shapes),
  };
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllGatherThunk(
    const HloInstruction* instruction) {
  auto* all_gather = Cast<HloAllGatherInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(AllGatherThunk::OpParams op_params,
                      GetCollectiveOpParams(all_gather));
  TF_ASSIGN_OR_RETURN(AllGatherThunk::OpBuffers op_buffers,
                      GetCollectiveOpBuffers(all_gather, buffer_assignment_));
  AllGatherThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<AllGatherThunk>(
      ThunkInfo(all_gather), std::move(op_params), std::move(op_buffers),
      std::move(op_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllReduceThunk(
    const HloInstruction* instruction) {
  auto* all_reduce = Cast<HloAllReduceInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                      MatchReductionKind(all_reduce->to_apply()));
  TF_ASSIGN_OR_RETURN(AllReduceThunk::OpParams op_params,
                      GetCollectiveOpParams(all_reduce));
  TF_ASSIGN_OR_RETURN(AllReduceThunk::OpBuffers op_buffers,
                      GetCollectiveOpBuffers(all_reduce, buffer_assignment_));
  AllReduceThunk::OpResources op_resources = {communicator_resource_};

  bool single_replica = hlo_module_config_.replica_count() == 1 &&
                        hlo_module_config_.num_partitions() == 1;

  return ThunkSequence::Of<AllReduceThunk>(
      ThunkInfo(all_reduce), reduction_kind, std::move(op_params),
      std::move(op_buffers), std::move(op_resources), single_replica);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllToAllThunk(
    const HloInstruction* instruction) {
  auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(AllToAllThunk::OpParams op_params,
                      GetCollectiveOpParams(all_to_all));
  TF_ASSIGN_OR_RETURN(AllToAllThunk::OpBuffers op_buffers,
                      GetCollectiveOpBuffers(all_to_all, buffer_assignment_));
  AllToAllThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<AllToAllThunk>(
      ThunkInfo(all_to_all), std::move(op_params), std::move(op_buffers),
      std::move(op_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectivePermuteThunk(
    const HloInstruction* instruction) {
  auto* collective_permute = Cast<HloCollectivePermuteInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(CollectivePermuteThunk::OpParams op_params,
                      GetCollectiveOpParams(collective_permute));
  TF_ASSIGN_OR_RETURN(
      CollectivePermuteThunk::OpBuffers op_buffers,
      GetCollectiveOpBuffers(collective_permute, buffer_assignment_));
  CollectivePermuteThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<CollectivePermuteThunk>(
      ThunkInfo(collective_permute), std::move(op_params),
      std::move(op_buffers), std::move(op_resources),
      collective_permute->source_target_pairs());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReduceScatterThunk(
    const HloInstruction* instruction) {
  auto* reduce_scatter = Cast<HloReduceScatterInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                      MatchReductionKind(reduce_scatter->to_apply()));
  TF_ASSIGN_OR_RETURN(ReduceScatterThunk::OpParams op_params,
                      GetCollectiveOpParams(reduce_scatter));
  TF_ASSIGN_OR_RETURN(
      ReduceScatterThunk::OpBuffers op_buffers,
      GetCollectiveOpBuffers(reduce_scatter, buffer_assignment_));
  ReduceScatterThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<ReduceScatterThunk>(
      ThunkInfo(reduce_scatter), reduction_kind, std::move(op_params),
      std::move(op_buffers), std::move(op_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCallThunk(
    const HloInstruction* instruction) {
  if (std::optional<std::string> maybe_small_call =
          instruction->get_frontend_attribute("xla_cpu_small_call");
      maybe_small_call.has_value() && *maybe_small_call == "true") {
    ComputationKernelEmitter emitter(instruction, &buffer_assignment_,
                                     &target_machine_features_);
    TF_ASSIGN_OR_RETURN(LlvmKernelDefinition kernel_definition,
                        emitter.EmitKernelDefinition());

    auto [kernel_spec, kernel_source] =
        std::move(kernel_definition).ReleaseStorage();

    kernels_.push_back(
        {kernel_spec.name(), std::move(kernel_source).thread_safe_module()});

    return MakeKernelThunkSequence(
        instruction, std::move(kernel_spec),
        /*min_alignment=*/cpu_function_runtime::MinAlign());
  } else {
    TF_ASSIGN_OR_RETURN(
        ThunkSequence called_sequence,
        EmitHloComputation(instruction->called_computations().front()));
    return ThunkSequence::Of<CallThunk>(ThunkInfo(instruction),
                                        std::move(called_sequence));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConcatenateKernelThunk(
    const HloInstruction* instruction) {
  ConcatenateKernelEmitter emitter(instruction, &buffer_assignment_,
                                   &target_machine_features_);
  TF_ASSIGN_OR_RETURN(LlvmKernelDefinition kernel_definition,
                      emitter.EmitKernelDefinition());

  auto [kernel_spec, kernel_source] =
      std::move(kernel_definition).ReleaseStorage();

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instruction->backend_config<BackendConfig>());

  kernels_.push_back(
      {kernel_spec.name(), std::move(kernel_source).thread_safe_module()});

  if (backend_config.has_llvm_kernel_options()) {
    SetXlaCpuBackendOptions(*kernels_.back().module.getModuleUnlocked(),
                            backend_config.llvm_kernel_options());
  }

  return MakeKernelThunkSequence(
      instruction, std::move(kernel_spec),
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitGetDimensionSizeThunk(
    const HloInstruction* instruction) {
  return Unimplemented("GetDimensionSize should be rewritten for CPU.");
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSetDimensionSizeThunk(
    const HloInstruction* instruction) {
  return Unimplemented("SetDimensionSize should be rewritten for CPU.");
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitBatchNormGradThunk(
    const HloInstruction* instruction) {
  return Unimplemented("BatchNormGrad should be rewritten for CPU.");
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitBatchNormTrainingThunk(
    const HloInstruction* instruction) {
  return Unimplemented("BatchNormTraining should be rewritten for CPU.");
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionThunk(
    const HloInstruction* instruction) {
  // NOTE: The following code (along with TODOs and comments) partially
  // duplicates IrEmitter::HandleConvolution. This duplication is temporary,
  // as IrEmitter will be removed when we switch to thunks runtime.
  const HloInstruction* input = instruction->operand(0);
  const HloInstruction* kernel = instruction->operand(1);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*instruction, /*operands=*/{input, kernel},
      /*supported_types=*/
      {PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32, F64, C64, C128}));

  // TODO(tonywy): Add PotentiallyImplementedAsMKLConvolution to support
  // different data layouts.
  if (PotentiallyImplementedAsEigenConvolution(*instruction,
                                               target_machine_features_)) {
    const Shape& input_shape = input->shape();
    const Shape& kernel_shape = kernel->shape();
    const Shape& output_shape = instruction->shape();

    // The input, kernel and output agree with respect to layout.
    if (LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout()) &&
        LayoutUtil::IsMonotonicWithDim0Major(kernel_shape.layout()) &&
        LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout())) {
      TF_ASSIGN_OR_RETURN(auto input_buffer, GetAllocationSlice(input));

      TF_ASSIGN_OR_RETURN(auto kernel_buffer, GetAllocationSlice(kernel));

      TF_ASSIGN_OR_RETURN(auto output_buffer, GetAllocationSlice(instruction));

      ConvolutionThunk::Options options;
      options.multi_threaded =
          hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
      return ThunkSequence::Of<ConvolutionThunk>(
          ThunkInfo(instruction), options, input_buffer, input_shape,
          kernel_buffer, kernel_shape, output_buffer, output_shape,
          instruction->convolution_dimension_numbers(), instruction->window(),
          instruction->feature_group_count());
    }
  }

  // This is a completely un-optimized version of convolution just to
  // have an early version that works. E.g. the input index and
  // padding calculation is not hoisted out of the inner loop.
  //
  // See the description of convolution in the XLA documentation for the pseudo
  // code for convolution.
  VLOG(2) << "Falling back to unoptimized convolution: " << instruction->name();
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
  ElementalKernelEmitter emitter(instruction, &buffer_assignment_,
                                 &target_machine_features_);
  TF_ASSIGN_OR_RETURN(LlvmKernelDefinition kernel_definition,
                      emitter.EmitKernelDefinition());

  auto [kernel_spec, kernel_source] =
      std::move(kernel_definition).ReleaseStorage();

  kernels_.push_back(
      {kernel_spec.name(), std::move(kernel_source).thread_safe_module()});

  return MakeKernelThunkSequence(
      instruction, std::move(kernel_spec),
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPadKernelThunk(
    const HloInstruction* instruction) {
  const HloPadInstruction* padInstr = Cast<HloPadInstruction>(instruction);
  TF_ASSIGN_OR_RETURN(auto kernel, ir_emitter_.EmitPadHostKernel(padInstr));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(padInstr));

  return MakeKernelThunkSequence(
      padInstr, buffers, kernel,
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFusionKernelThunk(
    const HloInstruction* instruction) {
  auto* fusion = Cast<HloFusionInstruction>(instruction);

  if (ir_emitter_.IsSupportedByFusionEmitter(fusion) &&
      fusion->fused_expression_root()->opcode() == HloOpcode::kScatter) {
    auto kernel_emitter =
        std::make_unique<CpuScatterFusion>(buffer_assignment_, fusion);

    TF_ASSIGN_OR_RETURN(MlirKernelDefinition kernel_definition,
                        kernel_emitter->EmitKernelDefinition());

    auto [kernel_spec, kernel_source] =
        std::move(kernel_definition).ReleaseStorage();

    TF_ASSIGN_OR_RETURN(LlvmIrKernelSource llvm_ir_kernel_source,
                        fusion_compiler_.Compile(std::move(kernel_source)));

    kernels_.push_back({kernel_spec.name(),
                        std::move(llvm_ir_kernel_source).thread_safe_module()});

    return MakeKernelThunkSequence(
        instruction, std::move(kernel_spec),
        /*min_alignment=*/cpu_function_runtime::MinAlign());
  }

  if (options::UseExperimentalLoopFusion(hlo_module_config_) &&
      fusion->fusion_kind() == HloFusionInstruction::FusionKind::kLoop) {
    std::unique_ptr<mlir::MLIRContext> context =
        FusionCompiler::CreateContext();
    TF_ASSIGN_OR_RETURN(
        MlirKernelDefinition kernel_definition,
        EmitFusionKernel(*context, *fusion, &buffer_assignment_));

    auto [kernel_spec, kernel_source] =
        std::move(kernel_definition).ReleaseStorage();

    TF_ASSIGN_OR_RETURN(LlvmIrKernelSource llvm_ir_kernel_source,
                        fusion_compiler_.Compile(std::move(kernel_source)));

    kernels_.push_back({kernel_spec.name(),
                        std::move(llvm_ir_kernel_source).thread_safe_module()});

    return MakeKernelThunkSequence(
        instruction, std::move(kernel_spec),
        /*min_alignment=*/cpu_function_runtime::MinAlign());
  }

  TF_ASSIGN_OR_RETURN(auto kernel, ir_emitter_.EmitFusionHostKernel(fusion));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(instruction));

  return MakeKernelThunkSequence(
      instruction, buffers, kernel,
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReductionKernelThunk(
    const HloInstruction* instruction) {
  // TODO(ezhulenev): Port vectorized reduction emitter from IrEmitter.
  return EmitElementalKernelThunk(instruction);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngThunk(
    const HloInstruction* instruction) {
  return Unimplemented("Rng should be expanded for CPU.");
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngBitGeneratorThunk(
    const HloInstruction* instruction) {
  return Unimplemented("RngBitGenerator should be expanded for CPU.");
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngGetAndUpdateStateThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto state_buffer, GetAllocationSlice(instruction));
  auto* rng_state = Cast<HloRngGetAndUpdateStateInstruction>(instruction);
  return ThunkSequence::Of<RngGetAndUpdateStateThunk>(
      ThunkInfo(instruction), state_buffer, rng_state->delta());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitStochasticConvertThunk(
    const HloInstruction* instruction) {
  return Unimplemented("StochasticConvert should be decomposed for CPU.");
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

  // Collect resources for consumed and produced tokens.
  InfeedThunk::InfeedResources infeed_resources;
  TF_ASSIGN_OR_RETURN(infeed_resources.consume_token,
                      GetTokenResource(infeed->operand(0)));
  TF_ASSIGN_OR_RETURN(infeed_resources.produce_token,
                      GetTokenResource(infeed, {1}));

  return ThunkSequence::Of<InfeedThunk>(ThunkInfo(instruction), infeed_buffers,
                                        std::move(infeed_resources));
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

  // Collect resources for consumed and produced tokens.
  OutfeedThunk::OutfeedResources outfeed_resources;
  TF_ASSIGN_OR_RETURN(outfeed_resources.consume_token,
                      GetTokenResource(outfeed->operand(1)));
  TF_ASSIGN_OR_RETURN(outfeed_resources.produce_token,
                      GetTokenResource(outfeed));

  return ThunkSequence::Of<OutfeedThunk>(
      ThunkInfo(instruction), outfeed_buffers, std::move(outfeed_resources));
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

  // Check if while loop has a statically known trip count.
  TF_ASSIGN_OR_RETURN(
      auto loop_config,
      instruction->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count;
  if (loop_config.has_known_trip_count()) {
    trip_count = loop_config.known_trip_count().n();
  }

  return ThunkSequence::Of<WhileThunk>(ThunkInfo(instruction), cond_buffer,
                                       std::move(cond_thunk),
                                       std::move(body_thunk), trip_count);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDotThunk(
    const HloInstruction* instruction) {
  const HloInstruction* lhs = instruction->operand(0);
  const HloInstruction* rhs = instruction->operand(1);

  TF_RETURN_IF_ERROR(
      ElementTypesSameAndSupported(*instruction, /*operands=*/{lhs, rhs},
                                   /*supported_types=*/
                                   {PRED, S8, U8, S16, U16, S32, U32, S64, U64,
                                    BF16, F16, F32, F64, C64, C128}));

  const DotDimensionNumbers& dnums = instruction->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1) {
    return Unimplemented(
        "Dot with multiple contracting dimensions is not implemented.");
  }

  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_config_, *instruction, target_machine_features_,
      /*allow_runtime_calls=*/true);

  switch (strategy) {
    // Emit host kernel implementing dot instruction.
    case DotImplementationStrategy::kNaiveLlvmIr:
    case DotImplementationStrategy::kTiledLlvmIrGemm:
    case DotImplementationStrategy::kTiledLlvmIrGemv: {
      DotKernelEmitter emitter(instruction, &buffer_assignment_,
                               &target_machine_features_);
      TF_ASSIGN_OR_RETURN(LlvmKernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());

      auto [kernel_spec, kernel_source] =
          std::move(kernel_definition).ReleaseStorage();

      kernels_.push_back(
          {kernel_spec.name(), std::move(kernel_source).thread_safe_module()});

      return MakeKernelThunkSequence(
          instruction, std::move(kernel_spec),
          /*min_alignment=*/cpu_function_runtime::MinAlign());
    }

    // Emit DotThunk implementing dot instruction as a library call.
    case DotImplementationStrategy::kEigen: {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_slice,
                          GetAllocationSlice(lhs));
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_slice,
                          GetAllocationSlice(rhs));
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice out_slice,
                          GetAllocationSlice(instruction));

      // Decide whether to use XNNPACK or Eigen.
      bool use_xnn = hlo_module_config_.debug_options().xla_cpu_use_xnnpack();
      if (use_xnn) {
        TF_ASSIGN_OR_RETURN(use_xnn,
                            IsXnnDotSupported(dnums, lhs->shape(), rhs->shape(),
                                              instruction->shape()));
      }

      if (use_xnn) {
        XnnDotThunk::Options options = {XnnShouldUseThreadPool(instruction)};
        bool capture_rhs = HloPredicateIsOp<HloOpcode::kParameter>(rhs);
        return ThunkSequence::Of<XnnDotThunk>(
            std::move(options), ThunkInfo(instruction), dnums, lhs_slice,
            lhs->shape(), rhs_slice, rhs->shape(), out_slice,
            instruction->shape(), capture_rhs);
      } else {
        return ThunkSequence::Of<DotThunk>(
            ThunkInfo(instruction), dnums, lhs_slice, lhs->shape(), rhs_slice,
            rhs->shape(), out_slice, instruction->shape());
      }
    }
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitTopKThunk(
    const HloCustomCallInstruction* custom_call) {
  const auto& result_shape = custom_call->shape();
  const HloInstruction* input = custom_call->operand(0);
  TF_RET_CHECK(input->shape().element_type() == F32)
      << "TopK expects F32 data type for input";
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      result_shape.tuple_shapes(0).layout()))
      << custom_call->ToString();
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      result_shape.tuple_shapes(1).layout()))
      << custom_call->ToString();
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(input->shape().layout()))
      << custom_call->ToString();

  // Deduce parameters from the result shape and operand shape
  const int64_t input_size = input->shape().dimensions().back();
  const bool has_batch = result_shape.tuple_shapes(0).dimensions().size() == 2;
  const int64_t batch_size =
      has_batch ? result_shape.tuple_shapes(0).dimensions(0) : 1;
  const int64_t k = result_shape.tuple_shapes(0).dimensions().back();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice values_slice,
                      GetAllocationSlice(input));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice indices_slice,
                      GetAllocationSlice(custom_call, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      GetAllocationSlice(custom_call, {1}));
  return ThunkSequence::Of<TopKThunk>(ThunkInfo(custom_call), values_slice,
                                      indices_slice, output_slice, batch_size,
                                      input_size, k);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReplicaIdThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice replica_id_buffer,
                      GetAllocationSlice(instruction));
  return ThunkSequence::Of<ReplicaIdThunk>(ThunkInfo(instruction),
                                           replica_id_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPartitionIdThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice partition_id_buffer,
                      GetAllocationSlice(instruction));
  return ThunkSequence::Of<PartitionIdThunk>(ThunkInfo(instruction),
                                             partition_id_buffer);
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

static absl::StatusOr<CustomCallThunk::OpBuffers> GetCustomCallOpBuffers(
    const HloInstruction* instruction,
    const BufferAssignment& buffer_assignment) {
  // Collect buffer slices for all operands.
  std::vector<BufferAllocation::Slice> arguments_buffers;
  std::vector<Shape> arguments_shapes;
  for (HloInstruction* operand : instruction->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(
          arguments_buffers.emplace_back(),
          buffer_assignment.GetUniqueSlice(operand, indexed.index));
      arguments_shapes.push_back(indexed.shape);
    }
  }

  // Collect buffer slices for all results.
  std::vector<BufferAllocation::Slice> results_buffers;
  std::vector<Shape> results_shapes;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(
        results_buffers.emplace_back(),
        buffer_assignment.GetUniqueSlice(instruction, indexed.index));
    results_shapes.push_back(indexed.shape);
  }

  return CustomCallThunk::OpBuffers{
      /*arguments_buffers=*/std::move(arguments_buffers),
      /*arguments_shapes=*/std::move(arguments_shapes),
      /*results_buffers=*/std::move(results_buffers),
      /*results_shapes=*/std::move(results_shapes),
      /*is_tuple_result=*/instruction->shape().IsTuple(),
  };
}

static bool IsValidCustomCallApiVersion(CustomCallApiVersion api_version) {
  switch (api_version) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCustomCallThunk(
    const HloInstruction* instruction) {
  auto custom_call = Cast<HloCustomCallInstruction>(instruction);

  // TODO(penporn): Support these existing targets.
  auto custom_call_target = custom_call->custom_call_target();
  if (custom_call_target == "PadToStatic" ||
      custom_call_target == "__onednn$matmul" ||
      custom_call_target == "__onednn$softmax" ||
      custom_call_target == "__onednn$layernorm" ||
      custom_call_target == "__onednn$matmul_reorder") {
    return Unimplemented("Custom call target %s is not implemented.",
                         custom_call_target);
  }
  if (custom_call_target == "TopK") {
    return EmitTopKThunk(custom_call);
  } else if (custom_call_target == "SliceToDynamic") {
    return EmitSliceToDynamicThunk(instruction);
  }

  // Check the API version.
  auto version = custom_call->api_version();
  if (!IsValidCustomCallApiVersion(version)) {
    return InvalidArgument(
        "Unknown custom-call API version enum value: %d (%s)", version,
        CustomCallApiVersion_Name(version));
  }

  // Get backend config and buffer assignments.
  auto backend_config = custom_call->backend_config<BackendConfig>();
  if (!backend_config.ok()) {
    VLOG(3) << "Unable to parse backend config for custom call: "
            << backend_config.status().message() << "\n"
            << "Fall back to parse the opaque str.";
  }
  auto& backend_config_str =
      !backend_config.ok()
          ? custom_call->opaque()
          : ((version == API_VERSION_TYPED_FFI)
                 ? backend_config->custom_call_config().attributes()
                 : backend_config->custom_call_config().opaque());
  TF_ASSIGN_OR_RETURN(auto op_buffers,
                      GetCustomCallOpBuffers(instruction, buffer_assignment_));

  return ThunkSequence::Of<CustomCallThunk>(ThunkInfo(instruction),
                                            custom_call_target, op_buffers,
                                            backend_config_str, version);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSliceToDynamicThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto kernel,
                      ir_emitter_.EmitSliceToDynamicHostKernel(instruction));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(instruction));

  return MakeKernelThunkSequence(
      instruction, buffers, kernel,
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSliceThunk(
    const HloInstruction* instruction) {
  // TODO(ezhulenev): Consider implementing slice operations as separate
  // Thunks because it might be easier to get peak performance from hand
  // written code (Eigen slice expression for example).
  return EmitElementalKernelThunk(instruction);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDynamicUpdateSliceThunk(
    const HloInstruction* instruction) {
  if (!ir_emitter_.CanUpdateDynamicSliceInPlace(instruction)) {
    VLOG(2) << "Could not emit in-place dynamic-update-slice kernel: "
            << instruction->name();
    return EmitElementalKernelThunk(instruction);
  }

  TF_ASSIGN_OR_RETURN(
      auto kernel, ir_emitter_.EmitDynamicUpdateSliceHostKernel(instruction));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(instruction));

  return MakeKernelThunkSequence(instruction, buffers, kernel);
}

// Parse the sort comparator to determine the sort direction. Comparator is
// expected to be an HloOpcode::kCompare with two parameters.
std::optional<SortThunk::SortDirection> ThunkEmitter::MatchSortDirection(
    const HloComputation* hlo_comparator) const {
  namespace m = match;
  std::optional<SortThunk::SortDirection> direction = std::nullopt;

  // TODO(tsilytskyi): Handle more than two input parameters.
  if (hlo_comparator->root_instruction()->opcode() == HloOpcode::kCompare &&
      hlo_comparator->root_instruction()->operand(0)->opcode() ==
          HloOpcode::kParameter &&
      hlo_comparator->root_instruction()->operand(1)->opcode() ==
          HloOpcode::kParameter &&
      hlo_comparator->num_parameters() == 2) {
    auto* compare =
        Cast<HloCompareInstruction>(hlo_comparator->root_instruction());

    // Take into account the order of the parameters. If they are swapped,
    // the sort direction will be reversed.
    const bool expected_param_order =
        (Match(compare, m::Op()
                            .WithOperand(0, m::Parameter(0))
                            .WithOperand(1, m::Parameter(1))));
    switch (compare->comparison_direction()) {
      case ComparisonDirection::kGe:
        direction = (expected_param_order)
                        ? SortThunk::SortDirection::kDescending
                        : SortThunk::SortDirection::kAscending;
        break;
      case ComparisonDirection::kLt:
        direction = (expected_param_order)
                        ? SortThunk::SortDirection::kAscending
                        : SortThunk::SortDirection::kDescending;
        break;
      default:
        break;
    }
  }

  return direction;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSortThunk(
    const HloInstruction* instruction) {
  auto* sort = Cast<HloSortInstruction>(instruction);

  HloComputation* hlocomparator = sort->to_apply();

  const std::optional<SortThunk::SortDirection> direction =
      MatchSortDirection(hlocomparator);

  TF_ASSIGN_OR_RETURN(auto comparator,
                      ir_emitter_.EmitSortComparator(hlocomparator));
  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(sort));

  if (buffers.arguments.size() != buffers.results.size()) {
    return Internal(
        "Sort operation expects the same number of operands and results");
  }

  ThunkSequence thunks;

  std::vector<SortThunk::Input> inputs;
  inputs.reserve(sort->operand_count());

  for (size_t i = 0; i < sort->operand_count(); ++i) {
    const Shape& shape = sort->operand(i)->shape();

    BufferAllocation::Slice arg = buffers.arguments[i];
    BufferAllocation::Slice result = buffers.results[i];

    // Copy argument to result if they are not the same buffer.
    if (arg != result) {
      TF_ASSIGN_OR_RETURN(
          thunks.emplace_back(),
          CopyThunk::Create(ThunkInfo(instruction), arg, shape, result, shape));
    }

    // Add sort thunk input to sort result buffer inplace.
    inputs.push_back(SortThunk::Input{result, shape});
  }

  TF_ASSIGN_OR_RETURN(
      thunks.emplace_back(),
      SortThunk::Create(ThunkInfo(instruction), inputs, sort->sort_dimension(),
                        sort->is_stable(), comparator.name, direction));

  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOneDnnFusionThunk(
    const HloInstruction* instruction) {
#if XLA_ONEDNN_USE_GRAPH_API
  auto* fusion = Cast<HloFusionInstruction>(instruction);

  // Collect oneDNN fusion arguments.
  std::vector<OneDnnFusionThunk::Argument> arguments;
  for (HloInstruction* operand : instruction->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          buffer_assignment_.GetUniqueSlice(operand, indexed.index));
      arguments.push_back(OneDnnFusionThunk::Argument{slice, indexed.shape});
    }
  }

  // Collect oneDNN fusion results.
  std::vector<OneDnnFusionThunk::Result> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        buffer_assignment_.GetUniqueSlice(instruction, indexed.index));
    results.push_back(OneDnnFusionThunk::Result{slice, indexed.shape});
  }

  const HloComputation* computation = fusion->fused_instructions_computation();

  // Construct oneDNN fusion builder from the fusion computation.
  TF_ASSIGN_OR_RETURN(auto builder, EmitOneDnnFusionBuilder(computation));

  return ThunkSequence::Of<OneDnnFusionThunk>(
      ThunkInfo(instruction), std::move(arguments), std::move(results),
      [b = std::move(builder)](auto, auto) mutable { return b(); });
#else
  return Unimplemented("oneDNN fusion is not supported");
#endif  // XLA_ONEDNN_USE_GRAPH_API
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitXnnFusionThunk(
    const HloInstruction* instruction) {
  auto* fusion = Cast<HloFusionInstruction>(instruction);

  // Collect XNNPACK fusion arguments.
  std::vector<XnnFusionThunk::Argument> arguments;
  for (HloInstruction* operand : instruction->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          buffer_assignment_.GetUniqueSlice(operand, indexed.index));
      arguments.push_back(XnnFusionThunk::Argument{slice, indexed.shape});
    }
  }

  // Collect XNNPACK fusion results.
  std::vector<XnnFusionThunk::Result> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        buffer_assignment_.GetUniqueSlice(instruction, indexed.index));
    results.push_back(XnnFusionThunk::Result{slice, indexed.shape});
  }

  const HloComputation* computation = fusion->fused_instructions_computation();

  // Construct XNNPACK subgraph builder from the fusion computation.
  TF_ASSIGN_OR_RETURN(auto builder, EmitXnnFusionBuilder(computation));

  XnnFusionThunk::Options options = {XnnShouldUseThreadPool(computation)};
  return ThunkSequence::Of<XnnFusionThunk>(
      std::move(options), ThunkInfo(instruction), std::move(arguments),
      std::move(results),
      [b = std::move(builder)](auto, auto) mutable { return b(); });
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

absl::StatusOr<ThunkSequence> ThunkEmitter::MakeKernelThunkSequence(
    const HloInstruction* instruction,
    const ThunkEmitter::HostKernelAllocationSlices& buffers,
    const IrEmitter2::KernelInfo& kernel,
    std::optional<uint64_t> min_alignment) {
  // TODO(ezhulenev): Migrate KernelSpec to use NumWorkGroups.
  NumWorkGroups num_workgroups{kernel.thread_dims.x, kernel.thread_dims.y,
                               kernel.thread_dims.z};
  return ThunkSequence::Of<KernelThunk>(
      ThunkInfo(instruction), buffers.arguments, buffers.results, kernel.name,
      num_workgroups, kernel.invariant_arguments, min_alignment);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::MakeKernelThunkSequence(
    const HloInstruction* instruction, const KernelSpec& kernel_spec,
    std::optional<uint64_t> min_alignment) {
  return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction), kernel_spec,
                                        min_alignment);
}

}  // namespace xla::cpu
