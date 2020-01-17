/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/replica_id_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/tuple_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/sort_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

using absl::InlinedVector;
using absl::nullopt;
using absl::optional;
using absl::StrCat;
using llvm_ir::IrArray;
using llvm_ir::IrName;

namespace m = match;

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
const int64 kMinDimensionToTransposeTiled = 16;

// Updates the launch dimensions in "thunk" and annotate the launch dimensions
// of the corresponding IR kernel in "llvm_module".
// Precondition: "thunk" must be a KernelThunk.
void UpdateLaunchDimensions(const LaunchDimensions& launch_dims, Thunk* thunk,
                            llvm::Module* llvm_module) {
  CHECK(Thunk::Kind::kKernel == thunk->kind());
  KernelThunk* kernel_thunk = static_cast<KernelThunk*>(thunk);
  kernel_thunk->SetLaunchDimensions(launch_dims);

  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.
  llvm::NamedMDNode* nvvm_annotations_node =
      llvm_module->getOrInsertNamedMetadata("nvvm.annotations");
  llvm::Function* ir_kernel =
      llvm_module->getFunction(kernel_thunk->kernel_name().c_str());
  llvm::LLVMContext& llvm_context = llvm_module->getContext();
  llvm::ConstantInt* threads_per_block_ir_value = llvm::ConstantInt::get(
      llvm::IntegerType::get(llvm_context, /*NumBits=*/32),
      launch_dims.threads_per_block());
  // Our launch bounds are exact, so we can specify them as reqntidx rather than
  // maxntidx.
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      llvm_context,
      {llvm::ConstantAsMetadata::get(ir_kernel),
       llvm::MDString::get(llvm_context, "reqntidx"),
       llvm::ConstantAsMetadata::get(threads_per_block_ir_value)}));
}

}  // namespace

IrEmitterUnnested::IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                                     const HloComputation* hlo_computation,
                                     IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/false),
      hlo_computation_(hlo_computation) {
  // Initialize thunk_sequence_ to an empty list of thunks.
  thunk_sequence_.reset(new ThunkSequence());
}

Status IrEmitterUnnested::Postprocess(HloInstruction* hlo) {
  bindings_.UnbindAllLocalIrValues();
  return DfsHloVisitor::Postprocess(hlo);
}

llvm::Function* IrEmitterUnnested::BuildKernelPrototype(
    const HloInstruction& inst,
    absl::Span<const BufferAllocation* const> args) {
  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(inst.name()));

  // Create the kernel and add it to the module.
  llvm::Module* module = ir_emitter_context_->llvm_module();
  llvm::LLVMContext& context = module->getContext();
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(args.size(), b_.getInt8PtrTy()),
      /*isVarArg=*/false);
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             kernel_name.c_str(), module);

  // Add dereferenceable and alignment information to each of the kernel's
  // parameters.
  auto arg_it = kernel->arg_begin();
  for (size_t arg_no = 0; arg_no < args.size(); ++arg_no) {
    const BufferAllocation* alloc = args[arg_no];
    llvm::Argument* fn_arg = &*arg_it;
    ++arg_it;

    kernel->addDereferenceableAttr(arg_no + 1, alloc->size());

    const int64 alignment = [&] {
      if (alloc->is_entry_computation_parameter()) {
        return kEntryParameterAlignBytes;
      } else if (alloc->is_constant()) {
        return kConstantBufferAlignBytes;
      } else {
        return kXlaAllocatedBufferAlignBytes;
      }
    }();

    kernel->addParamAttr(
        arg_no,
        llvm::Attribute::get(context, llvm::Attribute::Alignment, alignment));

    if (alloc->IsPreallocatedTempBuffer()) {
      fn_arg->setName("temp_buf");
    } else {
      fn_arg->setName(StrCat("alloc", alloc->index()));
    }
  }

  AnnotateFunctionAsGpuKernel(module, kernel, &b_);

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  b_.SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

  return kernel;
}

namespace {
// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(const HloInstruction* hlo) {
  int max_unroll_factor = hlo->GetModule()
                              ->config()
                              .debug_options()
                              .xla_gpu_max_kernel_unroll_factor();

  // Find the largest possible power of two to unroll by.
  // TODO(kramerb): Make this smarter.
  const Shape& element_shape = hlo->IsMultiOutputFusion()
                                   ? ShapeUtil::GetSubshape(hlo->shape(), {0})
                                   : hlo->shape();
  int64 num_elements = ShapeUtil::ElementsIn(element_shape);
  for (int i = max_unroll_factor; i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }

  // Cannot unroll.
  return 1;
}

// Returns the llvm type for the indices used in the kernel that contains the
// hlo instruction. Such indices include the index for the parallel loop and
// the indices for the tensors accessed by the kernel. The return type is i32
// iff the following conditions are met:
//  . The launch_size of the kernel is within the range of i32.
//  . The sizes of all the tensors accessed within the kernel are within the
//    range of i32.
// Otherwise, the return type is i64.
llvm::Type* GetIndexTypeForKernel(const HloInstruction* hlo, int64 launch_size,
                                  llvm::IRBuilder<>* b) {
  // Find the unnested hlo instruction for which the kernel is generated for.
  const HloInstruction* unnested_hlo = hlo;
  const HloComputation* computation = hlo->parent();
  if (computation->IsFusionComputation()) {
    unnested_hlo = computation->FusionInstruction();
  }

  auto shape_in_range = [&](const Shape& s) {
    bool in_range = true;
    ShapeUtil::ForEachSubshape(s, [&](const Shape& sub_shape,
                                      const ShapeIndex& /*index*/) {
      if (sub_shape.IsArray() && !IsInt32(ShapeUtil::ElementsIn(sub_shape))) {
        in_range = false;
      }
    });

    return in_range;
  };

  llvm::Type* i64_ty = b->getInt64Ty();
  // Check launch dimension
  if (!IsInt32(launch_size)) {
    return i64_ty;
  }

  // Check the size of result tensors
  if (!shape_in_range(unnested_hlo->shape())) {
    return i64_ty;
  }

  auto hlo_shape_in_range = [&](const HloInstruction* operand) -> bool {
    return shape_in_range(operand->shape());
  };

  // Check the size of input tensors
  if (!absl::c_all_of(unnested_hlo->operands(), hlo_shape_in_range)) {
    return i64_ty;
  }

  // Check the size of the internal result tensors
  if (unnested_hlo->opcode() == HloOpcode::kFusion) {
    if (!absl::c_all_of(
            unnested_hlo->fused_instructions_computation()->instructions(),
            hlo_shape_in_range)) {
      return i64_ty;
    }
  }

  return b->getInt32Ty();
}

// Gets the input shape of the ROOT slices, which will be used as the kernel
// launch dims. The slice input fusion requires the input shapes of the ROOT
// slices to be the same although the (slice) output shapes can be different.
//
// Returns the input shape of the ROOT slices if all the input shapes of ROOT
// slices are the same and the slices are non-strided. Otherwise, returns
// FailedPrecondition.
StatusOr<Shape> GetConsistentInputShapeForRootSlices(
    const HloInstruction& fusion) {
  if (!IsInputFusibleSlices(fusion, /*verify_no_strides=*/true)) {
    return FailedPrecondition(
        "Unsupported root for slice input fusion. "
        "Only non-strided slices are supported.");
  }

  const HloInstruction& root = *fusion.fused_expression_root();
  if (root.opcode() == HloOpcode::kSlice) {
    return root.operands()[0]->shape();
  }

  CHECK_EQ(root.opcode(), HloOpcode::kTuple);
  const Shape& first_slice_operand_shape =
      root.operands()[0]->operands()[0]->shape();
  for (size_t i = 1; i < root.operands().size(); ++i) {
    const HloInstruction* slice = root.operands()[i];
    const Shape& operand_shape = slice->operands()[0]->shape();
    if (!ShapeUtil::EqualIgnoringElementType(first_slice_operand_shape,
                                             operand_shape)) {
      return FailedPrecondition(
          "Fused slices do not have the same input shape, fused computation = "
          "%s.",
          root.parent()->name());
    }
  }

  return first_slice_operand_shape;
}

}  // namespace

Status IrEmitterUnnested::DefaultAction(HloInstruction* hlo) {
  return IrEmitter::DefaultAction(hlo);
}

Status IrEmitterUnnested::HandleDot(HloInstruction* dot) {
  AddThunkToThunkSequence(
      BuildKernelThunk(dot, /*implements_whole_instruction=*/true));
  return IrEmitter::HandleDot(dot);
}

Status IrEmitterUnnested::HandleConditional(HloInstruction* conditional) {
  AddThunkToThunkSequence(BuildConditionalThunk(conditional));
  return Status::OK();
}

Status IrEmitterUnnested::HandleConvolution(HloInstruction* convolution) {
  AddThunkToThunkSequence(
      BuildKernelThunk(convolution, /*implements_whole_instruction=*/true));
  return IrEmitter::HandleConvolution(convolution);
}

Status IrEmitterUnnested::HandleCustomCall(HloInstruction* custom_call) {
  return ThunkEmitter(this).HandleCustomCall(custom_call);
}

Status IrEmitterUnnested::HandleFft(HloInstruction* fft) {
  return ThunkEmitter(this).HandleFft(fft);
}

Status IrEmitterUnnested::HandleTriangularSolve(HloInstruction* hlo) {
  return ThunkEmitter(this).HandleTriangularSolve(hlo);
}

Status IrEmitterUnnested::HandleFusion(HloInstruction* fusion) {
  HloInstruction* root = fusion->fused_expression_root();
  if (fusion->IsInputFusion()) {
    switch (root->opcode()) {
      case HloOpcode::kScatter: {
        std::vector<std::unique_ptr<Thunk>> thunks;
        // The initialization from 'operand' is using different loop bounds, so
        // emit it in a separate kernel. Treat it like a loop fusion, writing to
        // the output buffer.
        {
          int unroll_factor = ComputeMaxUnrollFactor(fusion);
          thunks.push_back(BuildKernelThunk(
              fusion, /*implements_whole_instruction=*/false, unroll_factor));
          GpuElementalIrEmitter operand_elemental_emitter(
              hlo_module_config_, ir_emitter_context_->llvm_module(), &b_,
              GetNestedComputer());
          FusedIrEmitter operand_fused_emitter(
              GetGeneratorForOperandIrArrays(fusion),
              &operand_elemental_emitter);
          TF_RETURN_IF_ERROR(
              root->mutable_operand(0)->Accept(&operand_fused_emitter));

          TF_RETURN_IF_ERROR(EmitTargetElementLoopInThunk(
              *fusion, operand_fused_emitter.GetGenerator(root->operand(0)),
              static_cast<KernelThunk*>(thunks.back().get())));
        }

        // Now build the actual scatter, reading and writing to the freshly
        // filled output buffer.
        {
          thunks.push_back(
              BuildKernelThunk(fusion,
                               /*implements_whole_instruction=*/false));
          // Spin up a new fused emitter for the scatter kernel and emit it.
          GpuElementalIrEmitter scatter_elemental_emitter(
              hlo_module_config_, ir_emitter_context_->llvm_module(), &b_,
              GetNestedComputer());
          FusedIrEmitter scatter_fused_emitter(
              GetGeneratorForOperandIrArrays(fusion),
              &scatter_elemental_emitter);
          TF_RETURN_IF_ERROR(root->Accept(&scatter_fused_emitter));
          TF_RETURN_IF_ERROR(EmitScatter(
              thunks.back().get(), root,
              /*scatter_indices_gen=*/
              scatter_fused_emitter.GetGenerator(root->operand(1)),
              /*updates_gen=*/
              scatter_fused_emitter.GetGenerator(root->operand(2))));
        }
        AddThunkToThunkSequence(
            absl::make_unique<SequentialThunk>(std::move(thunks), fusion));
        return Status::OK();
      }
      // In the case of root tuple, it can be either reduce or slice input
      // fusion.
      case HloOpcode::kTuple: {
        if (IsInputFusibleSlices(*fusion)) {
          return EmitInputFusibleNonStridedSlices(fusion);
        }

        CHECK_GE(root->operand_count(), 1);
        return EmitReductionFromOrToContiguousDimensions(fusion,
                                                         root->operands());
      }
      case HloOpcode::kReduce: {
        // HandleFusion specializes reduction from a multi-dimensional array to
        // a 1D array. The specialized version requires a initializer thunk that
        // initializes the output array to the initial value of the reduce.
        if (root->shape().IsTuple()) {
          // TODO(b/129089333): Support tiled vectorized variadic reduce.
          return Unimplemented(
              "Vectorized variadic reduce is not supported on GPU");
        }
        return EmitReductionFromOrToContiguousDimensions(fusion, {root});
      }
      case HloOpcode::kSlice: {
        return EmitInputFusibleNonStridedSlices(fusion);
      }
      default:
        LOG(FATAL) << "Bad opcode for input fusion: "
                   << fusion->fused_expression_root()->opcode();
    }
  } else if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(
                 fusion, ir_emitter_context_->buffer_assignment())) {
    // Fusion node with dynamic-update-slice as the root where the op's input
    // (i.e. array to update) shares the same slice as its output.  In this case
    // we have a special algorithm that modifies the output in place without
    // touching the un-updated elements.

    // Set up kernel thunk and fused ir emitter.
    std::unique_ptr<KernelThunk> fusion_thunk =
        BuildKernelThunk(fusion, /*implements_whole_instruction=*/true);
    GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                            ir_emitter_context_->llvm_module(),
                                            &b_, GetNestedComputer());

    // Shape of the dynamic-update-slice's "update" operand.
    Shape update_shape = root->operand(1)->shape();

    // Array to write into.  Because this is an in-place operation, this is the
    // same as operand 0's array.
    IrArray output_array = GetIrArray(*fusion, *fusion);

    LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
        update_shape, ir_emitter_context_->device_description());
    UpdateLaunchDimensions(launch_dimensions, fusion_thunk.get(),
                           ir_emitter_context_->llvm_module());
    AddThunkToThunkSequence(std::move(fusion_thunk));

    return llvm_ir::EmitParallelFusedDynamicUpdateSliceInPlace(
        fusion, GetGeneratorForOperandIrArrays(fusion), output_array,
        &elemental_emitter, launch_dimensions, &b_);
  }

  CHECK_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kLoop)
      << ": " << fusion->ToString();

  if (CheckAndEmitHloWithTile021(fusion)) {
    return Status::OK();
  }

  return IrEmitter::HandleFusion(fusion);
}

Status IrEmitterUnnested::HandleCopy(HloInstruction* copy) {
  CHECK(ShapeUtil::Compatible(copy->operand(0)->shape(), copy->shape()));
  const BufferAssignment& buffer_assignment =
      ir_emitter_context_->buffer_assignment();
  if (LayoutUtil::Equal(copy->operand(0)->shape().layout(),
                        copy->shape().layout()) &&
      buffer_assignment.GetUniqueTopLevelSlice(copy->operand(0)).ok()) {
    // Copy the operand into the output if it's not the same buffer already.
    auto operand_buffer = GetAllocationSlice(*copy->operand(0));
    auto destination_buffer = GetAllocationSlice(*copy);
    if (operand_buffer != destination_buffer) {
      AddThunkToThunkSequence(absl::make_unique<DeviceToDeviceCopyThunk>(
          /*source_address=*/operand_buffer,
          /*destination_buffer=*/destination_buffer,
          /*mem_size=*/
          ByteSizeOf(copy->operand(0)->shape()), copy));
    }
    return Status::OK();
  }
  if (CheckAndEmitHloWithTile021(copy)) {
    return Status::OK();
  }

  return IrEmitter::HandleCopy(copy);
}

Status IrEmitterUnnested::EmitExtraOutputsForReduce(
    const HloInstruction* unnested_hlo, const IrArray::Index& index,
    bool use_linear_index,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  for (int i = 0; i != extra_output_gens.size(); ++i) {
    llvm::Value* extra_output_address =
        GetIrArray(*unnested_hlo, *unnested_hlo, extra_output_gens[i].second)
            .EmitArrayElementAddress(index, &b_, "extra_output_element_address",
                                     use_linear_index);
    TF_ASSIGN_OR_RETURN(llvm::Value* const extra_output_ir_value,
                        extra_output_gens[i].first(index));
    Store(extra_output_ir_value, extra_output_address);
  }
  return Status::OK();
}

Status IrEmitterUnnested::HandleReduce(HloInstruction* reduce) {
  if (IsReductionFromOrToContiguousDimensions(*reduce) &&
      reduce->shape().IsArray()) {
    return EmitReductionFromOrToContiguousDimensions(reduce, {reduce});
  }

  return IrEmitter::HandleReduce(reduce);
}

Status IrEmitterUnnested::HandleTuple(HloInstruction* tuple) {
  // For the root node of the entry computation we can elide writing the tuple
  // buffer. We can always figure out the contents of the tuples from buffer
  // assignment because we insert copies to ensure non-ambiguous output buffers.
  // GpuExecutable never reads the tuple buffer.
  if (tuple ==
      tuple->parent()->parent()->entry_computation()->root_instruction()) {
    return Status::OK();
  }
  bool all_tuple_elements_have_buffer =
      absl::c_all_of(tuple->operands(), [&](HloInstruction* tuple_element) {
        return ir_emitter_context_->buffer_assignment()
            .GetUniqueTopLevelSlice(tuple_element)
            .ok();
      });
  // TODO(b/111689850): This logic isn't quite correct.
  //
  // Tuples (especially tuples that are the final result of a computation) can
  // be so huge that if we were to emit a kernel that took each tuple element as
  // a parameter, we would exceed the max allowable number of parameters to a
  // GPU kernel, b/31336476. As an optimization, if all tuple elements have a
  // buffer, we collect their buffer addresses in a host array, and then copy
  // that array to the tuple's buffer.
  //
  // Some tuple elements might not have an unambiguous buffer (like the result
  // of a select-tuple). In that case, we fall back to emitting kernels which
  // have access to their buffer addresses in code.
  if (all_tuple_elements_have_buffer) {
    std::vector<BufferAllocation::Slice> tuple_element_buffers;
    for (const HloInstruction* tuple_element : tuple->operands()) {
      tuple_element_buffers.push_back(GetAllocationSlice(*tuple_element));
    }
    AddThunkToThunkSequence(absl::make_unique<TupleThunk>(
        tuple_element_buffers, GetAllocationSlice(*tuple), tuple));
    return Status::OK();
  }
  AddThunkToThunkSequence(
      BuildKernelThunk(tuple, /*implements_whole_instruction=*/true));
  return IrEmitter::HandleTuple(tuple);
}

Status IrEmitterUnnested::HandleGetTupleElement(HloInstruction*) {
  // GetTupleElement IR is emitted in the IR context of the user instruction,
  // and so we do not build a kernel for GetTupleElement instructions.
  return Status::OK();
}

Status IrEmitterUnnested::HandleSelectAndScatter(
    HloInstruction* select_and_scatter) {
  CHECK_EQ(select_and_scatter->operand_count(), 3);
  const auto* operand = select_and_scatter->operand(0);
  const auto* source = select_and_scatter->operand(1);
  const Window& window = select_and_scatter->window();
  PrimitiveType operand_element_type = operand->shape().element_type();
  const int64 rank = operand->shape().rank();
  CHECK_EQ(rank, source->shape().rank());
  CHECK_EQ(rank, window.dimensions_size());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> initializer_thunk,
                      BuildInitializerThunk(select_and_scatter));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(initializer_thunk));
  thunks.push_back(BuildKernelThunk(select_and_scatter,
                                    /*implements_whole_instruction=*/false));
  std::unique_ptr<SequentialThunk> select_and_scatter_thunk =
      absl::make_unique<SequentialThunk>(std::move(thunks), select_and_scatter);

  // TODO(b/31410564): Implement dilation rate for select-and-scatter.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for SelectAndScatter not implemented on GPU.");
  }

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      source->shape(), ir_emitter_context_->device_description());
  llvm::Type* index_type = GetIndexTypeForKernel(
      select_and_scatter, launch_dimensions.launch_bound(), &b_);
  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_type, c);
  };

  // kSelectAndScatter is implemented as two kernel launches: the first launch
  // initializes the output array to the given initial value,
  // and the second accumulates the "source" matrix to the
  // selected elements in the output array. The first launch is already
  // implemented by the initializer thunk generated earlier, so this function
  // only needs to take care of the select-and-scatter part.
  //
  // Pseudo code for select-and-scatter:
  //
  // for (coordinates S in the source):  # This loop is parallel.
  //   initialized_flag = false
  //   for (coordinates W in the window):
  //     I = S * stride + W - pad_low
  //     if I within bounds of operand:
  //       if !(initialized_flag and select(selected_value, operand(I))):
  //         selected_value = operand(I)
  //         selected_index = I
  //         initialized_flag = true
  //   output(selected_index) = scatter(output(selected_index), source(S))
  auto loop_body_emitter = [=](const IrArray::Index& source_index) -> Status {
    // Allocate space to keep the currently selected value, its index, and a
    // boolean flag if the value is initialized. The initialized_flag is set
    // false.
    llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_element_type,
                                       ir_emitter_context_->llvm_module()),
        "selected_value_address", &b_);
    llvm::Value* selected_index_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            index_type, index_typed_constant(rank), "selected_index_address",
            &b_);
    llvm::Value* initialized_flag_address = llvm_ir::EmitAllocaAtFunctionEntry(
        b_.getInt1Ty(), "initialized_flag_address", &b_);
    Store(b_.getInt1(false), initialized_flag_address);

    // Create the inner loop to iterate over the window.
    llvm_ir::ForLoopNest window_loops(IrName(select_and_scatter, "inner"), &b_,
                                      index_type);
    DimensionVector window_size;
    for (const auto& dim : window.dimensions()) {
      window_size.push_back(dim.size());
      CHECK_GT(dim.size(), 0);
    }
    const IrArray::Index window_index = window_loops.AddLoopsForShape(
        ShapeUtil::MakeShape(operand_element_type, window_size), "window");
    llvm_ir::SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(),
                                   &b_);

    // Compute the operand index to visit and evaluate the condition whether the
    // operand index is within the bounds. The unsigned comparison includes
    // checking whether the operand index >= 0.
    std::vector<llvm::Value*> operand_multi_index(source_index.size());
    llvm::Value* in_bounds_condition = b_.getInt1(true);
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* strided_index = NSWMul(
          source_index[i], index_typed_constant(window.dimensions(i).stride()));
      operand_multi_index[i] =
          NSWSub(NSWAdd(strided_index, window_index[i]),
                 index_typed_constant(window.dimensions(i).padding_low()));
      llvm::Value* index_condition = ICmpULT(
          operand_multi_index[i],
          index_typed_constant(ShapeUtil::GetDimension(operand->shape(), i)));
      in_bounds_condition = And(in_bounds_condition, index_condition);
    }
    CHECK(in_bounds_condition != nullptr);

    // Only need to do something if the operand index is within the bounds.
    // First check if the initialized_flag is set.
    llvm_ir::LlvmIfData if_in_bounds =
        llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, &b_);
    llvm_ir::LlvmIfData if_initialized = llvm_ir::EmitIfThenElse(
        Load(initialized_flag_address), "initialized", &b_);

    // If the initialized_flag is false, initialize the selected value and index
    // with the currently visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.false_block, &b_);
    const auto save_operand_index = [&](const IrArray::Index& operand_index) {
      for (int64 i = 0; i < rank; ++i) {
        llvm::Value* selected_index_address_slot =
            InBoundsGEP(selected_index_address, {b_.getInt32(i)});
        Store(operand_index[i], selected_index_address_slot);
      }
    };
    IrArray operand_array = GetIrArray(*operand, *select_and_scatter);
    IrArray::Index operand_index(operand_multi_index, operand->shape(),
                                 index_type);
    llvm::Value* operand_data =
        operand_array.EmitReadArrayElement(operand_index, &b_);
    Store(operand_data, selected_value_address);
    save_operand_index(operand_index);
    Store(b_.getInt1(true), initialized_flag_address);

    // If the initialized_flag is true, call the `select` function to
    // potentially update the selected value and index with the currently
    // visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.true_block, &b_);
    llvm::Value* operand_address =
        operand_array.EmitArrayElementAddress(operand_index, &b_);
    llvm::Value* select_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(PRED,
                                       ir_emitter_context_->llvm_module()),
        "select_return_buffer", &b_);
    TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
        *select_and_scatter->select(),
        {selected_value_address, operand_address}, select_return_buffer));
    llvm::Value* result = Load(select_return_buffer);

    // If the 'select' function returns false, update the selected value and the
    // index to the currently visiting operand.
    llvm::Value* cond = ICmpNE(
        result,
        llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(
                                   PRED, ir_emitter_context_->llvm_module()),
                               0),
        "boolean_predicate");
    llvm_ir::LlvmIfData if_select_lhs =
        llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &b_);
    llvm_ir::SetToFirstInsertPoint(if_select_lhs.false_block, &b_);
    Store(Load(operand_address), selected_value_address);
    save_operand_index(operand_index);

    // After iterating over the window elements, scatter the source element to
    // the selected index of the output. The value we store at the output
    // location is computed by calling the `scatter` function with the source
    // value and the current output value.
    llvm_ir::SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                                   &b_);
    std::vector<llvm::Value*> selected_multi_index;
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot =
          InBoundsGEP(selected_index_address, {b_.getInt32(i)});
      selected_multi_index.push_back(Load(selected_index_address_slot));
    }
    llvm::Value* source_value_address =
        GetIrArray(*source, *select_and_scatter)
            .EmitArrayElementAddress(source_index, &b_);
    IrArray::Index selected_index(selected_multi_index,
                                  select_and_scatter->shape(),
                                  operand_index.GetType());
    llvm::Value* output_value_address =
        GetIrArray(*select_and_scatter, *select_and_scatter)
            .EmitArrayElementAddress(selected_index, &b_);
    return EmitAtomicOperationForNestedComputation(
        *select_and_scatter->scatter(), output_value_address,
        source_value_address);
  };

  UpdateLaunchDimensions(
      launch_dimensions,
      // IrEmitterUnnested implements kSelectAndScatter as a SequentialThunk
      // consisting of two thunks, an initializer KernelThunk that initializes
      // the output and another KernelThunk that accumulates the scattered
      // elements.
      select_and_scatter_thunk->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  AddThunkToThunkSequence(std::move(select_and_scatter_thunk));
  return ParallelLoopEmitter(loop_body_emitter, source->shape(),
                             launch_dimensions, &b_)
      .EmitLoop(IrName(select_and_scatter), index_type);
}

Status IrEmitterUnnested::HandleWhile(HloInstruction* xla_while) {
  HloComputation* condition = xla_while->while_condition();
  TF_RET_CHECK(ShapeUtil::IsScalar(condition->root_instruction()->shape()) &&
               condition->root_instruction()->shape().element_type() == PRED)
      << "While condition computation must return bool";
  // Build ForThunk for conformant while loops, otherwise build WhileThunk.
  auto config = xla_while->backend_config<WhileLoopBackendConfig>();
  if (config.ok() && config.ValueOrDie().has_known_trip_count()) {
    AddThunkToThunkSequence(
        BuildForThunk(xla_while, config.ValueOrDie().known_trip_count().n()));
  } else {
    AddThunkToThunkSequence(BuildWhileThunk(xla_while));
  }
  return Status::OK();
}

Status IrEmitterUnnested::HandleRng(HloInstruction* rng) {
  return Unimplemented("Rng should be expanded for GPU.");
}

Status IrEmitterUnnested::HandleRngGetAndUpdateState(
    HloInstruction* rng_state) {
  // Emit a kernel to increment the global state for Philox RNG algorithm.
  AddThunkToThunkSequence(
      BuildKernelThunk(rng_state, /*implements_whole_instruction=*/true));

  llvm::Value* old_state = llvm_ir::RngGetAndUpdateState(
      Cast<HloRngGetAndUpdateStateInstruction>(rng_state)->delta(), module_,
      &b_);

  llvm::Value* output_address =
      GetIrArray(*rng_state, *rng_state)
          .EmitArrayElementAddress(
              llvm_ir::IrArray::Index(
                  /*linear=*/b_.getInt64(0), rng_state->shape(), &b_),
              &b_, "rng_state_address");
  output_address = BitCast(
      output_address, llvm::PointerType::get(
                          old_state->getType(),
                          output_address->getType()->getPointerAddressSpace()));
  Store(old_state, output_address);

  return Status::OK();
}

Status IrEmitterUnnested::HandleScatter(HloInstruction* scatter) {
  const HloInstruction* operand = scatter->operand(0);
  const HloInstruction* scatter_indices = scatter->operand(1);
  const HloInstruction* updates = scatter->operand(2);
  std::vector<std::unique_ptr<Thunk>> thunks;

  // Copy the operand into the output if it's not the same buffer already.
  auto operand_buffer = GetAllocationSlice(*operand);
  auto destination_buffer = GetAllocationSlice(*scatter);
  if (operand_buffer != destination_buffer) {
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        /*source_address=*/operand_buffer,
        /*destination_buffer=*/destination_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(operand->shape()),
        /*hlo_instruction=*/nullptr));
  }

  thunks.push_back(
      BuildKernelThunk(scatter,
                       /*implements_whole_instruction=*/thunks.empty()));

  TF_RETURN_IF_ERROR(EmitScatter(
      thunks.back().get(), scatter,
      /*scatter_indices_gen=*/
      [=](const IrArray::Index& index) {
        return GetIrArray(*scatter_indices, *scatter)
            .EmitReadArrayElement(index, &b_, "scatter_index");
      },
      /*updates_gen=*/
      [=](const IrArray::Index& index) {
        return GetIrArray(*updates, *scatter)
            .EmitReadArrayElement(index, &b_, "update");
      }));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(
        absl::make_unique<SequentialThunk>(std::move(thunks), scatter));
  }

  return Status::OK();
}

Status IrEmitterUnnested::EmitScatter(
    Thunk* thunk, HloInstruction* scatter,
    const llvm_ir::ElementGenerator& scatter_indices_gen,
    const llvm_ir::ElementGenerator& updates_gen) {
  const HloInstruction* operand = scatter->operand(0);
  const HloInstruction* scatter_indices = scatter->operand(1);
  const HloInstruction* updates = scatter->operand(2);
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  CHECK(ShapeUtil::Equal(scatter->shape(), operand->shape()));

  auto loop_body_emitter = [&](const IrArray::Index& index) -> Status {
    std::vector<llvm::Value*> raw_window_multidim;
    std::vector<llvm::Value*> input_scatter_multidim;
    std::vector<int64> raw_window_bounds;

    // Partition the index into window indices and scatter indices.
    for (int64 i = 0, e = index.size(); i != e; ++i) {
      // For window indices also remember the window size, this comes in handy
      // later.
      if (absl::c_binary_search(dim_numbers.update_window_dims(), i)) {
        raw_window_multidim.push_back(index[i]);
        raw_window_bounds.push_back(updates->shape().dimensions(i));
      } else {
        input_scatter_multidim.push_back(index[i]);
      }
    }
    DCHECK_EQ(raw_window_multidim.size(),
              dim_numbers.update_window_dims_size());

    // Apply inserted_window_dims to the window dimensions.
    int64 raw_window_multidim_idx = 0;
    std::vector<llvm::Value*> input_window_multidim;
    std::vector<int64> input_window_bounds;
    for (int64 i = 0, e = operand->shape().rank(); i != e; ++i) {
      if (absl::c_binary_search(dim_numbers.inserted_window_dims(), i)) {
        input_window_bounds.push_back(1);  // Trivial dimension.
        input_window_multidim.push_back(index.GetConstantWithIndexType(0));
      } else {
        input_window_bounds.push_back(
            raw_window_bounds[raw_window_multidim_idx]);
        input_window_multidim.push_back(
            raw_window_multidim[raw_window_multidim_idx]);
        ++raw_window_multidim_idx;
      }
    }
    DCHECK_EQ(input_window_multidim.size(), operand->shape().rank());

    // Insert a 1 dimension at the end if index_vector_dim requests one.
    Shape scatter_indices_shape = scatter_indices->shape();
    if (dim_numbers.index_vector_dim() == scatter_indices_shape.rank()) {
      scatter_indices_shape.add_dimensions(1);
      scatter_indices_shape.mutable_layout()->add_minor_to_major(
          dim_numbers.index_vector_dim());
    }

    // Now load the indices corresponding to the current window from
    // scatter_indices.
    std::vector<llvm::Value*> raw_scatter_index_multidim =
        input_scatter_multidim;
    raw_scatter_index_multidim.insert(
        raw_scatter_index_multidim.begin() + dim_numbers.index_vector_dim(),
        nullptr);
    llvm::Value* is_in_bounds = b_.getTrue();
    for (int64 i = 0, e = dim_numbers.scatter_dims_to_operand_dims_size();
         i != e; ++i) {
      // Our index is stored along index_vector_dim, insert that into the lookup
      // index into scatter_indices.
      raw_scatter_index_multidim[dim_numbers.index_vector_dim()] =
          index.GetConstantWithIndexType(i);
      llvm_ir::IrArray::Index raw_scatter_index_index(
          raw_scatter_index_multidim, scatter_indices_shape, index.GetType());

      int64 operand_dim = dim_numbers.scatter_dims_to_operand_dims(i);
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const loaded_scatter_index,
          scatter_indices_gen(raw_scatter_index_index.SourceIndexOfReshape(
              scatter_indices_shape, scatter_indices->shape(), &b_)));
      // And add the index to our window index. This yields the output index.
      llvm::Value* casted_scatter_index =
          IntCast(loaded_scatter_index, index.GetType(),
                  /*isSigned=*/true);
      llvm::Value* dim_offset =
          Add(input_window_multidim[operand_dim], casted_scatter_index);
      input_window_multidim[operand_dim] = dim_offset;

      // Also do the bounds check now.
      int64 max_index = operand->shape().dimensions(operand_dim) -
                        input_window_bounds[operand_dim] + 1;
      // is_in_bounds = index >= 0 && index < dim_size-window_size+1
      //   --> index u< dim_size-window_size+1
      is_in_bounds =
          And(is_in_bounds, ICmpULT(casted_scatter_index,
                                    index.GetConstantWithIndexType(max_index)));
    }

    llvm_ir::LlvmIfData if_window_in_bounds_data = llvm_ir::EmitIfThenElse(
        is_in_bounds, "scatter.in_bounds", &b_, /*emit_else=*/false);
    llvm_ir::SetToFirstInsertPoint(if_window_in_bounds_data.true_block, &b_);
    // All done, now just read from the calculated input from the window, and do
    // an atomic store to the calculated location in the output.
    HloInstruction* output_hlo =
        scatter->IsFused() ? scatter->parent()->FusionInstruction() : scatter;
    llvm_ir::IrArray::Index input_window_index(
        input_window_multidim, output_hlo->shape(), index.GetType());
    llvm::Value* output_address =
        GetIrArray(*output_hlo, *output_hlo)
            .EmitArrayElementAddress(input_window_index, &b_);
    llvm::Value* input_address = Alloca(llvm_ir::PrimitiveTypeToIrType(
        updates->shape().element_type(), module_));
    TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value, updates_gen(index));
    Store(input_ir_value, input_address);

    if (!scatter->unique_indices()) {
      return EmitAtomicOperationForNestedComputation(
          *scatter->to_apply(), output_address, input_address);
    } else {
      return EmitCallToNestedComputation(*scatter->to_apply(),
                                         {output_address, input_address},
                                         output_address);
    }
  };

  // Launch a kernel that reads every element in the updates tensor. We could
  // also do one kernel per window instead if bounds checks turn out to be a
  // bottleneck.
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      updates->shape(), ir_emitter_context_->device_description());
  UpdateLaunchDimensions(launch_dimensions, thunk,
                         ir_emitter_context_->llvm_module());

  return ParallelLoopEmitter(loop_body_emitter, updates->shape(),
                             launch_dimensions, &b_)
      .EmitLoop(IrName(scatter),
                GetIndexTypeForKernel(scatter, launch_dimensions.launch_bound(),
                                      &b_));
}

Status IrEmitterUnnested::HandleSelect(HloInstruction* select) {
  return IrEmitter::HandleSelect(select);
}

Status IrEmitterUnnested::HandleSort(HloInstruction* sort) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  Shape keys_shape = sort->operand(0)->shape();
  int64 dimension_to_sort = sort->dimensions(0);
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    // We assume that the layout of all involved operands and outputs is the
    // same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(keys_shape,
                                                  sort->operand(i)->shape()));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index)));

    // If possible, we share buffers. If that is not possible, we need to copy
    // the values, because the emitter does the sorting in-place.
    auto destination_buffer = GetAllocationSlice(*sort, shape_index);
    auto source_address = GetAllocationSlice(*sort->operand(i));
    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share buffers for
      // key/value sort.
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          /*source_address=*/source_address,
          /*destination_buffer=*/destination_buffer,
          /*mem_size=*/ShapeUtil::ByteSizeOf(sort->operand(i)->shape()),
          nullptr));
    }
  }

  uint64 dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64 num_stages = tensorflow::Log2Ceiling(dimension_to_sort_bound);
  CHECK_GE(1ULL << num_stages, dimension_to_sort_bound);
  CHECK_LT(1ULL << (num_stages - 1), dimension_to_sort_bound);

  // Naive C++ code for the outer loops:
  //
  // for (int64 stage = 0; stage < Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64 first_xor_mask = (1LL << (stage + 1)) - 1;
  //   SortInPlace(first_xor_mask);
  //   for (int64 mask = stage - 1; mask >= 0; --mask) {
  //     int64 later_xor_mask = 1LL << mask;
  //     SortInPlace(later_xor_mask);
  //   }
  // }
  //
  // This follows the alternative representation of the algorithm described on
  // Wikipedia: https://en.wikipedia.org/wiki/Bitonic_sorter
  //
  // Each mask specifies how to derive from one position in the array the
  // position with which it should be compared (we calculate the xor of the
  // position with the mask).
  // As an optimization, we can move the 'mask' loop to inside the
  // sorting/comparison loop if the comparisons happen within a small block of
  // the array. To make this work, we collect all consecutive masks that are
  // smaller than our chosen power of 2 tile size, and pass them to SortInPlace.
  // Each thread then processes one tile of data.

  const uint64 kTileSize = std::min(2048ULL, 1ULL << num_stages);

  // If we cannot combine several xor masks together, we don't use tiling, so we
  // calculate the standard launch dimensions for the shape. However we only
  // need to iterate through ~half of the dimension to sort (rounded up to the
  // next highest power of 2), because each iteration compares one pair of
  // elements.
  Shape standard_iteration_shape = keys_shape;
  uint64 standard_num_iterations_in_sort_dim = 1ULL << (num_stages - 1);
  standard_iteration_shape.set_dimensions(dimension_to_sort,
                                          standard_num_iterations_in_sort_dim);
  LaunchDimensions standard_launch_dimensions = CalculateLaunchDimensions(
      standard_iteration_shape, ir_emitter_context_->device_description());

  // Calculate the launch dimensions for the case where we use tiling. We split
  // the dimension that should be sorted into tiles of size 'kTileSize'. This
  // means we first need to round 'dimension_to_sort_bound' up to be a multiple
  // of the tile size.
  int64 rounded_bound = RoundUpToNearest(dimension_to_sort_bound, kTileSize);
  Shape iteration_shape = keys_shape;

  // We iterate through the element pairs that should be compared.
  uint64 num_iterations_in_sort_dim = rounded_bound / 2;
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64 num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  // For correctness reasons we need exactly 'kTileSize' / 2 many threads per
  // block. Each thread is responsible for copying exactly two adjacent elements
  // into shared memory, and then does a comparison of two possibly different
  // elements taken from shared memory.
  const uint64 kThreadsPerBlock = kTileSize / 2;

  // Check whether we should use any tiling. We might not be able to use it if
  // we have not enough threads, or not enough shared memory. Also it does not
  // give a speedup if the tile size is < 128.
  int64 total_shared_memory_needed = 0;
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    total_shared_memory_needed +=
        kTileSize * ShapeUtil::ByteSizeOfPrimitiveType(
                        sort->operand(i)->shape().element_type());
  }
  bool no_tiling =
      kTileSize < 128 ||
      kThreadsPerBlock >
          ir_emitter_context_->device_description().threads_per_block_limit() ||
      total_shared_memory_needed >
          ir_emitter_context_->device_description().shared_memory_per_block();

  uint64 num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);

  auto emit_kernel = [&](absl::Span<const int64> xor_masks) {
    thunks.push_back(
        BuildKernelThunk(sort, /*implements_whole_instruction=*/false));
    LaunchDimensions launch_dimensions = xor_masks.size() > 1
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    UpdateLaunchDimensions(launch_dimensions, thunks.back().get(),
                           ir_emitter_context_->llvm_module());
    std::vector<IrArray> values_arrays;
    values_arrays.reserve(sort->operand_count());
    for (int64 i = 0; i < sort->operand_count(); ++i) {
      ShapeIndex shape_index =
          sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
      values_arrays.push_back(GetIrArray(*sort, *sort, shape_index));
    }
    return llvm_ir::EmitSortInPlace(
        dimension_to_sort, values_arrays, IrName(sort), xor_masks, &b_,
        launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        kTileSize,
        [&](absl::Span<llvm::Value* const> operands, llvm::Value* output) {
          return EmitCallToNestedComputation(*sort->to_apply(), operands,
                                             output);
        });
  };
  std::vector<int64> xor_masks;
  for (int64 stage = 0; stage < num_stages; ++stage) {
    for (int64 mask = stage; mask >= 0; --mask) {
      int64 xor_mask;
      if (mask == stage) {
        xor_mask = (1LL << (stage + 1)) - 1;
      } else {
        xor_mask = 1LL << mask;
      }
      if (xor_mask >= kTileSize || no_tiling) {
        if (!xor_masks.empty()) {
          TF_RETURN_IF_ERROR(emit_kernel(xor_masks));
          xor_masks.clear();
        }
        TF_RETURN_IF_ERROR(emit_kernel({xor_mask}));
      } else {
        xor_masks.push_back(xor_mask);
      }
    }
  }
  if (!xor_masks.empty()) {
    TF_RETURN_IF_ERROR(emit_kernel(xor_masks));
  }

  AddThunkToThunkSequence(
      absl::make_unique<SequentialThunk>(std::move(thunks), sort));
  return Status::OK();
}

Status IrEmitterUnnested::HandleTupleSelect(HloInstruction* tuple_select) {
  AddThunkToThunkSequence(
      BuildKernelThunk(tuple_select, /*implements_whole_instruction=*/true));
  return IrEmitter::HandleTupleSelect(tuple_select);
}

Status IrEmitterUnnested::HandleReplicaId(HloInstruction* hlo) {
  AddThunkToThunkSequence(
      absl::make_unique<ReplicaIdThunk>(GetAllocationSlice(*hlo), hlo));
  return Status::OK();
}

Status IrEmitterUnnested::HandleCollectivePermute(HloInstruction* hlo) {
  AddThunkToThunkSequence(absl::make_unique<CollectivePermuteThunk>(
      GetAllocationSlice(*hlo->operand(0)), GetAllocationSlice(*hlo), hlo));
  return Status::OK();
}

namespace {


}  // namespace

Status IrEmitterUnnested::HandleAllReduce(HloInstruction* crs) {
  VLOG(2) << "AllReduce; replica count: " << hlo_module_config_.replica_count()
          << "; operand count: " << crs->operand_count()
          << "; NCCL is enabled: " << NcclAllReduceThunk::NcclIsEnabled();

  // Note the replica_count == 1 case is handled via device-to-device copy
  // below.
  bool should_use_nccl_thunk = hlo_module_config_.replica_count() > 1 &&
                               NcclAllReduceThunk::CanImplement(crs);

  if (should_use_nccl_thunk) {
    CHECK(crs->operand(0)->shape().IsArray())
        << "Operands to all-reduce must be arrays: " << crs->ToString();
    AddThunkToThunkSequence(absl::make_unique<NcclAllReduceThunk>(
        /*replica_count=*/hlo_module_config_.replica_count(),
        /*elements=*/ShapeUtil::ElementsIn(crs->operand(0)->shape()),
        /*source_address=*/GetAllocationSlice(*crs->operand(0)),
        /*destination_buffer=*/GetAllocationSlice(*crs), crs));
    return Status::OK();
  }

  if (hlo_module_config_.replica_count() != 1) {
    // TODO(b/33011107): Support more AllReduce configurations on GPU.
    string message = absl::StrFormat(
        "Requested AllReduce not implemented on GPU; replica_count: %d; "
        "operand_count: %d; IsCrossReplicaAllReduce: %d; NCCL support: %d",
        hlo_module_config_.replica_count(), crs->operand_count(),
        crs->IsCrossReplicaAllReduce(), NcclAllReduceThunk::NcclIsEnabled());
    if (crs->operand_count() > 0) {
      absl::StrAppendFormat(
          &message, "; first operand array element-type: %s",
          PrimitiveType_Name(crs->operand(0)->shape().element_type()));
    }
    return Unimplemented("%s", message);
  }

  // CRS with one operand and one replica is simply the identity function.
  // Buffer assignment expects a copy, so that's what we do.
  //
  // TODO(b/80100934): We would like to eliminate one-replica CRS nodes entirely
  // in algebraic-simplifier, but currently on some platforms
  // HloModuleConfig::num_replicas changes between when the module is compiled
  // and when it's run.
  if (crs->operand_count() == 1) {
    CHECK(crs->operand(0)->shape().IsArray())
        << "Operands to all-reduce must be arrays: " << crs->ToString();
    AddThunkToThunkSequence(absl::make_unique<DeviceToDeviceCopyThunk>(
        /*source_address=*/GetAllocationSlice(*crs->operand(0)),
        /*destination_buffer=*/GetAllocationSlice(*crs),
        /*mem_size=*/ShapeUtil::ByteSizeOf(crs->shape()), crs));
    return Status::OK();
  }

  // One-replica CRS with multiple operands produces a tuple of the inputs.
  // Again, buffer assignment expects us to copy each.
  std::vector<std::unique_ptr<Thunk>> thunks;
  std::vector<BufferAllocation::Slice> tuple_element_buffers;
  for (int64 i = 0; i < crs->operand_count(); ++i) {
    tuple_element_buffers.push_back(ir_emitter_context_->buffer_assignment()
                                        .GetUniqueSlice(crs, {i})
                                        .ValueOrDie());
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        /*source_address=*/GetAllocationSlice(*crs->operand(i)),
        /*destination_buffer=*/tuple_element_buffers.back(),
        /*mem_size=*/ShapeUtil::ByteSizeOf(crs->operand(i)->shape()), nullptr));
  }

  // Output a tuple of the buffers above.
  thunks.push_back(absl::make_unique<TupleThunk>(
      tuple_element_buffers, GetAllocationSlice(*crs), nullptr));
  AddThunkToThunkSequence(
      absl::make_unique<SequentialThunk>(std::move(thunks), crs));
  return Status::OK();
}

Status IrEmitterUnnested::HandleInfeed(HloInstruction* xla_infeed) {
  return ThunkEmitter(this).HandleInfeed(xla_infeed);
}

Status IrEmitterUnnested::HandleOutfeed(HloInstruction* outfeed) {
  return ThunkEmitter(this).HandleOutfeed(outfeed);
}

Status IrEmitterUnnested::HandleAfterAll(HloInstruction* after_all) {
  return Status::OK();
}

// Figures out how to access the buffers for all subshapes of hlo's operands and
// for hlo itself (i.e. all the buffers produced by HLO).
//
// Returns a map keyed on the pair {HloInstruction, ShapeIndex}.  The value for
// this key is a pair {Slice, ShapeIndex}, where the slice tells you the root
// buffer to look in, and the ShapeIndex describes how to dereference starting
// at that buffer to get to the buffer in question.
//
// For example, if {hlo, {1}} is mapped to {slice, {3, 4}}, then the buffer for
// hlo at ShapeIndex {1} (i.e. the buffer for the second tuple element of hlo)
// is found at slice[3][4].  That is, slice is a void***, which we dereference
// twice -- first at index 3, and then at index 4 -- to get the address of our
// buffer.
//
// This function conservatively assumes that we'll touch all sub-buffers of
// every operand and of the output.
static std::map<std::pair<const HloInstruction*, ShapeIndex>,
                std::pair<BufferAllocation::Slice, ShapeIndex>>
GetHloBufferSlices(const HloInstruction* hlo,
                   const BufferAssignment& buffer_assn) {
  std::map<std::pair<const HloInstruction*, ShapeIndex>,
           std::pair<BufferAllocation::Slice, ShapeIndex>>
      slices;

  // Tries to find a slice plus an array of indices i1, ..., iN such that the
  // sub-buffer for instr at index can be found at slice[i1]...[iN].
  auto find_slice_for = [&](const HloInstruction* instr,
                            const ShapeIndex& index)
      -> optional<std::pair<BufferAllocation::Slice, ShapeIndex>> {
    // Simple, common case: Is the buffer for instr known at runtime?  If so,
    // we're done.
    auto slice = buffer_assn.GetUniqueSlice(instr, index);
    if (slice.ok()) {
      return {{slice.ValueOrDie(), ShapeIndex()}};
    }

    // If that didn't work, walk up any bitcasts that we might see.  These must
    // appear before any GTE instructions, because it's illegal to bitcast to a
    // tuple type.
    const HloInstruction* parent = instr;
    while (parent->opcode() == HloOpcode::kBitcast) {
      parent = parent->operand(0);

      auto slice = buffer_assn.GetUniqueSlice(parent, {});
      if (slice.ok()) {
        return {{slice.ValueOrDie(), ShapeIndex()}};
      }
    }

    // Check whether instr is a GTE instruction.  If it is, see if we can get a
    // buffer for its parent, and continue walking up parents until we find a
    // defined buffer or we hit something that's not a GTE.
    ShapeIndex gte_indices;
    while (parent->opcode() == HloOpcode::kGetTupleElement) {
      gte_indices.push_front(parent->tuple_index());
      parent = parent->operand(0);

      auto slice = buffer_assn.GetUniqueSlice(parent, {});
      if (slice.ok()) {
        return {{slice.ValueOrDie(), gte_indices}};
      }
    }

    // Finally, if we don't know the buffer for instr at index, see if we know
    // the buffer for instr at index without its last element.  If so, we can
    // dynamically find the buffer for instr by dereferencing a pointer in that
    // buffer.  Continue looking this way until we run out of elements in
    // 'index'.
    //
    // We can almost always get a buffer without resorting to this.  The only
    // exception is for cases where the relevant sub-buffer is truly unknowable,
    // for example the sub-buffer of a tuple-shaped select.
    ShapeIndex new_index = index;
    while (!new_index.empty()) {
      gte_indices.push_front(new_index.back());
      new_index.pop_back();
      auto slice = buffer_assn.GetUniqueSlice(instr, new_index);
      if (slice.ok()) {
        return {{slice.ValueOrDie(), gte_indices}};
      }
    }

    return nullopt;
  };

  // Adds entries for all subshapes of instr to `slices`.
  auto add_slices_for = [&](const HloInstruction* instr) {
    ShapeUtil::ForEachSubshape(
        instr->shape(), [&](const Shape& /*shape*/, const ShapeIndex& index) {
          if (slices.count({instr, index})) {
            // HLOs can have duplicate operands; don't bother redoing work.
            return;
          }
          auto maybe_slice = find_slice_for(instr, index);
          if (maybe_slice.has_value()) {
            slices[{instr, index}] = *maybe_slice;
          } else {
            VLOG(1) << "Couldn't find buffer for " << instr->ToString()
                    << " at index " << index.ToString();
          }
        });
  };

  add_slices_for(hlo);
  for (const HloInstruction* operand : hlo->operands()) {
    // Conservatively assume we'll need the buffers for all subshapes of the
    // operand.
    add_slices_for(operand);
  }

  return slices;
}

std::unique_ptr<KernelThunk> IrEmitterUnnested::BuildKernelThunk(
    const HloInstruction* inst, bool implements_whole_instruction,
    int unroll_factor) {
  const BufferAssignment& buffer_assn =
      ir_emitter_context_->buffer_assignment();

  std::map<std::pair<const HloInstruction*, ShapeIndex>,
           std::pair<BufferAllocation::Slice, ShapeIndex>>
      hlo_slices = GetHloBufferSlices(inst, buffer_assn);

  // Figure out which buffer allocations need to be passed as arguments to our
  // kernel.  This is simply all of the allocations referenced in hlo_slices,
  // plus the XLA temp buffer (if we have it).  We always include the temp
  // buffer because even if the kernel itself doesn't use it, a nested
  // subcomputation within the kernel (e.g. a kMap's computation) might.
  std::unordered_set<const BufferAllocation*> buffers_needed;
  for (const auto& kv : hlo_slices) {
    buffers_needed.insert(kv.second.first.allocation());
  }
  absl::optional<const BufferAllocation*> temp_buffer;
  for (const BufferAllocation& alloc : buffer_assn.Allocations()) {
    if (alloc.IsPreallocatedTempBuffer()) {
      if (!temp_buffer.has_value()) {
        temp_buffer = &alloc;
      } else {
        LOG(FATAL) << "Multiple temp buffers found, but only one is allowed!";
      }
    }
  }
  if (temp_buffer.has_value()) {
    buffers_needed.insert(*temp_buffer);
  }

  // We'll pass a pointer to each of the elements of `buffers` to our kernel, in
  // this order.
  std::vector<const BufferAllocation*> non_constant_buffers;
  absl::c_copy_if(buffers_needed, std::back_inserter(non_constant_buffers),
                  [](const BufferAllocation* allocation) {
                    return !allocation->is_constant();
                  });

  absl::c_sort(non_constant_buffers,
               [](const BufferAllocation* a, const BufferAllocation* b) {
                 return a->index() < b->index();
               });

  llvm::Function* kernel = BuildKernelPrototype(*inst, non_constant_buffers);

  // Build a map from a BufferAllocation to the corresponding argument in our
  // kernel.
  std::unordered_map<const BufferAllocation*, llvm::Value*> kernel_args;
  {
    auto arg_it = kernel->arg_begin();
    auto buffers_it = non_constant_buffers.begin();
    for (; arg_it != kernel->arg_end(); ++arg_it, ++buffers_it) {
      kernel_args[*buffers_it] = arg_it;
    }
  }

  // For each buffer our kernel might want to touch, bind it to a value derived
  // from our kernel args.
  for (const auto& kv : hlo_slices) {
    const HloInstruction* instr = kv.first.first;
    const ShapeIndex& index = kv.first.second;
    const BufferAllocation::Slice& slice = kv.second.first;
    const ShapeIndex& gte_index = kv.second.second;

    VLOG(3) << "Buffer for " << instr->ToString() << " at " << index.ToString()
            << " is found in slice " << slice.ToString() << " at GTE index "
            << gte_index.ToString();

    llvm::Value* loc;
    if (slice.allocation()->is_constant()) {
      loc = ir_emitter_context_->llvm_module()->getGlobalVariable(
          llvm_ir::ConstantBufferAllocationToGlobalName(*slice.allocation()));
      CHECK_NE(loc, nullptr);
    } else {
      loc = InBoundsGEP(kernel_args.at(slice.allocation()),
                        {b_.getInt64(slice.offset())});
    }

    // If gte_index is nonempty, we have to dereference `loc` to get to the
    // value we're ultimately interested in.
    llvm::Type* int8_double_pointer =
        llvm::PointerType::get(b_.getInt8PtrTy(), /*AddressSpace=*/0);
    for (int64 idx : gte_index) {
      loc = BitCast(loc, int8_double_pointer);
      loc = Load(InBoundsGEP(loc, {b_.getInt64(idx)}));
    }

    bindings_.BindHloToIrValue(*instr, loc, index);
  }

  // Bind the temp buffer so that nested subcomputations can find it if they
  // need.
  if (temp_buffer.has_value()) {
    bindings_.SetTempBufferBase(kernel_args.at(*temp_buffer));
  } else {
    bindings_.SetTempBufferBase(
        llvm::ConstantPointerNull::get(b_.getInt8PtrTy()));
  }

  return absl::make_unique<KernelThunk>(
      non_constant_buffers, kernel->getName(),
      implements_whole_instruction ? inst : nullptr, unroll_factor);
}

StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildInitializerThunk(
    HloInstruction* hlo, const ShapeIndex& index) {
  bool fused = HloOpcode::kFusion == hlo->opcode();
  HloInstruction* inst = fused ? hlo->fused_expression_root() : hlo;
  HloInstruction* init_value_operand = [&] {
    switch (inst->opcode()) {
      case HloOpcode::kSelectAndScatter:
        return inst->mutable_operand(2);
      case HloOpcode::kReduce:
        return inst->mutable_operand(1);
      case HloOpcode::kTuple:
        CHECK(hlo->IsMultiOutputFusion())
            << ": " << hlo->ToString() << " is not a multi-output fusion.";
        CHECK(inst->operand(index.back())->opcode() == HloOpcode::kReduce)
            << ": Found '" << inst->operand(index.back())->opcode() << "' in "
            << inst->ToString() << " but expected 'reduce'.";
        // For multi-output fusion look through the tuple.
        return inst->mutable_operand(index.back())->mutable_operand(1);
      default:
        LOG(FATAL) << "Opcode " << inst->opcode()
                   << " should not need an initializer.";
    }
  }();

  const HloInstruction* init_value = init_value_operand;
  if (fused && init_value->opcode() == HloOpcode::kParameter) {
    init_value = hlo->operand(init_value->parameter_number());
  }

  // Initializer thunks don't implement a whole instruction, and we want to
  // profile the whole instruction instead of the individual thunks it consists
  // of. Therefore we pass nullptr as the HloInstruction* to the thunks we
  // generate below.
  //
  // In the common case, the initializer is a constant.  In this case, emit a
  // device-memset call if we can.  Currently StreamExecutor only supports
  // zeroing and 32-bit memsets.
  if (init_value->IsConstant()) {
    CHECK(ShapeUtil::IsScalar(init_value->shape()));
    int64 num_bytes = ShapeUtil::ByteSizeOfElements(init_value->shape());
    const auto& literal = init_value->literal();

    // Are all the bytes of this scalar equal to 0?  If so, we can create a
    // MemzeroThunk.
    absl::Span<const uint8> literal_bytes(
        reinterpret_cast<const uint8*>(literal.untyped_data()), num_bytes);
    if (absl::c_all_of(literal_bytes, [](uint8 byte) { return byte == 0; })) {
      return {absl::make_unique<MemzeroThunk>(GetAllocationSlice(*hlo, index),
                                              nullptr)};
    }

    // If the literal is 8 or 16 bits wide, we can emit a 32-bit memset by
    // repeating the literal 4 or 2 times, so long as the destination buffer is
    // an even multiple of 32 bits long.
    const Shape& output_shape = ShapeUtil::GetSubshape(hlo->shape(), index);
    if ((num_bytes == 1 || num_bytes == 2) &&
        ShapeUtil::ByteSizeOf(output_shape) % 4 == 0) {
      uint16 pattern16;
      if (num_bytes == 1) {
        uint8 b = literal_bytes.front();
        pattern16 = uint16{b} | (uint16{b} << 8);
      } else {
        memcpy(&pattern16, literal_bytes.data(), sizeof(pattern16));
      }
      uint32 pattern32 = uint32{pattern16} | (uint32{pattern16} << 16);
      return {absl::make_unique<Memset32BitValueThunk>(
          pattern32, GetAllocationSlice(*hlo, index), nullptr)};
    }

    // If the literal is an even multiple of 32 bits wide, we can emit a 32-bit
    // memset so long as all 32-bit words of the scalar are equal to each other.
    if (num_bytes >= 4 && num_bytes % 4 == 0 &&
        memcmp(literal_bytes.data(), literal_bytes.data() + 4,
               literal_bytes.size() - 4) == 0) {
      uint32 word;
      memcpy(&word, literal_bytes.data(), sizeof(word));
      return {absl::make_unique<Memset32BitValueThunk>(
          word, GetAllocationSlice(*hlo, index), nullptr)};
    }
  }

  // Otherwise fall back to our slow initializer code.
  std::unique_ptr<KernelThunk> kernel_thunk =
      BuildKernelThunk(hlo, /*implements_whole_instruction=*/false);
  LaunchDimensions launch_dimensions =
      CalculateLaunchDimensions(ShapeUtil::GetSubshape(hlo->shape(), index),
                                ir_emitter_context_->device_description());
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  if (fused) {
    // If init_value was fused into this reduce we have to generate it first.
    GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                            ir_emitter_context_->llvm_module(),
                                            &b_, GetNestedComputer());

    FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(hlo),
                                 &elemental_emitter);
    TF_RETURN_IF_ERROR(init_value_operand->Accept(&fused_emitter));
    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(fused_emitter.GetGenerator(init_value_operand),
                            GetIrArray(*hlo, *hlo, index), launch_dimensions,
                            &b_)
            .EmitLoop(IrName(hlo)));
  } else {
    // In the unfused case the element is already there, just read from it.
    TF_RETURN_IF_ERROR(ParallelLoopEmitter(
                           [=](const IrArray::Index& index) {
                             return GetIrArray(*init_value, *hlo)
                                 .EmitReadArrayElement(index, &b_);
                           },
                           GetIrArray(*hlo, *hlo, index), launch_dimensions,
                           &b_)
                           .EmitLoop(IrName(hlo)));
  }

  // Clean up state left behind by emitting the loop above.  (This is normally
  // done in IrEmitterUnnested::Postprocess().)
  bindings_.UnbindAllLocalIrValues();

  // Convert unique_ptr<KernelThunk> to StatusOr<unique_ptr<Thunk>>.
  return {std::move(kernel_thunk)};
}

namespace {

// Checks that the buffers corresponding to the given two HLOs share the same
// allocation.
Status CheckHloBuffersShareAllocation(
    const HloInstruction* a, const HloInstruction* b, const ShapeIndex& index,
    const BufferAssignment& buffer_assignment) {
  const BufferAllocation::Slice slice_a =
      buffer_assignment.GetUniqueSlice(a, index).ConsumeValueOrDie();
  const BufferAllocation::Slice slice_b =
      buffer_assignment.GetUniqueSlice(b, index).ConsumeValueOrDie();
  if (slice_a != slice_b) {
    return InternalError(
        "instruction %s %s does not share allocation with instruction %s %s",
        a->ToString(), slice_a.ToString(), b->ToString(), slice_b.ToString());
  }
  return Status::OK();
}

// Checks that all buffers used during while loop iteration share the same
// buffer allocation. This includes buffers for while result, while init
// operand, condition parameter, body parameter and body result.
// Returns OK on success, error status otherwise.
Status CheckWhileBuffersShareAllocation(
    const HloInstruction* xla_while,
    const BufferAssignment& buffer_assignment) {
  return ShapeUtil::ForEachSubshapeWithStatus(
      xla_while->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) -> Status {
        const HloInstruction* condition_parameter =
            xla_while->while_condition()->parameter_instruction(0);
        const HloComputation* body = xla_while->while_body();
        const HloInstruction* body_parameter = body->parameter_instruction(0);
        const HloInstruction* body_result = body->root_instruction();
        TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
            xla_while, xla_while->operand(0), index, buffer_assignment));
        TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
            xla_while, condition_parameter, index, buffer_assignment));
        TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
            xla_while, body_parameter, index, buffer_assignment));
        TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
            xla_while, body_result, index, buffer_assignment));
        return Status::OK();
      });
}

// Checks that the buffers used in a conditional instruction are shared with the
// operands and result as follows:
//   * The result buffer of the conditional should share the allocation with the
//     result buffers of each branch computation.
//   * The buffer of operand b+1 should share the allocation with the buffer of
//     the parameter 0 instruction of the b'th computation.
Status CheckConditionalBuffersShareAllocation(
    const HloInstruction* conditional,
    const BufferAssignment& buffer_assignment) {
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      conditional->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) -> Status {
        for (auto branch_computation : conditional->branch_computations()) {
          TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
              conditional, branch_computation->root_instruction(), index,
              buffer_assignment));
        }
        return Status::OK();
      }));
  for (int j = 0; j < conditional->branch_count(); ++j) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        conditional->operand(j + 1)->shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) -> Status {
          return CheckHloBuffersShareAllocation(
              conditional->operand(j + 1),
              conditional->branch_computation(j)->parameter_instruction(0),
              index, buffer_assignment);
        }));
  }
  return Status::OK();
}

}  // namespace

std::unique_ptr<Thunk> IrEmitterUnnested::BuildWhileThunk(
    const HloInstruction* hlo) {
  // Check that all while-related buffers share an allocation.
  TF_CHECK_OK(CheckWhileBuffersShareAllocation(
      hlo, ir_emitter_context_->buffer_assignment()));

  // Generate thunk sequence for while 'condition'.
  HloComputation* condition = hlo->while_condition();
  IrEmitterUnnested ir_emitter_condition(hlo_module_config_, condition,
                                         ir_emitter_context_);
  TF_CHECK_OK(condition->Accept(&ir_emitter_condition));

  // Generate thunk sequence for while 'body'.
  HloComputation* body = hlo->while_body();
  IrEmitterUnnested ir_emitter_body(hlo_module_config_, body,
                                    ir_emitter_context_);
  TF_CHECK_OK(body->Accept(&ir_emitter_body));

  return absl::make_unique<WhileThunk>(
      GetAllocationSlice(*condition->root_instruction()),  // cond result
      ir_emitter_condition.ConsumeThunkSequence(),
      ir_emitter_body.ConsumeThunkSequence(), hlo);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildForThunk(
    const HloInstruction* hlo, const int64 loop_limit) {
  // Check that all while-related buffers share an allocation.
  TF_CHECK_OK(CheckWhileBuffersShareAllocation(
      hlo, ir_emitter_context_->buffer_assignment()));

  // Generate thunk sequence for while 'body' (will be used a For loop body).
  HloComputation* body = hlo->while_body();
  IrEmitterUnnested ir_emitter_body(hlo_module_config_, body,
                                    ir_emitter_context_);
  TF_CHECK_OK(body->Accept(&ir_emitter_body));

  return absl::make_unique<ForThunk>(
      loop_limit, ir_emitter_body.ConsumeThunkSequence(), hlo);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildConditionalThunk(
    const HloInstruction* hlo) {
  // Check that the buffers used in conditional are shared with the operands and
  // result appropriately.
  TF_CHECK_OK(CheckConditionalBuffersShareAllocation(
      hlo, ir_emitter_context_->buffer_assignment()));

  std::vector<BufferAllocation::Slice> branch_operands;
  std::vector<ThunkSequence> branch_thunks;
  for (int j = 0; j < hlo->branch_count(); ++j) {
    branch_operands.emplace_back(GetAllocationSlice(*hlo->operand(j + 1)));
    HloComputation* branch_computation = hlo->branch_computation(j);
    IrEmitterUnnested ir_emitter(hlo_module_config_, branch_computation,
                                 ir_emitter_context_);
    TF_CHECK_OK(branch_computation->Accept(&ir_emitter));
    branch_thunks.push_back(std::move(*ir_emitter.ConsumeThunkSequence()));
  }

  return absl::make_unique<ConditionalThunk>(
      GetAllocationSlice(*hlo->operand(0)), branch_operands,
      std::move(branch_thunks), hlo);
}

Status IrEmitterUnnested::EmitTargetElementLoopInThunk(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator, KernelThunk* thunk) {
  int unroll_factor = thunk->unroll_factor();
  VLOG(3) << bindings_.ToString();

  bool multi_output = hlo.shape().IsTuple();

  const Shape& element_shape =
      multi_output ? ShapeUtil::GetSubshape(hlo.shape(), {0}) : hlo.shape();
  VLOG(3) << "EmitTargetElementLoopInThunk "
          << ShapeUtil::HumanStringWithLayout(hlo.shape())
          << " for unroll_factor " << unroll_factor;
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      element_shape, ir_emitter_context_->device_description(), unroll_factor);
  UpdateLaunchDimensions(launch_dimensions, thunk,
                         ir_emitter_context_->llvm_module());
  if (!multi_output) {
    return ParallelLoopEmitter(element_generator, GetIrArray(hlo, hlo),
                               launch_dimensions, &b_, unroll_factor)
        .EmitLoop(
            IrName(&hlo),
            GetIndexTypeForKernel(&hlo, launch_dimensions.launch_bound(), &b_));
  }

  // Emit the tuple pointers in one thread.  We could do this at any point in
  // the kernel, but we do it at the beginning in the hopes of reducing register
  // pressure, since we touch threadIdx.x and blockIdx.x at the beginning of the
  // kernel *anyway*.
  std::vector<IrArray> output_arrays = ConstructIrArrayForOutputs(hlo);
  KernelSupportLibrary{&b_}.If("emit_mof_tuple", IsBlock0Thread0(&b_), [&] {
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), output_arrays, &b_);
  });

  // For multioutput fusion, we need to emit each operand and the root.
  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(element_generator, output_arrays, launch_dimensions,
                          &b_, unroll_factor)
          .EmitLoop(IrName(&hlo),
                    GetIndexTypeForKernel(
                        &hlo, launch_dimensions.launch_bound(), &b_)));

  b_.SetInsertPoint(b_.GetInsertBlock()->getTerminator());
  return Status::OK();
}

namespace {

// Returns true if the fusion contains any instruction that is likely
// translated to complex LLVM IR, such as loops, and prevent vectorization.
bool MayPreventVectorization(const HloInstruction& fusion_hlo) {
  CHECK_EQ(fusion_hlo.opcode(), HloOpcode::kFusion);
  return absl::c_any_of(
      fusion_hlo.fused_instructions_computation()->instructions(),
      [&](const HloInstruction* instr) {
        switch (instr->opcode()) {
          case HloOpcode::kReduce:
          case HloOpcode::kReduceWindow:
          case HloOpcode::kSort:
          case HloOpcode::kDot:
            return true;
          default:
            return false;
        }
      });
}

}  // namespace

Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  int unroll_factor = 1;
  // Unfused elementwise operations are usually memory bound, unroll them.
  if (hlo.IsElementwise() ||
      (hlo.opcode() == HloOpcode::kFusion && !MayPreventVectorization(hlo))) {
    unroll_factor = ComputeMaxUnrollFactor(&hlo);
  }

  std::unique_ptr<KernelThunk> kernel_thunk = BuildKernelThunk(
      &hlo, /*implements_whole_instruction=*/true, unroll_factor);
  Status emit_status =
      EmitTargetElementLoopInThunk(hlo, element_generator, kernel_thunk.get());
  thunk_sequence_->emplace_back(std::move(kernel_thunk));

  return emit_status;
}

// Emits code to process up to
// (tile_size_x/num_threads_x * tile_size_y/num_threads_y) elements in a tile,
// given `emit_elem_function` is the function to emit code to process one
// element, `y` and `x` are the intra-tile coordinates for the first element
// to process, and `index` is the index for the origin of the tile. Information
// about tile_size_x/y and num_threads_x/y are stored in `mapping_scheme`. Emits
// bounds check to ensure that each processed element is within the boundary
// defined by `tile_width` and `tile_height`.
//
// Pseudocode:
//
// for (y_loc = 0; y_loc < tile_height; y_loc += num_threads_y) {
//   for (j = 0; j < tile_size_x / num_threads_x; j++) { // unrolled
//     if (dilated) {
//       x_loc = x + j * num_threads_x;
//     } else {
//       x_loc = x * (tile_size_x / num_threads_x) + j;
//     }
//
//     if (x_loc < tile_width) {
//       emit_elem_function(y + y_loc, x_loc);
//     }
//   }
// }
//
static void EmitTile(
    const KernelMappingScheme& mapping_scheme,
    const IrArray::Index& tile_origin_index, const string& loop_name,
    KernelSupportLibrary* ksl, llvm::IRBuilder<>* b, llvm::Value* y,
    llvm::Value* x, llvm::Value* tile_height, llvm::Value* tile_width,
    const IrEmitterUnnested::EmitElementFunction& emit_elem_function) {
  llvm::Type* index_ty = tile_width->getType();
  auto constant = [&](int64 val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  int64 num_threads_x = mapping_scheme.GetNumThreadsX();
  int64 num_threads_y = mapping_scheme.GetNumThreadsY();
  int64 tile_size_x = mapping_scheme.GetTileSizeX();

  int64 x_num_steps = tile_size_x / num_threads_x;
  llvm::Value* start_offset_x;
  int64 step_x;

  if (mapping_scheme.DilatedX()) {
    // Using dilated mapping scheme, each thread steps with a stride of number
    // of threads.
    start_offset_x = x;
    step_x = num_threads_x;
  } else {
    // Otherwise, the stride is one, but we multiply each offset by the limit of
    // number of steps which can be made.
    start_offset_x = b->CreateMul(x, constant(x_num_steps));
    step_x = 1;
  }

  IrArray::Index source_idx = tile_origin_index.AddOffsetToDim(
      start_offset_x, KernelMappingScheme::DimX, b);

  ksl->For(
      loop_name + "_y_in_tile",
      /*start=*/y,
      /*end=*/tile_height,
      /*step=*/constant(num_threads_y), [&](llvm::Value* y_loc) {
        IrArray::Index source_idx_y =
            source_idx.AddOffsetToDim(y_loc, KernelMappingScheme::DimY, b);
        for (int64 j = 0; j < x_num_steps; j++) {
          llvm::Value* x_loc =
              b->CreateAdd(constant(j * step_x), start_offset_x, "x_loc");
          IrArray::Index source_idx_x = source_idx_y.AddOffsetToDim(
              constant(j * step_x), KernelMappingScheme::DimX, b);
          // The if-statement below always evaluates to true for the blocks
          // where the entire processed tile fits within the input buffer.
          ksl->If(loop_name + "_x_in_tile", b->CreateICmpULT(x_loc, tile_width),
                  [&] { emit_elem_function(source_idx_x, y_loc, x_loc, j); });
        }
      });
}

// Emits code to process a tensor element in a tile for the given kCopy HLO that
// performs a 0-2-1 transpose.
//
// index: The index for the first output element in the normalized tensor. The
//   normalized tensor is the resulting tensor after collapsing contiguous
//   dimensions that play the same role in the transpose.
// y_loc: The y coordinate within a tile.
// x_loc: The x coordinate within a tile.
// mapping_scheme: Kernel mapping scheme specifying the tiling
void IrEmitterUnnested::EmitTileElementForCopy(
    HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
    const KernelMappingScheme& mapping_scheme, llvm::Value* y_loc,
    llvm::Value* x_loc, int64 /*x_iter_num*/,
    absl::Span<llvm::Value* const> param_shmem_buffers) {
  // TODO(jlebar): Add AA metadata to this load.
  llvm::Instruction* load_from_shmem_buffer =
      Load(GEP(param_shmem_buffers[0], {b_.getInt64(0), x_loc, y_loc}),
           "output_element");
  llvm_ir::IrArray output_array = GetIrArray(*hlo, *hlo);
  Shape output_reduced_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      hlo->shape().element_type(), mapping_scheme.GetDimsInElems());
  // When the output_reduced_shape is a 0-2-1 transpose of the input shape,
  // the 0-2-1 transpose is achieved through EmitWriteArrayElement.
  output_array.CastToShape(output_reduced_shape, &b_)
      .EmitWriteArrayElement(index, load_from_shmem_buffer, &b_);
}

static IrArray::Index GetUnnormalizedIndex(
    const IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* b_,
    const KernelMappingScheme& kernel_mapping_scheme) {
  DCHECK_EQ(normalized_shape_index.size(), 3);
  llvm::Value* linear = normalized_shape_index.Linearize(
      kernel_mapping_scheme.GetDimsInElems(), b_);
  return IrArray::Index(linear, unnormalized_shape, b_);
}

// Emits code to process a tensor element in a tile for the given kLoop fusion
// HLO containing parameters that are 0-2-1 transpose of its outputs.
//
// index: The index for the first output element in the normalized tensor, that
//   is the resulting tensor after collapsing contiguous dimensions that play
//   the same role in the transpose.
// kernel_info: Other information to support the kernel code generation.
// y_loc: The y coordinate within a tile.
// x_loc: The x coordinate within a tile.
void IrEmitterUnnested::EmitTileElementForFusion(
    HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
    const KernelMappingScheme& mapping_scheme, llvm::Value* y_loc,
    llvm::Value* x_loc, int64 /*x_iter_num*/,
    absl::Span<llvm::Value* const> param_shmem_buffers) {
  std::vector<IrArray> output_arrays = ConstructIrArrayForOutputs(*hlo);
  GpuElementalIrEmitter elem_emitter(hlo_module_config_, module_, &b_,
                                     GetNestedComputer());
  FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(hlo),
                               &elem_emitter, x_loc, y_loc,
                               param_shmem_buffers);

  TF_CHECK_OK(hlo->fused_expression_root()->Accept(&fused_emitter));
  IrArray::Index untiled_index = GetUnnormalizedIndex(
      index, output_arrays[0].GetShape(), &b_, mapping_scheme);
  const llvm_ir::ElementGenerator& output_generator =
      fused_emitter.GetRootGenerator();
  llvm::Value* output_value = output_generator(untiled_index).ValueOrDie();
  if (hlo->IsMultiOutputFusion()) {
    DCHECK(output_value->getType()->isStructTy());
    DCHECK_EQ(output_value->getType()->getStructNumElements(),
              output_arrays.size());
    for (int64 i = 0; i < output_arrays.size(); ++i) {
      output_arrays[i].EmitWriteArrayElement(
          untiled_index, ExtractValue(output_value, i), &b_);
    }
  } else {
    output_arrays[0].EmitWriteArrayElement(untiled_index, output_value, &b_);
  }
}

static int GetNumberOfPartialResults(
    const ReductionCodegenInfo& reduction_info) {
  const KernelMappingScheme& mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  if (reduction_info.IsRowReduction()) {
    return 1;
  }
  int64 num_partial_results = mapping_scheme.DilatedX() ? 1 : 2;
  CHECK_EQ(num_partial_results,
           (mapping_scheme.GetTileSizeX() / mapping_scheme.GetNumThreadsX()));
  return num_partial_results;
}

void IrEmitterUnnested::EmitPrologueForOneReduction(
    HloInstruction* unnested_hlo, HloInstruction* reduce_inst, int reduce_idx,
    ReductionCodegenInfo* reduction_info,
    GpuElementalIrEmitter* elemental_emitter) {
  AddressVector* reduction_input_addresses =
      reduction_info->GetMutableReductionInputAddresses();
  llvm::Type* element_type = llvm_ir::PrimitiveTypeToIrType(
      reduce_inst->shape().element_type(), ir_emitter_context_->llvm_module());
  llvm::AllocaInst* reduction_input_address = Alloca(element_type);
  reduction_input_addresses->push_back(reduction_input_address);

  int num_partial_results = GetNumberOfPartialResults(*reduction_info);
  AddressVector* partial_result_addresses =
      reduction_info->GetMutablePartialResultAddresses();
  llvm::AllocaInst* partial_result_address =
      Alloca(element_type, /*ArraySize=*/b_.getInt32(num_partial_results),
             "partial_reduction_result." + llvm::Twine(reduce_idx));
  partial_result_addresses->push_back(partial_result_address);

  // Initialize the partial result with the initial value of the reduction.
  llvm::Value* init_ir_value;
  const HloInstruction* init_value = reduce_inst->operand(1);
  if (unnested_hlo->opcode() == HloOpcode::kFusion) {
    FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(unnested_hlo),
                                 elemental_emitter);

    TF_CHECK_OK(init_value->Accept(&fused_emitter));
    init_ir_value =
        fused_emitter.GetGenerator(init_value)(IrArray::Index(b_.getInt32Ty()))
            .ValueOrDie();
  } else {
    init_ir_value =
        GetIrArray(*init_value, *unnested_hlo)
            .EmitReadArrayElement(IrArray::Index(b_.getInt32Ty()), &b_);
  }

  for (int i = 0; i < num_partial_results; ++i) {
    Store(init_ir_value, InBoundsGEP(partial_result_address, {b_.getInt32(i)}));
  }
}

void IrEmitterUnnested::EmitPrologueForReduction(
    HloInstruction* unnested_hlo, ReductionCodegenInfo* reduction_info,
    absl::Span<HloInstruction* const> reduce_instructions,
    llvm::Type* index_type) {
  VLOG(10) << "Emit prologue for reduction: " << unnested_hlo->ToString();
  GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                          ir_emitter_context_->llvm_module(),
                                          &b_, GetNestedComputer());
  const HloInstruction* first_reduce = nullptr;
  for (int i = 0; i < reduce_instructions.size(); i++) {
    HloInstruction* reduce_inst = reduce_instructions[i];
    VLOG(10) << "Emit prologue for reduction: " << reduce_inst->ToString();
    if (first_reduce == nullptr) {
      first_reduce = reduce_inst;
    } else {
      CHECK(first_reduce->dimensions() == reduce_inst->dimensions());
    }
    EmitPrologueForOneReduction(unnested_hlo, reduce_inst, i, reduction_info,
                                &elemental_emitter);
  }

  int num_partial_results = GetNumberOfPartialResults(*reduction_info);

  // Allocate stack storage to store the linear indices for the current output,
  // and record the address of the storage.
  reduction_info->SetCurrentOutputLinearIndexAddress(
      Alloca(index_type,
             /*ArraySize=*/b_.getInt32(num_partial_results),
             "current_output_linear_index_address"));

  if (!reduction_info->IsRowReduction()) {
    llvm::Type* bool_ty = b_.getInt1Ty();
    llvm::AllocaInst* output_inbound_addr = Alloca(bool_ty);
    Store(llvm::ConstantInt::get(bool_ty, 0), output_inbound_addr);
    reduction_info->SetCurrentOutputInboundAddress(output_inbound_addr);
  }
}

void IrEmitterUnnested::EmitFullWarpShuffleDownLoopForAllReduces(
    absl::Span<HloComputation* const> reducers,
    absl::Span<llvm::AllocaInst* const> partial_result_addresses) {
  for (int distance = 16; distance >= 1; distance /= 2) {
    for (int i = 0; i != reducers.size(); ++i) {
      llvm::Type* element_type =
          partial_result_addresses[i]->getType()->getElementType();
      int bit_width = llvm_ir::GetSizeInBits(element_type);
      llvm::Value* result_from_other_lane = Alloca(
          element_type, nullptr, "result_from_other_lane" + llvm::Twine(i));
      // Bitcast cannot be applied to aggregate types (even packed ones), so
      // we bitcast addresses of load/store to intN* of the same bit-width.
      llvm::Type* shuffled_value_type =
          element_type->isStructTy() ? b_.getIntNTy(bit_width) : element_type;
      auto convert_pointer_for_shuffle = [&](llvm::Value* ptr) {
        return BitCast(ptr, shuffled_value_type->getPointerTo());
      };
      llvm::Value* partial_result =
          Load(convert_pointer_for_shuffle(partial_result_addresses[i]),
               "partial_reduction_result");
      Store(EmitFullWarpShuffleDown(partial_result, b_.getInt32(distance), &b_),
            convert_pointer_for_shuffle(result_from_other_lane));
      TF_CHECK_OK(EmitCallToNestedComputation(
          *reducers[i], {partial_result_addresses[i], result_from_other_lane},
          partial_result_addresses[i]));
    }
  }
}

void IrEmitterUnnested::EmitEpilogueForReduction(
    HloInstruction* unnested_hlo, const ReductionCodegenInfo& reduction_info,
    absl::Span<const HloInstruction* const> reduce_instructions,
    absl::Span<const ShapeIndex> reduction_output_shape_indices,
    absl::Span<HloComputation* const> reducers) {
  const KernelMappingScheme& mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  llvm::Type* index_ty = b_.getInt32Ty();
  auto constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  llvm::Value* thread_id =
      EmitThreadId(mapping_scheme.GetThreadsPerBlock(), index_ty);
  llvm::Value* lane_id =
      b_.CreateURem(thread_id, constant(kWarpSize), "lane_id");

  int num_reduces = reducers.size();
  absl::Span<llvm::AllocaInst* const> partial_result_addresses =
      reduction_info.GetPartialResultAddresses();
  if (reduction_info.IsRowReduction()) {
    EmitFullWarpShuffleDownLoopForAllReduces(reducers,
                                             partial_result_addresses);
    llvm_ir::LlvmIfData if_lane_id_is_zero_data = llvm_ir::EmitIfThenElse(
        ICmpEQ(lane_id, constant(0)), "lane_id_is_zero", &b_);
    llvm_ir::SetToFirstInsertPoint(if_lane_id_is_zero_data.true_block, &b_);
  } else {
    llvm::Value* output_inbound_addr =
        reduction_info.GetCurrentOutputInboundAddress();
    llvm::Value* output_inbound = Load(output_inbound_addr);
    llvm_ir::LlvmIfData if_output_inbound_data = llvm_ir::EmitIfThenElse(
        ICmpEQ(output_inbound,
               llvm::ConstantInt::get(output_inbound->getType(), 1)),
        "output_inbound", &b_);
    llvm_ir::SetToFirstInsertPoint(if_output_inbound_data.true_block, &b_);
  }

  int num_partial_results = GetNumberOfPartialResults(reduction_info);

  // Emit an atomic operation that accumulates the partial reduction to the
  // output element. For row reduction, this is only for lane 0 due to the
  // if-statement emitted above.
  for (int i = 0; i != num_reduces; ++i) {
    const HloInstruction* reduce_hlo = reduce_instructions[i];
    Shape reduction_kept_element_shape = ShapeUtil::FilterDimensions(
        [&](int64 dim) {
          return !absl::c_linear_search(reduce_hlo->dimensions(), dim);
        },
        reduce_hlo->operand(0)->shape());
    for (int j = 0; j < num_partial_results; ++j) {
      // A reduction is allowed to transpose its output.  For example, suppose
      // we are reducing the second dimension of f32[10,20,30]{3,2,1}.  We are
      // allowed to produce as output either f32[10,30]{1,0} (no transpose) or
      // f32[10,30]{0,1} (transposing the two output dims).
      //
      // At this point in the function we have a "partial sum" of input elements
      // (stored in partial_result_addresses), and we need to accumulate it into
      // the correct output element.
      //
      // *reduction_info->GetCurrentOutputLinearIndexAddress() stores the linear
      // index in the output into which we would need to accumulate *if the
      // output layout matched the input layout*. This is why we use
      // `reduction_kept_element_shape` rather than `unnested_hlo->shape()` when
      // computing `element_index` below.
      auto output_array = GetIrArray(*unnested_hlo, *unnested_hlo,
                                     reduction_output_shape_indices[i]);
      IrArray::Index element_index(
          /*linear=*/Load(
              InBoundsGEP(reduction_info.GetCurrentOutputLinearIndexAddress(),
                          {constant(j)}),
              "untransposed_output_linear_addr"),
          reduction_kept_element_shape, &b_);
      IrArray::Index output_index(element_index.multidim(),
                                  output_array.GetShape(),
                                  element_index.GetType());
      llvm::Value* output_address = output_array.EmitArrayElementAddress(
          output_index, &b_, "output_element_address");
      TF_CHECK_OK(EmitAtomicOperationForNestedComputation(
          *reducers[i], output_address,
          InBoundsGEP(partial_result_addresses[i], {constant(j)})));
    }
  }
}

// Given the IrArray index of a reduction input, returns the linear address of
// the reduction output as if the reduction were going to keep the input
// shape with the dimensions being reduced moved.
static llvm::Value* GetUntransposedOutputLinearAddress(
    llvm::IRBuilder<>* b, const llvm_ir::IrArray::Index& index,
    const ReductionCodegenInfo& reduction_info) {
  const KernelMappingScheme& kernel_mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  if (reduction_info.IsRowReduction()) {
    return index[KernelMappingScheme::DimY];
  }
  absl::Span<const int64> dims_in_elem = kernel_mapping_scheme.GetDimsInElems();
  llvm::Value* x_dim_size =
      index.GetConstantWithIndexType(dims_in_elem[KernelMappingScheme::DimX]);
  llvm::Value* x_block_offset =
      b->CreateMul(index[KernelMappingScheme::DimZ], x_dim_size);
  return b->CreateAdd(x_block_offset, index[KernelMappingScheme::DimX]);
}

void IrEmitterUnnested::EmitTileElementForReduction(
    HloInstruction* unnested_hlo, const Shape& reduction_operand_shape,
    absl::Span<HloInstruction* const> output_instructions,
    const llvm_ir::IrArray::Index& index,
    const ReductionCodegenInfo& reduction_info,
    absl::Span<HloComputation* const> reducers, int64 x_iter_num) {
  VLOG(10) << "Emit tile element for reduce " << unnested_hlo->ToString();
  bool returns_tuple = output_instructions.size() > 1;
  // Record the untransposed output linear address for the reduction.
  int partial_result_index = reduction_info.IsRowReduction() ? 0 : x_iter_num;
  b_.CreateStore(
      GetUntransposedOutputLinearAddress(&b_, index, reduction_info),
      InBoundsGEP(reduction_info.GetCurrentOutputLinearIndexAddress(),
                  {b_.getInt32(partial_result_index)}));

  if (!reduction_info.IsRowReduction()) {
    llvm::Type* bool_ty = b_.getInt1Ty();
    llvm::AllocaInst* output_inbound_addr =
        reduction_info.GetCurrentOutputInboundAddress();
    Store(llvm::ConstantInt::get(bool_ty, 1), output_inbound_addr);
  }

  InlinedVector<llvm_ir::ElementGenerator, 1> input_gens;
  std::vector<std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
      extra_output_gens;
  GpuElementalIrEmitter elem_emitter(hlo_module_config_, module_, &b_,
                                     GetNestedComputer());
  FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(unnested_hlo),
                               &elem_emitter);
  // Construct the ElementGenerator for each reduction and extra output in the
  // the group of output instructions.
  if (unnested_hlo->opcode() == HloOpcode::kFusion) {
    TF_CHECK_OK(unnested_hlo->fused_expression_root()->Accept(&fused_emitter));

    for (int i = 0, e = output_instructions.size(); i != e; ++i) {
      const HloInstruction* inst = output_instructions[i];
      ShapeIndex idx = returns_tuple ? ShapeIndex({i}) : ShapeIndex({});
      if (IsReductionFromOrToContiguousDimensions(*inst)) {
        input_gens.push_back(fused_emitter.GetGenerator(inst->operand(0)));
      } else {
        extra_output_gens.emplace_back(fused_emitter.GetGenerator(inst),
                                       std::move(idx));
      }
    }
  } else {
    input_gens.push_back([&](const IrArray::Index& index) {
      return GetIrArray(*unnested_hlo->operand(0), *unnested_hlo)
          .EmitReadArrayElement(index, &b_);
    });
  }

  IrArray::Index input_index =
      GetUnnormalizedIndex(index, reduction_operand_shape, &b_,
                           reduction_info.GetKernelMappingScheme());
  // Clear the linear index field of the IrArray::Index to enable the use of
  // GetElementPointer with array types. This enables the vectorization of
  // the computation for different partial results. Use this index if
  // 'num_partial_results > 1'.
  int num_partial_results = GetNumberOfPartialResults(reduction_info);
  auto index_without_linear = IrArray::Index(
      input_index.multidim(), reduction_operand_shape, input_index.GetType());

  // Emit code to generate the input and perform the reduction computation for
  // each reduction instruction.
  for (int i = 0; i != reducers.size(); ++i) {
    llvm::AllocaInst* input_address =
        reduction_info.GetReductionInputAddresses()[i];
    llvm::AllocaInst* partial_reduction_result_address =
        reduction_info.GetPartialResultAddresses()[i];
    llvm::Value* const input_ir_value =
        input_gens[i](num_partial_results > 1 ? index_without_linear
                                              : input_index)
            .ValueOrDie();
    Store(input_ir_value, input_address);
    llvm::Value* partial_result_address = InBoundsGEP(
        partial_reduction_result_address, {b_.getInt32(partial_result_index)});
    TF_CHECK_OK(EmitCallToNestedComputation(
        *reducers[i], {partial_result_address, input_address},
        partial_result_address));
  }

  // Emit code to generate the output for the non-reduction instructions in the
  // fusion, if any.
  TF_CHECK_OK(EmitExtraOutputsForReduce(
      unnested_hlo, input_index,
      /*use_linear_index=*/num_partial_results == 1, extra_output_gens));
}

// Returns the index for the first element in the tile with the given tile
// index.
static IrArray::Index GetElementIndexForTileOrigin(
    const IrArray::Index& tile_index, const KernelMappingScheme& mapping_scheme,
    llvm::IRBuilder<>* b_) {
  std::vector<llvm::Value*> elem_multi_index = tile_index.multidim();
  for (int i = KernelMappingScheme::DimY; i < KernelMappingScheme::DimTot;
       ++i) {
    elem_multi_index[i] =
        b_->CreateMul(tile_index[i],
                      llvm::ConstantInt::get(tile_index[i]->getType(),
                                             mapping_scheme.GetTileSizeFor(i)),
                      "tile_origin." + std::to_string(i));
  }
  return IrArray::Index(elem_multi_index, mapping_scheme.GetDimsInElems(),
                        tile_index.GetType());
}

llvm::Value* IrEmitterUnnested::EmitThreadId(int64 threads_per_block,
                                             llvm::Type* index_ty) {
  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::CallInst* thread_id_raw = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kThreadIdx, {}, {}, &b_);
  llvm_ir::AddRangeMetadata(0, threads_per_block, thread_id_raw);
  return b_.CreateIntCast(thread_id_raw, index_ty,
                          /*isSigned=*/true, "thread.id.x");
}

void IrEmitterUnnested::EmitTilingKernel(
    const KernelMappingScheme& mapping_scheme, llvm::Type* index_ty,
    const TileElementGenerator& tile_element_generator) {
  absl::Span<const int64> dims_in_elems = mapping_scheme.GetDimsInElems();
  std::vector<int64> dims_in_blocks = {
      CeilOfRatio(dims_in_elems[0], mapping_scheme.GetTileSizeZ()),
      CeilOfRatio(dims_in_elems[1], mapping_scheme.GetTileSizeY()),
      CeilOfRatio(dims_in_elems[2], mapping_scheme.GetTileSizeX())};
  auto constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  llvm::Value* thread_id =
      EmitThreadId(mapping_scheme.GetThreadsPerBlock(), index_ty);
  llvm::Value* num_thread_x = constant(mapping_scheme.GetNumThreadsX());

  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::Value* x = b_.CreateURem(thread_id, num_thread_x, "thread.x");
  llvm::Value* y = b_.CreateUDiv(thread_id, num_thread_x, "thread.y");

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);

  // Calculate the starting tile.
  const IrArray::Index starting_tile = [&] {
    llvm::Value* block_id = gpu::EmitCallToTargetIntrinsic(
        gpu::TargetIntrinsicID::kBlockIdx, {}, {}, &b_);
    llvm_ir::AddRangeMetadata(0, mapping_scheme.GetNumberOfBlocks(),
                              llvm::cast<llvm::Instruction>(block_id));
    llvm::Value* linear_block_id =
        b_.CreateIntCast(block_id, index_ty, /*isSigned=*/true, "block.id.x");
    IrArray::Index starting_block(linear_block_id,
                                  ShapeUtil::MakeShapeWithDescendingLayout(
                                      PRED /*arbitrary*/, dims_in_blocks),
                                  &b_);

    std::vector<llvm::Value*> multidim = {
        b_.CreateMul(starting_block[0], constant(mapping_scheme.GetTileSizeZ()),
                     "block_origin.z"),
        starting_block[1], starting_block[2]};
    return IrArray::Index(multidim, dims_in_blocks, index_ty);
  }();

  auto emit_tile = [&](const IrArray::Index& tile_index) {
    std::vector<llvm::Value*> output_tile_bounds(3);
    for (int i = KernelMappingScheme::DimY; i < KernelMappingScheme::DimTot;
         ++i) {
      int64 tile_size_for_dim = mapping_scheme.GetTileSizeFor(i);
      // Only last row or column may not have full size.
      llvm::Value* is_last_row =
          b_.CreateICmpEQ(tile_index[i], constant(dims_in_blocks[i] - 1));
      int64 partial_row_size =
          dims_in_elems[i] - (dims_in_blocks[i] - 1) * tile_size_for_dim;
      output_tile_bounds[i] =
          b_.CreateSelect(is_last_row, constant(partial_row_size),
                          constant(tile_size_for_dim), "tile_bound");
    }
    IrArray::Index tile_origin =
        GetElementIndexForTileOrigin(tile_index, mapping_scheme, &b_);
    tile_element_generator(y, x, tile_origin, "output", output_tile_bounds[1],
                           output_tile_bounds[2], &ksl);
  };

  int dim_z = KernelMappingScheme::DimZ;
  if (mapping_scheme.GetTileSizeZ() == 1) {
    emit_tile(starting_tile);
  } else {
    llvm::Value* starting_tile_index_for_dim = starting_tile[dim_z];
    llvm::Value* block_size_for_dim = constant(mapping_scheme.GetTileSizeZ());
    llvm::Value* block_id_for_dim =
        b_.CreateUDiv(starting_tile_index_for_dim, block_size_for_dim);
    llvm::Value* last_block_for_dim = constant(dims_in_blocks[dim_z] - 1);
    llvm::Value* last_block_size_for_dim =
        constant(dims_in_elems[dim_z] -
                 (dims_in_blocks[dim_z] - 1) * mapping_scheme.GetTileSizeZ());

    llvm::Value* num_tiles_in_block =
        b_.CreateSelect(b_.CreateICmpEQ(last_block_for_dim, block_id_for_dim),
                        last_block_size_for_dim, block_size_for_dim);
    ksl.For("loop_z",
            /*start=*/constant(0),
            /*end=*/num_tiles_in_block,
            /*step=*/1, [&](llvm::Value* block_dim_induction_var) {
              IrArray::Index tile_index = starting_tile.AddOffsetToDim(
                  block_dim_induction_var, dim_z, &b_);
              emit_tile(tile_index);
            });
  }
}

// Emits a kernel for the given hlo instruction using a tiled 0-2-1 transpose
// algorithm to improve the memory access patterns for the input parameters
// with a shape that is a 0-2-1 transpose of the output tensor shape. The caller
// is responsible for making sure that it is safe to apply the shared memory
// transpose on the input parameters.
//
//
// For the purpose of tiling, the output tensors have a logical shape of three
// components 0-2-1 while the relevant input parameters have a logical shape
// of three components 0-1-2 in the order major to minor. The x- and y-
// dimensions of the tensors are tiled in square tiles with an edge length
// `kTileSize`. Each thread block of `kTileSize` x `kNumRows` threads
// transposes one tile: each thread copies kTileSize/kNumRows elements from
// the input to a shared memory tile, then the otherwise "regular HLO kernel"
// reads from the shared memory instead of the original input.
//
// This is similar to the following CUDA algorithm in TensorFlow:
// https://goo.gl/MStRV6.
//
// `kTileSize` should usually be same as warp size. We currently choose 32 for
// `kTileSize` and 4 for `kNumRows`. The CUDA algorithm uses 8 for `kNumRows`.
//
// TODO(b/33320379): Here each block transposes 1 tile. It may be more
// efficient to launch fewer blocks so each transposes many tiles.
void IrEmitterUnnested::EmitHlo021Tile(
    HloInstruction* hlo, Thunk* kernel_thunk,
    absl::Span<const int64> reduced_output_dims,
    absl::Span<const int64> tiled_param_ids) {
  constexpr int kNumRows = 4;
  KernelMappingScheme mapping_scheme(reduced_output_dims,
                                     /*tile_sizes=*/{1, kWarpSize, kWarpSize},
                                     /*num_threads_y=*/kNumRows,
                                     /*num_threads_x=*/kWarpSize,
                                     /*is_dilated_x=*/false);
  LaunchDimensions launch_dimensions(mapping_scheme.GetNumberOfBlocks(),
                                     mapping_scheme.GetThreadsPerBlock());
  llvm::Type* index_type =
      GetIndexTypeForKernel(hlo, launch_dimensions.launch_bound(), &b_);
  std::vector<IrArray> param_arrays;

  // For each tiled parameter, cast its input IrArray to the corresponding
  // reduced shape and keep the reduced shape live during IR emission.
  std::vector<IrArray> param_in_reduced_shape_arrays;
  std::vector<llvm::Value*> param_shmem_buffers(hlo->operand_count(), nullptr);

  auto get_shared_memory_buffer = [&](llvm::Type* elem_ty,
                                      absl::string_view buffer_name) {
    // For Nvidia GPUs, the warp size is 32 threads and the shared memory bank
    // is organized into 32-way. We usually use the warp size or a multiplier or
    // a the warp size as the size for tiling. This may cause all elements in
    // the same column of a tile use the same memory bank and therefore shared
    // memory bank conflicts. Adding 1 to the minor dimension of the shared
    // memory buffer can reduce such shared memory bank conflicts.
    llvm::Type* buffer_type = llvm::ArrayType::get(
        llvm::ArrayType::get(elem_ty, mapping_scheme.GetTileSizeX() + 1),
        mapping_scheme.GetTileSizeY());
    return llvm_ir::AllocateSharedMemoryTile(b_.GetInsertBlock()->getModule(),
                                             buffer_type, buffer_name);
  };

  for (int64 id = 0; id < hlo->operand_count(); id++) {
    const HloInstruction* param = hlo->operand(id);
    param_arrays.push_back(GetIrArray(*param, *hlo));

    if (absl::c_linear_search(tiled_param_ids, id)) {
      param_shmem_buffers[id] =
          get_shared_memory_buffer(llvm_ir::PrimitiveTypeToIrType(
                                       param->shape().element_type(), module_),
                                   IrName(hlo, StrCat("tile", id)));
      VLOG(3) << "Added shmem buffer for parameter " << id << ": "
              << llvm_ir::DumpToString(*param_shmem_buffers[id]);
      Shape reduced_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          param->shape().element_type(),
          Permute({0, 2, 1}, reduced_output_dims));
      param_in_reduced_shape_arrays.push_back(
          param_arrays[id].CastToShape(reduced_shape, &b_));
    } else {
      param_in_reduced_shape_arrays.push_back(IrArray());
    }
  }

  EmitElementFunction element_generator =
      [&](const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
          llvm::Value* x_loc, int64 x_iter_num) {
        if (hlo->opcode() == HloOpcode::kCopy) {
          EmitTileElementForCopy(hlo, index, mapping_scheme, y_loc, x_loc,
                                 x_iter_num, param_shmem_buffers);
        } else {
          CHECK_EQ(hlo->opcode(), HloOpcode::kFusion);
          EmitTileElementForFusion(hlo, index, mapping_scheme, y_loc, x_loc,
                                   x_iter_num, param_shmem_buffers);
        }
      };

  TileElementGenerator tile_generator =
      [&](llvm::Value* y, llvm::Value* x, const IrArray::Index& index,
          const string& loop_name, llvm::Value* tile_height,
          llvm::Value* tile_width, KernelSupportLibrary* ksl) {
        // If shared memory transpose is needed, wait for all threads to reach
        // this point, lest we copy a value from tile to output before the other
        // thread copies it from input to tile. This is `__syncthreads` in CUDA.
        if (!tiled_param_ids.empty()) {
          // Calculate the input tile origin from the output tile origin.
          const IrArray::Index input_tile_origin(
              Permute({0, 2, 1}, index.multidim()),
              Permute({0, 2, 1}, index.dims()), index.GetType());

          // Copy input parameter values to shared memory buffers:
          // tile[y, x] = input[index]
          // Note that tile_width and tile_height are flipped here because we
          // are reading a transposed tile.
          EmitTile(mapping_scheme, input_tile_origin, "input", ksl, &b_, y, x,
                   tile_width, tile_height,
                   [&](const IrArray::Index& index, llvm::Value* y_loc,
                       llvm::Value* x_loc, int64 /*x_iter_num*/) {
                     for (int64 id : tiled_param_ids) {
                       IrArray& input_in_logical_shape =
                           param_in_reduced_shape_arrays[id];

                       llvm::Value* shmem_buffer = param_shmem_buffers[id];
                       llvm::Value* zero =
                           llvm::ConstantInt::get(index_type, 0);
                       // TODO(jlebar): Add AA metadata to this store.  Tile
                       // buffers are global variables, so LLVM can't infer much
                       // about it.
                       Store(input_in_logical_shape.EmitReadArrayElement(
                                 index, &b_, "input_element"),
                             GEP(shmem_buffer, {zero, y_loc, x_loc}));
                     }
                   });

          // Wait for all threads to reach this point using `__syncthreads` in
          // CUDA.
          EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, &b_);
        }

        EmitTile(mapping_scheme, index, loop_name, ksl, &b_, y, x, tile_height,
                 tile_width, element_generator);
        bool block_contains_multi_tiles = mapping_scheme.GetTileSizeZ() > 1;

        // If a tile block contains multiple tiles and shared memory buffers are
        // used, we need to wait for all threads to finish using the shared
        // memory buffer for the current tile before we move on to process the
        // next tile and overwrite the shared memory buffers.
        if (block_contains_multi_tiles && !tiled_param_ids.empty()) {
          EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, &b_);
        }
      };

  // For multioutput fusion, one thread needs to output a tuple
  // with pointers to all the individual outputs.  We could do this
  // at any point in the kernel, but we do it at the beginning in
  // the hopes of reducing register pressure, since we touch
  // threadIdx.x and blockIdx.x at the beginning of the kernel
  // *anyway*.
  if (hlo->IsMultiOutputFusion()) {
    KernelSupportLibrary{&b_}.If("emit_mof_tuple", IsBlock0Thread0(&b_), [&] {
      llvm_ir::EmitTuple(GetIrArray(*hlo, *hlo),
                         ConstructIrArrayForOutputs(*hlo), &b_);
    });
  }

  EmitTilingKernel(mapping_scheme, index_type, tile_generator);
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk,
                         ir_emitter_context_->llvm_module());
}

namespace {
// A recursive function to inspect the users of a parameter to determine
// whether it's safe for a parameter to participate in a shared-memory
// transpose.
//
// Consider a fusion parameter P for which we might want to use a shmem
// transpose.  If we do, we use a GPU thread block to preload a tile of P with
// indices [z, y..y+31, x..x+31] to compute an output tile with the same indices
// cooperatively, where z, y, x are the indices for the normalized input/output
// tensor (see the document for FindTranspose021 for the definition of
// normalized tensor for 0-2-1 transpose). This shmem transpose implementation
// requires that the computation of the output tile only read elements within
// the preload tile. If this is not true, we can't use a shmem transpose for P.
//
// If the computation of output element [z, y, x] only requires the element of
// P with the same indices, the shmem transpose implementation can be applied
// to P safely. This is a sufficient but not necessary condition. We check all
// the transitive users of P to see if we can find a user that may cause an
// exception to the situation. If such a user is not found, we conclude that P
// is safe for shmem transpose.
//
// This is trivially true for elementwise operations and some "data-movement"
// ops like kTuple. However, it's not true for operations that can change the
// dimensions of the inputs (e.g. pad, slice) and bitcast operation.
// For example:
//
// fused_computation {
//   param_0 = f32[64,64]{1,0} parameter(0)
//   ROOT bitcast = f32[64,64]{0,1} bitcast(param_0)
// }
// The output element at logical address [0, 63] depends on the input element
// at logical address [63, 0], which would not be within the shared-memory
// block.
//
// TODO(bixia): In order to extend this for kInput fusion, that is reduction
// with transpose, we only need to end the use-chain checking with the input of
// a reduce operations. In this case, the above description on "output" apply
// to the result of such a use-chain, which provides the input to the reduce
// operation.
bool IsInstructionSafeForShmemTranspose(const HloInstruction* hlo) {
  if (hlo->IsElementwise()) {
    return absl::c_all_of(hlo->users(), [&](const HloInstruction* user) {
      return IsInstructionSafeForShmemTranspose(user);
    });
  }

  switch (hlo->opcode()) {
    // Non-elementwise instructions that don't cause the shmem transpose
    // to be unsafe, including the instructions that don't currently fuse.
    case HloOpcode::kGetDimensionSize:
      // The result of the operation doesn't rely on the content of the
      // tensor. As such, there is no need to further inspect its users.
      return true;
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    case HloOpcode::kTupleSelect:
      return absl::c_all_of(hlo->users(), [&](const HloInstruction* user) {
        return IsInstructionSafeForShmemTranspose(user);
      });

    default:
      return false;
  }
}

// Given a group of input parameters that are 0-2-1 transpose of the outputs of
// a fusion kernel, returns the input parameters that are safe for the shared
// memory transpose implementation.
//
// When a tile based shared memory transpose is used to implement an input with
// 0-2-1 transpose, we preload a tile of the input elements
// [z, y..y+31, x..x+31] to compute the output tile elements of the same
// indices. Preloading the input tile this way is only safe when the computation
// of the output tile elements do not need any input element outside the
// preloaded tile. We inspect all the transitive users of the input parameter
// up to the fusion root instruction to see if we can find any instruction
// that can make preloading the input tile unsafe.
std::vector<int64> FilterInputsForShmemTranspose(const HloInstruction* fusion,
                                                 std::vector<int64> input_ids) {
  std::vector<int64> filtered_input_ids;
  for (int64 i = 0; i < input_ids.size(); ++i) {
    const HloInstruction* input = fusion->fused_parameter(input_ids[i]);
    if (IsInstructionSafeForShmemTranspose(input)) {
      filtered_input_ids.push_back(input_ids[i]);
    } else {
      VLOG(10) << "Input not safe for shmem transpose " << input->ToString();
    }
  }
  return filtered_input_ids;
}

}  // namespace

bool IrEmitterUnnested::CheckAndEmitHloWithTile021(HloInstruction* hlo) {
  HloOpcode opcode = hlo->opcode();

  CHECK(hlo->IsLoopFusion() || opcode == HloOpcode::kCopy);

  const Shape& output_shape = hlo->IsMultiOutputFusion()
                                  ? ShapeUtil::GetSubshape(hlo->shape(), {0})
                                  : hlo->shape();

  // If the output_shape is reduced to 021 shape, find all the parameters of
  // the HLO that are in the corresponding 012 shape.
  std::vector<int64> params_012;
  optional<std::vector<int64>> reduced_dims_021;
  for (int64 operand_idx = 0; operand_idx < hlo->operand_count();
       ++operand_idx) {
    HloInstruction* operand = hlo->mutable_operand(operand_idx);
    auto find_transpose_result =
        ShapeUtil::FindTranspose021(operand->shape(), output_shape);
    if (!find_transpose_result.has_value()) {
      continue;
    }
    const std::vector<int64>& curr_reduced_dims_021 = *find_transpose_result;
    if (!reduced_dims_021.has_value()) {
      reduced_dims_021 = curr_reduced_dims_021;
    }
    if (!absl::c_equal(*reduced_dims_021, curr_reduced_dims_021)) {
      // There is more than one possible transpose. Instead of picking one
      // transpose, we simply give up here.
      return false;
    }
    params_012.push_back(operand_idx);
  }

  if (!reduced_dims_021.has_value()) {
    return false;
  }

  if ((*reduced_dims_021)[1] < kMinDimensionToTransposeTiled ||
      (*reduced_dims_021)[2] < kMinDimensionToTransposeTiled) {
    return false;
  }

  if (opcode == HloOpcode::kFusion) {
    params_012 = FilterInputsForShmemTranspose(hlo, params_012);
    if (params_012.empty()) {
      return false;
    }
  }

  // Each of our shared memory tiles has 32*33 elements (so ~4kb, if the
  // elements are of size 4 bytes), and CUDA has an architectural limit of
  // 48kb shared memory per SM.  (This is increased to 96kb in Volta, but we
  // don't use this, in part because it eats into our L1 cache space.)
  //
  // For correctness we need to ensure that we don't make more than 48kb worth
  // of shmem tiles per block.  And for performance, we'd probably like to use
  // significantly less, so that we can fit more than one block at a time on a
  // gpu core.
  //
  // We say without benchmarks that we want at least 3 threads/block,
  // corresponding to 3 shmem tiles if the elements are 32 bits wide.  We
  // choose which params get the shmem transpose treatment arbitrarily; it's
  // not clear if there's a Right Choice.
  //
  // This is only sound if tiled transposes are the only place where we use
  // shared memory in fusions.  If in the future other fusible ops use shared
  // memory, we'll have to adjust this heuristic.
  constexpr int kMinBlocksPerCore = 3;
  constexpr int64 kShmemPerCore = 48 * 1024;
  int64 shmem_used = 0;
  for (int64 i = 0; i < params_012.size(); ++i) {
    const HloInstruction* operand = hlo->operand(params_012[i]);
    shmem_used +=
        32 * 33 *
        ShapeUtil::ByteSizeOfPrimitiveType(operand->shape().element_type());

    if (kMinBlocksPerCore * shmem_used > kShmemPerCore) {
      // Erase this element and everything after it from params_012.
      params_012.resize(i);
      break;
    }
  }

  if (params_012.empty()) {
    return false;
  }

  VLOG(3) << "EmitHlo021Tile Emitting hlo tile 0-2-1" << hlo->ToString();
  std::unique_ptr<KernelThunk> kernel_thunk =
      BuildKernelThunk(hlo, /*implements_whole_instruction=*/true);
  EmitHlo021Tile(hlo, kernel_thunk.get(), *reduced_dims_021, params_012);
  AddThunkToThunkSequence(std::move(kernel_thunk));
  return true;
}

namespace {

// Returns true if all the transitive users of hlo before hitting users in
// use_chain_endings are elementwise operations.
bool AreUsersElementwise(const HloInstruction* hlo,
                         const ConstHloInstructionSet& use_chain_endings) {
  return absl::c_all_of(hlo->users(), [&](const HloInstruction* user) {
    return use_chain_endings.count(user) ||
           (user->IsElementwise() &&
            AreUsersElementwise(user, use_chain_endings));
  });
}

// Returns the number of fusion inputs that have the same dimension as the
// given shape, and involve in only elementwise operations.
int64 NumInputsInvolveInOnlyElementwiseOps(
    const HloInstruction* unnested_hlo, const Shape& op_shape,
    const ConstHloInstructionSet& use_chain_endings) {
  return absl::c_count_if(
      unnested_hlo->fused_parameters(), [&](const HloInstruction* parameter) {
        const Shape& parameter_shape = parameter->shape();
        return ShapeUtil::SameDimensions(op_shape, parameter_shape) &&
               AreUsersElementwise(parameter, use_chain_endings);
      });
}

// Returns the number of fusion inputs that have more elements than the given
// shape.
int64 NumInputsWithMoreElementsThan(const HloInstruction* unnested_hlo,
                                    const Shape& shape) {
  int64 num_elements = ShapeUtil::ElementsIn(shape);
  return absl::c_count_if(
      unnested_hlo->fused_parameters(), [&](const HloInstruction* parameter) {
        return ShapeUtil::ElementsIn(parameter->shape()) > num_elements;
      });
}

// The benefit of unrolling a kInput fusion that is a column reduction comes
// from the vectorization of non-reduction fusion outputs and fusion inputs.
// On the other hand, unrolling can also introduce factors that can cause
// the kernel to run slower. This routine uses a simple heuristic to estimate
// the benefit as well as the overhead of unrolling in order to decide whether
// unrolling is beneficial for the given kInput fusion.
bool IsUnrollingColumnReductionBeneficial(const HloInstruction* unnested_hlo,
                                          const Shape& input_shape,
                                          int64 num_kept_minor) {
  // TODO(b/122468062): Need further investigate to see whether we can
  // remove the constraint on IsPowerOfTwo.
  if (!IsPowerOfTwo(static_cast<uint64>(num_kept_minor))) {
    return false;
  }

  if (IsReductionFromOrToContiguousDimensions(*unnested_hlo)) {
    return true;
  }

  CHECK_EQ(unnested_hlo->opcode(), HloOpcode::kFusion);
  int64 can_be_vectorized = 0;
  int64 cannot_be_vectorized = 0;
  const HloInstruction* fused_root = unnested_hlo->fused_expression_root();
  ConstHloInstructionSet use_chain_endings;
  if (IsReductionFromOrToContiguousDimensions(*fused_root)) {
    use_chain_endings.insert(fused_root);
    // Atomic.add of the reduction result can't be vectorized.
    cannot_be_vectorized++;
  } else {
    CHECK_EQ(fused_root->opcode(), HloOpcode::kTuple);
    for (const HloInstruction* instr : fused_root->operands()) {
      if (IsReductionFromOrToContiguousDimensions(*instr)) {
        // Atomic.add of the reduction result can't be vectorized.
        cannot_be_vectorized++;
      } else {
        // Write of the non-reduction result can be vectorized.
        can_be_vectorized++;
      }
      use_chain_endings.insert(instr);
    }
  }
  // Fusion inputs that have the same dimension as the reduce input and
  // only involve in elementwise operations can be vectorized.
  can_be_vectorized += NumInputsInvolveInOnlyElementwiseOps(
      unnested_hlo, input_shape, use_chain_endings);
  // Fusion inputs with more elements than the reduce op input must participate
  // in non-elementwise operations and we assume that they are not vectorizable
  // for the purpose of estimating the benefit of unrolling. If the kernel is
  // unrolled even with such an assumption,  and the accesses to those inputs
  // turn out to be vectorizable, the compiler will still vectorize them.
  cannot_be_vectorized +=
      NumInputsWithMoreElementsThan(unnested_hlo, input_shape);
  return can_be_vectorized >= cannot_be_vectorized;
}

}  // namespace

ReductionCodegenInfo IrEmitterUnnested::ComputeReductionCodegenInfo(
    const HloInstruction* unnested_hlo, const HloInstruction* first_reduce) {
  const Shape& input_shape = first_reduce->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*first_reduce);
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << reduction_dimensions.dimensions[0] << " "
           << reduction_dimensions.dimensions[1] << " "
           << reduction_dimensions.dimensions[2];

  std::array<int64, 3> reduction_tiling =
      GetReductionTiling(reduction_dimensions);
  bool dilated_x =
      reduction_dimensions.is_row_reduction ||
      !IsUnrollingColumnReductionBeneficial(unnested_hlo, input_shape,
                                            reduction_dimensions.dimensions[2]);

  if (!dilated_x && !reduction_dimensions.is_row_reduction) {
    // Vectorized loads: a single thread reduces two adjacent columns.
    reduction_tiling[2] *= 2;
  }

  int64 num_threads_y = 1;
  int64 num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      return kWarpSize;
    }
    return std::min(
        ThreadsPerBlockLimit(ir_emitter_context_->device_description()),
        CeilOfRatio(reduction_dimensions.dimensions[2], reduction_tiling[2]));
  }();

  KernelMappingScheme mapping_scheme(
      reduction_dimensions.dimensions,
      {reduction_tiling[0], reduction_tiling[1] * num_threads_y,
       reduction_tiling[2] * num_threads_x},
      num_threads_y, num_threads_x, dilated_x);
  return ReductionCodegenInfo(mapping_scheme,
                              reduction_dimensions.is_row_reduction);
}

Status IrEmitterUnnested::EmitReductionFromOrToContiguousDimensions(
    HloInstruction* unnested_hlo,
    absl::Span<HloInstruction* const> output_instructions) {
  bool returns_tuple = output_instructions.size() > 1;
  VLOG(10) << "Emitting reduction to vector " << unnested_hlo->ToString();

  std::vector<HloInstruction*> reduce_instructions;
  InlinedVector<ShapeIndex, 1> reduction_output_shape_indices;
  InlinedVector<HloComputation*, 1> reducers;

  // Build an initializer thunk to initialize each reduction output.
  std::vector<std::unique_ptr<Thunk>> thunks;
  for (int i = 0; i < output_instructions.size(); ++i) {
    if (!IsReductionFromOrToContiguousDimensions(*output_instructions[i])) {
      continue;
    }

    HloInstruction* output_instruction = output_instructions[i];
    reduce_instructions.push_back(output_instruction);
    ShapeIndex idx = returns_tuple ? ShapeIndex({i}) : ShapeIndex({});
    reduction_output_shape_indices.push_back(idx);
    reducers.push_back(output_instruction->to_apply());

    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> initializer_thunk,
                        BuildInitializerThunk(unnested_hlo, idx));
    thunks.push_back(std::move(initializer_thunk));
  }

  const HloInstruction* first_reduce = reduce_instructions.at(0);
  if (output_instructions.size() > 1) {
    if (!AreFusedReductionOutputsConsistent(output_instructions,
                                            first_reduce)) {
      return InternalError("Inconsistent reduction fusion outputs");
    }
  }

  // Build a kernel thunk to compute all the outputs.
  std::unique_ptr<KernelThunk> kernel_thunk =
      BuildKernelThunk(unnested_hlo, /*implements_whole_instruction=*/false);

  const Shape& input_shape = first_reduce->operand(0)->shape();
  // The layout of a reduction input is either set by LayoutAssignment for
  // unnested kReduce or by InstructionFusion for fused kReduce.
  CHECK(input_shape.has_layout()) << "LayoutAssignment or InstructionFusion "
                                     "doesn't set the input layout of "
                                  << first_reduce->ToString();

  ReductionCodegenInfo reduction_info =
      ComputeReductionCodegenInfo(unnested_hlo, first_reduce);
  const KernelMappingScheme& mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  LaunchDimensions launch_dimensions(mapping_scheme.GetNumberOfBlocks(),
                                     mapping_scheme.GetThreadsPerBlock());
  llvm::Type* index_ty = GetIndexTypeForKernel(
      unnested_hlo, launch_dimensions.launch_bound(), &b_);
  EmitPrologueForReduction(unnested_hlo, &reduction_info, reduce_instructions,
                           index_ty);
  EmitElementFunction emit_reduction_tile =
      [&](const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
          llvm::Value* x_loc, int64 x_iter_num) {
        EmitTileElementForReduction(unnested_hlo, input_shape,
                                    output_instructions, index, reduction_info,
                                    reducers, x_iter_num);
      };

  EmitTilingKernel(
      mapping_scheme, index_ty,
      /*tile_element_generator=*/
      [&](llvm::Value* y, llvm::Value* x, const IrArray::Index& index,
          const string& loop_name, llvm::Value* tile_height,
          llvm::Value* tile_width, KernelSupportLibrary* ksl) {
        EmitTile(reduction_info.GetKernelMappingScheme(), index, loop_name, ksl,
                 &b_, y, x, tile_height, tile_width, emit_reduction_tile);
      });
  EmitEpilogueForReduction(unnested_hlo, reduction_info, reduce_instructions,
                           reduction_output_shape_indices, reducers);

  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  thunks.push_back(std::move(kernel_thunk));
  auto sequential_thunk =
      absl::make_unique<SequentialThunk>(std::move(thunks), unnested_hlo);
  AddThunkToThunkSequence(std::move(sequential_thunk));

  return Status::OK();
}

Status IrEmitterUnnested::EmitConstantGlobals() {
  for (const BufferAllocation& allocation :
       ir_emitter_context_->buffer_assignment().Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    const Literal& literal = llvm_ir::LiteralForConstantAllocation(allocation);
    const bool should_emit_initializer = ShouldEmitLiteralInLlvmIr(literal);
    llvm::ArrayType* global_type =
        llvm::ArrayType::get(b_.getInt8Ty(), allocation.size());
    llvm::Constant* initializer =
        should_emit_initializer
            ? llvm_ir::ConvertLiteralToIrConstant(literal, module_)
            : llvm::ConstantAggregateZero::get(global_type);
    if (should_emit_initializer) {
      VLOG(3) << "Emitted initializer for constant with shape "
              << ShapeUtil::HumanString(literal.shape());
    }

    // These globals will be looked up by name by GpuExecutable so we need to
    // give them an external linkage.  Not all of their uses are visible in
    // the LLVM IR (e.g. TupleThunk) so we can't give then a linkage that
    // merely preserves their names (like available_externally), we also need
    // to ensure that they stick around even if they're "unused".
    //
    // We may have to be more more clever here in the future if we notice that
    // we're keeping around too many globals because of their linkage.
    unsigned global_address_space = llvm_ir::GetGlobalMemoryAddressSpace(
        *ir_emitter_context_->llvm_module());
    llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
        global_type, /*isConstant=*/should_emit_initializer,
        llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/initializer,
        llvm_ir::ConstantBufferAllocationToGlobalName(allocation),
        /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/global_address_space,
        /*isExternallyInitialized=*/false);
    global_for_const->setAlignment(kConstantBufferAlignBytes);
    ir_emitter_context_->llvm_module()->getGlobalList().push_back(
        global_for_const);
  }

  return Status::OK();
}

// Emits code for slices based on the below structure. An if statement with
// a guarding condition is generated for each ROOT slice.
//
// Pseudo code:
//
// Compute values of slice input operands
//
// Compute guarding_cond0
// if (guarding_cond0) {
//   Write to output of slice0
// }
//
// Compute guarding_cond1
// if (guarding_cond1) {
//   Write to output of slice1
// }
//
void IrEmitterUnnested::EmitElementForInputFusibleSlices(
    HloInstruction* unnested_hlo, const llvm_ir::IrArray::Index& index) {
  VLOG(10) << "Emitting slice input fusion for " << unnested_hlo->ToString();

  HloInstruction* slice_or_tuple = unnested_hlo->fused_expression_root();
  auto slice_instructions = [&]() -> absl::Span<HloInstruction* const> {
    if (slice_or_tuple->opcode() == HloOpcode::kSlice) {
      return absl::Span<HloInstruction* const>(&slice_or_tuple, 1);
    }
    CHECK_EQ(slice_or_tuple->opcode(), HloOpcode::kTuple);
    return slice_or_tuple->operands();
  }();

  // Emit input operand values of slices.
  std::vector<llvm::Value*> input_ir_values;
  GpuElementalIrEmitter elem_emitter(hlo_module_config_, module_, &b_,
                                     GetNestedComputer());
  FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(unnested_hlo),
                               &elem_emitter);
  TF_CHECK_OK(unnested_hlo->fused_expression_root()->Accept(&fused_emitter));
  for (const HloInstruction* slice : slice_instructions) {
    auto input_generator = fused_emitter.GetGenerator(slice->operand(0));
    input_ir_values.push_back(input_generator(index).ValueOrDie());
  }

  // Emit for slice_instructions.
  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);
  for (int64 i = 0; i < slice_instructions.size(); ++i) {
    HloInstruction* slice = slice_instructions[i];

    // guarding_cond := index >= start && index < limit, for each dim.
    std::vector<llvm::Value*> index_within_ranges;
    for (size_t dim = 0; dim < slice->slice_starts().size(); ++dim) {
      CHECK_EQ(slice->slice_strides(dim), 1);
      auto larger_or_equal_than_start = b_.CreateICmpSGE(
          index.multidim()[dim],
          index.GetConstantWithIndexType(slice->slice_starts(dim)));
      llvm::Value* smaller_than_limit = b_.CreateICmpSLT(
          index.multidim()[dim],
          index.GetConstantWithIndexType(slice->slice_limits(dim)));
      llvm::Value* within_range =
          b_.CreateAnd(larger_or_equal_than_start, smaller_than_limit);
      index_within_ranges.push_back(within_range);
    }
    llvm::Value* guarding_cond = b_.CreateAnd(index_within_ranges);

    auto emit_slice_elem_func = [&] {
      const std::vector<llvm::Value*>& src_multidim = index.multidim();
      std::vector<llvm::Value*> dst_multidim(src_multidim.size());
      for (size_t dim = 0; dim < src_multidim.size(); ++dim) {
        dst_multidim[dim] =
            Sub(src_multidim[dim],
                index.GetConstantWithIndexType(slice->slice_starts(dim)));
      }
      ShapeIndex shape_index = (slice_or_tuple->opcode() == HloOpcode::kSlice)
                                   ? ShapeIndex()
                                   : ShapeIndex({i});
      llvm_ir::IrArray src_ir_array =
          GetIrArray(*unnested_hlo, *unnested_hlo, shape_index);
      IrArray::Index slice_dst_index(dst_multidim, slice->shape(),
                                     index.GetType());
      llvm::Value* dst_addr = src_ir_array.EmitArrayElementAddress(
          slice_dst_index, &b_, "slice.dest");
      b_.CreateStore(input_ir_values[i], dst_addr);
    };

    ksl.If(StrCat("slice", i), guarding_cond, emit_slice_elem_func);
  }
}

Status IrEmitterUnnested::EmitInputFusibleNonStridedSlices(
    HloInstruction* unnested_hlo) {
  constexpr int unroll_factor = 1;
  std::unique_ptr<KernelThunk> kernel_thunk = BuildKernelThunk(
      unnested_hlo, /*implements_whole_instruction=*/true, unroll_factor);

  TF_ASSIGN_OR_RETURN(Shape element_shape,
                      GetConsistentInputShapeForRootSlices(*unnested_hlo));
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      element_shape, ir_emitter_context_->device_description(), unroll_factor);
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  Status emit_status =
      ParallelLoopEmitter(
          [&](const llvm_ir::IrArray::Index index) -> Status {
            EmitElementForInputFusibleSlices(unnested_hlo, index);
            return Status::OK();
          },
          element_shape, launch_dimensions, &b_)
          .EmitLoop(IrName(unnested_hlo),
                    GetIndexTypeForKernel(
                        unnested_hlo, launch_dimensions.launch_bound(), &b_));

  thunk_sequence_->emplace_back(std::move(kernel_thunk));

  return emit_status;
}

}  // namespace gpu
}  // namespace xla
