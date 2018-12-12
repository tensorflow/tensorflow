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

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"

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
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
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
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/sort_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
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

using llvm_ir::KernelMappingScheme;

namespace {

using absl::InlinedVector;
using absl::nullopt;
using absl::optional;
using absl::StrCat;
using llvm_ir::IrArray;
using llvm_ir::IrName;

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
const int64 kMinDimensionToTransposeTiled = 16;

// Returns true if all paths from `hlo` to `root` contain only tuples. The
// result of such an HloInstruction does not need to be materialized, when the
// computation can have a hybrid result.
bool ReachRootViaOnlyTuples(const HloInstruction& hlo,
                            const HloInstruction& root) {
  if (hlo.opcode() != HloOpcode::kTuple) {
    return false;
  }

  if (&hlo == &root) {
    return true;
  }

  for (HloInstruction* user : hlo.users()) {
    if (!ReachRootViaOnlyTuples(*user, root)) {
      return false;
    }
  }

  return true;
}

// If `hlo` is a Transpose, returns its operand; otherwise returns `hlo` itself.
const HloInstruction* StripTranspose(const HloInstruction& hlo) {
  if (hlo.IsRank2Transpose()) {
    return hlo.operand(0);
  }
  return &hlo;
}

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
      fn_arg->setName(llvm_ir::AsStringRef(StrCat("alloc", alloc->index())));
    }
  }

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  // Add the declaration of this kernel to llvm.nvvm.annotations so that NVPTX
  // treats it as a CUDA kernel.
  llvm::NamedMDNode* nvvm_annotations_node =
      module->getOrInsertNamedMetadata("nvvm.annotations");
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      context, {llvm::ConstantAsMetadata::get(kernel),
                llvm::MDString::get(context, "kernel"),
                llvm::ConstantAsMetadata::get(b_.getInt32(1))}));

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
  // Find the unnested hlo instructon for which the kernel is generated for.
  const HloInstruction* unnested_hlo = hlo;
  const HloComputation* computation = hlo->parent();
  if (computation->IsFusionComputation()) {
    unnested_hlo = computation->FusionInstruction();
  }

  auto shape_in_range = [&](const Shape& s) {
    bool in_range = true;
    ShapeUtil::ForEachSubshape(
        s, [&](const Shape& sub_shape, const ShapeIndex& /*index*/) {
          if (ShapeUtil::IsArray(sub_shape) &&
              !IsInt32(ShapeUtil::ElementsIn(sub_shape))) {
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

}  // namespace

Status IrEmitterUnnested::DefaultAction(HloInstruction* hlo) {
  return IrEmitter::DefaultAction(hlo);
}

Status IrEmitterUnnested::HandleDot(HloInstruction* dot) {
  if (ImplementedAsGemm(*dot)) {
    AddThunkToThunkSequence(BuildGemmThunk(dot));
    return Status::OK();
  }
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
  // A CustomCall on the GPU backend can either be a custom-call to a
  // user-supplied kernel, or a call into a library like cudnn.

  // Lower custom-calls to cudnn batchnorm ops to specialized thunks.  It's part
  // of the contract of these cudnn batchnorm calls that the epsilon and
  // feature_index operands be constants.
  if (custom_call->custom_call_target() ==
      kCudnnBatchNormForwardInferenceCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(5);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(6);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardInferenceThunk>(
            /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
            /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
            /*offset=*/GetAllocationSlice(*custom_call->operand(2)),
            /*mean=*/GetAllocationSlice(*custom_call->operand(3)),
            /*variance=*/GetAllocationSlice(*custom_call->operand(4)),
            /*epsilon=*/epsilon_value,
            /*feature_index=*/feature_index_value,
            /*output=*/GetAllocationSlice(*custom_call),
            /*hlo=*/custom_call));
    return Status::OK();
  }

  if (custom_call->custom_call_target() ==
      kCudnnBatchNormForwardTrainingCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(3);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(4);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    // BatchNormTraining returns a tuple of three elements: data, calculated
    // mean, and calculated 1/sqrt(variance + epsilon).
    const auto& assn = ir_emitter_context_->buffer_assignment();
    auto output_data = assn.GetUniqueSlice(custom_call, {0}).ValueOrDie();
    auto output_mean = assn.GetUniqueSlice(custom_call, {1}).ValueOrDie();
    auto output_inv_stddev = assn.GetUniqueSlice(custom_call, {2}).ValueOrDie();
    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardTrainingThunk>(
            /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
            /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
            /*offset=*/GetAllocationSlice(*custom_call->operand(2)),
            /*epsilon=*/epsilon_value,
            /*feature_index=*/feature_index_value,
            /*output_data=*/output_data,
            /*output_mean=*/output_mean,
            /*output_inv_stddev=*/output_inv_stddev,
            /*output_tuple=*/GetAllocationSlice(*custom_call),
            /*hlo=*/custom_call));
    return Status::OK();
  }

  if (custom_call->custom_call_target() == kCudnnBatchNormBackwardCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(5);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(6);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    // BatchNormGrad returns a tuple of three elements: grad_data, grad_scale,
    // grad_offset.
    const auto& assn = ir_emitter_context_->buffer_assignment();
    auto output_grad_data = assn.GetUniqueSlice(custom_call, {0}).ValueOrDie();
    auto output_grad_scale = assn.GetUniqueSlice(custom_call, {1}).ValueOrDie();
    auto output_grad_offset =
        assn.GetUniqueSlice(custom_call, {2}).ValueOrDie();
    AddThunkToThunkSequence(absl::make_unique<CudnnBatchNormBackwardThunk>(
        /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
        /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
        /*mean=*/GetAllocationSlice(*custom_call->operand(2)),
        /*inv_stddev=*/GetAllocationSlice(*custom_call->operand(3)),
        /*grad_output=*/GetAllocationSlice(*custom_call->operand(4)),
        /*epsilon=*/epsilon_value,
        /*feature_index=*/feature_index_value,
        /*output_grad_data=*/output_grad_data,
        /*output_grad_scale=*/output_grad_scale,
        /*output_grad_offset=*/output_grad_offset,
        /*output_tuple=*/GetAllocationSlice(*custom_call),
        /*hlo=*/custom_call));
    return Status::OK();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    const auto& assn = ir_emitter_context_->buffer_assignment();
    std::vector<BufferAllocation::Slice> operand_slices;
    operand_slices.reserve(custom_call->operand_count());
    for (const auto* operand : custom_call->operands()) {
      operand_slices.push_back(GetAllocationSlice(*operand));
    }
    auto tuple_result_slice = GetAllocationSlice(*custom_call);
    auto conv_result_slice = assn.GetUniqueSlice(custom_call, {0}).ValueOrDie();
    auto scratch_slice = assn.GetUniqueSlice(custom_call, {1}).ValueOrDie();

    AddThunkToThunkSequence(absl::make_unique<ConvolutionThunk>(
        Cast<HloCustomCallInstruction>(custom_call), std::move(operand_slices),
        conv_result_slice, scratch_slice, tuple_result_slice));
    return Status::OK();
  }

  return IrEmitter::HandleCustomCall(custom_call);
}

Status IrEmitterUnnested::HandleFft(HloInstruction* fft) {
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(fft->operand(0)->shape().layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(fft->shape().layout()));
  AddThunkToThunkSequence(BuildFftThunk(fft));
  return Status::OK();
}

Status IrEmitterUnnested::HandleFusion(HloInstruction* fusion) {
  HloInstruction* root = fusion->fused_expression_root();
  if (HloInstruction::FusionKind::kInput == fusion->fusion_kind()) {
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
      case HloOpcode::kTuple:
      case HloOpcode::kReduce: {
        // HandleFusion specializes reduction from a multi-dimensional array to
        // a 1D array. The specialized version requires a initializer thunk that
        // initializes the output array to the initial value of the reduce.
        if (root->opcode() == HloOpcode::kReduce &&
            ShapeUtil::IsTuple(root->shape())) {
          // TODO(b/112040122): Support variadic reduce.
          return Unimplemented("Variadic reduce is not supported on GPU");
        }
        VLOG(3) << "Emitting fused reduction to vector: " << fusion->ToString();
        std::vector<std::unique_ptr<Thunk>> thunks;
        absl::Span<HloInstruction* const> output_instructions =
            root->opcode() == HloOpcode::kTuple
                ? root->operands()
                : absl::Span<HloInstruction* const>(&root, 1);

        // For multi-output fusion emit an initializer for each tuple element.
        // Otherwise it's sufficient to just initialize the single output.
        HloInstruction* first_reduce = nullptr;
        for (int i = 0, e = output_instructions.size(); i != e; ++i) {
          if (output_instructions[i]->opcode() == HloOpcode::kReduce) {
            TF_ASSIGN_OR_RETURN(
                std::unique_ptr<Thunk> initializer_thunk,
                BuildInitializerThunk(fusion, output_instructions[i] == root
                                                  ? ShapeIndex()
                                                  : ShapeIndex({i})));
            thunks.push_back(std::move(initializer_thunk));
            first_reduce =
                first_reduce == nullptr ? output_instructions[i] : first_reduce;
          }
        }
        CHECK(first_reduce != nullptr);
        std::unique_ptr<KernelThunk> kernel_thunk =
            BuildKernelThunk(fusion, /*implements_whole_instruction=*/false);
        GpuElementalIrEmitter elemental_emitter(
            hlo_module_config_, ir_emitter_context_->llvm_module(), &b_,
            GetNestedComputer());
        FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(fusion),
                                     &elemental_emitter);
        TF_RETURN_IF_ERROR(root->Accept(&fused_emitter));

        // For multi-output fusion CHECK the constraints and feed all the
        // reduces into a single loop code generator. Single-output reduce
        // fusion is a special case of that.
        InlinedVector<llvm_ir::ElementGenerator, 1> input_gens;
        InlinedVector<llvm_ir::ElementGenerator, 1> init_value_gens;
        std::vector<std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
            extra_output_gens;
        InlinedVector<HloComputation*, 1> reducers;
        InlinedVector<ShapeIndex, 1> reduce_output_shapes;
        for (int i = 0, e = output_instructions.size(); i != e; ++i) {
          const HloInstruction* inst = output_instructions[i];
          ShapeIndex output_shape_index;
          if (root->opcode() == HloOpcode::kTuple) {
            output_shape_index = {i};
          }
          if (inst->opcode() == HloOpcode::kReduce) {
            CHECK(IsReductionToVector(*inst))
                << "Only reductions to vector are supported";
            // Shapes, layouts and dimensions must be the same for all reduces
            // inside of this fusion.
            CHECK(ShapeUtil::Equal(first_reduce->shape(), inst->shape()));
            CHECK(ShapeUtil::Equal(first_reduce->operand(0)->shape(),
                                   inst->operand(0)->shape()));
            CHECK(ShapeUtil::Equal(first_reduce->operand(1)->shape(),
                                   inst->operand(1)->shape()));
            CHECK(first_reduce->dimensions() == inst->dimensions());
            input_gens.push_back(fused_emitter.GetGenerator(inst->operand(0)));
            init_value_gens.push_back(
                fused_emitter.GetGenerator(inst->operand(1)));
            reducers.push_back(inst->to_apply());
            reduce_output_shapes.push_back(std::move(output_shape_index));
          } else {
            // For extra outputs we can relax shape equality to allow different
            // types (with the same number of elements). Layouts still have to
            // match.
            CHECK(ShapeUtil::CompatibleIgnoringElementType(
                first_reduce->operand(0)->shape(), inst->shape()));
            CHECK(LayoutUtil::Equal(first_reduce->operand(0)->shape().layout(),
                                    inst->shape().layout()));
            extra_output_gens.emplace_back(fused_emitter.GetGenerator(inst),
                                           std::move(output_shape_index));
          }
        }
        const Shape& input_shape = first_reduce->operand(0)->shape();
        TF_CHECK_OK(EmitReductionToVector(
            kernel_thunk.get(), first_reduce, input_shape, input_gens,
            init_value_gens, first_reduce->dimensions(), reducers,
            reduce_output_shapes, extra_output_gens));
        thunks.push_back(std::move(kernel_thunk));
        std::unique_ptr<SequentialThunk> sequential_thunk =
            absl::make_unique<SequentialThunk>(std::move(thunks), fusion);
        AddThunkToThunkSequence(std::move(sequential_thunk));
        return Status::OK();
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

  if (ImplementedAsGemm(*fusion)) {
    AddThunkToThunkSequence(BuildGemmThunk(fusion));
    return Status::OK();
  }

  CHECK_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kLoop);

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
    AddThunkToThunkSequence(BuildDeviceToDeviceCopyThunk(copy));
    return Status::OK();
  }
  if (CheckAndEmitHloWithTile021(copy)) {
    return Status::OK();
  }

  return IrEmitter::HandleCopy(copy);
}

Status IrEmitterUnnested::EmitExtraOutputsForReduce(
    const HloInstruction* reduce, const IrArray::Index& index,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  for (int i = 0; i != extra_output_gens.size(); ++i) {
    const HloInstruction* output = reduce->parent()->FusionInstruction();
    llvm::Value* extra_output_address =
        GetIrArray(*output, *output, extra_output_gens[i].second)
            .EmitArrayElementAddress(index, &b_,
                                     "extra_output_element_address");
    TF_ASSIGN_OR_RETURN(llvm::Value* const extra_output_ir_value,
                        extra_output_gens[i].first(index));
    Store(extra_output_ir_value, extra_output_address);
  }
  return Status::OK();
}

Status IrEmitterUnnested::EmitReductionToScalar(
    KernelThunk* kernel_thunk, HloInstruction* reduce, const Shape& input_shape,
    absl::Span<const llvm_ir::ElementGenerator> input_gens,
    absl::Span<const llvm_ir::ElementGenerator> init_value_gens,
    absl::Span<HloComputation* const> reducers,
    absl::Span<const ShapeIndex> reduce_output_shapes,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  // Number of elements processed by a single thread.
  constexpr int64 kTileSize = 16;
  int64 num_elems = ShapeUtil::ElementsIn(input_shape);

  // Round up the number of tiles to a multiple of the warp size.  This is
  // necessary for correctness.  We launch one thread per tile, and if the
  // number of threads isn't a multiple of the number of the warp size, our
  // shuffles will read from inactive threads, producing undefined values.
  int64 num_tiles =
      RoundUpToNearest(CeilOfRatio(num_elems, kTileSize), kWarpSize);

  Shape tiled_input_shape = ShapeUtil::MakeShapeWithLayout(
      reduce->shape().element_type(), {num_tiles}, {0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());

  llvm::Type* index_ty =
      GetIndexTypeForKernel(reduce, launch_dimensions.launch_bound(), &b_);

  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  // Check whether every thread will process a full tile's worth of elements
  // without reading outside the bounds of the input.  If this is true, we can
  // skip some bounds checks in the final algorithm.
  bool all_threads_in_bounds = num_tiles * kTileSize == num_elems;

  // __global__ void full_reduce_kernel() {
  //   x_in_tiles = threadIdx.x + blockIdx.x * blockDim.x;
  //   x = x_in_tiles * kTileSize;
  //
  //   partial_result = init_value;
  //   if (all_threads_in_bounds || x + kTileSize <= num_elems) {
  //     for (i = 0; i < kTileSize; ++i) {
  //       partial_result = Reducer(partial_result, input[x + i]);
  //     }
  //   } else {
  //     for (i = 0; i < kTileSize; ++i) {
  //       if (x + i < num_elems) {
  //         partial_result = Reducer(partial_result, input[x + i]);
  //       }
  //     }
  //   }
  //   for (i = warpSize / 2; i > 0; i /= 2) {
  //     partial_result = Reducer(partial_result,
  //                              __shfl_down(partial_result, i));
  //   }
  //   if (lane_id == 0) {
  //     AtomicReducer(&output[y], partial_result);
  //   }
  // }
  //
  // // Choose num_blocks and threads_per_block such that:
  // //
  // //   num_blocks * threads_per_block =
  // //     RoundUpToNextMultipleOf(Ceil(num_elems / kTileSize), warpSize),
  // //
  // // and threads_per_block is a multiple of warpSize.
  // reduce_kernel  //
  auto loop_body_emitter = [=](const IrArray::Index& tile_index) -> Status {
    const int num_reduces = reducers.size();
    llvm::Type* element_ir_type =
        llvm_ir::PrimitiveTypeToIrType(input_shape.element_type(), module_);
    std::vector<llvm::Value*> partial_reduction_result_addresses;
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* partial_reduction_result_address =
          Alloca(element_ir_type, /*ArraySize=*/nullptr,
                 "partial_reduction_result." + llvm::Twine(i));
      TF_ASSIGN_OR_RETURN(llvm::Value* const init_ir_value,
                          init_value_gens[i](IrArray::Index(index_ty)));
      Store(init_ir_value, partial_reduction_result_address);
      partial_reduction_result_addresses.push_back(
          partial_reduction_result_address);
    }

    llvm::Value* x_in_tiles = tile_index[0];
    x_in_tiles = ZExtOrTrunc(x_in_tiles, index_ty);

    // Emit an inner for-loop that reduces the elements in the tile.
    auto emit_tile_element_loop = [=](bool tile_in_bounds) -> Status {
      std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
          llvm_ir::ForLoop::EmitForLoop(
              "element_id_in_tile", index_typed_constant(0),
              index_typed_constant(kTileSize), index_typed_constant(1), &b_);

      // Emit the body of the partial reduction loop.
      llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(),
                                     &b_);
      llvm::Value* x =
          NSWAdd(NSWMul(x_in_tiles, index_typed_constant(kTileSize)),
                 tile_element_loop->GetIndVarValue());
      // Unless we know the tile is entirely in bounds, we have to emit a
      // x-in-bounds check before reading from the input.
      if (!tile_in_bounds) {
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            ICmpULT(x, index_typed_constant(num_elems)), "x_in_bounds", &b_);

        // Emit code that reads the input element and accumulates it to
        // the partial reduction result.
        llvm_ir::SetToFirstInsertPoint(if_data.true_block, &b_);
      }

      IrArray::Index input_index(
          /*linear=*/x, input_shape, &b_);
      llvm::Value* input_address = Alloca(element_ir_type);
      for (int i = 0; i != num_reduces; ++i) {
        TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                            input_gens[i](input_index));
        Store(input_ir_value, input_address);
        TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
            *reducers[i],
            {partial_reduction_result_addresses[i], input_address},
            partial_reduction_result_addresses[i]));
      }
      return EmitExtraOutputsForReduce(reduce, input_index, extra_output_gens);
    };

    // x_end = kTileSize + x_in_tiles * kTileSize, i.e., the location that's
    // immediately beyond the tile.
    llvm::Value* x_end =
        NSWAdd(index_typed_constant(kTileSize),
               NSWMul(x_in_tiles, index_typed_constant(kTileSize)));
    // The tile is entirely in bound if all_threads_in_bounds or
    // x_end <= num_elems.
    llvm::Value* tile_in_bounds =
        Or(ICmpULE(x_end, index_typed_constant(num_elems)),
           b_.getInt1(all_threads_in_bounds));
    llvm_ir::LlvmIfData if_tile_in_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_bounds, "tile_in_bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.true_block, &b_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.false_block, &b_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/false));

    // After the if-then-else statement on tile_in_bounds, emit calls to
    // shfl_down that accumulate the partial reduction results of all threads
    // from the warp.
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.after_block, &b_);
    int bit_width = llvm_ir::GetSizeInBits(element_ir_type);
    // bitcast cannot be applied to aggregate types (even packed ones), so we
    // instead bitcast addresses of load/store to intN* of the same bit-width.
    llvm::Type* shuffle_ir_type = element_ir_type->isStructTy()
                                      ? b_.getIntNTy(bit_width)
                                      : element_ir_type;
    for (int shuffle_distance = kWarpSize / 2; shuffle_distance >= 1;
         shuffle_distance /= 2) {
      llvm::Value* result_from_other_lane =
          Alloca(element_ir_type, nullptr, "result_from_other_lane");
      for (int i = 0; i != num_reduces; ++i) {
        llvm::Value* partial_reduction_result =
            Load(BitCast(partial_reduction_result_addresses[i],
                         shuffle_ir_type->getPointerTo()),
                 "partial_reduction_result");
        CHECK_EQ(launch_dimensions.threads_per_block() % kWarpSize, 0)
            << "Requires block size a multiple of the warp size, otherwise we "
               "will read undefined elements.";
        Store(EmitFullWarpShuffleDown(partial_reduction_result,
                                      b_.getInt32(shuffle_distance), &b_),
              BitCast(result_from_other_lane, shuffle_ir_type->getPointerTo()));
        TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
            *reducers[i],
            {partial_reduction_result_addresses[i], result_from_other_lane},
            partial_reduction_result_addresses[i]));
      }
    }

    const HloInstruction* output =
        reduce->IsFused() ? reduce->parent()->FusionInstruction() : reduce;

    // Emit an atomic operation that accumulates the partial reduction result of
    // lane 0 (which holds the partially accumulated result for its warp) to the
    // output element.
    llvm::Value* lane_id =
        URem(x_in_tiles, index_typed_constant(kWarpSize), "lane_id");
    llvm_ir::LlvmIfData if_lane_id_is_zero_data = llvm_ir::EmitIfThenElse(
        ICmpEQ(lane_id, index_typed_constant(0)), "lane_id_is_zero", &b_);
    llvm_ir::SetToFirstInsertPoint(if_lane_id_is_zero_data.true_block, &b_);

    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* output_address =
          GetIrArray(*output, *output, reduce_output_shapes[i])
              .EmitArrayElementAddress(
                  IrArray::Index(
                      /*linear=*/b_.getInt64(0),
                      ShapeUtil::GetSubshape(output->shape(),
                                             reduce_output_shapes[i]),
                      &b_),
                  &b_, "output_element_address");
      TF_RETURN_IF_ERROR(EmitAtomicOperationForNestedComputation(
          *reducers[i], output_address, partial_reduction_result_addresses[i]));
    }
    return Status::OK();
  };

  // Emit a parallel loop that iterates through all input tiles, one per thread.
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk,
                         ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &b_)
      .EmitLoop(IrName(reduce), index_ty);
}

Status IrEmitterUnnested::EmitColumnReduction(
    KernelThunk* kernel_thunk, int64 height, int64 width,
    HloInstruction* reduce, const Shape& input_shape,
    absl::Span<const llvm_ir::ElementGenerator> input_gens,
    absl::Span<const llvm_ir::ElementGenerator> init_value_gens,
    absl::Span<HloComputation* const> reducers,
    absl::Span<const ShapeIndex> reduce_output_shapes,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  // Divide the input matrix into tiles of size KxL. For example, when the
  // input matrix is 4x4, K=2, and L=1 the tiled matrix looks like
  //
  //   0123
  //   0123
  //   4567
  //   4567  // Numbers indicate tile IDs.
  //
  // Each tile is first partially reduced to a scalar by a thread, and then the
  // scalar is accumulated to the output vector using atomic operations.
  //
  // We choose 128 as the tile size based on empirical evidence. It's big enough
  // to reduce the amount of atomic adds in the end, maximizing the memory
  // bandwidth. A tile width of 2 allows for high memory bandwidth utilization
  // on 16b input data.
  constexpr int64 kTileHeight = 128;
  constexpr int64 kTileWidth = 2;

  // If the height is not a multiple of kTileHeight, we pad the bottom of the
  // input matrix.
  const int64 height_in_tiles = CeilOfRatio(height, kTileHeight);
  // If width is not a multiple of kTileWidth the rightmost thread will process
  // fewer input elements.
  const int64 width_in_tiles = CeilOfRatio(width, kTileWidth);
  Shape tiled_input_shape =
      ShapeUtil::MakeShapeWithLayout(reduce->shape().element_type(),
                                     {height_in_tiles, width_in_tiles}, {1, 0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());

  // TODO(b/110211620): Convert to use i32 index_type when it is possible.
  llvm::Type* index_ty = b_.getInt64Ty();

  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < height_in_tiles * width_in_tiles;
  //      linear_index += blockDim.x * gridDim.x) {
  //   y_in_tiles = linear_index / width_in_tiles;
  //   x_in_tiles = linear_index % width_in_tiles;
  //
  //   partial_results[kTileWidth] = init_values;
  //   tile_in_y_bounds = height % kTileHeight == 0 ||
  //       y_in_tiles * kTileHeight + kTileHeight <= height;
  //   tile_in_x_bounds = width % kTileWidth == 0 ||
  //       x_in_tiles * kTileWidth + kTileWidth <= width;
  //   // The implementation handles y and x bound checks separately.
  //   if (tile_in_y_bounds && tile_in_x_bounds) {
  //     for (y_offset : range(kTileHeight)) {
  //       y = y_in_tiles * kTileHeight + y_offset;
  //       for (x_offset : range(kTileWidth)) {
  //         x = x_in_tiles * kTileWidth + x_offset;
  //         partial_result = Reducer(partial_result[x_offset], input[y][x]);
  //       }
  //     }
  //   } else {
  //     for (y_offset : range(kTileHeight)) {
  //       y = y_in_tiles * kTileHeight + y_offset;
  //       for (y_offset : range(kTileHeight)) {
  //         x = x_in_tiles * kTileWidth + x_offset;
  //         if (y < height && x < width) {
  //           partial_result = Reducer(partial_result, input[y][x]);
  //         }
  //       }
  //     }
  //   }
  //   for (x_offset : range(kTileWidth)) {
  //     AtomicReducer(&output[x + x_offset], partial_result[x_offset]);
  //   }
  // }
  auto loop_body_emitter = [=](const IrArray::Index& tile_index) -> Status {
    const int num_reduces = reducers.size();
    // Emit the loop body that reduces one tile.
    llvm::Type* element_ir_type =
        llvm_ir::PrimitiveTypeToIrType(input_shape.element_type(), module_);
    std::vector<llvm::Value*> partial_reduction_result_addresses;
    for (int i = 0; i != num_reduces; ++i) {
      for (int x_offset = 0; x_offset < kTileWidth; ++x_offset) {
        llvm::Value* partial_reduction_result_address =
            Alloca(element_ir_type, /*ArraySize=*/nullptr,
                   "partial_reduction_result." +
                       llvm::Twine(i * kTileWidth + x_offset));
        TF_ASSIGN_OR_RETURN(llvm::Value* const init_ir_value,
                            init_value_gens[i](IrArray::Index(index_ty)));
        Store(init_ir_value, partial_reduction_result_address);
        partial_reduction_result_addresses.push_back(
            partial_reduction_result_address);
      }
    }

    // Emit an inner for-loop that partially reduces the elements in the given
    // tile.
    llvm::Value* y_in_tiles = tile_index[0];
    llvm::Value* x_in_tiles = tile_index[1];

    y_in_tiles = ZExtOrTrunc(y_in_tiles, index_ty);
    x_in_tiles = ZExtOrTrunc(x_in_tiles, index_ty);

    auto emit_tile_element_loop = [=](bool tile_in_y_bounds,
                                      bool tile_in_x_bounds) -> Status {
      std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
          llvm_ir::ForLoop::EmitForLoop(
              "element_id_in_tile", index_typed_constant(0),
              index_typed_constant(kTileHeight), index_typed_constant(1), &b_);

      // Emit the body of the partial reduction loop.
      llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(),
                                     &b_);
      llvm::Value* y =
          NSWAdd(NSWMul(y_in_tiles, index_typed_constant(kTileHeight)),
                 tile_element_loop->GetIndVarValue());

      // Unless we know that y is in bounds, we have to emit a check before
      // reading from the input.
      if (!tile_in_y_bounds) {
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            ICmpULT(y, index_typed_constant(height)), "y_in_bounds", &b_);

        // Emit code that reads the input element and accumulates it to
        // the partial reduction result.
        llvm_ir::SetToFirstInsertPoint(if_data.true_block, &b_);
      }
      for (int x_offset = 0; x_offset < kTileWidth; ++x_offset) {
        llvm::Value* x =
            NSWAdd(NSWMul(x_in_tiles, index_typed_constant(kTileWidth)),
                   index_typed_constant(x_offset));
        // Unless we know that x is in bounds, we have to emit a check before
        // reading from the input.
        if (!tile_in_x_bounds) {
          llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
              ICmpULT(x, index_typed_constant(width)), "x_in_bounds", &b_);
          llvm_ir::SetToFirstInsertPoint(if_data.true_block, &b_);
        }
        llvm::Value* input_address = Alloca(element_ir_type);
        // {y,x} is an index to input_matrix_shape [height,width]. We need to
        // convert that to an index to input_shape (the shape of the operand of
        // "reduce"). This conversion is composed of a transposition from
        // input_shape to normalized_input_shape and a reshape from
        // normalized_input_shape to input_matrix_shape.
        const Shape normalized_input_shape =
            ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                input_shape);
        auto input_shape_min2maj = LayoutUtil::MinorToMajor(input_shape);
        const std::vector<int64> transpose_dimension_mapping(
            input_shape_min2maj.rbegin(), input_shape_min2maj.rend());

        const Shape input_matrix_shape =
            ShapeUtil::MakeShapeWithDescendingLayout(input_shape.element_type(),
                                                     {height, width});
        const IrArray::Index input_matrix_index({y, x}, input_matrix_shape,
                                                &b_);
        const IrArray::Index input_index =
            input_matrix_index
                .SourceIndexOfReshape(input_matrix_shape,
                                      normalized_input_shape, &b_)
                .SourceIndexOfTranspose(normalized_input_shape, input_shape,
                                        transpose_dimension_mapping, &b_);
        for (int i = 0; i != num_reduces; ++i) {
          TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                              input_gens[i](input_index));
          Store(input_ir_value, input_address);
          TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
              *reducers[i],
              {partial_reduction_result_addresses[i * kTileWidth + x_offset],
               input_address},
              partial_reduction_result_addresses[i * kTileWidth + x_offset]));
          TF_RETURN_IF_ERROR(EmitExtraOutputsForReduce(reduce, input_index,
                                                       extra_output_gens));
        }
      }
      return Status::OK();
    };

    // y_end = kTileHeight + y_in_tiles * kTileHeight, i.e., the y location
    // that's immediately beyond the tile.
    llvm::Value* y_end =
        NSWAdd(index_typed_constant(kTileHeight),
               NSWMul(y_in_tiles, index_typed_constant(kTileHeight)));
    // x_end = kTileWidth + x_in_tiles * kTileWidth, i.e., the x location
    // that's immediately beyond the tile.
    llvm::Value* x_end =
        NSWAdd(index_typed_constant(kTileWidth),
               NSWMul(x_in_tiles, index_typed_constant(kTileWidth)));
    llvm::Value* tile_in_y_bounds =
        Or(ICmpULE(y_end, index_typed_constant(height)),
           b_.getInt1(height % kTileHeight == 0));
    llvm::Value* tile_in_x_bounds =
        Or(ICmpULE(x_end, index_typed_constant(width)),
           b_.getInt1(width % kTileWidth == 0));
    // The tile is in y bounds if "height" is a multiple of kTileHeight or
    // y_end <= height.
    llvm_ir::LlvmIfData if_tile_in_y_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_y_bounds, "tile_in_y_bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_y_bounds_data.true_block, &b_);
    // The tile is in x bounds if "width" is a multiple of kTileWidth or
    // x_end <= width.
    llvm_ir::LlvmIfData if_tile_in_x_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_x_bounds, "tile_in_x_bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_x_bounds_data.true_block, &b_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_y_bounds=*/true,
                                              /*tile_in_x_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_x_bounds_data.false_block, &b_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_y_bounds=*/true,
                                              /*tile_in_x_bounds=*/false));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_y_bounds_data.false_block, &b_);
    if_tile_in_x_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_x_bounds, "tile_in_x_bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_x_bounds_data.true_block, &b_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_y_bounds=*/false,
                                              /*tile_in_x_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_x_bounds_data.false_block, &b_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_y_bounds=*/false,
                                              /*tile_in_x_bounds=*/false));

    // After the nested if-then-else statement on tile_in_y_bounds and
    // tile_in_x_bounds, emit atomic operations to accumulate the partial
    // reduction result to the output element.
    llvm_ir::SetToFirstInsertPoint(if_tile_in_y_bounds_data.after_block, &b_);
    const HloInstruction* output =
        reduce->IsFused() ? reduce->parent()->FusionInstruction() : reduce;
    for (int i = 0; i != num_reduces; ++i) {
      for (int x_offset = 0; x_offset < kTileWidth; ++x_offset) {
        llvm::Value* x =
            NSWAdd(NSWMul(x_in_tiles, index_typed_constant(kTileWidth)),
                   index_typed_constant(x_offset));
        llvm::Value* output_address =
            GetIrArray(*output, *output, reduce_output_shapes[i])
                .EmitArrayElementAddress(
                    IrArray::Index(
                        x,
                        ShapeUtil::GetSubshape(output->shape(),
                                               reduce_output_shapes[i]),
                        &b_),
                    &b_, "output_element_address");
        TF_RETURN_IF_ERROR(EmitAtomicOperationForNestedComputation(
            *reducers[i], output_address,
            partial_reduction_result_addresses[i * kTileWidth + x_offset]));
      }
    }
    return Status::OK();
  };

  // Emit a parallel loop that iterate through all input tiles.
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk,
                         ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &b_)
      .EmitLoop(IrName(reduce), index_ty);
}

static std::pair<int64, int64> ComputeKernelMappingSchemeForReduction(
    int64 depth, int64 width, int64 kWarpSize) {
  constexpr int64 kTargetNumElementsPerThread = 64;
  int64 x_tile_size = kTargetNumElementsPerThread;
  int64 z_tile_size = 1;

  // Only tile along the x dimension with tile size kTargetNumElementsPerThread
  // if doing so doesn't require a slow version of loop with bound check on each
  // dimension. A more sophisticated heuristics is to enable tile along the
  // x dimension with tile size kTargetNumElementsPerThread when either width is
  // a factor of (kWarpSize * kTargetNumElementsPerThread) or width is big
  // enough so that only a small fraction of the threads execute the slow
  // version of loop with bound check.
  if (width % (kWarpSize * kTargetNumElementsPerThread) != 0) {
    x_tile_size = 8;
    z_tile_size = 8;
    while (depth % z_tile_size != 0) {
      z_tile_size -= 1;
    }
  }

  return std::pair<int64, int64>(x_tile_size, z_tile_size);
}

Status IrEmitterUnnested::EmitRowReduction(
    KernelThunk* kernel_thunk, int64 depth, int64 height, int64 width,
    HloInstruction* reduce, const Shape& input_shape,
    absl::Span<const llvm_ir::ElementGenerator> input_gens,
    absl::Span<const llvm_ir::ElementGenerator> init_value_gens,
    absl::Span<HloComputation* const> reducers,
    absl::Span<const ShapeIndex> reduce_output_shapes,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  // A naive algorithm is:
  // 1. Divide the x dimension of the input tensor into tiles of size 1x1xX.
  // 2. Partially reduces each tile to a scalar using one thread.
  // 3. Accumulates that scalar to the output vector using atomic operations.
  //
  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < depth * height * width_in_tiles;
  //      linear_index += blockDim.x * gridDim.x) {
  //   int x_in_tiles = linear_index % width_in_tiles;
  //   int y = linear_index / width_in_tiles % height;
  //   int z = linear_index / (height * width_in_tiles);
  //   float partial_result = 0;
  //   for (element_id_in_tile : range(x_tile_size)) {
  //     int x = x_in_tiles * x_tile_size + element_id_in_tile;
  //     if (x < width)
  //       partial_result = reducer(partial_result, input[z][y][x]);
  //   }
  //   AtomicReducer(&output[y], partial_result);
  // }
  //
  // Four optimizations are performed.
  //
  // 1. To coalesce global memory accesses, dilate the tile with a factor of 32
  // (i.e. the warp size). For example, suppose the width is 8x32=256. Instead
  // of making each tile consecutive, we let make tile 0 column
  // [0,32,64,...,224], tile 1 column [1,33,65,...,225], and so on. This ensures
  // that threads in a warp access consecutive memory in one iteration (i.e.
  // coalesced). In the above example, the warp that contains thread 0-31
  // accesses column 0-31 in the first iteration, and 32-63 in the second
  // iteration, and so on.
  //
  // 2. Partially accumulate partial reduced results computed by threads in the
  // same warp using shfl_down. Using shfl_down is faster than directly using
  // atomic operations because shfl_down transfers the data between threads
  // using shared memory and threads in the same warp run in lock step (thus no
  // extra synchronization needed). See
  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  // for details. The downside is, to produce correct results when using
  // shfl_down, we need to guarantee threads in the same warp work on input
  // elements with the same y, so the number of tiles in each row must be a
  // multiple of 32.
  //
  // 3. Specialize the case that the entire tile is in bounds. When that is
  // true, we don't need to emit "if(x<width)" inside the loop on
  // element_id_in_tile, which makes the code more friendly to optimizations
  // such as LICM.
  //
  // 4. When the width is too small and x_tile_size is less than the target
  //    number of elements per thread and use a small factor of depth as
  //    z_tile_size to increase the number of elements calculated by each
  //    partial sum. This can reduce the needed number of dynamic shfl_down and
  //    atomic operations.
  //
  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < depth * height * width_in_tiles;
  //      linear_index += blockDim.x * gridDim.x) {
  //   int x_in_tiles = linear_index % width_in_tiles;
  //   int y = linear_index / width_in_tiles % height;
  //   int z_in_tiles = linear_index / (height * width_in_tiles);
  //   int warp_id = x_in_tiles / warpSize;
  //   int lane_id = x_in_tiles % warpSize;
  //   float partial_result = 0;
  //   int x = warp_id * kTileSize * warpSize + lane_id;
  //   if (width % (x_tile_size * warpSize) == 0 ||
  //       x + (x_tile_size - 1) * warpSize < width) {
  //     // The entire x_tile is in bounds.
  //     for (int element_id_in_z_tile = 0; element_id_in_z_tile < z_tile_size;
  //          ++element_id_in_z_tile) {
  //       z = z_in_tiles * z_tile_size + element_id_in_z_tile;
  //       int tx = x;
  //       for (int element_id_in_x_tile = 0;
  //            element_id_in_x_tile < x_tile_size;
  //            ++element_id_in_x_tile, tx += warpSize) {
  //         partial_result = Reducer(partial_result, input[z][y][tx]);
  //       }
  //     }
  //   } else {
  //     // The tile is partially in bounds.
  //     for (int element_id_in_z_tile = 0; element_id_in_z_tile < z_tile_size;
  //          ++element_id_in_z_tile) {
  //       z = z_in_tiles * z_tile_size + element_id_in_z_tile;
  //       int tx = x;
  //       for (int element_id_in_x_tile = 0; element_id_in_x_tile <
  //            x_tile_size; ++element_id_in_tile, tx += warpSize) {
  //         if (tx < width)
  //           partial_result = Reducer(partial_result, input[z][y][tx]);
  //       }
  //     }
  //   }
  //   for (shuffle_distance = 16; shuffle_distance > 0; shuffle_distance /= 2)
  //     partial_result = Reducer(
  //         partial_result,
  //         __shfl_down_sync(CUDA_WARP_ALL, partial_result, shuffle_distance));
  //   if (lane_id == 0)
  //     AtomicReducer(&output[y], partial_result);
  // }
  //

  int64 x_tile_size;
  int64 z_tile_size;
  std::tie(x_tile_size, z_tile_size) =
      ComputeKernelMappingSchemeForReduction(depth, width, kWarpSize);

  // Round the width in tiles up to the nearest multiple of kWarpSize, so that
  // the use of shfl_down is valid.
  const int64 width_in_tiles =
      RoundUpToNearest(CeilOfRatio(width, x_tile_size), kWarpSize);
  Shape tiled_input_shape = ShapeUtil::MakeShapeWithLayout(
      reduce->shape().element_type(),
      {depth / z_tile_size, height, width_in_tiles}, {2, 1, 0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());
  llvm::Type* index_ty =
      GetIndexTypeForKernel(reduce, launch_dimensions.launch_bound(), &b_);

  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  auto loop_body_emitter = [=](const IrArray::Index& tile_index) {
    const int num_reduces = reducers.size();
    llvm::Type* element_ir_type = llvm_ir::PrimitiveTypeToIrType(
        input_shape.element_type(), ir_emitter_context_->llvm_module());
    std::vector<llvm::Value*> partial_reduction_result_addresses;
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* partial_reduction_result_address =
          Alloca(element_ir_type, /*ArraySize=*/nullptr,
                 "partial_reduction_result." + llvm::Twine(i));
      TF_ASSIGN_OR_RETURN(llvm::Value* const init_ir_value,
                          init_value_gens[i](IrArray::Index(index_ty)));
      Store(init_ir_value, partial_reduction_result_address);
      partial_reduction_result_addresses.push_back(
          partial_reduction_result_address);
    }

    llvm::Value* z_tile = tile_index[0];
    llvm::Value* y = tile_index[1];
    llvm::Value* x_tile = tile_index[2];

    x_tile = ZExtOrTrunc(x_tile, index_ty);

    llvm::Value* warp_id =
        UDiv(x_tile, index_typed_constant(kWarpSize), "warp_id");
    llvm::Value* lane_id =
        URem(x_tile, index_typed_constant(kWarpSize), "lane_id");

    // The x-location of the last element in this z-x-tile.
    // last_x = lane_id + warpSize * (x_tile_size - 1 + warp_id * x_tile_size);
    llvm::Value* last_x = NSWAdd(
        lane_id,
        NSWMul(index_typed_constant(kWarpSize),
               NSWAdd(index_typed_constant(x_tile_size - 1),
                      NSWMul(warp_id, index_typed_constant(x_tile_size)))));

    KernelSupportLibrary ksl(
        &b_,
        /*unroll_mode=*/xla::llvm_ir::UnrollMode::kFullyUnroll,
        /*prevent_vectorization=*/false);

    // Emit a for-loop that partially reduces the elements in the given
    // z-x-tile.
    auto emit_z_x_tile_element_loop = [&](bool x_tile_in_bounds,
                                          int64 x_tile_loop_bound) -> Status {
      auto emit_z_tile_element_loop = [&](llvm::Value* z_indvar) -> Status {
        llvm::Value* z =
            NSWAdd(z_indvar, NSWMul(index_typed_constant(z_tile_size), z_tile));
        TF_RETURN_IF_ERROR(ksl.ForWithStatus(
            "x_tile",
            /*start=*/index_typed_constant(0),
            /*end=*/index_typed_constant(x_tile_loop_bound),
            /*step=*/1, [&](llvm::Value* x_indvar) -> Status {
              // x = lane_id +
              //     warpSize * (element_id_in_x_tile + warp_id * x_tile_size);
              llvm::Value* x = NSWAdd(
                  lane_id,
                  NSWMul(index_typed_constant(kWarpSize),
                         NSWAdd(x_indvar,
                                NSWMul(warp_id, llvm::ConstantInt::get(
                                                    index_ty, x_tile_size)))));

              // Unless we know the x-tile is entirely in bounds, we have to
              // emit a x-in-bounds check before reading from the input.
              if (!x_tile_in_bounds) {
                llvm_ir::LlvmIfData if_x_in_bounds_data =
                    llvm_ir::EmitIfThenElse(
                        ICmpULT(x, index_typed_constant(width)), "x_in_bounds",
                        &b_);
                // Points b_ to the then-block.
                llvm_ir::SetToFirstInsertPoint(if_x_in_bounds_data.true_block,
                                               &b_);
              }

              // Emit code that reads the input element and accumulates it
              // to the partial reduction result.
              llvm::Value* input_address = Alloca(element_ir_type);
              {
                // {z,y,x} is an index to input_3d_tensor_shape
                // [depth,height,width]. We need to convert that to an index
                // to input_shape (the shape of the operand of "reduce").
                // This conversion is composed of a transposition from
                // input_shape to normalized_input_shape and a reshape from
                // normalized_input_shape to input_3d_tensor_shape.
                const Shape normalized_input_shape = ShapeUtil::
                    MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                        input_shape);
                auto input_shape_min2maj =
                    LayoutUtil::MinorToMajor(input_shape);
                const std::vector<int64> transpose_dimension_mapping(
                    input_shape_min2maj.rbegin(), input_shape_min2maj.rend());
                const Shape input_3d_tensor_shape =
                    ShapeUtil::MakeShapeWithDescendingLayout(
                        input_shape.element_type(), {depth, height, width});
                const IrArray::Index input_3d_tensor_index(
                    {z, y, x}, input_3d_tensor_shape, &b_);
                const IrArray::Index input_index =
                    input_3d_tensor_index
                        .SourceIndexOfReshape(input_3d_tensor_shape,
                                              normalized_input_shape, &b_)
                        .SourceIndexOfTranspose(
                            normalized_input_shape, input_shape,
                            transpose_dimension_mapping, &b_);

                for (int i = 0; i != num_reduces; ++i) {
                  TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                                      input_gens[i](input_index));
                  Store(input_ir_value, input_address);
                  TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
                      *reducers[i],
                      {partial_reduction_result_addresses[i], input_address},
                      partial_reduction_result_addresses[i]));
                }
                return EmitExtraOutputsForReduce(reduce, input_index,
                                                 extra_output_gens);
              }
            }));
        return Status::OK();
      };

      return ksl.ForWithStatus("z_tile",
                               /*start=*/index_typed_constant(0),
                               /*end=*/index_typed_constant(z_tile_size),
                               /*step=*/1, emit_z_tile_element_loop);
    };

    llvm::Value* tile_in_bounds =
        Or(b_.getInt1(width % (x_tile_size * kWarpSize) == 0),
           ICmpULT(last_x, index_typed_constant(width)));

    TF_RETURN_IF_ERROR(ksl.IfWithStatus(
        tile_in_bounds,
        /*true_block_generator=*/
        [&]() -> Status {
          return emit_z_x_tile_element_loop(/*x_tile_in_bounds=*/true,
                                            x_tile_size);
        },
        /*false_block_generator=*/
        [&]() -> Status {
          return emit_z_x_tile_element_loop(
              /*x_tile_in_bounds=*/false,
              CeilOfRatio(width % (x_tile_size * kWarpSize), kWarpSize));
        }));

    // After accumulating the elements of the z_x_tile, emit calls to
    // shfl_down that accumulate the partial reduction results of all
    // threads in a warp.
    int bit_width = llvm_ir::GetSizeInBits(element_ir_type);
    // bitcast cannot be applied to aggregate types (even packed ones), so we
    // instead bitcast addresses of load/store to intN* of the same bit-width.
    llvm::Type* shuffle_ir_type = element_ir_type->isStructTy()
                                      ? b_.getIntNTy(bit_width)
                                      : element_ir_type;
    for (int shuffle_distance = 16; shuffle_distance >= 1;
         shuffle_distance /= 2) {
      llvm::Value* result_from_other_lane =
          Alloca(element_ir_type, nullptr, "result_from_other_lane");
      for (int i = 0; i != num_reduces; ++i) {
        llvm::Value* partial_reduction_result =
            Load(BitCast(partial_reduction_result_addresses[i],
                         shuffle_ir_type->getPointerTo()),
                 "partial_reduction_result");
        CHECK_EQ(launch_dimensions.threads_per_block() % kWarpSize, 0)
            << "Requires block size a multiple of the warp size, otherwise we "
               "will read undefined elements.";
        Store(EmitFullWarpShuffleDown(partial_reduction_result,
                                      b_.getInt32(shuffle_distance), &b_),
              BitCast(result_from_other_lane, shuffle_ir_type->getPointerTo()));
        TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
            *reducers[i],
            {partial_reduction_result_addresses[i], result_from_other_lane},
            partial_reduction_result_addresses[i]));
      }
    }

    const HloInstruction* output =
        reduce->IsFused() ? reduce->parent()->FusionInstruction() : reduce;

    // Emit an atomic operation that accumulates the partial reduction result of
    // lane 0 (which holds the partially accumulated result for its warp) to the
    // output element.
    llvm_ir::LlvmIfData if_lane_id_is_zero_data = llvm_ir::EmitIfThenElse(
        ICmpEQ(lane_id, index_typed_constant(0)), "lane_id_is_zero", &b_);
    llvm_ir::SetToFirstInsertPoint(if_lane_id_is_zero_data.true_block, &b_);
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* output_address =
          GetIrArray(*output, *output, reduce_output_shapes[i])
              .EmitArrayElementAddress(
                  IrArray::Index(y,
                                 ShapeUtil::GetSubshape(
                                     output->shape(), reduce_output_shapes[i]),
                                 &b_),
                  &b_, "output_element_address");
      // We don't need to emit atomic operations if there is only one tile of
      // results. 'depth' is the z dimension, 'width' is the x dimension.
      if (z_tile_size >= depth && x_tile_size >= width) {
        TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
            *reducers[i],
            {output_address, partial_reduction_result_addresses[i]},
            output_address));
      } else {
        TF_RETURN_IF_ERROR(EmitAtomicOperationForNestedComputation(
            *reducers[i], output_address,
            partial_reduction_result_addresses[i]));
      }
    }
    return Status::OK();
  };

  // Emit a parallel loop that iterates through every input tiles.
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk,
                         ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &b_)
      .EmitLoop(IrName(reduce), index_ty);
}

// Figures out whether `reduce` is a row or column reduction, and which
// dimensions to reduce, and calls either `EmitRowReduction` or
// `EmitColumnReduction` as appropriate.
// Prerequisite: all the dimensions to keep are contiguous in the input layout
//               and, if `reduce` is fused, the fused subgraph is pure
//               elementwise.
Status IrEmitterUnnested::EmitReductionToVector(
    KernelThunk* kernel_thunk, HloInstruction* reduce, const Shape& input_shape,
    absl::Span<const llvm_ir::ElementGenerator> input_gens,
    absl::Span<const llvm_ir::ElementGenerator> init_value_gens,
    absl::Span<const int64> dimensions_to_reduce,
    absl::Span<HloComputation* const> reducers,
    absl::Span<const ShapeIndex> reduce_output_shapes,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  // This emission requires "reduce" to have an input layout. It is either set
  // by LayoutAssignment (for a top-level kReduce) or by InstructionFusion (for
  // a fused kReduce).
  CHECK(input_shape.has_layout()) << "LayoutAssignment or InstructionFusion "
                                     "doesn't set the input layout of "
                                  << reduce->ToString();

  // Specialize multi-dimensional-array-to-vector reduction.
  std::vector<int64> input_dims_to_keep;
  for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
       ++input_dim) {
    if (std::find(dimensions_to_reduce.begin(), dimensions_to_reduce.end(),
                  input_dim) == dimensions_to_reduce.end()) {
      input_dims_to_keep.push_back(input_dim);
    }
  }

  // Sort the dimensions to keep from minor to major, to facilitate checking
  // whether another dimension is major or minor of them.
  std::sort(input_dims_to_keep.begin(), input_dims_to_keep.end(),
            [&input_shape](int64 dim_a, int64 dim_b) {
              return PositionInContainer(LayoutUtil::MinorToMajor(input_shape),
                                         dim_a) <
                     PositionInContainer(LayoutUtil::MinorToMajor(input_shape),
                                         dim_b);
            });
  // Now, if output rank is at least 1, `input_dims_to_keep.front()` is
  // minormost and `input_dims_to_keep.back()` is majormost.

  // If the dimensions to keep are minormost, emit a column reduction. As all
  // the dimensions to keep are contiguous, by prerequisite of
  // `EmitReductionToVector`, we only need to check whether the minormost
  // dimension of the input is to keep.
  if (ShapeUtil::IsEffectiveScalar(reduce->shape())) {
    return EmitReductionToScalar(kernel_thunk, reduce, input_shape, input_gens,
                                 init_value_gens, reducers,
                                 reduce_output_shapes, extra_output_gens);
  } else if (input_dims_to_keep.front() ==
             LayoutUtil::Minor(input_shape.layout(), 0)) {
    // Column reduction. Treat the result of "input" as a matrix whose width
    // is the most minor dimension and height the product of other dimensions,
    // and treat "reduce" as a column reduction of the input matrix.
    const int64 width = ShapeUtil::ElementsIn(reduce->shape());
    // "width" can be zero, so don't do
    //   height = ShapeUtil::ElementsIn(input_shape) / width;
    int64 height = 1;
    for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
         ++input_dim) {
      if (!std::count(input_dims_to_keep.begin(), input_dims_to_keep.end(),
                      input_dim)) {
        height *= input_shape.dimensions(input_dim);
      }
    }
    return EmitColumnReduction(kernel_thunk, height, width, reduce, input_shape,
                               input_gens, init_value_gens, reducers,
                               reduce_output_shapes, extra_output_gens);
  } else {
    // Reduce the row dimension of a matrix or reduce dimension 0 and 2 in a
    // 3D tensor. The size of dimension 1 (the height) is the size of the
    // dimension to keep, the size of dimension 0 (the depth) is the product
    // of dimensions that are more major than the dimension to keep, and the
    // size of dimension 2 (the width) is the product of more minor
    // dimensions.
    int64 depth = 1;
    int64 width = 1;
    for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
         ++input_dim) {
      if (PositionInContainer(LayoutUtil::MinorToMajor(input_shape),
                              input_dim) >
          PositionInContainer(LayoutUtil::MinorToMajor(input_shape),
                              input_dims_to_keep.back())) {
        depth *= input_shape.dimensions(input_dim);
      } else if (PositionInContainer(LayoutUtil::MinorToMajor(input_shape),
                                     input_dim) <
                 PositionInContainer(LayoutUtil::MinorToMajor(input_shape),
                                     input_dims_to_keep.front())) {
        width *= input_shape.dimensions(input_dim);
      }
    }
    const int64 height = ShapeUtil::ElementsIn(reduce->shape());
    return EmitRowReduction(kernel_thunk, depth, height, width, reduce,
                            input_shape, input_gens, init_value_gens, reducers,
                            reduce_output_shapes, extra_output_gens);
  }
}

Status IrEmitterUnnested::HandleReduce(HloInstruction* reduce) {
  // TODO(b/112040122): Support multi-output reduce.
  if (!ShapeUtil::IsArray(reduce->shape())) {
    return Unimplemented("Multi-output reduce is not supported on GPU");
  }
  auto input = reduce->operand(0);
  auto init_value = reduce->operand(1);
  absl::Span<const int64> dimensions_to_reduce(reduce->dimensions());
  HloComputation* reducer = reduce->to_apply();
  // HandleReduce specializes reduction from a multi-dimensional array to a 1D
  // array. The specialized version requires an initializer thunk that
  // initializes the output array to the initial value of the reduce.
  if (IsReductionToVector(*reduce)) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> initializer_thunk,
                        BuildInitializerThunk(reduce));
    std::vector<std::unique_ptr<Thunk>> thunks;
    thunks.push_back(std::move(initializer_thunk));
    std::unique_ptr<KernelThunk> kernel_thunk =
        BuildKernelThunk(reduce, /*implements_whole_instruction=*/false);

    TF_CHECK_OK(EmitReductionToVector(
        kernel_thunk.get(), reduce, input->shape(),
        {[&](const IrArray::Index& index) {
          return GetIrArray(*input, *reduce).EmitReadArrayElement(index, &b_);
        }},
        {[&](const IrArray::Index& index) {
          return GetIrArray(*init_value, *reduce)
              .EmitReadArrayElement(index, &b_);
        }},
        dimensions_to_reduce, {reducer}, {{}}, {}));

    thunks.push_back(std::move(kernel_thunk));

    std::unique_ptr<SequentialThunk> sequential_thunk =
        absl::make_unique<SequentialThunk>(std::move(thunks), reduce);
    AddThunkToThunkSequence(std::move(sequential_thunk));
    return Status::OK();
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
  const int64 rank = ShapeUtil::Rank(operand->shape());
  CHECK_EQ(rank, ShapeUtil::Rank(source->shape()));
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
    std::vector<int64> window_size;
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
    IrArray::Index operand_index(index_type, source_index.size());
    llvm::Value* in_bounds_condition = b_.getInt1(true);
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* strided_index = NSWMul(
          source_index[i], index_typed_constant(window.dimensions(i).stride()));
      operand_index[i] =
          NSWSub(NSWAdd(strided_index, window_index[i]),
                 index_typed_constant(window.dimensions(i).padding_low()));
      llvm::Value* index_condition = ICmpULT(
          operand_index[i],
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
    llvm::Value* operand_data =
        operand_array.EmitReadArrayElement(operand_index, &b_);
    Store(operand_data, selected_value_address);
    save_operand_index(operand_index);
    Store(b_.getInt1(true), initialized_flag_address);

    // If the initialized_flag is true, call the `select` function to
    // potentially update the selected value and index with the currently
    // visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.true_block, &b_);
    const Shape output_shape = ShapeUtil::MakeShape(PRED, {});
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
    IrArray::Index selected_index(operand_index.GetType());
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot =
          InBoundsGEP(selected_index_address, {b_.getInt32(i)});
      selected_index.push_back(Load(selected_index_address_slot));
    }
    llvm::Value* source_value_address =
        GetIrArray(*source, *select_and_scatter)
            .EmitArrayElementAddress(source_index, &b_);
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
  // TODO(b/112163966): Move trip count computation earlier in the pipeline.
  if (auto loop_trip_count = ComputeWhileLoopTripCount(xla_while)) {
    AddThunkToThunkSequence(BuildForThunk(xla_while, *loop_trip_count));
    VLOG(3) << "Built ForThunk for while: " << xla_while->name();
  } else {
    AddThunkToThunkSequence(BuildWhileThunk(xla_while));
    VLOG(3) << "Built WhileThunk for while: " << xla_while->name();
  }
  return Status::OK();
}

Status IrEmitterUnnested::HandleRng(HloInstruction* rng) {
  // Build the kernel to generate the random numbers.
  //
  // Unroll the kernel so that the duplicated computation that calculates the
  // 128 bit sample can be optimized away by LLVM.
  std::unique_ptr<KernelThunk> rng_thunk = BuildKernelThunk(
      rng, /*implements_whole_instruction=*/false, ComputeMaxUnrollFactor(rng));
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : rng->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArray(*operand, *rng).EmitReadArrayElement(index, &b_);
    };
  }
  TF_RETURN_IF_ERROR(EmitTargetElementLoopInThunk(
      *rng,
      GpuElementalIrEmitter(hlo_module_config_, module_, &b_,
                            GetNestedComputer())
          .MakeElementGenerator(rng, operand_to_generator),
      rng_thunk.get()));

  // Emit a kernel to increment the global state for Philox RNG algorithm.
  std::unique_ptr<Thunk> increment_seed_thunk =
      BuildKernelThunk(rng, /*implements_whole_instruction=*/false);
  llvm_ir::IncrementVariableForPhiloxRngState(1, module_, &b_);

  // Build the SequentialThunk for the RNG hlo.
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.reserve(2);
  thunks.push_back(std::move(rng_thunk));
  thunks.push_back(std::move(increment_seed_thunk));
  AddThunkToThunkSequence(
      absl::make_unique<SequentialThunk>(std::move(thunks), rng));

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
        /*mem_size=*/ShapeUtil::ByteSizeOf(operand->shape()), scatter));
  }

  thunks.push_back(
      BuildKernelThunk(scatter,
                       /*implements_whole_instruction=*/thunks.empty()));

  TF_RETURN_IF_ERROR(
      EmitScatter(thunks.back().get(), scatter,
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
    for (int64 i = 0, e = ShapeUtil::Rank(operand->shape()); i != e; ++i) {
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
    DCHECK_EQ(input_window_multidim.size(), ShapeUtil::Rank(operand->shape()));

    // Insert a 1 dimension at the end if index_vector_dim requests one.
    Shape scatter_indices_shape = scatter_indices->shape();
    if (dim_numbers.index_vector_dim() ==
        ShapeUtil::Rank(scatter_indices_shape)) {
      scatter_indices_shape.add_dimensions(1);
      scatter_indices_shape.mutable_layout()->add_minor_to_major(
          dim_numbers.index_vector_dim());
    }

    // Now load the indices corresponding to the current window from
    // scatter_indices.
    llvm_ir::IrArray::Index raw_scatter_index_index(input_scatter_multidim,
                                                    index.GetType());
    raw_scatter_index_index.InsertAt(dim_numbers.index_vector_dim(), nullptr);
    llvm::Value* is_in_bounds = b_.getTrue();
    for (int64 i = 0, e = dim_numbers.scatter_dims_to_operand_dims_size();
         i != e; ++i) {
      // Our index is stored along index_vector_dim, insert that into the lookup
      // index into scatter_indices.
      raw_scatter_index_index[dim_numbers.index_vector_dim()] =
          raw_scatter_index_index.GetConstantWithIndexType(i);

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
    llvm_ir::IrArray::Index input_window_index(input_window_multidim,
                                               index.GetType());
    HloInstruction* output_hlo =
        scatter->IsFused() ? scatter->parent()->FusionInstruction() : scatter;
    llvm::Value* output_address =
        GetIrArray(*output_hlo, *output_hlo)
            .EmitArrayElementAddress(input_window_index, &b_);
    llvm::Value* input_address = Alloca(llvm_ir::PrimitiveTypeToIrType(
        updates->shape().element_type(), module_));
    TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value, updates_gen(index));
    Store(input_ir_value, input_address);
    return EmitAtomicOperationForNestedComputation(
        *scatter->to_apply(), output_address, input_address);
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
  // In case there is a 'values' parameter that is a iota, we take note and use
  // it later to ensure a stable sort. Otherwise, we don't guarantee a stable
  // sort.
  int64 iota_values_parameter_index = -1;
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    if (i > 0 && sort->operand(i)->opcode() == HloOpcode::kIota &&
        ShapeUtil::ElementIsIntegral(sort->operand(i)->shape()) &&
        Cast<HloIotaInstruction>(sort->operand(i))->iota_dimension() ==
            dimension_to_sort) {
      iota_values_parameter_index = i;
    }
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
    IrArray keys_array;
    std::vector<IrArray> values_arrays;
    values_arrays.reserve(sort->operand_count() - 1);
    for (int64 i = 0; i < sort->operand_count(); ++i) {
      ShapeIndex shape_index =
          sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
      if (i == 0) {
        keys_array = GetIrArray(*sort, *sort, shape_index);
      } else {
        values_arrays.push_back(GetIrArray(*sort, *sort, shape_index));
      }
    }
    return llvm_ir::EmitSortInPlace(
        dimension_to_sort, keys_array, values_arrays,
        iota_values_parameter_index, IrName(sort), xor_masks, &b_,
        launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        kTileSize);
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

Status IrEmitterUnnested::HandleCrossReplicaSum(HloInstruction* crs) {
  if (hlo_module_config_.replica_count() != 1) {
    // TODO(b/33011107): Support nontrivial cross replica sum on GPU.
    return Unimplemented(
        "CrossReplicaSum with >1 replica is not implemented on GPU.");
  }

  // CRS with one operand and one replica is simply the identity function.
  // Buffer assignment expects a copy, so that's what we do.
  //
  // TODO(b/80100934): We would like to eliminate one-replica CRS nodes entirely
  // in algebraic-simplifier, but currently on some platforms
  // HloModuleConfig::num_replicas changes between when the module is compiled
  // and when it's run.
  if (crs->operand_count() == 1) {
    CHECK(ShapeUtil::IsArray(crs->operand(0)->shape()))
        << "Operands to cross-replica-sum must be arrays: " << crs->ToString();
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

Status IrEmitterUnnested::HandleAfterAll(HloInstruction* after_all) {
  return Status::OK();
}

Status IrEmitterUnnested::HandleInfeed(HloInstruction* infeed) {
  AddThunkToThunkSequence(BuildInfeedThunk(infeed));
  return Status::OK();
}

Status IrEmitterUnnested::HandleOutfeed(HloInstruction* outfeed) {
  AddThunkToThunkSequence(BuildOutfeedThunk(outfeed));
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

  std::sort(non_constant_buffers.begin(), non_constant_buffers.end(),
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
          llvm_ir::AsStringRef(llvm_ir::ConstantBufferAllocationToGlobalName(
              *slice.allocation())));
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
      non_constant_buffers, llvm_ir::AsString(kernel->getName()),
      implements_whole_instruction ? inst : nullptr, unroll_factor);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildHostToDeviceCopyThunk(
    const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  CHECK_EQ(HloOpcode::kConstant, operand->opcode());
  return absl::make_unique<HostToDeviceCopyThunk>(
      /*source_address=*/operand->literal().untyped_data(),
      /*destination_buffer=*/GetAllocationSlice(*inst),
      /*mem_size=*/
      llvm_ir::ByteSizeOf(operand->shape(),
                          ir_emitter_context_->llvm_module()->getDataLayout()),
      inst);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildDeviceToDeviceCopyThunk(
    const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  return absl::make_unique<DeviceToDeviceCopyThunk>(
      /*source_address=*/GetAllocationSlice(*operand),
      /*destination_buffer=*/GetAllocationSlice(*inst),
      /*mem_size=*/
      llvm_ir::ByteSizeOf(operand->shape(),
                          ir_emitter_context_->llvm_module()->getDataLayout()),
      inst);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildInfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kInfeed, inst->opcode());

  ShapeTree<BufferAllocation::Slice> slices(inst->shape());
  slices.ForEachMutableElement(
      [&](const ShapeIndex& index, BufferAllocation::Slice* slice) {
        *slice = ir_emitter_context_->buffer_assignment()
                     .GetUniqueSlice(inst, index)
                     .ConsumeValueOrDie();
      });
  return absl::make_unique<InfeedThunk>(slices, inst);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildOutfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kOutfeed, inst->opcode());

  ShapeTree<BufferAllocation::Slice> slices(inst->operand(0)->shape());
  slices.ForEachMutableElement(
      [&](const ShapeIndex& index, BufferAllocation::Slice* slice) {
        auto status_or_slice =
            ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                inst->operand(0), index);
        if (status_or_slice.ok()) {
          *slice = status_or_slice.ConsumeValueOrDie();
        }
      });
  return absl::make_unique<OutfeedThunk>(std::move(slices), inst);
}

namespace {
double GetScalarConstantAsDouble(const Literal& literal) {
  switch (literal.shape().element_type()) {
    case F16:
      return static_cast<double>(literal.Get<Eigen::half>({}));
    case F32:
      return literal.Get<float>({});
    case F64:
      return literal.Get<double>({});
    default:
      LOG(FATAL) << "Unsupported type.";
  }
}
}  // namespace

std::unique_ptr<Thunk> IrEmitterUnnested::BuildGemmThunk(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDot) {
    const HloInstruction* lhs = inst->operand(0);
    const HloInstruction* rhs = inst->operand(1);
    return absl::make_unique<GemmThunk>(
        GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
        GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
        GetAllocationSlice(*inst),  // The output buffer.
        lhs->shape(),               // The shape of LHS.
        rhs->shape(),               // The shape of RHS.
        inst->shape(),              // The shape of the output.
        1.0,                        // alpha.
        0.0,                        // beta.
        inst, /*implements_whole_instruction=*/true);
  }

  if (inst->opcode() == HloOpcode::kFusion) {
    CHECK_EQ(inst->fusion_kind(), HloInstruction::FusionKind::kOutput);
    const HloInstruction* output_fused_op = inst->fused_expression_root();

    double alpha_value = 1.0;
    const HloInstruction* bias = nullptr;
    const HloInstruction* dot = output_fused_op->operand(0);
    if (output_fused_op->opcode() == HloOpcode::kMultiply) {
      const HloInstruction* alpha = output_fused_op->operand(1);
      if (dot->opcode() != HloOpcode::kDot) {
        std::swap(dot, alpha);
      }
      if (alpha->opcode() == HloOpcode::kBroadcast) {
        alpha = alpha->operand(0);
      }
      if (alpha->opcode() == HloOpcode::kParameter) {
        alpha = inst->operand(alpha->parameter_number());
      }
      // TODO(b/74185543): Remove the following if block once we support fusion
      // with a non-constant as well. Then we will just always use the constant
      // on the device.
      if (alpha->opcode() == HloOpcode::kCopy) {
        alpha = alpha->operand(0);
      }
      alpha_value = GetScalarConstantAsDouble(alpha->literal());
    } else {
      // Fused bias add.
      CHECK_EQ(output_fused_op->opcode(), HloOpcode::kAdd);
      bias = output_fused_op->operand(1);
      if (dot->opcode() != HloOpcode::kDot) {
        std::swap(dot, bias);
      }
      bias = inst->operand(bias->parameter_number());
    }

    DCHECK(dot->opcode() == HloOpcode::kDot);
    const HloInstruction* lhs_parameter = StripTranspose(*dot->operand(0));
    const HloInstruction* rhs_parameter = StripTranspose(*dot->operand(1));
    DCHECK(lhs_parameter->opcode() == HloOpcode::kParameter &&
           rhs_parameter->opcode() == HloOpcode::kParameter);
    const HloInstruction* lhs =
        inst->operand(lhs_parameter->parameter_number());
    const HloInstruction* rhs =
        inst->operand(rhs_parameter->parameter_number());

    // The bias is passed inside the output buffer. If those buffers are shared
    // we can just use it, otherwise copy the bias values into the output buffer
    // first.
    if (bias != nullptr &&
        GetAllocationSlice(*bias) != GetAllocationSlice(*inst)) {
      std::vector<std::unique_ptr<Thunk>> thunks;
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          /*source_buffer=*/GetAllocationSlice(*bias),
          /*destination_buffer=*/GetAllocationSlice(*inst),
          /*mem_size=*/ShapeUtil::ByteSizeOf(inst->shape()), nullptr));
      thunks.push_back(absl::make_unique<GemmThunk>(
          GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
          GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
          GetAllocationSlice(*inst),  // The output buffer.
          lhs->shape(),               // The shape of LHS.
          rhs->shape(),               // The shape of RHS.
          inst->shape(),              // The shape of the output.
          alpha_value,                // alpha.
          1.0,                        // beta.
          inst, /*implements_whole_instruction=*/false));
      return absl::make_unique<SequentialThunk>(std::move(thunks), inst);
    }
    return absl::make_unique<GemmThunk>(
        GetAllocationSlice(*lhs),     // The buffer assigned to LHS.
        GetAllocationSlice(*rhs),     // The buffer assigned to RHS.
        GetAllocationSlice(*inst),    // The output buffer.
        lhs->shape(),                 // The shape of LHS.
        rhs->shape(),                 // The shape of RHS.
        inst->shape(),                // The shape of the output.
        alpha_value,                  // alpha.
        bias != nullptr ? 1.0 : 0.0,  // beta.
        inst, /*implements_whole_instruction=*/true);
  }

  LOG(FATAL) << "Cannot build a GemmThunk for " << inst->ToString();
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildFftThunk(
    const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  return absl::make_unique<FftThunk>(
      inst->fft_type(), inst->fft_length(),
      /*input_buffer=*/GetAllocationSlice(*operand),
      /*output_buffer=*/GetAllocationSlice(*inst),
      /*input_shape=*/operand->shape(),
      /*output_shape=*/inst->shape(), inst);
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
//     result buffers of the true and false computations.
//   * The buffer of operand 1 should share the allocation with the buffer of
//     the parameter 0 instruction of the true computation.
//   * The buffer of operand 2 should share the allocation with the buffer of
//     the parameter 0 instruction of the false computation.
Status CheckConditionalBuffersShareAllocation(
    const HloInstruction* conditional,
    const BufferAssignment& buffer_assignment) {
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      conditional->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) -> Status {
        TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
            conditional, conditional->true_computation()->root_instruction(),
            index, buffer_assignment));
        TF_RETURN_IF_ERROR(CheckHloBuffersShareAllocation(
            conditional, conditional->false_computation()->root_instruction(),
            index, buffer_assignment));
        return Status::OK();
      }));
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      conditional->operand(1)->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) -> Status {
        return CheckHloBuffersShareAllocation(
            conditional->operand(1),
            conditional->true_computation()->parameter_instruction(0), index,
            buffer_assignment);
      }));
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      conditional->operand(2)->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) -> Status {
        return CheckHloBuffersShareAllocation(
            conditional->operand(2),
            conditional->false_computation()->parameter_instruction(0), index,
            buffer_assignment);
      }));
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

  HloComputation* true_computation = hlo->true_computation();
  IrEmitterUnnested ir_emitter_true(hlo_module_config_, true_computation,
                                    ir_emitter_context_);
  TF_CHECK_OK(true_computation->Accept(&ir_emitter_true));

  HloComputation* false_computation = hlo->false_computation();
  IrEmitterUnnested ir_emitter_false(hlo_module_config_, false_computation,
                                     ir_emitter_context_);
  TF_CHECK_OK(false_computation->Accept(&ir_emitter_false));

  return absl::make_unique<ConditionalThunk>(
      GetAllocationSlice(*hlo->operand(0)),
      GetAllocationSlice(*hlo->operand(1)),
      GetAllocationSlice(*hlo->operand(2)),
      std::move(*ir_emitter_true.ConsumeThunkSequence()),
      std::move(*ir_emitter_false.ConsumeThunkSequence()), hlo);
}

Status IrEmitterUnnested::EmitTargetElementLoopInThunk(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator, KernelThunk* thunk) {
  int unroll_factor = thunk->unroll_factor();
  VLOG(3) << bindings_.ToString();

  const Shape& element_shape = hlo.IsMultiOutputFusion()
                                   ? ShapeUtil::GetSubshape(hlo.shape(), {0})
                                   : hlo.shape();
  VLOG(3) << "EmitTargetElementLoopInThunk "
          << ShapeUtil::HumanStringWithLayout(hlo.shape())
          << " for unroll_factor " << unroll_factor;
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      element_shape, ir_emitter_context_->device_description(), unroll_factor);
  UpdateLaunchDimensions(launch_dimensions, thunk,
                         ir_emitter_context_->llvm_module());
  if (!hlo.IsMultiOutputFusion()) {
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
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), output_arrays, &b_, module_);
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

Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  int unroll_factor = 1;
  // Unfused elementwise operations are usually memory bound, unroll them.
  if (hlo.IsElementwise() || hlo.opcode() == HloOpcode::kFusion) {
    unroll_factor = ComputeMaxUnrollFactor(&hlo);
  }

  std::unique_ptr<KernelThunk> kernel_thunk = BuildKernelThunk(
      &hlo, /*implements_whole_instruction=*/true, unroll_factor);
  Status emit_status =
      EmitTargetElementLoopInThunk(hlo, element_generator, kernel_thunk.get());
  thunk_sequence_->emplace_back(std::move(kernel_thunk));

  return emit_status;
}

std::vector<IrArray> IrEmitterUnnested::ConstructIrArrayForInputs(
    const HloInstruction& hlo) {
  std::vector<IrArray> param_arrays;
  param_arrays.reserve(hlo.operands().size());
  for (const HloInstruction* param : hlo.operands()) {
    param_arrays.push_back(GetIrArray(*param, hlo));
  }
  return param_arrays;
}


int IrEmitterUnnested::ConstructInputReducedShapeAndCastInputIrArrayToShape(
    const HloInstruction& hlo, const std::vector<IrArray>& param_arrays,
    const std::vector<llvm::Value*>& param_buffers,
    absl::Span<const int64> reduced_output_dims,
    std::vector<Shape>* param_reduced_shapes,
    std::vector<IrArray>* param_in_reduced_shape_arrays) {
  int64 num_params = hlo.operands().size();
  param_in_reduced_shape_arrays->reserve(num_params);
  param_reduced_shapes->reserve(num_params);
  for (int64 id = 0; id < num_params; ++id) {
    if (param_buffers[id] == nullptr) {
      param_reduced_shapes->push_back(Shape());
      param_in_reduced_shape_arrays->push_back(IrArray());
      continue;
    }
    const HloInstruction* param = hlo.operand(id);
    param_reduced_shapes->push_back(ShapeUtil::MakeShapeWithDescendingLayout(
        param->shape().element_type(),
        Permute({0, 2, 1}, reduced_output_dims)));
    param_in_reduced_shape_arrays->push_back(
        param_arrays[id].CastToShape((*param_reduced_shapes)[id], &b_));
  }
  return num_params;
}

namespace {

void EmitFullTile(const KernelMappingScheme* mapping_scheme,
                  const IrArray::Index& tile_origin_index,
                  llvm::IRBuilder<>* builder, llvm::Value* y, llvm::Value* x,
                  llvm::Type* index_ty,
                  const std::function<void(const IrArray::Index&, llvm::Value*,
                                           llvm::Value*)>& emit_elem_function) {
  int64 num_threads_x = mapping_scheme->GetNumberOfThreadsForDimensionX();
  int64 num_threads_y = mapping_scheme->GetNumberOfThreadsForDimensionY();
  int64 tile_size_x = mapping_scheme->GetTileSizeForDimensionX();
  int64 tile_size_y = mapping_scheme->GetTileSizeForDimensionY();
  for (int64 i = 0; i < tile_size_y; i += num_threads_y) {
    IrArray::Index source_idx_y =
        tile_origin_index.AddOffsetToDim(llvm::ConstantInt::get(index_ty, i),
                                         KernelMappingScheme::DimY, builder);
    llvm::Value* y_loc =
        builder->CreateAdd(llvm::ConstantInt::get(index_ty, i), y);
    for (int64 j = 0; j < tile_size_x; j += num_threads_x) {
      IrArray::Index source_idx =
          source_idx_y.AddOffsetToDim(llvm::ConstantInt::get(index_ty, j),
                                      KernelMappingScheme::DimX, builder);
      llvm::Value* x_loc =
          builder->CreateAdd(llvm::ConstantInt::get(index_ty, j), x);
      emit_elem_function(source_idx, y_loc, x_loc);
    }
  }
}

void EmitPartialTile(
    const KernelMappingScheme* mapping_scheme,
    const IrArray::Index& tile_origin_index, const string& loop_name,
    KernelSupportLibrary* ksl, llvm::IRBuilder<>* builder, llvm::Value* y,
    llvm::Value* x, llvm::Value* tile_height, llvm::Value* tile_width,
    llvm::Type* index_ty,
    const std::function<void(const IrArray::Index&, llvm::Value*,
                             llvm::Value*)>& emit_elem_function) {
  int64 num_threads_x = mapping_scheme->GetNumberOfThreadsForDimensionX();
  int64 num_threads_y = mapping_scheme->GetNumberOfThreadsForDimensionY();
  int64 tile_size_x = mapping_scheme->GetTileSizeForDimensionX();

  for (int64 j = 0; j < tile_size_x; j += num_threads_x) {
    IrArray::Index source_idx =
        tile_origin_index.AddOffsetToDim(llvm::ConstantInt::get(index_ty, j),
                                         KernelMappingScheme::DimX, builder);
    llvm::Value* x_loc =
        builder->CreateAdd(llvm::ConstantInt::get(index_ty, j), x);

    ksl->If(
        "x_in_tile", builder->CreateICmpULT(x_loc, tile_width), [&] {
          // tile_height_bound =
          //   ceil(tile_height / num_threads_y) * num_threads_y
          llvm::Value* ceiling_of_ratio = builder->CreateUDiv(
              builder->CreateAdd(tile_height, llvm::ConstantInt::get(
                                                  index_ty, num_threads_y - 1)),
              llvm::ConstantInt::get(index_ty, num_threads_y));
          llvm::Value* tile_height_bound = builder->CreateMul(
              ceiling_of_ratio,
              llvm::ConstantInt::get(index_ty, num_threads_y));
          ksl->For(
              loop_name, /*start=*/llvm::ConstantInt::get(index_ty, 0),
              /*end=*/tile_height_bound,
              /*step=*/llvm::ConstantInt::get(index_ty, num_threads_y),
              [&](llvm::Value* y_indvar) {
                llvm::Value* y_loc = builder->CreateAdd(y_indvar, y);
                ksl->If(
                    "y_in_tile", builder->CreateICmpULT(y_loc, tile_height),
                    [&] {
                      emit_elem_function(
                          source_idx.AddOffsetToDim(
                              y_indvar, KernelMappingScheme::DimY, builder),
                          y_loc, x_loc);
                    });
              });
        });
  }
}

// Emits code to process up to
// (tile_size_x/num_threads_x * tile_size_y/num_threads_y) elements in a tile,
// given `emit_elem_function` is the function to emit code to process one
// element, `y` and `x` are the intra-tile coordinates for the first element
// to process, and `index` is the index for the origin of the tile. Information
// about tile_size_x/y and num_threads_x/y are stored in `mapping_scheme`. Emits
// bounds check to ensure that each processed element is within the boundary
// defined by `tile_width` and `tile_height`.
void EmitTiledElementalCodeWithBoundsCheck(
    const KernelMappingScheme* mapping_scheme,
    const IrArray::Index& tile_origin_index, const string& loop_name,
    KernelSupportLibrary* ksl, llvm::IRBuilder<>* builder, llvm::Value* y,
    llvm::Value* x, llvm::Value* tile_height, llvm::Value* tile_width,
    const std::function<void(const IrArray::Index&, llvm::Value*,
                             llvm::Value*)>& emit_elem_function) {
  int64 tile_size_x = mapping_scheme->GetTileSizeForDimensionX();
  int64 tile_size_y = mapping_scheme->GetTileSizeForDimensionY();
  llvm::Type* index_ty = tile_width->getType();

  ksl->If(
      "full_tile",
      builder->CreateAnd(
          builder->CreateICmpEQ(llvm::ConstantInt::get(index_ty, tile_size_x),
                                tile_width),
          builder->CreateICmpEQ(llvm::ConstantInt::get(index_ty, tile_size_y),
                                tile_height)),
      [&] {
        EmitFullTile(mapping_scheme, tile_origin_index, builder, y, x, index_ty,
                     emit_elem_function);
      },
      [&] {
        EmitPartialTile(mapping_scheme, tile_origin_index, loop_name, ksl,
                        builder, y, x, tile_height, tile_width, index_ty,
                        emit_elem_function);
      });
}
}  // namespace

// Emits code to process a tensor element in a tile for the given kCopy HLO that
// performs a 0-2-1 transpose.
//
// index: The index for the first output element in the normalized tensor. The
//   normalized tensor is the resulting tensor after collapsing contiguous
//   dimensions that play the same role in the transpose.
// y_loc: The y coordinate within a tile.
// x_loc: The x coordinate within a tile.
// kernel_info: Other information to support the kernel code generation.
void IrEmitterUnnested::EmitTileElementForCopy(
    HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
    const KernelCodegenInfo* kernel_info, llvm::Value* y_loc,
    llvm::Value* x_loc) {
  llvm_ir::TiledParameterInfo* tiled_param_info =
      kernel_info->GetTiledParameterInfo();
  // TODO(jlebar): Add AA metadata to this load.
  llvm::Instruction* load_from_shmem_buffer =
      Load(GEP(tiled_param_info->GetBufferForParameter(0),
               {b_.getInt64(0), x_loc, y_loc}),
           "output_element");
  llvm_ir::IrArray output_array = GetIrArray(*hlo, *hlo);
  Shape output_reduced_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      hlo->shape().element_type(),
      kernel_info->GetKernelMappingScheme()->GetDimensionsInElements());
  // When the output_reduced_shape is a 0-2-1 transpose of the input shape,
  // the 0-2-1 transpose is achieved through EmitWriteArrayElement.
  output_array.CastToShape(output_reduced_shape, &b_)
      .EmitWriteArrayElement(index, load_from_shmem_buffer, &b_);
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
    const KernelCodegenInfo* kernel_info, llvm::Value* y_loc,
    llvm::Value* x_loc) {
  llvm_ir::TiledParameterInfo* tiled_param_info =
      kernel_info->GetTiledParameterInfo();
  std::vector<IrArray> output_arrays = ConstructIrArrayForOutputs(*hlo);
  GpuElementalIrEmitter elem_emitter(hlo_module_config_, module_, &b_,
                                     GetNestedComputer());
  FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(hlo),
                               &elem_emitter);
  tiled_param_info->set_y(y_loc);
  tiled_param_info->set_x(x_loc);
  fused_emitter.SetTiledParameterInfo(tiled_param_info);
  TF_CHECK_OK(hlo->fused_expression_root()->Accept(&fused_emitter));
  IrArray::Index untiled_index =
      kernel_info->GetKernelMappingScheme()->GetUnnormalizedIndex(
          index, output_arrays[0].GetShape());
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

// Emits a block of tiles, given a function object to emit one tile.
void IrEmitterUnnested::EmitBlock(const TileGenerator& emit_one_tile,
                                  const KernelCodegenInfo* kernel_info,
                                  KernelSupportLibrary& ksl,
                                  llvm::Type* index_ty) {
  KernelMappingScheme* mapping_scheme = kernel_info->GetKernelMappingScheme();
  absl::Span<const int64> dims_in_tile = mapping_scheme->GetDimensionsInTiles();
  absl::Span<const int64> dims_in_block =
      mapping_scheme->GetDimensionsInBlocks();
  absl::Span<const int64> block_sizes = mapping_scheme->GetBlockSizes();
  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  // Emit all the tiles for a given dimension in a tile block.
  auto emit_tiles_for_block_dim =
      [&](const string& loop_name, const IrArray::Index& starting_tile,
          int dim_id,
          const std::function<void(const IrArray::Index& tile_index)>
              emit_next_block_dim) {
        if (block_sizes[dim_id] == 1) {
          emit_next_block_dim(starting_tile);
        } else {
          llvm::Value* starting_tile_index_for_dim = starting_tile[dim_id];
          llvm::Value* block_size_for_dim =
              index_typed_constant(block_sizes[dim_id]);
          llvm::Value* block_id_for_dim =
              b_.CreateUDiv(starting_tile_index_for_dim, block_size_for_dim);
          llvm::Value* last_block_for_dim =
              index_typed_constant(dims_in_block[dim_id] - 1);
          llvm::Value* last_block_size_for_dim = index_typed_constant(
              dims_in_tile[dim_id] -
              (dims_in_block[dim_id] - 1) * block_sizes[dim_id]);
          llvm::Value* num_tiles_in_block =
              Select(ICmpEQ(last_block_for_dim, block_id_for_dim),
                     last_block_size_for_dim, block_size_for_dim);

          ksl.For(loop_name,
                  /*start=*/index_typed_constant(0),
                  /*end=*/num_tiles_in_block,
                  /*step=*/1, [&](llvm::Value* block_dim_induction_var) {
                    IrArray::Index tile_index = starting_tile.AddOffsetToDim(
                        block_dim_induction_var, dim_id, &b_);
                    emit_next_block_dim(tile_index);
                  });
        }
      };

  absl::Span<const int64> reduced_dims =
      mapping_scheme->GetDimensionsInElements();
  const bool block_contains_multi_tiles =
      mapping_scheme->GetNumberOfTilesInOneBlock() > 1;

  // Emit the tile with a given tile_index, by calculating the tight bounds for
  // each dimension of the tile and then calling emit_one_tile.
  auto emit_one_tile_for_tile_index = [&](const IrArray::Index& tile_index) {
    std::vector<llvm::Value*> output_tile_bounds(3);
    for (int i = KernelMappingScheme::DimY; i < KernelMappingScheme::DimTot;
         ++i) {
      int64 tile_size_for_dim = mapping_scheme->GetTileSizeForDimension(i);
      // Only last row or column may not have full size.
      llvm::Value* is_last_row =
          ICmpEQ(tile_index[i], index_typed_constant(dims_in_tile[i] - 1));
      int64 partial_row_size =
          reduced_dims[i] - (dims_in_tile[i] - 1) * tile_size_for_dim;
      output_tile_bounds[i] =
          Select(is_last_row, index_typed_constant(partial_row_size),
                 index_typed_constant(tile_size_for_dim), "tile_bound");
    }

    IrArray::Index tile_origin =
        mapping_scheme->GetElementIndexForTileOrigin(tile_index);
    emit_one_tile(tile_origin, output_tile_bounds, block_contains_multi_tiles);
  };

  const IrArray::Index starting_block =
      mapping_scheme->EmitBlockIndex(index_ty);
  const IrArray::Index starting_tile_for_dim_z =
      mapping_scheme->GetTileIndexForBlockOrigin(starting_block);

  // Emit the three dimensional block of tiles.
  emit_tiles_for_block_dim(
      "block_dim_z", starting_tile_for_dim_z, KernelMappingScheme::DimZ,
      [&](const IrArray::Index& starting_tile_for_dim_y) {
        emit_tiles_for_block_dim(
            "block_dim_y", starting_tile_for_dim_y, KernelMappingScheme::DimY,
            [&](const IrArray::Index& starting_tile_for_dim_x) {
              emit_tiles_for_block_dim("block_dim_x", starting_tile_for_dim_x,
                                       KernelMappingScheme::DimX,
                                       emit_one_tile_for_tile_index);
            });
      });
}

// Emits a kernel for the hlo instruction using the given kernel mapping scheme.
//
// unnested_hlo: The unnested hlo instruction for which the kernel is generated.
//   Currently, these hlo instructions are supported: kLoop fusion, kCopy.
// tiled_param_ids: The IDs for the parameters that are 0-2-1 transpose of
//   other tensors with the same dimensions and need to be tiled and tranposed.
// mapping_scheme: The tiling scheme to use.
// kernel_generator: Contains function objects for code generation, such as
//   element generator, block prologue and epilogue generators.
// kernel_info: Represent other information to support the code generation
//   of the tiled kernel for the hlo.
LaunchDimensions IrEmitterUnnested::EmitKernel(
    HloInstruction* unnested_hlo, absl::Span<const int64> tiled_param_ids,
    const KernelCodeGenerator& kernel_generator,
    KernelCodegenInfo* kernel_info) {
  KernelMappingScheme* mapping_scheme = kernel_info->GetKernelMappingScheme();

  std::vector<IrArray> param_arrays = ConstructIrArrayForInputs(*unnested_hlo);
  int64 num_params = param_arrays.size();
  // Allocate shared memory buffers to store the tiled inputs.
  std::vector<llvm::Value*> param_shmem_buffers(num_params, nullptr);
  for (int64 id : tiled_param_ids) {
    const HloInstruction* param = unnested_hlo->operand(id);
    param_shmem_buffers[id] =
        mapping_scheme->GetSharedMemoryBufferForElementType(
            llvm_ir::PrimitiveTypeToIrType(param->shape().element_type(),
                                           module_),
            IrName(unnested_hlo, StrCat("tile", id)));
    VLOG(3) << "Added shmem buffer for parameter " << id << ": "
            << llvm_ir::DumpToString(*param_shmem_buffers[id]);
  }

  CHECK_EQ(mapping_scheme->GetThreadsPerTile() % kWarpSize, 0);
  LaunchDimensions launch_dimensions = LaunchDimensions(
      mapping_scheme->GetNumberOfBlocks(), mapping_scheme->GetThreadsPerTile());
  llvm::Type* index_ty = GetIndexTypeForKernel(
      unnested_hlo, launch_dimensions.launch_bound(), &b_);
  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  // For multioutput fusion, one thread needs to output a tuple with pointers to
  // all the individual outputs.  We could do this at any point in the kernel,
  // but we do it at the beginning in the hopes of reducing register pressure,
  // since we touch threadIdx.x and blockIdx.x at the beginning of the kernel
  // *anyway*.
  if (unnested_hlo->IsMultiOutputFusion()) {
    KernelSupportLibrary{&b_}.If(
        "emit_mof_tuple", IsBlock0Thread0(&b_), [&] {
          llvm_ir::EmitTuple(GetIrArray(*unnested_hlo, *unnested_hlo),
                             ConstructIrArrayForOutputs(*unnested_hlo), &b_,
                             module_);
        });
  }

  // For each tiled parameter, cast its input IrArray to the corresponding
  // reduced shape and keep the reduced shape live during IR emission.
  std::vector<IrArray> param_in_reduced_shape_arrays;
  std::vector<Shape> param_reduced_shapes;
  absl::Span<const int64> reduced_dims =
      mapping_scheme->GetDimensionsInElements();
  int num_shapes = ConstructInputReducedShapeAndCastInputIrArrayToShape(
      *unnested_hlo, param_arrays, param_shmem_buffers, reduced_dims,
      &param_reduced_shapes, &param_in_reduced_shape_arrays);
  DCHECK_EQ(num_shapes, num_params);

  // Calculate the starting element coordinate within a tile for the current
  // thread, (y, x) from thread_id.
  llvm::Value* x;
  llvm::Value* y;
  std::tie(y, x) = mapping_scheme->EmitThreadYXCoordinate(index_ty);

  kernel_info->SetLaneId(
      mapping_scheme->GetNumberOfThreadsForDimensionX() == kWarpSize ? x
                                                                     : nullptr);

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);
  // Curry a few parameters to EmitTiledElementalCodeWithBoundsCheck.
  auto emit_tiled_elemental_code_with_bounds_check =
      [&](const IrArray::Index& index, const string& loop_name,
          llvm::Value* tile_height, llvm::Value* tile_width,
          const std::function<void(const IrArray::Index&, llvm::Value*,
                                   llvm::Value*)>& emit_elem_function) {
        EmitTiledElementalCodeWithBoundsCheck(mapping_scheme, index, loop_name,
                                              &ksl, &b_, y, x, tile_height,
                                              tile_width, emit_elem_function);
      };

  auto emit_one_tile = [&](const IrArray::Index& output_tile_origin,
                           absl::Span<llvm::Value* const> output_tile_bounds,
                           bool block_contains_multi_tiles) {
    // Calculate the input tile origin from the output tile origin.
    const IrArray::Index input_tile_origin(
        Permute({0, 2, 1}, output_tile_origin.multidim()));

    const IrArray::Index input_index =
        input_tile_origin.AddOffsetToDim(x, KernelMappingScheme::DimX, &b_)
            .AddOffsetToDim(y, KernelMappingScheme::DimY, &b_);

    // Copy input parameter values to shared memory buffers:
    // tile[y, x] = input[index]
    // Note that tile_width and tile_height are flipped here because we are
    // reading a transposed tile.
    emit_tiled_elemental_code_with_bounds_check(
        input_index, "input", output_tile_bounds[2], output_tile_bounds[1],
        [&](const IrArray::Index& index, llvm::Value* y_loc,
            llvm::Value* x_loc) {
          for (int64 id : tiled_param_ids) {
            IrArray& input_in_logical_shape = param_in_reduced_shape_arrays[id];
            llvm::Value* shmem_buffer = param_shmem_buffers[id];
            // TODO(jlebar): Add AA metadata to this store.  Tile buffers are
            // global variables, so LLVM can't infer much about it.
            Store(input_in_logical_shape.EmitReadArrayElement(index, &b_,
                                                              "input_element"),
                  GEP(shmem_buffer, {index_typed_constant(0), y_loc, x_loc}));
          }
        });

    // If shared memory transpose is needed, wait for all threads to reach this
    // point, lest we copy a value from tile to output before the other thread
    // copies it from input to tile. This is `__syncthreads` in CUDA.
    if (!tiled_param_ids.empty()) {
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {}, &b_);
    }

    llvm_ir::TiledParameterInfo tiled_param_info(param_shmem_buffers, y, x);
    kernel_info->SetTiledParamInfo(&tiled_param_info);

    const IrArray::Index output_index =
        output_tile_origin.AddOffsetToDim(x, KernelMappingScheme::DimX, &b_)
            .AddOffsetToDim(y, KernelMappingScheme::DimY, &b_);

    // Write to output[index] by emitting code like normal, except that values
    // for the tiled parameters are read from the shmem buffers.
    emit_tiled_elemental_code_with_bounds_check(
        output_index, "output", output_tile_bounds[1], output_tile_bounds[2],
        [&](const IrArray::Index& index, llvm::Value* y_loc,
            llvm::Value* x_loc) {
          kernel_generator.GetTileElementGenerator()(unnested_hlo, index,
                                                     kernel_info, y_loc, x_loc);
        });
    // If a tile block contains multiple tiles and shared memory buffers are
    // used, we need to wait for all threads to finish using the shared memory
    // buffer for the current tile before we move on to process the next tile
    // and overwrite the shared memory buffers.
    if (block_contains_multi_tiles && !tiled_param_ids.empty()) {
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {}, &b_);
    }
  };

  const BlockPrologueGenerator& block_prologue_generator =
      kernel_generator.GetBlockPrologueGenerator();
  if (block_prologue_generator) {
    block_prologue_generator(unnested_hlo, kernel_info);
  }

  EmitBlock(std::move(emit_one_tile), kernel_info, ksl, index_ty);

  const BlockEpilogueGenerator& block_epilogue_generator =
      kernel_generator.GetBlockEpilogueGenerator();
  if (block_epilogue_generator) {
    block_epilogue_generator(unnested_hlo, kernel_info);
  }

  return launch_dimensions;
}

// Emits a kernel for the given hlo instruction using a tiled 0-2-1 transpose
// algorithm to improve the memory access patterns for the input parameters
// with a shape that is a 0-2-1 transpose of the output tensor shape.
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
LaunchDimensions IrEmitterUnnested::EmitHlo021Tile(
    HloInstruction* hlo, absl::Span<const int64> reduced_output_dims,
    absl::Span<const int64> tiled_param_ids) {
  constexpr int kNumRows = 4;
  KernelMappingScheme mapping_scheme(
      reduced_output_dims, /*tile_size_y=*/kWarpSize,
      /*tile_size_x=*/kWarpSize, /*req_block_sizes=*/{1, 1, 1},
      /*num_threads_y=*/kNumRows,
      /*num_threads_x=*/kWarpSize, &b_);
  TileElementGenerator element_generator;
  if (hlo->opcode() == HloOpcode::kCopy) {
    element_generator = [&](HloInstruction* hlo,
                            const llvm_ir::IrArray::Index& index,
                            const KernelCodegenInfo* kernel_info,
                            llvm::Value* y_loc, llvm::Value* x_loc) {
      EmitTileElementForCopy(hlo, index, kernel_info, y_loc, x_loc);
    };
  } else {
    DCHECK_EQ(hlo->opcode(), HloOpcode::kFusion);
    element_generator = [&](HloInstruction* hlo,
                            const llvm_ir::IrArray::Index& index,
                            const KernelCodegenInfo* kernel_info,
                            llvm::Value* y_loc, llvm::Value* x_loc) {
      EmitTileElementForFusion(hlo, index, kernel_info, y_loc, x_loc);
    };
  }
  KernelCodegenInfo kernel_info(&mapping_scheme);
  KernelCodeGenerator kernel_generator(std::move(element_generator));
  return EmitKernel(hlo, tiled_param_ids, kernel_generator, &kernel_info);
}

namespace {
// Returns true to indicate it is safe to use the tile based shared memory
// transpose implementation to implement the kernel for the instruction.
//
// An instruction is not safe for such an implementation if it can change the
// element order of a tensor without changing the dimension of the tensor, and
// the instruction has a corresponding elemental_ir_emitter.
bool IsInstructionSafeForTileBasedTranspose(const HloInstruction* hlo) {
  auto is_safe_for_tile_based_transpose = [&](const HloInstruction* instr) {
    HloOpcode opcode = instr->opcode();
    CHECK_NE(opcode, HloOpcode::kFusion);
    return (opcode != HloOpcode::kReverse && opcode != HloOpcode::kGather);
  };

  if (hlo->opcode() == HloOpcode::kFusion) {
    return absl::c_all_of(hlo->fused_instructions_computation()->instructions(),
                          is_safe_for_tile_based_transpose);
  }

  return is_safe_for_tile_based_transpose(hlo);
}
}  // namespace

bool IrEmitterUnnested::CheckAndEmitHloWithTile021(HloInstruction* hlo) {
  HloOpcode opcode = hlo->opcode();
  CHECK(opcode == HloOpcode::kFusion || opcode == HloOpcode::kCopy);
  CHECK(opcode != HloOpcode::kFusion ||
        hlo->fusion_kind() == HloInstruction::FusionKind::kLoop)
      << "Only loop fusions are supported.";

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
        llvm_ir::FindTranspose021(operand->shape(), output_shape);
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

  if (!IsInstructionSafeForTileBasedTranspose(hlo)) {
    return false;
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

  VLOG(3) << "EmitHlo021Tile Emitting hlo tile 0-2-1" << hlo->ToString();
  std::unique_ptr<KernelThunk> kernel_thunk =
      BuildKernelThunk(hlo, /*implements_whole_instruction=*/true);
  const LaunchDimensions launch_dimensions =
      EmitHlo021Tile(hlo, *reduced_dims_021, params_012);
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());
  AddThunkToThunkSequence(std::move(kernel_thunk));

  return true;
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
    llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
        global_type, /*isConstant=*/should_emit_initializer,
        llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/initializer,
        llvm_ir::AsStringRef(
            llvm_ir::ConstantBufferAllocationToGlobalName(allocation)));
    global_for_const->setAlignment(kConstantBufferAlignBytes);
    ir_emitter_context_->llvm_module()->getGlobalList().push_back(
        global_for_const);
  }

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
