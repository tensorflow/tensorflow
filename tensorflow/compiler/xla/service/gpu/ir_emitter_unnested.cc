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

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_runner.h"
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
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/tuple_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_transformer.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

using llvm_ir::IrName;
using tensorflow::gtl::ArraySlice;
using tensorflow::gtl::InlinedVector;
using tensorflow::gtl::nullopt;
using tensorflow::gtl::optional;
using tensorflow::strings::StrCat;

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

namespace {
bool ImplementedAsHostToDeviceMemcpy(const BufferAssignment& buffer_assignment,
                                     const HloInstruction& hlo) {
  // `hlo` needs to satisfy the following conditions to be implemented as a
  // host-to-device cuMemcpy.
  //
  // 1. `hlo` is a kCopy instruction.
  // 2. `hlo`'s only operand is a kConstant instruction.
  // 3. `hlo` and its operand have the same shape (thus the same layout too).
  // 4. The address of `hlo`'s buffer is known at runtime (without dereferencing
  //    pointers in a tuple).
  return hlo.opcode() == HloOpcode::kCopy &&
         hlo.operand(0)->opcode() == HloOpcode::kConstant &&
         ShapeUtil::Equal(hlo.operand(0)->shape(), hlo.shape()) &&
         buffer_assignment.GetUniqueTopLevelSlice(&hlo).ok();
}

bool ImplementedAsDeviceToDeviceMemcpy(
    const BufferAssignment& buffer_assignment, const HloInstruction& hlo) {
  // `hlo` needs to satisfy three conditions to be implemented as a
  // device-to-device cuMemcpy.
  //
  // 1. `hlo` is a kCopy instruction.
  // 2. `hlo` and its operand have the same shape (thus the same layout too).
  // 3. `hlo` and its operand have a statically-known buffer assignment
  //     (constants do not, for instance), which means the source buffer also
  //     resides on the device.
  return hlo.opcode() == HloOpcode::kCopy &&
         ShapeUtil::Equal(hlo.operand(0)->shape(), hlo.shape()) &&
         buffer_assignment.GetUniqueTopLevelSlice(&hlo).ok() &&
         buffer_assignment.GetUniqueTopLevelSlice(hlo.operand(0)).ok();
}
}  // namespace

llvm::Function* IrEmitterUnnested::BuildKernelPrototype(
    const HloInstruction& inst,
    tensorflow::gtl::ArraySlice<const BufferAllocation*> args) {
  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(inst.name()));

  // Create the kernel and add it to the module.
  llvm::Module* module = ir_emitter_context_->llvm_module();
  llvm::LLVMContext& context = module->getContext();
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(args.size(), ir_builder_.getInt8PtrTy()),
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
    kernel->addParamAttr(
        arg_no, llvm::Attribute::get(context, llvm::Attribute::Alignment,
                                     kCudaMallocAlignBytes));

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
                llvm::ConstantAsMetadata::get(ir_builder_.getInt32(1))}));

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  ir_builder_.SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

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
                                  llvm::IRBuilder<>* ir_builder) {
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

  llvm::Type* i64_ty = ir_builder->getInt64Ty();
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
  if (!c_all_of(unnested_hlo->operands(), hlo_shape_in_range)) {
    return i64_ty;
  }

  // Check the size of the internal result tensors
  if (unnested_hlo->opcode() == HloOpcode::kFusion) {
    if (!c_all_of(
            unnested_hlo->fused_instructions_computation()->instructions(),
            hlo_shape_in_range)) {
      return i64_ty;
    }
  }

  return ir_builder->getInt32Ty();
}

}  // namespace

Status IrEmitterUnnested::DefaultAction(HloInstruction* hlo) {
  int unroll_factor = 1;
  // Unfused elementwise operations are usually memory bound, unroll them.
  if (hlo->IsElementwise()) {
    unroll_factor = ComputeMaxUnrollFactor(hlo);
  }

  thunk_sequence_->emplace_back(BuildKernelThunk(hlo, unroll_factor));
  return IrEmitter::DefaultAction(hlo);
}

Status IrEmitterUnnested::HandleDot(HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_batch_dimensions_size() > 0 ||
      dnums.rhs_batch_dimensions_size() > 0) {
    return Unimplemented("Dot with batch dimensions not implemented.");
  }
  if (ImplementedAsGemm(*dot)) {
    thunk_sequence_->emplace_back(BuildGemmThunk(dot));
    return Status::OK();
  }
  thunk_sequence_->emplace_back(BuildKernelThunk(dot));
  return IrEmitter::HandleDot(dot);
}

Status IrEmitterUnnested::HandleConditional(HloInstruction* conditional) {
  thunk_sequence_->emplace_back(BuildConditionalThunk(conditional));
  return Status::OK();
}

Status IrEmitterUnnested::HandleConvolution(HloInstruction* convolution) {
  thunk_sequence_->emplace_back(BuildKernelThunk(convolution));
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

    thunk_sequence_->emplace_back(
        MakeUnique<CudnnBatchNormForwardInferenceThunk>(
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
    thunk_sequence_->emplace_back(
        MakeUnique<CudnnBatchNormForwardTrainingThunk>(
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
    thunk_sequence_->emplace_back(MakeUnique<CudnnBatchNormBackwardThunk>(
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
    const auto& lhs_shape = custom_call->operand(0)->shape();
    const auto& rhs_shape = custom_call->operand(1)->shape();
    const auto& conv_result_shape = custom_call->shape().tuple_shapes(0);
    auto lhs_slice = GetAllocationSlice(*custom_call->operand(0));
    auto rhs_slice = GetAllocationSlice(*custom_call->operand(1));
    auto tuple_result_slice = GetAllocationSlice(*custom_call);
    auto conv_result_slice = assn.GetUniqueSlice(custom_call, {0}).ValueOrDie();
    auto scratch_slice = assn.GetUniqueSlice(custom_call, {1}).ValueOrDie();

    TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                        custom_call->backend_config<CudnnConvBackendConfig>());
    const auto& target = custom_call->custom_call_target();
    std::unique_ptr<ConvolutionThunk> thunk;
    if (target == kCudnnConvForwardCallTarget) {
      thunk = MakeUnique<ConvolutionThunk>(
          CudnnConvKind::kForward,
          /*input_buffer=*/lhs_slice,
          /*filter_buffer=*/rhs_slice,
          /*output_buffer=*/conv_result_slice,
          /*tuple_result_buffer=*/tuple_result_slice,
          /*scratch_buffer=*/scratch_slice,
          /*input_shape=*/lhs_shape,
          /*filter_shape=*/rhs_shape,
          /*output_shape=*/conv_result_shape,  //
          custom_call->window(), custom_call->convolution_dimension_numbers(),
          backend_config.algorithm(), backend_config.tensor_ops_enabled(),
          custom_call);
    } else if (target == kCudnnConvBackwardInputCallTarget) {
      thunk = MakeUnique<ConvolutionThunk>(
          CudnnConvKind::kBackwardInput,
          /*input_buffer=*/conv_result_slice,
          /*filter_buffer=*/rhs_slice,
          /*output_buffer=*/lhs_slice,
          /*tuple_result_buffer=*/tuple_result_slice,
          /*scratch_buffer=*/scratch_slice,
          /*input_shape=*/conv_result_shape,
          /*filter_shape=*/rhs_shape,
          /*output_shape=*/lhs_shape,  //
          custom_call->window(), custom_call->convolution_dimension_numbers(),
          backend_config.algorithm(), backend_config.tensor_ops_enabled(),
          custom_call);
    } else if (target == kCudnnConvBackwardFilterCallTarget) {
      thunk = MakeUnique<ConvolutionThunk>(
          CudnnConvKind::kBackwardFilter,
          /*input_buffer=*/lhs_slice,
          /*filter_buffer=*/conv_result_slice,
          /*output_buffer=*/rhs_slice,
          /*tuple_result_buffer=*/tuple_result_slice,
          /*scratch_buffer=*/scratch_slice,
          /*input_shape=*/lhs_shape,
          /*filter_shape=*/conv_result_shape,
          /*output_shape=*/rhs_shape,  //
          custom_call->window(), custom_call->convolution_dimension_numbers(),
          backend_config.algorithm(), backend_config.tensor_ops_enabled(),
          custom_call);
    } else {
      LOG(FATAL) << "Unexpected custom call target: "
                 << custom_call->custom_call_target();
    }

    thunk_sequence_->emplace_back(std::move(thunk));
    return Status::OK();
  }

  return IrEmitter::HandleCustomCall(custom_call);
}

Status IrEmitterUnnested::HandleFft(HloInstruction* fft) {
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(fft->operand(0)->shape().layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(fft->shape().layout()));
  thunk_sequence_->emplace_back(BuildFftThunk(fft));
  return Status::OK();
}

Status IrEmitterUnnested::HandleFusion(HloInstruction* fusion) {
  HloInstruction* root = fusion->fused_expression_root();
  // HandleFusion specializes reduction from a multi-dimensional array to a 1D
  // array. The specialized version requires a initializer thunk that
  // initializes the output array to the initial value of the reduce.
  if (HloInstruction::FusionKind::kInput == fusion->fusion_kind()) {
    switch (root->opcode()) {
      case HloOpcode::kTuple:
      case HloOpcode::kReduce: {
        VLOG(3) << "Emitting fused reduction to vector: " << fusion->ToString();
        std::vector<std::unique_ptr<Thunk>> thunks;
        ArraySlice<HloInstruction*> output_instructions =
            root->opcode() == HloOpcode::kTuple
                ? root->operands()
                : ArraySlice<HloInstruction*>(&root, 1);

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
        thunks.push_back(BuildKernelThunk(fusion));
        thunk_sequence_->emplace_back(
            MakeUnique<SequentialThunk>(std::move(thunks), fusion));
        std::vector<llvm_ir::IrArray> parameter_arrays;
        for (HloInstruction* operand : fusion->operands()) {
          parameter_arrays.push_back(GetIrArray(*operand, *fusion));
        }
        GpuElementalIrEmitter elemental_emitter(
            hlo_module_config_, ir_emitter_context_->llvm_module(),
            &ir_builder_, GetNestedComputer());
        FusedIrEmitter fused_emitter(parameter_arrays, &elemental_emitter);
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
        return EmitReductionToVector(first_reduce, input_shape, input_gens,
                                     init_value_gens,
                                     first_reduce->dimensions(), reducers,
                                     reduce_output_shapes, extra_output_gens);
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
    thunk_sequence_->emplace_back(BuildKernelThunk(fusion));
    std::vector<llvm_ir::IrArray> operand_arrays;
    for (HloInstruction* operand : fusion->operands()) {
      operand_arrays.push_back(GetIrArray(*operand, *fusion));
    }
    GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                            ir_emitter_context_->llvm_module(),
                                            &ir_builder_, GetNestedComputer());

    // Shape of the dynamic-update-slice's "update" operand.
    Shape update_shape = root->operand(1)->shape();

    // Array to write into.  Because this is an in-place operation, this is the
    // same as operand 0's array.
    llvm_ir::IrArray output_array = GetIrArray(*fusion, *fusion);

    LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
        update_shape, ir_emitter_context_->device_description());
    CHECK(Thunk::Kind::kKernel == LastThunk()->kind());
    UpdateLaunchDimensions(launch_dimensions,
                           static_cast<KernelThunk*>(LastThunk()),
                           ir_emitter_context_->llvm_module());

    return llvm_ir::EmitParallelFusedDynamicUpdateSliceInPlace(
        fusion, operand_arrays, output_array, &elemental_emitter,
        launch_dimensions, &ir_builder_);
  }
  if (ImplementedAsGemm(*fusion)) {
    thunk_sequence_->emplace_back(BuildGemmThunk(fusion));
    return Status::OK();
  }

  CHECK(fusion->fusion_kind() == HloInstruction::FusionKind::kLoop);
  int unroll_factor = ComputeMaxUnrollFactor(fusion);

  thunk_sequence_->emplace_back(BuildKernelThunk(fusion, unroll_factor));
  return IrEmitter::HandleFusion(fusion);
}

namespace {

// Returns the indices of the first elements of all consecutive subarrays of the
// given array. For example:
// ConsecutiveSegments({m, m+1, m+2, n, k, k+1}) = {0, 3, 4}
std::vector<size_t> ConsecutiveSegments(tensorflow::gtl::ArraySlice<int64> xs) {
  std::vector<size_t> is = {0};
  for (size_t i = 1; i < xs.size(); ++i) {
    if (1 != xs[i] - xs[i - 1]) {
      is.push_back(i);
    }
  }
  return is;
}

// Merges the sequences of dimensions of the given shape which start at the
// given indices `segs`.
Shape MergeDimensions(tensorflow::gtl::ArraySlice<size_t> segs,
                      const Shape& shape) {
  std::vector<int64> dimensions;
  for (size_t i = 1; i <= segs.size(); ++i) {
    dimensions.push_back(std::accumulate(
        shape.dimensions().begin() + segs[i - 1],
        shape.dimensions().begin() +
            (segs.size() == i ? shape.dimensions().size() : segs[i]),
        1, std::multiplies<int64>()));
  }
  return ShapeUtil::MakeShapeWithDescendingLayout(shape.element_type(),
                                                  dimensions);
}

// Returns whether the given shapes and permutation are a 0-2-1 transpose, and
// if so, the normalized and rank-reduced shapes. The shapes must have the same
// dimensions, so this considers layout only.
//
// This function recognizes higher-rank transposes which are elementwise
// equivalent to a 0-2-1 transpose.
std::tuple<bool, Shape, Shape> IsTranspose021(const Shape& a, const Shape& b) {
  CHECK(ShapeUtil::Compatible(a, b));
  std::vector<int64> perm(a.dimensions().size());
  {
    auto layout_a_orig = LayoutUtil::MinorToMajor(a);
    std::vector<int64> layout_a(layout_a_orig.rbegin(), layout_a_orig.rend());
    auto layout_b_orig = LayoutUtil::MinorToMajor(b);
    std::vector<int64> layout_b(layout_b_orig.rbegin(), layout_b_orig.rend());
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = PositionInContainer(layout_b, layout_a[i]);
    }
  }
  auto segs = ConsecutiveSegments(perm);
  Shape norm_a =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(a);
  Shape norm_b =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(b);
  if (3 == segs.size() && 0 == perm[0]) {
    Shape reduced_a = MergeDimensions(segs, norm_a);
    Shape reduced_b = ShapeUtil::MakeShapeWithDescendingLayout(
        b.element_type(),
        Permute({0, 2, 1}, AsInt64Slice(reduced_a.dimensions())));
    return std::make_tuple(true, reduced_a, reduced_b);
  }
  return std::make_tuple(false, ShapeUtil::MakeNil(), ShapeUtil::MakeNil());
}

// Returns whether the given shapes are potentially of a 0-2-1 transpose.
// As 0-2-1 is a self-inverse permutation, which shape is input or output is
// arbitrary.
bool AreShapesForTranspose021(const Shape& a, const Shape& b) {
  return 3 == b.dimensions().size() &&
         ShapeUtil::Compatible(
             ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(a),
             ShapeUtil::PermuteDimensions(
                 {0, 2, 1},
                 ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                     b)));
}

// Emits a tiled 0-2-1 transpose, assuming both input and output lain out from
// major to minor. The x- and y- dimensions are tiled in square tiles of edge
// length `tile_size`. Each thread block of `tile_size` x `num_rows` threads
// transposes one tile: each thread copies a row from the input to a shared
// memory tile, then copies a column from the shared memory tile to the output.
//
// `tile_size` should usually be same as warp size.
//
// Returns (number of tiles = number of thread blocks needed).
//
// TODO(b/33320379): Here each block transposes 1 tile. It may be more efficient
//                   to launch fewer blocks so each transposes many tiles, and
//                   in any case, the number of blocks we can launch is limited.
//
// This is the same algorithm in CUDA:
// https://github.com/tensorflow/tensorflow/blob/d2693c8a70567cc78b2e8a9ac8020d321620ca83/tensorflow/core/kernels/conv_ops_gpu_3.cu.cc#L189
int64 EmitTranspose021Tiled(llvm_ir::IrArray input, llvm_ir::IrArray output,
                            const int64 tile_size, const int64 num_rows,
                            llvm::IRBuilder<>* builder) {
  // Adds `addend` to the given `dim` of `index`.
  auto offset_dim = [builder](llvm_ir::IrArray::Index index,
                              llvm::Value* addend, int64 dim) {
    index[dim] = builder->CreateAdd(index[dim], addend);
    return index;
  };

  CHECK(AreShapesForTranspose021(input.GetShape(), output.GetShape()));

  Shape input_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          input.GetShape());
  Shape output_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          output.GetShape());
  input = input.CastToShape(input_shape, builder);
  output = output.CastToShape(output_shape, builder);

  llvm::Type* tile_type = llvm::ArrayType::get(
      llvm::ArrayType::get(input.GetElementLlvmType(), tile_size),
      // One extra here to avoid share memory bank conflict
      tile_size + 1);
  auto* tile = new llvm::GlobalVariable(
      *builder->GetInsertBlock()->getParent()->getParent(), tile_type,
      /*isConstant=*/false, llvm::GlobalValue::PrivateLinkage,
      llvm::UndefValue::get(tile_type), "tile", nullptr,
      llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/3 /* GPU shared memory */);

  // let x = threadIdx.x
  llvm::Value* x = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {}, builder);
  llvm_ir::AddRangeMetadata(0, num_rows * tile_size,
                            static_cast<llvm::Instruction*>(x));
  x = builder->CreateIntCast(x, builder->getInt64Ty(), /*isSigned=*/true,
                             "thread.id.x");

  // computing logical thread ids
  // logical_x = x % tile_size
  auto logical_x = builder->CreateURem(x, builder->getInt64(tile_size));

  // logical_y = x / tile_size
  auto logical_y = builder->CreateUDiv(x, builder->getInt64(tile_size));

  // `emit_cp` emits equivalent to following pseudocode:
  // if (tile_size == tile_width && tile_size == tile_height) {
  //   unroll for (i in range(0, tile_size, num_rows)) {
  //     emit_cp_element(index + {0, i, 0}, y + logical_y);
  //   }
  // } else if (x < tile_width) {
  //   tile_height_upperbound = ceil(tile_height / num_rows) * num_rows;
  //   for (i in range(0, tile_height_upperbound, num_rows)) {
  //     y_loc = i + logical_y;
  //     if (y_loc < tile_height)
  //      emit_cp_element(index + {0, i, 0}, y_loc);
  //   }
  // }
  //
  // We use this to emit both the copy from input to tile and the copy from tile
  // to output.
  //
  // `index` is the origin of the row or column in the input or output array.
  //
  // `emit_cp_element(index, y)` emits code to copy a single element between the
  // tile and the input or output array, where `y` is the `y`-position in the
  // tile, whether which is row or column is a function of whether we're copying
  // from input or to output, and `index` is the index into the input or output
  // array.
  auto emit_cp_tile = [builder, tile_size, &offset_dim, num_rows, logical_x,
                       logical_y](
                          std::function<void(const llvm_ir::IrArray::Index&,
                                             llvm::Value*)>
                              emit_cp_element,
                          llvm::Value* tile_width, llvm::Value* tile_height,
                          const llvm_ir::IrArray::Index& index,
                          const string& loop_name) {
    llvm_ir::LlvmIfData if_not_last_row = llvm_ir::EmitIfThenElse(
        builder->CreateAnd(
            builder->CreateICmpEQ(builder->getInt64(tile_size), tile_width),
            builder->CreateICmpEQ(builder->getInt64(tile_size), tile_height)),
        "not_last_row", builder);
    builder->SetInsertPoint(if_not_last_row.true_block->getTerminator());
    for (int64 i = 0; i < tile_size; i += num_rows) {
      auto source_idx = offset_dim(index, builder->getInt64(i), /*dim=*/1);
      auto y_loc = builder->CreateAdd(builder->getInt64(i), logical_y);
      emit_cp_element(source_idx, y_loc);
    }
    builder->SetInsertPoint(if_not_last_row.false_block->getTerminator());
    llvm_ir::LlvmIfData if_in_tile = llvm_ir::EmitIfThenElse(
        builder->CreateICmpULT(logical_x, tile_width), "x_in_tile", builder);
    builder->SetInsertPoint(if_in_tile.true_block->getTerminator());

    // tile_height_upper_bound = ceil(tile_height / num_rows) * num_rows
    auto tile_height_upper_bound = builder->CreateMul(
        builder->CreateUDiv(
            builder->CreateAdd(tile_height, builder->getInt64(num_rows - 1)),
            builder->getInt64(num_rows)),
        builder->getInt64(num_rows));

    auto loop = llvm_ir::ForLoop::EmitForLoop(
        loop_name, builder->getInt64(0), tile_height_upper_bound,
        builder->getInt64(num_rows), builder);
    llvm_ir::SetToFirstInsertPoint(loop->GetHeaderBasicBlock(), builder);
    builder->SetInsertPoint(loop->GetBodyBasicBlock()->getTerminator());

    auto y_loc = builder->CreateAdd(loop->GetIndVarValue(), logical_y);
    auto if_y_in_tile = llvm_ir::EmitIfThenElse(
        builder->CreateICmpULT(y_loc, tile_height), "y_in_tile", builder);
    builder->SetInsertPoint(if_y_in_tile.true_block->getTerminator());

    emit_cp_element(offset_dim(index, loop->GetIndVarValue(), /*dim=*/1),
                    y_loc);
    builder->SetInsertPoint(if_not_last_row.after_block->getTerminator());
  };

  auto input_dims_in_tiles = input_shape.dimensions();
  // Unpermuted dimensions are untiled.
  for (int i = 1; i < 3; ++i) {
    input_dims_in_tiles[i] =
        CeilOfRatio<int64>(input_dims_in_tiles[i], tile_size);
  }
  int64 num_tiles =
      std::accumulate(input_dims_in_tiles.begin(), input_dims_in_tiles.end(), 1,
                      std::multiplies<int64>());
  const llvm_ir::IrArray::Index input_tile_index(
      /*linear=*/builder->CreateIntCast(
          llvm_ir::AddRangeMetadata(
              0, num_tiles,
              static_cast<llvm::Instruction*>(llvm_ir::EmitCallToIntrinsic(
                  llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, {},
                  builder))),
          builder->getInt64Ty(), /*isSigned=*/true, "block.id.x"),
      ShapeUtil::MakeShapeWithDescendingLayout(
          PRED /*arbitrary*/, AsInt64Slice(input_dims_in_tiles)),
      builder);
  const llvm_ir::IrArray::Index input_tile_origin = ({
    llvm_ir::IrArray::Index index = input_tile_index;
    for (int i = 1; i < 3; ++i) {
      index[i] = builder->CreateMul(index[i], builder->getInt64(tile_size),
                                    "tile_origin." + std::to_string(i));
    }
    index;
  });
  const llvm_ir::IrArray::Index input_index =
      offset_dim(offset_dim(input_tile_origin, logical_x, /*dim=*/2), logical_y,
                 /*dim=*/1);
  std::vector<llvm::Value*> tile_dims(input_shape.dimensions().size());
  // Only last row or column may not have full size.
  for (int i = 1; i < 3; ++i) {
    tile_dims[i] = builder->CreateSelect(
        builder->CreateICmpEQ(input_tile_index[i],
                              builder->getInt64(input_dims_in_tiles[i] - 1)),
        builder->getInt64(input_shape.dimensions(i) -
                          (input_dims_in_tiles[i] - 1) * tile_size),
        builder->getInt64(tile_size), "tile_size");
  }

  // Load data from input memory to shared memory tile.
  emit_cp_tile(
      // tile[y, x] = input_array[index]
      [builder, tile, &input, logical_x](const llvm_ir::IrArray::Index& index,
                                         llvm::Value* y) {
        builder->CreateStore(
            input.EmitReadArrayElement(index, builder, "input_element"),
            builder->CreateGEP(tile, {builder->getInt64(0), y, logical_x}));
      },
      tile_dims[2], tile_dims[1], input_index, "input");

  // Wait for all threads to reach this point, lest we copy a value from tile to
  // output before the other thread copies it from input to tile.
  // This is `__syncthreads` in CUDA.
  llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {}, builder);

  const llvm_ir::IrArray::Index output_tile_index(
      Permute({0, 2, 1}, input_tile_index.multidim()));
  const llvm_ir::IrArray::Index output_tile_origin(
      Permute({0, 2, 1}, input_tile_origin.multidim()));
  const llvm_ir::IrArray::Index output_index =
      offset_dim(offset_dim(output_tile_origin, logical_x, /*dim=*/2),
                 logical_y, /*dim=*/1);

  // Store data from shared memory tile to output memory.
  emit_cp_tile(
      // output_array[index] = tile[x, y]
      [builder, tile, &output, logical_x](const llvm_ir::IrArray::Index& index,
                                          llvm::Value* y) {
        output.EmitWriteArrayElement(
            index,
            builder->CreateLoad(
                builder->CreateGEP(tile, {builder->getInt64(0), logical_x, y}),
                "output_element"),
            builder);
      },
      tile_dims[1], tile_dims[2], output_index, "output");

  return num_tiles;
}

}  // namespace

Status IrEmitterUnnested::HandleCopy(HloInstruction* copy) {
  if (ImplementedAsHostToDeviceMemcpy(ir_emitter_context_->buffer_assignment(),
                                      *copy)) {
    thunk_sequence_->emplace_back(BuildHostToDeviceCopyThunk(copy));
    return Status::OK();
  }
  if (ImplementedAsDeviceToDeviceMemcpy(
          ir_emitter_context_->buffer_assignment(), *copy)) {
    thunk_sequence_->emplace_back(BuildDeviceToDeviceCopyThunk(copy));
    return Status::OK();
  }
  bool is_transpose_021;
  Shape reduced_input_shape, reduced_output_shape;
  std::tie(is_transpose_021, reduced_input_shape, reduced_output_shape) =
      IsTranspose021(copy->operand(0)->shape(), copy->shape());
  if (is_transpose_021 &&
      reduced_input_shape.dimensions(1) >= kMinDimensionToTransposeTiled &&
      reduced_input_shape.dimensions(2) >= kMinDimensionToTransposeTiled) {
    thunk_sequence_->emplace_back(BuildKernelThunk(copy));
    VLOG(3) << "Emitting tiled 0-2-1 transposition";
    constexpr int64 tile_size = 32;
    constexpr int64 num_rows = 8;
    int64 num_tiles = EmitTranspose021Tiled(
        GetIrArray(*copy->operand(0), *copy)
            .CastToShape(reduced_input_shape, &ir_builder_),
        GetIrArray(*copy, *copy)
            .CastToShape(reduced_output_shape, &ir_builder_),
        tile_size, num_rows, &ir_builder_);
    UpdateLaunchDimensions(LaunchDimensions(num_tiles, num_rows * tile_size),
                           LastThunk(), ir_emitter_context_->llvm_module());
    return Status::OK();
  }

  return IrEmitter::HandleCopy(copy);
}

Status IrEmitterUnnested::EmitExtraOutputsForReduce(
    const HloInstruction* reduce, const llvm_ir::IrArray::Index& index,
    tensorflow::gtl::ArraySlice<
        std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  for (int i = 0; i != extra_output_gens.size(); ++i) {
    const HloInstruction* output = reduce->parent()->FusionInstruction();
    llvm::Value* extra_output_address =
        GetIrArray(*output, *output, extra_output_gens[i].second)
            .EmitArrayElementAddress(index, &ir_builder_,
                                     "extra_output_element_address");
    TF_ASSIGN_OR_RETURN(llvm::Value* const extra_output_ir_value,
                        extra_output_gens[i].first(index));
    ir_builder_.CreateStore(extra_output_ir_value, extra_output_address);
  }
  return Status::OK();
}

Status IrEmitterUnnested::EmitReductionToScalar(
    HloInstruction* reduce, const Shape& input_shape,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
    tensorflow::gtl::ArraySlice<HloComputation*> reducers,
    tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
    tensorflow::gtl::ArraySlice<
        std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
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

  llvm::Type* index_ty = GetIndexTypeForKernel(
      reduce,
      launch_dimensions.block_count() * launch_dimensions.threads_per_block(),
      &ir_builder_);

  auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
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
  // reduce_kernel<<<num_blocks, threads_per_block>>>();
  //
  auto loop_body_emitter =
      [=](const llvm_ir::IrArray::Index& tile_index) -> Status {
    const int num_reduces = reducers.size();
    llvm::Type* element_ir_type =
        llvm_ir::PrimitiveTypeToIrType(input_shape.element_type(), module_);
    std::vector<llvm::Value*> partial_reduction_result_addresses;
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* partial_reduction_result_address = ir_builder_.CreateAlloca(
          element_ir_type, /*ArraySize=*/nullptr,
          "partial_reduction_result." + llvm::Twine(i));
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const init_ir_value,
          init_value_gens[i](llvm_ir::IrArray::Index(index_ty)));
      ir_builder_.CreateStore(init_ir_value, partial_reduction_result_address);
      partial_reduction_result_addresses.push_back(
          partial_reduction_result_address);
    }

    llvm::Value* x_in_tiles = tile_index[0];
    x_in_tiles = ir_builder_.CreateZExtOrTrunc(x_in_tiles, index_ty);

    // Emit an inner for-loop that reduces the elements in the tile.
    auto emit_tile_element_loop = [=](bool tile_in_bounds) -> Status {
      std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
          llvm_ir::ForLoop::EmitForLoop(
              "element_id_in_tile", index_typed_const(0),
              index_typed_const(kTileSize), index_typed_const(1), &ir_builder_);

      // Emit the body of the partial reduction loop.
      llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(),
                                     &ir_builder_);
      llvm::Value* x = ir_builder_.CreateNSWAdd(
          ir_builder_.CreateNSWMul(x_in_tiles, index_typed_const(kTileSize)),
          tile_element_loop->GetIndVarValue());
      // Unless we know the tile is entirely in bounds, we have to emit a
      // x-in-bounds check before reading from the input.
      if (!tile_in_bounds) {
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            ir_builder_.CreateICmpULT(x, index_typed_const(num_elems)),
            "x_in_bounds", &ir_builder_);

        // Emit code that reads the input element and accumulates it to
        // the partial reduction result.
        llvm_ir::SetToFirstInsertPoint(if_data.true_block, &ir_builder_);
      }

      llvm_ir::IrArray::Index input_index(
          /*linear=*/x, input_shape, &ir_builder_);
      llvm::Value* input_address = ir_builder_.CreateAlloca(element_ir_type);
      for (int i = 0; i != num_reduces; ++i) {
        TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                            input_gens[i](input_index));
        ir_builder_.CreateStore(input_ir_value, input_address);
        TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
            *reducers[i],
            {partial_reduction_result_addresses[i], input_address},
            partial_reduction_result_addresses[i]));
      }
      return EmitExtraOutputsForReduce(reduce, input_index, extra_output_gens);
    };

    // x_end = kTileSize + x_in_tiles * kTileSize, i.e., the location that's
    // immediately beyond the tile.
    llvm::Value* x_end = ir_builder_.CreateNSWAdd(
        index_typed_const(kTileSize),
        ir_builder_.CreateNSWMul(x_in_tiles, index_typed_const(kTileSize)));
    // The tile is entirely in bound if all_threads_in_bounds or
    // x_end <= num_elems.
    llvm::Value* tile_in_bounds = ir_builder_.CreateOr(
        ir_builder_.CreateICmpULE(x_end, index_typed_const(num_elems)),
        ir_builder_.getInt1(all_threads_in_bounds));
    llvm_ir::LlvmIfData if_tile_in_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_bounds, "tile_in_bounds", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.true_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.false_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/false));

    // After the if-then-else statement on tile_in_bounds, emit calls to
    // shfl_down that accumulate the partial reduction results of all threads
    // from the warp.
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.after_block,
                                   &ir_builder_);
    int bit_width = llvm_ir::GetSizeInBits(element_ir_type);
    // bitcast cannot be applied to aggregate types (even packed ones), so we
    // instead bitcast addresses of load/store to intN* of the same bit-width.
    llvm::Type* shuffle_ir_type = element_ir_type->isStructTy()
                                      ? ir_builder_.getIntNTy(bit_width)
                                      : element_ir_type;
    for (int shuffle_distance = kWarpSize / 2; shuffle_distance >= 1;
         shuffle_distance /= 2) {
      llvm::Value* result_from_other_lane = ir_builder_.CreateAlloca(
          element_ir_type, nullptr, "result_from_other_lane");
      for (int i = 0; i != num_reduces; ++i) {
        llvm::Value* partial_reduction_result = ir_builder_.CreateLoad(
            ir_builder_.CreateBitCast(partial_reduction_result_addresses[i],
                                      shuffle_ir_type->getPointerTo()),
            "partial_reduction_result");
        ir_builder_.CreateStore(
            EmitShuffleDown(partial_reduction_result,
                            ir_builder_.getInt32(shuffle_distance),
                            &ir_builder_),
            ir_builder_.CreateBitCast(result_from_other_lane,
                                      shuffle_ir_type->getPointerTo()));
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
    llvm::Value* lane_id = ir_builder_.CreateURem(
        x_in_tiles, index_typed_const(kWarpSize), "lane_id");
    llvm_ir::LlvmIfData if_lane_id_is_zero_data = llvm_ir::EmitIfThenElse(
        ir_builder_.CreateICmpEQ(lane_id, index_typed_const(0)),
        "lane_id_is_zero", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_lane_id_is_zero_data.true_block,
                                   &ir_builder_);

    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* output_address =
          GetIrArray(*output, *output, reduce_output_shapes[i])
              .EmitArrayElementAddress(
                  llvm_ir::IrArray::Index(
                      /*linear=*/ir_builder_.getInt64(0),
                      ShapeUtil::GetSubshape(output->shape(),
                                             reduce_output_shapes[i]),
                      &ir_builder_),
                  &ir_builder_, "output_element_address");
      TF_RETURN_IF_ERROR(EmitAtomicOperationForNestedComputation(
          *reducers[i], output_address, partial_reduction_result_addresses[i]));
    }
    return Status::OK();
  };

  // Emit a parallel loop that iterates through all input tiles, one per thread.
  CHECK(LastThunk()->kind() == Thunk::Kind::kSequential);
  UpdateLaunchDimensions(
      launch_dimensions,
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &ir_builder_)
      .EmitLoop(IrName(reduce), index_ty);
}

Status IrEmitterUnnested::EmitColumnReduction(
    int64 height, int64 width, HloInstruction* reduce, const Shape& input_shape,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
    tensorflow::gtl::ArraySlice<HloComputation*> reducers,
    tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
    tensorflow::gtl::ArraySlice<
        std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
        extra_output_gens) {
  // Divide the input matrix into tiles of size Kx1. For example, when the
  // input matrix is 4x4 and K=2, the tiled matrix looks like
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
  // bandwidth.
  constexpr int64 kTileSize = 128;

  // If the height is not a multiple of the tile size, we pad the bottom of the
  // input matrix.
  const int64 height_in_tiles = CeilOfRatio(height, kTileSize);
  Shape tiled_input_shape = ShapeUtil::MakeShapeWithLayout(
      reduce->shape().element_type(), {height_in_tiles, width}, {1, 0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());

  // TODO(b/110211620): Convert to use i32 index_type when it is possible.
  llvm::Type* index_ty = ir_builder_.getInt64Ty();

  auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < height_in_tiles * width;
  //      linear_index += blockDim.x * gridDim.x) {
  //   y_in_tiles = linear_index / width;
  //   x = linear_index % width;
  //
  //   partial_result = init_value;
  //   if (height % kTileSize == 0 ||
  //       y_in_tiles * kTileSize + kTileSize <= height) {
  //     for (element_id_in_tile : range(kTileSize)) {
  //       y = y_in_tiles * kTileSize + element_id_in_tile;
  //       partial_result = Reducer(partial_result, input[y][x]);
  //     }
  //   } else {
  //     for (element_id_in_tile : range(kTileSize)) {
  //       y = y_in_tiles * kTileSize + element_id_in_tile;
  //       if (y < height) {
  //         partial_result = Reducer(partial_result, input[y][x]);
  //       }
  //     }
  //   }
  //   AtomicReducer(&output[x], partial_result);
  // }
  auto loop_body_emitter =
      [=](const llvm_ir::IrArray::Index& tile_index) -> Status {
    const int num_reduces = reducers.size();
    // Emit the loop body that reduces one tile.
    llvm::Type* element_ir_type =
        llvm_ir::PrimitiveTypeToIrType(input_shape.element_type(), module_);
    std::vector<llvm::Value*> partial_reduction_result_addresses;
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* partial_reduction_result_address = ir_builder_.CreateAlloca(
          element_ir_type, /*ArraySize=*/nullptr,
          "partial_reduction_result." + llvm::Twine(i));
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const init_ir_value,
          init_value_gens[i](llvm_ir::IrArray::Index(index_ty)));
      ir_builder_.CreateStore(init_ir_value, partial_reduction_result_address);
      partial_reduction_result_addresses.push_back(
          partial_reduction_result_address);
    }

    // Emit an inner for-loop that partially reduces the elements in the given
    // tile.
    llvm::Value* y_in_tiles = tile_index[0];
    llvm::Value* x = tile_index[1];

    y_in_tiles = ir_builder_.CreateZExtOrTrunc(y_in_tiles, index_ty);
    x = ir_builder_.CreateZExtOrTrunc(x, index_ty);

    auto emit_tile_element_loop = [=](bool tile_in_bounds) -> Status {
      std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
          llvm_ir::ForLoop::EmitForLoop(
              "element_id_in_tile", index_typed_const(0),
              index_typed_const(kTileSize), index_typed_const(1), &ir_builder_);

      // Emit the body of the partial reduction loop.
      llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(),
                                     &ir_builder_);
      llvm::Value* y = ir_builder_.CreateNSWAdd(
          ir_builder_.CreateNSWMul(y_in_tiles, index_typed_const(kTileSize)),
          tile_element_loop->GetIndVarValue());

      // Unless we know the tile is entirely in bounds, we have to emit a
      // y-in-bounds check before reading from the input.
      if (!tile_in_bounds) {
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            ir_builder_.CreateICmpULT(y, index_typed_const(height)),
            "y_in_bounds", &ir_builder_);

        // Emit code that reads the input element and accumulates it to
        // the partial reduction result.
        llvm_ir::SetToFirstInsertPoint(if_data.true_block, &ir_builder_);
      }
      llvm::Value* input_address = ir_builder_.CreateAlloca(element_ir_type);
      {
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
        const llvm_ir::IrArray::Index input_matrix_index(
            {y, x}, input_matrix_shape, &ir_builder_);
        const llvm_ir::IrArray::Index input_index =
            input_matrix_index
                .SourceIndexOfReshape(input_matrix_shape,
                                      normalized_input_shape, &ir_builder_)
                .SourceIndexOfTranspose(normalized_input_shape, input_shape,
                                        transpose_dimension_mapping,
                                        &ir_builder_);
        for (int i = 0; i != num_reduces; ++i) {
          TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                              input_gens[i](input_index));
          ir_builder_.CreateStore(input_ir_value, input_address);
          TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
              *reducers[i],
              {partial_reduction_result_addresses[i], input_address},
              partial_reduction_result_addresses[i]));
        }
        return EmitExtraOutputsForReduce(reduce, input_index,
                                         extra_output_gens);
      }
    };

    // y_end = kTileSize + y_in_tiles * kTileSize, i.e., the y location that's
    // immediately beyond the tile.
    llvm::Value* y_end = ir_builder_.CreateNSWAdd(
        index_typed_const(kTileSize),
        ir_builder_.CreateNSWMul(y_in_tiles, index_typed_const(kTileSize)));
    llvm::Value* tile_in_bounds = ir_builder_.CreateOr(
        ir_builder_.CreateICmpULE(y_end, index_typed_const(height)),
        ir_builder_.getInt1(height % kTileSize == 0));
    // The tile is entirely in bound if "height" is a multiple of kTileSize or
    // y_end <= height.
    llvm_ir::LlvmIfData if_tile_in_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_bounds, "tile_in_bounds", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.true_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.false_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/false));

    // After the if-then-else statement on tile_in_bounds, emit atomic
    // operations to accumulate the partial reduction result to the output
    // element.
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.after_block,
                                   &ir_builder_);
    const HloInstruction* output =
        reduce->IsFused() ? reduce->parent()->FusionInstruction() : reduce;
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* output_address =
          GetIrArray(*output, *output, reduce_output_shapes[i])
              .EmitArrayElementAddress(
                  llvm_ir::IrArray::Index(
                      x,
                      ShapeUtil::GetSubshape(output->shape(),
                                             reduce_output_shapes[i]),
                      &ir_builder_),
                  &ir_builder_, "output_element_address");
      TF_RETURN_IF_ERROR(EmitAtomicOperationForNestedComputation(
          *reducers[i], output_address, partial_reduction_result_addresses[i]));
    }
    return Status::OK();
  };

  // Emit a parallel loop that iterate through all input tiles.
  CHECK(LastThunk()->kind() == Thunk::Kind::kSequential);
  UpdateLaunchDimensions(
      launch_dimensions,
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &ir_builder_)
      .EmitLoop(IrName(reduce), index_ty);
}

static std::pair<int64, int64> ComputeTilingSchemeForReduction(
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
    int64 depth, int64 height, int64 width, HloInstruction* reduce,
    const Shape& input_shape,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
    tensorflow::gtl::ArraySlice<HloComputation*> reducers,
    tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
    tensorflow::gtl::ArraySlice<
        std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
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
  //       partial_result = reducer(partial_result, input[z][y][z]);
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
  //       for (int element_id_in_x_tile = 0;
  //            element_id_in_x_tile < x_tile_size;
  //            ++element_id_in_x_tile, x += warpSize) {
  //         partial_result = Reducer(partial_result, input[z][y][x]);
  //       }
  //     }
  //   } else {
  //     // The tile is partially in bounds.
  //     for (int element_id_in_z_tile = 0; element_id_in_z_tile < z_tile_size;
  //          ++element_id_in_z_tile) {
  //       z = z_in_tiles * z_tile_size + element_id_in_z_tile;
  //       for (int element_id_in_x_tile = 0; element_id_in_x_tile <
  //            x_tile_size; ++element_id_in_tile, x += warpSize) {
  //         if (x < width)
  //           partial_result = Reducer(partial_result, input[z][y][x]);
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
      ComputeTilingSchemeForReduction(depth, width, kWarpSize);

  // Round the width in tiles up to the nearest multiple of kWarpSize, so that
  // the use of shfl_down is valid.
  const int64 width_in_tiles =
      RoundUpToNearest(CeilOfRatio(width, x_tile_size), kWarpSize);
  Shape tiled_input_shape = ShapeUtil::MakeShapeWithLayout(
      reduce->shape().element_type(),
      {depth / z_tile_size, height, width_in_tiles}, {2, 1, 0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());
  llvm::Type* index_ty = GetIndexTypeForKernel(
      reduce,
      launch_dimensions.block_count() * launch_dimensions.threads_per_block(),
      &ir_builder_);

  auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  auto loop_body_emitter = [=](const llvm_ir::IrArray::Index& tile_index) {
    const int num_reduces = reducers.size();
    llvm::Type* element_ir_type = llvm_ir::PrimitiveTypeToIrType(
        input_shape.element_type(), ir_emitter_context_->llvm_module());
    std::vector<llvm::Value*> partial_reduction_result_addresses;
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* partial_reduction_result_address = ir_builder_.CreateAlloca(
          element_ir_type, /*ArraySize=*/nullptr,
          "partial_reduction_result." + llvm::Twine(i));
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const init_ir_value,
          init_value_gens[i](llvm_ir::IrArray::Index(index_ty)));
      ir_builder_.CreateStore(init_ir_value, partial_reduction_result_address);
      partial_reduction_result_addresses.push_back(
          partial_reduction_result_address);
    }

    llvm::Value* z_tile = tile_index[0];
    llvm::Value* y = tile_index[1];
    llvm::Value* x_tile = tile_index[2];

    x_tile = ir_builder_.CreateZExtOrTrunc(x_tile, index_ty);

    llvm::Value* warp_id =
        ir_builder_.CreateUDiv(x_tile, index_typed_const(kWarpSize), "warp_id");
    llvm::Value* lane_id =
        ir_builder_.CreateURem(x_tile, index_typed_const(kWarpSize), "lane_id");

    // The x-location of the last element in this z-x-tile.
    // last_x = lane_id + warpSize * (x_tile_size - 1 + warp_id * x_tile_size);
    llvm::Value* last_x = ir_builder_.CreateNSWAdd(
        lane_id, ir_builder_.CreateNSWMul(
                     index_typed_const(kWarpSize),
                     ir_builder_.CreateNSWAdd(
                         index_typed_const(x_tile_size - 1),
                         ir_builder_.CreateNSWMul(
                             warp_id, index_typed_const(x_tile_size)))));

    KernelSupportLibrary ksl(
        &ir_builder_,
        /*unroll_mode=*/xla::llvm_ir::UnrollMode::kFullyUnroll,
        /*prevent_vectorization=*/false);

    // Emit a for-loop that partially reduces the elements in the given
    // z-x-tile.
    auto emit_z_x_tile_element_loop = [&](bool x_tile_in_bounds,
                                          int64 x_tile_loop_bound) -> Status {
      auto emit_z_tile_element_loop = [&](llvm::Value* z_indvar) -> Status {
        llvm::Value* z = ir_builder_.CreateNSWAdd(
            z_indvar,
            ir_builder_.CreateNSWMul(index_typed_const(z_tile_size), z_tile));
        TF_RETURN_IF_ERROR(ksl.For(
            "x_tile",
            /*start=*/index_typed_const(0),
            /*end=*/index_typed_const(x_tile_loop_bound),
            /*step=*/1, [&](llvm::Value* x_indvar) -> Status {
              // x = lane_id +
              //     warpSize * (element_id_in_x_tile + warp_id * x_tile_size);
              llvm::Value* x = ir_builder_.CreateNSWAdd(
                  lane_id,
                  ir_builder_.CreateNSWMul(
                      index_typed_const(kWarpSize),
                      ir_builder_.CreateNSWAdd(
                          x_indvar, ir_builder_.CreateNSWMul(
                                        warp_id, llvm::ConstantInt::get(
                                                     index_ty, x_tile_size)))));

              // Unless we know the x-tile is entirely in bounds, we have to
              // emit a x-in-bounds check before reading from the input.
              if (!x_tile_in_bounds) {
                llvm_ir::LlvmIfData if_x_in_bounds_data =
                    llvm_ir::EmitIfThenElse(
                        ir_builder_.CreateICmpULT(x, index_typed_const(width)),
                        "x_in_bounds", &ir_builder_);
                // Points ir_builder_ to the then-block.
                llvm_ir::SetToFirstInsertPoint(if_x_in_bounds_data.true_block,
                                               &ir_builder_);
              }

              // Emit code that reads the input element and accumulates it
              // to the partial reduction result.
              llvm::Value* input_address =
                  ir_builder_.CreateAlloca(element_ir_type);
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
                const llvm_ir::IrArray::Index input_3d_tensor_index(
                    {z, y, x}, input_3d_tensor_shape, &ir_builder_);
                const llvm_ir::IrArray::Index input_index =
                    input_3d_tensor_index
                        .SourceIndexOfReshape(input_3d_tensor_shape,
                                              normalized_input_shape,
                                              &ir_builder_)
                        .SourceIndexOfTranspose(
                            normalized_input_shape, input_shape,
                            transpose_dimension_mapping, &ir_builder_);

                for (int i = 0; i != num_reduces; ++i) {
                  TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                                      input_gens[i](input_index));
                  ir_builder_.CreateStore(input_ir_value, input_address);
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

      return ksl.For("z_tile",
                     /*start=*/index_typed_const(0),
                     /*end=*/index_typed_const(z_tile_size),
                     /*step=*/1, emit_z_tile_element_loop);
    };

    llvm::Value* tile_in_bounds = ir_builder_.CreateOr(
        ir_builder_.getInt1(width % (x_tile_size * kWarpSize) == 0),
        ir_builder_.CreateICmpULT(last_x, index_typed_const(width)));

    TF_RETURN_IF_ERROR(
        ksl.If(tile_in_bounds,
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
                                      ? ir_builder_.getIntNTy(bit_width)
                                      : element_ir_type;
    for (int shuffle_distance = 16; shuffle_distance >= 1;
         shuffle_distance /= 2) {
      llvm::Value* result_from_other_lane = ir_builder_.CreateAlloca(
          element_ir_type, nullptr, "result_from_other_lane");
      for (int i = 0; i != num_reduces; ++i) {
        llvm::Value* partial_reduction_result = ir_builder_.CreateLoad(
            ir_builder_.CreateBitCast(partial_reduction_result_addresses[i],
                                      shuffle_ir_type->getPointerTo()),
            "partial_reduction_result");
        ir_builder_.CreateStore(
            EmitShuffleDown(partial_reduction_result,
                            ir_builder_.getInt32(shuffle_distance),
                            &ir_builder_),
            ir_builder_.CreateBitCast(result_from_other_lane,
                                      shuffle_ir_type->getPointerTo()));
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
        ir_builder_.CreateICmpEQ(lane_id, index_typed_const(0)),
        "lane_id_is_zero", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_lane_id_is_zero_data.true_block,
                                   &ir_builder_);
    for (int i = 0; i != num_reduces; ++i) {
      llvm::Value* output_address =
          GetIrArray(*output, *output, reduce_output_shapes[i])
              .EmitArrayElementAddress(
                  llvm_ir::IrArray::Index(
                      y,
                      ShapeUtil::GetSubshape(output->shape(),
                                             reduce_output_shapes[i]),
                      &ir_builder_),
                  &ir_builder_, "output_element_address");
      if (x_tile_size * z_tile_size < depth * width) {
        TF_RETURN_IF_ERROR(EmitAtomicOperationForNestedComputation(
            *reducers[i], output_address,
            partial_reduction_result_addresses[i]));
      } else {
        TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
            *reducers[i],
            {output_address, partial_reduction_result_addresses[i]},
            output_address));
      }
    }
    return Status::OK();
  };

  // Emit a parallel loop that iterates through every input tiles.
  CHECK(LastThunk()->kind() == Thunk::Kind::kSequential);
  UpdateLaunchDimensions(
      launch_dimensions,
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &ir_builder_)
      .EmitLoop(IrName(reduce), index_ty);
}

// Figures out whether `reduce` is a row or column reduction, and which
// dimensions to reduce, and calls either `EmitRowReduction` or
// `EmitColumnReduction` as appropriate.
// Prerequisite: all the dimensions to keep are contiguous in the input layout
//               and, if `reduce` is fused, the fused subgraph is pure
//               elementwise.
Status IrEmitterUnnested::EmitReductionToVector(
    HloInstruction* reduce, const Shape& input_shape,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
    tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    tensorflow::gtl::ArraySlice<HloComputation*> reducers,
    tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
    tensorflow::gtl::ArraySlice<
        std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
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
  if (input_dims_to_keep.empty()) {
    return EmitReductionToScalar(reduce, input_shape, input_gens,
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
    return EmitColumnReduction(height, width, reduce, input_shape, input_gens,
                               init_value_gens, reducers, reduce_output_shapes,
                               extra_output_gens);
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
    return EmitRowReduction(depth, height, width, reduce, input_shape,
                            input_gens, init_value_gens, reducers,
                            reduce_output_shapes, extra_output_gens);
  }
}

Status IrEmitterUnnested::HandleReduce(HloInstruction* reduce) {
  auto input = reduce->operand(0);
  auto init_value = reduce->operand(1);
  tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce(reduce->dimensions());
  HloComputation* reducer = reduce->to_apply();
  // HandleReduce specializes reduction from a multi-dimensional array to a 1D
  // array. The specialized version requires an initializer thunk that
  // ingitializes the output array to the initial value of the reduce.
  if (IsReductionToVector(*reduce) &&
      // NVPTX backend can't do atomic cmpxchg any narrower than 32 bits
      32 <= primitive_util::BitWidth(reduce->shape().element_type())) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> initializer_thunk,
                        BuildInitializerThunk(reduce));
    std::vector<std::unique_ptr<Thunk>> thunks;
    thunks.push_back(std::move(initializer_thunk));
    thunks.push_back(BuildKernelThunk(reduce));
    thunk_sequence_->emplace_back(
        MakeUnique<SequentialThunk>(std::move(thunks), reduce));

    return EmitReductionToVector(
        reduce, input->shape(), {[&](const llvm_ir::IrArray::Index& index) {
          return GetIrArray(*input, *reduce)
              .EmitReadArrayElement(index, &ir_builder_);
        }},
        {[&](const llvm_ir::IrArray::Index& index) {
          return GetIrArray(*init_value, *reduce)
              .EmitReadArrayElement(index, &ir_builder_);
        }},
        dimensions_to_reduce, {reducer}, {{}}, {});
  }

  thunk_sequence_->emplace_back(BuildKernelThunk(reduce));
  return IrEmitter::HandleReduce(reduce);
}

Status IrEmitterUnnested::HandleTuple(HloInstruction* tuple) {
  bool all_tuple_elements_have_buffer =
      c_all_of(tuple->operands(), [&](HloInstruction* tuple_element) {
        return ir_emitter_context_->buffer_assignment().HasTopLevelAllocation(
            tuple_element);
      });
  // Tuples (especially tuples that are the final result of a computation) can
  // be so huge that if we were to emit a kernel that took each tuple element as
  // a parameter, we would exceed the max allowable number of parameters to a
  // GPU kernel, b/31336476. As an optimization, if all tuple elements have a
  // buffer, we collect their buffer addresses in a host array, and then copy
  // that array to the tuple's buffer.
  //
  // Some tuple elements (e.g. const or bitcast of const) might not have a
  // buffer -- their contents are stored in code. In that case, we fall back to
  // emitting kernels which have access to their buffer addresses in code.
  if (all_tuple_elements_have_buffer) {
    std::vector<BufferAllocation::Slice> tuple_element_buffers;
    for (const HloInstruction* tuple_element : tuple->operands()) {
      tuple_element_buffers.push_back(GetAllocationSlice(*tuple_element));
    }
    thunk_sequence_->emplace_back(MakeUnique<TupleThunk>(
        tuple_element_buffers, GetAllocationSlice(*tuple), tuple));
    return Status::OK();
  }
  thunk_sequence_->emplace_back(BuildKernelThunk(tuple));
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
  thunks.push_back(BuildKernelThunk(select_and_scatter));
  thunk_sequence_->emplace_back(
      MakeUnique<SequentialThunk>(std::move(thunks), select_and_scatter));

  // TODO(b/31410564): Implement dilation rate for select-and-scatter.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for SelectAndScatter not implemented on GPU.");
  }

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      source->shape(), ir_emitter_context_->device_description());
  llvm::Type* index_type = GetIndexTypeForKernel(
      select_and_scatter, launch_dimensions.launch_bound(), &ir_builder_);
  auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
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
  auto loop_body_emitter =
      [=](const llvm_ir::IrArray::Index& source_index) -> Status {
    // Allocate space to keep the currently selected value, its index, and a
    // boolean flag if the value is initialized. The initialized_flag is set
    // false.
    llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_element_type,
                                       ir_emitter_context_->llvm_module()),
        "selected_value_address", &ir_builder_);
    llvm::Value* selected_index_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            index_type, index_typed_const(rank), "selected_index_address",
            &ir_builder_);
    llvm::Value* initialized_flag_address = llvm_ir::EmitAllocaAtFunctionEntry(
        ir_builder_.getInt1Ty(), "initialized_flag_address", &ir_builder_);
    ir_builder_.CreateStore(ir_builder_.getInt1(false),
                            initialized_flag_address);

    // Create the inner loop to iterate over the window.
    llvm_ir::ForLoopNest window_loops(IrName(select_and_scatter, "inner"),
                                      &ir_builder_, index_type);
    std::vector<int64> window_size;
    for (const auto& dim : window.dimensions()) {
      window_size.push_back(dim.size());
      CHECK_GT(dim.size(), 0);
    }
    const llvm_ir::IrArray::Index window_index = window_loops.AddLoopsForShape(
        ShapeUtil::MakeShape(operand_element_type, window_size), "window");
    llvm_ir::SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(),
                                   &ir_builder_);

    // Compute the operand index to visit and evaluate the condition whether the
    // operand index is within the bounds. The unsigned comparison includes
    // checking whether the operand index >= 0.
    llvm_ir::IrArray::Index operand_index(index_type, source_index.size());
    llvm::Value* in_bounds_condition = ir_builder_.getInt1(true);
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* strided_index = ir_builder_.CreateNSWMul(
          source_index[i], index_typed_const(window.dimensions(i).stride()));
      operand_index[i] = ir_builder_.CreateNSWSub(
          ir_builder_.CreateNSWAdd(strided_index, window_index[i]),
          index_typed_const(window.dimensions(i).padding_low()));
      llvm::Value* index_condition = ir_builder_.CreateICmpULT(
          operand_index[i],
          index_typed_const(ShapeUtil::GetDimension(operand->shape(), i)));
      in_bounds_condition =
          ir_builder_.CreateAnd(in_bounds_condition, index_condition);
    }
    CHECK(in_bounds_condition != nullptr);

    // Only need to do something if the operand index is within the bounds.
    // First check if the initialized_flag is set.
    llvm_ir::LlvmIfData if_in_bounds =
        llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, &ir_builder_);
    llvm_ir::LlvmIfData if_initialized = llvm_ir::EmitIfThenElse(
        ir_builder_.CreateLoad(initialized_flag_address), "initialized",
        &ir_builder_);

    // If the initialized_flag is false, initialize the selected value and index
    // with the currently visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.false_block, &ir_builder_);
    const auto save_operand_index = [&](
        const llvm_ir::IrArray::Index& operand_index) {
      for (int64 i = 0; i < rank; ++i) {
        llvm::Value* selected_index_address_slot =
            ir_builder_.CreateInBoundsGEP(selected_index_address,
                                          {ir_builder_.getInt32(i)});
        ir_builder_.CreateStore(operand_index[i], selected_index_address_slot);
      }
    };
    llvm_ir::IrArray operand_array = GetIrArray(*operand, *select_and_scatter);
    llvm::Value* operand_data =
        operand_array.EmitReadArrayElement(operand_index, &ir_builder_);
    ir_builder_.CreateStore(operand_data, selected_value_address);
    save_operand_index(operand_index);
    ir_builder_.CreateStore(ir_builder_.getInt1(true),
                            initialized_flag_address);

    // If the initialized_flag is true, call the `select` function to
    // potentially update the selected value and index with the currently
    // visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.true_block, &ir_builder_);
    const Shape output_shape = ShapeUtil::MakeShape(PRED, {});
    llvm::Value* operand_address =
        operand_array.EmitArrayElementAddress(operand_index, &ir_builder_);
    llvm::Value* select_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(PRED,
                                       ir_emitter_context_->llvm_module()),
        "select_return_buffer", &ir_builder_);
    TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
        *select_and_scatter->select(),
        {selected_value_address, operand_address}, select_return_buffer));
    llvm::Value* result = ir_builder_.CreateLoad(select_return_buffer);

    // If the 'select' function returns false, update the selected value and the
    // index to the currently visiting operand.
    llvm::Value* cond = ir_builder_.CreateICmpNE(
        result,
        llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(
                                   PRED, ir_emitter_context_->llvm_module()),
                               0),
        "boolean_predicate");
    llvm_ir::LlvmIfData if_select_lhs =
        llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_select_lhs.false_block, &ir_builder_);
    ir_builder_.CreateStore(ir_builder_.CreateLoad(operand_address),
                            selected_value_address);
    save_operand_index(operand_index);

    // After iterating over the window elements, scatter the source element to
    // the selected index of the output. The value we store at the output
    // location is computed by calling the `scatter` function with the source
    // value and the current output value.
    llvm_ir::SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                                   &ir_builder_);
    llvm_ir::IrArray::Index selected_index(operand_index.GetType());
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot = ir_builder_.CreateInBoundsGEP(
          selected_index_address, {ir_builder_.getInt32(i)});
      selected_index.push_back(
          ir_builder_.CreateLoad(selected_index_address_slot));
    }
    llvm::Value* source_value_address =
        GetIrArray(*source, *select_and_scatter)
            .EmitArrayElementAddress(source_index, &ir_builder_);
    llvm::Value* output_value_address =
        GetIrArray(*select_and_scatter, *select_and_scatter)
            .EmitArrayElementAddress(selected_index, &ir_builder_);
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
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, source->shape(),
                             launch_dimensions, &ir_builder_)
      .EmitLoop(IrName(select_and_scatter), index_type);
}

Status IrEmitterUnnested::HandleWhile(HloInstruction* xla_while) {
  HloComputation* condition = xla_while->while_condition();
  TF_RET_CHECK(ShapeUtil::IsScalar(condition->root_instruction()->shape()) &&
               condition->root_instruction()->shape().element_type() == PRED)
      << "While condition computation must return bool";
  // Build ForThunk for conformant while loops, otherwise build WhileThunk.
  auto result = CanTransformWhileToFor(xla_while);
  if (result.ok()) {
    auto tuple = result.ConsumeValueOrDie();
    // loop_trip_count = (limit - start + increment - 1) / increment
    const int64 loop_trip_count =
        (std::get<1>(tuple) - std::get<0>(tuple) + std::get<2>(tuple) - 1) /
        std::get<2>(tuple);
    thunk_sequence_->emplace_back(BuildForThunk(xla_while, loop_trip_count));
    VLOG(3) << "Built ForThunk for while: " << xla_while->name();
  } else {
    thunk_sequence_->emplace_back(BuildWhileThunk(xla_while));
    VLOG(3) << "Built WhileThunk for while: " << xla_while->name()
            << " while-to-for transform status: " << result.status();
  }
  return Status::OK();
}

Status IrEmitterUnnested::HandleRng(HloInstruction* random) {
  thunk_sequence_->push_back(BuildKernelThunk(random));
  return IrEmitter::HandleRng(random);
}

Status IrEmitterUnnested::HandleSelect(HloInstruction* select) {
  thunk_sequence_->push_back(BuildKernelThunk(select));
  return IrEmitter::HandleSelect(select);
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
    thunk_sequence_->push_back(MakeUnique<DeviceToDeviceCopyThunk>(
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
    thunks.push_back(MakeUnique<DeviceToDeviceCopyThunk>(
        /*source_address=*/GetAllocationSlice(*crs->operand(i)),
        /*destination_buffer=*/tuple_element_buffers.back(),
        /*mem_size=*/ShapeUtil::ByteSizeOf(crs->operand(i)->shape()), crs));
  }

  // Output a tuple of the buffers above.
  thunks.push_back(MakeUnique<TupleThunk>(tuple_element_buffers,
                                          GetAllocationSlice(*crs), crs));
  thunk_sequence_->push_back(
      MakeUnique<SequentialThunk>(std::move(thunks), crs));
  return Status::OK();
}

Status IrEmitterUnnested::HandleGenerateToken(HloInstruction* gen_token) {
  return Status::OK();
}

Status IrEmitterUnnested::HandleInfeed(HloInstruction* infeed) {
  thunk_sequence_->emplace_back(BuildInfeedThunk(infeed));
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
    // GPU constants don't have buffers; don't bother looking for one.
    if (instr->IsConstant()) {
      return;
    }

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
    const HloInstruction* inst, int unroll_factor) {
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
  tensorflow::gtl::optional<const BufferAllocation*> temp_buffer;
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
  std::vector<const BufferAllocation*> buffers(buffers_needed.begin(),
                                               buffers_needed.end());
  std::sort(buffers.begin(), buffers.end(),
            [](const BufferAllocation* a, const BufferAllocation* b) {
              return a->index() < b->index();
            });

  llvm::Function* kernel = BuildKernelPrototype(*inst, buffers);

  // Build a map from a BufferAllocation to the corresponding argument in our
  // kernel.
  std::unordered_map<const BufferAllocation*, llvm::Value*> kernel_args;
  {
    auto arg_it = kernel->arg_begin();
    auto buffers_it = buffers.begin();
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

    llvm::Value* loc =
        ir_builder_.CreateInBoundsGEP(kernel_args.at(slice.allocation()),
                                      {ir_builder_.getInt64(slice.offset())});

    // If gte_index is nonempty, we have to dereference `loc` to get to the
    // value we're ultimately interested in.
    llvm::Type* int8_double_pointer =
        llvm::PointerType::get(ir_builder_.getInt8PtrTy(), /*AddressSpace=*/0);
    for (int64 idx : gte_index) {
      loc = ir_builder_.CreateBitCast(loc, int8_double_pointer);
      loc = ir_builder_.CreateLoad(
          ir_builder_.CreateInBoundsGEP(loc, {ir_builder_.getInt64(idx)}));
    }

    bindings_.BindHloToIrValue(*instr, loc, index);
  }

  // Bind the temp buffer so that nested subcomputations can find it if they
  // need.
  if (temp_buffer.has_value()) {
    bindings_.SetTempBufferBase(kernel_args.at(*temp_buffer));
  } else {
    bindings_.SetTempBufferBase(
        llvm::ConstantPointerNull::get(ir_builder_.getInt8PtrTy()));
  }

  return MakeUnique<KernelThunk>(buffers, llvm_ir::AsString(kernel->getName()),
                                 inst, unroll_factor);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildHostToDeviceCopyThunk(
    const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  CHECK_EQ(HloOpcode::kConstant, operand->opcode());
  return MakeUnique<HostToDeviceCopyThunk>(
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
  return MakeUnique<DeviceToDeviceCopyThunk>(
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
      [this, inst](const ShapeIndex& index, BufferAllocation::Slice* slice) {
        *slice = ir_emitter_context_->buffer_assignment()
                     .GetUniqueSlice(inst, index)
                     .ConsumeValueOrDie();
      });
  return MakeUnique<InfeedThunk>(slices, inst);
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
    return MakeUnique<GemmThunk>(
        GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
        GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
        GetAllocationSlice(*inst),  // The output buffer.
        lhs->shape(),               // The shape of LHS.
        rhs->shape(),               // The shape of RHS.
        inst->shape(),              // The shape of the output.
        1.0,                        // alpha.
        inst);
  }

  if (inst->opcode() == HloOpcode::kFusion) {
    CHECK_EQ(inst->fusion_kind(), HloInstruction::FusionKind::kOutput);
    const HloInstruction* mul = inst->fused_expression_root();
    const HloInstruction* dot = mul->operand(0);
    const HloInstruction* alpha = mul->operand(1);
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

    DCHECK(dot->opcode() == HloOpcode::kDot);
    const HloInstruction* lhs_parameter = StripTranspose(*dot->operand(0));
    const HloInstruction* rhs_parameter = StripTranspose(*dot->operand(1));
    DCHECK(lhs_parameter->opcode() == HloOpcode::kParameter &&
           rhs_parameter->opcode() == HloOpcode::kParameter);
    const HloInstruction* lhs =
        inst->operand(lhs_parameter->parameter_number());
    const HloInstruction* rhs =
        inst->operand(rhs_parameter->parameter_number());

    return MakeUnique<GemmThunk>(
        GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
        GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
        GetAllocationSlice(*inst),  // The output buffer.
        lhs->shape(),               // The shape of LHS.
        rhs->shape(),               // The shape of RHS.
        inst->shape(),              // The shape of the output.
        GetScalarConstantAsDouble(alpha->literal()),  // alpha.
        inst);
  }

  LOG(FATAL) << "Cannot build a GemmThunk for " << inst->ToString();
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildFftThunk(
    const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  return MakeUnique<FftThunk>(inst->fft_type(), inst->fft_length(),
                              /*input_buffer=*/GetAllocationSlice(*operand),
                              /*output_buffer=*/GetAllocationSlice(*inst),
                              /*input_shape=*/operand->shape(),
                              /*output_shape=*/inst->shape(), inst);
}

StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildInitializerThunk(
    const HloInstruction* hlo, const ShapeIndex& index) {
  bool fused = HloOpcode::kFusion == hlo->opcode();
  const HloInstruction* inst = fused ? hlo->fused_expression_root() : hlo;
  const HloInstruction* init_value_operand = [&] {
    switch (inst->opcode()) {
      case HloOpcode::kSelectAndScatter:
        return inst->operand(2);
      case HloOpcode::kReduce:
        return inst->operand(1);
      case HloOpcode::kTuple:
        CHECK(hlo->IsMultiOutputFusion())
            << ": " << hlo->ToString() << " is not a multi-output fusion.";
        CHECK(inst->operand(index.back())->opcode() == HloOpcode::kReduce)
            << ": Found '" << inst->operand(index.back())->opcode() << "' in "
            << inst->ToString() << " but expected 'reduce'.";
        // For multi-output fusion look through the tuple.
        return inst->operand(index.back())->operand(1);
      default:
        LOG(FATAL) << "Opcode " << inst->opcode()
                   << " should not need an initializer.";
    }
  }();

  const HloInstruction* init_value = init_value_operand;
  if (fused && init_value->opcode() == HloOpcode::kParameter) {
    init_value = hlo->operand(init_value->parameter_number());
  }

  // In the common case, the initializer is a constant.  In this case, emit a
  // device-memset call if we can.  Currently StreamExecutor only supports
  // zeroing and 32-bit memsets.
  if (init_value->IsConstant()) {
    CHECK(ShapeUtil::IsScalar(init_value->shape()));
    int64 num_bytes = ShapeUtil::ByteSizeOfElements(init_value->shape());
    const auto& literal = init_value->literal();

    // Are all the bytes of this scalar equal to 0?  If so, we can create a
    // MemzeroThunk.
    ArraySlice<uint8> literal_bytes(
        reinterpret_cast<const uint8*>(literal.untyped_data()), num_bytes);
    if (c_all_of(literal_bytes, [](uint8 byte) { return byte == 0; })) {
      return {MakeUnique<MemzeroThunk>(GetAllocationSlice(*hlo, index), hlo)};
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
        pattern16 = literal_bytes.front();
      }
      uint32 pattern32 = uint32{pattern16} | (uint32{pattern16} << 16);
      return {MakeUnique<Memset32BitValueThunk>(
          pattern32, GetAllocationSlice(*hlo, index), hlo)};
    }

    // If the literal is an even multiple of 32 bits wide, we can emit a 32-bit
    // memset so long as all 32-bit words of the scalar are equal to each other.
    if (num_bytes >= 4 && num_bytes % 4 == 0 &&
        memcmp(literal_bytes.data(), literal_bytes.data() + 4,
               literal_bytes.size() - 4) == 0) {
      uint32 word;
      memcpy(&word, literal_bytes.data(), sizeof(word));
      return {MakeUnique<Memset32BitValueThunk>(
          word, GetAllocationSlice(*hlo, index), hlo)};
    }
  }

  // Otherwise fall back to our slow initializer code.
  std::unique_ptr<KernelThunk> kernel_thunk = BuildKernelThunk(hlo);
  LaunchDimensions launch_dimensions =
      CalculateLaunchDimensions(ShapeUtil::GetSubshape(hlo->shape(), index),
                                ir_emitter_context_->device_description());
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());
  // If the init_value was fused into this reduce we have to generate it first.
  if (fused && init_value_operand->opcode() != HloOpcode::kParameter) {
    CHECK_EQ(HloOpcode::kConstant, init_value_operand->opcode());
    TF_RETURN_IF_ERROR(HandleConstant(const_cast<HloInstruction*>(init_value)));
  }
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(
                         [=](const llvm_ir::IrArray::Index& index) {
                           return GetIrArray(*init_value, *hlo)
                               .EmitReadArrayElement(index, &ir_builder_);
                         },
                         GetIrArray(*hlo, *hlo, index), launch_dimensions,
                         &ir_builder_)
                         .EmitLoop(IrName(hlo)));

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
        a->ToString().c_str(), slice_a.ToString().c_str(),
        b->ToString().c_str(), slice_b.ToString().c_str());
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
  TF_CHECK_OK(condition->root_instruction()->Accept(&ir_emitter_condition));

  // Generate thunk sequence for while 'body'.
  HloComputation* body = hlo->while_body();
  IrEmitterUnnested ir_emitter_body(hlo_module_config_, body,
                                    ir_emitter_context_);
  TF_CHECK_OK(body->root_instruction()->Accept(&ir_emitter_body));

  return MakeUnique<WhileThunk>(
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
  TF_CHECK_OK(body->root_instruction()->Accept(&ir_emitter_body));

  return MakeUnique<ForThunk>(loop_limit,
                              ir_emitter_body.ConsumeThunkSequence(), hlo);
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
  TF_CHECK_OK(true_computation->root_instruction()->Accept(&ir_emitter_true));

  HloComputation* false_computation = hlo->false_computation();
  IrEmitterUnnested ir_emitter_false(hlo_module_config_, false_computation,
                                     ir_emitter_context_);
  TF_CHECK_OK(false_computation->root_instruction()->Accept(&ir_emitter_false));

  return MakeUnique<ConditionalThunk>(
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
                               launch_dimensions, &ir_builder_, unroll_factor)
        .EmitLoop(IrName(&hlo),
                  GetIndexTypeForKernel(&hlo, launch_dimensions.launch_bound(),
                                        &ir_builder_));
  }

  // For multiple outputs fusion, we need to emit each operand and the root.
  std::vector<llvm_ir::IrArray> output_arrays;
  for (int64 i = 0; i < ShapeUtil::TupleElementCount(hlo.shape()); ++i) {
    output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
  }
  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(element_generator, output_arrays, launch_dimensions,
                          &ir_builder_, unroll_factor)
          .EmitLoop(IrName(&hlo),
                    GetIndexTypeForKernel(
                        &hlo, launch_dimensions.launch_bound(), &ir_builder_)));

  std::vector<llvm::Value*> tuple_operand_ptrs;
  for (int64 i = 0; i < output_arrays.size(); ++i) {
    tuple_operand_ptrs.push_back(output_arrays[i].GetBasePointer());
  }
  ir_builder_.SetInsertPoint(ir_builder_.GetInsertBlock()->getTerminator());
  llvm_ir::EmitTuple(GetIrArray(hlo, hlo), tuple_operand_ptrs, &ir_builder_,
                     module_);
  return Status::OK();
}

Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  CHECK(Thunk::Kind::kKernel == LastThunk()->kind());
  return EmitTargetElementLoopInThunk(hlo, element_generator,
                                      static_cast<KernelThunk*>(LastThunk()));
}

}  // namespace gpu
}  // namespace xla
