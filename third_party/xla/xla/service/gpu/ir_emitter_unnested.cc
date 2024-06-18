/*Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/ir_emitter_unnested.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir_hlo/transforms/gpu_passes.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/fusions/thunk_util.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_fused_mha_runner.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/topk_custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/service/gpu/runtime/command_buffer_thunk.h"
#include "xla/service/gpu/runtime/conditional_thunk.h"
#include "xla/service/gpu/runtime/convolution_thunk.h"
#include "xla/service/gpu/runtime/copy_thunk.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/fft_thunk.h"
#include "xla/service/gpu/runtime/fused_mha_thunk.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/runtime/infeed_thunk.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_gather_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_to_all_thunk.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_broadcast_thunk.h"
#include "xla/service/gpu/runtime/nccl_collective_permute_thunk.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/runtime/nccl_recv_thunk.h"
#include "xla/service/gpu/runtime/nccl_send_thunk.h"
#include "xla/service/gpu/runtime/norm_thunk.h"
#include "xla/service/gpu/runtime/outfeed_thunk.h"
#include "xla/service/gpu/runtime/replica_id_thunk.h"
#include "xla/service/gpu/runtime/send_recv_thunk.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/runtime/wait_for_streams_thunk.h"
#include "xla/service/gpu/runtime/while_thunk.h"
#include "xla/service/gpu/triton_call.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/sort_util.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/dnn.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#if GOOGLE_CUDA || TF_HIPBLASLT
#include "xla/service/gpu/runtime/gpublas_lt_matmul_thunk.h"
#endif  // GOOGLE_CUDA || TF_HIPBLASLT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/runtime/cholesky_thunk.h"
#include "xla/service/gpu/runtime/cub_sort_thunk.h"
#include "xla/service/gpu/runtime/triangular_solve_thunk.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {
namespace {

// Construct the key for looking up the AsyncEvents for Send and Recv. Input
// kind is the thunk kind for the corresponding done thunk.
inline std::pair<bool, int64_t> GetSendRecvAsyncEventsKey(Thunk::Kind kind,
                                                          int64_t channel_id) {
  return std::make_pair(kind == Thunk::Kind::kNcclRecvDone, channel_id);
}

}  // namespace

IrEmitterUnnested::IrEmitterUnnested(IrEmitterContext* ir_emitter_context)
    : IrEmitter(ir_emitter_context, /*is_nested=*/false),
      send_recv_events_(std::make_shared<SendRecvAsyncEvents>()),
      copy_events_(std::make_shared<CopyThunk::AsyncEvents>()),
      elemental_emitter_(*ir_emitter_context, &b_) {}

std::unique_ptr<IrEmitterUnnested> IrEmitterUnnested::Create(
    IrEmitterContext* ir_emitter_context) {
  return std::unique_ptr<IrEmitterUnnested>(
      new IrEmitterUnnested(ir_emitter_context));
}

absl::Status IrEmitterUnnested::EmitConstant(
    const HloConstantInstruction* instr) {
  TF_ASSIGN_OR_RETURN(DenseDataIntermediate content,
                      LiteralToXlaFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(instr, {}));

  ir_emitter_context_->emit_constant(num_elements, element_bytes, global_name,
                                     slice.index(), std::move(content), &b_);
  return absl::OkStatus();
}

static ConditionalThunkConfig GetConditionalThunkConfig(
    const HloInstruction* instr,
    std::vector<ThunkSequence> branch_thunk_sequences) {
  ConditionalThunkConfig config;
  config.branch_index_is_bool =
      instr->operand(0)->shape().element_type() == PRED;
  config.branch_count = instr->branch_count();
  config.branch_thunks.reserve(config.branch_count);
  for (auto& branch_thunk_sequence : branch_thunk_sequences) {
    config.branch_thunks.emplace_back(
        new SequentialThunk(Thunk::ThunkInfo::WithProfileAnnotation(instr),
                            std::move(branch_thunk_sequence)));
  }
  return config;
}

absl::Status IrEmitterUnnested::EmitConditional(const HloInstruction* instr) {
  std::vector<ThunkSequence> branch_thunks;
  branch_thunks.reserve(instr->branch_count());

  for (auto comp : instr->branch_computations()) {
    auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
    TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(comp));
    branch_thunks.push_back(std::move(*ir_emitter->ConsumeThunkSequence()));
  }

  ConditionalThunkConfig config =
      GetConditionalThunkConfig(instr, std::move(branch_thunks));

  TF_ASSIGN_OR_RETURN(auto slice,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  AddThunkToThunkSequence(std::unique_ptr<Thunk>(
      new ConditionalThunk(Thunk::ThunkInfo::WithProfileAnnotation(instr),
                           std::move(config), slice)));
  return absl::OkStatus();
}

llvm::Value* IrEmitterUnnested::CreateLoad(llvm::Value* address,
                                           llvm::Type* data_type,
                                           int alignment_bytes) {
  int data_bytes = data_type->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  if (alignment_bytes == 0) {
    return b_.CreateLoad(data_type, address);
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  llvm::Value* output = llvm::ConstantInt::get(data_type, 0);
  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* partial_value = b_.CreateLoad(b_.getIntNTy(alignment_bitwidth),
                                               offset_address, "partial_value");
    llvm::Value* zextd =
        b_.CreateZExt(partial_value, output->getType(), "partial_value_zextd");
    llvm::Value* shifted = b_.CreateShl(
        zextd, llvm::ConstantInt::get(b_.getInt32Ty(), offset_bytes),
        "partial_input_shifted");
    output = b_.CreateAdd(output, shifted, "output_updated");
  }
  return output;
}

void IrEmitterUnnested::CreateStore(llvm::Value* data, llvm::Value* address,
                                    int alignment_bytes) {
  int data_bytes = data->getType()->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  CHECK_GE(data_bytes, alignment_bytes);
  if (alignment_bytes == 0) {
    b_.CreateStore(data, address);
    return;
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* shifted_partial = b_.CreateTrunc(
        b_.CreateLShr(data,
                      llvm::ConstantInt::get(b_.getInt32Ty(), offset_bytes)),
        b_.getIntNTy(alignment_bitwidth), "truncated_value");
    b_.CreateStore(shifted_partial, offset_address);
  }
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
absl::Status IrEmitterUnnested::EmitPadToStatic(
    const HloCustomCallInstruction* instr) {
  int unroll_factor = 1;
  std::string ir_name = std::string(instr->name());

  const Shape& input_shape = instr->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context_->gpu_device_info(), {unroll_factor});
  std::vector<llvm_ir::IrArray> input_arrays;
  std::vector<llvm_ir::IrArray> output_arrays;
  TF_ASSIGN_OR_RETURN(std::tie(input_arrays, output_arrays),
                      BuildKernelThunkForNonFusionOp(instr, instr->operands(),
                                                     launch_dimensions));

  CHECK_EQ(output_arrays.size(), 0);
  const llvm_ir::IrArray source_array = input_arrays[0];
  const llvm_ir::IrArray output_array = input_arrays[1];
  auto output_dim_arrays =
      absl::Span<const llvm_ir::IrArray>(input_arrays).subspan(2);

  llvm::Type* index_ty =
      GetIndexTypeForKernel(instr, launch_dimensions.launch_bound(), &b_);

  // pseudo code for PadToStatic on a 2d array
  //   int* source_array = input[0];
  //   int* dest_array = output[0];
  llvm::Value* source_buffer = source_array.GetBasePointer();

  // TODO(jurahul): input_shape here is the static shape of the input (which has
  // a dynamic shape in XLA). Currently, we are mapping that to a static shaped
  // memref. When we change that to a more appropriate representation in MLIR,
  // fix this code to correctly deduce the static shape backing the dynamically
  // shaped memref.
  int64_t raw_data_size = ShapeUtil::ByteSizeOf(input_shape);

  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  std::vector<ShapeUtil::IndexedShape> output_shapes =
      ShapeUtil::GetLeafShapes(instr->shape());

  for (int64_t i = 1; i < output_shapes.size(); ++i) {
    // Dynamic size of each dimension is attached at the end of the source
    // array(operand(0)). We need to extract these value.
    const Shape& dim_shape = output_shapes[i].shape;
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));

    const int64_t dim_index = i - 1;
    llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), source_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    llvm::Value* dyn_dim_size =
        CreateLoad(metadata, b_.getInt32Ty(), alignment);
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }
  KernelSupportLibrary{&b_}.If("is_thread_0", IsBlock0Thread0(&b_), [&] {
    for (int64_t i = 1; i < output_shapes.size(); ++i) {
      const int64_t dim_index = i - 1;
      llvm::Value* dest_dim_size_address =
          output_dim_arrays[dim_index].GetBasePointer();
      // output[i] stores dynamic_dim_(i-1)
      CreateStore(dynamic_dims[dim_index], dest_dim_size_address, alignment);
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= *dyn_dim0_size;
  //     dyn_element_total *= *dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total =
        b_.CreateMul(dyn_element_total,
                     b_.CreateIntCast(dynamic_dim, dyn_element_total->getType(),
                                      /*isSigned=*/true),
                     /*Name=*/"dyn_element_total_pad");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), &b_);
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        b_.CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), &b_, false);
    // Set IR builder insertion point to the body of the if structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block, &b_);
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims), &b_);
    output_array.EmitWriteArrayElement(
        dyn_index,
        source_array.EmitReadArrayElement(array_index, &b_, /*name=*/""), &b_,
        /*use_linear_index=*/false);
    return absl::OkStatus();
  };

  const Shape& data_shape = instr->shape().tuple_shapes(0);
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                         launch_dimensions, &b_,
                                         {unroll_factor})
                         .EmitLoop(ir_name, index_ty));
  return absl::OkStatus();
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
absl::Status IrEmitterUnnested::EmitSliceToDynamic(
    const HloCustomCallInstruction* instr) {
  // TODO(jurahul): Create an op to represent SliceToDynamic.
  int unroll_factor = 1;
  std::string ir_name = std::string(instr->name());

  const Shape& input_shape = instr->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context_->gpu_device_info(), {unroll_factor});
  llvm::Type* index_ty =
      GetIndexTypeForKernel(instr, launch_dimensions.launch_bound(), &b_);
  std::vector<llvm_ir::IrArray> input_arrays, output_arrays;
  TF_ASSIGN_OR_RETURN(std::tie(input_arrays, output_arrays),
                      BuildKernelThunkForNonFusionOp(instr, instr->operands(),
                                                     launch_dimensions));

  const Shape& data_shape = ShapeUtil::MakeStaticShape(instr->shape());
  TF_RET_CHECK(data_shape.IsArray());

  // TODO(jurahul): data_shape here is the static shape of the output (which has
  // a dynamic shape in XLA). Currently, we are mapping that to a static shaped
  // memref. When we change that to a more appropriate representation in MLIR,
  // fix this code to correctly deduce the static shape backing the dynamically
  // shaped memref.

  // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset + sizeof(int);
  int32_t raw_data_size = ShapeUtil::ByteSizeOf(data_shape);

  // pseudo code for sliceToDynamic on a 2d array
  //   int* source_array = input[0];
  //   int* dest_array = output[0];
  const llvm_ir::IrArray data_array = input_arrays.back();
  llvm::Value* dest_buffer = data_array.GetBasePointer();

  // Load dynamic dimensions from memory.
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  for (int64_t i = 1; i < instr->operand_count(); ++i) {
    llvm::Value* source_buffer = input_arrays[i].GetBasePointer();
    llvm::Type* source_buffer_pointee_type =
        input_arrays[i].GetBasePointeeType();
    llvm::LoadInst* dyn_dim_size =
        Load(source_buffer_pointee_type, source_buffer, "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }
  KernelSupportLibrary{&b_}.If("is_thread_0", IsBlock0Thread0(&b_), [&] {
    for (int64_t i = 1; i < instr->operand_count(); ++i) {
      const int64_t dim_index = i - 1;
      llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
          b_.getInt8Ty(), dest_buffer,
          raw_data_size + dim_index * sizeof(int32_t));
      // output[i] stores dynamic_dim_(i-1)
      CreateStore(dynamic_dims[dim_index], metadata, alignment);
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= dyn_dim0_size;
  //     dyn_element_total *= dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total =
        b_.CreateMul(dyn_element_total,
                     b_.CreateIntCast(dynamic_dim, dyn_element_total->getType(),
                                      /*isSigned=*/true),
                     /*Name=*/"dyn_element_total_slice");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.di] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), &b_);
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        b_.CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), &b_, false);
    // Set IR builder insertion point to the body of the if structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block, &b_);
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims), &b_);

    data_array.EmitWriteArrayElement(
        array_index,
        input_arrays[0].EmitReadArrayElement(dyn_index, &b_, /*name=*/"",
                                             /*use_linear_index=*/false),
        &b_);
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                         launch_dimensions, &b_,
                                         {unroll_factor})
                         .EmitLoop(ir_name, index_ty));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCommandBufferThunk(
    const HloInstruction* instr) {
  // Spawn a new IrEmitterUnnested to emit thunks for the command buffer
  // computation. Then convert emitted thunks to a sequence of CommandBufferCmd.
  // The resulting thunk added to the thunk sequence is a CommandBufferThunk.
  // Thunks emitted from the command buffer computation are discarded.
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* command_buffer = instr->called_computations().front();
  auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(command_buffer));
  std::unique_ptr<ThunkSequence> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();

  // Maybe serialize all commands in a sequence by forcing barriers between all
  // recorded commands. This guarantees that we execute all device operations
  // in the exact same order as a thunk sequence.
  CommandBufferCmdSequence::SynchronizationMode synchronization_mode =
      ir_emitter_context_->debug_options()
              .xla_gpu_graph_enable_concurrent_region()
          ? CommandBufferCmdSequence::SynchronizationMode::kAutomatic
          : CommandBufferCmdSequence::SynchronizationMode::kSerialize;

  TF_ASSIGN_OR_RETURN(CommandBufferCmdSequence cmd_sequence,
                      ConvertToCommands(*thunk_sequence, synchronization_mode));

  AddThunkToThunkSequence(std::make_unique<CommandBufferThunk>(
      std::move(cmd_sequence), Thunk::ThunkInfo::WithProfileAnnotation(instr),
      std::move(*thunk_sequence)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitConvolutionThunk(
    const HloCustomCallInstruction* instr) {
  std::vector<BufferAllocation::Slice> operand_slices;
  operand_slices.reserve(instr->operand_count());
  for (const HloInstruction* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        GetAllocationSliceForHlo(operand, {}));
    operand_slices.push_back(slice);
  }

  // The first and the last element in the result tuple for a convolution are
  // always the result and the scratch buffer. It may have auxiliary results in
  // addition to the main result.
  std::vector<BufferAllocation::Slice> result_slices;
  for (int i = 0; i < instr->shape().tuple_shapes_size() - 1; i++) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                        GetAllocationSliceForHlo(instr, {i}));
    result_slices.push_back(result_slice);
  }

  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      instr->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& backend_config =
      gpu_config.cudnn_conv_backend_config();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      GetAllocationSliceForHlo(
                          instr, {instr->shape().tuple_shapes_size() - 1}));
  GpuConvDescriptor descriptor = {kind,
                                  backend_config,
                                  instr->operand(0)->shape(),
                                  instr->operand(1)->shape(),
                                  instr->shape().tuple_shapes(0),
                                  static_cast<size_t>(scratch_slice.size()),
                                  instr->window(),
                                  instr->convolution_dimension_numbers(),
                                  instr->feature_group_count()};

  TF_ASSIGN_OR_RETURN(GpuConvConfig config, GetGpuConvConfig(descriptor, ""));
  AddThunkToThunkSequence(std::make_unique<ConvolutionThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(config),
      std::move(operand_slices), std::move(result_slices), scratch_slice));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitGemmThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1), {}));

  // Result of a legacy cuBLAS custom call can be a tuple if we explicitly
  // allocate workspace buffer in HLO. If result is an array, it means that
  // workspace is not available, and cuBLAS will allocate its own workspace.
  BufferAllocation::Slice c;
  std::optional<BufferAllocation::Slice> workspace;

  if (instr->shape().IsArray()) {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, {}));
  } else {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, {0}));
    TF_ASSIGN_OR_RETURN(workspace, GetAllocationSliceForHlo(instr, {1}));
  }

  bool deterministic_ops =
      ir_emitter_context_->debug_options().xla_gpu_deterministic_ops() ||
      ir_emitter_context_->debug_options()
          .xla_gpu_exclude_nondeterministic_ops();

  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr)));
  auto thunk = std::make_unique<GemmThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(config), a, b,
      c, workspace, deterministic_ops);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

#if GOOGLE_CUDA || TF_HIPBLASLT

absl::Status IrEmitterUnnested::EmitCublasLtMatmulThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_ASSIGN_OR_RETURN(bool has_vector_bias,
                      xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  bool has_matrix_bias = config.beta() != 0;

  TF_RET_CHECK(instr->operand_count() ==
               2 + int{has_matrix_bias} + int{has_vector_bias});

  TF_ASSIGN_OR_RETURN(
      bool has_aux_output,
      xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));
  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1)));
  BufferAllocation::Slice c;
  if (has_matrix_bias) {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr->operand(2)));
  } else {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, output_index));
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d,
                      GetAllocationSliceForHlo(instr, output_index));

  BufferAllocation::Slice bias;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(bias, GetAllocationSliceForHlo(
                                  instr->operand(has_matrix_bias ? 3 : 2)));
  }

  BufferAllocation::Slice aux;
  if (has_aux_output) {
    TF_ASSIGN_OR_RETURN(aux, GetAllocationSliceForHlo(instr, {1}));
  }

  std::optional<BufferAllocation::Slice> workspace_buffer;
  if (instr->shape().IsTuple() &&
      (instr->shape().tuple_shapes_size() - has_aux_output - 1)) {
    TF_RET_CHECK((has_aux_output && instr->shape().tuple_shapes_size() == 3) ||
                 (!has_aux_output && instr->shape().tuple_shapes_size() == 2));
    TF_ASSIGN_OR_RETURN(workspace_buffer,
                        GetAllocationSliceForHlo(
                            instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr)));

  // Use the first algorithm by default (i.e. fastest according to heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  BufferAllocation::Slice a_scale, b_scale, c_scale, d_scale, d_amax;
  TF_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                      gpublas_lt::AsBlasLtEpilogue(epilogue));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(gemm_config),
      blas_lt_epilogue, algorithm, a, b, c, d, bias, aux, a_scale, b_scale,
      c_scale, d_scale, d_amax, workspace_buffer);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCublasLtMatmulThunkF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 6 || instr->operand_count() == 7 ||
               instr->operand_count() == 8);
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_ASSIGN_OR_RETURN(bool has_vector_bias,
                      xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  TF_ASSIGN_OR_RETURN(
      bool has_aux_output,
      xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1)));
  BufferAllocation::Slice c;
  bool has_matrix_bias = config.beta() != 0;
  if (has_matrix_bias) {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr->operand(2)));
  } else {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, output_index));
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d,
                      GetAllocationSliceForHlo(instr, output_index));

  int a_scale_index = has_matrix_bias ? 3 : 2;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_scale,
                      GetAllocationSliceForHlo(instr->operand(a_scale_index)));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice b_scale,
      GetAllocationSliceForHlo(instr->operand(a_scale_index + 1)));
#if GOOGLE_CUDA
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice c_scale,
      GetAllocationSliceForHlo(instr->operand(a_scale_index + 2)));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice d_scale,
      GetAllocationSliceForHlo(instr->operand(a_scale_index + 3)));
#else  // TENSORFLOW_USE_ROCM
  BufferAllocation::Slice c_scale;
  BufferAllocation::Slice d_scale;
#endif

  BufferAllocation::Slice bias;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(
        bias, GetAllocationSliceForHlo(instr->operand(a_scale_index + 4)));
  }

  BufferAllocation::Slice d_amax;
  if (config.damax_output()) {
    TF_ASSIGN_OR_RETURN(d_amax, GetAllocationSliceForHlo(instr, {1}));
  }

  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr)));

  // Use the first algorithm by default (i.e. fastest according to heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  BufferAllocation::Slice aux;  // Not used.
  TF_RET_CHECK(!has_aux_output);
  std::optional<BufferAllocation::Slice> workspace_buffer;
  if (instr->shape().tuple_shapes_size() - config.damax_output() == 2) {
    TF_ASSIGN_OR_RETURN(workspace_buffer,
                        GetAllocationSliceForHlo(
                            instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  TF_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                      gpublas_lt::AsBlasLtEpilogue(epilogue));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(gemm_config),
      blas_lt_epilogue, algorithm, a, b, c, d, bias, aux, a_scale, b_scale,
      c_scale, d_scale, d_amax, workspace_buffer);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}
#endif  // GOOGLE_CUDA || TF_HIPBLASLT

#if GOOGLE_CUDA
absl::Status IrEmitterUnnested::EmitConvolutionReorderThunk(
    const HloCustomCallInstruction* instr) {
  bool has_bias = instr->operand_count() > 1;
  Shape shape = has_bias ? instr->shape().tuple_shapes(0) : instr->shape();
  if (shape.rank() != 5 || shape.dimensions(4) != 32) {
    return Internal("Unexpected shape for convolution reorder: %s",
                    instr->ToString());
  }
  absl::InlinedVector<int64_t, 4> filter_dims = {
      shape.dimensions(0), shape.dimensions(1) * 32, shape.dimensions(2),
      shape.dimensions(3)};

  absl::InlinedVector<BufferAllocation::Slice, 2> operand_slices;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_input,
                      GetAllocationSliceForHlo(instr->operand(0)));
  operand_slices.push_back(filter_input);
  if (has_bias) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bias_input,
                        GetAllocationSliceForHlo(instr->operand(1)));
    operand_slices.push_back(bias_input);
  }

  absl::InlinedVector<BufferAllocation::Slice, 2> result_slices;
  if (has_bias) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_output,
                        GetAllocationSliceForHlo(instr, {0}));
    result_slices.push_back(filter_output);
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bias_output,
                        GetAllocationSliceForHlo(instr, {1}));
    result_slices.push_back(bias_output);
  } else {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_output,
                        GetAllocationSliceForHlo(instr));
    result_slices.push_back(filter_output);
  }

  auto thunk = std::make_unique<ConvolutionReorderThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      absl::MakeSpan(filter_dims), operand_slices, result_slices);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitNormThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto const gpu_backend_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnNormBackendConfig& backend_config =
      gpu_backend_config.cudnn_norm_backend_config();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x_slice,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scale_slice,
                      GetAllocationSliceForHlo(instr->operand(1)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice y_or_dx_slice,
                      GetAllocationSliceForHlo(instr, {0}));

  std::optional<BufferAllocation::Slice> bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    TF_ASSIGN_OR_RETURN(bias_slice,
                        GetAllocationSliceForHlo(instr->operand(2)));
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    TF_ASSIGN_OR_RETURN(expectation_slice,
                        GetAllocationSliceForHlo(instr, {1}));
    TF_ASSIGN_OR_RETURN(norm_factor_slice,
                        GetAllocationSliceForHlo(instr, {2}));
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    TF_ASSIGN_OR_RETURN(dy_slice, GetAllocationSliceForHlo(instr->operand(2)));
    TF_ASSIGN_OR_RETURN(expectation_slice,
                        GetAllocationSliceForHlo(instr->operand(3)));
    TF_ASSIGN_OR_RETURN(norm_factor_slice,
                        GetAllocationSliceForHlo(instr->operand(4)));
    TF_ASSIGN_OR_RETURN(dscale_slice, GetAllocationSliceForHlo(instr, {1}));
    TF_ASSIGN_OR_RETURN(dbias_slice, GetAllocationSliceForHlo(instr, {2}));
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      GetAllocationSliceForHlo(
                          instr, {instr->shape().tuple_shapes_size() - 1}));

  GpuNormDescriptor descriptor;
  descriptor.backend_config = backend_config;

  descriptor.x_shape = instr->operand(0)->shape();
  descriptor.scale_shape = instr->operand(1)->shape();
  descriptor.y_or_dx_shape = ShapeUtil::GetSubshape(instr->shape(), {0});
  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    descriptor.bias_shape = instr->operand(2)->shape();
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    descriptor.expectation_shape = ShapeUtil::GetSubshape(instr->shape(), {1});
    descriptor.norm_factor_shape = ShapeUtil::GetSubshape(instr->shape(), {2});
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    descriptor.dy_shape = instr->operand(2)->shape();
    descriptor.expectation_shape = instr->operand(3)->shape();
    descriptor.norm_factor_shape = instr->operand(4)->shape();
    descriptor.dscale_shape = ShapeUtil::GetSubshape(instr->shape(), {1});
    descriptor.dbias_shape = ShapeUtil::GetSubshape(instr->shape(), {2});
  }

  TF_ASSIGN_OR_RETURN(GpuNormConfig config, GpuNormConfig::For(descriptor));

  auto thunk = std::make_unique<NormThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(config),
      x_slice, scale_slice, y_or_dx_slice, bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice, scratch_slice);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitFusedMHAThunk(
    const HloCustomCallInstruction* instr) {
  const HloInstruction* lhs_bmm1 = instr->operand(0);
  const HloInstruction* rhs_bmm1 = instr->operand(1);
  const HloInstruction* rhs_bmm2 = instr->operand(2);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_bmm1_slice,
                      GetAllocationSliceForHlo(lhs_bmm1));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_bmm1_slice,
                      GetAllocationSliceForHlo(rhs_bmm1));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_bmm2_slice,
                      GetAllocationSliceForHlo(rhs_bmm2));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      GetAllocationSliceForHlo(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      GetAllocationSliceForHlo(
                          instr, {instr->shape().tuple_shapes_size() - 1}));
  BufferAllocation::Slice activation_slice;
  bool has_activation = xla::ShapeUtil::TupleElementCount(instr->shape()) == 3;
  if (has_activation) {
    TF_ASSIGN_OR_RETURN(activation_slice, GetAllocationSliceForHlo(instr, {1}));
  }

  TF_ASSIGN_OR_RETURN(const xla::gpu::CudnnfMHAKind kind,
                      xla::gpu::GetCudnnfMHAKind(instr));
  BufferAllocation::Slice mask_slice, bias_slice;
  BufferAllocation::Slice seqlen_q_slice, seqlen_k_slice;
  std::optional<Shape> mask_shape, bias_shape;
  {
    bool has_bias = kind == CudnnfMHAKind::kScaleBiasSoftmax ||
                    kind == CudnnfMHAKind::kScaleBiasSoftmaxDropout;

    if (has_bias) {
      const HloInstruction* bias = instr->operand(3);
      TF_ASSIGN_OR_RETURN(bias_slice, GetAllocationSliceForHlo(bias));
      bias_shape = bias->shape();
    }
    int64_t seqlen_qk_operand_index = 3 + has_bias;
    bool has_seqlen_qk = seqlen_qk_operand_index == instr->operand_count() - 2;
    if (has_seqlen_qk) {
      const HloInstruction* seqlen_q = instr->operand(seqlen_qk_operand_index);
      TF_ASSIGN_OR_RETURN(seqlen_q_slice, GetAllocationSliceForHlo(seqlen_q));
      const HloInstruction* seqlen_k =
          instr->operand(seqlen_qk_operand_index + 1);
      TF_ASSIGN_OR_RETURN(seqlen_k_slice, GetAllocationSliceForHlo(seqlen_k));
    }
  }

  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnfMHABackendConfig& config =
      gpu_config.cudnn_fmha_backend_config();
  Shape intermediate_tensor_shape(config.intermediate_tensor_shape());
  absl::InlinedVector<Shape, 2> output_shapes = {
      ShapeUtil::GetSubshape(instr->shape(), {0})};
  if (has_activation) {
    output_shapes.push_back(ShapeUtil::GetSubshape(instr->shape(), {1}));
  }
  TF_ASSIGN_OR_RETURN(const auto mask_type,
                      AsCudnnFmhaMaskKind(config.mask_type()));
  GpufMHADescriptor descriptor = {kind,
                                  config,
                                  mask_type,
                                  lhs_bmm1->shape(),
                                  rhs_bmm1->shape(),
                                  rhs_bmm2->shape(),
                                  intermediate_tensor_shape,
                                  output_shapes,
                                  config.bmm1_dot_dimension_numbers(),
                                  config.bmm2_dot_dimension_numbers(),
                                  mask_shape,
                                  bias_shape};

  TF_ASSIGN_OR_RETURN(GpufMHAConfig fmha_config,
                      GpufMHAConfig::For(descriptor));
  AddThunkToThunkSequence(std::make_unique<FusedMHAThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(fmha_config),
      lhs_bmm1_slice, rhs_bmm1_slice, rhs_bmm2_slice, output_slice,
      scratch_slice, mask_slice, bias_slice, activation_slice, seqlen_q_slice,
      seqlen_k_slice));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitFusedMHABackwardThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnfMHABackendConfig& config =
      gpu_config.cudnn_fmha_backend_config();

  int input_index = 0;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bmm1_grad_gemm1_rhs_slice,
                      GetAllocationSliceForHlo(instr->operand(input_index)));
  Shape bmm1_grad_gemm1_rhs_shape = instr->operand(input_index++)->shape();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bmm1_grad_gemm2_rhs_slice,
                      GetAllocationSliceForHlo(instr->operand(input_index)));
  Shape bmm1_grad_gemm2_rhs_shape = instr->operand(input_index++)->shape();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bmm2_grad_gemm2_rhs_slice,
                      GetAllocationSliceForHlo(instr->operand(input_index)));
  Shape bmm2_grad_gemm2_rhs_shape = instr->operand(input_index++)->shape();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bmm2_grad_gemm1_lhs_slice,
                      GetAllocationSliceForHlo(instr->operand(input_index)));
  Shape bmm2_grad_gemm1_lhs_shape;

  Shape intermediate_tensor_shape(config.intermediate_tensor_shape());
  bmm2_grad_gemm1_lhs_shape = intermediate_tensor_shape;
  input_index++;

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d_output_slice,
                      GetAllocationSliceForHlo(instr->operand(input_index)));
  Shape d_output_shape = instr->operand(input_index++)->shape();

  TF_ASSIGN_OR_RETURN(const CudnnfMHAKind kind, GetCudnnfMHAKind(instr));
  BufferAllocation::Slice mask_slice;
  std::optional<Shape> mask_shape;

  bool has_bias = (kind == CudnnfMHAKind::kBackwardScaleBiasSoftmax ||
                   kind == CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout);
  BufferAllocation::Slice bias_slice;
  std::optional<Shape> bias_shape;
  if (has_bias) {
    TF_ASSIGN_OR_RETURN(bias_slice,
                        GetAllocationSliceForHlo(instr->operand(input_index)));
    bias_shape = instr->operand(input_index++)->shape();
  }

  BufferAllocation::Slice fwd_output_slice;
  std::optional<Shape> fwd_output_shape;

  TF_ASSIGN_OR_RETURN(fwd_output_slice,
                      GetAllocationSliceForHlo(instr->operand(input_index)));
  fwd_output_shape = instr->operand(input_index++)->shape();

  BufferAllocation::Slice seqlen_q_slice, seqlen_k_slice;
  bool has_seqlen_qk = input_index == instr->operand_count() - 2;
  if (has_seqlen_qk) {
    const HloInstruction* seqlen_q = instr->operand(input_index);
    TF_ASSIGN_OR_RETURN(seqlen_q_slice, GetAllocationSliceForHlo(seqlen_q));
    const HloInstruction* seqlen_k = instr->operand(input_index + 1);
    TF_ASSIGN_OR_RETURN(seqlen_k_slice, GetAllocationSliceForHlo(seqlen_k));
    input_index += 2;
  }
  TF_RET_CHECK(input_index == instr->operand_count());

  int output_index = 0;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d_bmm1_lhs_slice,
                      GetAllocationSliceForHlo(instr, {output_index}));
  Shape d_bmm1_lhs_shape =
      ShapeUtil::GetSubshape(instr->shape(), {output_index++});

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d_bmm1_rhs_slice,
                      GetAllocationSliceForHlo(instr, {output_index}));
  Shape d_bmm1_rhs_shape =
      ShapeUtil::GetSubshape(instr->shape(), {output_index++});

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d_bmm2_rhs_slice,
                      GetAllocationSliceForHlo(instr, {output_index}));
  Shape d_bmm2_rhs_shape =
      ShapeUtil::GetSubshape(instr->shape(), {output_index++});

  BufferAllocation::Slice d_s_slice;
  std::optional<Shape> d_s_shape;

  bool has_dbias = instr->shape().tuple_shapes().size() == 5;
  BufferAllocation::Slice d_bias_slice;
  std::optional<Shape> d_bias_shape;
  if (has_dbias) {
    TF_ASSIGN_OR_RETURN(d_bias_slice,
                        GetAllocationSliceForHlo(instr, {output_index}));
    d_bias_shape = ShapeUtil::GetSubshape(instr->shape(), {output_index++});
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      GetAllocationSliceForHlo(instr, {output_index++}));
  TF_RET_CHECK(output_index == instr->shape().tuple_shapes().size());
  TF_ASSIGN_OR_RETURN(const auto mask_type,
                      AsCudnnFmhaMaskKind(config.mask_type()));
  GpufMHABackwardDescriptor descriptor = {
      kind,
      config,
      mask_type,
      bmm1_grad_gemm1_rhs_shape,
      bmm1_grad_gemm2_rhs_shape,
      bmm2_grad_gemm1_lhs_shape,
      bmm2_grad_gemm2_rhs_shape,
      d_output_shape,
      d_bmm1_lhs_shape,
      d_bmm1_rhs_shape,
      d_bmm2_rhs_shape,
      config.bmm1_grad_gemm1_dot_dimension_numbers(),
      config.bmm1_grad_gemm2_dot_dimension_numbers(),
      config.bmm2_grad_gemm1_dot_dimension_numbers(),
      config.bmm2_grad_gemm2_dot_dimension_numbers(),
      d_s_shape,
      fwd_output_shape,
      mask_shape,
      d_bias_shape,
      bias_shape};

  TF_ASSIGN_OR_RETURN(GpufMHABackwardConfig fmha_backward_config,
                      GpufMHABackwardConfig::For(descriptor));

  AddThunkToThunkSequence(std::make_unique<FusedMHABackwardThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      std::move(fmha_backward_config), bmm1_grad_gemm1_rhs_slice,
      bmm1_grad_gemm2_rhs_slice, bmm2_grad_gemm1_lhs_slice,
      bmm2_grad_gemm2_rhs_slice, d_output_slice, scratch_slice,
      d_bmm1_lhs_slice, d_bmm1_rhs_slice, d_bmm2_rhs_slice, d_s_slice,
      mask_slice, d_bias_slice, fwd_output_slice, bias_slice, seqlen_q_slice,
      seqlen_k_slice));

  return absl::OkStatus();
}

#endif  // GOOGLE_CUDA

absl::StatusOr<BufferAllocation::Slice>
IrEmitterUnnested::GetAllocationSliceForHlo(const HloInstruction* instr,
                                            const ShapeIndex& index) const {
  return xla::gpu::GetAllocationSlice(ir_emitter_context_->buffer_assignment(),
                                      instr, index);
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

absl::Status IrEmitterUnnested::EmitCubDeviceRadixSort(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 1 && instr->operand_count() != 2) {
    return Internal("Invalid number of operands for radix sort");
  }

  absl::InlinedVector<BufferAllocation::Slice, 2> operands;
  for (int i = 0; i < instr->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice operand,
                        GetAllocationSliceForHlo(instr->operand(i), {}));
    operands.push_back(operand);
  }

  absl::InlinedVector<BufferAllocation::Slice, 2> results;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result,
                      GetAllocationSliceForHlo(instr, {0}));
  results.push_back(result);

  BufferAllocation::Slice scratch;
  if (instr->operand_count() == 1) {
    TF_ASSIGN_OR_RETURN(scratch, GetAllocationSliceForHlo(instr, {1}));
  } else {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result,
                        GetAllocationSliceForHlo(instr, {1}));
    results.push_back(result);
    TF_ASSIGN_OR_RETURN(scratch, GetAllocationSliceForHlo(instr, {2}));
  }

  TF_ASSIGN_OR_RETURN(xla::SortOptions options,
                      instr->backend_config<xla::SortOptions>());
  const Shape& operand_shape = instr->operand(0)->shape();
  auto thunk = std::make_unique<CubSortThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      operand_shape.element_type(),
      instr->operand_count() == 2
          ? std::optional(instr->operand(1)->shape().element_type())
          : std::nullopt,
      operands, results, scratch, options.descending(),
      Product(operand_shape.dimensions()) /
          operand_shape.dimensions(operand_shape.rank() - 1));
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCholeskyThunk(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(CholeskyOptions options,
                      instr->backend_config<CholeskyOptions>());
  const Shape& shape = instr->operand(0)->shape();
  int ndim = shape.dimensions_size();
  CHECK_GE(ndim, 2);
  int64_t n = shape.dimensions(ndim - 1);

  const absl::Span<const int64_t>& dims = shape.dimensions();
  int64_t batch_size =
      std::accumulate(dims.begin(), dims.end() - 2, int64_t{1},
                      [](int64_t a, int64_t b) { return a * b; });

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice operand_buffer,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_buffer,
                      GetAllocationSliceForHlo(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice workspace_buffer,
                      GetAllocationSliceForHlo(instr, {1}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice info_buffer,
                      GetAllocationSliceForHlo(instr, {2}));

  ThunkSequence thunks;

  if (operand_buffer != a_buffer) {
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr),
        /*source_buffer=*/operand_buffer,
        /*destination_buffer=*/a_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
  }

  thunks.push_back(std::make_unique<CholeskyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), options,
      PtxOptsFromDebugOptions(ir_emitter_context_->debug_options()), a_buffer,
      workspace_buffer, info_buffer, shape.element_type(), batch_size, n));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(thunks)));
  }

  return absl::OkStatus();
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

absl::Status IrEmitterUnnested::EmitCustomCallThunk(
    const HloCustomCallInstruction* instr) {
  const std::string& call_target_name = instr->custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls with
  // a rich type safe API. It's under construction and not fully supported.
  bool is_ffi_custom_call =
      instr->api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(platform_name()));

  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(call_target_name, platform_name());

  // At least one implementation should be available at run time.
  bool found_custom_call = !is_ffi_custom_call && call_target != nullptr;
  bool found_ffi_handler = is_ffi_custom_call && registration.ok();

  if (!found_custom_call && !found_ffi_handler) {
    auto& debug_options = ir_emitter_context_->debug_options();

    // If true, then all custom calls that are not found in custom call or FFI
    // registries will become no-op (we don't emit any thunks for them).
    if (debug_options.xla_gpu_mock_custom_calls()) {
      return absl::OkStatus();
    }

    return absl::UnimplementedError(
        absl::StrCat("No registered implementation for custom call to ",
                     call_target_name, " for platform ", platform_name()));
  }

  using Slices = std::vector<std::optional<CustomCallThunk::Slice>>;

  Slices operands;
  for (auto* operand : instr->operands()) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          TF_ASSIGN_OR_RETURN(auto slice,
                              GetAllocationSliceForHlo(operand, index));
          operands.push_back(CustomCallThunk::Slice{slice, subshape});
          return absl::OkStatus();
        }));
  }

  Slices results;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSliceForHlo(instr, index));
        results.push_back(CustomCallThunk::Slice{slice, subshape});
        return absl::OkStatus();
      }));

  // For legacy custom calls we convert all API versions into the latest
  // status-returning one and pass backend config as an opaque string.
  CustomCallThunk::CustomCallTarget custom_call_target;
  std::string opaque;

  // For XLA FFI handlers we decode opaque backend config into attributes map
  // at IR emission time, so that we do not need to parse MLIR at run time. For
  // FFI handlers backend config must be a compatible MLIR dictionary.
  CustomCallThunk::AttributesMap attributes;

  // For information about this calling convention, see
  // xla/g3doc/custom_call.md.
  switch (instr->api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
      using original_call_type =
          void (*)(CustomCallThunk::Stream /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/);
      custom_call_target = [call_target](CustomCallThunk::Stream stream,
                                         void** buffers, const char* opaque,
                                         size_t opaque_len,
                                         XlaCustomCallStatus*) {
        auto typed_call_target =
            reinterpret_cast<original_call_type>(call_target);
        typed_call_target(stream, buffers, opaque, opaque_len);
      };
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      using status_returning_call_type =
          void (*)(CustomCallThunk::Stream /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/,
                   XlaCustomCallStatus* /*status*/);
      custom_call_target =
          reinterpret_cast<status_returning_call_type>(call_target);
      break;
    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      // We already checked `handler` above.
      break;
    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      instr->api_version());
  }

  auto& backend_config_str = instr->raw_backend_config_string();
  switch (instr->api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      if (!backend_config_str.empty()) {
        opaque = backend_config_str;
      }
      break;

    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      if (!backend_config_str.empty()) {
        mlir::Attribute attr = mlir::parseAttribute(
            backend_config_str, ir_emitter_context_->mlir_context());
        if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
          TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
          break;
        }
        return absl::InternalError(
            "Unsupported backend config. Expected a string parsable into "
            "dictionary attribute");
      }
      break;

    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      instr->api_version());
  }

  auto ffi_thunk = [&] {
    auto& called_computations = instr->called_computations();
    return std::make_unique<CustomCallThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), registration->bundle,
        std::move(operands), std::move(results), std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0]);
  };

  auto legacy_thunk = [&] {
    return std::make_unique<CustomCallThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr),
        std::move(custom_call_target), std::move(operands), std::move(results),
        std::move(opaque));
  };

  AddThunkToThunkSequence(found_ffi_handler ? ffi_thunk() : legacy_thunk());

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitFftThunk(const HloFftInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                      GetAllocationSliceForHlo(instr));
  AddThunkToThunkSequence(
      std::make_unique<FftThunk>(Thunk::ThunkInfo::WithProfileAnnotation(instr),
                                 instr->fft_type(), instr->fft_length(),
                                 /*input_buffer=*/arg_slice,
                                 /*output_buffer=*/dest_slice,
                                 /*input_shape=*/instr->operand(0)->shape(),
                                 /*output_shape=*/instr->shape()));
  return absl::OkStatus();
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

absl::Status IrEmitterUnnested::EmitTriangularSolveCustomCall(
    const HloInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 2);
  auto operands = instr->operands();
  TF_RET_CHECK(instr->shape().IsTuple() &&
               instr->shape().tuple_shapes_size() == 2);

  // We expect Fortran layout for everything other than the temp buffer (the
  // last operand).  Fortran layout is not XLA default layout with elements 0
  // and 1 swapped.  For example instead of default layout {3,2,1,0} we'd have
  // Fortran layout {2,3,1,0}.
  auto has_fortran_layout = [](const Layout& layout) {
    int n = layout.minor_to_major_size();
    return layout.minor_to_major(0) == n - 2 &&
           layout.minor_to_major(1) == n - 1;
  };
  TF_RET_CHECK(has_fortran_layout(operands[0]->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(operands[1]->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(instr->shape().tuple_shapes(0).layout()));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_slice,
                      GetAllocationSliceForHlo(operands[0]));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b_slice,
                      GetAllocationSliceForHlo(operands[1]));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForHlo(instr, {0}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice temp_slice,
                      GetAllocationSliceForHlo(instr, {1}));

  const Shape b_shape = operands[1]->shape();
  const PrimitiveType elem_ty = b_shape.element_type();

  TriangularSolveOptions backend_config;
  auto& backend_config_str = instr->raw_backend_config_string();
  if (!backend_config_str.empty()) {
    TF_RETURN_IF_ERROR(
        tsl::HumanReadableJsonToProto(backend_config_str, &backend_config));
  }

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  if (b_slice != result_slice) {
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr),
        /*source_buffer=*/b_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_shape)));
  }

  int64_t m = b_shape.dimensions(b_shape.rank() - 2);
  int64_t n = b_shape.dimensions(b_shape.rank() - 1);
  int64_t batch_size = std::accumulate(
      b_shape.dimensions().begin(), b_shape.dimensions().end() - 2, int64_t{1},
      [](int64_t a, int64_t b) { return a * b; });
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_ty);
  int64_t a_batch_stride =
      backend_config.left_side() ? m * m * elem_size : n * n * elem_size;
  int64_t b_batch_stride = m * n * elem_size;
  thunks.push_back(std::make_unique<TriangularSolveThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), backend_config,
      PtxOptsFromDebugOptions(ir_emitter_context_->debug_options()),
      /*a_buffer=*/a_slice, /*b_buffer=*/result_slice, temp_slice, elem_ty,
      batch_size, m, n, a_batch_stride, b_batch_stride));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(instr);
    // Don't repeat the annotation from inside thunks
    thunk_info.profile_annotation = {};
    AddThunkToThunkSequence(
        std::make_unique<SequentialThunk>(thunk_info, std::move(thunks)));
  }
  return absl::OkStatus();
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

absl::Status IrEmitterUnnested::EmitTopKCustomCall(
    const HloCustomCallInstruction* instr) {
  auto operands = instr->operands();
  const auto& shape = instr->shape();
  TF_RET_CHECK(operands.size() == 1)
      << "Expect only 1 operand for TopK custom call.";
  TF_RET_CHECK(shape.IsTuple())
      << "Expect TopK custom call to have tuple shape.";
  TF_RET_CHECK(shape.tuple_shapes_size() == 2)
      << "Expect TopK custom call shape to have exactly 2 sub-shapes.";

  auto data_shape = operands[0]->shape();
  auto top_elements_shape = shape.tuple_shapes()[0];
  auto indices_shape = shape.tuple_shapes()[1];

  TF_RET_CHECK(data_shape.rank() <= 2) << "Invalid input shape.";
  TF_RET_CHECK(indices_shape.element_type() == PrimitiveType::S32)
      << "Indices should be S32.";

  bool has_batch = data_shape.rank() == 2;
  auto [batch_size, n, k] =
      has_batch
          ? std::tuple<size_t, size_t, size_t>{data_shape.dimensions(0),
                                               data_shape.dimensions(1),
                                               top_elements_shape.dimensions(1)}
          : std::tuple<size_t, size_t, size_t>{
                1, data_shape.dimensions(0), top_elements_shape.dimensions(0)};

  // Load TopK custom kernel.
  TF_ASSIGN_OR_RETURN(CustomKernel kernel,
                      kernel::topk::GetTopKKernel(
                          "topk", data_shape.element_type(), n, k, batch_size));

  // Prepare kernel arguments.
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context_->buffer_assignment(), instr,
                              operands));

  auto thunk = std::make_unique<CustomKernelThunk>(
      instr, std::move(kernel), std::move(kernel_arguments.args()));
  AddThunkToThunkSequence(std::move(thunk));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitTritonCustomCall(
    const HloCustomCallInstruction* instr) {
#if !GOOGLE_CUDA
  return absl::UnimplementedError("Triton support requires CUDA");
#else
  auto generate = [this, &instr]() -> absl::StatusOr<KernelReuseCache::Entry> {
    mlir::MLIRContext& mlir_context = *ir_emitter_context_->mlir_context();
    LoadMlirDialectsForTriton(mlir_context);
    auto call =
        TritonCall::Parse(instr->raw_backend_config_string(), &mlir_context);
    auto kernel_name =
        ir_emitter_context_->name_uniquer()->GetUniqueName(call.name);
    VLOG(3) << "Generating: " << kernel_name;

    auto triton_module =
        mlir::parseSourceString<mlir::ModuleOp>(call.ir, &mlir_context);
    TF_RET_CHECK(triton_module);
    auto triton_fn =
        triton_module->lookupSymbol<mlir::triton::FuncOp>(call.name);
    triton_fn.setName(kernel_name);

    HloModule* hlo_module = instr->GetModule();

    BlockLevelParameters block_level_parameters;
    block_level_parameters.num_stages = call.num_stages;
    block_level_parameters.num_warps = call.num_warps;
    block_level_parameters.num_ctas = 1;

    TF_ASSIGN_OR_RETURN(
        auto result,
        CompileTritonToLLVM(hlo_module->config(), hlo_module->name(),
                            ir_emitter_context_->cuda_compute_capability(),
                            ir_emitter_context_->gpu_device_info(),
                            block_level_parameters, triton_module.get(),
                            ir_emitter_context_->llvm_module(), mlir_context));

    llvm::Function* impl_fn =
        ir_emitter_context_->llvm_module()->getFunction(kernel_name);
    TF_RET_CHECK(impl_fn);
    impl_fn->setName(ir_emitter_context_->name_uniquer()->GetUniqueName(
        kernel_name + "_impl"));

    TF_ASSIGN_OR_RETURN(
        auto kernel_arguments,
        KernelArguments::Create(ir_emitter_context_->buffer_assignment(), instr,
                                instr->operands(),
                                /*dedup=*/false));
    auto launch_dimensions =
        LaunchDimensions(se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
                         se::ThreadDim(call.num_warps * 32));

    llvm::IRBuilder builder(ir_emitter_context_->llvm_module()->getContext());

    llvm::Function* kernel;
    std::vector<llvm_ir::IrArray> inputs;
    std::vector<llvm_ir::IrArray> outputs;
    TF_ASSIGN_OR_RETURN(
        std::tie(kernel, inputs, outputs),
        BuildKernelPrototype(*ir_emitter_context_, kernel_name,
                             kernel_arguments.args(), impl_fn->arg_size(),
                             launch_dimensions, &builder));

    // Move function body into kernel prototype.
    llvm::Function* prototype_func = builder.GetInsertBlock()->getParent();
    prototype_func->splice(prototype_func->begin(), impl_fn);
    for (const auto& [arg, input] : llvm::zip(impl_fn->args(), inputs)) {
      arg.replaceAllUsesWith(input.GetBasePointer());
    }
    impl_fn->eraseFromParent();

    for (auto& arg : prototype_func->args()) {
      // Remove the alignment and aliasing attributes to avoid recompiling the
      // kernel for each alignment/aliasing combination.
      arg.removeAttr(llvm::Attribute::Alignment);
      arg.removeAttr(llvm::Attribute::NoAlias);
    }

    return {{kernel->getName().str(), launch_dimensions, result.cluster_dim,
             result.shmem_bytes}};
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context_->kernel_cache().GetWithStatus(
          instr->raw_backend_config_string(), generate);
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context_->buffer_assignment(), instr,
                              instr->operands(),
                              /*dedup=*/false));

  AddThunkToThunkSequence(std::make_unique<KernelThunk>(
      instr, entry->kernel_name, kernel_arguments.args(),
      entry->launch_dimensions, entry->cluster_dim, entry->shmem_bytes));
  return absl::OkStatus();
#endif  // GOOGLE_CUDA
}

absl::Status IrEmitterUnnested::EmitFusion(const HloFusionInstruction* instr) {
  const se::DeviceDescription& device_info =
      ir_emitter_context_->gpu_device_info();
  const HloFusionAnalysis fusion_analysis =
      HloFusionAnalysis::Create(instr, &device_info);

  std::unique_ptr<FusionInterface> emitter =
      GetFusionEmitter(HloFusionInfo(fusion_analysis, instr,
                                     &ir_emitter_context_->buffer_assignment()),
                       /*is_emission_phase=*/true);
  TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  for (std::unique_ptr<Thunk>& thunk : result.thunks) {
    TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                        stream_assignment.GetSyncExecutionStreamId(instr));
    thunk->set_execution_stream_id(execution_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  }
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitAsyncCustomCallStart(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  auto* async_start = Cast<HloAsyncInstruction>(instr);
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
      stream_assignment.GetAsyncExecutionStreamIds(async_start));
  AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      streams.destination_stream_id, streams.source_stream_id));
  TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                      stream_assignment.GetSyncExecutionStreamId(wrapped));

  auto* custom_call = Cast<HloCustomCallInstruction>(wrapped);
  if (IsLegacyCublasMatmul(*wrapped)) {
    auto status = EmitGemmThunk(custom_call);
    if (status.ok()) {
      thunk_sequence_.back()->set_execution_stream_id(execution_stream_id);
    }
    return status;
  }
#if GOOGLE_CUDA || TF_HIPBLASLT
  if (IsCublasLtMatmul(*wrapped)) {
    auto status = EmitGemmThunk(custom_call);
    if (status.ok()) {
      thunk_sequence_.back()->set_execution_stream_id(execution_stream_id);
    }
    return status;
  }
  if (IsCublasLtMatmulF8(*wrapped)) {
    auto status = EmitGemmThunk(custom_call);
    if (status.ok()) {
      thunk_sequence_.back()->set_execution_stream_id(execution_stream_id);
    }
    return status;
  }
#endif  // GOOGLE_CUDA || TF_HIPBLASLT
  return Internal("Unsupported async custom call instruction: %s",
                  HloOpcodeString(wrapped->opcode()));
}

absl::Status IrEmitterUnnested::AssertNonDeterminismIsOkay(
    const std::string& op_name) {
  if (ir_emitter_context_->debug_options().xla_gpu_deterministic_ops() ||
      ir_emitter_context_->debug_options()
          .xla_gpu_exclude_nondeterministic_ops()) {
    return Unimplemented(
        "HLO instruction %s does not have a deterministic implementation, "
        "but run-to-run determinism is required by --xla_gpu_deterministic_ops "
        "or --xla_gpu_exclude_nondeterministic_ops.",
        op_name);
  }
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSelectAndScatter(
    const HloSelectAndScatterInstruction* instr) {
  const HloInstruction* operand = instr->operand(0);
  const HloInstruction* source = instr->operand(1);
  const Shape source_shape = source->shape();
  const Shape operand_shape = operand->shape();
  const int64_t rank = operand_shape.rank();

  Window window = instr->window();

  CHECK_EQ(rank, source_shape.rank());
  CHECK_EQ(rank, window.dimensions_size());

  std::string name = llvm_ir::IrName(instr);

  TF_RETURN_IF_ERROR(AssertNonDeterminismIsOkay(name));

  const HloInstruction* init_value = instr->operand(2);
  // IrEmitterUnnested implements kSelectAndScatter as a SequentialThunk
  // consisting of two thunks, an initializer KernelThunk that initializes
  // the output and another KernelThunk that accumulates the scattered
  // elements.
  TF_RETURN_IF_ERROR(BuildInitializerThunk(instr, init_value));

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      source_shape, ir_emitter_context_->gpu_device_info());

  // Init value is not needed in IR emission.
  TF_ASSIGN_OR_RETURN(auto ir_arrays,
                      BuildKernelThunkForNonFusionOp(instr, {operand, source},
                                                     launch_dimensions));

  auto& [inputs, outputs] = ir_arrays;
  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 0);
  const llvm_ir::IrArray& operand_array = inputs[0];
  const llvm_ir::IrArray& source_array = inputs[1];
  const llvm_ir::IrArray& out_array = inputs[2];

  llvm::Type* index_type =
      GetIndexTypeForKernel(instr, launch_dimensions.launch_bound(), &b_);
  auto index_typed_constant = [&](uint64_t c) -> llvm::Constant* {
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
  //   if initialized_flag:
  //     output(selected_index) = scatter(output(selected_index), source(S))
  auto loop_body_emitter =
      [&](const llvm_ir::IrArray::Index& source_index) -> absl::Status {
    // Allocate space to keep the currently selected value, its index, and a
    // boolean flag if the value is initialized. The initialized_flag is set
    // false.
    llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_shape.element_type(), module_),
        "selected_value_address", &b_);

    llvm::AllocaInst* selected_index_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            index_type, index_typed_constant(rank), "selected_index_address",
            &b_);

    llvm::AllocaInst* initialized_flag_address =
        llvm_ir::EmitAllocaAtFunctionEntry(b_.getInt1Ty(),
                                           "initialized_flag_address", &b_);
    Store(b_.getInt1(false), initialized_flag_address);

    // Create the inner loop to iterate over the window.
    llvm_ir::ForLoopNest window_loops(absl::StrCat(name, "inner"), &b_,
                                      index_type);

    DimensionVector window_size;
    for (const WindowDimension& dim : window.dimensions()) {
      auto size = static_cast<int64_t>(dim.size());
      window_size.push_back(size);
      CHECK_GT(size, 0);
    }

    const llvm_ir::IrArray::Index window_index = window_loops.AddLoopsForShape(
        ShapeUtil::MakeShape(operand_shape.element_type(), window_size),
        "window");
    llvm_ir::SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(),
                                   &b_);

    // Compute the operand index to visit and evaluate the condition whether the
    // operand index is within the bounds. The unsigned comparison includes
    // checking whether the operand index >= 0.
    std::vector<llvm::Value*> operand_multi_index(source_index.size());
    llvm::Value* in_bounds_condition = b_.getInt1(true);

    for (const auto [i, value] : llvm::enumerate(window.dimensions())) {
      auto stride = static_cast<int64_t>(value.stride());
      auto padding = static_cast<int64_t>(value.padding_low());

      llvm::Value* strided_index =
          NSWMul(source_index[i], index_typed_constant(stride));
      operand_multi_index[i] = NSWSub(NSWAdd(strided_index, window_index[i]),
                                      index_typed_constant(padding));
      llvm::Value* index_condition = ICmpULT(
          operand_multi_index[i],
          index_typed_constant(ShapeUtil::GetDimension(operand_shape, i)));
      in_bounds_condition = And(in_bounds_condition, index_condition);
    }

    // Only need to do something if the operand index is within the bounds.
    // First check if the initialized_flag is set.
    llvm_ir::LlvmIfData if_in_bounds =
        llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, &b_);
    llvm_ir::LlvmIfData if_initialized = llvm_ir::EmitIfThenElse(
        Load(initialized_flag_address->getAllocatedType(),
             initialized_flag_address),
        "initialized", &b_);

    // If the initialized_flag is false, initialize the selected value and index
    // with the currently visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.false_block, &b_);
    const auto save_operand_index =
        [&](const llvm_ir::IrArray::Index& operand_index) {
          for (int64_t i = 0; i < rank; ++i) {
            llvm::Value* selected_index_address_slot =
                InBoundsGEP(selected_index_address->getAllocatedType(),
                            selected_index_address, {b_.getInt32(i)});
            Store(operand_index[i], selected_index_address_slot);
          }
        };
    llvm_ir::IrArray::Index operand_index(operand_multi_index, operand_shape,
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
    llvm::AllocaInst* select_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(PRED, module_), "select_return_buffer",
        &b_);

    const HloComputation* select_computation = instr->select();
    TF_RETURN_IF_ERROR(CallNestedComputation(
        &b_, *ir_emitter_context_, *select_computation,
        {selected_value_address, operand_address}, select_return_buffer));
    llvm::Value* result =
        Load(select_return_buffer->getAllocatedType(), select_return_buffer);

    // If the 'select' function returns false, update the selected value and the
    // index to the currently visiting operand.
    llvm::Value* cond =
        ICmpNE(result,
               llvm::ConstantInt::get(
                   llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0),
               "boolean_predicate");
    llvm_ir::LlvmIfData if_select_lhs =
        llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &b_);
    llvm_ir::SetToFirstInsertPoint(if_select_lhs.false_block, &b_);
    Store(Load(operand_array.GetElementLlvmType(), operand_address),
          selected_value_address);
    save_operand_index(operand_index);

    // If the initialized_flag is true, write to the selected index of the
    // output; otherwise the window is outside the source (in the padding) and
    // should be ignored.
    llvm_ir::SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                                   &b_);
    llvm_ir::LlvmIfData if_should_store = llvm_ir::EmitIfThenElse(
        Load(initialized_flag_address->getAllocatedType(),
             initialized_flag_address),
        "should-store", &b_, /*emit_else=*/false);
    llvm_ir::SetToFirstInsertPoint(if_should_store.true_block, &b_);

    // After iterating over the window elements, scatter the source element to
    // the selected index of the output. The value we store at the output
    // location is computed by calling the `scatter` function with the source
    // value and the current output value.
    std::vector<llvm::Value*> selected_multi_index;
    for (int64_t i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot =
          InBoundsGEP(selected_index_address->getAllocatedType(),
                      selected_index_address, {b_.getInt32(i)});
      selected_multi_index.push_back(
          Load(selected_index_address->getAllocatedType(),
               selected_index_address_slot));
    }
    const Shape& output_shape = instr->shape();
    llvm::Value* source_value_address =
        source_array.EmitArrayElementAddress(source_index, &b_);
    llvm_ir::IrArray::Index selected_index(selected_multi_index, output_shape,
                                           operand_index.GetType());
    llvm::Value* output_value_address =
        out_array.EmitArrayElementAddress(selected_index, &b_);

    const HloComputation* scatter_computation = instr->scatter();
    return EmitAtomicOperationForNestedComputation(
        &b_, *ir_emitter_context_, *scatter_computation, output_value_address,
        source_value_address, source_array.GetElementLlvmType());
  };

  return ParallelLoopEmitter(loop_body_emitter, source_shape, launch_dimensions,
                             &b_)
      .EmitLoop(name, index_type);
}

absl::Status IrEmitterUnnested::EmitWhile(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto config,
                      instr->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) trip_count = config.known_trip_count().n();

  TF_ASSIGN_OR_RETURN(
      auto thunk,
      BuildWhileThunk(instr, Thunk::ThunkInfo::WithProfileAnnotation(instr),
                      trip_count));

  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr) {
  // Emit a kernel to increment the global state for Philox RNG algorithm.
  TF_ASSIGN_OR_RETURN(auto ir_arrays, BuildKernelThunkForNonFusionOp(
                                          instr, {}, LaunchDimensions()));
  auto& [inputs, outputs] = ir_arrays;
  llvm::Value* old_state =
      llvm_ir::RngGetAndUpdateState(instr->delta(), module_, &b_);
  llvm::Value* output_address = inputs[0].EmitArrayElementAddress(
      llvm_ir::IrArray::Index(
          /*linear=*/b_.getInt64(0), instr->shape(), &b_),
      &b_, "rng_state_address");
  Store(old_state, output_address);
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSort(const HloSortInstruction* sort) {
  std::string op_name(sort->name());
  const Shape& keys_shape = sort->operand(0)->shape();
  int64_t dimension_to_sort = sort->sort_dimension();
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    // We assume that the layout of all involved operands and outputs is the
    // same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(keys_shape,
                                                  sort->operand(i)->shape()));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index)));

    BufferAllocation::Slice destination_buffer;
    BufferAllocation::Slice source_address;

    // If possible, we share buffers. If that is not possible, we need to
    // copy the values, because the emitter does the sorting in-place.
    TF_ASSIGN_OR_RETURN(destination_buffer,
                        GetAllocationSliceForHlo(sort, shape_index));
    TF_ASSIGN_OR_RETURN(source_address,
                        GetAllocationSliceForHlo(sort->operand(i), {}));

    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share buffers for
      // key/value sort.
      VLOG(2) << op_name << " requires initial D2D copy for operand " << i;
      AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(sort),
          /*source_buffer=*/source_address,
          /*destination_buffer=*/destination_buffer,
          /*mem_size=*/ShapeUtil::ByteSizeOf(sort->operand(i)->shape())));
    }
  }

  uint64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << op_name << " requires " << num_stages << " stages.";
  CHECK_GE(1ULL << num_stages, dimension_to_sort_bound);
  CHECK_LT(1ULL << (num_stages - 1), dimension_to_sort_bound);

  // Naive C++ code for the outer loops:
  //
  // for (int64_t stage = 0; stage < Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64_t first_xor_mask = (1LL << (stage + 1)) - 1;
  //   SortInPlace(first_xor_mask);
  //   for (int64_t mask = stage - 1; mask >= 0; --mask) {
  //     int64_t later_xor_mask = 1LL << mask;
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

  const uint64_t kTileSize = std::min(2048ULL, 1ULL << num_stages);

  // If we cannot combine several xor masks together, we don't use tiling, so we
  // calculate the standard launch dimensions for the shape. However we only
  // need to iterate through ~half of the dimension to sort (rounded up to the
  // next highest power of 2), because each iteration compares one pair of
  // elements.
  Shape standard_iteration_shape = keys_shape;
  uint64_t standard_num_iterations_in_sort_dim = 1ULL << (num_stages - 1);
  standard_iteration_shape.set_dimensions(dimension_to_sort,
                                          standard_num_iterations_in_sort_dim);

  LaunchDimensions standard_launch_dimensions = CalculateLaunchDimensions(
      standard_iteration_shape, ir_emitter_context_->gpu_device_info());

  // Calculate the launch dimensions for the case where we use tiling. We split
  // the dimension that should be sorted into tiles of size 'kTileSize'. This
  // means we first need to round 'dimension_to_sort_bound' up to be a multiple
  // of the tile size.
  int64_t rounded_bound = RoundUpTo(dimension_to_sort_bound, kTileSize);
  Shape iteration_shape = keys_shape;

  // We iterate through the element pairs that should be compared.
  uint64_t num_iterations_in_sort_dim = rounded_bound / 2;
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  // For correctness reasons we need exactly 'kTileSize' / 2 many threads per
  // block. Each thread is responsible for copying exactly two adjacent elements
  // into shared memory, and then does a comparison of two possibly different
  // elements taken from shared memory.
  const uint64_t kThreadsPerBlock = kTileSize / 2;

  // Check whether we should use any tiling. We might not be able to use it if
  // we have not enough threads, or not enough shared memory.
  int64_t total_shared_memory_needed = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    total_shared_memory_needed +=
        kTileSize * ShapeUtil::ByteSizeOfPrimitiveType(
                        sort->operand(i)->shape().element_type());
  }
  bool no_tiling =
      kThreadsPerBlock >
          ir_emitter_context_->gpu_device_info().threads_per_block_limit() ||
      total_shared_memory_needed >
          ir_emitter_context_->gpu_device_info().shared_memory_per_block();
  VLOG(2) << absl::StreamFormat(
      "%s %s use tiling. No tiling if any of the following is true: "
      "kThreadsPerBlock=%d > threads_per_block_limit=%d, "
      "total_shared_memory_needed=%d > shared_memory_per_block=%d",
      op_name, (no_tiling ? "won't" : "will"), kThreadsPerBlock,
      ir_emitter_context_->gpu_device_info().threads_per_block_limit(),
      total_shared_memory_needed,
      ir_emitter_context_->gpu_device_info().shared_memory_per_block());

  uint64_t num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                op_name, num_blocks, kThreadsPerBlock);
  auto emit_kernel = [&](absl::Span<const int64_t> xor_masks) {
    VLOG(2) << absl::StreamFormat(
        "%s uses kernel for xor masks [%s]", op_name,
        absl::StrJoin(xor_masks, ", ", [](std::string* out, int64_t xor_mask) {
          absl::StrAppendFormat(out, "0x%x", xor_mask);
        }));
    LaunchDimensions launch_dimensions = xor_masks.size() > 1
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    TF_ASSIGN_OR_RETURN(auto ir_arrays, BuildKernelThunkForNonFusionOp(
                                            sort, {}, launch_dimensions));

    auto& [inputs, outputs] = ir_arrays;
    auto* comparator = sort->called_computations().front();
    return llvm_ir::EmitSortInPlace(
        dimension_to_sort, inputs, llvm_ir::IrName(op_name), xor_masks, &b_,
        launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        kTileSize,
        [&](absl::Span<llvm::Value* const> operands, llvm::Value* output) {
          return CallNestedComputation(&b_, *ir_emitter_context_, *comparator,
                                       operands, output);
        });
  };
  std::vector<int64_t> xor_masks;
  for (int64_t stage = 0; stage < num_stages; ++stage) {
    for (int64_t mask = stage; mask >= 0; --mask) {
      int64_t xor_mask;
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
  return absl::OkStatus();
}

template <typename ThunkType>
absl::Status IrEmitterUnnested::EmitReplicaOrPartitionId(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForHlo(instr, {}));
  auto thunk = std::make_unique<ThunkType>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), result_slice);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCollectivePermute(
    const HloCollectivePermuteInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 1);
  auto* operand = instr->operand(0);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice source_slice,
                      GetAllocationSliceForHlo(operand));
  // First output is aliased.
  TF_RET_CHECK(
      instr->shape().IsTuple() && instr->shape().tuple_shapes_size() == 2 &&
      Shape::Equal().IgnoreMemorySpaceInLayout()(
          instr->shape().tuple_shapes(0), instr->shape().tuple_shapes(1)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForHlo(instr, {1}));

  const Shape shape = operand->shape();
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();
  const int64_t src_memory_space = shape.layout().memory_space();
  const int64_t dst_memory_space =
      instr->shape().tuple_shapes(1).layout().memory_space();

  if (NcclCollectivePermuteStartThunk::IsDegenerate(instr, replica_count,
                                                    partition_count)) {
    // For a degenerate collective permute, just generate a copy thunk.
    AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
    // Signal that start thunk not created with nullptr.
    GetCollectivesAsyncEvents().try_emplace(instr, nullptr);
  } else {
    const NcclCollectiveThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(shape),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/result_slice,
        /*source_memory_space=*/src_memory_space,
        /*destination_memory_space=*/dst_memory_space};
    auto thunk = std::make_unique<NcclCollectivePermuteStartThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), NcclApi::Default(),
        instr, replica_count, partition_count, buffer,
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
    GetCollectivesAsyncEvents().try_emplace(instr, thunk->async_events());
    AddThunkToThunkSequence(std::move(thunk));
  }
  return absl::OkStatus();
}

template <typename NcclThunkType, typename HloInstType>
absl::Status IrEmitterUnnested::EmitNcclThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloInstType* inst, std::optional<bool> use_global_device_ids) {
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  VLOG(2) << NcclThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << inst->operand_count();

  // A given collective op can be degenerate if across all groups formed
  // by it are singleton. In such a case, we don't need to do any communication
  // and we can just copy the input to the output.
  bool is_degenerate = GetNcclCollectiveConfig(inst, use_global_device_ids)
                           .IsDegenerate(replica_count, partition_count);
  absl::Status implementable_status =
      NcclThunkType::CheckImplementable(inst, replica_count, partition_count);
  bool should_use_nccl_thunk = !is_degenerate && implementable_status.ok();

  // Stash relevant information in NcclCollectiveThunk::Buffer even if we may
  // not generate an NcclCollectiveThunk.
  std::vector<NcclCollectiveThunk::Buffer> buffers;

  int64_t operand_count = inst->operand_count();
  buffers.reserve(operand_count);

  // Adds a source and destination buffers pair to `buffers`.
  auto add_buffer = [&](int64_t element_count, BufferAllocation::Slice src,
                        int64_t src_memory_space, BufferAllocation::Slice dst,
                        int64_t dst_memory_space) {
    buffers.push_back(NcclCollectiveThunk::Buffer{
        /*element_count=*/element_count,
        /*source_buffer=*/src,
        /*destination_buffer=*/dst,
        /*source_memory_space=*/src_memory_space,
        /*destination_memory_space=*/dst_memory_space,
        /*source_value=*/nullptr,
        /*destination_value=*/nullptr});
  };

  if (kind == Thunk::Kind::kNcclAllGatherStart) {
    // Start operations return a tuple of (<<inputs>>, <<outputs>>) where
    // outputs can be a tuple itself (if operation has multiple operands).
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx = operand_count > 1 ? ShapeIndex({1, i}) : ShapeIndex({1});
      const Shape& src_shape = inst->operand(i)->shape();
      const Shape& dst_shape = ShapeUtil::GetSubshape(inst->shape(), idx);
      TF_ASSIGN_OR_RETURN(auto src, GetAllocationSliceForHlo(inst->operand(i)));
      TF_ASSIGN_OR_RETURN(auto dst, GetAllocationSliceForHlo(inst, idx));
      add_buffer(ShapeUtil::ElementsIn(src_shape), src,
                 src_shape.layout().memory_space(), dst,
                 dst_shape.layout().memory_space());
    }

  } else {
    // For other operations simply zip operands with results.
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx = operand_count > 1 ? ShapeIndex({i}) : ShapeIndex({});
      const Shape& src_shape = inst->operand(i)->shape();
      const Shape& dst_shape = ShapeUtil::GetSubshape(inst->shape(), idx);
      TF_ASSIGN_OR_RETURN(auto src, GetAllocationSliceForHlo(inst->operand(i)));
      TF_ASSIGN_OR_RETURN(auto dst, GetAllocationSliceForHlo(inst, idx));
      add_buffer(ShapeUtil::ElementsIn(src_shape), src,
                 src_shape.layout().memory_space(), dst,
                 dst_shape.layout().memory_space());
    }
  }

  if (should_use_nccl_thunk) {
    auto thunk = std::make_unique<NcclThunkType>(
        Thunk::ThunkInfo::WithProfileAnnotation(inst), NcclApi::Default(), inst,
        /*buffers=*/std::move(buffers));
    GetCollectivesAsyncEvents().insert({async_start, thunk->async_events()});
    AddThunkToThunkSequence(std::move(thunk));
    return absl::OkStatus();
  }

  if (!is_degenerate) {
    return implementable_status;
  }

  // Signal that start thunk not created with nullptr.
  GetCollectivesAsyncEvents().insert({async_start, nullptr});

  VLOG(1) << "Collective call is degenerate, not doing NCCL call";

  // Degenerate collectives are simply identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64_t i = 0; i < buffers.size(); i++) {
    const Shape shape = inst->operand(i)->shape();
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(inst),
        /*source_buffer=*/buffers[i].source_buffer,
        /*destination_buffer=*/buffers[i].destination_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
  }
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(inst), std::move(thunks)));
  }
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitNcclAsyncDone(Thunk::Kind kind,
                                                  const HloInstruction* inst) {
  CollectivesAsyncEvents& collectives_async_events =
      GetCollectivesAsyncEvents();
  if (kind == Thunk::Kind::kNcclRecvDone ||
      kind == Thunk::Kind::kNcclSendDone) {
    const HloChannelInstruction* done = DynCast<HloChannelInstruction>(inst);
    int64_t channel_id = done->channel_id().value();
    // We only pipeline Send/Recv when channel_id > 0, and allows multiple
    // and potentially interleaving Send/Recv chains using channel_id = 0.
    if (MayPipelineSendRecvChannel(channel_id)) {
      auto it = collectives_async_events.find(
          GetSendRecvAsyncEventsKey(kind, channel_id));
      TF_RET_CHECK(it != collectives_async_events.end())
          << "couldn't find async events for channel_id " << channel_id;
      AddThunkToThunkSequence(std::make_unique<NcclCollectiveDoneThunk>(
          kind, Thunk::ThunkInfo::WithProfileAnnotation(inst), it->second,
          GetStreamKindForSendRecv(DynCast<HloSendRecvInstruction>(inst))));
      return absl::OkStatus();
    }
  }

  const HloInstruction* start = inst->operand(0);
  auto async_events = collectives_async_events.extract(start);
  TF_RET_CHECK(async_events)
      << "couldn't find async events for start operation";

  // Can be null if no start thunk was created (e.g. if the start op is
  // degenerate), in which case there's nothing to do here.
  if (async_events.mapped()) {
    AddThunkToThunkSequence(std::make_unique<NcclCollectiveDoneThunk>(
        kind, Thunk::ThunkInfo::WithProfileAnnotation(inst),
        std::move(async_events.mapped()), AsyncStreamKind::kCollective));
  }
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitInfeed(const HloInfeedInstruction* instr) {
  // Infeed instruction returns a tuple containing the result data and a token.
  // We only need the result data to construct the infeed thunk.
  std::vector<ShapedSlice> shaped_slices;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple() || subshape.IsToken()) return absl::OkStatus();
        if (subshape.IsArray()) {
          TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                              GetAllocationSliceForHlo(instr, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        instr->ToString(), index.ToString());
      }));

  auto thunk = std::make_unique<InfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(shaped_slices));
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitOutfeed(
    const HloOutfeedInstruction* instr) {
  // HLO outfeed instruction has 2 operands, the source and a token, and a
  // single token output.
  const HloInstruction* source = instr->operand(0);
  std::vector<ShapedSlice> shaped_slices;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      source->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) return absl::OkStatus();
        if (subshape.IsArray()) {
          TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                              GetAllocationSliceForHlo(source, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        source->ToString(), index.ToString());
      }));

  auto thunk = std::make_unique<OutfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(shaped_slices));
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::vector<llvm_ir::IrArray> /*inputs*/,
                         std::vector<llvm_ir::IrArray> /*outputs*/>>
IrEmitterUnnested::BuildKernelThunkForNonFusionOp(
    const HloInstruction* hlo,
    absl::Span<const HloInstruction* const> needed_operands,
    const LaunchDimensions& launch_dimensions) {
  std::string suggested_kernel_name(hlo->name());

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context_->buffer_assignment(), hlo,
                              needed_operands));

  VLOG(3) << "Generating (without reuse check): " << suggested_kernel_name;

  llvm::Function* kernel;
  std::vector<llvm_ir::IrArray> inputs;
  std::vector<llvm_ir::IrArray> outputs;
  TF_ASSIGN_OR_RETURN(
      std::tie(kernel, inputs, outputs),
      BuildKernelPrototype(
          *ir_emitter_context_, suggested_kernel_name, kernel_arguments.args(),
          kernel_arguments.args().size(), launch_dimensions, &b_));

  AddThunkToThunkSequence(std::make_unique<KernelThunk>(
      hlo, kernel->getName().str(), kernel_arguments.args(), launch_dimensions,
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0));

  return {{inputs, outputs}};
}

absl::Status IrEmitterUnnested::BuildInitializerThunk(
    const HloInstruction* instr, const HloInstruction* init_value) {
  // initial value must be a scalar memref.
  TF_RET_CHECK(init_value->shape().rank() == 0);

  auto maybe_dest_slice = GetAllocationSliceForHlo(instr, {});
  if (!maybe_dest_slice.ok()) return maybe_dest_slice.status();

  BufferAllocation::Slice dest_slice = *maybe_dest_slice;

  TF_ASSIGN_OR_RETURN(std::optional<std::unique_ptr<Thunk>> constant_init_thunk,
                      BuildConstantInitializerThunk(*ir_emitter_context_, instr,
                                                    init_value, dest_slice));
  if (constant_init_thunk) {
    AddThunkToThunkSequence(*std::move(constant_init_thunk));
    return absl::OkStatus();
  }

  // Otherwise fall back to our slow initializer code. The thunk in this case
  // will just need the IR arrays for the initial value and the destination.
  const Shape& dest_shape = instr->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      dest_shape, ir_emitter_context_->gpu_device_info());
  TF_ASSIGN_OR_RETURN(
      auto ir_arrays,
      BuildKernelThunkForNonFusionOp(instr, {init_value}, launch_dimensions));
  auto& [inputs, outputs] = ir_arrays;
  auto init_array = inputs[0];

  std::string name = llvm_ir::IrName(instr, "init");
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(
                         [=](const llvm_ir::IrArray::Index& index) {
                           return init_array.EmitReadArrayElement(index, &b_);
                         },
                         {inputs[1]}, launch_dimensions, &b_)
                         .EmitLoop(name));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildWhileThunk(
    const HloInstruction* instr, const Thunk::ThunkInfo& thunk_info,
    std::optional<int64_t> trip_count) {
  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Generate thunk sequence for while 'condition'.
  auto ir_emitter_condition = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_condition->EmitHloComputation(condition));

  // Generate thunk sequence for while 'body'.
  auto ir_emitter_body = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_body->EmitHloComputation(body));

  // Buffer slice holding while loop predicate.
  TF_ASSIGN_OR_RETURN(
      auto pred, GetAllocationSliceForHlo(condition->root_instruction(), {}));

  return std::unique_ptr<Thunk>(new WhileThunk(
      thunk_info, pred, ir_emitter_condition->ConsumeThunkSequence(),
      ir_emitter_body->ConsumeThunkSequence(), trip_count));
}

absl::Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter) {
  return Internal("This should be unreachable");
}

static absl::flat_hash_map<std::string, std::string> ConvertFrontendAttributes(
    const FrontendAttributes& attrs) {
  absl::flat_hash_map<std::string, std::string> result;
  for (auto& [k, v] : attrs.map()) result[k] = v;
  return result;
}

static std::optional<GlobalDeviceId> DeviceConstraint(
    const HloInstruction* hlo) {
  if (hlo->has_sharding() && hlo->sharding().HasUniqueDevice()) {
    return GlobalDeviceId(hlo->sharding().GetUniqueDevice());
  }
  return std::nullopt;
}

absl::StatusOr<bool> ShapeHasHostMemorySpace(Shape shape, int index,
                                             int host_memory_space) {
  return shape.tuple_shapes(index).has_layout() &&
         shape.tuple_shapes(index).layout().memory_space() == host_memory_space;
}

absl::Status IrEmitterUnnested::EmitCopyStartThunk(
    const HloCopyStartInstruction* copy_start_instr) {
  // copy-start has a tuple shape: {host, device, context},
  // or {device, host, context}.
  // Only the destination shape is needed to get the output buffer.
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(copy_start_instr,
                                               /*index=*/{0}));

  const HloInstruction* src = copy_start_instr->operand(0);
  const Shape& input_shape = src->shape();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(src, {}));
  const Shape& shape = copy_start_instr->shape();
  CHECK(shape.IsTuple());
  int host_memory_space = static_cast<int>(stream_executor::MemoryType::kHost);
  TF_ASSIGN_OR_RETURN(bool is_dst_host_memory,
                      ShapeHasHostMemorySpace(shape, 0, host_memory_space));
  TF_ASSIGN_OR_RETURN(bool is_src_host_memory,
                      ShapeHasHostMemorySpace(shape, 1, host_memory_space));
  if (is_dst_host_memory == is_src_host_memory) {
    return absl::InternalError(absl::StrFormat(
        "Copy-start %s doesn't have correct host memory space color S(%d)",
        copy_start_instr->ToString(),
        static_cast<int>(stream_executor::MemoryType::kHost)));
  }
  if (is_dst_host_memory) {
    auto thunk = std::make_unique<DeviceToHostCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    AddThunkToThunkSequence(std::move(thunk));
  } else {
    auto thunk = std::make_unique<HostToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    AddThunkToThunkSequence(std::move(thunk));
  }

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCopyDoneThunk(const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  auto thunk = std::make_unique<CopyDoneThunk>(
      Thunk::kCopyDone,
      Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
      /*copy_events=*/copy_events_,
      /*copy_start_instr=*/copy_start_instr);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSendThunk(const HloSendInstruction* instr) {
  if (!instr->channel_id().has_value())
    return absl::InternalError("Unknown send instruction channel id");

  const HloInstruction* src = instr->operand(0);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice buffer,
                      GetAllocationSliceForHlo(src, {}));
  if (!instr->is_host_transfer()) {
    const auto& hlo_config = ir_emitter_context_->hlo_module().config();
    const int64_t replica_count = hlo_config.replica_count();
    const int64_t partition_count = hlo_config.num_partitions();
    const int64_t memory_space =
        instr->shape().IsTuple()
            ? instr->shape().tuple_shapes(0).layout().memory_space()
            : instr->shape().layout().memory_space();
    const NcclCollectiveThunk::Buffer nccl_buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(src->shape()),
        /*source_buffer=*/buffer,
        /*destination_buffer=*/buffer,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    auto thunk = std::make_unique<NcclSendThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), NcclApi::Default(),
        instr, replica_count, partition_count, nccl_buffer);
    CollectivesAsyncEvents& collectives_async_events =
        GetCollectivesAsyncEvents();
    int64_t channel_id = instr->channel_id().value();
    if (MayPipelineSendRecvChannel(channel_id)) {
      std::pair<bool, int64_t> async_events_key =
          GetSendRecvAsyncEventsKey(Thunk::Kind::kNcclSendDone, channel_id);
      auto it = collectives_async_events.find(async_events_key);
      if (it != collectives_async_events.end()) {
        VLOG(0) << "Found async events " << it->second.get();
        thunk->set_async_events(it->second);
      } else {
        VLOG(0) << "Used Async events create for thunk "
                << thunk->async_events().get();
        collectives_async_events.emplace(async_events_key,
                                         thunk->async_events());
      }
    } else {
      collectives_async_events.try_emplace(instr, thunk->async_events());
    }

    AddThunkToThunkSequence(std::move(thunk));
    return absl::OkStatus();
  }

  AddThunkToThunkSequence(std::make_unique<SendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), src->shape(), buffer,
      *instr->channel_id(), send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSendDoneThunk(
    const HloSendDoneInstruction* instr) {
  if (!instr->channel_id().has_value())
    return absl::InternalError("Unknown send done instruction channel id");

  if (!instr->is_host_transfer()) {
    return EmitNcclAsyncDone(Thunk::kNcclSendDone, instr);
  }

  AddThunkToThunkSequence(std::make_unique<SendDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), *instr->channel_id(),
      send_recv_events_, DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitRecvThunk(const HloRecvInstruction* instr) {
  if (!instr->channel_id().has_value())
    return absl::InternalError("Unknown recv instruction channel id");
  TF_RET_CHECK(instr->shape().IsTuple());
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice buffer,
                      GetAllocationSliceForHlo(instr, {0}));

  if (!instr->is_host_transfer()) {
    const auto& hlo_config = ir_emitter_context_->hlo_module().config();
    const int64_t replica_count = hlo_config.replica_count();
    const int64_t partition_count = hlo_config.num_partitions();

    const int64_t memory_space =
        instr->shape().IsTuple()
            ? instr->shape().tuple_shapes(0).layout().memory_space()
            : instr->shape().layout().memory_space();

    const NcclCollectiveThunk::Buffer nccl_buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(instr->shape().tuple_shapes(0)),
        /*source_buffer=*/buffer,
        /*destination_buffer=*/buffer,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    auto thunk = std::make_unique<NcclRecvThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), NcclApi::Default(),
        instr, replica_count, partition_count, nccl_buffer);
    CollectivesAsyncEvents& collectives_async_events =
        GetCollectivesAsyncEvents();
    int64_t channel_id = instr->channel_id().value();
    if (MayPipelineSendRecvChannel(channel_id)) {
      std::pair<bool, int64_t> async_events_key =
          GetSendRecvAsyncEventsKey(Thunk::Kind::kNcclRecvDone, channel_id);
      auto it = collectives_async_events.find(async_events_key);

      if (it != GetCollectivesAsyncEvents().end()) {
        thunk->set_async_events(it->second);
      } else {
        collectives_async_events.emplace(async_events_key,
                                         thunk->async_events());
      }
    } else {
      collectives_async_events.try_emplace(instr, thunk->async_events());
    }

    AddThunkToThunkSequence(std::move(thunk));
    return absl::OkStatus();
  }

  AddThunkToThunkSequence(std::make_unique<RecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      instr->shape().tuple_shapes()[0], buffer, *instr->channel_id(),
      send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitRecvDoneThunk(
    const HloRecvDoneInstruction* instr) {
  if (!instr->channel_id().has_value())
    return absl::InternalError("Unknown recv done instruction channel id");

  if (!instr->is_host_transfer()) {
    return EmitNcclAsyncDone(Thunk::kNcclRecvDone, instr);
  }

  AddThunkToThunkSequence(std::make_unique<RecvDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), *instr->channel_id(),
      send_recv_events_, DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kAllGatherDone:
      return EmitNcclAsyncDone(Thunk::kNcclAllGatherDone, instr);
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(instr);
      return EmitNcclThunk<NcclAllGatherStartThunk, HloAllGatherInstruction>(
          Thunk::kNcclAllGatherStart, all_gather, all_gather,
          all_gather->use_global_device_ids());
    }

    case HloOpcode::kAllReduceDone:
      return EmitNcclAsyncDone(Thunk::kNcclAllReduceDone, instr);
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(instr);
      return EmitNcclThunk<NcclAllReduceStartThunk, HloAllReduceInstruction>(
          Thunk::kNcclAllReduceStart, all_reduce, all_reduce,
          all_reduce->use_global_device_ids());
    }
    case HloOpcode::kAsyncDone: {
      const HloInstruction* wrapped = instr->async_wrapped_instruction();
      switch (wrapped->opcode()) {
        case HloOpcode::kReduceScatter:
          return EmitNcclAsyncDone(Thunk::kNcclReduceScatterDone, instr);
        case HloOpcode::kAllToAll:
          return EmitNcclAsyncDone(Thunk::kNcclAllToAllDone, instr);
        case HloOpcode::kCollectiveBroadcast:
          return EmitNcclAsyncDone(Thunk::kNcclCollectiveBroadcastDone, instr);
        case HloOpcode::kFusion:
        case HloOpcode::kCustomCall: {
          // Wait until the concurrent stream has finished.
          auto* async_done = Cast<HloAsyncInstruction>(instr);
          const ExecutionStreamAssignment& stream_assignment =
              ir_emitter_context_->execution_stream_assignment();
          TF_ASSIGN_OR_RETURN(
              ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
              stream_assignment.GetAsyncExecutionStreamIds(async_done));
          AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
              Thunk::ThunkInfo::WithProfileAnnotation(instr),
              streams.source_stream_id, streams.destination_stream_id));
          return absl::OkStatus();
        }
        default:
          return Internal("Unsupported async done wrapped instruction: %s",
                          HloOpcodeString(wrapped->opcode()));
      }
    }
    case HloOpcode::kAsyncStart: {
      const HloInstruction* wrapped = instr->async_wrapped_instruction();
      switch (wrapped->opcode()) {
        case HloOpcode::kReduceScatter: {
          auto* reduce_scatter = Cast<HloReduceScatterInstruction>(wrapped);
          return EmitNcclThunk<NcclReduceScatterStartThunk,
                               HloReduceScatterInstruction>(
              Thunk::kNcclReduceScatter, instr, reduce_scatter,
              reduce_scatter->use_global_device_ids());
        }
        case HloOpcode::kAllToAll: {
          auto* all_to_all = Cast<HloAllToAllInstruction>(wrapped);
          return EmitNcclThunk<NcclAllToAllStartThunk, HloAllToAllInstruction>(
              Thunk::kNcclAllToAll, instr, all_to_all, std::nullopt);
        }
        case HloOpcode::kCollectiveBroadcast: {
          auto* collective_broadcast =
              Cast<HloCollectiveBroadcastInstruction>(wrapped);
          return EmitNcclThunk<NcclCollectiveBroadcastStartThunk,
                               HloCollectiveBroadcastInstruction>(
              Thunk::kNcclCollectiveBroadcast, instr, collective_broadcast,
              std::nullopt);
        }
        case HloOpcode::kFusion: {
          // We'll launch the fusion computation on a concurrent stream. The
          // concurrent stream needs to first wait until the main stream has
          // finished calculating any values that may be used as inputs to the
          // fusion computation. We enforce this by inlining a `WaitForStreams`
          // thunk.
          auto* async_start = Cast<HloAsyncInstruction>(instr);
          const ExecutionStreamAssignment& stream_assignment =
              ir_emitter_context_->execution_stream_assignment();
          TF_ASSIGN_OR_RETURN(
              ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
              stream_assignment.GetAsyncExecutionStreamIds(async_start));
          AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
              Thunk::ThunkInfo::WithProfileAnnotation(instr),
              streams.destination_stream_id, streams.source_stream_id));
          return EmitFusion(Cast<HloFusionInstruction>(wrapped));
        }
        case HloOpcode::kCustomCall: {
          return EmitAsyncCustomCallStart(instr);
        }
        default:
          return Internal("Unsupported async start wrapped instruction: %s",
                          HloOpcodeString(wrapped->opcode()));
      }
    }

    case HloOpcode::kCall:
      return EmitCommandBufferThunk(instr);
    case HloOpcode::kCollectivePermuteDone:
      return EmitNcclAsyncDone(Thunk::kNcclCollectivePermuteDone, instr);
    case HloOpcode::kCollectivePermuteStart:
      return EmitCollectivePermute(
          Cast<HloCollectivePermuteInstruction>(instr));
    case HloOpcode::kConditional:
      return EmitConditional(instr);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(instr));
    case HloOpcode::kCustomCall: {
      auto* custom_call = Cast<HloCustomCallInstruction>(instr);
      if (IsLegacyCublasMatmul(*instr)) {
        return EmitGemmThunk(custom_call);
      }
#if GOOGLE_CUDA || TF_HIPBLASLT
      if (IsCublasLtMatmul(*instr)) {
        return EmitCublasLtMatmulThunk(custom_call);
      }
      if (IsCublasLtMatmulF8(*instr)) {
        return EmitCublasLtMatmulThunkF8(custom_call);
      }
#endif  // GOOGLE_CUDA || TF_HIPBLASLT
#if GOOGLE_CUDA
      if (IsCudnnConvolutionReorder(*instr)) {
        return EmitConvolutionReorderThunk(custom_call);
      }
      if (IsCustomCallToDnnNorm(*instr)) {
        return EmitNormThunk(custom_call);
      }
      if (IsFwdCustomCallTofMHA(*instr)) {
        return EmitFusedMHAThunk(custom_call);
      }
      if (IsBwdCustomCallTofMHA(*instr)) {
        return EmitFusedMHABackwardThunk(custom_call);
      }
#endif  // GOOGLE_CUDA
      if (IsCustomCallToTopK(*instr)) {
        return EmitTopKCustomCall(custom_call);
      }
      if (IsCustomCallToDnnConvolution(*instr)) {
        return EmitConvolutionThunk(custom_call);
      }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (IsCustomCallToCusolver(*instr)) {
        return EmitCholeskyThunk(instr);
      }
      if (IsTriangularSolve(*instr)) {
        return EmitTriangularSolveCustomCall(instr);
      }
      if (IsCubDeviceRadixSort(*instr)) {
        return EmitCubDeviceRadixSort(custom_call);
      }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (custom_call->custom_call_target() == "PadToStatic") {
        return EmitPadToStatic(custom_call);
      }
      if (instr->custom_call_target() == "SliceToDynamic") {
        return EmitSliceToDynamic(custom_call);
      }
      if (instr->custom_call_target() == "__gpu$xla.gpu.triton") {
        return EmitTritonCustomCall(custom_call);
      }
      if (instr->custom_call_target() == kNopCustomCallTarget) {
        return absl::OkStatus();
      }
      return EmitCustomCallThunk(custom_call);
    }
    case HloOpcode::kFusion: {
      return EmitFusion(Cast<HloFusionInstruction>(instr));
    }
    case HloOpcode::kInfeed:
      return EmitInfeed(Cast<HloInfeedInstruction>(instr));
    case HloOpcode::kOutfeed:
      return EmitOutfeed(Cast<HloOutfeedInstruction>(instr));
    case HloOpcode::kPartitionId:
      return EmitReplicaOrPartitionId<PartitionIdThunk>(instr);
    case HloOpcode::kFft:
      return EmitFftThunk(Cast<HloFftInstruction>(instr));

    case HloOpcode::kRecv:
      return EmitRecvThunk(Cast<HloRecvInstruction>(instr));
    case HloOpcode::kRecvDone:
      return EmitRecvDoneThunk(Cast<HloRecvDoneInstruction>(instr));

    case HloOpcode::kReplicaId:
      return EmitReplicaOrPartitionId<ReplicaIdThunk>(instr);
    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateState(
          Cast<HloRngGetAndUpdateStateInstruction>(instr));
    case HloOpcode::kSelectAndScatter:
      return EmitSelectAndScatter(Cast<HloSelectAndScatterInstruction>(instr));

    case HloOpcode::kSend:
      return EmitSendThunk(Cast<HloSendInstruction>(instr));
    case HloOpcode::kSendDone:
      return EmitSendDoneThunk(Cast<HloSendDoneInstruction>(instr));

    case HloOpcode::kSort:
      return EmitSort(Cast<HloSortInstruction>(instr));
    case HloOpcode::kWhile:
      return EmitWhile(instr);
    case HloOpcode::kCopyStart:
      return EmitCopyStartThunk(Cast<HloCopyStartInstruction>(instr));
    case HloOpcode::kCopyDone:
      return EmitCopyDoneThunk(instr);

    // HLO module is already scheduled, so instructions for ordering are noops.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    // We don't need to emit thunks for these operations because their semantics
    // are encoded by buffers.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return absl::OkStatus();
    default:
      return Internal("Unsupported instruction opcode: %s",
                      HloOpcodeString(instr->opcode()));
  }

  return Internal("Unhandled HLO instruction");
}

absl::Status IrEmitterUnnested::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation))
    return Internal("Sequence not found for computation: %s",
                    computation->name());

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_RETURN_IF_ERROR(EmitHloInstruction(instr));
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
