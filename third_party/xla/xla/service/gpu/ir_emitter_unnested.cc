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
#include <cassert>
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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/transforms/gpu_passes.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#ifdef GOOGLE_CUDA
#include "xla/stream_executor/cuda/cuda_solver_context.h"
#endif  // GOOGLE_CUDA
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/runtime/topk.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#ifdef TENSORFLOW_USE_ROCM
#include "xla/stream_executor/rocm/rocm_solver_context.h"
#endif  // TENSORFLOW_USE_ROCM
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/cholesky_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/cub_sort_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_target.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/norm_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"
#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/triton_call.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/sort_util.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/human_readable_json.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {

IrEmitterUnnested::IrEmitterUnnested(IrEmitterContext* ir_emitter_context)
    : IrEmitter(ir_emitter_context, /*is_nested=*/false),
      send_recv_events_(std::make_shared<HostSendRecvAsyncEvents>()),
      copy_events_(std::make_shared<CopyThunk::AsyncEvents>()),
      call_graph_(CallGraph::Build(&ir_emitter_context->hlo_module())) {}

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
    std::vector<std::unique_ptr<SequentialThunk>> branch_thunk_sequences) {
  ConditionalThunkConfig config;
  config.branch_index_is_bool =
      instr->operand(0)->shape().element_type() == PRED;
  config.branch_count = instr->branch_count();
  config.branch_thunks = std::move(branch_thunk_sequences);
  return config;
}

absl::Status IrEmitterUnnested::EmitConditional(const HloInstruction* instr) {
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  branch_thunks.reserve(instr->branch_count());

  for (auto comp : instr->branch_computations()) {
    auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
    TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(comp));
    Thunk::ThunkInfo branch_thunk_info =
        Thunk::ThunkInfo::WithProfileAnnotation(instr);
    branch_thunk_info.profile_annotation +=
        absl::StrCat("_branch_", comp->name());
    branch_thunks.push_back(
        ir_emitter->ConsumeThunkSequence(branch_thunk_info));
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
  std::unique_ptr<SequentialThunk> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();

  // Maybe serialize all commands in a sequence by forcing barriers between all
  // recorded commands. This guarantees that we execute all device operations
  // in the exact same order as a thunk sequence.
  CommandBufferCmdExecutor::SynchronizationMode synchronization_mode =
      ir_emitter_context_->debug_options()
              .xla_gpu_graph_enable_concurrent_region()
          ? CommandBufferCmdExecutor::SynchronizationMode::kAutomatic
          : CommandBufferCmdExecutor::SynchronizationMode::kSerialize;

  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor cmd_executor,
      ConvertToCommands(thunk_sequence->thunks(),
                        ConvertToCommandsOptions{synchronization_mode}));

  AddThunkToThunkSequence(std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor), Thunk::ThunkInfo::WithProfileAnnotation(instr),
      std::move(thunk_sequence),
      ir_emitter_context_->debug_options()
          .xla_enable_command_buffers_during_profiling()));

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

  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr),
                      ir_emitter_context_->gpu_compute_capability()));
  auto thunk = std::make_unique<GemmThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(config), a, b,
      c, workspace,
      RequireDeterminism(ir_emitter_context_->hlo_module().config()));
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

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
      GemmConfig::For(static_cast<const HloInstruction*>(instr),
                      ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  BufferAllocation::Slice a_scale, b_scale, c_scale, d_scale, d_amax;
  TF_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                      gpublas_lt::AsBlasLtEpilogue(epilogue));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      instr, std::move(gemm_config), blas_lt_epilogue, algorithm, a, b, c, d,
      bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax, workspace_buffer);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCublasLtMatmulThunkF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() > 3 && instr->operand_count() < 8);
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

  bool is_cuda = std::holds_alternative<stream_executor::CudaComputeCapability>(
      ir_emitter_context_->gpu_compute_capability());
  bool is_fp8 = instr->shape().tuple_shapes(0).element_type() == F8E4M3FN ||
                instr->shape().tuple_shapes(0).element_type() == F8E5M2;
  // cublasLT requires c_scale/d_scale to be null when C/D is not FP8.
  // Currently, C cannot be FP8.
  BufferAllocation::Slice c_scale, d_scale;
  if (is_cuda && is_fp8) {
    TF_ASSIGN_OR_RETURN(d_scale,
                        GetAllocationSliceForHlo(instr->operands().back()));
  }

  BufferAllocation::Slice bias;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(
        bias, GetAllocationSliceForHlo(instr->operand(a_scale_index + 2)));
  }

  BufferAllocation::Slice d_amax;
  if (config.damax_output()) {
    TF_ASSIGN_OR_RETURN(d_amax, GetAllocationSliceForHlo(instr, {1}));
  }

  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr),
                      ir_emitter_context_->gpu_compute_capability()));

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
      instr, std::move(gemm_config), blas_lt_epilogue, algorithm, a, b, c, d,
      bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax, workspace_buffer);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitConvolutionReorderThunk(
    const HloCustomCallInstruction* instr) {
  bool has_bias = instr->operand_count() > 1;
  Shape shape = has_bias ? instr->shape().tuple_shapes(0) : instr->shape();
  if (shape.dimensions().size() != 5 || shape.dimensions(4) != 32) {
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

absl::Status IrEmitterUnnested::EmitCuDnnThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context_->buffer_assignment(), instr,
                              instr->operands()));
  TF_ASSIGN_OR_RETURN(const std::string fingerprint,
                      FingerprintWithBackendConfig<GpuBackendConfig>(*instr));
  // check if sdpa dropout is enabled
  std::optional<int64_t> dropout_seed = std::nullopt;
  if (MHACallHasDropout(instr->custom_call_target())) {
    TF_ASSIGN_OR_RETURN(const auto gpu_config,
                        instr->backend_config<xla::gpu::GpuBackendConfig>());
    dropout_seed = gpu_config.cudnn_fmha_backend_config().seed();
  }
  AddThunkToThunkSequence(std::make_unique<CuDnnThunk>(
      fingerprint, Thunk::ThunkInfo::WithProfileAnnotation(instr),
      kernel_arguments.args(), dropout_seed));
  return absl::OkStatus();
}

absl::StatusOr<BufferAllocation::Slice>
IrEmitterUnnested::GetAllocationSliceForHlo(const HloInstruction* instr,
                                            const ShapeIndex& index) const {
  return xla::gpu::GetAllocationSlice(ir_emitter_context_->buffer_assignment(),
                                      instr, index);
}

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
          operand_shape.dimensions(operand_shape.dimensions().size() - 1),
      ir_emitter_context_->platform_name());
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCholeskyThunk(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(CholeskyOptions options,
                      instr->backend_config<CholeskyOptions>());
  const Shape& shape = instr->operand(0)->shape();
  int ndim = shape.dimensions().size();
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

#if GOOGLE_CUDA
  auto solver_creator = stream_executor::CudaSolverContext::Create;
#else
  auto solver_creator = stream_executor::RocmSolverContext::Create;
#endif

  thunks.push_back(std::make_unique<CholeskyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), options, a_buffer,
      workspace_buffer, info_buffer, shape.element_type(), batch_size, n,
      solver_creator));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), std::move(thunks)));
  }

  return absl::OkStatus();
}

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

  // For XLA FFI handlers we decode opaque backend config into attributes map
  // at IR emission time, so that we do not need to parse MLIR at run time. For
  // FFI handlers backend config must be a compatible MLIR dictionary.
  CustomCallThunk::AttributesMap attributes;

  // For information about this calling convention, see
  // xla/g3doc/custom_call.md.
  switch (instr->api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
      custom_call_target = [call_target](stream_executor::Stream* stream,
                                         void** buffers, const char* opaque,
                                         size_t opaque_len,
                                         XlaCustomCallStatus*) {
        reinterpret_cast<CustomCallWithOpaqueStreamHandle>(call_target)(
            stream->platform_specific_handle().stream, buffers, opaque,
            opaque_len);
      };
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      custom_call_target = [call_target](stream_executor::Stream* stream,
                                         void** buffers, const char* opaque,
                                         size_t opaque_len,
                                         XlaCustomCallStatus* status) {
        reinterpret_cast<CustomCallWithStatusAndOpaqueStreamHandle>(
            call_target)(stream->platform_specific_handle().stream, buffers,
                         opaque, opaque_len, status);
      };
      break;
    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      // We already checked `handler` above.
      break;
    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      instr->api_version());
  }

  auto backend_config = instr->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    VLOG(3) << "Unable to parse backend config for custom call: "
            << backend_config.status().message() << "\n"
            << "Fall back to parse the raw backend config str.";
  }

  auto ffi_thunk = [&]() -> absl::StatusOr<std::unique_ptr<CustomCallThunk>> {
    auto& called_computations = instr->called_computations();
    auto& backend_config_str =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().attributes()
            : instr->raw_backend_config_string();
    if (!backend_config_str.empty()) {
      mlir::Attribute attr = mlir::parseAttribute(
          backend_config_str, ir_emitter_context_->mlir_context());
      auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
      if (dict == nullptr) {
        return absl::InternalError(
            "Unsupported backend config. Expected a string parsable into "
            "dictionary attribute");
      }
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    }
    return CustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), call_target_name,
        registration->bundle, std::move(operands), std::move(results),
        std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0]);
  };

  auto legacy_thunk =
      [&]() -> absl::StatusOr<std::unique_ptr<CustomCallThunk>> {
    std::string opaque =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().opaque()
            : instr->raw_backend_config_string();
    return CustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), call_target_name,
        std::move(custom_call_target), std::move(operands), std::move(results),
        std::move(opaque));
  };

  TF_ASSIGN_OR_RETURN(std::unique_ptr<CustomCallThunk> custom_call_thunk,
                      found_ffi_handler ? ffi_thunk() : legacy_thunk());
  AddThunkToThunkSequence(std::move(custom_call_thunk));

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

  int64_t m = b_shape.dimensions(b_shape.dimensions().size() - 2);
  int64_t n = b_shape.dimensions(b_shape.dimensions().size() - 1);
  int64_t batch_size = std::accumulate(
      b_shape.dimensions().begin(), b_shape.dimensions().end() - 2, int64_t{1},
      [](int64_t a, int64_t b) { return a * b; });
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_ty);
  int64_t a_batch_stride =
      backend_config.left_side() ? m * m * elem_size : n * n * elem_size;
  int64_t b_batch_stride = m * n * elem_size;
  thunks.push_back(std::make_unique<TriangularSolveThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), backend_config,
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

  TF_RET_CHECK(data_shape.dimensions().size() <= 2) << "Invalid input shape.";
  TF_RET_CHECK(indices_shape.element_type() == PrimitiveType::S32)
      << "Indices should be S32.";

  bool has_batch = data_shape.dimensions().size() == 2;
  auto [batch_size, n, k] =
      has_batch
          ? std::tuple<size_t, size_t, size_t>{data_shape.dimensions(0),
                                               data_shape.dimensions(1),
                                               top_elements_shape.dimensions(1)}
          : std::tuple<size_t, size_t, size_t>{
                1, data_shape.dimensions(0), top_elements_shape.dimensions(0)};

  auto wavefront_size =
      ir_emitter_context_->gpu_device_info().threads_per_warp();

  // Load TopK custom kernel.
  TF_ASSIGN_OR_RETURN(
      CustomKernel kernel,
      kernel::topk::GetTopKKernel("topk", data_shape.element_type(), n, k,
                                  batch_size, platform_name(), wavefront_size));

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
  auto generate = [this, &instr]() -> absl::StatusOr<KernelReuseCache::Entry> {
    mlir::MLIRContext& mlir_context = *ir_emitter_context_->mlir_context();
    LoadMlirDialectsForTriton(mlir_context);
    auto call =
        TritonCall::Parse(instr->raw_backend_config_string(), &mlir_context);
    auto kernel_name =
        ir_emitter_context_->name_uniquer()->GetUniqueName(call.name);
    VLOG(3) << "Generating: " << kernel_name;

    mlir::OwningOpRef<mlir::ModuleOp> triton_module;
    {
      mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context);
      triton_module =
          mlir::parseSourceString<mlir::ModuleOp>(call.ir, &mlir_context);
      if (!triton_module) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to parse Triton module: ",
                         diagnostic_handler.ConsumeStatus().message(),
                         "\ninput ir: ", call.ir));
      }
    }

    auto triton_fn =
        triton_module->lookupSymbol<mlir::triton::FuncOp>(call.name);
    TF_RET_CHECK(triton_fn)
        << "Call name not found in the Triton module: " << call.name;
    triton_fn.setName(kernel_name);
    size_t arg_size = triton_fn.getNumArguments();

    HloModule* hlo_module = instr->GetModule();
    // If emit_kernels if false (i.e., when deserializing an already compiled
    // executable), we do not emit code, but we still need to run part of the
    // compiler to figure out the size of the shared memory and the cluster
    // dimensions for the thunk. We also must call the name uniqifier as if
    // emitting code so that the future generated names remain in sync.
    bool emit_kernels = ir_emitter_context_->emit_kernels();

    BlockLevelParameters block_level_parameters;
    block_level_parameters.num_stages = call.num_stages;
    block_level_parameters.num_warps = call.num_warps;
    block_level_parameters.num_ctas = 1;

    TF_ASSIGN_OR_RETURN(
        auto result,
        CompileTritonToLLVM(kernel_name, *hlo_module,
                            ir_emitter_context_->gpu_device_info(),
                            block_level_parameters, triton_module.get(),
                            ir_emitter_context_->llvm_module(), mlir_context,
                            /*is_xla_fusion=*/false, emit_kernels));

    TF_ASSIGN_OR_RETURN(
        auto kernel_arguments,
        KernelArguments::Create(ir_emitter_context_->buffer_assignment(), instr,
                                instr->operands(),
                                /*dedup=*/false));
    auto launch_dimensions =
        LaunchDimensions(se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
                         se::ThreadDim(call.num_warps * 32));

    std::string sanitized_kernel_name =
        GetSanitizedUniqueName(*ir_emitter_context_, kernel_name);

    if (emit_kernels) {
      llvm::Function* impl_fn =
          ir_emitter_context_->llvm_module()->getFunction(kernel_name);
      TF_RET_CHECK(impl_fn);
      impl_fn->setName(ir_emitter_context_->name_uniquer()->GetUniqueName(
          kernel_name + "_impl"));

      llvm::IRBuilder builder(ir_emitter_context_->llvm_module()->getContext());

      llvm::Function* kernel;
      std::vector<llvm_ir::IrArray> inputs;
      std::vector<llvm_ir::IrArray> outputs;
      TF_ASSIGN_OR_RETURN(std::tie(kernel, inputs, outputs),
                          BuildKernelPrototypeFromUniqueName(
                              *ir_emitter_context_, impl_fn->getName().str(),
                              sanitized_kernel_name, kernel_arguments.args(),
                              arg_size, launch_dimensions, &builder));

      // Move function body into kernel prototype.
      llvm::Function* prototype_func = builder.GetInsertBlock()->getParent();
      prototype_func->splice(prototype_func->begin(), impl_fn);
      for (const auto& [arg, input] : llvm::zip(impl_fn->args(), inputs)) {
        arg.replaceAllUsesWith(input.GetBasePointer());
      }
      // Triton's kernel ABI expects an additional scratchpad global memory.
      // For now it is only used for on-device creation of TMA descriptors,
      // which we do not use yet, so we are just replacing this argument with a
      // null pointer.
      // TODO: b/381242007 - Allocate a proper buffer if we want to use
      // device-side TMA APIs.
      auto scratchpad_arg = impl_fn->getArg(impl_fn->arg_size() - 1);
      scratchpad_arg->replaceAllUsesWith(llvm::ConstantPointerNull::get(
          llvm::cast<llvm::PointerType>(scratchpad_arg->getType())));

      impl_fn->eraseFromParent();

      for (auto& arg : prototype_func->args()) {
        // Remove the alignment and aliasing attributes to avoid recompiling the
        // kernel for each alignment/aliasing combination.
        arg.removeAttr(llvm::Attribute::Alignment);
        arg.removeAttr(llvm::Attribute::NoAlias);
      }
    }

    return {{sanitized_kernel_name, launch_dimensions, result.cluster_dim,
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
}

absl::Status IrEmitterUnnested::EmitAsyncComputation(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(auto stream,
                      stream_assignment.GetSyncExecutionStreamId(wrapped));
  TF_RET_CHECK(wrapped->called_computations().size() == 1);
  auto computation = wrapped->called_computations().front();
  auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(computation));
  std::unique_ptr<SequentialThunk> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();
  for (auto& thunk : thunk_sequence->thunks()) {
    thunk->set_execution_stream_id(stream);
  }
  auto* async_start = Cast<HloAsyncInstruction>(instr);
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds async_streams,
      stream_assignment.GetAsyncExecutionStreamIds(async_start));
  // We launch the thunk sequence computation on a concurrent stream.
  // The concurrent stream needs to first wait until the main stream has
  // finished calculating any values that may be used as input.
  // We enforce this by inlining a `WaitForStreams` thunk on the main
  // stream.
  AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      async_streams.destination_stream_id, async_streams.source_stream_id));
  AddThunkToThunkSequence(std::move(thunk_sequence));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitFusion(const HloFusionInstruction* instr) {
  const se::DeviceDescription& device_info =
      ir_emitter_context_->gpu_device_info();
  const HloFusionAnalysis fusion_analysis =
      HloFusionAnalysis::Create(*instr, device_info);
  VLOG(3) << "IrEmitterUnnested::EmitFusion:start";
  std::unique_ptr<FusionInterface> emitter = GetFusionEmitter(
      /*fusion_info=*/HloFusionInfo(
          /*analysis=*/fusion_analysis, instr,
          /*buffer_assignment=*/&ir_emitter_context_->buffer_assignment(),
          /*call_graph=*/*call_graph_));
  TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  for (std::unique_ptr<Thunk>& thunk : result.thunks) {
    TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                        stream_assignment.GetSyncExecutionStreamId(instr));
    thunk->set_execution_stream_id(execution_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  }
  VLOG(3) << "IrEmitterUnnested::EmitFusion:complete";
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCopy(const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(instr));
  AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      /*source_buffer=*/src_buffer,
      /*destination_buffer=*/dst_buffer,
      /*mem_size=*/src_buffer.size()));
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
  if (IsCublasLtMatmul(*wrapped)) {
    auto status = EmitCublasLtMatmulThunk(custom_call);
    if (status.ok()) {
      thunk_sequence_.back()->set_execution_stream_id(execution_stream_id);
    }
    return status;
  }
  if (IsCublasLtMatmulF8(*wrapped)) {
    auto status = EmitCublasLtMatmulThunkF8(custom_call);
    if (status.ok()) {
      thunk_sequence_.back()->set_execution_stream_id(execution_stream_id);
    }
    return status;
  }
  return Internal("Unsupported async custom call instruction: %s",
                  HloOpcodeString(wrapped->opcode()));
}

absl::Status IrEmitterUnnested::AssertNonDeterminismIsOkay(
    const std::string& op_name) {
  if (RequireDeterminism(ir_emitter_context_->hlo_module().config())) {
    return Unimplemented(
        "HLO instruction %s does not have a deterministic implementation, "
        "but run-to-run determinism is required.",
        op_name);
  }
  return absl::OkStatus();
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
    TF_RET_CHECK(
        LayoutUtil::LayoutsInShapesEqual(keys_shape, sort->operand(i)->shape(),
                                         Layout::Equal().IgnoreMemorySpace()));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace()));

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
  // First output is aliased.
  TF_RET_CHECK(
      instr->shape().IsTuple() && instr->shape().tuple_shapes_size() == 2 &&
      Shape::Equal().IgnoreMemorySpaceInLayout()(
          instr->shape().tuple_shapes(0), instr->shape().tuple_shapes(1)));

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();

  auto operands = instr->operands();
  std::vector<CollectiveThunk::Buffer> buffers;
  for (int oprd_idx = 0; oprd_idx < operands.size(); ++oprd_idx) {
    const auto operand = operands.at(oprd_idx);
    const ShapeIndex nested_shape_idx = {1, oprd_idx}, normal_shape_idx = {1};
    const Shape operand_shape = operand->shape();
    const Shape result_shape = instr->shape().tuple_shapes(1);
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                        GetAllocationSliceForHlo(
                            instr, result_shape.IsTuple() ? nested_shape_idx
                                                          : normal_shape_idx));

    const int64_t src_memory_space = operand_shape.layout().memory_space();
    const int64_t dst_memory_space =
        (result_shape.IsTuple())
            ? result_shape.tuple_shapes(0).layout().memory_space()
            : result_shape.layout().memory_space();

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice source_slice,
                        GetAllocationSliceForHlo(operand));
    if (CollectivePermuteStartThunk::IsDegenerate(instr, replica_count,
                                                  partition_count)) {
      // For a degenerate collective permute, just generate a copy thunk.
      AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(instr),
          /*source_buffer=*/source_slice,
          /*destination_buffer=*/result_slice,
          /*mem_size=*/ShapeUtil::ByteSizeOf(operand_shape)));
      // Signal that start thunk not created with nullptr.
      GetCollectivesAsyncEvents().try_emplace(instr, nullptr);
    } else {
      const CollectiveThunk::Buffer buffer = {
          /*element_count=*/ShapeUtil::ElementsIn(operand_shape),
          /*source_buffer=*/source_slice,
          /*destination_buffer=*/result_slice,
          /*source_memory_space=*/src_memory_space,
          /*destination_memory_space=*/dst_memory_space};
      buffers.push_back(buffer);
    }
  }
  if (!CollectivePermuteStartThunk::IsDegenerate(instr, replica_count,
                                                 partition_count)) {
    auto thunk = std::make_unique<CollectivePermuteStartThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), instr, replica_count,
        partition_count, buffers,
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p(),
        GetStreamKindForP2P(instr));
    GetCollectivesAsyncEvents().try_emplace(instr, thunk->async_events());
    AddThunkToThunkSequence(std::move(thunk));
  }
  return absl::OkStatus();
}

template <typename CollectiveThunkType, typename HloInstType>
absl::Status IrEmitterUnnested::EmitCollectiveThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloInstType* inst, std::optional<bool> use_global_device_ids) {
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  VLOG(2) << CollectiveThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << inst->operand_count();

  // A given collective op can be degenerate if across all groups formed
  // by it are singleton. In such a case, we don't need to do any communication
  // and we can just copy the input to the output.
  //
  // The only exception is RaggedAllToAll, which is not degenerate even if
  // all groups are singleton. In a singleton group case, RaggedAllToAll becomes
  // a generic equivalent of DynamicUpdateSlice, except update size is not
  // statically known. This operation can not be expressed in term of standard
  // HLO instructions, so the best solution we have is to use NCCL thunk even
  // for degenerate cases.
  bool is_degenerate = kind != Thunk::Kind::kRaggedAllToAll &&
                       GetCollectiveConfig(inst, use_global_device_ids)
                           .IsDegenerate(replica_count, partition_count);
  absl::Status implementable_status = CollectiveThunkType::CheckImplementable(
      inst, replica_count, partition_count);
  bool should_use_nccl_thunk = !is_degenerate && implementable_status.ok();

  // Stash relevant information in CollectiveThunk::Buffer even if we may
  // not generate an CollectiveThunk.
  std::vector<CollectiveThunk::Buffer> buffers;

  int64_t operand_count = inst->operand_count();
  buffers.reserve(operand_count);

  // Adds a source and destination buffers pair to `buffers`.
  auto add_buffer = [&](int64_t element_count, BufferAllocation::Slice src,
                        int64_t src_memory_space, BufferAllocation::Slice dst,
                        int64_t dst_memory_space) {
    buffers.push_back(
        CollectiveThunk::Buffer{/*element_count=*/element_count,
                                /*source_buffer=*/src,
                                /*destination_buffer=*/dst,
                                /*source_memory_space=*/src_memory_space,
                                /*destination_memory_space=*/dst_memory_space,
                                /*source_value=*/nullptr,
                                /*destination_value=*/nullptr});
  };

  if (kind == Thunk::Kind::kAllGatherStart) {
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
  } else if (kind == Thunk::Kind::kRaggedAllToAll) {
    // RaggedAllToAll operation has 6 operands: input, output, input_offset,
    // send_size, output_offset, recv_size.
    // `output` operand is aliased with the instruction result. All other
    // operands are not aliased.
    const Shape& input_shape = inst->operand(0)->shape();
    TF_ASSIGN_OR_RETURN(auto input_buffer,
                        GetAllocationSliceForHlo(inst->operand(0)));
    add_buffer(ShapeUtil::ElementsIn(input_shape), input_buffer,
               input_shape.layout().memory_space(), input_buffer,
               input_shape.layout().memory_space());

    const Shape& output_shape = inst->operand(1)->shape();
    const Shape& result_shape = inst->shape();
    TF_ASSIGN_OR_RETURN(auto output_buffer,
                        GetAllocationSliceForHlo(inst->operand(1)));
    TF_ASSIGN_OR_RETURN(auto result_buffer, GetAllocationSliceForHlo(inst));

    add_buffer(ShapeUtil::ElementsIn(result_shape), output_buffer,
               output_shape.layout().memory_space(), result_buffer,
               result_shape.layout().memory_space());

    for (int64_t i = 2; i < operand_count; i++) {
      const Shape& shape = inst->operand(i)->shape();
      TF_ASSIGN_OR_RETURN(auto slice,
                          GetAllocationSliceForHlo(inst->operand(i)));
      add_buffer(ShapeUtil::ElementsIn(shape), slice,
                 shape.layout().memory_space(), slice,
                 shape.layout().memory_space());
    }
  } else {
    // For other operations simply zip operands with results.
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx =
          inst->shape().IsTuple() ? ShapeIndex({i}) : ShapeIndex({});
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
    auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(inst);
    // The wrapper name is used when syntactic sugar is turned on.
    if (ir_emitter_context_->debug_options().xla_syntax_sugar_async_ops()) {
      thunk_info.profile_annotation = async_start->name();
    }
    auto thunk = std::make_unique<CollectiveThunkType>(
        thunk_info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
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

// Find the canonical send/recv start op for one of send, recv, send-done, or
// recv-done. For trivial cases send/recv and send-done/recv-done come in pairs
// and the canonical start op is the send/recv op of the pair. If send/recv is
// partially pipelined, we will use the send/recv leading into the while loop as
// the canonical start op, which will serve as a key for the async events.
//
// Example:
// ```
// send_ctx = send(src, ...)  <-- canonical start op
// send_ctx_final = while(send_ctx) {
//   send_ctx_in = parameter(0)
//   send-done(send_ctx_in)
//   ...
//   ROOT send_ctx_out = send(next_src, ...)
// }
// send-done(send_ctx_final)
// ```
static const HloInstruction* FindCanonicalSendRecvStartOp(
    const HloInstruction* inst) {
  CHECK(inst->opcode() == HloOpcode::kSend ||
        inst->opcode() == HloOpcode::kRecv ||
        inst->opcode() == HloOpcode::kSendDone ||
        inst->opcode() == HloOpcode::kRecvDone);
  // If the instruction is wrapped in an async computation, return the
  // instruction itself.
  if (inst->parent()->IsAsyncComputation()) {
    return inst;
  }

  // Find container while loop and index for the send/recv case or return
  // canonical start op directly.
  const HloInstruction* while_op = nullptr;
  int64_t i = -1;
  if (inst->opcode() == HloOpcode::kSend ||
      inst->opcode() == HloOpcode::kRecv) {
    CHECK_EQ(inst->users().size(), 1);
    const HloInstruction* unique_user = inst->users().front();

    // Return send/recv inst directly if this is a simple send/recv pair.
    if (unique_user->opcode() == HloOpcode::kSendDone ||
        unique_user->opcode() == HloOpcode::kRecvDone) {
      return inst;
    }

    // Find while loop and index, otherwise.
    CHECK(unique_user->opcode() == HloOpcode::kTuple ||
          unique_user->opcode() == HloOpcode::kWhile);
    if (unique_user->IsRoot()) {
      // send/recv op in the loop body.
      auto maybe_while_op =
          unique_user->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while_op);
      while_op = *maybe_while_op;
      i = unique_user->operand_index(inst);
    } else {
      // send/recv leading into the loop.
      CHECK_EQ(unique_user->users().size(), 1);
      CHECK(unique_user->users().front()->opcode() == HloOpcode::kWhile);
      while_op = unique_user->users().front();
      i = unique_user->operand_index(inst);
    }
  }

  // Find container while loop and index for the send-done/recv-done case or
  // return canonical start op directly.
  if (inst->opcode() == HloOpcode::kSendDone ||
      inst->opcode() == HloOpcode::kRecvDone) {
    const HloInstruction* operand = inst->operand(0);

    // Return send/recv inst directly if this is a simple send/recv pair.
    if (operand->opcode() == HloOpcode::kSend ||
        operand->opcode() == HloOpcode::kRecv) {
      return operand;
    }

    // Find while loop and index, otherwise.
    CHECK(operand->opcode() == HloOpcode::kGetTupleElement);
    const auto* gte = Cast<HloGetTupleElementInstruction>(operand);
    const HloInstruction* iter_tuple = operand->operand(0);
    if (iter_tuple->opcode() == HloOpcode::kParameter) {
      // send-done/recv-done in the loop body.
      CHECK(Cast<HloParameterInstruction>(iter_tuple)->parameter_number() == 0);
      auto maybe_while =
          iter_tuple->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while);
      while_op = *maybe_while;
      i = gte->tuple_index();
    } else {
      // send-done/recv-done proceeding the loop.
      CHECK(iter_tuple->opcode() == HloOpcode::kWhile);
      while_op = iter_tuple;
      i = gte->tuple_index();
    }
  }

  // Extract canonical start op from while loop's init.
  CHECK(while_op != nullptr);
  CHECK(0 <= i && i < while_op->shape().tuple_shapes_size());
  const HloInstruction* init = while_op->operand(0);
  const HloInstruction* canonical_start_op = init->operand(i);
  CHECK(canonical_start_op->opcode() == HloOpcode::kSend ||
        canonical_start_op->opcode() == HloOpcode::kRecv);
  return canonical_start_op;
}

absl::Status IrEmitterUnnested::EmitCollectiveGroupStartThunk(
    const HloInstruction* instr) {
  emit_group_thunks_ = true;
  std::optional<AsyncStreamKind> stream_kind;
  for (const HloInstruction* nested_instruction :
       instr->async_wrapped_computation()->instructions()) {
    TF_RETURN_IF_ERROR(EmitHloInstruction(nested_instruction));
    if ((nested_instruction->opcode() == HloOpcode::kSend ||
         nested_instruction->opcode() == HloOpcode::kRecv) &&
        !stream_kind.has_value()) {
      // We only need to modify the stream kind once, since all send/recv
      // instructions in a group should have the same stream kind.
      stream_kind = GetStreamKindForP2P(nested_instruction);
    }
  }
  auto thunk = std::make_unique<CollectiveGroupThunk>(
      instr, Thunk::Kind::kGroupStart, std::move(scoped_thunk_sequence_),
      stream_kind.value_or(AsyncStreamKind::kCollective));
  emit_group_thunks_ = false;

  GetCollectivesAsyncEvents().insert({instr, thunk->async_events()});
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCollectiveAsyncDone(
    Thunk::Kind kind, const HloInstruction* inst) {
  // Partial pipelining is only implemented for send/recv.
  bool is_send_recv =
      kind == Thunk::Kind::kRecvDone || kind == Thunk::Kind::kSendDone;
  const HloInstruction* start =
      is_send_recv ? FindCanonicalSendRecvStartOp(inst) : inst->operand(0);

  // Find canonical async event.
  CollectivesAsyncEvents& collectives_async_events =
      GetCollectivesAsyncEvents();
  auto async_events_it = collectives_async_events.find(start);
  TF_RET_CHECK(async_events_it != collectives_async_events.end())
      << "couldn't find async events for start operation";

  // Can be null if no start thunk was created (e.g. if the start op is
  // degenerate), in which case there's nothing to do here.
  if (!async_events_it->second) return absl::OkStatus();

  AsyncStreamKind stream_kind = AsyncStreamKind::kCollective;
  if (is_send_recv) {
    stream_kind = GetStreamKindForP2P(start);
  }
  AddThunkToThunkSequence(std::make_unique<CollectiveDoneThunk>(
      kind, Thunk::ThunkInfo::WithProfileAnnotation(inst),
      async_events_it->second, stream_kind));
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
      BuildKernelPrototype(*ir_emitter_context_, suggested_kernel_name,
                           suggested_kernel_name, kernel_arguments.args(),
                           kernel_arguments.args().size(), launch_dimensions,
                           &b_));

  AddThunkToThunkSequence(std::make_unique<KernelThunk>(
      hlo, kernel->getName().str(), kernel_arguments.args(), launch_dimensions,
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0));

  return {{inputs, outputs}};
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

  Thunk::ThunkInfo cond_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr);
  cond_thunk_info.profile_annotation += "_condition";
  Thunk::ThunkInfo body_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr);
  body_thunk_info.profile_annotation += "_body";

  return std::unique_ptr<Thunk>(new WhileThunk(
      thunk_info, instr, pred,
      ir_emitter_condition->ConsumeThunkSequence(cond_thunk_info),
      ir_emitter_body->ConsumeThunkSequence(body_thunk_info), trip_count));
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
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
      stream_assignment.GetAsyncExecutionStreamIds(copy_start_instr));
  // Insert a waitFor() thunk for asynchronous memcpy only when the source
  // and destination stream IDs differ. If the IDs are the same, the memcpy
  // operation is synchronous within that stream.
  if (streams.destination_stream_id != streams.source_stream_id) {
    AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        streams.destination_stream_id, streams.source_stream_id));
  }
  if (is_dst_host_memory) {
    auto thunk = std::make_unique<DeviceToHostCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    thunk->set_execution_stream_id(streams.destination_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  } else {
    auto thunk = std::make_unique<HostToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    thunk->set_execution_stream_id(streams.destination_stream_id);
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
    const CollectiveThunk::Buffer nccl_buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(src->shape()),
        /*source_buffer=*/buffer,
        /*destination_buffer=*/buffer,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    auto thunk = std::make_unique<SendThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), instr, replica_count,
        partition_count, nccl_buffer);
    CollectivesAsyncEvents& collectives_async_events =
        GetCollectivesAsyncEvents();

    // Wire up async events if the send thunk isn't emitted as a part of a
    // group thunk.
    if (!emit_group_thunks_) {
      const HloInstruction* canonical_send_instr =
          FindCanonicalSendRecvStartOp(instr);
      if (collectives_async_events.contains(canonical_send_instr)) {
        thunk->set_async_events(collectives_async_events[canonical_send_instr]);
      } else {
        collectives_async_events.try_emplace(instr, thunk->async_events());
      }
    }
    AddThunkToThunkSequence(std::move(thunk));
    return absl::OkStatus();
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send instruction");
  }

  AddThunkToThunkSequence(std::make_unique<HostSendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), src->shape(), buffer,
      *instr->channel_id(), send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSendDoneThunk(
    const HloSendDoneInstruction* instr) {
  if (!instr->is_host_transfer()) {
    return EmitCollectiveAsyncDone(Thunk::kSendDone, instr);
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send done instruction");
  }

  AddThunkToThunkSequence(std::make_unique<HostSendDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), *instr->channel_id(),
      send_recv_events_, DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitRecvThunk(const HloRecvInstruction* instr) {
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

    const CollectiveThunk::Buffer nccl_buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(instr->shape().tuple_shapes(0)),
        /*source_buffer=*/buffer,
        /*destination_buffer=*/buffer,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    auto thunk = std::make_unique<RecvThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(instr), instr, replica_count,
        partition_count, nccl_buffer);
    CollectivesAsyncEvents& collectives_async_events =
        GetCollectivesAsyncEvents();

    // Wire up async events.
    if (!emit_group_thunks_) {
      const HloInstruction* canonical_recv_instr =
          FindCanonicalSendRecvStartOp(instr);
      if (collectives_async_events.contains(canonical_recv_instr)) {
        thunk->set_async_events(collectives_async_events[canonical_recv_instr]);
      } else {
        collectives_async_events.try_emplace(instr, thunk->async_events());
      }
    }
    AddThunkToThunkSequence(std::move(thunk));
    return absl::OkStatus();
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv instruction");
  }

  AddThunkToThunkSequence(std::make_unique<HostRecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      instr->shape().tuple_shapes()[0], buffer, *instr->channel_id(),
      send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitRecvDoneThunk(
    const HloRecvDoneInstruction* instr) {
  if (!instr->is_host_transfer()) {
    return EmitCollectiveAsyncDone(Thunk::kRecvDone, instr);
  }
  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv done instruction");
  }
  AddThunkToThunkSequence(std::make_unique<HostRecvDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), *instr->channel_id(),
      send_recv_events_, DeviceConstraint(instr)));

  return absl::OkStatus();
}

// If the fusion instruction is a dynamic-slice-fusion instruction, with a
// collective hero operation, then this function returns the collective
// operation. Returns std::nullopt otherwise.
std::optional<const HloInstruction*> GetCollectiveHeroForDynamicSliceFusion(
    const HloFusionInstruction* instruction) {
  if (!IsDynamicSliceFusion(instruction)) {
    return std::nullopt;
  }
  return HloBfsFindIf(
      {instruction->fused_instructions_computation()->root_instruction()},
      [](const HloInstruction* instr) { return IsCollective(instr); });
}

absl::Status IrEmitterUnnested::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kAllGatherDone:
      return EmitCollectiveAsyncDone(Thunk::kAllGatherDone, instr);
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(instr);
      return EmitCollectiveThunk<AllGatherStartThunk, HloAllGatherInstruction>(
          Thunk::kAllGatherStart, all_gather, all_gather,
          all_gather->use_global_device_ids());
    }

    case HloOpcode::kAllReduceDone:
      return EmitCollectiveAsyncDone(Thunk::kAllReduceDone, instr);
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(instr);
      return EmitCollectiveThunk<AllReduceStartThunk, HloAllReduceInstruction>(
          Thunk::kAllReduceStart, all_reduce, all_reduce,
          all_reduce->use_global_device_ids());
    }
    case HloOpcode::kAsyncDone: {
      if (!instr->async_wrapped_computation()
               ->CanExpandIntoSingleInstruction()) {
        return EmitCollectiveAsyncDone(Thunk::kGroupDone, instr);
      }
      const HloInstruction* wrapped = instr->async_wrapped_instruction();
      switch (wrapped->opcode()) {
        case HloOpcode::kReduceScatter:
          return EmitCollectiveAsyncDone(Thunk::kReduceScatterDone, instr);
        case HloOpcode::kAllToAll:
          return EmitCollectiveAsyncDone(Thunk::kAllToAllDone, instr);
        case HloOpcode::kRaggedAllToAll:
          return EmitCollectiveAsyncDone(Thunk::kRaggedAllToAllDone, instr);
        case HloOpcode::kCollectiveBroadcast:
          return EmitCollectiveAsyncDone(Thunk::kCollectiveBroadcastDone,
                                         instr);
        case HloOpcode::kFusion: {
          auto collective_hero = GetCollectiveHeroForDynamicSliceFusion(
              Cast<HloFusionInstruction>(wrapped));
          if (collective_hero.has_value()) {
            switch ((*collective_hero)->opcode()) {
              case HloOpcode::kReduceScatter:
                TF_RETURN_IF_ERROR(
                    EmitCollectiveAsyncDone(Thunk::kReduceScatterDone, instr));
                break;
              default:
                return absl::InternalError(absl::StrFormat(
                    "Unhandled collective in dynamic slice fusion "
                    "instruction: %s",
                    (*collective_hero)
                        ->fused_instructions_computation()
                        ->ToString()));
            }
          }
          // We still want to emit the stream done thunk.
          [[clang::fallthrough]];
        }
        case HloOpcode::kCall:
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
      // Multi-op async start will emit a NCCL group thunk.
      if (!instr->async_wrapped_computation()
               ->CanExpandIntoSingleInstruction()) {
        return EmitCollectiveGroupStartThunk(instr);
      }
      const HloInstruction* wrapped = instr->async_wrapped_instruction();
      switch (wrapped->opcode()) {
        case HloOpcode::kReduceScatter: {
          auto* reduce_scatter = Cast<HloReduceScatterInstruction>(wrapped);
          return EmitCollectiveThunk<ReduceScatterStartThunk,
                                     HloReduceScatterInstruction>(
              Thunk::kReduceScatter, instr, reduce_scatter,
              reduce_scatter->use_global_device_ids());
        }
        case HloOpcode::kAllToAll: {
          auto* all_to_all = Cast<HloAllToAllInstruction>(wrapped);
          return EmitCollectiveThunk<AllToAllStartThunk,
                                     HloAllToAllInstruction>(
              Thunk::kAllToAll, instr, all_to_all, std::nullopt);
        }
        case HloOpcode::kRaggedAllToAll: {
          auto* ragged_all_to_all = Cast<HloRaggedAllToAllInstruction>(wrapped);
          return EmitCollectiveThunk<RaggedAllToAllStartThunk,
                                     HloRaggedAllToAllInstruction>(
              Thunk::kRaggedAllToAll, instr, ragged_all_to_all, std::nullopt);
        }
        case HloOpcode::kCollectiveBroadcast: {
          auto* collective_broadcast =
              Cast<HloCollectiveBroadcastInstruction>(wrapped);
          return EmitCollectiveThunk<CollectiveBroadcastStartThunk,
                                     HloCollectiveBroadcastInstruction>(
              Thunk::kCollectiveBroadcast, instr, collective_broadcast,
              std::nullopt);
        }
        case HloOpcode::kFusion: {
          // We'll launch the fusion computation on a concurrent stream. The
          // concurrent stream needs to first wait until the main stream has
          // finished calculating any values that may be used as inputs to the
          // fusion computation. We enforce this by inlining a
          // `WaitForStreams` thunk.
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
        case HloOpcode::kCall: {
          return EmitAsyncComputation(instr);
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
      return EmitCollectiveAsyncDone(Thunk::kCollectivePermuteDone, instr);
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
      if (IsCublasLtMatmul(*instr)) {
        return EmitCublasLtMatmulThunk(custom_call);
      }
      if (IsCublasLtMatmulF8(*instr)) {
        return EmitCublasLtMatmulThunkF8(custom_call);
      }
      if (IsCudnnConvolutionReorder(*instr)) {
        return EmitConvolutionReorderThunk(custom_call);
      }
      if (IsCustomCallToDnnNorm(*instr)) {
        return EmitNormThunk(custom_call);
      }
      if (IsCustomCallTofMHA(*instr) || IsCustomCallTofMHAF8(*instr) ||
          IsCustomCallToBlockScaledDot(*instr)) {
        return EmitCuDnnThunk(custom_call);
      }
      if (IsCustomCallToTopK(*instr)) {
        return EmitTopKCustomCall(custom_call);
      }
      if (IsCustomCallToDnnConvolution(*instr)) {
        return EmitConvolutionThunk(custom_call);
      }
      if (IsCustomCallToCusolver(*instr)) {
        return EmitCholeskyThunk(instr);
      }
      if (IsTriangularSolve(*instr)) {
        return EmitTriangularSolveCustomCall(instr);
      }
      if (IsCubDeviceRadixSort(*instr)) {
        return EmitCubDeviceRadixSort(custom_call);
      }
      if (custom_call->custom_call_target() == "PadToStatic") {
        return EmitPadToStatic(custom_call);
      }
      if (instr->custom_call_target() == "SliceToDynamic") {
        return EmitSliceToDynamic(custom_call);
      }
      if (instr->custom_call_target() == "__gpu$xla.gpu.triton") {
        // TODO(slebedev): Remove this after June 15th 2025.
        return EmitTritonCustomCall(custom_call);
      }
      if (instr->custom_call_target() == kNopCustomCallTarget) {
        return absl::OkStatus();
      }
      return EmitCustomCallThunk(custom_call);
    }
    case HloOpcode::kFusion:
      return EmitFusion(Cast<HloFusionInstruction>(instr));
    case HloOpcode::kCopy:
      return EmitCopy(instr);
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
