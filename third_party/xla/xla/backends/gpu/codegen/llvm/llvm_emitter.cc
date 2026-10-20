/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/llvm/parallel_loop_emitter.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

std::vector<llvm_ir::IrArray> IrArraysFor(
    llvm::Function* kernel, const emitters::KernelArguments& kernel_arguments) {
  std::vector<llvm_ir::IrArray> ir_arrays;
  ir_arrays.reserve(kernel_arguments.args().size());
  for (const auto& [kernel_argument, llvm_arg] :
       llvm::zip(kernel_arguments.args(), kernel->args())) {
    llvm::Type* ir_type =
        llvm_ir::ShapeToIrType(kernel_argument.shape(), llvm_arg.getContext());
    llvm_ir::IrArray ir_array(&llvm_arg, ir_type, kernel_argument.shape());

    if (!kernel_argument.written()) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }
    ir_arrays.push_back(ir_array);
  }
  return ir_arrays;
}

llvm::Value* CreateLoad(llvm::Value* address, llvm::Type* data_type,
                        int alignment_bytes, llvm::IRBuilderBase* b) {
  int data_bytes = data_type->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  if (alignment_bytes == 0) {
    return b->CreateLoad(data_type, address);
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  llvm::Value* output = llvm::ConstantInt::get(data_type, 0);
  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b->CreateConstInBoundsGEP1_32(
        b->getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* partial_value = b->CreateLoad(b->getIntNTy(alignment_bitwidth),
                                               offset_address, "partial_value");
    llvm::Value* zextd =
        b->CreateZExt(partial_value, output->getType(), "partial_value_zextd");
    llvm::Value* shifted = b->CreateShl(
        zextd, llvm::ConstantInt::get(b->getInt32Ty(), offset_bytes * 8),
        "partial_input_shifted");
    output = b->CreateAdd(output, shifted, "output_updated");
  }
  return output;
}

void CreateStore(llvm::Value* data, llvm::Value* address, int alignment_bytes,
                 llvm::IRBuilderBase* b) {
  int data_bytes = data->getType()->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  CHECK_GE(data_bytes, alignment_bytes);
  if (alignment_bytes == 0) {
    b->CreateStore(data, address);
    return;
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b->CreateConstInBoundsGEP1_32(
        b->getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* shifted_partial = b->CreateTrunc(
        b->CreateLShr(
            data, llvm::ConstantInt::get(b->getInt32Ty(), offset_bytes * 8)),
        b->getIntNTy(alignment_bitwidth), "truncated_value");
    b->CreateStore(shifted_partial, offset_address);
  }
}

}  // namespace

void AppendGlobalConstant(llvm::Module* module, int64_t num_elements,
                          int64_t bytes_per_element,
                          absl::string_view symbol_name, int allocation_idx,
                          DenseDataIntermediate content,
                          bool emit_initializer) {
  llvm::IRBuilder<> b(module->getContext());
  // Ptxas has issues if the constant allocation is smaller than 64 bytes.
  // TODO(b/253259975): Remove when fixed ptxas version is submitted.
  constexpr int64_t kMinConstAllocationInBytes = 64;
  bool needs_padding =
      num_elements * bytes_per_element < kMinConstAllocationInBytes;

  llvm::ArrayType* global_type = llvm::ArrayType::get(
      b.getInt8Ty(),
      std::max(num_elements * bytes_per_element, kMinConstAllocationInBytes));

  llvm::Constant* initializer = [&]() -> llvm::Constant* {
    if (!emit_initializer) {
      // Content is filled by GpuExecutable during initialization.
      return llvm::ConstantAggregateZero::get(global_type);
    }

    std::vector<uint8_t> padded(kMinConstAllocationInBytes, 0);
    absl::c_copy(content.span(), padded.begin());
    return llvm::ConstantDataArray::get<uint8_t>(
        module->getContext(),
        needs_padding ? llvm::ArrayRef<uint8_t>(padded)
                      : llvm::ArrayRef<uint8_t>(content.span().data(),
                                                content.span().size()));
  }();

  // Explicitly set global addrspace for SPIR-V backend.
  int addrspace =
      llvm::Triple(module->getTargetTriple()).isSPIROrSPIRV() ? 1 : 0;
  // These globals will be looked up by name by GpuExecutable so we need to
  // give them an external linkage.  Not all of their uses are visible in
  // the LLVM IR so we can't give then a linkage that merely preserves their
  // names (like available_externally), we also need to ensure that they stick
  // around even if they're "unused".
  //
  // We may have to be more clever here in the future if we notice that we're
  // keeping around too many globals because of their linkage.
  auto* global_for_const =
      new llvm::GlobalVariable(global_type, /*isConstant=*/emit_initializer,
                               llvm::GlobalValue::ExternalLinkage,
                               /*Initializer=*/initializer, symbol_name,
                               /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
                               /*AddressSpace=*/addrspace,
                               /*isExternallyInitialized=*/false);
  global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
  module->insertGlobalVariable(global_for_const);
}

// Input = {dynamic array(with dynamic dimension meta data at the
// end)} Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<KernelDefinition<LlvmKernelSource>> EmitPadToStaticLLVMIR(
    const HloCustomCallInstruction* hlo, IrEmitterContext* ir_emitter_context,
    const emitters::KernelArguments& kernel_arguments) {
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  std::string op_name = std::string(hlo->name());

  std::unique_ptr<llvm::Module> llvm_module =
      ir_emitter_context->CreateLLVMModule(op_name, *llvm_context);
  llvm::IRBuilder<> b(*llvm_context);

  constexpr int kUnrollFactor = 1;
  const Shape& input_shape = hlo->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context->gpu_device_info(), kUnrollFactor);

  std::string sanitized_kernel_name =
      ir_emitter_context->GetSanitizedUniqueName(op_name);

  ABSL_ASSIGN_OR_RETURN(
      llvm::Function * kernel,
      BuildKernelPrototype(
          llvm_module.get(), ir_emitter_context->gpu_device_info(), op_name,
          sanitized_kernel_name, kernel_arguments, launch_dimensions, &b));

  std::vector<llvm_ir::IrArray> ir_arrays =
      IrArraysFor(kernel, kernel_arguments);

  const llvm_ir::IrArray& source_array = ir_arrays[0];
  const llvm_ir::IrArray& output_array = ir_arrays[1];
  auto output_dim_arrays =
      absl::Span<const llvm_ir::IrArray>(ir_arrays).subspan(2);

  llvm::Type* index_ty =
      GetIndexTypeForKernel(hlo, launch_dimensions.launch_bound(), &b);

  // pseudo code for PadToStatic on a 2d array
  //   int* source_array = args[0];
  //   int* dest_array = args[1];
  llvm::Value* source_buffer = source_array.GetBasePointer();

  // TODO(jurahul): input_shape here is the static shape of the
  // input (which has a dynamic shape in XLA). Currently, we are
  // mapping that to a static shaped memref. When we change that to
  // a more appropriate representation in MLIR, fix this code to
  // correctly deduce the static shape backing the dynamically
  // shaped memref.
  int64_t raw_data_size = ShapeUtil::ByteSizeOf(input_shape);

  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset +
  //   sizeof(int);
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  std::vector<ShapeUtil::IndexedShape> output_shapes =
      ShapeUtil::GetLeafShapes(hlo->shape());

  for (int64_t i = 1; i < output_shapes.size(); ++i) {
    // Dynamic size of each dimension is attached at the end of the
    // source array(operand(0)). We need to extract these value.
    const Shape& dim_shape = output_shapes[i].shape;
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));

    const int64_t dim_index = i - 1;
    llvm::Value* metadata = b.CreateConstInBoundsGEP1_32(
        b.getInt8Ty(), source_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    llvm::Value* dyn_dim_size =
        CreateLoad(metadata, b.getInt32Ty(), alignment, &b);
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }
  KernelSupportLibrary{&b}.If("is_thread_0", IsBlock0Thread0(&b), [&] {
    for (int64_t i = 1; i < output_shapes.size(); ++i) {
      const int64_t dim_index = i - 1;
      llvm::Value* dest_dim_size_address =
          output_dim_arrays[dim_index].GetBasePointer();
      // output[i] stores dynamic_dim_(i-1)
      CreateStore(dynamic_dims[dim_index], dest_dim_size_address, alignment,
                  &b);
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= *dyn_dim0_size;
  //     dyn_element_total *= *dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total =
        b.CreateMul(dyn_element_total,
                    b.CreateIntCast(dynamic_dim, dyn_element_total->getType(),
                                    /*isSigned=*/true),
                    /*Name=*/"dyn_element_total_pad");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size,
  //         static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size,
  //           *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), &b);
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        b.CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(op_name, "in_dyn_bounds"), &b, false);
    // Set IR builder insertion point to the body of the if
    // structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block, &b);
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims), &b);
    output_array.EmitWriteArrayElement(
        dyn_index,
        source_array.EmitReadArrayElement(array_index, &b, /*name=*/""), &b,
        /*use_linear_index=*/false);
    return absl::OkStatus();
  };

  const Shape& data_shape = hlo->shape().tuple_shapes(0);
  ABSL_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                      launch_dimensions, &b, kUnrollFactor)
                      .EmitLoop(op_name, index_ty));

  ABSL_ASSIGN_OR_RETURN(
      KernelSpec kernel_spec,
      emitters::GetKernelSpec(sanitized_kernel_name, *hlo,
                              &ir_emitter_context->buffer_assignment(),
                              launch_dimensions.AsWorkDimensions()));
  return KernelDefinition<LlvmKernelSource>(
      std::move(kernel_spec),
      LlvmKernelSource{std::move(llvm_context), std::move(llvm_module)});
}

// Input = {dynamic array(with dynamic dimension meta data at the
// end)} Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<KernelDefinition<LlvmKernelSource>> EmitSliceToDynamicLLVMIR(
    const HloCustomCallInstruction* hlo, IrEmitterContext* ir_emitter_context,
    const emitters::KernelArguments& kernel_arguments) {
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  std::string op_name = std::string(hlo->name());

  std::unique_ptr<llvm::Module> llvm_module =
      ir_emitter_context->CreateLLVMModule(op_name, *llvm_context);
  llvm::IRBuilder<> b(*llvm_context);

  constexpr int kUnrollFactor = 1;
  const Shape& input_shape = hlo->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context->gpu_device_info(), kUnrollFactor);
  llvm::Type* index_ty =
      GetIndexTypeForKernel(hlo, launch_dimensions.launch_bound(), &b);

  std::string sanitized_kernel_name =
      ir_emitter_context->GetSanitizedUniqueName(op_name);

  ABSL_ASSIGN_OR_RETURN(
      llvm::Function * kernel,
      BuildKernelPrototype(
          llvm_module.get(), ir_emitter_context->gpu_device_info(), op_name,
          sanitized_kernel_name, kernel_arguments, launch_dimensions, &b));

  const Shape& data_shape = ShapeUtil::MakeStaticShape(hlo->shape());
  TF_RET_CHECK(data_shape.IsArray());

  // TODO(jurahul): data_shape here is the static shape of the
  // output (which has a dynamic shape in XLA). Currently, we are
  // mapping that to a static shaped memref. When we change that to
  // a more appropriate representation in MLIR, fix this code to
  // correctly deduce the static shape backing the dynamically
  // shaped memref.

  // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset +
  //   sizeof(int);
  int32_t raw_data_size = ShapeUtil::ByteSizeOf(data_shape);

  // pseudo code for sliceToDynamic on a 2d array
  //   int* source_array = args[0];
  //   int* dest_array = args.back();
  std::vector<llvm_ir::IrArray> ir_arrays =
      IrArraysFor(kernel, kernel_arguments);
  const llvm_ir::IrArray& data_array = ir_arrays.back();
  llvm::Value* dest_buffer = data_array.GetBasePointer();

  // Load dynamic dimensions from memory.
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  for (int64_t i = 1; i < hlo->operand_count(); ++i) {
    llvm::Value* source_buffer = ir_arrays[i].GetBasePointer();
    llvm::Type* source_buffer_pointee_type = ir_arrays[i].GetBasePointeeType();
    llvm::LoadInst* dyn_dim_size =
        b.CreateLoad(source_buffer_pointee_type, source_buffer, "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }
  KernelSupportLibrary{&b}.If("is_thread_0", IsBlock0Thread0(&b), [&] {
    for (int64_t i = 1; i < hlo->operand_count(); ++i) {
      const int64_t dim_index = i - 1;
      llvm::Value* metadata = b.CreateConstInBoundsGEP1_32(
          b.getInt8Ty(), dest_buffer,
          raw_data_size + dim_index * sizeof(int32_t));
      // output[i] stores dynamic_dim_(i-1)
      CreateStore(dynamic_dims[dim_index], metadata, alignment, &b);
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= dyn_dim0_size;
  //     dyn_element_total *= dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total =
        b.CreateMul(dyn_element_total,
                    b.CreateIntCast(dynamic_dim, dyn_element_total->getType(),
                                    /*isSigned=*/true),
                    /*Name=*/"dyn_element_total_slice");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size,
  //         static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size,
  //           *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.di] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), &b);
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        b.CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(op_name, "in_dyn_bounds"), &b, false);
    // Set IR builder insertion point to the body of the if
    // structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block, &b);
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims), &b);

    data_array.EmitWriteArrayElement(
        array_index,
        ir_arrays[0].EmitReadArrayElement(dyn_index, &b, /*name=*/"",
                                          /*use_linear_index=*/false),
        &b);
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                      launch_dimensions, &b, kUnrollFactor)
                      .EmitLoop(op_name, index_ty));

  ABSL_ASSIGN_OR_RETURN(
      KernelSpec kernel_spec,
      emitters::GetKernelSpec(sanitized_kernel_name, *hlo,
                              &ir_emitter_context->buffer_assignment(),
                              launch_dimensions.AsWorkDimensions()));
  return KernelDefinition<LlvmKernelSource>(
      std::move(kernel_spec),
      LlvmKernelSource{std::move(llvm_context), std::move(llvm_module)});
}

absl::StatusOr<KernelDefinition<LlvmKernelSource>>
EmitRngGetAndUpdateStateLLVMIR(
    const HloRngGetAndUpdateStateInstruction* hlo,
    IrEmitterContext* ir_emitter_context,
    const emitters::KernelArguments& kernel_arguments) {
  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  std::string op_name(hlo->name());
  std::unique_ptr<llvm::Module> llvm_module =
      ir_emitter_context->CreateLLVMModule(op_name, *llvm_context);
  llvm::IRBuilder<> b(*llvm_context);

  VLOG(3) << "Generating (without reuse check): " << op_name;

  LaunchDimensions launch_dimensions;
  std::string sanitized_kernel_name =
      ir_emitter_context->GetSanitizedUniqueName(op_name);

  ABSL_ASSIGN_OR_RETURN(
      llvm::Function * kernel,
      BuildKernelPrototype(
          llvm_module.get(), ir_emitter_context->gpu_device_info(), op_name,
          sanitized_kernel_name, kernel_arguments, launch_dimensions, &b));

  std::vector<llvm_ir::IrArray> ir_arrays =
      IrArraysFor(kernel, kernel_arguments);
  llvm::Value* old_state =
      llvm_ir::RngGetAndUpdateState(hlo->delta(), llvm_module.get(), &b);
  llvm::Value* output_address = ir_arrays[0].EmitArrayElementAddress(
      llvm_ir::IrArray::Index(
          /*linear=*/b.getInt64(0), hlo->shape(), &b),
      &b, "rng_state_address");
  b.CreateStore(old_state, output_address);

  ABSL_ASSIGN_OR_RETURN(
      KernelSpec kernel_spec,
      emitters::GetKernelSpec(sanitized_kernel_name, *hlo,
                              &ir_emitter_context->buffer_assignment(),
                              launch_dimensions.AsWorkDimensions()));
  return KernelDefinition<LlvmKernelSource>(
      std::move(kernel_spec),
      LlvmKernelSource{std::move(llvm_context), std::move(llvm_module)});
}

}  // namespace xla::gpu
