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

#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CodeGen.h"
#include "xla/cpu_function_runtime.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {

namespace {

// Following struct types correspond to HostKernel C API.
// See: xla/backends/cpu/runtime/kernel_c_api.h

llvm::StructType* Dim3StructTy(llvm::LLVMContext& ctx, absl::string_view name) {
  llvm::IntegerType* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create(name, i64, i64, i64);
}

llvm::StructType* KernelThreadDimTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "XLA_CPU_KernelThreadDim");
}

llvm::StructType* KernelThreadTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "XLA_CPU_KernelThread");
}

llvm::StructType* KernelArgTy(llvm::LLVMContext& ctx) {
  llvm::PointerType* ptr = llvm::PointerType::getUnqual(ctx);
  llvm::IntegerType* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("XLA_CPU_KernelArg", ptr, i64);
}

llvm::StructType* KernelCallFrameTy(llvm::LLVMContext& ctx) {
  llvm::PointerType* ptr = llvm::PointerType::getUnqual(ctx);
  llvm::IntegerType* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("XLA_CPU_KernelCallFrame", ptr, ptr, i64,
                                  ptr);
}

llvm::FunctionType* KernelFunctionTy(llvm::LLVMContext& ctx) {
  return llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx),
                                 llvm::PointerType::getUnqual(ctx),
                                 /*isVarArg=*/false);
}

}  // namespace

KernelApiIrBuilder::KernelApiIrBuilder(llvm::LLVMContext& context,
                                       Options options)
    : context_(context), options_(std::move(options)) {
  thread_dim_ty_ = KernelThreadDimTy(context_);
  thread_ty_ = KernelThreadTy(context_);
  arg_ty_ = KernelArgTy(context_);
  call_frame_ty_ = KernelCallFrameTy(context_);
  kernel_function_ty_ = KernelFunctionTy(context_);
}

auto KernelApiIrBuilder::EmitKernelThreadDims(llvm::IRBuilderBase& builder,
                                              llvm::Value* call_frame)
    -> ThreadDims {
  llvm::Value* td_gep =
      builder.CreateStructGEP(call_frame_ty_, call_frame, 0, "tdims_gep");
  llvm::Value* tdims = builder.CreateLoad(builder.getPtrTy(), td_gep, "tdims");
  llvm::Value* x_gep =
      builder.CreateStructGEP(thread_dim_ty_, tdims, 0, "tdim_x_gep");
  llvm::Value* y_gep =
      builder.CreateStructGEP(thread_dim_ty_, tdims, 1, "tdim_y_gep");
  llvm::Value* z_gep =
      builder.CreateStructGEP(thread_dim_ty_, tdims, 2, "tdim_z_gep");

  return {builder.CreateLoad(builder.getInt64Ty(), x_gep, "tdim_x"),
          builder.CreateLoad(builder.getInt64Ty(), y_gep, "tdim_y"),
          builder.CreateLoad(builder.getInt64Ty(), z_gep, "tdim_z")};
}

auto KernelApiIrBuilder::EmitKernelThread(llvm::IRBuilderBase& builder,
                                          llvm::Value* call_frame) -> ThreadId {
  llvm::Value* t_gep =
      builder.CreateStructGEP(call_frame_ty_, call_frame, 1, "tid_gep");
  llvm::LoadInst* tids = builder.CreateLoad(builder.getPtrTy(), t_gep, "tids");
  llvm::Value* x_gep =
      builder.CreateStructGEP(thread_ty_, tids, 0, "tid_x_gep");
  llvm::Value* y_gep =
      builder.CreateStructGEP(thread_ty_, tids, 1, "tid_y_gep");
  llvm::Value* z_gep =
      builder.CreateStructGEP(thread_ty_, tids, 2, "tid_z_gep");

  return {builder.CreateLoad(builder.getInt64Ty(), x_gep, "tid_x"),
          builder.CreateLoad(builder.getInt64Ty(), y_gep, "tid_y"),
          builder.CreateLoad(builder.getInt64Ty(), z_gep, "tid_z")};
}

llvm_ir::IrArray KernelApiIrBuilder::EmitKernelArgument(
    llvm::IRBuilderBase& builder, llvm::Value* call_frame, int64_t index,
    const Shape& shape) {
  llvm::LLVMContext& ctx = builder.getContext();

  llvm::Type* ptr = llvm::PointerType::get(ctx, 0);
  std::string name = absl::StrCat("arg", index);

  llvm::Value* args_gep =
      builder.CreateStructGEP(call_frame_ty_, call_frame, 3, "args_gep");
  llvm::LoadInst* args = builder.CreateLoad(ptr, args_gep, "args");
  llvm::Value* data_gep =
      builder.CreateConstGEP2_32(arg_ty_, args, index, 0, name + "_gep");
  llvm::LoadInst* data = builder.CreateLoad(ptr, data_gep, name);

  // All buffers passed to host kernels are expected to be properly aligned,
  // emit metadata to allow LLVM to use that information for optimization.
  llvm_ir::SetAlignmentMetadataForLoad(data, cpu_function_runtime::MinAlign());

  // All buffers pointers passed to host kernels are expected to be
  // dereferenceable.
  llvm_ir::SetDereferenceableMetadataForLoad(data,
                                             ShapeUtil::ByteSizeOf(shape));

  // All buffers pointers passed to host kernels are expected to be invariant
  // over the whole program. Note the metadata is attached only to loading
  // buffer pointers, not to loading actual buffers.
  if (options_.enable_invariant_load_metadata) {
    data->setMetadata(llvm::LLVMContext::MD_invariant_load,
                      llvm::MDNode::get(data->getContext(), /*MDs=*/{}));
  }

  return llvm_ir::IrArray(data, llvm_ir::ShapeToIrType(shape, ctx), shape);
}

llvm::Function* KernelApiIrBuilder::EmitKernelFunction(llvm::Module& module,
                                                       absl::string_view name) {
  llvm::Function* function = llvm::Function::Create(
      kernel_function_ty_, llvm::GlobalValue::ExternalLinkage, name, module);

  // We use external linkage because we'll be resolving this function from the
  // XLA runtime.
  function->setCallingConv(llvm::CallingConv::C);

  // Generate unwind information so that GDB can crawl through the stack frames
  // created by the JIT compiled code.
  function->setUWTableKind(llvm::UWTableKind::Default);

  // Set prefer-vector-width attribute to allow LLVM to use wider vector
  // registers (by default LLVM uses at most 256-bit registers).
  function->addFnAttr("prefer-vector-width",
                      absl::StrCat(options_.prefer_vector_width));

  // Always keep a frame pointer for the host kernel so we can see them in all
  // performance profiling tools.
  function->addFnAttr("frame-pointer", "all");

  return function;
}

}  // namespace xla::cpu
