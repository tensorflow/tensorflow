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

#include "xla/service/cpu/host_kernel_emitter.h"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

static llvm::StructType* Dim3StructType(llvm::LLVMContext& ctx,
                                        std::string_view name) {
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create(name, i64, i64, i64);
}

// The following struct types correspond to HostKernel C API.
// See: xla/stream_executor/host/host_kernel_c_api.h

static llvm::StructType* KernelThreadDimType(llvm::LLVMContext& ctx) {
  return Dim3StructType(ctx, "SE_HOST_KernelThreadDim");
}

static llvm::StructType* KernelThreadType(llvm::LLVMContext& ctx) {
  return Dim3StructType(ctx, "SE_HOST_KernelThread");
}

static llvm::StructType* KernelArgType(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelArg", ptr, i64);
}

static llvm::StructType* KernelCallFrameType(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelCallFrame", ptr, ptr, i64,
                                  ptr);
}

static llvm::FunctionType* KernelFunctionType(llvm::LLVMContext& ctx) {
  return llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx),
                                 llvm::PointerType::getUnqual(ctx),
                                 /*isVarArg=*/false);
}

HostKernelEmitter::HostKernelEmitter(llvm::Module* module)
    : module_(module),
      call_frame_ty_(KernelCallFrameType(module_->getContext())),
      thread_dims_ty_(KernelThreadDimType(module_->getContext())),
      thread_ty_(KernelThreadType(module_->getContext())),
      arg_ty_(KernelArgType(module_->getContext())) {}

HostKernelEmitter::IrKernelThreadDims HostKernelEmitter::BuildKernelThreadDims(
    llvm::Value* call_frame, llvm::IRBuilder<>& b) {
  auto* thread_dims = b.CreateConstGEP2_64(call_frame_ty_, call_frame, 0, 0);
  auto* x_ptr = b.CreateConstGEP2_32(thread_dims_ty_, thread_dims, 0, 0);
  auto* y_ptr = b.CreateConstGEP2_32(thread_dims_ty_, thread_dims, 0, 1);
  auto* z_ptr = b.CreateConstGEP2_32(thread_dims_ty_, thread_dims, 0, 2);

  return {b.CreateLoad(b.getInt64Ty(), x_ptr),
          b.CreateLoad(b.getInt64Ty(), y_ptr),
          b.CreateLoad(b.getInt64Ty(), z_ptr)};
}

HostKernelEmitter::IrKernelThread HostKernelEmitter::BuildKernelThread(
    llvm::Value* call_frame, llvm::IRBuilder<>& b) {
  auto* thread_dims = b.CreateConstGEP2_64(call_frame_ty_, call_frame, 0, 1);
  auto* x_ptr = b.CreateConstGEP2_32(thread_ty_, thread_dims, 0, 0);
  auto* y_ptr = b.CreateConstGEP2_32(thread_ty_, thread_dims, 0, 1);
  auto* z_ptr = b.CreateConstGEP2_32(thread_ty_, thread_dims, 0, 2);

  return {b.CreateLoad(b.getInt64Ty(), x_ptr),
          b.CreateLoad(b.getInt64Ty(), y_ptr),
          b.CreateLoad(b.getInt64Ty(), z_ptr)};
}

llvm_ir::IrArray HostKernelEmitter::BuildArgument(llvm::IRBuilder<>& b,
                                                  llvm::Value* call_frame,
                                                  int64_t idx,
                                                  const Shape& shape) {
  auto* args_ptr = b.CreateConstGEP2_64(call_frame_ty_, call_frame, 0, 3);
  auto* arg_ptr = b.CreateConstGEP1_64(arg_ty_, args_ptr, idx);
  auto* data_ptr = b.CreateConstGEP2_64(arg_ty_, arg_ptr, 0, 0);

  llvm::Type* ptr = llvm::PointerType::get(b.getContext(), 0);
  return llvm_ir::IrArray(b.CreateLoad(ptr, data_ptr),
                          llvm_ir::ShapeToIrType(shape, module_), shape);
}

HostKernelEmitter::KernelPrototype HostKernelEmitter::BuildKernelPrototype(
    std::string_view name, absl::Span<const Shape> parameters,
    absl::Span<const Shape> results) {
  VLOG(3) << "Build kernel prototype for: " << name << " with "
          << parameters.size() << " parameters and " << results.size()
          << " results:";
  for (auto& parameter : parameters) {
    VLOG(3) << "  parameter: " << parameter.ToString(true);
  }
  for (auto& result : results) {
    VLOG(3) << "  result: " << result.ToString(true);
  }

  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> b(ctx);

  // Create a kernel function with HostKernel API.
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module_->getOrInsertFunction(name, KernelFunctionType(ctx)).getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "", function));

  llvm::Value* call_frame = function->getArg(0);

  // Build thread coordinates from the call frame.
  IrKernelThreadDims kernel_thread_dims = BuildKernelThreadDims(call_frame, b);
  IrKernelThread kernel_thread = BuildKernelThread(call_frame, b);

  int64_t idx = 0;

  // IrArrays for the parameters.
  std::vector<llvm_ir::IrArray> ir_parameters;
  for (const Shape& parameter : parameters) {
    ir_parameters.push_back(BuildArgument(b, call_frame, idx++, parameter));
  }

  // IrArrays for the results.
  std::vector<llvm_ir::IrArray> ir_results;
  for (const Shape& result : results) {
    ir_results.push_back(BuildArgument(b, call_frame, idx++, result));
  }

  // Return null pointer to signal success as we do not support error handling
  // in the compiled host kernel.
  b.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx)));

  return KernelPrototype{function, kernel_thread_dims, kernel_thread,
                         std::move(ir_parameters), std::move(ir_results)};
}

}  // namespace xla::cpu
