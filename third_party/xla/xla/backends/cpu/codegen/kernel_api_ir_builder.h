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

#ifndef XLA_BACKENDS_CPU_CODEGEN_KERNEL_API_IR_BUILDER_H_
#define XLA_BACKENDS_CPU_CODEGEN_KERNEL_API_IR_BUILDER_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"

namespace xla::cpu {

class KernelApiIrBuilder {
 public:
  struct Options {
    bool enable_invariant_load_metadata;
    int32_t prefer_vector_width;
  };

  // Thread dimensions of the kernel invocation.
  struct ThreadDims {
    llvm::Value* x;
    llvm::Value* y;
    llvm::Value* z;
  };

  // Thread coordinates of the kernel invocation.
  struct ThreadId {
    llvm::Value* x;
    llvm::Value* y;
    llvm::Value* z;
  };

  KernelApiIrBuilder(llvm::LLVMContext& context_, Options options);

  ThreadDims EmitKernelThreadDims(llvm::IRBuilderBase& builder,
                                  llvm::Value* call_frame);
  ThreadId EmitKernelThread(llvm::IRBuilderBase& builder,
                            llvm::Value* call_frame);
  llvm_ir::IrArray EmitKernelArgument(llvm::IRBuilderBase& builder,
                                      llvm::Value* call_frame, int64_t index,
                                      const Shape& shape);
  llvm::Function* EmitKernelFunction(llvm::Module& module,
                                     absl::string_view name);

 private:
  llvm::LLVMContext& context_;

  Options options_;

  llvm::StructType* thread_dim_ty_;
  llvm::StructType* thread_ty_;
  llvm::StructType* arg_ty_;
  llvm::StructType* call_frame_ty_;
  llvm::FunctionType* kernel_function_ty_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_KERNEL_API_IR_BUILDER_H_
