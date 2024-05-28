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

#ifndef XLA_SERVICE_CPU_HOST_KERNEL_EMITTER_H_
#define XLA_SERVICE_CPU_HOST_KERNEL_EMITTER_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"

namespace xla::cpu {

// Collection of LLVM utilities to emit functions compatible with XLA HostKernel
// API (ABI) for compiled HLO operations.
class HostKernelEmitter {
 public:
  // Thread dimensions of the kernel invocation.
  struct IrKernelThreadDims {
    llvm::Value* x;
    llvm::Value* y;
    llvm::Value* z;
  };

  // Thread coordinates of the kernel invocation.
  struct IrKernelThread {
    llvm::Value* x;
    llvm::Value* y;
    llvm::Value* z;
  };

  struct KernelPrototype {
    llvm::Function* function;

    // LLVM values identifying kernel invocation thread coordinates.
    IrKernelThreadDims thread_dims;
    IrKernelThread thread;

    // LLVM values corresponding to the kernel parameters and results arrays.
    std::vector<llvm_ir::IrArray> parameters;
    std::vector<llvm_ir::IrArray> results;
  };

  explicit HostKernelEmitter(llvm::Module* module);

  KernelPrototype BuildKernelPrototype(std::string_view name,
                                       absl::Span<const Shape> parameters,
                                       absl::Span<const Shape> results);

 private:
  IrKernelThreadDims BuildKernelThreadDims(llvm::Value* call_frame,
                                           llvm::IRBuilder<>& b);

  IrKernelThread BuildKernelThread(llvm::Value* call_frame,
                                   llvm::IRBuilder<>& b);

  llvm_ir::IrArray BuildArgument(llvm::IRBuilder<>& b, llvm::Value* call_frame,
                                 int64_t idx, const Shape& shape);

  llvm::Module* module_;

  llvm::StructType* call_frame_ty_;
  llvm::StructType* thread_dims_ty_;
  llvm::StructType* thread_ty_;
  llvm::StructType* arg_ty_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_HOST_KERNEL_EMITTER_H_
