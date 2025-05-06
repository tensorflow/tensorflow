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

#ifndef XLA_CODEGEN_LLVM_IR_KERNEL_SOURCE_H_
#define XLA_CODEGEN_LLVM_IR_KERNEL_SOURCE_H_

#include <memory>
#include <string>
#include <utility>

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/kernel_source.h"

namespace xla {

// XLA kernel compiled to LLVM IR. Depending on the concrete kernel emitter
// implementation we might emit a single LLVM module with multiple kernels or a
// separate LLVM module for each kernel. Kernel function signature is defined by
// the backend specific ABI.
class LlvmIrKernelSource final : public KernelSource {
 public:
  LlvmIrKernelSource(llvm::orc::ThreadSafeContext context,
                     std::unique_ptr<llvm::Module> module)
      : module_(std::move(module), std::move(context)) {}

  LlvmIrKernelSource(LlvmIrKernelSource&& other) = default;
  LlvmIrKernelSource& operator=(LlvmIrKernelSource&& other) noexcept = default;

  llvm::orc::ThreadSafeModule thread_safe_module() && {
    return std::move(module_);
  }

  std::string ToString() const final;

 private:
  llvm::orc::ThreadSafeModule module_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_LLVM_IR_KERNEL_SOURCE_H_
