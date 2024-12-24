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

#include "xla/backends/cpu/testlib/llvm_ir_kernel_emitter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_spec.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

}  // namespace

LlvmIrKernelEmitter::LlvmIrKernelEmitter(absl::string_view llvm_ir,
                                         absl::string_view kernel_name,
                                         se::ThreadDim thread_dim,
                                         absl::Span<const KernelArg> args)
    : llvm_ir_(llvm_ir),
      kernel_name_(kernel_name),
      thread_dim_(thread_dim),
      args_(args.begin(), args.end()) {}

absl::StatusOr<std::unique_ptr<KernelSpec>>
LlvmIrKernelEmitter::EmitKernelSpec() {
  auto context = std::make_unique<llvm::LLVMContext>();

  // Parse LLVM IR into a module and create a LlvmIrKernelSource.
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssembly(
      llvm::MemoryBufferRef(llvm_ir_, kernel_name_), diagnostic, *context);

  if (module == nullptr) {
    return Internal("Failed to parse kernel LLVM IR: %s",
                    diagnostic.getMessage().str());
  }

  auto source = std::make_unique<LlvmIrKernelSource>(
      std::move(context), std::move(module), kernel_name_);

  // Convert kernel arguments to fake allocations and buffer uses.
  std::vector<BufferAllocation> buffer_allocations;
  KernelSpec::BufferUses buffer_uses;

  buffer_allocations.reserve(args_.size());
  buffer_uses.reserve(args_.size());

  for (const LlvmIrKernelEmitter::KernelArg& arg : args_) {
    auto& allocation = buffer_allocations.emplace_back(
        buffer_allocations.size(), arg.size_bytes, /*color=*/0);
    BufferAllocation::Slice slice(&allocation, 0, arg.size_bytes);
    buffer_uses.push_back(BufferUse(slice, arg.memory_access));
  }

  return std::make_unique<LlvmIrKernelSpec>(
      thread_dim_, std::move(buffer_allocations), std::move(buffer_uses),
      std::move(source));
}

}  // namespace xla::cpu
