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

#include "xla/service/cpu/ir_emitter2.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

namespace {

std::string SortCsv(absl::string_view csv) {
  std::vector<absl::string_view> v =
      absl::StrSplit(csv, ',', absl::SkipEmpty());
  std::sort(v.begin(), v.end());
  return absl::StrJoin(v, ",");
}

}  // namespace

//===----------------------------------------------------------------------===//
// IrEmitter2
//===----------------------------------------------------------------------===//

IrEmitter2::IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
                       IrEmitter* nested_ir_emitter)
    : hlo_module_(hlo_module),
      module_(module),
      nested_ir_emitter_(nested_ir_emitter),
      kernel_api_ir_builder_(module_->getContext(),
                             KernelApiIrBuilder::Options::FromHloModuleConfig(
                                 hlo_module_.config())) {}

IrEmitter2::KernelInfo::KernelInfo(KernelPrototype prototype,
                                   const se::BlockDim& block_dims,
                                   const se::ThreadDim& thread_dims)
    : name(prototype.function->getName().str()),
      block_dims(block_dims),
      thread_dims(thread_dims),
      invariant_arguments(std::move(prototype.invariant_arguments)) {}

IrEmitter2::KernelInfo::KernelInfo(
    const std::string& name, const se::BlockDim& block_dims,
    const se::ThreadDim& thread_dims,
    const absl::flat_hash_set<int64_t>& invariant_arguments,
    absl::string_view backend_extra_options)
    : name(name),
      block_dims(block_dims),
      thread_dims(thread_dims),
      invariant_arguments(invariant_arguments),
      backend_extra_options(SortCsv(backend_extra_options)) {}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitPadHostKernel(
    const HloInstruction* pad) {
  VLOG(2) << "Emit Pad host kernel.";

  ABSL_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype, EmitKernelPrototype(pad));

  llvm_ir::IrArray operand_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray padvalue_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray output_array = kernel_prototype.results[0];

  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> b(ctx);
  auto builder_overwrite = nested_ir_emitter_->WithBuilder(b);

  nested_ir_emitter_->PushComputeFunction(
      &b, module_, kernel_prototype.function, kernel_prototype.return_block);

  ABSL_RETURN_IF_ERROR(nested_ir_emitter_->HandlePad(
      const_cast<HloInstruction*>(pad), operand_array, padvalue_array,
      output_array));

  nested_ir_emitter_->PopComputeFunction();

  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitSliceToDynamicHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit slice-to-dynamic host kernel: " << instr->name();

  ABSL_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                   EmitKernelPrototype(instr));
  llvm::IRBuilder<> ir_builder(module_->getContext());
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  auto guard = nested_ir_emitter_->WithBuilder(ir_builder);
  ABSL_RETURN_IF_ERROR(nested_ir_emitter_->EmitSliceToDynamic(
      instr, kernel_prototype.arguments, output_array));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

//===----------------------------------------------------------------------===//
// Building HostKernel prototypes.
//===----------------------------------------------------------------------===//

absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  return kernel_api_ir_builder_.EmitKernelPrototype(
      *module_, instr, &nested_ir_emitter_->assignment(), "ir_emitter2");
}

}  // namespace xla::cpu
