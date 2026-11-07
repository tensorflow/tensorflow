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
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape_util.h"
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
                       const BufferAssignment* assignment)
    : hlo_module_(hlo_module),
      module_(module),
      assignment_(assignment),
      kernel_api_ir_builder_(module_->getContext(),
                             KernelApiIrBuilder::Options::FromHloModuleConfig(
                                 hlo_module_.config())) {
  SetModuleMemoryRegionName(*module_, "ir_emitter2");
}

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

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitSliceToDynamicHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit slice-to-dynamic host kernel: " << instr->name();

  ABSL_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                   EmitKernelPrototype(instr));
  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> ir_builder(ctx);
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  std::vector<llvm::Value*> dynamic_dims;
  int32_t raw_data_size =
      ShapeUtil::ByteSizeOf(ShapeUtil::MakeStaticShape(instr->shape()));
  llvm::Value* dest_buffer = output_array.GetBasePointer();
  for (int64_t i = 1; i < instr->operand_count(); ++i) {
    const int64_t dim_index = i - 1;
    llvm::Value* source_buffer = kernel_prototype.arguments[i].GetBasePointer();
    llvm::LoadInst* dyn_dim_size = ir_builder.CreateLoad(
        llvm_ir::ShapeToIrType(instr->operand(i)->shape(), ctx), source_buffer,
        "dyn_dim_size");

    llvm::Value* metadata = ir_builder.CreateConstInBoundsGEP1_32(
        ir_builder.getInt8Ty(), dest_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    ir_builder.CreateStore(dyn_dim_size, metadata);
    dynamic_dims.push_back(
        ir_builder.CreateIntCast(dyn_dim_size, ir_builder.getInt64Ty(),
                                 /*isSigned=*/true, "i64_dyn_dim_size"));
  }

  auto loop_body_emitter =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* source_element =
        kernel_prototype.arguments[0].EmitReadArrayElement(array_index,
                                                           &ir_builder);
    llvm::Value* linear_index =
        array_index.Linearize(dynamic_dims, &ir_builder);
    llvm_ir::IrArray::Index dest_index(linear_index, output_array.GetShape(),
                                       &ir_builder);
    output_array.EmitWriteArrayElement(dest_index, source_element, &ir_builder);
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(llvm_ir::LoopEmitter(loop_body_emitter,
                                       output_array.GetShape(), dynamic_dims,
                                       &ir_builder)
                      .EmitLoop(llvm_ir::IrName(instr)));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}

//===----------------------------------------------------------------------===//
// Building HostKernel prototypes.
//===----------------------------------------------------------------------===//

absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  return kernel_api_ir_builder_.EmitKernelPrototype(*module_, instr,
                                                    assignment_, "ir_emitter2");
}

}  // namespace xla::cpu
