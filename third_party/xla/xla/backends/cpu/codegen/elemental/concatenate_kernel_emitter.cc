/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/elemental/concatenate_kernel_emitter.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

static absl::Status CanDoFastConcatenate(const HloInstruction* concatenate) {
  const Shape& output_shape = concatenate->shape();
  for (auto* op : concatenate->operands()) {
    if (!LayoutUtil::Equal(op->shape().layout(), output_shape.layout())) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Operand has mismatching layouts");
    }
  }
  return absl::OkStatus();
};

ConcatenateKernelEmitter::ConcatenateKernelEmitter(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment,
    const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<ConcatenateKernelEmitter::KernelDefinition>
ConcatenateKernelEmitter::EmitKernelDefinition() {
  if (absl::Status status = CanDoFastConcatenate(instr_); !status.ok()) {
    VLOG(1) << "Could not emit fast concatenate for " << instr_->ToString()
            << ": " << status.message();
    return ElementalKernelEmitter(instr_, buffer_assignment_, target_machine_)
        .EmitKernelDefinition();
  }

  auto ctx = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }

  const auto& backend_config = instr_->backend_config<BackendConfig>();
  const auto& partitions = backend_config->outer_dimension_partitions();
  auto total_workgroups =
      absl::c_accumulate(partitions, 1, std::multiplies<int64_t>());

  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_elemental_kernel_module"), *ctx);

  TF_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(
          *llvm_module, instr_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> ir_builder(*ctx);
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  TF_ASSIGN_OR_RETURN(
      bool is_parallel,
      EmitFastConcatenate(instr_, kernel_prototype.arguments, output_array,
                          llvm_module.get(), ir_builder,
                          kernel_prototype.workgroup_id.x, total_workgroups));

  LlvmKernelSource source(std::move(ctx), std::move(llvm_module));
  NumWorkGroups num_workgroups;
  if (is_parallel) {
    num_workgroups.x = total_workgroups;
  }

  KernelSpec::Buffers arguments, results;
  for (const auto& buffer : kernel_prototype.argument_buffers) {
    arguments.push_back({buffer.slice, buffer.shape});
  }
  for (const auto& buffer : kernel_prototype.result_buffers) {
    results.push_back({buffer.slice, buffer.shape});
  }

  KernelSpec spec(kernel_prototype.function->getName(), num_workgroups,
                  std::move(arguments), std::move(results),
                  std::move(kernel_prototype.invariant_arguments));

  return KernelDefinition(std::move(spec), std::move(source));
}

}  // namespace xla::cpu
