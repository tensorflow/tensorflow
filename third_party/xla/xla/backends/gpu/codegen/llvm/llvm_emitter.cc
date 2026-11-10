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
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Argument.h"
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
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"

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
    ir_arrays.push_back(
        llvm_ir::IrArray(&llvm_arg, ir_type, kernel_argument.shape()));
  }
  return ir_arrays;
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

absl::StatusOr<KernelDefinition<LlvmKernelSource, GpuKernelSpec>>
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

  return KernelDefinition<LlvmKernelSource, GpuKernelSpec>(
      GpuKernelSpec{sanitized_kernel_name, launch_dimensions, kernel_arguments},
      LlvmKernelSource{std::move(llvm_context), std::move(llvm_module)});
}

}  // namespace xla::gpu
