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

#include "xla/backends/cpu/testlib/elemental_kernel_emitter.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_spec.h"  // Move this outside of testlib?
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/errors.h"

namespace xla::cpu {

class TemporraryCpuElementalIrEmitter : public CpuElementalIrEmitter {
 public:
  using CpuElementalIrEmitter::CpuElementalIrEmitter;

 private:
  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    return absl::UnimplementedError("");
  }
};

ElementalKernelEmitter::ElementalKernelEmitter(
    std::unique_ptr<HloInstruction> op_hlo)
    : op_hlo_(std::move(op_hlo)),
      context_(std::make_unique<llvm::LLVMContext>()),
      kernel_api_ir_builder_(*context_.getContext(),
                             KernelApiIrBuilder::Options{true, 256}) {}

absl::StatusOr<std::unique_ptr<KernelSpec>>
ElementalKernelEmitter::EmitKernelSpec() {
  llvm::LLVMContext& ctx = *context_.getContext();
  auto module = std::make_unique<llvm::Module>(
      absl::StrCat(op_hlo_->name(), "_elemental_kernel_module"), ctx);

  llvm::IRBuilder<> ir_builder(ctx);

  std::string function_name = absl::StrCat(op_hlo_->name(), "_kernel");
  llvm::Function* function =
      kernel_api_ir_builder_.EmitKernelFunction(*module, function_name);

  ir_builder.SetInsertPoint(llvm::BasicBlock::Create(ctx, "", function));

  llvm::Value* call_frame = function->getArg(0);

  std::vector<llvm_ir::IrArray> input_arrays;
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;

  input_arrays.reserve(op_hlo_->operand_count());
  for (size_t idx = 0; idx < op_hlo_->operand_count(); ++idx) {
    const HloInstruction* operand = op_hlo_->operand(idx);
    const Shape& input_shape = operand->shape();

    llvm_ir::IrArray& input_array =
        input_arrays.emplace_back(kernel_api_ir_builder_.EmitKernelArgument(
            ir_builder, call_frame, idx, input_shape));

    // We are treading a fine line here, but as we have reserved enough space
    // for the input arrays, we can safely use references to them.
    operand_to_generator[operand] =
        [&input_array, &ir_builder](const llvm_ir::IrArray::Index& index)
        -> absl::StatusOr<llvm::Value*> {
      return input_array.EmitReadArrayElement(index, &ir_builder);
    };
  }

  // TODO(willfroom): use real IR emitter here.
  TemporraryCpuElementalIrEmitter elemental_ir_emitter(module.get(),
                                                       &ir_builder, true, true);

  llvm_ir::ElementGenerator element_generator =
      elemental_ir_emitter.MakeElementGenerator(op_hlo_.get(),
                                                operand_to_generator);

  llvm_ir::IrArray output_array = kernel_api_ir_builder_.EmitKernelArgument(
      ir_builder, call_frame, op_hlo_->operand_count(), op_hlo_->shape());

  llvm_ir::LoopEmitter loop_emitter(element_generator, output_array,
                                    &ir_builder);

  TF_RETURN_IF_ERROR(loop_emitter.EmitLoop());

  // Return null pointer to signal success as we do not support error handling
  // in the compiled host kernel.
  ir_builder.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx)));

  auto source = std::make_unique<LlvmIrKernelSource>(
      context_, std::move(module), function_name);

  // TODO(willfroom): fill in buffer allocations and buffer uses when we support
  // creation from a real HLO instruction.
  std::vector<BufferAllocation> buffer_allocations;
  KernelSpec::BufferUses buffer_uses;

  return std::make_unique<LlvmIrKernelSpec>(
      se::ThreadDim(), std::move(buffer_allocations), std::move(buffer_uses),
      std::move(source));
}

}  // namespace xla::cpu
