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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/cpu/host_kernel_emitter.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla::cpu {
namespace {

static std::vector<Shape> FlattenedParameters(const HloInstruction* instr) {
  std::vector<Shape> parameters;
  for (auto* operand : instr->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      parameters.push_back(indexed.shape);
    }
  }
  return parameters;
}

static std::vector<Shape> FlattenedResults(const HloInstruction* instr) {
  std::vector<Shape> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instr->shape())) {
    results.push_back(indexed.shape);
  }
  return results;
}

}  // namespace

//===----------------------------------------------------------------------===//
// ElementalIrEmitter
//===----------------------------------------------------------------------===//

class IrEmitter2::ElementalIrEmitter : public xla::ElementalIrEmitter {
 public:
  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilder<>* b,
                     bool fast_min_max)
      : xla::ElementalIrEmitter(module, b), fast_min_max_(fast_min_max) {}

 protected:
  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view) override {
    return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
  }

  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override {
    return xla::cpu::EmitTanh(module(), *b(), prim_type, value);
  }

  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override {
    return xla::cpu::EmitErf(module(), *b(), prim_type, value);
  }

  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    return absl::UnimplementedError("Not implemented");
  }

  bool fast_min_max() override { return fast_min_max_; }

 private:
  bool fast_min_max_;
};

//===----------------------------------------------------------------------===//
// IrEmitter2
//===----------------------------------------------------------------------===//

IrEmitter2::IrEmitter2(llvm::Module* module) : module_(module) {}

absl::StatusOr<IrEmitter2::HostKernelSym> IrEmitter2::EmitElementalHostKernel(
    const HloInstruction* instr) {
  llvm::IRBuilder<> b(module_->getContext());
  HostKernelEmitter emitter(module_);

  std::vector<Shape> parameters = FlattenedParameters(instr);
  std::vector<Shape> results = FlattenedResults(instr);

  HostKernelEmitter::KernelPrototype kernel_prototype =
      emitter.BuildKernelPrototype(instr->name(), parameters, results);
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : instr->operands()) {
    operand_to_generator[operand] = [&](const llvm_ir::IrArray::Index& index) {
      return kernel_prototype.parameters[0].EmitReadArrayElement(index, &b);
    };
  }

  // TODO(ezhulenev): Get `fast_min_max` from the HLO module config.
  ElementalIrEmitter elemental_emitter(module_, &b, /*fast_min_max_=*/true);
  auto element_generator =
      elemental_emitter.MakeElementGenerator(instr, operand_to_generator);

  TF_RETURN_IF_ERROR(
      llvm_ir::LoopEmitter(element_generator, kernel_prototype.results[0], &b)
          .EmitLoop(llvm_ir::IrName(instr)));

  return HostKernelSym{kernel_prototype.function->getName().str()};
}

}  // namespace xla::cpu
