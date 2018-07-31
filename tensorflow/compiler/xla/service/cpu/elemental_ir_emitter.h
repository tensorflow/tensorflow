/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace cpu {

class CpuElementalIrEmitter : public ElementalIrEmitter {
 public:
  CpuElementalIrEmitter(const HloModuleConfig& module_config,
                        IrEmitter* ir_emitter, llvm::Module* module)
      : ElementalIrEmitter(module_config, module, ir_emitter->b()),
        ir_emitter_(ir_emitter) {}

  llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator) const override;

 protected:
  StatusOr<llvm::Value*> EmitFloatUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value) const override;
  StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type, llvm::Value* lhs,
                                   llvm::Value* rhs) const override;

  IrEmitter* ir_emitter_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_
