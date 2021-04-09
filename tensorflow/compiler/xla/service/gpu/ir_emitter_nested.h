/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_

#include "llvm/IR/Function.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"

namespace xla {
namespace gpu {

// Emits LLVM IR for a "nested computation" into a non-kernel device function.
//
// This is used to emit code for HloComputations that don't require a separate
// kernel call.  For example, IrEmitterNested is used to emit code for a kReduce
// HLO's elementwise reduction computation.  Notably, IrEmitterNested is *not*
// used to emit code for fusion nodes -- fusion nodes use FusedIrEmitter, which
// is a different beast altogether.
//
// IrEmitterNested generates a non-kernel function with the following
// parameters:
//
//   - N pointers to the buffers of each of the N parameters to the computation,
//   - a pointer to the output buffer of the computation, and
//   - a pointer to the top-level temp buffer.
//
class IrEmitterNested : public IrEmitter {
 public:
  static StatusOr<std::unique_ptr<IrEmitterNested>> Create(
      const HloModuleConfig& hlo_module_config,
      const HloComputation& nested_computation,
      IrEmitterContext* ir_emitter_context);

  IrEmitterNested(const IrEmitterNested&) = delete;
  IrEmitterNested& operator=(const IrEmitterNested&) = delete;

  // Overrides the default empty implementation. Binds the given instruction
  // "parameter" with the parameter of the IR function.
  Status HandleParameter(HloInstruction* parameter) override;

  llvm::Function* GetEmittedFunction() const { return emitted_function_; }

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Generate the code for the computation passed in the constructor.
  Status CodegenNestedComputation();

 private:
  // Constructs an LLVM IR emitter for a nested HLO computation. `function` is
  // the containing IR function this emitter produces IR to. See
  // IrEmitter::IrEmitter for the meanings of other arguments.
  IrEmitterNested(const HloModuleConfig& hlo_module_config,
                  const HloComputation& nested_computation,
                  IrEmitterContext* ir_emitter_context);

  const HloComputation& nested_computation_;
  llvm::Function* emitted_function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_NESTED_H_
