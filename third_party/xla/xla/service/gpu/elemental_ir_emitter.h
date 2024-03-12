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

#ifndef XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
#define XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class GpuElementalIrEmitter : public ElementalIrEmitter {
 public:
  GpuElementalIrEmitter(IrEmitterContext& ir_emitter_context,
                        llvm::IRBuilder<>* b);

 protected:
  llvm_ir::IrArray::Index GetSourceIndexOfBitcast(
      const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) override;

  absl::StatusOr<llvm::Value*> EmitFloatBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value,
      llvm::Value* rhs_value) override;

  absl::StatusOr<llvm::Value*> EmitLog(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitLog1p(PrimitiveType prim_type,
                                         llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitSin(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitCos(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitTan(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type,
                                       llvm::Value* value,
                                       absl::string_view name) override;

  absl::StatusOr<llvm::Value*> EmitExpm1(PrimitiveType prim_type,
                                         llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitSqrt(PrimitiveType prim_type,
                                        llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitRsqrt(PrimitiveType prim_type,
                                         llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitPow(PrimitiveType prim_type,
                                       llvm::Value* lhs, llvm::Value* rhs,
                                       absl::string_view name) override;

  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view name) override;

  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitComplexAbs(PrimitiveType prim_type,
                                              llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitCbrt(PrimitiveType prim_type,
                                        llvm::Value* value) override;

  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view, bool /*is_reducer*/) override;

  bool fast_min_max() override {
    return ir_emitter_context_.debug_options().xla_gpu_enable_fast_min_max();
  }

 private:
  // Emits IR for op, which must have opcode kPower.
  absl::StatusOr<llvm::Value*> EmitPowerOp(const HloInstruction* op,
                                           llvm::Value* lhs_value,
                                           llvm::Value* rhs_value);

  // Emits IR to call a device function of type [T] -> T.  Adjusts
  // callee_name according to T.  Returns the IR value that represents the
  // return value of the function.
  absl::StatusOr<llvm::Value*> EmitDeviceMathCall(
      TargetDeviceFunctionID funcid, absl::Span<llvm::Value* const> operands,
      absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
      absl::string_view name = "");

  // Emits IR to call a function of type [T] -> T.  Does not munge callee_name.
  // Returns the IR value that represents the return value of the function.
  absl::StatusOr<llvm::Value*> EmitMathCall(
      const std::string& callee_name, absl::Span<llvm::Value* const> operands,
      absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
      absl::string_view name = "");

  IrEmitterContext& ir_emitter_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
