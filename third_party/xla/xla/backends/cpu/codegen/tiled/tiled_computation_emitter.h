/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TILED_TILED_COMPUTATION_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_TILED_TILED_COMPUTATION_EMITTER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Environment mapping HloInstruction (and tuple element indices) to MLIR Value.
class ComputationEnvironment {
 public:
  void SetValue(const HloInstruction* instr, mlir::Value val) {
    scalar_values_[instr] = val;
  }

  void SetTupleElement(const HloInstruction* instr, int64_t index,
                       mlir::Value val) {
    tuple_values_[{instr, index}] = val;
  }

  mlir::Value GetValue(const HloInstruction* instr) const {
    auto it = scalar_values_.find(instr);
    if (it != scalar_values_.end()) return it->second;
    return nullptr;
  }

  mlir::Value GetTupleElement(const HloInstruction* instr,
                              int64_t index) const {
    auto it = tuple_values_.find({instr, index});
    if (it != tuple_values_.end()) return it->second;
    return nullptr;
  }

  bool Contains(const HloInstruction* instr) const {
    if (scalar_values_.contains(instr)) return true;
    if (tuple_values_.contains({instr, 0})) return true;
    return false;
  }

  llvm::SmallVector<mlir::Value> GetAllValues(
      const HloInstruction* instr) const {
    llvm::SmallVector<mlir::Value> values;
    if (instr->shape().IsTuple()) {
      for (int64_t i = 0; i < ShapeUtil::TupleElementCount(instr->shape());
           ++i) {
        mlir::Value v = GetTupleElement(instr, i);
        if (v) values.push_back(v);
      }
    } else {
      mlir::Value v = GetValue(instr);
      if (v) values.push_back(v);
    }
    return values;
  }

 private:
  absl::flat_hash_map<const HloInstruction*, mlir::Value> scalar_values_;
  absl::flat_hash_map<std::pair<const HloInstruction*, int64_t>, mlir::Value>
      tuple_values_;
};

// Whole-computation kernel emitter for XLA:CPU tiled execution.
// Drives MLIR lowering for whole computations containing control flow
// (scf.while, scf.if), tuple & multi-output roots, topological instruction
// scheduling, and edge materialization (memref.alloca & BufferAssignment
// slices).
class TiledComputationKernelEmitter final
    : public KernelEmitter<MlirKernelSource> {
 public:
  TiledComputationKernelEmitter(mlir::MLIRContext& mlir_context,
                                const HloInstruction* instr,
                                const BufferAssignment* buffer_assignment,
                                absl::string_view name);

  absl::string_view name() const final {
    return "tiled_computation_kernel_emitter";
  }

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() override;

 private:
  absl::Status EmitComputation(mlir::ImplicitLocOpBuilder& b,
                               const HloComputation* computation,
                               ComputationEnvironment& env);

  absl::Status EmitInstruction(mlir::ImplicitLocOpBuilder& b,
                               const HloInstruction* instr,
                               ComputationEnvironment& env);

  absl::Status EmitWhile(mlir::ImplicitLocOpBuilder& b,
                         const HloInstruction* while_instr,
                         ComputationEnvironment& env);

  absl::Status EmitConditional(mlir::ImplicitLocOpBuilder& b,
                               const HloInstruction* cond_instr,
                               ComputationEnvironment& env);

  mlir::MLIRContext& mlir_context_;
  const HloInstruction* instr_;
  const BufferAssignment* buffer_assignment_;
  std::string name_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TILED_TILED_COMPUTATION_EMITTER_H_
