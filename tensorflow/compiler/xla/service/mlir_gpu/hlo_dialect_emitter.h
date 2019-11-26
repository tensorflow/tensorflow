/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_HLO_DIALECT_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_HLO_DIALECT_EMITTER_H_

#include <memory>

#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/emission_context.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace mlir_gpu {

class HloDialectEmitter : public DfsHloVisitorWithDefault {
 public:
  HloDialectEmitter(xla::mlir_gpu::EmissionContext* emission_context,
                    ::mlir::Region* region,
                    llvm::ArrayRef<::mlir::Value*> arguments)
      : emission_context_(emission_context),
        builder_(region),
        arguments_(arguments) {}

  HloDialectEmitter(xla::mlir_gpu::EmissionContext* emission_context,
                    ::mlir::OpBuilder builder,
                    llvm::ArrayRef<::mlir::Value*> arguments)
      : emission_context_(emission_context),
        builder_(builder),
        arguments_(arguments) {}

  StatusOr<mlir::Value*> EmitComputation(const HloComputation& computation);

  Status DefaultAction(HloInstruction* instr) override;
  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandleCompare(HloInstruction* compare) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleIota(HloInstruction* iota) override;
  Status HandleParameter(HloInstruction* param) override;
  Status HandleReduce(HloInstruction* reduce) override;

 private:
  mlir::Location getLocation(const HloInstruction* instr) const;

  xla::mlir_gpu::EmissionContext* emission_context_;
  ::mlir::OpBuilder builder_;
  llvm::ArrayRef<::mlir::Value*> arguments_;
  absl::flat_hash_map<const xla::HloInstruction*, ::mlir::Value*>
      instruction_to_values_;
};

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_HLO_DIALECT_EMITTER_H_
