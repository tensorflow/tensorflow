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

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace gpu {

// This class is the top-level API for the HLO --> HLO dialect compiler. It
// implements the DfsHloVisitor interface and emits HLO computations as MLIR IR
// functions.
class HloDialectEmitter : public DfsHloVisitorWithDefault {
 public:
  HloDialectEmitter(const HloModule& hlo_module,
                    const BufferAssignment& assignment,
                    ::mlir::ModuleOp mlir_module);
  ~HloDialectEmitter() override = default;

  // The following methods implement the DfsHloVisitor interface.
  //
  // Default action which emits code for most operations. Operations which are
  // special in some way are handled explicitly in HandleFoo methods.
  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;

  Status FinishVisit(HloInstruction* root) override;

 private:
  ::mlir::ModuleOp mlir_module_;
  ::mlir::Builder builder_;
  absl::flat_hash_map<const xla::HloComputation*, ::mlir::FuncOp>
      computation_to_mlir_function_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloDialectEmitter);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_HLO_DIALECT_EMITTER_H_
