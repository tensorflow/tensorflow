/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_XTILE_TEST_BASE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_XTILE_TEST_BASE_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/cost_model/block_level_parameters.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::gpu {

class XTileTestBase {
 public:
  XTileTestBase() { RegisterSymbolicExprStorage(&mlir_context_); }
  mlir::MLIRContext* mlir_context() { return &mlir_context_; }

  // Creates a shared dialect IR for the fusion `triton_fusion_name` inside the
  // computation defined by `hlo_text`.
  // The function returns the shared dialect IR and the HLO module. The HLO
  // module is returned so that the user can work with the computation that
  // generated the fusion if needed. This function also checks the generated
  // shared dialect IR against the `filecheck_pattern`.
  absl::StatusOr<
      std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::unique_ptr<HloModule>>>
  CreateXTileIrAndFileCheck(std::unique_ptr<HloModule> hlo_module,
                            absl::string_view triton_fusion_name,
                            absl::string_view filecheck_pattern);

  // Creates a shared dialect IR from the given HLO computation and returns it.
  // This function also checks the generated shared dialect IR against the
  // `filecheck_pattern`.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateXTileIrAndFileCheck(
      const HloComputation& computation,
      const BlockLevelParameters& block_level_parameters,
      absl::string_view filecheck_pattern);

  // Lowers the given shared dialect IR to Triton IR and checks the result
  // against the `filecheck_pattern`.
  absl::Status LowerXTileIrToTritonAndFileCheck(
      mlir::ModuleOp xtile_dialect_module, absl::string_view filecheck_pattern,
      const HloFusionInstruction& fusion);

 private:
  mlir::MLIRContext mlir_context_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_XTILE_TEST_BASE_H_
