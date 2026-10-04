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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_HLO_TO_TENSOR_IR_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_HLO_TO_TENSOR_IR_H_

#include "tensor_ir/Dialect/TensorIR.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla::gpu::tensor_ir {

// Imports `computation` into `module` as a StableHLO `func.func` (via
// `HloFunctionImporter`) and legalizes it to a single `nv_tensor_ir.graph`
// (via the `legalize-stablehlo-to-tensor-ir` pass). Returns the resulting
// graph, which is owned by `module`.
//
// `module` must be empty of graphs; the caller is expected to pass a freshly
// created module. Any MLIR diagnostic emitted during import or legalization is
// folded into the returned status.
//
// Note: the computation is imported as the module's main function, but the
// resulting graph is renamed after `computation` (sanitized for use as a
// symbol name), so that the compiled kernel is attributable in profiles.
absl::StatusOr<mlir::nv_tensor_ir::GraphOp> ImportAndLegalizeComputation(
    const HloComputation& computation, mlir::ModuleOp module);

// Same as above, but creates the containing module in `context` and returns
// it. The returned module contains exactly one `nv_tensor_ir.graph`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportAndLegalizeComputation(
    const HloComputation& computation, mlir::MLIRContext* context);

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_HLO_TO_TENSOR_IR_H_
