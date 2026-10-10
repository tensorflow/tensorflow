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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CONVERSION_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CONVERSION_H_

#include "tensor_ir/Dialect/TensorIR.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla::gpu::tensor_ir {

// Converts an HLO fusion computation into a TensorIR `GraphOp` appended
// to the body of `target`. If conversion fails, `target` is unmodified.
absl::StatusOr<mlir::nv_tensor_ir::GraphOp> ConvertFusionComputation(
    const HloComputation& source, mlir::ModuleOp target);

// Creates a new MLIR module containing the converted TensorIR graph for
// `source`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertFusionComputation(
    const HloComputation& source, mlir::MLIRContext* context);

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CONVERSION_H_
