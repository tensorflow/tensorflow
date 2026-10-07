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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_TRANSFORMS_PASSES_H_

#include "tensor_ir/Dialect/TensorIR.h"  // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"  // IWYU pragma: keep

namespace mlir::nv_tensor_ir::xla {

#define GEN_PASS_DECL
#include "xla/backends/gpu/codegen/tensor_ir/transforms/passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xla/backends/gpu/codegen/tensor_ir/transforms/passes.h.inc"

}  // namespace mlir::nv_tensor_ir::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_TRANSFORMS_PASSES_H_
