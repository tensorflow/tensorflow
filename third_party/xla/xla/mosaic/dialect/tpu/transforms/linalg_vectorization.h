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

#ifndef XLA_MOSAIC_DIALECT_TPU_TRANSFORMS_LINALG_VECTORIZATION_H_
#define XLA_MOSAIC_DIALECT_TPU_TRANSFORMS_LINALG_VECTORIZATION_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "xla/mosaic/pass_boilerplate.h"

namespace mlir::tpu {

struct LinalgVectorizationPassOptions {
  bool supports_bf16_alu_instructions = false;
  bool supports_bf16_matmul = false;
};

struct LinalgVectorizationPass
    : public jaxlib::mlir::Pass<LinalgVectorizationPass, func::FuncOp> {
  using jaxlib::mlir::Pass<LinalgVectorizationPass, func::FuncOp>::Pass;

  static constexpr llvm::StringLiteral kArgumentName = "linalg-vectorization";
  static constexpr llvm::StringLiteral kPassName = "LinalgVectorizationPass";

  LinalgVectorizationPass() = default;

  explicit LinalgVectorizationPass(LinalgVectorizationPassOptions options) {
    supports_bf16_alu_instructions = options.supports_bf16_alu_instructions;
    supports_bf16_matmul = options.supports_bf16_matmul;
  }

  LinalgVectorizationPass(const LinalgVectorizationPass& other) {
    supports_bf16_alu_instructions = other.supports_bf16_alu_instructions;
    supports_bf16_matmul = other.supports_bf16_matmul;
  }

  LinalgVectorizationPass& operator=(const LinalgVectorizationPass& other) {
    supports_bf16_alu_instructions = other.supports_bf16_alu_instructions;
    supports_bf16_matmul = other.supports_bf16_matmul;
    return *this;
  }

  void getDependentDialects(DialectRegistry& registry) const override;
  void runOnOperation() override;

 protected:
  ::mlir::Pass::Option<bool> supports_bf16_alu_instructions{
      *this, "supports-bf16-alu-instructions", llvm::cl::desc("")};
  ::mlir::Pass::Option<bool> supports_bf16_matmul{*this, "supports-bf16-matmul",
                                                  llvm::cl::desc("")};
};

inline std::unique_ptr<::mlir::Pass> createLinalgVectorizationPass(
    bool supports_bf16_alu_instructions = false,
    bool supports_bf16_matmul = false) {
  return std::make_unique<LinalgVectorizationPass>(
      LinalgVectorizationPassOptions{
          .supports_bf16_alu_instructions = supports_bf16_alu_instructions,
          .supports_bf16_matmul = supports_bf16_matmul,
      });
}

inline std::unique_ptr<::mlir::Pass> createLinalgVectorizationPass(
    LinalgVectorizationPassOptions options) {
  return std::make_unique<LinalgVectorizationPass>(std::move(options));
}

inline void registerLinalgVectorizationPass() {
  registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLinalgVectorizationPass();
  });
}

}  // namespace mlir::tpu

#endif  // XLA_MOSAIC_DIALECT_TPU_TRANSFORMS_LINALG_VECTORIZATION_H_
