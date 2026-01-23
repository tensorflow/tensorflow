/* Copyright 2025 The OpenXLA Authors.

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

#include <optional>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_dialect.h"

namespace xla::xtile {

#define GEN_PASS_DEF_VERIFYLEGALXTILEOPSPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

namespace {

bool WholeDialectIsLegal(mlir::Dialect* dialect) {
  return mlir::isa<XTileDialect, XlaDialect, mlir::arith::ArithDialect,
                   mlir::math::MathDialect, mlir::func::FuncDialect,
                   mlir::BuiltinDialect>(dialect);
}

std::optional<absl::string_view> IsLegalSCFOp(mlir::Operation* op) {
  if (mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::YieldOp>(op)) {
    return std::nullopt;
  }

  return "unsupported SCF op";
}

std::optional<absl::string_view> IsLegalTensorOp(mlir::Operation* op) {
  if (mlir::isa<mlir::tensor::BitcastOp>(op)) {
    return std::nullopt;
  }

  // TODO(willfroom): remove this ExtractOp & FromElementsOp once the special
  // handling of 0D tensors is removed from the emitter.
  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op)) {
    if (extract.getTensor().getType().getRank() != 0) {
      return "Expected rank 0";
    }
    return std::nullopt;
  }

  if (auto from_elements = mlir::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
    if (from_elements.getType().getRank() != 0) {
      return "Expected rank 0";
    }
    return std::nullopt;
  }

  return "unsupported Tensor op";
}

std::optional<absl::string_view> IsLegalStablehloOp(mlir::Operation* op) {
  if (mlir::isa<
          // go/keep-sorted start
          mlir::stablehlo::AllReduceOp, mlir::stablehlo::AddOp,
          mlir::stablehlo::AndOp, mlir::stablehlo::BroadcastInDimOp,
          mlir::stablehlo::ConvertOp, mlir::stablehlo::DivOp,
          mlir::stablehlo::DotGeneralOp, mlir::stablehlo::MaxOp,
          mlir::stablehlo::MinOp, mlir::stablehlo::MulOp,
          mlir::stablehlo::ReduceOp, mlir::stablehlo::RemOp,
          mlir::stablehlo::ReshapeOp, mlir::stablehlo::ReturnOp,
          mlir::stablehlo::SubtractOp, mlir::stablehlo::TransposeOp,
          mlir::stablehlo::XorOp, mlir::stablehlo::OrOp
          // go/keep-sorted end
          >(op)) {
    return std::nullopt;
  }

  if (auto iota = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
    if (iota.getType().getRank() != 1) {
      return "Only 1D iota is supported";
    }

    return std::nullopt;
  }

  return "unsupported StableHLO op";
}

// Check if a given op is xtile legal, if it is return std::nullopt else,
// returns a diagnostic string.
std::optional<absl::string_view> IsLegalOp(mlir::Operation* op) {
  mlir::Dialect* dialect = op->getDialect();
  if (WholeDialectIsLegal(dialect)) {
    return std::nullopt;
  }

  if (mlir::isa<mlir::scf::SCFDialect>(dialect)) {
    return IsLegalSCFOp(op);
  }

  if (mlir::isa<mlir::tensor::TensorDialect>(dialect)) {
    return IsLegalTensorOp(op);
  }

  if (mlir::isa<mlir::stablehlo::StablehloDialect>(dialect)) {
    return IsLegalStablehloOp(op);
  }

  return "unsupported op";
}

struct VerifyLegalXTileOpsPass
    : public impl::VerifyLegalXTileOpsPassBase<VerifyLegalXTileOpsPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    bool failed = false;
    module->walk([&failed](mlir::Operation* op) {
      if (std::optional<absl::string_view> diagnostic = IsLegalOp(op)) {
        op->emitError() << op->getName() << ": " << *diagnostic;
        failed = true;
      }
      return mlir::WalkResult::advance();
    });

    if (failed) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla::xtile
