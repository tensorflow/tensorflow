/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/Base.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/errors.h"

namespace mlir {
namespace mhlo {

namespace {

#define GEN_PASS_DEF_VERIFYTFXLALEGALIZATION
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

auto* mlir_failed_legalization_op_count =
    tensorflow::monitoring::Counter<1>::New(
        "/tensorflow/core/tf2xla/"
        "mlir_second_phase_failed_legalization_op_count",
        "Counts which op fails to legalize", "op_name");

auto* mlir_non_static_op_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/"
    "mlir_second_phase_non_static_op_count",
    "Counts which ops do not have static results", "op_name");

auto* mlir_non_static_op_skip_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/"
    "mlir_second_phase_non_static_op_skip_count",
    "Counts skipped ops which do not have static results", "op_name");

static const char* kMustBeConstantError =
    "must have compile-time constant inputs and outputs.\n\n"
    "XLA compilation requires that operator arguments that represent shapes or "
    "dimensions be evaluated to concrete values at compile time.  This error "
    "means that a shape or dimension argument could not be evaluated at "
    "compile time, usually because the value of the argument depends on a "
    "parameter to the computation, on a variable, or on a stateful operation "
    "such as a random number generator.";

// TODO(b/282188914) remove the operations to skip once tests are fixed.
static const DenseSet<mlir::TypeID>* operations_to_skip =
    new DenseSet<mlir::TypeID>{mlir::TypeID::get<mhlo::EinsumOp>()};

class VerifyTFXLALegalization
    : public impl::VerifyTFXLALegalizationBase<VerifyTFXLALegalization> {
 public:
  explicit VerifyTFXLALegalization(bool legalize_chlo) {
    legalize_chlo_ = legalize_chlo;
  }

  void runOnOperation() override;
};

static void IncrementCounterFor(tensorflow::monitoring::Counter<1>* counter,
                                Operation* op) {
  counter->GetCell(op->getName().getStringRef().str())->IncrementBy(1);
}

bool HasBounds(RankedTensorType type) {
  auto bounds = hlo::encodingToBounds(type.getEncoding());
  return !bounds.empty();
}

bool HasStaticShapeOrBounded(Value val) {
  auto type = val.getType();
  if (mlir::isa<UnrankedTensorType>(type)) {
    return false;
  }
  if (mlir::isa<RankedTensorType>(type)) {
    auto ranked_tensor = mlir::dyn_cast<RankedTensorType>(type);
    if (ranked_tensor.hasStaticShape()) {
      return true;
    }
    return HasBounds(ranked_tensor);
  }
  return true;
}

bool EmitMustBeConstantError(Operation* op) {
  if (operations_to_skip->contains(op->getRegisteredInfo()->getTypeID())) {
    IncrementCounterFor(mlir_non_static_op_skip_count, op);
    return true;
  }
  emitError(op->getLoc()) << absl::StrCat(
      "Node `", op->getName().getStringRef().str(), "` ", kMustBeConstantError);
  return false;
}

bool IsStaticOperation(Operation* op) {
  for (auto o : op->getResults()) {
    if (!HasStaticShapeOrBounded(o)) {
      return EmitMustBeConstantError(op);
    }
  }
  return true;
}

bool IsMhloAndStatic(Operation* op) {
  if (!llvm::isa<mlir::mhlo::MhloDialect>(op->getDialect())) {
    // Skip this op if it isn't an mhlo op.
    return true;
  }
  return IsStaticOperation(op);
}

bool IsDefaultConversionLegal(
    Operation* op, const ConversionTarget& default_conversion_target) {
  if (!default_conversion_target.isLegal(op)) {
    emitError(op->getLoc()) << "Could not legalize op: " << op->getName();
    return false;
  }
  return true;
}

void VerifyTFXLALegalization::runOnOperation() {
  Operation* func_op = getOperation();
  ConversionTarget default_conversion_target =
      GetDefaultLegalConversionTargets(getContext(), legalize_chlo_);

  bool has_invalid_ops = false;
  func_op->walk([&](Operation* op) {
    if (!IsMhloAndStatic(op)) {
      has_invalid_ops = true;
      IncrementCounterFor(mlir_non_static_op_count, op);
      return WalkResult::interrupt();
    }
    if (!IsDefaultConversionLegal(op, default_conversion_target)) {
      has_invalid_ops = true;
      IncrementCounterFor(mlir_failed_legalization_op_count, op);
    }
    return WalkResult::advance();
  });

  if (has_invalid_ops) signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyTFXLALegalizationPass(bool legalize_chlo) {
  return std::make_unique<VerifyTFXLALegalization>(legalize_chlo);
}

}  // namespace mhlo
}  // namespace mlir
