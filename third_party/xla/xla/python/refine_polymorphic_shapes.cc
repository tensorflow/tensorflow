/* Copyright 2023 The JAX Authors.

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
#include "xla/python/refine_polymorphic_shapes.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

namespace {

constexpr llvm::StringRef shapeAssertionName = "shape_assertion";
constexpr llvm::StringRef errorMessageAttrName = "error_message";
// We bound the number of error_message_inputs for using llvm::formatv
constexpr int maxErrorMessageInputs = 32;  // TODO(necula): Remove this bound

// This pass is needed when we have shape assertions. A shape assertion is
// represented via the `stablehlo.custom_call @shape_assertion`
// custom call, and represents an assertion that the first operand
// (`assert_what`) evaluates to `true`. The custom call also has an
// `error_message` string attribute, and a variadic number
// of integer scalar operands that may be used to format the error message.
// The `error_message` may contain format specifiers `{0}`, `{1}`, ..., that
// are replaced with the values of the error message inputs. The formatting is
// done with the `llvm::formatv` function
// (https://llvm.org/docs/ProgrammersManual.html#formatting-strings-the-formatv-function).
//
struct CheckShapeAssertionsPass
    : public mlir::PassWrapper<CheckShapeAssertionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckShapeAssertionsPass)

  explicit CheckShapeAssertionsPass(bool enable_shape_assertions = true)
      : PassWrapper() {
    this->enable_shape_assertions = enable_shape_assertions;
  }

  CheckShapeAssertionsPass(const CheckShapeAssertionsPass &pass) {
    enable_shape_assertions = pass.enable_shape_assertions;
  }

 private:
  void runOnOperation() final {
    mlir::func::FuncOp func_op = getOperation();
    func_op.walk([this](mlir::stablehlo::CustomCallOp op) {
      if (op.getCallTargetName() != shapeAssertionName) return;
      if (!enable_shape_assertions) {
        op.erase();
        return;
      }
      // Check first for ill-formed assertions, rather than silently fail.
      if (mlir::failed(verifyShapeAssertion(op))) {
        signalPassFailure();
        return;
      }
      mlir::OperandRange inputs = op.getInputs();
      mlir::SmallVector<int64_t> assertWhat;
      if (mlir::failed(mlir::hlo::matchInts(inputs[0], assertWhat))) {
        op.emitError() << "expects static assert_what (operand #0)";
        signalPassFailure();
        return;
      }
      if (assertWhat[0] != 0) {
        op.erase();
        return;
      }
      llvm::StringRef errorMessage = getErrorMessage(op);
      mlir::SmallVector<int64_t> errorMessageInputs;
      for (int i = 1; i < inputs.size(); ++i) {
        mlir::SmallVector<int64_t> input;
        if (failed(mlir::hlo::matchInts(inputs[i], input))) {
          op.emitError() << "expects static error_message_input (operand #" << i
                         << ")";
          signalPassFailure();
          return;
        }
        errorMessageInputs.push_back(input[0]);
      }
      op.emitError(formatErrorMessage(errorMessage, errorMessageInputs));
      signalPassFailure();
    });
  }

  mlir::LogicalResult verifyShapeAssertion(mlir::stablehlo::CustomCallOp op) {
    if (!(1 <= op->getNumOperands() &&
          op->getNumOperands() <= 1 + maxErrorMessageInputs))
      return op.emitError() << "expects 1 <= size(operands) <= "
                            << (1 + maxErrorMessageInputs);
    int nrErrorMessageInputs = op.getNumOperands() - 1;
    if (op->getNumResults() != 0)
      return op.emitError("expects size(results) = 0");
    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() != "api_version" &&
          attr.getName() != "backend_config" &&
          attr.getName() != "call_target_name" &&
          attr.getName() != "error_message" &&
          attr.getName() != "has_side_effect")
        return op.emitError()
               << attr.getName() << " is not a supported attribute";
    }
    if (!op.hasEmptyBackendConfig())
      return op.emitError() << "expects an empty backend_config";
    if (op.getCallTargetName() != shapeAssertionName)
      return op.emitError() << "expects @shape_assertion";

    // input[0] (assert_what) : tensor<i1>
    auto assertWhatType =
        mlir::dyn_cast<mlir::ShapedType>(op.getInputs()[0].getType());
    if (!assertWhatType || !assertWhatType.hasRank() ||
        assertWhatType.getRank() != 0 ||
        !assertWhatType.getElementType().isSignlessInteger() ||
        assertWhatType.getElementTypeBitWidth() != 1)
      return op.emitError() << "expects assert_what (operand #0) "
                            << "to be a constant of type tensor<i1>";

    // input[1:] (error_message_inputs) : tensor<i32> or tensor<i64>
    for (int i = 0; i < nrErrorMessageInputs; ++i) {
      auto errorMessageInputType =
          mlir::dyn_cast<mlir::ShapedType>(op.getInputs()[i + 1].getType());
      if (!errorMessageInputType || !errorMessageInputType.hasRank() ||
          errorMessageInputType.getRank() != 0 ||
          !errorMessageInputType.getElementType().isSignlessInteger() ||
          (errorMessageInputType.getElementTypeBitWidth() != 32 &&
           errorMessageInputType.getElementTypeBitWidth() != 64))
        return op.emitError()
               << "expects error_message_input (operand #" << (i + 1) << ") "
               << "to be a constant of type tensor<i32> or tensor<i64>";
    }

    if (!op->hasAttr(errorMessageAttrName))
      return op.emitError() << "expects an error_message attribute";

    // error_message contains valid format specifiers.
    std::string errorMessage = getErrorMessage(op).data();
    // format specs: "{" index ["," layout] [":" format] "}"
    llvm::Regex formatSpecifierRE = llvm::Regex("{([0-9]+)[,:}]");
    do {
      mlir::SmallVector<llvm::StringRef> formatSpec;
      if (!formatSpecifierRE.match(errorMessage, &formatSpec)) {
        break;
      }
      int index = std::stoi(formatSpec[1].data());
      if (!(0 <= index && index < nrErrorMessageInputs)) {
        return op.emitError()
               << "expects error_message to contain format specifiers with "
               << "error_message_input index less than " << nrErrorMessageInputs
               << ". Found specifier " << formatSpec[0];
      }
      errorMessage = formatSpecifierRE.sub("", errorMessage);
    } while (true);

    return mlir::success();
  }

  llvm::StringRef getErrorMessage(mlir::stablehlo::CustomCallOp op) const {
    return mlir::cast<mlir::StringAttr>(op->getAttr(errorMessageAttrName))
        .getValue();
  }

  std::string formatErrorMessage(
      llvm::StringRef errorMessage,
      const mlir::SmallVector<int64_t> &errorMessageInputs) const {
    int nrErrorMessageInputs = errorMessageInputs.size();
    auto errorMessageFormat = errorMessage.data();
    if (nrErrorMessageInputs == 0) return errorMessageFormat;
    auto errInput = [nrErrorMessageInputs, &errorMessageInputs](int idx) {
      return (idx < nrErrorMessageInputs ? errorMessageInputs[idx] : -1);
    };
    return llvm::formatv(
        false, errorMessageFormat, errInput(0), errInput(1), errInput(2),
        errInput(3), errInput(4), errInput(5), errInput(6), errInput(7),
        errInput(8), errInput(9), errInput(10), errInput(11), errInput(12),
        errInput(13), errInput(14), errInput(15), errInput(16), errInput(17),
        errInput(18), errInput(19), errInput(20), errInput(21), errInput(22),
        errInput(23), errInput(24), errInput(25), errInput(26), errInput(27),
        errInput(28), errInput(29), errInput(30), errInput(31));
  }

  mlir::StringRef getArgument() const override {
    return "check-shape-assertions";
  }

  mlir::StringRef getDescription() const override {
    return "Check stablehlo.custom_call @shape_assertion ops.";
  }

  Option<bool> enable_shape_assertions{
      *this, "enable-shape-assertions",
      llvm::cl::desc("Whether shape assertions may generate errors."),
      llvm::cl::init(true)};
};

}  // namespace

absl::Status RefinePolymorphicShapes(mlir::ModuleOp module,
                                     bool enable_shape_assertions) {
  mlir::MLIRContext *context = module->getContext();
  if (VLOG_IS_ON(3)) context->disableMultithreading();

  // Verify the module before running passes on it.
  // If the module doesn't pass verification, all sorts of weirdness might
  // happen if we run the pass manager.
  mlir::BaseScopedDiagnosticHandler diag_handler(context);

  if (mlir::failed(mlir::verify(module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module verification failed: ",
                     diag_handler.ConsumeStatus().ToString()));
  }

  mlir::PassManager pm(context);
  if (VLOG_IS_ON(3)) {
    auto print_before = [](mlir::Pass *, mlir::Operation *) { return true; };
    auto print_after = [](mlir::Pass *, mlir::Operation *) { return true; };
    pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/true);
  }

  // TODO(necula): we should not need the inliner.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createChloRecomposeOpsPass());
  pm.addPass(mlir::stablehlo_ext::createStablehloRefineShapesPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<CheckShapeAssertionsPass>(enable_shape_assertions));
  if (!mlir::succeeded(pm.run(module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module shape refinement failed: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  return absl::OkStatus();
}

absl::Status RefinePolymorphicShapes(llvm::StringRef module_str,
                                     llvm::raw_ostream &os,
                                     bool enable_shape_assertions,
                                     bool validate_static_shapes,
                                     bool enable_shardy) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  context.loadDialect<mlir::sdy::SdyDialect>();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(module_str.data(), module_str.size()), &context);
  if (!module) {
    return absl::InvalidArgumentError("Cannot parse module.");
  }
  if (enable_shardy) {
    mlir::PassManager pm(module.get()->getName(),
                         mlir::OpPassManager::Nesting::Implicit);
    // NOTE: JAX shape refinement has `@shape_assertion` custom calls that
    // require constant folding. As such, we cannot import constants here just
    // yet. We have to delay it until after shape refinement.
    xla::sdy::addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/false);
    mlir::BaseScopedDiagnosticHandler diag_handler(module.get()->getContext());
    if (mlir::failed(pm.run(*module))) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error importing Sdy dialect: ",
                       diag_handler.ConsumeStatus().ToString()));
    }
  }

  TF_RETURN_IF_ERROR(RefinePolymorphicShapes(*module, enable_shape_assertions));
  if (validate_static_shapes) TF_RETURN_IF_ERROR(ValidateStaticShapes(*module));
  if (mlir::failed(mlir::writeBytecodeToFile(*module, os))) {
    return absl::InternalError("Cannot serialize module.");
  }
  if (enable_shardy) {
    mlir::PassManager pm(module.get()->getName(),
                         mlir::OpPassManager::Nesting::Implicit);
    pm.addNestedPass<mlir::func::FuncOp>(xla::sdy::createImportConstantsPass());
    mlir::BaseScopedDiagnosticHandler diag_handler(module.get()->getContext());
    if (mlir::failed(pm.run(*module))) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error importing Sdy constants: ",
                       diag_handler.ConsumeStatus().ToString()));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateStaticShapes(mlir::ModuleOp module) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module->getContext());
  bool moduleHasDynamicShapes = false;
  bool moduleHasShapeAssertions = false;

  module->walk([&](mlir::Operation *op) {
    // It's sufficient to only check results because operands either come from
    // results or from block arguments which are checked below.
    auto hasDynamicShape = [](mlir::Value value) {
      auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(value.getType());
      return shaped_type ? !shaped_type.hasStaticShape() : false;
    };
    bool opHasDynamicShapes = false;
    opHasDynamicShapes |= llvm::any_of(op->getResults(), hasDynamicShape);
    for (mlir::Region &region : op->getRegions()) {
      opHasDynamicShapes |=
          llvm::any_of(region.getArguments(), hasDynamicShape);
    }
    if (opHasDynamicShapes) {
      moduleHasDynamicShapes = true;
      op->emitOpError() << "has dynamic shapes";
    }

    auto customCall = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (customCall && customCall.getCallTargetName() == shapeAssertionName) {
      moduleHasShapeAssertions = true;
      op->emitOpError() << "has residual shape assertions";
    }
  });

  if (moduleHasDynamicShapes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module has dynamic shapes: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  if (moduleHasShapeAssertions) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module has residual shape assertions: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  return absl::OkStatus();
}

}  // namespace xla
