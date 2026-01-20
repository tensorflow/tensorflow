/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/python/model_utils_core/pybind_api_internal.h"

#include <memory>

#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TFL {
namespace model_utils {

std::unique_ptr<mlir::MLIRContext> CreateIRContext() {
  auto context = std::make_unique<mlir::MLIRContext>();

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::quant::QuantDialect, mlir::TFL::TensorFlowLiteDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  context->disableMultithreading();
  context->allowUnregisteredDialects();
  return context;
}

absl::Status MlirVerify(mlir::Operation* op) {
  mlir::StatusScopedDiagnosticHandler diag_handler(op->getContext());
  if (mlir::failed(mlir::verify(op))) {
    return diag_handler.ConsumeStatus();
  }
  return absl::OkStatus();
}

}  // namespace model_utils
}  // namespace TFL
}  // namespace mlir
