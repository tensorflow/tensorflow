/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/translate/mhlo_to_hlo/translate_registration.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "xla/hlo/translate/mhlo_to_hlo/translate.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

static mlir::LogicalResult MlirHloToHloTranslate(mlir::ModuleOp module,
                                                 llvm::raw_ostream& output) {
  return xla::MlirHloToHloTranslateFunction(module, output, emit_return_tuple,
                                            emit_use_tuple_arg);
}

static mlir::LogicalResult MlirHloToHloTextTranslate(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  return xla::MlirHloToHloTextTranslateFunction(
      module, output, emit_return_tuple, emit_use_tuple_arg, print_layouts,
      print_large_constants, print_sugar, via_builder, with_layouts);
}

static void RegisterInputDialects(mlir::DialectRegistry& registry) {
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();
}

static mlir::TranslateFromMLIRRegistration MlirHloToHloTranslateRegistration(
    "mlir-hlo-to-hlo", "mlir-hlo-to-hlo", MlirHloToHloTranslate,
    RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration
    MlirHloToHloTextTranslateRegistration("mlir-hlo-to-hlo-text",
                                          "mlir-hlo-to-hlo-text",
                                          MlirHloToHloTextTranslate,
                                          RegisterInputDialects);
