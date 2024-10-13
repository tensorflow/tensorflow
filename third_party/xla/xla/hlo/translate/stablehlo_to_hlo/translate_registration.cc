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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/dialect/Register.h"
#include "xla/hlo/translate/stablehlo_to_hlo/translate.h"

// The following symbols are defined in
// tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo/translate_registration.h
extern llvm::cl::opt<bool> emit_use_tuple_arg;
extern llvm::cl::opt<bool> emit_return_tuple;
extern llvm::cl::opt<bool> with_layouts;
extern llvm::cl::opt<bool> print_layouts;
extern llvm::cl::opt<bool> print_large_constants;
extern llvm::cl::opt<bool> print_sugar;
extern llvm::cl::opt<bool> via_builder;

static mlir::LogicalResult StablehloToHloTranslate(mlir::ModuleOp module,
                                                   llvm::raw_ostream& output) {
  return xla::StablehloToHloTranslateFunction(module, output, emit_return_tuple,
                                              emit_use_tuple_arg);
}

static mlir::LogicalResult StablehloToHloTextTranslate(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  return xla::StablehloToHloTextTranslateFunction(
      module, output, emit_return_tuple, emit_use_tuple_arg, print_layouts,
      print_large_constants, print_sugar, via_builder, with_layouts);
}

static void RegisterInputDialects(mlir::DialectRegistry& registry) {
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();
}

static mlir::TranslateFromMLIRRegistration StablehloToHloTranslateRegistration(
    "stablehlo-to-hlo", "stablehlo-to-hlo", StablehloToHloTranslate,
    RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration
    StablehloToHloTextTranslateRegistration("stablehlo-to-hlo-text",
                                            "stablehlo-to-hlo-text",
                                            StablehloToHloTextTranslate,
                                            RegisterInputDialects);
