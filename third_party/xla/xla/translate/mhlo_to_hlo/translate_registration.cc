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

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/translate/mhlo_to_hlo/translate.h"

namespace {
// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_use_tuple_arg(
    "emit-use-tuple-args",
    llvm::cl::desc(
        "Emit HLO modules using tuples as args for the entry computation"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_return_tuple(
    "emit-return-tuple",
    llvm::cl::desc("Emit HLO modules with entry computations returning tuple"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> with_layouts(
    "with-layouts",
    llvm::cl::desc("Propagate layouts when translating MHLO->XLA HLO"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_layouts(
    "print-layouts", llvm::cl::desc("Print layouts in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_large_constants(
    "print-large-constants",
    llvm::cl::desc("Print large constants in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_sugar(
    "print-sugar",
    llvm::cl::desc(
        "Print async ops using syntactic sugar in the generated HLO text"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> via_builder(
    "via-builder", llvm::cl::desc("Translate MHLO->XLA HLO via XLA Builder"),
    llvm::cl::init(false));
}  // namespace

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
