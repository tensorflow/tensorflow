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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "xla/translate/hlo_to_mhlo/translate.h"

namespace {
// NOLINTNEXTLINE
llvm::cl::opt<bool> import_all_computations(
    "hlo-import-all-computations",
    llvm::cl::desc("Enable importing unreachable computations."));

// NOLINTNEXTLINE
llvm::cl::opt<bool> flatten_computation_args_result(
    "hlo-flatten-computation-args-result",
    llvm::cl::desc("Enable flattening computation arguments and results."));
}  // namespace

static mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloToMlirHloTranslateFunction(
      input, context, import_all_computations, flatten_computation_args_result);
}

static mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloTextToMlirHloTranslateFunction(
      input, context, import_all_computations, flatten_computation_args_result);
}

static mlir::TranslateToMLIRRegistration HloToMlirHloTranslateRegistration(
    "hlo-to-mlir-hlo", "hlo-to-mlir-hlo", HloToMlirHloTranslate);

static mlir::TranslateToMLIRRegistration HloTextToMlirHloTranslateRegistration(
    "hlo-text-to-mlir-hlo", "hlo-text-to-mlir-hlo", HloTextToMlirHloTranslate);
