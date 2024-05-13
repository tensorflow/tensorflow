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

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/odml_converter/passes.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

const char* art = R"(
  ___  ___  __  __ _       ___                     _
 / _ \|   \|  \/  | |     / __|___ _ ___ _____ _ _| |_ ___ _ _
| (_) | |) | |\/| | |__  | (__/ _ \ ' \ V / -_) '_|  _/ -_) '_|
 \___/|___/|_|  |_|____|  \___\___/_||_\_/\___|_|  \__\___|_|
)";

int main(int argc, char* argv[]) {
  tensorflow::InitMlir y(&argc, &argv);
  llvm::errs() << art << "\n";

  mlir::odml::registerODMLConverterPasses();
  mlir::odml::registerLegalizeStablehloToVhloPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                  mlir::TFL::TFLDialect, mlir::arith::ArithDialect,
                  mlir::TF::TensorFlowDialect, mlir::chlo::ChloDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "ODML Converter Driver\n", registry));
}
