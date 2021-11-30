/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Support/MlirOptMain.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.h"
#include "tensorflow/compiler/mlir/tfjs/transforms/passes.h"

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::tfjs::registerTFJSPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::TF::TensorFlowDialect>();
  registry.insert<mlir::tfjs::TFJSDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "TF JS pass driver\n", registry));
}
