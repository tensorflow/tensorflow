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

#include "mlir/InitAllPasses.h"               // from @llvm-project
#include "mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"           // from @llvm-project
#include "tensorflow//compiler/mlir/emitc/emitc_passes.h"
#include "tensorflow/compiler/mlir/emitc/transforms/passes.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/register_common_dialects.h"

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::emitc::registerAddReflectionMapPipeline();

  mlir::DialectRegistry registry;
  mlir::RegisterCommonToolingDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "TensorFlow pass driver\n", registry));
}
