/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_registry.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/transforms/pass_registration.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerCanonicalizerPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerViewOpGraphPass();
  mlir::registerSymbolDCEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::tfg::registerTFGraphPasses();
  registry.insert<mlir::tfg::TFGraphDialect, mlir::tf_type::TFTypeDialect>();
  // Inject the op registry.
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::tfg::TFGraphDialect *dialect) {
        dialect->addInterfaces<mlir::tfg::TensorFlowOpRegistryInterface>();
      });
  return failed(
      mlir::MlirOptMain(argc, argv, "TFGraph Transforms Driver", registry));
}
