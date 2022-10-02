/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir/transforms/cpu/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/register_gml_st_passes.h"

int main(int argc, char **argv) {
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::gml_st::registerAllGmlStPasses();

  mlir::DialectRegistry registry;
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::lmhlo::LmhloDialect,
                  mlir::gml_st::GmlStDialect, mlir::thlo::THLODialect>();

  xla::cpu::registerCpuTransformsPasses();

  return failed(MlirOptMain(argc, argv, "Xla Cpu Pass Driver\n", registry));
}
