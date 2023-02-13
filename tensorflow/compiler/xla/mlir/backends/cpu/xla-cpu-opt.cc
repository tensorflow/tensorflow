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

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/transforms/test_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/thlo/IR/thlo_ops.h"

int main(int argc, char **argv) {
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::gml_st::registerGmlStPasses();
  mlir::gml_st::registerGmlStTestPasses();
  mlir::bufferization::registerBufferizationPasses();

  mlir::DialectRegistry registry;
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::lmhlo::LmhloDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::gml_st::GmlStDialect, mlir::thlo::THLODialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::vector::VectorDialect, mlir::xla_cpu::XlaCpuDialect>();

  xla::cpu::registerCpuTransformsPasses();

  return failed(MlirOptMain(argc, argv, "Xla Cpu Pass Driver\n", registry));
}
