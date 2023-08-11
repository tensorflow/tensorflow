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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/tsl/platform/init_main.h"

using namespace mlir;  // NOLINT

int main(int argc, char **argv) {
  // Initialize the process. On OSS this is a no-op.
  // Note: we do not parse any flags here; all flags are parsed by
  // `MlirOptMain` below.
  int32_t argc1 = 1;
  tsl::port::InitMain("Xla Gpu Pass Driver", &argc1, &argv);

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                  mhlo::MhloDialect, bufferization::BufferizationDialect,
                  lmhlo::LmhloDialect, lmhlo_gpu::LmhloGpuDialect,
                  xla::gpu::XlaGpuDialect>();

  // General MLIR passes like `-cse` and `-canonicalize`.
  registerTransformsPasses();

  // Lowering to XLA:GPU runtime.
  xla::gpu::registerGpu2Pases();

  return failed(MlirOptMain(argc, argv, "Xla Gpu Pass Driver\n", registry));
}
