/* Copyright 2024 The OpenXLA Authors.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "third_party/triton/bin/RegisterTritonDialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  registerTritonDialects(registry);  // This registers all passes as well.
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::triton::xla::XlaTritonDialect, xla::XlaDialect>();
  mlir::triton::xla::registerTritonXlaTransformsPasses();
  xla::emitters::registerTransformsPasses();
  xla::gpu::registerGpuFusionTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "xla-opt modular optimizer driver\n", registry));
}
