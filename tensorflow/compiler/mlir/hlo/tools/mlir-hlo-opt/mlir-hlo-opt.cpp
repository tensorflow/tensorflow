/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/disc-ral/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/disc-ral/transforms/register_passes.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "mlir-hlo/Transforms/register_passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::disc_ral::registerAllDiscRalPasses();
  mlir::hlo::registerAllHloPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
  registry.insert<mlir::disc_ral::RalDialect>();

  return failed(mlir::MlirOptMain(argc, argv, "MLIR HLO pass driver\n",
                                  registry,
                                  /*preloadDialectsInContext=*/false));
}
