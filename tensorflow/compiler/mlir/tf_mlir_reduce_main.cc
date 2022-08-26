/* Copyright 2021 Google Inc. All Rights Reserved.

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

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Reducer/ReductionPatternInterface.h"  // from @llvm-project
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/stablehlo/stablehlo/dialect/Register.h"

namespace {

#include "tensorflow/compiler/mlir/tensorflow/transforms/reducer/tf_reduce_patterns.inc"

struct TFReductionPatternInterface
    : public mlir::DialectReductionPatternInterface {
 public:
  explicit TFReductionPatternInterface(mlir::Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(
      mlir::RewritePatternSet &patterns) const final {
    populateWithGenerated(patterns);
  }
};

}  // namespace

int main(int argc, char *argv[]) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::TFDevice::registerTensorFlowDevicePasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  // These are in compiler/mlir/xla and not part of the above MHLO passes.
  mlir::mhlo::registerTfXlaPasses();
  mlir::mhlo::registerXlaPasses();
  mlir::mhlo::registerLegalizeTFPass();
  mlir::mhlo::registerLegalizeTFControlFlowPass();
  mlir::mhlo::registerLegalizeTfTypesPassPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::TF::TensorFlowDialect *dialect) {
        dialect->addInterfaces<TFReductionPatternInterface>();
      });
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();

  mlir::MLIRContext context(registry);

  return failed(mlirReduceMain(argc, argv, context));
}
