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

#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_chlo {

namespace {

struct TestChloLegalizeToHloPass
    : public PassWrapper<TestChloLegalizeToHloPass, FunctionPass> {
  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;

    conversionTarget.addIllegalDialect<XlaHloClientDialect>();
    // Consider the xla_hlo dialect legal for tests.
    conversionTarget.addLegalDialect<xla_hlo::XlaHloDialect>();
    // The conversion uses helpers from the Standard dialect.
    conversionTarget.addLegalDialect<mlir::StandardOpsDialect>();
    conversionTarget.addLegalDialect<mlir::shape::ShapeDialect>();

    PopulateLegalizeChloToHloPatterns(&getContext(), &conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla_chlo
}  // namespace mlir

static mlir::PassRegistration<mlir::xla_chlo::TestChloLegalizeToHloPass> pass(
    "test-xla-chlo-legalize-to-hlo",
    "Test pass for applying chlo -> hlo legalization patterns");
