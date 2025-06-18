/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/UB/IR/UBOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/tf_quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir::tf_quant::stablehlo::testing {

#define GEN_PASS_DEF_TESTTFTOSTABLEHLOPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/testing/tf_passes.h.inc"

namespace {

using ::tensorflow::quantization::AddTFToStablehloPasses;
using ::tensorflow::quantization::RunPassesOnModuleOp;

class TestTFToStablehloPass
    : public impl::TestTFToStablehloPassBase<TestTFToStablehloPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTFToStablehloPass)

 private:
  void runOnOperation() override;
};

void TestTFToStablehloPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = &getContext();
  mlir::PassManager pm(ctx);

  AddTFToStablehloPasses(pm);
  if (!RunPassesOnModuleOp(
           /*mlir_dump_file_name=*/"test_tf_to_stablehlo_pass", pm, module_op)
           .ok()) {
    return signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::tf_quant::stablehlo::testing
