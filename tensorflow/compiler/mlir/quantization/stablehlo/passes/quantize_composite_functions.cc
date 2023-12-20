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
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/uniform_quantized_types.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "quantize-composite-functions"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_QUANTIZECOMPOSITEFUNCTIONSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using QuantMethod = tensorflow::quantization::QuantizationMethod::PresetMethod;
using ::tensorflow::quantization::RunPassesOnModuleOp;

constexpr StringRef kCompositeFuncPrefix = "composite_";
constexpr StringRef kQuantizedFuncPrefix = "quantized_";
constexpr StringRef kEntryFuncAttrName = "_entry_function";

class QuantizeCompositeFunctionsPass
    : public impl::QuantizeCompositeFunctionsPassBase<
          QuantizeCompositeFunctionsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeCompositeFunctionsPass)

  using impl::QuantizeCompositeFunctionsPassBase<
      QuantizeCompositeFunctionsPass>::QuantizeCompositeFunctionsPassBase;

 private:
  void runOnOperation() override;
};

void QuantizeCompositeFunctionsPass::runOnOperation() {
  MLIRContext& ctx = getContext();

  QuantizationSpecs quant_specs;
  quant_specs.inference_type = tensorflow::DT_QINT8;

  PassManager pm(&ctx);
  // Intermediate output from QuantizePass will have quantized ops
  // (XlaCallModuleOps) with quantized input and output types, which are not
  // allowed in the TF dialect.
  pm.enableVerifier(false);

  pm.addNestedPass<func::FuncOp>(CreatePrepareQuantizePass());
  // QuantizePass modifies FuncOps referenced outside of its given scope
  // and therefore requires a module-level context.
  pm.addPass(CreateQuantizePass(quant_specs));
  pm.addNestedPass<func::FuncOp>(createPostQuantizePass());

  ModuleOp module_op = getOperation();
  if (const absl::Status pm_run_status =
          RunPassesOnModuleOp(mlir_dump_file_name_, pm, module_op);
      !pm_run_status.ok()) {
    signalPassFailure();
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo
