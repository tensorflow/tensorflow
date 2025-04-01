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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/platform/test.h"

namespace mlir::quant {

using ::testing::Test;

class QuantizationTestBase : public Test {
 protected:
  QuantizationTestBase()
      : ctx_(stablehlo::CreateMlirContextForQuantization()),
        builder_(ctx_.get()) {
    ctx_->loadDialect<
        arith::ArithDialect, mlir::stablehlo::StablehloDialect,
        func::FuncDialect, TF::TensorFlowDialect, TFL::TensorFlowLiteDialect,
        tf_saved_model::TensorFlowSavedModelDialect,
        tf_executor::TensorFlowExecutorDialect, quant::QuantDialect,
        quantfork::QuantizationForkDialect, ir::TFQuantDialect>();
  }

  // Parses `module_op_str` to create a `ModuleOp`.
  OwningOpRef<ModuleOp> ParseModuleOpString(
      const absl::string_view module_op_str) {
    return parseSourceString<ModuleOp>(module_op_str, ctx_.get());
  }

  // Convenience function that returns the first operation of type `OpT` from
  // the `@main` function in `module_op`. Useful when testing with a text
  // representation of a `ModuleOp` containing a single function `@main`.
  // Returns `failure` iff there is no `@main` or no such operation is found in
  // `@main`.
  template <typename OpT>
  FailureOr<OpT> FindFirstOpFromMainFunc(ModuleOp module_op) {
    func::FuncOp main_func_op = FindMainFuncOp(module_op);
    if (main_func_op == nullptr) return failure();

    auto ops = main_func_op.getOps<OpT>();
    if (ops.empty()) return failure();

    return *ops.begin();
  }

  std::unique_ptr<MLIRContext> ctx_;
  OpBuilder builder_;
};

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_
