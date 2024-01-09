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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/platform/test.h"

namespace mlir::quant {

using ::testing::Test;

class QuantizationTestBase : public Test {
 protected:
  QuantizationTestBase() {
    ctx_.loadDialect<arith::ArithDialect, mlir::stablehlo::StablehloDialect,
                     func::FuncDialect, TF::TensorFlowDialect,
                     tf_saved_model::TensorFlowSavedModelDialect,
                     tf_executor::TensorFlowExecutorDialect,
                     quant::QuantizationDialect,
                     quantfork::QuantizationForkDialect>();
  }

  // Parses `module_op_str` to create a `ModuleOp`. Checks whether the created
  // module op is valid.
  OwningOpRef<ModuleOp> ParseModuleOpString(
      const absl::string_view module_op_str) {
    auto module_op_ref = parseSourceString<ModuleOp>(module_op_str, &ctx_);
    EXPECT_TRUE(module_op_ref);
    return module_op_ref;
  }

  // Gets the function with the given name from the module.
  func::FuncOp GetFunctionFromModule(ModuleOp module,
                                     absl::string_view function_name) {
    SymbolTable symbol_table(module);
    return symbol_table.lookup<func::FuncOp>(function_name);
  }

  // Returns the first operation with the given type in the function.
  template <typename OpType>
  OpType FindOperationOfType(func::FuncOp function) {
    for (auto op : function.getBody().getOps<OpType>()) {
      return op;
    }
    return nullptr;
  }

  mlir::MLIRContext ctx_{};
  OpBuilder builder_{&ctx_};
};

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_TEST_BASE_H_
