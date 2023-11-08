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

#include "tensorflow/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"

#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::UnknownLoc;
using mlir::chlo::ChloDialect;
using mlir::TF::TensorFlowDialect;
using tensorflow::tf2xla::internal::IsInBridgeAcceptableDialects;

class SharedUtilsTest : public ::testing::Test {};

TEST_F(SharedUtilsTest, IsInFunctionalDialectPasses) {
  MLIRContext context;
  context.loadDialect<TensorFlowDialect>();
  OpBuilder opBuilder(&context);
  OperationState state(UnknownLoc::get(opBuilder.getContext()),
                       /*OperationName=*/"tf.Const");
  mlir::Operation* op = Operation::create(state);

  bool result = IsInBridgeAcceptableDialects(op);

  EXPECT_TRUE(result);
  op->destroy();
}

TEST_F(SharedUtilsTest, IsInFunctionalDialectFails) {
  MLIRContext context;
  context.loadDialect<ChloDialect>();
  OpBuilder opBuilder(&context);
  OperationState state(UnknownLoc::get(opBuilder.getContext()),
                       /*OperationName=*/"chlo.broadcast_add");
  Operation* op = Operation::create(state);

  bool result = IsInBridgeAcceptableDialects(op);

  EXPECT_FALSE(result);
  op->destroy();
}

}  // namespace
}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
