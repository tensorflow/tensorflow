/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"

#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
namespace TFL {
namespace {
using stream_executor::port::StatusOr;

// MockSuccessPass reports errors but doesn't fail.
class MockSuccessPass
    : public PassWrapper<MockSuccessPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MockSuccessPass)

  explicit MockSuccessPass() {}

 private:
  void runOnOperation() override {
    getOperation().walk([](Operation* nestedOp) {
      nestedOp->emitError()
          << "Error at " << nestedOp->getName().getStringRef().str() << " op";
    });
  };
};

// MockFailurePass reports errors and fails.
class MockFailurePass
    : public PassWrapper<MockFailurePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MockFailurePass)

  explicit MockFailurePass() {}

 private:
  void runOnOperation() override {
    getOperation().walk([](Operation* nestedOp) {
      if (nestedOp->getName().getStringRef().str().rfind("tf.") != -1) {
        AttachErrorCode(
            nestedOp->emitError()
                << "Failed at " << nestedOp->getName().getStringRef().str()
                << " op",
            tflite::metrics::ConverterErrorData::ERROR_NEEDS_FLEX_OPS);
      }
    });
    signalPassFailure();
  };
};

StatusOr<OwningOpRef<mlir::ModuleOp>> LoadModule(MLIRContext* context,
                                                 const std::string& file_name) {
  std::string error_message;
  auto file = openInputFile(file_name, &error_message);
  if (!file) {
    return tensorflow::errors::InvalidArgument("fail to open input file");
  }

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningOpRef<mlir::ModuleOp>(
      parseSourceFile<mlir::ModuleOp>(source_mgr, context));
}

TEST(ErrorCollectorTest, TessSuccessPass) {
  std::string input_file = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/metrics/testdata/strided_slice.mlir");
  MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  context.enableMultithreading();

  auto module = LoadModule(&context, input_file);
  EXPECT_EQ(module.ok(), true);

  PassManager pm(&context, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<MockSuccessPass>());

  pm.addInstrumentation(
      std::make_unique<ErrorCollectorInstrumentation>(&context));
  EXPECT_EQ(succeeded(pm.run(module.ValueOrDie().get())), true);

  auto collected_errors =
      ErrorCollector::GetErrorCollector()->CollectedErrors();
  EXPECT_EQ(collected_errors.size(), 0);
}

TEST(ErrorCollectorTest, TessFailurePass) {
  using tflite::metrics::ConverterErrorData;
  MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  const std::string input_file =
      "tensorflow/compiler/mlir/lite/metrics/testdata/strided_slice.mlir";
  auto input_file_id = StringAttr::get(&context, input_file);

  context.allowUnregisteredDialects();
  context.enableMultithreading();

  auto module =
      LoadModule(&context, tensorflow::GetDataDependencyFilepath(input_file));
  EXPECT_EQ(module.ok(), true);

  PassManager pm(&context, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<MockSuccessPass>());
  pm.addPass(std::make_unique<MockFailurePass>());

  pm.addInstrumentation(
      std::make_unique<ErrorCollectorInstrumentation>(&context));
  EXPECT_EQ(succeeded(pm.run(module.ValueOrDie().get())), false);

  auto collected_errors =
      ErrorCollector::GetErrorCollector()->CollectedErrors();

  EXPECT_EQ(collected_errors.size(), 3);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass",
                "Failed at tf.Const op\nsee current operation: %0 = "
                "\"tf.Const\"() {value = dense<1> : tensor<4xi32>} : () -> "
                "tensor<4xi32>\nError code: ERROR_NEEDS_FLEX_OPS",
                ConverterErrorData::ERROR_NEEDS_FLEX_OPS, "tf.Const",
                mlir::FileLineColLoc::get(input_file_id, 2, 9))),
            1);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass",
                "Failed at tf.Const op\nsee current operation: %1 = "
                "\"tf.Const\"() {value = dense<0> : tensor<4xi32>} : () -> "
                "tensor<4xi32>\nError code: ERROR_NEEDS_FLEX_OPS",
                ConverterErrorData::ERROR_NEEDS_FLEX_OPS, "tf.Const",
                mlir::FileLineColLoc::get(input_file_id, 2, 9))),
            1);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass",
                "Failed at tf.StridedSlice op\nsee current operation: %2 = "
                "\"tf.StridedSlice\"(%arg0, %1, %1, %0) {begin_mask = 11 : "
                "i64, device = \"\", ellipsis_mask = 0 : i64, end_mask = 11 : "
                "i64, new_axis_mask = 4 : i64, shrink_axis_mask = 0 : i64} : "
                "(tensor<*xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) "
                "-> tensor<*xf32>\nError code: ERROR_NEEDS_FLEX_OPS",
                ConverterErrorData::ERROR_NEEDS_FLEX_OPS, "tf.StridedSlice",
                mlir::FileLineColLoc::get(input_file_id, 4, 10))),
            1);

  // Check the location information.
  std::vector<std::string> locations;
  for (const auto& error : collected_errors) {
    EXPECT_TRUE(error.has_location());
    locations.push_back(error.location().DebugString());
  }

  EXPECT_THAT(locations, Each(testing::HasSubstr("CALLSITELOC")));
  EXPECT_THAT(locations, Each(testing::HasSubstr(input_file)));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("line: 2")));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("column: 9")));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("line: 4")));
  EXPECT_THAT(locations, Contains(testing::HasSubstr("column: 10")));
}
}  // namespace
}  // namespace TFL
}  // namespace mlir
