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
#include "tensorflow/compiler/mlir/lite/metrics/error_collector.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/resource_loader.h"
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

StatusOr<OwningModuleRef> LoadModule(MLIRContext* context,
                                     const std::string& file_name) {
  std::string error_message;
  auto file = openInputFile(file_name, &error_message);
  if (!file) {
    return tensorflow::errors::InvalidArgument("fail to open input file");
  }

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningModuleRef(parseSourceFile(source_mgr, context));
}

TEST(ErrorCollectorTest, TessSuccessPass) {
  std::string input_file = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/metrics/testdata/"
      "strided_slice.mlir");
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.enableMultithreading();

  auto module = LoadModule(&context, input_file);
  EXPECT_EQ(module.ok(), true);

  PassManager pm(&context, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<MockSuccessPass>());

  pm.addInstrumentation(
      std::make_unique<ErrorCollectorInstrumentation>(&context));
  EXPECT_EQ(succeeded(pm.run(module.ValueOrDie().get())), true);

  auto collected_errors = GetErrorCollector()->CollectedErrors();
  EXPECT_EQ(collected_errors.size(), 0);
}

TEST(ErrorCollectorTest, TessFailurePass) {
  std::string input_file = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/metrics/testdata/"
      "strided_slice.mlir");
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.enableMultithreading();

  auto module = LoadModule(&context, input_file);
  EXPECT_EQ(module.ok(), true);

  PassManager pm(&context, OpPassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<MockSuccessPass>());
  pm.addPass(std::make_unique<MockFailurePass>());

  pm.addInstrumentation(
      std::make_unique<ErrorCollectorInstrumentation>(&context));
  EXPECT_EQ(succeeded(pm.run(module.ValueOrDie().get())), false);

  auto collected_errors = GetErrorCollector()->CollectedErrors();

  EXPECT_EQ(collected_errors.size(), 2);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass", "Failed at tf.Const op",
                tflite::metrics::ConverterErrorData::ERROR_NEEDS_FLEX_OPS,
                "tf.Const")),
            1);
  EXPECT_EQ(collected_errors.count(NewConverterErrorData(
                "MockFailurePass", "Failed at tf.StridedSlice op",
                tflite::metrics::ConverterErrorData::ERROR_NEEDS_FLEX_OPS,
                "tf.StridedSlice")),
            1);
}
}  // namespace
}  // namespace TFL
}  // namespace mlir
