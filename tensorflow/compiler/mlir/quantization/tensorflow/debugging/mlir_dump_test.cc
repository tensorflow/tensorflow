/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace tensorflow {
namespace quantization {
namespace mlir_dump_test {

class NoOpPass
    : public mlir::PassWrapper<NoOpPass, mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoOpPass)

  NoOpPass() = default;

  llvm::StringRef getArgument() const final { return "no-op-pass"; }

  void runOnOperation() override {
    // Noop pass does nothing on the operation.
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateNoOpPass() {
  return std::make_unique<NoOpPass>();
}

class ParentPass
    : public mlir::PassWrapper<ParentPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParentPass)

  ParentPass() = default;

  llvm::StringRef getArgument() const final { return "parent-pass"; }

  void runOnOperation() override {
    mlir::MLIRContext* ctx = &getContext();
    mlir::ModuleOp module_op = getOperation();
    mlir::PassManager pm(ctx);

    pm.addPass(CreateNoOpPass());

    EnableIrPrinting(pm, "dump2");

    if (failed(pm.run(module_op))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateParentPass() {
  return std::make_unique<ParentPass>();
}

}  // namespace mlir_dump_test

namespace {

using namespace tensorflow::quantization::mlir_dump_test;

class EnableIrPrintingTest : public ::testing::Test {
 protected:
  EnableIrPrintingTest() : env_(tsl::Env::Default()) {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&test_dir_)) {
      test_dir_ = tsl::testing::TmpDir();
    }
  }

  void SetUp() override {
    tsl::setenv("TF_QUANT_MLIR_DUMP_PREFIX", test_dir_.c_str(), 1);

    mlir::DialectRegistry dialects;
    dialects.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::stablehlo::StablehloDialect>();
    ctx_ = std::make_unique<mlir::MLIRContext>(dialects);
    ctx_->loadAllAvailableDialects();
  }

  void TearDown() override {
    // Delete files in the test directory.
    std::vector<std::string> files;
    TF_ASSERT_OK(
        env_->GetMatchingPaths(tsl::io::JoinPath(test_dir_, "*"), &files));
    for (const std::string& file : files) {
      TF_ASSERT_OK(env_->DeleteFile(file));
    }
  }

  tsl::Env* env_;
  std::string test_dir_;
  std::unique_ptr<mlir::MLIRContext> ctx_;
};

TEST_F(EnableIrPrintingTest, PassSuccessfullyRuns) {
  mlir::PassManager pm = {ctx_.get()};
  pm.addPass(CreateNoOpPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  EnableIrPrinting(pm, "dump");

  constexpr absl::string_view program = R"mlir(
module{
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    return %arg0 : tensor<10xf32>
  }
  func.func @func1(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
    %1 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
})mlir";
  auto module_op = mlir::parseSourceString<mlir::ModuleOp>(program, ctx_.get());

  const mlir::LogicalResult result = pm.run(module_op.get());
  EXPECT_FALSE(failed(result));

  TF_EXPECT_OK(tsl::Env::Default()->FileExists(
      tsl::io::JoinPath(test_dir_,
                        "dump_0001_tensorflow::quantization::mlir_dump_test"
                        "::NoOpPass_before.mlir")));
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(
      tsl::io::JoinPath(test_dir_, "dump_0002_Canonicalizer_before.mlir")));
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(
      tsl::io::JoinPath(test_dir_, "dump_0002_Canonicalizer_after.mlir")));
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(
      tsl::io::JoinPath(test_dir_, "dump_0003_Canonicalizer_before.mlir")));
}

TEST_F(EnableIrPrintingTest, NestedPassSuccessfullyRuns) {
  mlir::MLIRContext ctx{};

  mlir::PassManager pm = {&ctx};
  pm.addPass(CreateParentPass());

  EnableIrPrinting(pm, "dump");

  mlir::OpBuilder builder(&ctx);
  auto module_op = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  // Destroy by calling destroy() to avoid memory leak since it is allocated
  // with malloc().
  const absl::Cleanup module_op_cleanup = [module_op] { module_op->destroy(); };

  const mlir::LogicalResult result = pm.run(module_op);
  EXPECT_FALSE(failed(result));

  TF_EXPECT_OK(tsl::Env::Default()->FileExists(
      tsl::io::JoinPath(test_dir_,
                        "dump_0001_tensorflow::quantization::mlir_dump_test"
                        "::ParentPass_before.mlir")));
  TF_EXPECT_OK(tsl::Env::Default()->FileExists(
      tsl::io::JoinPath(test_dir_,
                        "dump2_0001_tensorflow::quantization::mlir_dump_test"
                        "::NoOpPass_before.mlir")));
}
}  // namespace
}  // namespace quantization
}  // namespace tensorflow
