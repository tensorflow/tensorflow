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

#include <memory>

#include <gtest/gtest.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using mlir::mhlo::test::GetMlirModuleFromString;

class VerifyClusteringPassTest : public testing::Test {
 protected:
  void CreateModule(const char* module_string) {
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            GetMlirModuleFromString(module_string, &context_));
    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addNestedPass<mlir::func::FuncOp>(CreateVerifyClusteringPass());
  }

  mlir::LogicalResult Run() { return pm_->run(module_.get()); }

 private:
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(VerifyClusteringPassTest, OnlyTfFunctionalPasses) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      return %0 : tensor<1xi32>
    }
  })";
  CreateModule(kMlirModuleStr);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());
}

TEST_F(VerifyClusteringPassTest, NotTfFunctionalFails) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<3x32x32x3xf32> {
      %0 = mhlo.constant dense<2.550000e+02> : tensor<3x32x32x3xf32>
      return %0 : tensor<3x32x32x3xf32>
    }
  })";
  CreateModule(kMlirModuleStr);

  auto result = Run();

  EXPECT_TRUE(result.failed());
}

}  // namespace
}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
