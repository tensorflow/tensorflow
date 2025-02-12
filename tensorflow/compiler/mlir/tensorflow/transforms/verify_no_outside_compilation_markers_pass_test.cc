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
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace mlir {
namespace TFDevice {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::mhlo::test::GetMlirModuleFromString;

class VerifyNoOutsideCompilationMarkersPassTest : public ::testing::Test {
 protected:
  void CreateModule(const char* module_string) {
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            GetMlirModuleFromString(module_string, &context_));

    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addNestedPass<func::FuncOp>(
        CreateVerifyNoOutsideCompilationMarkersPass());
  }

  mlir::LogicalResult Run() { return pm_->run(module_.get()); }

 private:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(VerifyNoOutsideCompilationMarkersPassTest, PassesValidOps) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

  CreateModule(kMlirModuleStr);
  auto result = Run();

  EXPECT_TRUE(result.succeeded());
}

TEST_F(VerifyNoOutsideCompilationMarkersPassTest,
       FailsXlaOutsideCompilationMarkers) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> () {
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      func.return
    }
  })";

  CreateModule(kMlirModuleStr);
  auto result = Run();

  EXPECT_TRUE(result.failed());
}

TEST_F(VerifyNoOutsideCompilationMarkersPassTest,
       FailsWithLaunchOpsInsideCluster) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> () {
     %0 = "tf_device.cluster"() ({
        "tf_device.launch"() ({
          "tf.B"() : () -> ()
          tf_device.return // end device
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()

        tf_device.return // end cluster
      }) {cluster_attr = "cluster_attr"} : () -> tensor<*xi32>
      func.return
    }
  })";

  CreateModule(kMlirModuleStr);
  auto result = Run();

  EXPECT_TRUE(result.failed());
}

TEST_F(VerifyNoOutsideCompilationMarkersPassTest,
       PassesWithLaunchOpsOutsideCluster) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> () {
      "tf_device.launch"() ({
        "tf.B"() : () -> ()
        tf_device.return
      }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
      func.return
    }
  })";

  CreateModule(kMlirModuleStr);
  auto result = Run();

  EXPECT_TRUE(result.succeeded());
}

}  // namespace TFDevice
}  // namespace mlir
