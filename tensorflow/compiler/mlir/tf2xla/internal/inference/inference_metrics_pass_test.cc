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
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/inference/inference_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace tf2xla {
namespace internal {
namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::mhlo::test::GetMlirModuleFromString;
using ::tensorflow::monitoring::testing::CellReader;

static constexpr char kHasTpuPartitionedCallStreamzName[] =
    "/tensorflow/core/tf2xla/internal/inference/tpu_partitioned_call";

class InferenceMetricsPassTest : public ::testing::Test {
 protected:
  void CreateModule(const char* module_string) {
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            GetMlirModuleFromString(module_string, &context_));

    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addPass(CreateInferenceMetricsPass());
  }
  mlir::LogicalResult Run() { return pm_->run(module_.get()); }

 private:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(InferenceMetricsPassTest, RecordsTrueForTPUPartitionedCallOp) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @tpu_partitioned_call_func(%arg0: tensor<?xi32>) -> (tensor<?xi32>) {
      func.return %arg0 : tensor<?xi32>
    }

    func.func @main(%arg0: tensor<20xi32>, %arg1: tensor<?xi32>) -> tensor<*xi32> {
      %2 = "tf.TPUPartitionedCall"(%arg0, %arg1) {f = @tpu_partitioned_call_func} : (tensor<20xi32>, tensor<?xi32>) -> tensor<*xi32>
      func.return %2 : tensor<*xi32>
    }
  })";

  CellReader<int64_t> error(kHasTpuPartitionedCallStreamzName);
  CreateModule(kMlirModuleStr);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());

  EXPECT_EQ(error.Delta("true"), 1);
  EXPECT_EQ(error.Delta("false"), 0);
}

TEST_F(InferenceMetricsPassTest, RecordsFalseForNonTPUPartitionedCallOp) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.BadValue"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  CellReader<int64_t> error(kHasTpuPartitionedCallStreamzName);
  CreateModule(kMlirModuleStr);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(error.Delta("false"), 1);
  EXPECT_EQ(error.Delta("true"), 0);
}

}  // namespace
}  // namespace internal
}  // namespace tf2xla
}  // namespace mlir
