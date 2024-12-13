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
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::mhlo::test::GetMlirModuleFromString;
using ::tensorflow::monitoring::testing::CellReader;

static constexpr char kFailedLegalizationStreamz[] =
    "/tensorflow/core/tf2xla/mlir_second_phase_failed_legalization_op_count";
static constexpr char kNonStaticOpStreamz[] =
    "/tensorflow/core/tf2xla/mlir_second_phase_non_static_op_count";
static constexpr char kNonStaticOpSkipStreamz[] =
    "/tensorflow/core/tf2xla/mlir_second_phase_non_static_op_skip_count";

class VerifyTfxlaLegalizationTest : public ::testing::Test {
 protected:
  void CreateModule(const char* module_string) {
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            GetMlirModuleFromString(module_string, &context_));

    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::CreateVerifyTFXLALegalizationPass(/*legalize_chlo=*/false));
  }
  mlir::LogicalResult Run() { return pm_->run(module_.get()); }

 private:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(VerifyTfxlaLegalizationTest, RecordsStreamzFailedVerification) {
  // Using a string constant here instead of testdata to make this compatible
  // with open source.
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.BadValue"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  CellReader<int64_t> error(kFailedLegalizationStreamz);
  CreateModule(kMlirModuleStr);

  auto result = Run();

  EXPECT_TRUE(result.failed());
  EXPECT_EQ(error.Delta("tf.BadValue"), 1);
}

TEST_F(VerifyTfxlaLegalizationTest, ErrorsNonStaticInputs) {
  // Using a string constant here instead of testdata to make this compatible
  // with open source.
  static constexpr char kNonStaticFailure[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1504 : i32}} {
    func.func @main() -> tensor<?xi32> attributes {tf.entry_function = {control_outputs = "", inputs = "i,j", outputs = "identity_RetVal"}} {
      %0 = mhlo.constant dense<1.000000e+00> : tensor<f64>
      %1 = mhlo.convert %0 : (tensor<f64>) -> tensor<i64>
      %2 = mhlo.reshape %1 : (tensor<i64>) -> tensor<1xi64>
      %3 = "mhlo.dynamic_iota"(%2) {iota_dimension = 0 : i64} : (tensor<1xi64>) -> tensor<?xi32>
      %4 = mhlo.multiply %3, %3 : tensor<?xi32>
      return %4 : tensor<?xi32>
    }
  })";
  CellReader<int64_t> legal_error(kFailedLegalizationStreamz);
  CellReader<int64_t> static_error(kNonStaticOpStreamz);
  CreateModule(kNonStaticFailure);

  auto result = Run();

  EXPECT_TRUE(result.failed());
  EXPECT_EQ(legal_error.Delta("mhlo.dynamic_iota"), 0);
  EXPECT_EQ(static_error.Delta("mhlo.dynamic_iota"), 1);
}

TEST_F(VerifyTfxlaLegalizationTest, SkipsSpecificNonStaticInputs) {
  // Using a string constant here instead of testdata to make this compatible
  // with open source.
  static constexpr char kNonStaticFailure[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1504 : i32}} {
    func.func @main(%a : tensor<5x14x1xf32>, %b : tensor<1x14x32xf32>) -> tensor<?x?x?xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "i,j", outputs = "identity_RetVal"}} {
      %c = "mhlo.einsum"(%a, %b) {einsum_config = "bji,bjk->bik"} : (tensor<5x14x1xf32>, tensor<1x14x32xf32>) -> tensor<?x?x?xf32>
      return %c : tensor<?x?x?xf32>
    }
  })";
  CellReader<int64_t> static_error(kNonStaticOpStreamz);
  CellReader<int64_t> skipped(kNonStaticOpSkipStreamz);
  CreateModule(kNonStaticFailure);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(static_error.Delta("mhlo.einsum"), 0);
  EXPECT_EQ(skipped.Delta("mhlo.einsum"), 1);
}

TEST_F(VerifyTfxlaLegalizationTest, SkipsNonStaticInputsWithBounds) {
  // Using a string constant here instead of testdata to make this compatible
  // with open source.
  static constexpr char kNonStaticWithBoundsSuccess[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1504 : i32}} {
    func.func @main() -> tensor<?xi32, #mhlo.type_extensions<bounds = [4]>> attributes {tf.entry_function = {control_outputs = "", inputs = "i,j", outputs = "identity_RetVal"}} {
      %0 = mhlo.constant dense<1.000000e+00> : tensor<f64>
      %1 = mhlo.convert %0 : (tensor<f64>) -> tensor<i64>
      %2 = mhlo.reshape %1 : (tensor<i64>) -> tensor<1xi64>
      %3 = "mhlo.dynamic_iota"(%2) {iota_dimension = 0 : i64} : (tensor<1xi64>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [4]>>
      %4 = mhlo.multiply %3, %3 : tensor<?xi32, #mhlo.type_extensions<bounds = [4]>>
      return %4 : tensor<?xi32, #mhlo.type_extensions<bounds = [4]>>
    }
  })";
  CellReader<int64_t> legal_error(kFailedLegalizationStreamz);
  CellReader<int64_t> static_error(kNonStaticOpStreamz);
  CreateModule(kNonStaticWithBoundsSuccess);

  auto result = Run();

  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(legal_error.Delta("mhlo.multiply"), 0);
  EXPECT_EQ(static_error.Delta("mhlo.multiply"), 0);
}

TEST_F(VerifyTfxlaLegalizationTest, RecordsMultipleFailures) {
  // Using a string constant here instead of testdata to make this compatible
  // with open source.
  static constexpr char kMultipleFailures[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.BadValue"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      %1 = "tf.AlsoBad"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  CellReader<int64_t> error(kFailedLegalizationStreamz);
  CreateModule(kMultipleFailures);

  auto result = Run();

  EXPECT_TRUE(result.failed());
  EXPECT_EQ(error.Delta("tf.BadValue"), 1);
  EXPECT_EQ(error.Delta("tf.AlsoBad"), 1);
}

}  // namespace
}  // namespace tensorflow
